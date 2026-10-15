import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def _bootstrap_tt_metal_home():
    if os.environ.get("TT_METAL_HOME"):
        return

    for parent in Path(__file__).resolve().parents:
        if (parent / "runtime" / "hw").is_dir() and (parent / "runtime" / "sfpi").is_dir():
            os.environ["TT_METAL_HOME"] = str(parent)
            return


_bootstrap_tt_metal_home()

import torch
import ttnn
### Profiling ###
class Profiler:
    def __init__(self):
        self.start_times = dict()
        self.times = dict()
        self.disabled = False

    def clear(self):
        self.start_times = dict()
        self.times = dict()
        self.disabled = False

    def enable(self):
        self.disabled = False

    def disable(self):
        self.disabled = True

    def start(self, key, force_enable=False):
        if self.disabled and not force_enable:
            return

        self.start_times[key] = time.time()

    def end(self, key, PERF_CNT=1, force_enable=False):
        if self.disabled and not force_enable:
            return

        if key not in self.start_times:
            return

        diff = time.time() - self.start_times[key]

        if key not in self.times:
            self.times[key] = []

        self.times[key].append(diff / PERF_CNT)

    def get(self, key):
        if key not in self.times:
            return 0

        return sum(self.times[key]) / len(self.times[key])

    def print(self, units="s"):
        for key in self.times:
            average = self.get(key)
            if units == "s":
                pass
            elif units == "ms":
                average *= 1000
            elif units == "us":
                average *= 1000000
            elif units == "ns":
                average *= 1000000000
            else:
                raise ValueError(f"Invalid units: {units}")
            print(f"{key}: {average:.3f}{units}")


profiler = Profiler()


KERNELS_DIR = os.environ.get("LOOM_KERNELS_DIR", "/workspace/loom/tmp_output/kernels")
sys.path.insert(0, str(KERNELS_DIR))

from host_ttnn import run as run_generated_flashdecode


def _log(message):
    print(message, flush=True)


def _get_decode_tops(cfg):
    return 4 * cfg.batch * cfg.nhead * cfg.q_len * cfg.seq_len * cfg.head_dim


@dataclass(frozen=True)
class FlashDecodeConfig:
    batch: int
    nhead: int
    nkv_head: int
    q_len: int
    seq_len: int
    head_dim: int


@dataclass
class GeneratedDecodeDeviceTensors:
    q: ttnn.Tensor
    k_transposed: ttnn.Tensor
    v: ttnn.Tensor
    dst: ttnn.Tensor


GENERATED_CONFIG = FlashDecodeConfig(
    batch=24,
    nhead=32,
    nkv_head=1,
    q_len=1,
    seq_len=4096,
    head_dim=64,
)


SHAPE_SPEC_RE = re.compile(
    r"(?:^|[^A-Za-z0-9])"
    r"B(?P<batch>\d+)_H(?P<nhead>\d+)_L(?P<seq_len>\d+)_D(?P<head_dim>\d+)"
    r"(?=$|[^A-Za-z0-9])",
    re.IGNORECASE,
)


SHAPE_ARG_DEFAULTS = {
    "batch": GENERATED_CONFIG.batch,
    "seq_len": GENERATED_CONFIG.seq_len,
    "nhead": GENERATED_CONFIG.nhead,
    "head_dim": GENERATED_CONFIG.head_dim,
}


def _parse_shape_spec(shape_spec):
    match = SHAPE_SPEC_RE.search(shape_spec)
    if match is None:
        raise ValueError(f"invalid shape spec '{shape_spec}', expected something like B8_H32_L4096_D128")
    return {name: int(value) for name, value in match.groupdict().items()}


def _apply_shape_spec_args(args, parser):
    if args.shape and args.shape_spec and args.shape != args.shape_spec:
        parser.error("use either positional SHAPE or --shape, not both")

    shape_spec = args.shape_spec or args.shape
    shape_values = {}
    if shape_spec:
        try:
            shape_values = _parse_shape_spec(shape_spec)
        except ValueError as exc:
            parser.error(str(exc))

    for name, default_value in SHAPE_ARG_DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, shape_values.get(name, default_value))

    return args


def _torch_rand_bf16(shape, seed, low, high):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensor = torch.rand(shape, generator=generator, dtype=torch.float32)
    tensor = tensor * (high - low) + low
    return tensor.to(torch.bfloat16)


def _to_device_tile(tensor, device):
    return ttnn.as_tensor(
        tensor.contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_device_raw_pages(tensor, device):
    flat = tensor.contiguous().flatten()
    if flat.numel() % 1024 != 0:
        raise ValueError(f"raw page tensor must contain a whole number of tiles, got {flat.numel()} elements")
    return ttnn.as_tensor(
        flat.reshape(1, 1, -1, 1024),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _tilize_nfaces(row_major, rows, cols):
    if rows % 32 != 0 or cols % 32 != 0:
        raise ValueError(f"tilize_nfaces expects rows/cols divisible by 32, got {rows}x{cols}")
    flat = row_major.contiguous().flatten()
    batch = flat.numel() // (rows * cols)
    if batch * rows * cols != flat.numel():
        raise ValueError(f"input with {flat.numel()} elements is not divisible by rows*cols={rows * cols}")

    return (
        flat.reshape(batch, rows // 32, 2, 16, cols // 32, 2, 16)
        .permute(0, 1, 4, 2, 5, 3, 6)
        .contiguous()
        .flatten()
    )


def _untilize_nfaces(tiled, rows, cols):
    if rows % 32 != 0 or cols % 32 != 0:
        raise ValueError(f"untilize_nfaces expects rows/cols divisible by 32, got {rows}x{cols}")
    flat = tiled.contiguous().flatten()
    batch = flat.numel() // (rows * cols)
    if batch * rows * cols != flat.numel():
        raise ValueError(f"input with {flat.numel()} elements is not divisible by rows*cols={rows * cols}")

    return (
        flat.reshape(batch, rows // 32, cols // 32, 2, 2, 16, 16)
        .permute(0, 1, 3, 5, 2, 4, 6)
        .contiguous()
        .reshape(batch, rows, cols)
    )


def _nearest_power_of_two_factor_chunk(seq_len):
    chunk_size = 1
    while seq_len % (chunk_size * 2) == 0:
        chunk_size *= 2
    return min(512, chunk_size)


def _get_grid_size(device, grid_x, grid_y):
    hw_grid = device.compute_with_storage_grid_size()
    grid = (
        grid_x if grid_x is not None else int(hw_grid.x),
        grid_y if grid_y is not None else int(hw_grid.y),
    )
    if grid[0] > int(hw_grid.x) or grid[1] > int(hw_grid.y):
        raise ValueError(f"requested grid {grid} exceeds device grid ({int(hw_grid.x)}, {int(hw_grid.y)})")
    return grid


def _build_inputs(cfg, low, high):
    if cfg.q_len != 1:
        raise ValueError(f"decode comparison expects q_len=1, got {cfg.q_len}")

    q = _torch_rand_bf16((cfg.q_len, cfg.batch, cfg.nhead, cfg.head_dim), seed=123, low=low, high=high)
    k = _torch_rand_bf16((cfg.batch, cfg.nkv_head, cfg.seq_len, cfg.head_dim), seed=1234, low=low, high=high)
    v = _torch_rand_bf16((cfg.batch, cfg.nkv_head, cfg.seq_len, cfg.head_dim), seed=12345, low=low, high=high)
    return q, k, v


def _make_decode_program_config(grid_size, q_chunk_size, k_chunk_size):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )


def _make_decode_compute_kernel_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _prepare_official_decode_inputs(device, q, k, v, log_prefix="  official"):
    start = time.perf_counter()
    _log(f"{log_prefix}: moving Q/K/V to device as TTNN TILE tensors")
    q_dev = _to_device_tile(q, device)
    k_dev = _to_device_tile(k, device)
    v_dev = _to_device_tile(v, device)
    _log(f"{log_prefix}: device upload done in {time.perf_counter() - start:.3f}s")
    return q_dev, k_dev, v_dev


def _launch_official_decode(q_dev, k_dev, v_dev, cfg, program_config, compute_kernel_config):
    return ttnn.transformer.scaled_dot_product_attention_decode(
        q_dev,
        k_dev,
        v_dev,
        is_causal=False,
        cur_pos=[cfg.seq_len - 1 for _ in range(cfg.batch)],
        scale=cfg.head_dim**-0.5,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run_torch_decode(q, k, v, cfg):
    if cfg.nhead % cfg.nkv_head != 0:
        raise ValueError(f"nhead must be divisible by nkv_head, got {cfg.nhead} and {cfg.nkv_head}")

    start = time.perf_counter()
    _log("Running torch scaled_dot_product_attention baseline...")
    q_ref = q.permute(1, 2, 0, 3).to(torch.float32)
    k_ref = k.repeat_interleave(cfg.nhead // cfg.nkv_head, dim=1).to(torch.float32)
    v_ref = v.repeat_interleave(cfg.nhead // cfg.nkv_head, dim=1).to(torch.float32)
    with torch.inference_mode():
        out = torch.nn.functional.scaled_dot_product_attention(
            q_ref,
            k_ref,
            v_ref,
            scale=cfg.head_dim**-0.5,
            is_causal=False,
        )
    _log(f"  torch: baseline done in {time.perf_counter() - start:.3f}s")
    return out.permute(2, 0, 1, 3).contiguous().to(torch.float32)


def _run_official_decode(device, q, k, v, cfg, grid_size, k_chunk_size, q_chunk_size):
    q_dev, k_dev, v_dev = _prepare_official_decode_inputs(device, q, k, v)
    program_config = _make_decode_program_config(grid_size, q_chunk_size, k_chunk_size)
    compute_kernel_config = _make_decode_compute_kernel_config()

    start = time.perf_counter()
    _log("  official: launching scaled_dot_product_attention_decode")
    out_dev = _launch_official_decode(q_dev, k_dev, v_dev, cfg, program_config, compute_kernel_config)
    ttnn.synchronize_device(device)
    _log(f"  official: launch/sync done in {time.perf_counter() - start:.3f}s")
    return ttnn.to_torch(out_dev).to(torch.float32)


def _prepare_generated_decode_inputs(device, q, k, v, cfg, log_prefix="  generated"):
    # The generated dataflow expects K as [B, KVH, D, S], unlike official SDPA decode.
    # It also expects each DRAM page in the low-level TILED_NFACES order used by host_flashdecode.cpp.
    start = time.perf_counter()
    _log(f"{log_prefix}: transposing K and tilizing Q/K^T/V into TILED_NFACES pages")
    k_transposed = k.transpose(-2, -1).contiguous()
    q_tiled = _tilize_nfaces(q, rows=cfg.nhead * cfg.q_len, cols=cfg.head_dim)
    k_transposed_tiled = _tilize_nfaces(k_transposed, rows=cfg.head_dim, cols=cfg.seq_len)
    v_tiled = _tilize_nfaces(v, rows=cfg.seq_len, cols=cfg.head_dim)
    # Make unwritten pages obvious in correctness checks instead of inheriting stale DRAM contents.
    dst_tiled = torch.zeros_like(q_tiled)
    _log(f"{log_prefix}: Python tilize done in {time.perf_counter() - start:.3f}s")

    start = time.perf_counter()
    _log(f"{log_prefix}: moving raw pages to device as ROW_MAJOR backing tensors")
    q_dev = _to_device_raw_pages(q_tiled, device)
    k_transposed_dev = _to_device_raw_pages(k_transposed_tiled, device)
    v_dev = _to_device_raw_pages(v_tiled, device)
    dst_dev = _to_device_raw_pages(dst_tiled, device)
    _log(f"{log_prefix}: raw page upload done in {time.perf_counter() - start:.3f}s")

    return GeneratedDecodeDeviceTensors(q=q_dev, k_transposed=k_transposed_dev, v=v_dev, dst=dst_dev)


def _launch_generated_decode(device_tensors):
    device_tensors.dst = run_generated_flashdecode(
        device_tensors.k_transposed,
        device_tensors.v,
        device_tensors.q,
        device_tensors.dst,
    )
    return device_tensors.dst


def _read_generated_decode_output(dst_dev, cfg):
    start = time.perf_counter()
    _log("  generated: reading output and untilizing TILED_NFACES pages")
    generated_tiled = ttnn.to_torch(dst_dev).flatten().to(torch.bfloat16)
    generated = _untilize_nfaces(generated_tiled, rows=cfg.nhead * cfg.q_len, cols=cfg.head_dim)
    _log(f"  generated: output read/untilize done in {time.perf_counter() - start:.3f}s")
    return generated.reshape(cfg.batch, cfg.q_len, cfg.nhead, cfg.head_dim).permute(1, 0, 2, 3).to(torch.float32)


def _run_generated_decode(device, q, k, v, cfg):
    device_tensors = _prepare_generated_decode_inputs(device, q, k, v, cfg)

    start = time.perf_counter()
    _log("  generated: launching generated generic_op")
    dst_dev = _launch_generated_decode(device_tensors)
    ttnn.synchronize_device(device)
    _log(f"  generated: launch/sync done in {time.perf_counter() - start:.3f}s")

    return _read_generated_decode_output(dst_dev, cfg)


def _benchmark(device, run_once, warmup, iters, use_trace):
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    if iters <= 0:
        raise ValueError(f"iters must be > 0, got {iters}")

    last_result = None
    for _ in range(warmup):
        last_result = run_once()
    ttnn.synchronize_device(device)
    last_result = None

    profiler.clear()
    if use_trace:
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(iters):
            last_result = run_once()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)

        profiler.start("run")
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        profiler.end("run")
        ttnn.release_trace(device, tid)
        last_result = None
        return profiler.get("run") / iters

    profiler.start("run")
    for _ in range(iters):
        last_result = run_once()
    ttnn.synchronize_device(device)
    profiler.end("run")
    last_result = None
    return profiler.get("run") / iters


def _benchmark_official_decode(device, q, k, v, cfg, grid_size, k_chunk_size, q_chunk_size, warmup, iters, use_trace):
    q_dev, k_dev, v_dev = _prepare_official_decode_inputs(device, q, k, v, log_prefix="  official benchmark")
    program_config = _make_decode_program_config(grid_size, q_chunk_size, k_chunk_size)
    compute_kernel_config = _make_decode_compute_kernel_config()

    def run_once():
        return _launch_official_decode(q_dev, k_dev, v_dev, cfg, program_config, compute_kernel_config)

    return _benchmark(
        device=device,
        run_once=run_once,
        warmup=warmup,
        iters=iters,
        use_trace=use_trace,
    )


def _benchmark_generated_decode(device, q, k, v, cfg, warmup, iters, use_trace):
    device_tensors = _prepare_generated_decode_inputs(device, q, k, v, cfg, log_prefix="  benchmark")

    def run_once():
        return _launch_generated_decode(device_tensors)

    return _benchmark(
        device=device,
        run_once=run_once,
        warmup=warmup,
        iters=iters,
        use_trace=use_trace,
    )


def _report_decode_performance(label, cfg, grid_size, avg_seconds, warmup, iters, use_trace):
    avg_ms = avg_seconds * 1000.0
    tflops = _get_decode_tops(cfg) / avg_seconds / 1e12

    _log(f"{label} performance:")
    _log(f"  Grid: {grid_size}")
    _log(f"  Warmup iterations: {warmup}")
    _log(f"  Measurement iterations: {iters}")
    _log(f"  Trace capture: {'enabled' if use_trace else 'disabled'}")
    _log(f"  Latency per token: {avg_ms:.6f} ms")
    _log(f"  Decode TFLOPS: {tflops:.6f}")


def _report_speedup(generated_seconds, official_seconds):
    if generated_seconds is None or official_seconds is None:
        return
    if generated_seconds <= 0:
        _log("Speedup generated over official TTNN: unavailable because generated latency is non-positive")
        return
    _log(f"Speedup generated over official TTNN: {official_seconds / generated_seconds:.3f}x")


def _pcc(expected, actual):
    expected = expected.flatten().to(torch.float64)
    actual = actual.flatten().to(torch.float64)
    if expected.numel() != actual.numel():
        raise ValueError(f"PCC shape mismatch: {tuple(expected.shape)} vs {tuple(actual.shape)}")
    expected_centered = expected - expected.mean()
    actual_centered = actual - actual.mean()
    denom = torch.linalg.vector_norm(expected_centered) * torch.linalg.vector_norm(actual_centered)
    if denom == 0:
        return float("nan")
    return float(torch.dot(expected_centered, actual_centered) / denom)


def _comparison_stats(expected, actual):
    expected = expected.to(torch.float32)
    actual = actual.to(torch.float32)
    diff = (expected - actual).abs()
    return {
        "pcc": _pcc(expected, actual),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "num_different": int((diff != 0).sum()),
    }


def run_flashdecode_comparison(args):
    cfg = FlashDecodeConfig(
        batch=args.batch,
        nhead=args.nhead,
        nkv_head=args.nkv_head,
        q_len=args.q_len,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
    )

    device_params = {
        "l1_small_size": args.l1_small_size,
        "trace_region_size": args.trace_region_size,
    }

    q, k, v = _build_inputs(cfg, low=args.low, high=args.high)
    torch_baseline = _run_torch_decode(q, k, v, cfg)
    official_perf_seconds = None
    generated_perf_seconds = None

    official_device = ttnn.CreateDevice(device_id=args.device_id, **device_params)
    try:
        official_grid = _get_grid_size(official_device, args.grid_x, args.grid_y)
        k_chunk_size = args.k_chunk_size or _nearest_power_of_two_factor_chunk(args.seq_len)
        q_chunk_size = args.q_chunk_size or args.nhead

        _log("Running ttnn.transformer.scaled_dot_product_attention_decode...")
        official = _run_official_decode(
            device=official_device,
            q=q,
            k=k,
            v=v,
            cfg=cfg,
            grid_size=official_grid,
            k_chunk_size=k_chunk_size,
            q_chunk_size=q_chunk_size,
        )
        if not args.skip_perf and args.perf_target in ("official", "both"):
            _log("Benchmarking official TTNN flashdecode implementation...")
            official_perf_seconds = _benchmark_official_decode(
                device=official_device,
                q=q,
                k=k,
                v=v,
                cfg=cfg,
                grid_size=official_grid,
                k_chunk_size=k_chunk_size,
                q_chunk_size=q_chunk_size,
                warmup=args.warmup,
                iters=args.iters,
                use_trace=not args.no_trace,
            )
            _report_decode_performance(
                label="Official TTNN flashdecode",
                cfg=cfg,
                grid_size=official_grid,
                avg_seconds=official_perf_seconds,
                warmup=args.warmup,
                iters=args.iters,
                use_trace=not args.no_trace,
            )
    finally:
        ttnn.close_device(official_device)

    generated_device = ttnn.CreateDevice(device_id=args.device_id, **device_params)
    try:
        generated_grid = _get_grid_size(generated_device, 8, 8)

        _log("Running generated flashdecode...")
        generated = _run_generated_decode(generated_device, q, k, v, cfg)

        pcc_torch_vs_official = _pcc(torch_baseline, official)
        pcc_torch_vs_generated = _pcc(torch_baseline, generated)
        pcc_official_vs_generated = _pcc(official, generated)
        min_pcc = min(pcc_torch_vs_official, pcc_torch_vs_generated, pcc_official_vs_generated)
        passed = min_pcc >= args.pcc

        _log(
            f"Shape: B={cfg.batch}, H={cfg.nhead}, KVH={cfg.nkv_head}, "
            f"Q={cfg.q_len}, S={cfg.seq_len}, D={cfg.head_dim}"
        )
        _log(f"Generated grid: {generated_grid}")
        _log(f"Official grid: {official_grid}")
        _log(f"Official q_chunk_size: {q_chunk_size}")
        _log(f"Official k_chunk_size: {k_chunk_size}")
        _log(f"PCC(torch_baseline, ttnn_sdpa_decode): {pcc_torch_vs_official:.9f}")
        _log(f"PCC(torch_baseline, generated): {pcc_torch_vs_generated:.9f}")
        _log(f"PCC(ttnn_sdpa_decode, generated): {pcc_official_vs_generated:.9f}")
        _log(f"Result: {'PASS' if passed else 'FAIL'} (min PCC >= {args.pcc})")

        if args.print_values:
            count = min(args.print_values, torch_baseline.numel())
            _log(f"torch_baseline[:{count}] = {torch_baseline.flatten()[:count].tolist()}")
            _log(f"official[:{count}] = {official.flatten()[:count].tolist()}")
            _log(f"generated[:{count}] = {generated.flatten()[:count].tolist()}")

        if not args.skip_perf and args.perf_target in ("generated", "both"):
            _log("Benchmarking generated flashdecode kernel...")
            generated_perf_seconds = _benchmark_generated_decode(
                device=generated_device,
                q=q,
                k=k,
                v=v,
                cfg=cfg,
                warmup=args.warmup,
                iters=args.iters,
                use_trace=not args.no_trace,
            )
            _report_decode_performance(
                label="Generated flashdecode",
                cfg=cfg,
                grid_size=generated_grid,
                avg_seconds=generated_perf_seconds,
                warmup=args.warmup,
                iters=args.iters,
                use_trace=not args.no_trace,
            )
            _report_speedup(generated_perf_seconds, official_perf_seconds)

        if args.raise_on_fail and not passed:
            raise AssertionError(f"Minimum PCC {min_pcc} is below threshold {args.pcc}")
    finally:
        ttnn.close_device(generated_device)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the generated flash-decode kernel against "
            "ttnn.transformer.scaled_dot_product_attention_decode. "
            "Official TTNN receives K as [B, KVH, S, D]; the generated path receives K^T as [B, KVH, D, S]."
        )
    )
    parser.add_argument(
        "shape",
        nargs="?",
        help="Optional shape shorthand like B8_H32_L4096_D128.",
    )
    parser.add_argument(
        "--shape",
        dest="shape_spec",
        type=str,
        default=None,
        help="Shape shorthand like B8_H32_L4096_D128. Explicit dimension flags override it.",
    )
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--nkv_head", type=int, default=GENERATED_CONFIG.nkv_head)
    parser.add_argument("--head_dim", type=int, default=None)
    parser.add_argument("--q_len", type=int, default=GENERATED_CONFIG.q_len)
    parser.add_argument("--grid_x", type=int, default=8)
    parser.add_argument("--grid_y", type=int, default=8)
    parser.add_argument("--q_chunk_size", type=int, default=None)
    parser.add_argument("--k_chunk_size", type=int, default=None)
    parser.add_argument("--pcc", type=float, default=0.98)
    parser.add_argument("--low", type=float, default=-0.5)
    parser.add_argument("--high", type=float, default=0.5)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--l1_small_size", type=int, default=24576)
    parser.add_argument("--trace_region_size", type=int, default=7520256 * 8)
    parser.add_argument("--print_values", type=int, default=0)
    parser.add_argument("--raise_on_fail", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no_trace", action="store_true")
    parser.add_argument("--skip_perf", action="store_true")
    parser.add_argument("--perf_target", choices=("official", "generated", "both"), default="both")
    return _apply_shape_spec_args(parser.parse_args(), parser)


if __name__ == "__main__":
    run_flashdecode_comparison(parse_args())
