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

from host_ttnn import run as run_generated_flashattn
from host_flashdecode import _tilize_nfaces, _untilize_nfaces

TORCH_DTYPE = torch.bfloat16
TTNN_DTYPE = ttnn.bfloat16


def _log(message):
    print(message, flush=True)


@dataclass(frozen=True)
class FlashAttnConfig:
    batch: int
    nhead: int
    seq_len: int
    head_dim: int
    is_causal: bool


@dataclass
class GeneratedFlashAttnDeviceTensors:
    k_transposed: ttnn.Tensor
    v: ttnn.Tensor
    q: ttnn.Tensor
    dst: ttnn.Tensor


GENERATED_CONFIG = FlashAttnConfig(
    batch=8,
    nhead=64,
    seq_len=2048,
    head_dim=1024,
    is_causal=False,
)

TOTAL_HEAD_WIDTH = 1024 * 64


SHAPE_TOKEN_RE = re.compile(
    r"(?:^|[^A-Za-z0-9])(?P<name>[BNLHD])(?P<value>\d+)(?=$|[^A-Za-z0-9])",
    re.IGNORECASE,
)


SHAPE_ARG_DEFAULTS = {
    "batch": GENERATED_CONFIG.batch,
    "nhead": GENERATED_CONFIG.nhead,
    "seq_len": GENERATED_CONFIG.seq_len,
}


def _parse_shape_spec(shape_spec):
    token_to_arg = {
        "B": "batch",
        "N": "nhead",
        "L": "seq_len",
        "H": "nhead",
        "D": "head_dim",
    }
    shape_values = {}
    for match in SHAPE_TOKEN_RE.finditer(shape_spec):
        name = token_to_arg[match.group("name").upper()]
        value = int(match.group("value"))
        if name in shape_values and shape_values[name] != value:
            raise ValueError(f"conflicting {name} values in shape spec '{shape_spec}'")
        shape_values[name] = value

    required = ("batch", "seq_len", "nhead")
    if any(name not in shape_values for name in required):
        raise ValueError(
            f"invalid shape spec '{shape_spec}', expected something like B1_L16384_H128 "
            "or B1_H128_L16384_D512"
        )
    return shape_values


def _derive_head_dim(nhead):
    if nhead <= 0:
        raise ValueError(f"nhead must be > 0, got {nhead}")
    if TOTAL_HEAD_WIDTH % nhead != 0:
        raise ValueError(f"nhead must divide {TOTAL_HEAD_WIDTH} so D=1024*64/H is integral, got H={nhead}")
    return TOTAL_HEAD_WIDTH // nhead


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

    if args.head_dim is None and "head_dim" in shape_values:
        args.head_dim = shape_values["head_dim"]

    explicit_head_dim = args.head_dim is not None or "head_dim" in shape_values
    try:
        derived_head_dim = _derive_head_dim(args.nhead)
    except ValueError as exc:
        parser.error(str(exc))
    if explicit_head_dim and args.head_dim != derived_head_dim:
        parser.error(f"head_dim/D is fixed to 1024*64/H = {derived_head_dim} for H={args.nhead}")
    args.head_dim = derived_head_dim

    return args


def _get_sdpa_tops(cfg):
    return 4 * cfg.batch * cfg.nhead * cfg.seq_len * cfg.seq_len * cfg.head_dim


def _torch_rand_bf16(shape, seed, low, high):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensor = torch.rand(shape, generator=generator, dtype=TORCH_DTYPE)
    return tensor * (high - low) + low


def _to_device_tile(tensor, device):
    return ttnn.as_tensor(
        tensor.contiguous(),
        device=device,
        dtype=TTNN_DTYPE,
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
        dtype=TTNN_DTYPE,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


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
    q = _torch_rand_bf16((cfg.batch, cfg.nhead, cfg.seq_len, cfg.head_dim), seed=123, low=low, high=high)
    k = _torch_rand_bf16((cfg.batch, cfg.nhead, cfg.seq_len, cfg.head_dim), seed=1234, low=low, high=high)
    v = _torch_rand_bf16((cfg.batch, cfg.nhead, cfg.seq_len, cfg.head_dim), seed=12345, low=low, high=high)
    return q, k, v


def _make_sdpa_program_config(grid_size, q_chunk_size, k_chunk_size):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size
    )


def _make_compute_kernel_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4
    )


def _run_official_sdpa(device, q, k, v, cfg, grid_size, q_chunk_size, k_chunk_size):
    start = time.perf_counter()
    _log("  official: moving Q/K/V to device as TTNN TILE tensors")
    q_dev = _to_device_tile(q, device)
    k_dev = _to_device_tile(k, device)
    v_dev = _to_device_tile(v, device)
    _log(f"  official: device upload done in {time.perf_counter() - start:.3f}s")

    program_config = _make_sdpa_program_config(grid_size, q_chunk_size, k_chunk_size)

    start = time.perf_counter()
    _log("  official: launching ttnn.transformer.scaled_dot_product_attention")
    out_dev = ttnn.transformer.scaled_dot_product_attention(
        q_dev,
        k_dev,
        v_dev,
        is_causal=cfg.is_causal,
        scale=cfg.head_dim**-0.5,
        program_config=program_config,
        compute_kernel_config=_make_compute_kernel_config(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.synchronize_device(device)
    _log(f"  official: launch/sync done in {time.perf_counter() - start:.3f}s")
    return ttnn.to_torch(out_dev)


def _run_mlir_sdpa(device, q, k, v, cfg, grid_size, q_chunk_size, k_chunk_size):
    start = time.perf_counter()
    _log("  mlir: transposing K and tilizing K^T/V/Q into TILED_NFACES pages")
    k_transposed = k.transpose(-2, -1).contiguous()
    k_transposed_tiled = _tilize_nfaces(k_transposed, rows=cfg.head_dim, cols=cfg.seq_len)
    v_tiled = _tilize_nfaces(v, rows=cfg.seq_len, cols=cfg.head_dim)
    q_tiled = _tilize_nfaces(q, rows=cfg.seq_len, cols=cfg.head_dim)
    # Make unwritten pages obvious in correctness checks instead of inheriting stale DRAM contents.
    dst_tiled = torch.zeros_like(q_tiled)
    _log(f"  mlir: Python tilize done in {time.perf_counter() - start:.3f}s")

    start = time.perf_counter()
    _log("  mlir: moving raw pages to device as ROW_MAJOR backing tensors")
    k_transposed_dev = _to_device_raw_pages(k_transposed_tiled, device)
    v_dev = _to_device_raw_pages(v_tiled, device)
    q_dev = _to_device_raw_pages(q_tiled, device)
    dst_dev = _to_device_raw_pages(dst_tiled, device)
    _log(f"  mlir: raw page upload done in {time.perf_counter() - start:.3f}s")

    start = time.perf_counter()
    _log("  mlir: launching generated host_ttnn.run")
    out_dev = run_generated_flashattn(
        k_transposed_dev,
        v_dev,
        q_dev,
        dst_dev,
    )
    ttnn.synchronize_device(device)
    _log(f"  mlir: launch/sync done in {time.perf_counter() - start:.3f}s")
    return _read_generated_sdpa_output(out_dev, cfg)


def _prepare_sdpa_inputs(device, q, k, v):
    return _to_device_tile(q, device), _to_device_tile(k, device), _to_device_tile(v, device)


def _prepare_generated_sdpa_inputs(device, q, k, v, cfg):
    k_transposed = k.transpose(-2, -1).contiguous()
    k_transposed_tiled = _tilize_nfaces(k_transposed, rows=cfg.head_dim, cols=cfg.seq_len)
    v_tiled = _tilize_nfaces(v, rows=cfg.seq_len, cols=cfg.head_dim)
    q_tiled = _tilize_nfaces(q, rows=cfg.seq_len, cols=cfg.head_dim)
    # Make unwritten pages obvious in correctness checks instead of inheriting stale DRAM contents.
    dst_tiled = torch.zeros_like(q_tiled)
    return GeneratedFlashAttnDeviceTensors(
        k_transposed=_to_device_raw_pages(k_transposed_tiled, device),
        v=_to_device_raw_pages(v_tiled, device),
        q=_to_device_raw_pages(q_tiled, device),
        dst=_to_device_raw_pages(dst_tiled, device),
    )


def _launch_generated_sdpa(device_tensors):
    result = run_generated_flashattn(
        device_tensors.k_transposed,
        device_tensors.v,
        device_tensors.q,
        device_tensors.dst,
    )
    if result is None:
        return device_tensors.dst
    if isinstance(result, (list, tuple)):
        if len(result) != 1:
            raise ValueError(f"host_ttnn.run returned {len(result)} outputs; expected one")
        return result[0]
    return result


def _read_generated_sdpa_output(output_dev, cfg):
    start = time.perf_counter()
    _log("  mlir: reading output and untilizing TILED_NFACES pages")
    generated_tiled = ttnn.to_torch(output_dev).flatten()
    expected_numel = cfg.batch * cfg.nhead * cfg.seq_len * cfg.head_dim
    if generated_tiled.numel() != expected_numel:
        raise ValueError(f"generated output has {generated_tiled.numel()} elements, expected {expected_numel}")
    generated = _untilize_nfaces(generated_tiled, rows=cfg.seq_len, cols=cfg.head_dim)
    _log(f"  mlir: output read/untilize done in {time.perf_counter() - start:.3f}s")
    return generated.reshape(cfg.batch, cfg.nhead, cfg.seq_len, cfg.head_dim)


def _benchmark(device, run_once, warmup, iters, use_trace, sync_each_iter=False):
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    if iters <= 0:
        raise ValueError(f"iters must be > 0, got {iters}")

    last_result = None
    for _ in range(warmup):
        last_result = run_once()
        if sync_each_iter:
            ttnn.synchronize_device(device)
    ttnn.synchronize_device(device)
    last_result = None

    profiler.clear()
    if use_trace:
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        last_result = run_once()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)

        profiler.start("run")
        for _ in range(iters):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=sync_each_iter)
            if sync_each_iter:
                ttnn.synchronize_device(device)
        if not sync_each_iter:
            ttnn.synchronize_device(device)
        profiler.end("run")
        ttnn.release_trace(device, tid)
        last_result = None
        return profiler.get("run") / iters

    profiler.start("run")
    for _ in range(iters):
        last_result = run_once()
        if sync_each_iter:
            ttnn.synchronize_device(device)
    if not sync_each_iter:
        ttnn.synchronize_device(device)
    profiler.end("run")
    last_result = None
    return profiler.get("run") / iters


def _benchmark_official_sdpa(
    device, q, k, v, cfg, grid_size, q_chunk_size, k_chunk_size, warmup, iters, use_trace, sync_each_iter
):
    q_dev, k_dev, v_dev = _prepare_sdpa_inputs(device, q, k, v)
    program_config = _make_sdpa_program_config(grid_size, q_chunk_size, k_chunk_size)

    def run_once():
        return ttnn.transformer.scaled_dot_product_attention(
            q_dev,
            k_dev,
            v_dev,
            is_causal=cfg.is_causal,
            scale=cfg.head_dim**-0.5,
            program_config=program_config,
            compute_kernel_config=_make_compute_kernel_config(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return _benchmark(device, run_once, warmup, iters, use_trace, sync_each_iter=sync_each_iter)


def _benchmark_mlir_sdpa(
    device, q, k, v, cfg, grid_size, q_chunk_size, k_chunk_size, warmup, iters, use_trace, sync_each_iter
):
    device_tensors = _prepare_generated_sdpa_inputs(device, q, k, v, cfg)

    def run_once():
        return _launch_generated_sdpa(device_tensors)

    return _benchmark(device, run_once, warmup, iters, use_trace, sync_each_iter=sync_each_iter)


def _report_performance(label, cfg, grid_size, avg_seconds, warmup, iters, use_trace, sync_each_iter):
    avg_ms = avg_seconds * 1000.0
    tflops = _get_sdpa_tops(cfg) / avg_seconds / 1e12

    _log(f"{label} performance:")
    _log(f"  Grid: {grid_size}")
    _log(f"  Warmup iterations: {warmup}")
    _log(f"  Measurement iterations: {iters}")
    _log(f"  Trace capture: {'enabled' if use_trace else 'disabled'}")
    _log(f"  Sync each iteration: {'enabled' if sync_each_iter else 'disabled'}")
    _log(f"  Latency: {avg_ms:.6f} ms")
    _log(f"  SDPA TFLOPS: {tflops:.6f}")


def _report_speedup(mlir_seconds, official_seconds):
    if mlir_seconds is None or official_seconds is None:
        return
    if mlir_seconds <= 0:
        _log("Speedup generated host_ttnn over official TTNN: unavailable because generated latency is non-positive")
        return
    _log(f"Speedup generated host_ttnn over official TTNN: {official_seconds / mlir_seconds:.3f}x")


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


def _torch_baseline_generated_stats(q, k, v, generated, cfg):
    expected_shape = (cfg.batch, cfg.nhead, cfg.seq_len, cfg.head_dim)
    if tuple(generated.shape) != expected_shape:
        raise ValueError(
            "generated output shape mismatch: "
            f"got {tuple(generated.shape)}, expected {expected_shape}"
        )

    _log(
        "Running batched torch scaled_dot_product_attention baseline "
        f"for shape {expected_shape}..."
    )
    start = time.perf_counter()
    with torch.inference_mode():
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=cfg.is_causal,
            scale=cfg.head_dim**-0.5,
        )
    _log(f"  torch: baseline SDPA done in {time.perf_counter() - start:.3f}s")

    stats = _comparison_stats(expected, generated)
    stats["first_batch"] = 0
    stats["first_head"] = 0
    stats["first_expected"] = expected[0, 0]
    stats["first_actual"] = generated[0, 0]
    return stats


def run_flashattn_comparison(args):
    cfg = FlashAttnConfig(
        batch=args.batch,
        nhead=args.nhead,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        is_causal=args.is_causal,
    )
    run_mlir = args.perf_target in ("mlir", "both")
    run_official_for_perf = args.perf_target in ("official", "both") and not args.skip_perf
    run_correctness = run_mlir and not args.skip_correctness
    run_official_for_correctness = run_correctness
    run_official = run_official_for_perf or run_official_for_correctness
    run_torch_correctness = run_correctness

    device_params = {
        "l1_small_size": args.l1_small_size,
        "trace_region_size": args.trace_region_size,
    }

    q, k, v = _build_inputs(cfg, low=args.low, high=args.high)
    official_perf_seconds = None
    mlir_perf_seconds = None
    official = None
    official_grid = None

    if run_official:
        official_device = ttnn.CreateDevice(device_id=args.device_id, **device_params)
        try:
            official_grid = _get_grid_size(official_device, args.grid_x, args.grid_y)
            _log("Running official ttnn.transformer.scaled_dot_product_attention...")
            official = _run_official_sdpa(
                device=official_device,
                q=q,
                k=k,
                v=v,
                cfg=cfg,
                grid_size=official_grid,
                q_chunk_size=args.q_chunk_size,
                k_chunk_size=args.k_chunk_size,
            )
            if not args.skip_perf and args.perf_target in ("official", "both"):
                _log("Benchmarking official SDPA...")
                official_perf_seconds = _benchmark_official_sdpa(
                    device=official_device,
                    q=q,
                    k=k,
                    v=v,
                    cfg=cfg,
                    grid_size=official_grid,
                    q_chunk_size=args.q_chunk_size,
                    k_chunk_size=args.k_chunk_size,
                    warmup=args.warmup,
                    iters=args.iters,
                    use_trace=not args.no_trace,
                    sync_each_iter=args.sync_each_iter,
                )
                _report_performance(
                    "Official SDPA",
                    cfg,
                    official_grid,
                    official_perf_seconds,
                    args.warmup,
                    args.iters,
                    not args.no_trace,
                    args.sync_each_iter,
                )
        finally:
            ttnn.close_device(official_device)
    else:
        _log("Official SDPA path: skipped (--skip_correctness and official perf not requested)")

    mlir = None
    mlir_grid = None
    if run_mlir:
        mlir_device = ttnn.CreateDevice(device_id=args.device_id, **device_params)
        try:
            mlir_grid = _get_grid_size(mlir_device, args.grid_x, args.grid_y)
            _log("Running generated host_ttnn SDPA...")
            mlir = _run_mlir_sdpa(
                device=mlir_device,
                q=q,
                k=k,
                v=v,
                cfg=cfg,
                grid_size=mlir_grid,
                q_chunk_size=args.q_chunk_size,
                k_chunk_size=args.k_chunk_size,
            )
            if not args.skip_perf and args.perf_target in ("mlir", "both"):
                _log("Benchmarking generated host_ttnn SDPA...")
                mlir_perf_seconds = _benchmark_mlir_sdpa(
                    device=mlir_device,
                    q=q,
                    k=k,
                    v=v,
                    cfg=cfg,
                    grid_size=mlir_grid,
                    q_chunk_size=args.q_chunk_size,
                    k_chunk_size=args.k_chunk_size,
                    warmup=args.warmup,
                    iters=args.iters,
                    use_trace=not args.no_trace,
                    sync_each_iter=args.sync_each_iter,
                )
                _report_performance(
                    "Generated host_ttnn SDPA",
                    cfg,
                    mlir_grid,
                    mlir_perf_seconds,
                    args.warmup,
                    args.iters,
                    not args.no_trace,
                    args.sync_each_iter,
                )
                _report_speedup(mlir_perf_seconds, official_perf_seconds)
        finally:
            ttnn.close_device(mlir_device)

    pccs = []

    _log(f"Shape: B={cfg.batch}, H={cfg.nhead}, L={cfg.seq_len}, D={cfg.head_dim}, causal={cfg.is_causal}")
    if official_grid is not None:
        _log(f"Official grid: {official_grid}")
    else:
        _log("Official path: skipped")
    if mlir_grid is not None:
        _log(f"Generated host_ttnn grid: {mlir_grid}")
    else:
        _log("Generated host_ttnn path: skipped (--perf_target=official)")

    stats_official_vs_mlir = None
    if official is not None and mlir is not None and run_correctness:
        _log("Running official TTNN vs generated host_ttnn PCC comparison...")
        stats_official_vs_mlir = _comparison_stats(official, mlir)
        pccs.append(stats_official_vs_mlir["pcc"])
        _log(f"PCC(official_ttnn, generated_host_ttnn): {stats_official_vs_mlir['pcc']:.9f}")
        _log(f"Max abs diff(official_ttnn, generated_host_ttnn): {stats_official_vs_mlir['max_abs_diff']:.9f}")
        _log(f"Mean abs diff(official_ttnn, generated_host_ttnn): {stats_official_vs_mlir['mean_abs_diff']:.9f}")
        _log(f"Num different(official_ttnn, generated_host_ttnn): {stats_official_vs_mlir['num_different']}")

    stats_torch_vs_mlir = None
    if mlir is not None and run_torch_correctness:
        _log(f"Correctness debug: generated shape={tuple(mlir.shape)}, dtype={mlir.dtype}, numel={mlir.numel()}")
        stats_torch_vs_mlir = _torch_baseline_generated_stats(q, k, v, mlir, cfg)
        pccs.append(stats_torch_vs_mlir["pcc"])
        _log(f"PCC(torch_baseline, generated_host_ttnn): {stats_torch_vs_mlir['pcc']:.9f}")
        _log(f"Max abs diff(torch_baseline, generated_host_ttnn): {stats_torch_vs_mlir['max_abs_diff']:.9f}")
        _log(f"Mean abs diff(torch_baseline, generated_host_ttnn): {stats_torch_vs_mlir['mean_abs_diff']:.9f}")
        _log(f"Num different(torch_baseline, generated_host_ttnn): {stats_torch_vs_mlir['num_different']}")

    if pccs:
        min_pcc = min(pccs)
        passed = min_pcc >= args.pcc
        _log(f"Result: {'PASS' if passed else 'FAIL'} (min PCC >= {args.pcc})")
    else:
        min_pcc = None
        passed = True
        _log("Result: correctness skipped because no comparison path was selected")

    if args.print_values:
        source = official if official is not None else mlir
        count = min(args.print_values, source.numel()) if source is not None else 0
        if stats_torch_vs_mlir is not None:
            torch_values = stats_torch_vs_mlir["first_expected"].flatten()
            generated_values = stats_torch_vs_mlir["first_actual"].flatten()
            slice_count = min(args.print_values, torch_values.numel())
            batch = stats_torch_vs_mlir["first_batch"]
            head = stats_torch_vs_mlir["first_head"]
            _log(f"torch_baseline[B={batch},H={head}][:{slice_count}] = {torch_values[:slice_count].tolist()}")
            _log(f"generated[B={batch},H={head}][:{slice_count}] = {generated_values[:slice_count].tolist()}")
        elif stats_official_vs_mlir is not None:
            _log(f"official[:{count}] = {official.flatten()[:count].tolist()}")
            _log(f"generated[:{count}] = {mlir.flatten()[:count].tolist()}")
        elif official is not None:
            _log(f"official[:{count}] = {official.flatten()[:count].tolist()}")
        if mlir is not None and stats_official_vs_mlir is None:
            _log(f"generated[:{count}] = {mlir.flatten()[:count].tolist()}")

    if args.raise_on_fail and not passed:
        raise AssertionError(f"Minimum PCC {min_pcc} is below threshold {args.pcc}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Profile official ttnn.transformer.scaled_dot_product_attention and the generated-style "
            "host_ttnn.py SDPA path."
        )
    )
    parser.add_argument(
        "shape",
        nargs="?",
        help="Optional shape shorthand like B1_L16384_H128 or B1_H128_L16384_D512.",
    )
    parser.add_argument(
        "--shape",
        dest="shape_spec",
        type=str,
        default=None,
        help="Shape shorthand like B1_L16384_H128. H is nhead; D is fixed to 1024*64/H.",
    )
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--head_dim", type=int, default=None)
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--grid_x", type=int, default=8)
    parser.add_argument("--grid_y", type=int, default=8)
    parser.add_argument("--q_chunk_size", type=int, default=32)
    parser.add_argument("--k_chunk_size", type=int, default=32)
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
    parser.add_argument(
        "--sync_each_iter",
        action="store_true",
        help="Debug timing mode: synchronize after every measured iteration instead of only once at the end.",
    )
    parser.add_argument("--skip_perf", action="store_true")
    parser.add_argument(
        "--skip_correctness",
        action="store_true",
        help="Skip correctness checks (official TTNN vs generated, and torch baseline vs generated).",
    )
    parser.add_argument("--perf_target", choices=("official", "mlir", "both"), default="both")
    return _apply_shape_spec_args(parser.parse_args(), parser)


if __name__ == "__main__":
    run_flashattn_comparison(parse_args())
