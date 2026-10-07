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

from host_ttnn import run as run_generated_matmul


def _log(message):
    print(message, flush=True)


@dataclass(frozen=True)
class MatmulConfig:
    m: int
    n: int
    k: int


GENERATED_CONFIG = MatmulConfig(
    m=4096,
    n=4096,
    k=512,
)

SHAPE_SPEC_RE = re.compile(
    r"(?:^|[^A-Za-z0-9])"
    r"M(?P<m>\d+)_N(?P<n>\d+)_K(?P<k>\d+)"
    r"(?=$|[^A-Za-z0-9])",
    re.IGNORECASE,
)

SHAPE_ARG_DEFAULTS = {
    "m": GENERATED_CONFIG.m,
    "n": GENERATED_CONFIG.n,
    "k": GENERATED_CONFIG.k,
}


def _parse_shape_spec(shape_spec):
    match = SHAPE_SPEC_RE.search(shape_spec)
    if match is None:
        raise ValueError(f"invalid shape spec '{shape_spec}', expected something like M256_N256_K256")
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


def _get_matmul_tops(cfg):
    return 2 * cfg.m * cfg.n * cfg.k


def _torch_rand_bf16(shape, seed, low, high):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensor = torch.rand(shape, generator=generator, dtype=torch.float32)
    tensor = tensor * (high - low) + low
    return tensor.to(torch.bfloat16)


def _to_device_tile(tensor, device):
    return ttnn.as_tensor(
        tensor,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
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


def _validate_matmul_shape(cfg):
    if cfg.m % 32 != 0 or cfg.n % 32 != 0 or cfg.k % 32 != 0:
        raise ValueError(f"matmul dimensions must be divisible by 32, got M={cfg.m}, N={cfg.n}, K={cfg.k}")


def _build_dense_inputs(cfg, low, high):
    #a = _torch_rand_bf16((1, 1, cfg.m, cfg.k), seed=123, low=low, high=high)
    a = torch.ones((1, 1, cfg.m, cfg.k), dtype=torch.bfloat16)
    b = _torch_rand_bf16((1, 1, cfg.k, cfg.n), seed=1234, low=low, high=high)
    return a, b


def _make_compute_kernel_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
        throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
    )


def _official_matmul_kwargs(grid_size):
    return {
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "dtype": ttnn.bfloat16,
        "compute_kernel_config": _make_compute_kernel_config(),
        "output_tile": ttnn.Tile([32, 32]),
        "core_grid": ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
    }


def _run_official_matmul(device, a, b, grid_size):
    start = time.perf_counter()
    a_dev, b_dev = _prepare_matmul_inputs(device, a, b)
    _log(f"  official: device upload done in {time.perf_counter() - start:.3f}s")

    start = time.perf_counter()
    out_dev = ttnn.matmul(
        a_dev,
        b_dev,
        **_official_matmul_kwargs(grid_size),
    )
    ttnn.synchronize_device(device)
    _log(f"  official: launch/sync done in {time.perf_counter() - start:.3f}s")
    return ttnn.to_torch(out_dev).to(torch.float32)


def _run_generated_matmul(device, a, b, cfg):
    start = time.perf_counter()
    _log("  generated: moving A/[K,N] B to device as TTNN TILE tensors")
    a_dev, b_dev, dst_dev = _prepare_generated_matmul_inputs(device, a, b, cfg)
    _log(f"  generated: device upload done in {time.perf_counter() - start:.3f}s")

    start = time.perf_counter()
    _log("  generated: launching host_ttnn.run")
    out_dev = run_generated_matmul(
        a_dev,
        b_dev,
        dst_dev,
    )
    ttnn.synchronize_device(device)
    _log(f"  generated: launch/sync done in {time.perf_counter() - start:.3f}s")
    return ttnn.to_torch(out_dev).to(torch.float32)


def _prepare_matmul_inputs(device, a, b):
    return _to_device_tile(a, device), _to_device_tile(b, device)


def _prepare_generated_matmul_inputs(device, a, b, cfg):
    a_dev, b_dev = _prepare_matmul_inputs(device, a, b)
    return (
        a_dev,
        b_dev,
        ttnn.zeros(
            [1, 1, cfg.m, cfg.n],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )


def _benchmark(device, run_once, warmup, iters, use_trace):
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    if iters <= 0:
        raise ValueError(f"iters must be > 0, got {iters}")

    for _ in range(warmup):
        run_once()
    ttnn.synchronize_device(device)

    profiler.clear()
    if use_trace:
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(iters):
            run_once()
        ttnn.end_trace_capture(device, tid, cq_id=0)

        profiler.start("run")
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        profiler.end("run")
        ttnn.release_trace(device, tid)
        return profiler.get("run") / iters

    profiler.start("run")
    for _ in range(iters):
        run_once()
    ttnn.synchronize_device(device)
    profiler.end("run")
    return profiler.get("run") / iters


def _benchmark_official_matmul(device, a, b, grid_size, warmup, iters, use_trace):
    a_dev, b_dev = _prepare_matmul_inputs(device, a, b)
    def run_once():
        return ttnn.matmul(
            a_dev,
            b_dev,
            **_official_matmul_kwargs(grid_size),
        )

    return _benchmark(device, run_once, warmup, iters, use_trace)


def _effective_generated_warmup(warmup, use_trace):
    return max(warmup, 1) if use_trace else warmup


def _benchmark_generated_matmul(device, a, b, cfg, warmup, iters, use_trace):
    a_dev, b_dev, dst_dev = _prepare_generated_matmul_inputs(device, a, b, cfg)
    effective_warmup = _effective_generated_warmup(warmup, use_trace)

    def run_once():
        return run_generated_matmul(
            a_dev,
            b_dev,
            dst_dev,
        )

    return _benchmark(device, run_once, effective_warmup, iters, use_trace)


def _report_performance(label, cfg, grid_size, avg_seconds, warmup, iters, use_trace):
    avg_ms = avg_seconds * 1000.0
    tflops = _get_matmul_tops(cfg) / avg_seconds / 1e12

    _log(f"{label} performance:")
    _log(f"  Grid: {grid_size}")
    _log(f"  Warmup iterations: {warmup}")
    _log(f"  Measurement iterations: {iters}")
    _log(f"  Trace capture: {'enabled' if use_trace else 'disabled'}")
    _log(f"  Latency: {avg_ms:.6f} ms")
    _log(f"  Matmul TFLOPS: {tflops:.6f}")


def _report_mlir_speedup(official_avg_seconds, mlir_avg_seconds):
    speedup = official_avg_seconds / mlir_avg_seconds
    _log("MLIR speedup over official ttnn.matmul:")
    _log(f"  Speedup: {speedup:.6f}x")
    _log(f"  Official latency: {official_avg_seconds * 1000.0:.6f} ms")
    _log(f"  MLIR latency: {mlir_avg_seconds * 1000.0:.6f} ms")


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


def _get_device_params(args):
    return {
        "l1_small_size": args.l1_small_size,
        "trace_region_size": args.trace_region_size,
    }


def _run_official_perf_target(args, cfg, a, b):
    device = ttnn.CreateDevice(device_id=args.device_id, **_get_device_params(args))
    try:
        grid_size = _get_grid_size(device, args.grid_x, args.grid_y)
        avg_seconds = _benchmark_official_matmul(
            device=device,
            a=a,
            b=b,
            grid_size=grid_size,
            warmup=args.warmup,
            iters=args.iters,
            use_trace=not args.no_trace,
        )
        _report_performance(
            "Official ttnn.matmul",
            cfg,
            grid_size,
            avg_seconds,
            args.warmup,
            args.iters,
            not args.no_trace,
        )
        return avg_seconds
    finally:
        ttnn.close_device(device)


def _run_generated_perf_target(args, cfg, a, b):
    device = ttnn.CreateDevice(device_id=args.device_id, **_get_device_params(args))
    try:
        grid_size = _get_grid_size(device, args.grid_x, args.grid_y)
        avg_seconds = _benchmark_generated_matmul(
            device=device,
            a=a,
            b=b,
            cfg=cfg,
            warmup=args.warmup,
            iters=args.iters,
            use_trace=not args.no_trace,
        )
        _report_performance(
            "Generated host_ttnn matmul",
            cfg,
            grid_size,
            avg_seconds,
            _effective_generated_warmup(args.warmup, not args.no_trace),
            args.iters,
            not args.no_trace,
        )
        return avg_seconds
    finally:
        ttnn.close_device(device)


def run_performance(args, cfg, a, b):
    official_avg_seconds = None
    mlir_avg_seconds = None

    if args.perf_target in ("official", "both"):
        official_avg_seconds = _run_official_perf_target(args, cfg, a, b)
    if args.perf_target in ("mlir", "both"):
        mlir_avg_seconds = _run_generated_perf_target(args, cfg, a, b)

    if official_avg_seconds is not None and mlir_avg_seconds is not None:
        _report_mlir_speedup(official_avg_seconds, mlir_avg_seconds)


def run_correctness_check(args, cfg, a, b):
    device_params = _get_device_params(args)

    generated_device = ttnn.CreateDevice(device_id=args.device_id, **device_params)
    try:
        generated_grid = _get_grid_size(generated_device, args.grid_x, args.grid_y)
        _log("Running generated host_ttnn matmul...")
        generated = _run_generated_matmul(
            device=generated_device,
            a=a,
            b=b,
            cfg=cfg,
        )
    finally:
        ttnn.close_device(generated_device)

    official_device = ttnn.CreateDevice(device_id=args.device_id, **device_params)
    try:
        official_grid = _get_grid_size(official_device, args.grid_x, args.grid_y)
        _log("Running official ttnn.matmul...")
        official = _run_official_matmul(
            device=official_device,
            a=a,
            b=b,
            grid_size=official_grid,
        )

        stats_official_vs_generated = _comparison_stats(official, generated)
        pcc = stats_official_vs_generated["pcc"]

        _log(f"Shape: M={cfg.m}, N={cfg.n}, K={cfg.k}")
        _log(f"Official grid: {official_grid}")
        _log(f"Generated grid: {generated_grid}")
        _log(f"PCC(ttnn.matmul, generated): {pcc:.9f}")
        _log(f"Max abs diff(ttnn.matmul, generated): {stats_official_vs_generated['max_abs_diff']:.9f}")

        passed = pcc >= args.pcc
        _log(f"Result: {'PASS' if passed else 'FAIL'} (PCC >= {args.pcc})")

        if args.print_values:
            count = min(args.print_values, official.numel())
            _log(f"official[:{count}] = {official.flatten()[:count].tolist()}")
            _log(f"generated[:{count}] = {generated.flatten()[:count].tolist()}")

        if args.raise_on_fail and not passed:
            raise AssertionError(f"PCC {pcc} is below threshold {args.pcc}")
    finally:
        ttnn.close_device(official_device)


def run_matmul_comparison(args):
    if args.skip_correctness and args.skip_perf:
        raise ValueError("nothing to run: --skip_correctness and --skip_perf are both set")

    cfg = MatmulConfig(m=args.m, n=args.n, k=args.k)
    _validate_matmul_shape(cfg)

    a, b = _build_dense_inputs(cfg, low=args.low, high=args.high)

    if not args.skip_correctness:
        run_correctness_check(args, cfg, a, b)
    if not args.skip_perf:
        run_performance(args, cfg, a, b)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run official-vs-generated correctness and/or benchmark the selected matmul target."
        )
    )
    parser.add_argument(
        "shape",
        nargs="?",
        help="Optional shape shorthand like M256_N256_K256.",
    )
    parser.add_argument(
        "--shape",
        dest="shape_spec",
        type=str,
        default=None,
        help="Shape shorthand like M256_N256_K256. Explicit dimension flags override it.",
    )
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--grid_x", type=int, default=8)
    parser.add_argument("--grid_y", type=int, default=8)
    parser.add_argument("--pcc", type=float, default=0.99)
    parser.add_argument("--low", type=float, default=-0.5)
    parser.add_argument("--high", type=float, default=0.5)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--l1_small_size", type=int, default=24576)
    parser.add_argument("--trace_region_size", type=int, default=7520256 * 8)
    parser.add_argument("--print_values", type=int, default=0)
    parser.add_argument("--raise_on_fail", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", "--iter", type=int, default=50)
    parser.add_argument("--no_trace", action="store_true")
    parser.add_argument("--skip_correctness", action="store_true", help="Skip the official TTNN comparison.")
    parser.add_argument("--skip_perf", action="store_true", help="Skip performance benchmarking.")
    parser.add_argument("--perf_target", choices=("official", "mlir", "both"), default="mlir")
    return _apply_shape_spec_args(parser.parse_args(), parser)


if __name__ == "__main__":
    run_matmul_comparison(parse_args())