"""Pipeline step timer using a context manager."""

import time
from contextlib import contextmanager
from typing import Generator

_timings: list[tuple[str, float]] = []


@contextmanager
def PipelineTimer(name: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _timings.append((name, elapsed_ms))


def print_timing_summary() -> None:
    if not _timings:
        return
    col_w = max(len(n) for n, _ in _timings) + 2
    total = sum(ms for _, ms in _timings)
    width = col_w + 24
    print()
    print("=" * width)
    print("PIPELINE TIMING SUMMARY")
    print("=" * width)
    print(f"  {'Step':<{col_w}} {'Time (ms)':>10}  {'% Total':>8}")
    print("-" * width)
    for name, ms in _timings:
        pct = ms / total * 100 if total > 0 else 0.0
        print(f"  {name:<{col_w}} {ms:>10.1f}  {pct:>7.1f}%")
    print("-" * width)
    print(f"  {'TOTAL':<{col_w}} {total:>10.1f}")
    print("=" * width)
