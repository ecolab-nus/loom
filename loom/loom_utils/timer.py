import time
from contextlib import contextmanager
from typing import Generator


class TimingCollector:
    """Collects and summarizes pipeline step timings."""

    def __init__(self) -> None:
        self._timings: list[tuple[str, float]] = []

    def reset(self) -> None:
        """Clear all collected timings."""
        self._timings = []

    @contextmanager
    def timer(self, name: str) -> Generator[None, None, None]:
        """Context manager to time a pipeline step."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._timings.append((name, elapsed_ms))

    def print_summary(self) -> None:
        """Print a formatted summary of all collected timings."""
        if not self._timings:
            return
        col_w = max(len(n) for n, _ in self._timings) + 2
        total = sum(ms for _, ms in self._timings)
        width = col_w + 24
        print()
        print("=" * width)
        print("PIPELINE TIMING SUMMARY")
        print("=" * width)
        print(f"  {'Step':<{col_w}} {'Time (ms)':>10}  {'% Total':>8}")
        print("-" * width)
        for name, ms in self._timings:
            pct = ms / total * 100 if total > 0 else 0.0
            print(f"  {name:<{col_w}} {ms:>10.1f}  {pct:>7.1f}%")
        print("-" * width)
        print(f"  {'TOTAL':<{col_w}} {total:>10.1f}")
        print("=" * width)


# Default global instances for convenience
_DEFAULT_COLLECTOR = TimingCollector()
pipeline_timer = _DEFAULT_COLLECTOR.timer
print_timing_summary = _DEFAULT_COLLECTOR.print_summary
