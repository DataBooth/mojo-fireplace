"""Quick-and-dirty benchmark comparing `sortedcontainers.SortedList`
against the Mojo-backed `mojo_sortedlist.SortedList`.

This is intentionally simple and focuses on bulk `add` of random
integers so we can get an initial feel for relative performance.

Run with:

    uv run python src/sorted_containers/benchmark_sortedlist.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import mojo.importer  # type: ignore[import-not-found]
from sortedcontainers import SortedList as PySortedList  # type: ignore[import-not-found]

# Make sure we can import the Mojo extension module directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOJO_SORTEDLIST_DIR = PROJECT_ROOT / "src" / "sorted_containers"
if str(MOJO_SORTEDLIST_DIR) not in sys.path:
    sys.path.insert(0, str(MOJO_SORTEDLIST_DIR))

import mojo_sortedlist  # type: ignore[import-not-found]


def bench_build(label: str, cls, values: list[int]) -> float:
    """Time building a sorted list by repeated ``add`` calls."""

    t0 = time.perf_counter()
    s = cls()
    for v in values:
        s.add(v)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    ops_per_sec = len(values) / elapsed if elapsed > 0 else float("inf")
    print(f"{label:30s}: {elapsed:8.3f}s  ({ops_per_sec:12.0f} ops/s)")
    return elapsed


def main() -> None:
    # Problem size; tweak as desired.
    n = 100_000
    seed = 42

    print(f"Benchmarking bulk add of {n:,} random integers (seed={seed})")

    random.seed(seed)
    values = [random.randint(0, 1_000_000) for _ in range(n)]

    # Warm-up runs (small) to trigger any one-off overheads.
    warm_values = values[:1_000]
    bench_build("python SortedList (warmup)", PySortedList, warm_values)
    bench_build("mojo   SortedList (warmup)", mojo_sortedlist.SortedList, warm_values)
    print()

    # Real benchmark.
    t_py = bench_build("python SortedList", PySortedList, values)
    t_mojo = bench_build("mojo   SortedList", mojo_sortedlist.SortedList, values)

    print()
    if t_mojo > 0:
        print(f"Speedup (python / mojo): {t_py / t_mojo:0.2f}x")
    else:
        print("Mojo time is ~0; speedup is effectively infinite for this run.")


if __name__ == "__main__":  # pragma: no cover - manual benchmark
    main()
