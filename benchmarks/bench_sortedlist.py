"""Simple micro-benchmark for Mojo-backed SortedList vs sortedcontainers.

Run with something like:

    uv run benchmarks/bench_sortedlist.py

This is intentionally minimal and not a rigorous benchmark suite.
"""

from __future__ import annotations

import time

try:
    from sortedcontainers import SortedList as PySortedList  # type: ignore
except Exception:  # pragma: no cover
    PySortedList = None  # type: ignore[assignment]

from sorted_containers import SortedList as MojoSortedList


def bench(label: str, cls, n: int) -> None:
    start = time.perf_counter()
    sl = cls()
    for i in range(n):
        sl.add(i)
    duration = time.perf_counter() - start
    print(f"{label:24s} n={n:8d}  {duration:8.4f}s")


def main() -> None:
    n = 100_000

    if PySortedList is None:
        print("sortedcontainers is not installed; skipping Python baseline.")
    else:
        bench("Python SortedList", PySortedList, n)

    bench("Mojo SortedList", MojoSortedList, n)


if __name__ == "__main__":  # pragma: no cover
    main()
