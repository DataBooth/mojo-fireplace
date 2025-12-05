"""Basic behavioural parity tests for Mojo-backed SortedList.

These tests are intentionally small but compare our implementation
against the reference `sortedcontainers.SortedList` where available.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

# Quick path tweak so `src/sorted_containers` is importable without
# installing the package. We import the Mojo extension module
# `mojo_sortedlist.mojo` directly via `mojo.importer`.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MOJO_SORTEDLIST_DIR = PROJECT_ROOT / "src" / "sorted_containers"
if str(MOJO_SORTEDLIST_DIR) not in sys.path:
    sys.path.insert(0, str(MOJO_SORTEDLIST_DIR))

import mojo.importer  # type: ignore[import-not-found]

try:
    # Reference implementation from the upstream package.
    from sortedcontainers import SortedList as PySortedList  # type: ignore
except Exception:  # pragma: no cover
    PySortedList = None  # type: ignore[assignment]

import mojo_sortedlist  # provided by mojo.importer


class MojoSortedList:
    """Thin Python wrapper over the Mojo `SortedList` extension type.

    This gives us normal Python special methods (`len()`, indexing,
    iteration, and membership) by delegating to the underlying
    Mojo-implemented methods.
    """

    def __init__(self, iterable: list[int] | None = None) -> None:
        self._inner = mojo_sortedlist.SortedList()
        if iterable is not None:
            for v in iterable:
                self.add(v)

    def add(self, value: int) -> None:
        self._inner.add(value)

    def remove(self, value: int) -> None:
        self._inner.remove(value)

    def __len__(self) -> int:
        return self._inner.__len__()

    def __getitem__(self, index: int) -> int:
        return self._inner.__getitem__(index)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


pytestmark = pytest.mark.skipif(
    PySortedList is None,
    reason="sortedcontainers package not installed; install to run parity tests",
)


@pytest.mark.parametrize("values", [
    [],
    [3, 1, 2],
    list(range(10)),
    list(range(10))[::-1],
])
def test_add_and_order(values: list[int]) -> None:
    py = PySortedList()  # type: ignore[call-arg]
    mojo = MojoSortedList()

    for v in values:
        py.add(v)
        mojo.add(v)

    assert list(py) == list(mojo)


def test_remove_and_membership() -> None:
    values = [5, 1, 3, 3, 2]

    py = PySortedList(values)  # type: ignore[call-arg]
    mojo = MojoSortedList()
    for v in values:
        mojo.add(v)

    # Remove a value present in the list
    py.remove(3)
    mojo.remove(3)

    assert list(py) == list(mojo)
    assert 3 in py
    assert 3 in mojo

    # Removing a value not in the list should raise ValueError on both sides
    with pytest.raises(ValueError):
        py.remove(42)

    with pytest.raises(Exception):  # Mojo side currently raises a generic error
        mojo.remove(42)


@pytest.mark.parametrize("index", [0, -1])
def test_indexing_simple(index: int) -> None:
    values = [10, 1, 7]

    py = PySortedList(values)  # type: ignore[call-arg]
    mojo = MojoSortedList()
    for v in values:
        mojo.add(v)

    assert py[index] == mojo[index]


def test_len_and_iter() -> None:
    values = [4, 2, 9]

    py = PySortedList(values)  # type: ignore[call-arg]
    mojo = MojoSortedList(values)

    assert len(py) == len(mojo)
    assert list(py) == list(mojo)
