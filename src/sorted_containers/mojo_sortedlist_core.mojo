"""
Standalone Mojo implementation of a simple sorted list.

This avoids Python interop completely so we can validate the data
structure and algorithm in pure Mojo before wiring it up to Python.

We start with a generic `SortedList[T]` over orderable, copyable types
(e.g. `Int`, `Float64`).
"""


struct SortedList[T: Comparable & ImplicitlyCopyable & Copyable & Movable](Movable):
    var data: List[T]
    var size: Int

    fn __init__(out self):
        self.data = List[T]()
        self.size = 0

    fn __len__(self) -> Int:
        return self.size

    fn _bisect_left(self, value: T) -> Int:
        """Binary-search insertion position for `value`.

        Returns the index where `value` should be inserted to keep
        the list sorted, using a left-biased (stable) insertion
        point.
        """
        var lo = 0
        var hi = self.size
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.data[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        return lo

    fn add(mut self, value: T):
        """Insert value while keeping the list sorted (stable)."""
        var pos = self._bisect_left(value)

        # Grow list by one, then shift elements right from the end.
        self.data.append(value)
        var i = self.size
        while i > pos:
            self.data[i] = self.data[i - 1]
            i -= 1
        self.data[pos] = value
        self.size += 1

    fn remove(mut self, value: T) raises:
        """Remove a single matching value (first position)."""
        var pos = self._bisect_left(value)

        if pos >= self.size or self.data[pos] != value:
            raise Error("Value not in list")

        var i = pos
        while i < self.size - 1:
            self.data[i] = self.data[i + 1]
            i += 1
        self.size -= 1

    fn get_item(self, index: Int) raises -> T:
        var idx = index
        if idx < 0:
            idx += self.size
        if idx < 0 or idx >= self.size:
            raise Error("Index out of range")
        return self.data[idx]

    fn contains(self, value: T) -> Bool:
        var pos = self._bisect_left(value)
        if pos >= self.size:
            return False
        return self.data[pos] == value
