"""
Mojo-backed SortedList core for Python interop.

This module exposes a single Python-visible type, `SortedList`, that
wraps the pure-Mojo `SortedList[Int]` from `mojo_sortedlist_core`.

The design mirrors the working `Person` example in `src/docs_person_example`:

- `MojoIntSortedList` is a small value type with an `inner` field that
  holds the real `SortedList[Int]` data structure.
- A custom `py_init` static method provides the Python constructor
  behaviour for `SortedList()`.
- Additional static methods (`py_add`, `py_remove`, `py_len`,
  `py_get_item`) are bound as Python methods using
  `PythonModuleBuilder.def_method`.

Python never sees `SortedList[Int]` directly – it only works with the
`MojoIntSortedList` wrapper, which is exported to Python under the
simpler name `SortedList`.
"""

from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort

from mojo_sortedlist_core import SortedList


@fieldwise_init
struct MojoIntSortedList(Movable, Representable):
    """Thin Mojo wrapper around `SortedList[Int]` exposed to Python.

    The Python extension layer only ever sees this type. All the real
    sorted-list behaviour lives in the generic `SortedList[T]` core.
    """

    var inner: SortedList[Int]

    fn __init__(out self):
        self.inner = SortedList[Int]()

    fn __repr__(self) -> String:
        return String("SortedList(len=", self.inner.__len__(), ")")

    # --- Python construction -------------------------------------------------

    @staticmethod
    fn py_init(
        out self: MojoIntSortedList, args: PythonObject, kwargs: PythonObject
    ) raises:
        """Python-facing constructor for `SortedList`.

        We currently only support a no-argument constructor. The Python
        tests build up the contents by calling `add` repeatedly.
        """

        if len(args) != 0:
            raise Error("SortedList() currently takes no positional arguments")

        self = Self()

    # --- Python-bound methods ------------------------------------------------

    @staticmethod
    fn py_add(py_self: PythonObject, value_obj: PythonObject) raises -> PythonObject:
        """Bound as `SortedList.add(self, value)` on the Python side."""
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var v = Int(value_obj)
        self_ptr[].inner.add(v)
        return PythonObject(None)

    @staticmethod
    fn py_remove(py_self: PythonObject, value_obj: PythonObject) raises -> PythonObject:
        """Bound as `SortedList.remove(self, value)` in Python."""
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var v = Int(value_obj)
        self_ptr[].inner.remove(v)
        return PythonObject(None)

    @staticmethod
    fn py_len(py_self: PythonObject) raises -> PythonObject:
        """Implements `__len__` so Python's `len()` works on the object."""
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var n = self_ptr[].inner.__len__()
        return PythonObject(n)

    @staticmethod
    fn py_get_item(
        py_self: PythonObject, index_obj: PythonObject
    ) raises -> PythonObject:
        """Implements `__getitem__` for index-based access from Python."""
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var idx = Int(index_obj)
        var v = self_ptr[].inner.get_item(idx)
        return PythonObject(v)


# --- Python module init ------------------------------------------------------

@export
fn PyInit_mojo_sortedlist() -> PythonObject:
    try:
        var mb = PythonModuleBuilder("mojo_sortedlist")

        _ = mb.add_type[MojoIntSortedList]("SortedList")
            .def_py_init[MojoIntSortedList.py_init]()
            .def_method[MojoIntSortedList.py_add]("add")
            .def_method[MojoIntSortedList.py_remove]("remove")
            .def_method[MojoIntSortedList.py_len]("__len__")
            .def_method[MojoIntSortedList.py_get_item]("__getitem__")

        return mb.finalize()
    except e:
        return abort[PythonObject](
            String("error creating Python Mojo module:", e),
        )
