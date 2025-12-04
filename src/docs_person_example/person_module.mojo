"""
Person interop example used to validate Mojo ↔ Python bindings.

This file is intentionally small and mirrors the patterns used in the
Modular `person_module.mojo` example, but stripped down to the pieces
that matter for this repository:

- Define a `Person` struct in Mojo
- Provide a `py_init` static method so Python can construct `Person`
  instances with normal-looking syntax: `Person("Alice", 30)`
- Expose a simple Mojo method (`get_age`) to Python via
  `PythonModuleBuilder.def_method`
- Export a `PyInit_person_module` function so Python’s import machinery
  (and `mojo.importer`) can treat this as an extension module

There is deliberately **no** `main()` here used by Python. A separate
file `person_module_main.mojo` is used for Mojo-only testing so we don’t
couple the extension module to an executable entry point.
"""

from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort

# ---------------------------------------------------------------------------
# Core Mojo type
# ---------------------------------------------------------------------------

@fieldwise_init
struct Person(Movable, Representable):
    var name: String
    var age: Int

    fn __repr__(self) -> String:
        return String("Person(", self.name, ", ", self.age, ")")

    @staticmethod
    fn py_init(
        out self: Person, args: PythonObject, kwargs: PythonObject
    ) raises:
        """Python-facing constructor.

        This gives Python its `__init__`-like behaviour. It is registered
        below with `def_py_init[Person.py_init]()`, which makes this
        usable from Python as:

            import person_module
            p = person_module.Person("Alice", 30)
        """

        # Expect exactly two positional arguments: (name, age)
        if len(args) != 2:
            raise Error("Person() takes exactly 2 arguments")

        var name = String(args[0])
        var age = Int(args[1])

        self = Self(name, age)

    @staticmethod
    fn get_age(py_self: PythonObject) raises -> PythonObject:
        """Simple example of a Mojo method exposed to Python.

        Called from Python as:
            p = person_module.Person("Alice", 30)
            p.get_age()  # -> 30
        """
        var self_ptr = py_self.downcast_value_ptr[Self]()
        return PythonObject(self_ptr[].age)

# ---------------------------------------------------------------------------
# Python extension module entry point
# ---------------------------------------------------------------------------

@export
fn PyInit_person_module() -> PythonObject:
    try:
        var mb = PythonModuleBuilder("person_module")

        _ = mb.add_type[Person]("Person")
            .def_py_init[Person.py_init]()
            .def_method[Person.get_age]("get_age")

        return mb.finalize()
    except e:
        return abort[PythonObject](
            String("error creating Mojo module:", e),
        )
