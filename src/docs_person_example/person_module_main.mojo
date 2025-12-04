"""
Standalone Mojo-only test harness for the `Person` interop example.

This file is kept separate from `person_module.mojo` so that the
extension module (used via `mojo.importer` from Python) stays clean and
only defines `Person` + `PyInit_person_module`. That keeps the Python
interop story simple and avoids any confusion about `main()` when Python
is just trying to import the module.

You can run this directly to sanity-check the Mojo struct behaviour:

    uv run mojo src/docs_person_example/person_module_main.mojo
"""

from person_module import Person


fn main():
    var p = Person("Alice", 30)
    print("Object (repr):", repr(p))
    print("Name:", p.name)
    print("Age:", p.age)
