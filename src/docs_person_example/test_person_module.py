"""Tiny Python smoke test for the Mojo `person_module` example.

This deliberately mirrors the structure that `mojo.importer` expects:

- We add the directory containing `person_module.mojo` to `sys.path`
- We `import mojo.importer` so Python knows how to import `.mojo` files
- We import `person_module` like a normal Python extension module
- We exercise the bound constructor and one bound method

Run with:

    uv run python src/docs_person_example/test_person_module.py
"""

import sys
from pathlib import Path
import mojo.importer

MOJO_IMPORT_PATH = Path(__file__).resolve().parents[2] / "src" / "docs_person_example"
sys.path.insert(0, str(MOJO_IMPORT_PATH))

import person_module

p = person_module.Person("Alice", 30)

print(f"Object (repr) : {p}")
print(f"Type: {type(p)}")
print("Get age method:", p.get_age())
