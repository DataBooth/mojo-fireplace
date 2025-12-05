import sys
import mojo.importer  # Required for on-the-fly compilation

sys.path.insert(0, ".")  # Or your dir with .mojo files

import ld_list  # Triggers build to __mojocache__/ld_list.mojo.so

# Usage
print(ld_list.distance("hello", "world"))  # Returns 4 (as int)
