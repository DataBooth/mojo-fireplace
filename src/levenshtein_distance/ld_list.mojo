from python import Python
from collections.list import List
from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort


struct Levenshtein:
    @staticmethod
    fn distance(s1: String, s2: String) -> Int:
        m = len(s1)
        n = len(s2)
        # Two rows for O(min(m,n)) space
        var prev = List[Int](length=n + 1, fill=0)
        var curr = List[Int](length=n + 1, fill=0)
        # Initialize first row
        for j in range(n + 1):
            prev[j] = j
        # Fill DP table row-by-row
        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,  # deletion
                    curr[j - 1] + 1,  # insertion
                    prev[j - 1] + cost,  # substitution
                )
            # Swap rows
            temp = prev^
            prev = curr^
            curr = temp^
        return prev[n]


fn distance(py_s1: PythonObject, py_s2: PythonObject) raises -> PythonObject:
    s1 = String(py_s1)
    s2 = String(py_s2)
    result = Levenshtein.distance(s1, s2)
    # Int auto-converts to Python int
    return PythonObject(result)


# Python entry point – matches your working person_module pattern
@export
fn PyInit_ld_list() -> PythonObject:
    try:
        var mb = PythonModuleBuilder("ld_list")

        # Expose the top-level function (def_function, not add_function)
        mb.def_function[distance]("distance")

        return mb.finalize()
    except:
        return abort[PythonObject](String("Failed to create ld_list module: "))
