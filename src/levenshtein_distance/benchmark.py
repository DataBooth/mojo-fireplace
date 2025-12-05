import time
import sys
import pandas as pd  # For nice table output

# Note: Install deps with: pip install duckdb rapidfuzz python-Levenshtein polyleven textdistance
# For Mojo: Run `mojo build levenshtein_mojo.mojo` first, then import levenshtein_mojo
# Run this script: python benchmark_all.py


# 1. Pure Python (from the post)
def levenshtein_python(s1, s2):
    """Pure Python Levenshtein Distance - O(m*n) space."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize boundaries
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[m][n]


# 2. DuckDB (requires pip install duckdb)
import duckdb


def levenshtein_duckdb(s1, s2):
    """DuckDB native Levenshtein."""
    # Use a temp connection for single query
    con = duckdb.connect(":memory:")
    result = con.execute(f"SELECT levenshtein('{s1}', '{s2}')").fetchone()[0]
    con.close()
    return result


# 3. rapidfuzz (pip install rapidfuzz)
from rapidfuzz.distance import Levenshtein


def levenshtein_rapidfuzz(s1, s2):
    """RapidFuzz Levenshtein."""
    return Levenshtein.distance(s1, s2)


# 4. python-Levenshtein (pip install python-Levenshtein)
from Levenshtein import distance as levenshtein_c


def levenshtein_c_ext(s1, s2):
    """Python-Levenshtein C extension."""
    return levenshtein_c(s1, s2)


# 5. polyleven (pip install polyleven)
from polyleven import levenshtein as levenshtein_poly


def levenshtein_polyleven(s1, s2):
    """Polyleven Rust impl."""
    return levenshtein_poly(s1, s2)


# 6. textdistance (pip install textdistance) - pure Python fallback
import textdistance


def levenshtein_textdistance(s1, s2):
    """Textdistance Levenshtein (pure Python)."""
    return textdistance.levenshtein(s1, s2)


# 7. Mojo (after building: import levenshtein_mojo)
try:
    import levenshtein_mojo

    def levenshtein_mojo(s1, s2):
        """Mojo compiled version."""
        return levenshtein_mojo.levenshtein_py(s1, s2)
except ImportError:
    print(
        "Warning: Mojo not available (run 'mojo build levenshtein_mojo.mojo'). Skipping."
    )

    def levenshtein_mojo(s1, s2):
        raise NotImplementedError("Mojo module not built.")


# Benchmark function
def bench(name: str, func, s1, s2, repeats: int = 20):
    """Time the function over repeats, return min time."""
    times = []
    # Warmup
    func(s1, s2)
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(s1, s2)
        times.append(time.perf_counter() - t0)
    return min(times)


# Generate comprehensive test cases: vary lengths and similarity
def generate_test_cases():
    return [
        ("hello", "world"),  # Small, dissimilar (5,5)
        ("kitten", "sitting"),  # Small, similar (6,7)
        ("a" * 100, "b" * 100),  # Medium, dissimilar
        ("abc" * 34 + "def", "abc" * 33 + "xyz"),  # Medium, similar (~100 chars)
        ("a" * 500, "b" * 500),  # Large, dissimilar
        ("a" * 1000, "b" * 1000),  # X-Large, dissimilar
    ]


# Run benchmarks
test_cases = generate_test_cases()
results = []

for s1, s2 in test_cases:
    row = {"len1": len(s1), "len2": len(s2), "case": f"{s1[:10]}... vs {s2[:10]}..."}
    baseline_time = None
    baseline_name = "Python"

    # Time each impl
    impls = [
        ("Python", levenshtein_python),
        ("DuckDB", levenshtein_duckdb),
        ("RapidFuzz", levenshtein_rapidfuzz),
        ("C Ext", levenshtein_c_ext),
        ("Polyleven", levenshtein_polyleven),
        ("Textdistance", levenshtein_textdistance),
        ("Mojo", levenshtein_mojo),
    ]

    for name, func in impls:
        try:
            t = bench(name, func, s1, s2)
            row[name + "_s"] = f"{t:.6f}"
            if name == baseline_name:
                baseline_time = t
            if baseline_time:
                row[name + "_speedup"] = f"{baseline_time / t:.0f}x" if t > 0 else "N/A"
        except Exception as e:
            row[name + "_s"] = "Error"
            row[name + "_speedup"] = str(e)

    results.append(row)

# Output as DataFrame table (print and save to CSV)
df = pd.DataFrame(results)
print("\nComprehensive Levenshtein Distance Benchmarks")
print(df.to_string(index=False))

# Save to CSV for plotting/DuckDB
df.to_csv("ld_benchmarks.csv", index=False)
print(
    "\nResults saved to ld_benchmarks.csv. Columns: len1, len2, case, [Impl]_s, [Impl]_speedup"
)

# Optional: Verify correctness on first case (all should return 4 for "hello" vs "world")
print("\nVerification (expected: 4 for 'hello' vs 'world'):")
for name, func in impls:
    try:
        res = func("hello", "world")
        print(f"{name}: {res}")
    except:
        print(f"{name}: Error")
