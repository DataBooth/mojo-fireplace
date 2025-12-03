"""
run_grid_bench.mojo — Unified Grid Benchmark Runner

This single file serves as the template for ALL grid implementations.
The Python benchmark script replaces:
- "from gridv1 import Grid" → "from gridvX import Grid"
- alias values with config values
"""

from gridv1 import Grid
from time import perf_counter_ns
from pathlib import cwd

# These defaults are overridden by benchmark script
alias rows = 256
alias cols = 256
alias generations = 100
alias warmup = 5
alias initial_grid_path = "initial_grid.csv"


fn main() raises:
    print("Grid:", rows, "×", cols, "| Generations:", generations, "| Warmup:", warmup)

    var file = open(cwd() / initial_grid_path, "r")
    var content = file.read()
    if content == "":
        raise Error("Failed to read grid: " + initial_grid_path)

    var lines = content.split("\n")
    var current = Grid[rows, cols]()
    
    # Load initial grid from CSV
    for i in range(rows):
        var line = lines[i].strip()
        if not line:
            continue
        var cells = line.split(",")
        for j in range(cols):
            current[i, j] = atol(cells[j])
    file.close()
    
    # Warmup runs
    for _ in range(warmup):
        current = current.evolve()
    
    # Benchmark
    var start = perf_counter_ns()
    for _ in range(generations):
        current = current.evolve()
    var end = perf_counter_ns()
    
    var duration = Float64(end - start) / 1_000_000_000.0
    print("Mojo_time:", duration)
    print("Mojo_fingerprint_str:", current.fingerprint_str())
