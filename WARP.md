# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository overview

`mojo-fireplace` is a collection of paired Python and Mojo examples, aimed at helping Python developers incrementally adopt Mojo for performance-critical workloads. Each example lives under `src/` in its own directory and typically includes:

- A clear, idiomatic Python baseline
- One or more Mojo implementations (often progressively optimised)
- Optional benchmark / visualisation code and documentation

Current major areas:

- `src/advent_of_code/`: Advent of Code Day 1 "dial rotation" puzzle, showing a near one-to-one Python → Mojo translation
- `src/game_of_life/`: Conway’s Game of Life performance study with multiple Mojo versions, NumPy, and GPU variants, plus a rich documentation set
- `src/black_scholes/`: Monte Carlo option pricing in Python and Mojo, highlighting numerical performance and Python interop
- `src/sorted_containers/`: Experimental Mojo-backed `SortedList` implementation exposed to Python

The repository is managed as a regular Python project (see `pyproject.toml`) with additional Mojo source alongside.

## Tooling & environment

- Python: configured for Python `>=3.14` in `pyproject.toml`. Use `uv` for environment and dependency management.
- Core Python dependencies (declared):
  - `modular` (Mojo SDK / tooling integration)
  - `pytest` (tests)
  - `sortedcontainers` (reference implementation for `SortedList` tests)
- Example-specific extras (not declared in `pyproject.toml`, install as needed with `uv add`):
  - Game of Life benchmarks/docs: `numpy`, `matplotlib`, `loguru`, `tomli`/`tomllib` (Python 3.11+ has `tomllib` built in)
  - Black–Scholes comparisons: `scipy` (for `scipy.stats.norm`)
  - Visual Game of Life: `pygame`
  - GPU Game of Life (Python side): `torch` with Metal/MPS support on macOS

Mojo itself must be installed via Modular’s tooling (`modular install mojo`); commands below assume a working `mojo` on `PATH`.

## Common commands

All commands are relative to the repository root unless stated otherwise.

### Environment setup

```bash
# Install Python dependencies from pyproject + uv.lock
uv sync
```

### Running examples

#### Advent of Code Day 1

```bash
cd src/advent_of_code

# Python implementation
uv run day1.py          # uses test input / debug mode by default

# Mojo implementation
mojo day1.mojo         # mirrors Python logic and debug behaviour
```

#### Game of Life benchmark suite

```bash
cd src/game_of_life

# (Optional) install benchmark extras if you want plots/logging
uv add numpy matplotlib loguru

# Run all enabled implementations as configured in benchmark_config.toml
uv run python benchmark_grid_all_versions.py

# Adjust settings / enabled implementations via benchmark_config.toml
# (rows, cols, generations, which Python/Mojo/GPU backends to include, etc.)
```

To run only the Mojo benchmarks, set `run_python = false` in `[features]` in `benchmark_config.toml`. To include GPU variants, enable the relevant implementations in the `[[implementations]]` section and ensure your GPU/toolchain meets the documented requirements.

#### Game of Life visual demo

```bash
cd src/game_of_life

# Python visualiser (pygame window)
uv add pygame
uv run life.py

# Mojo visualiser (calls into pygame via Python interop)
mojo life.mojo
```

#### Black–Scholes Monte Carlo

```bash
cd src/black_scholes

# Python baseline
uv add scipy
uv run monte_carlo_options.py

# Mojo single-threaded + parallel implementation with Python/scipy comparison
mojo monte_carlo_options.mojo
```

### Tests

Tests currently target the Mojo-backed `SortedList` experiment.

```bash
# Run the full pytest suite
uv run pytest

# Run a single test file
uv run pytest tests/test_sortedlist_basic.py

# Run a single test case
uv run pytest tests/test_sortedlist_basic.py::test_add_and_order
```

`tests/test_sortedlist_basic.py` expects a built Mojo extension module `mojo_sortedlist` exposing `MojoSortedListPy`. The Python wrapper in `src/sorted_containers/__init__.py` will raise a clear `RuntimeError` if the extension is not available. Build and install the Mojo extension following Modular’s "Mojo from Python" guidelines before relying on these tests.

### Linting and formatting

There is no project-wide linter or formatter configured in `pyproject.toml`. If you introduce tooling (e.g. Ruff, Black), prefer wiring it through `uv run ...` so it integrates cleanly with the existing environment.

## High-level architecture

### Overall layout

- `pyproject.toml` defines a minimal Python project with shared dependencies for all examples.
- `src/` is organised by example domain rather than as a single cohesive package; each subdirectory is largely self-contained.
- Mojo files (`*.mojo`) live alongside their Python counterparts to make it easy to compare implementations line by line.
- Benchmarks and visualisers are colocated with the code they exercise (`src/game_of_life` and `src/black_scholes`).

### Game of Life performance study (`src/game_of_life`)

This is the most architecturally involved part of the repo.

Key components:

- **Config-driven benchmark runner (`benchmark_grid_all_versions.py`)**
  - Loads `benchmark_config.toml` to determine grid shape, number of generations, seed, and which implementations to run.
  - Generates an initial grid CSV (`initial_grid.csv`) that is shared across all implementations for fair comparison.
  - For Python implementations:
    - Dynamically imports the configured module/class (e.g. `gridv1.Grid`, `gridv1_np.GridNP`).
    - Runs warm-up generations followed by timed generations.
    - Collects timings and per-implementation fingerprints via each grid’s `fingerprint()` method.
  - For Mojo implementations:
    - Treats `run_grid_bench.mojo` as a **template**.
    - Rewrites the import (`from gridv1 import Grid`) to point at the configured Mojo grid module (e.g. `gridv5`).
    - Rewrites `alias` constants (rows, cols, generations, etc.) from `benchmark_config.toml`.
    - Writes a generated Mojo file per implementation and executes it via `mojo run`, parsing `Mojo_time:` and `Mojo_fingerprint_str:` from stdout.
  - Produces:
    - Console summary table with timings and speedups vs baseline.
    - Optional CSV output of all runs into `benchmark_results/`.
    - Log file with detailed run information.

- **Grid implementations (Python and Mojo)**
  - Python:
    - `gridv1.py`: straightforward `List[List[int]]` implementation with wrap-around indexing; provides `evolve()` and `fingerprint()`.
    - `gridv1_np.py`: NumPy implementation using `np.roll` for wrap-around and vectorised neighbour counting.
  - Mojo (all parameterised by `rows` and `cols`):
    - `gridv1.mojo`: idiomatic, list-of-lists baseline mirroring the Python version.
    - `gridv2.mojo`: flat `UnsafePointer[Int8]` storage and bitwise rules, still single-threaded.
    - `gridv3.mojo`: adds row-level parallelisation via `algorithm.parallelize`.
    - `gridv4.mojo`: optimises pointer arithmetic and cache behaviour using precomputed row pointers.
    - `gridv5.mojo`: further optimises edge handling to remove most modulo operations, splitting left/middle/right cases.
    - `gridv6_gpu.mojo`: GPU-accelerated version targeting Apple Silicon via Mojo’s GPU APIs, with a CPU fallback path.

- **Shared contracts**
  - All grid types (Python and Mojo) expose a way to evolve the grid (`evolve`/`evolve_gpu`) and to produce a fingerprint:
    - Python: `Grid.fingerprint() -> str` (SHA-256 of the flattened 0/1 grid).
    - Mojo: `Grid.fingerprint_str() -> String` (flattened 0/1 string, hashed in Python).
  - The benchmark runner treats these fingerprints as the correctness oracle and will fail the run if any implementation disagrees with the reference.

- **Template runner (`run_grid_bench.mojo`)**
  - Reads `initial_grid.csv` from the working directory.
  - Instantiates `Grid[rows, cols]` and runs warm-up plus timed generations.
  - Prints timing and fingerprint in a machine-readable format that the Python benchmark parses.
  - Is never edited per version; all per-version differences are injected by the Python script via string replacement and config.

- **Visual front-ends**
  - `life.py` is a pygame-based viewer that steps `Grid` forward in time while drawing to a window.
  - `life.mojo` demonstrates Mojo’s Python interop by calling into `pygame` from Mojo and delegating drawing/event handling to Python APIs.

The net effect is a single, unified benchmark harness that can grow to include new grid implementations (Python, Mojo CPU, Mojo GPU, or other languages) by editing `benchmark_config.toml` only, with correctness enforced via shared fingerprints.

### Sorted containers experiment (`src/sorted_containers`)

This directory explores how to implement a performance-critical data structure in Mojo and surface it as a drop-in Python replacement.

Layers:

- **Core Mojo type (`mojo_sortedlist.mojo`)**
  - `MojoSortedList[T]` manages a contiguous buffer via `UnsafePointer[T]` with manual capacity/size tracking.
  - Provides basic operations:
    - `add` inserts via binary search (`_bisect_left`) and a tail copy to maintain sorted order.
    - `remove` removes a single value by shifting the tail left.
    - `__len__`, `__getitem__`, and `__contains__` integrate with Mojo’s standard trait expectations.
  - `MojoSortedListPy` wraps `MojoSortedList[PythonObject]` to provide a Python-friendly container.
  - `PyInit_mojo_sortedlist()` builds a Python extension module via `PythonModuleBuilder`, exporting `MojoSortedListPy`.

- **Python wrapper (`sorted_containers/__init__.py`)**
  - `SortedList` is a minimal, Pythonic façade over the Mojo type, designed to approximate `sortedcontainers.SortedList`:
    - Construction from an iterable.
    - `add`, `remove`, indexing, iteration, and `len()`.
    - `insert` is currently implemented as `add` to preserve sorted order.
  - The module attempts to import the compiled extension `mojo_sortedlist`; if missing, it raises a `RuntimeError` with a clear message.

- **Tests (`tests/test_sortedlist_basic.py`)**
  - Use `pytest` and, when available, the upstream `sortedcontainers.SortedList` as a reference implementation.
  - Mark all tests as skipped if `sortedcontainers` is not installed.
  - Compare ordering, removal semantics, indexing, and basic iteration between the Python and Mojo-backed versions.

This stack demonstrates the "Mojo core + thin Python shell" pattern, with tests codifying behavioural parity with an established Python library.

### Advent of Code Day 1 (`src/advent_of_code`)

This directory is intentionally small but illustrative:

- `day1.py` is the clean, idiomatic Python solution using modular arithmetic to wrap a dial around a 100-position circle. It separates parsing (`load_rotations`), state update (`update_position`), and summary/printing.
- `day1.mojo` is a near-direct translation that keeps the structure and naming familiar to Python developers while adopting Mojo syntax (`alias` constants, explicit types, `List[Int]`).
- `day1_initial.py` captures an earlier, more verbose prototype; the refined versions intentionally simplify logic down to the core modular arithmetic idea.

Future Mojo ports of Advent of Code problems are expected to follow the same pattern: start from a clear Python baseline, then introduce Mojo-centric improvements while keeping the diff small and understandable.

### Black–Scholes Monte Carlo (`src/black_scholes`)

The Black–Scholes example showcases numerically heavy Monte Carlo simulations and how Mojo can accelerate them while still leaning on Python’s ecosystem.

- `monte_carlo_options.py`:
  - Pure Python baseline implementation that simulates geometric Brownian motion paths and prices a European call.
  - Uses a Box–Muller transform for normal variates and reports timing and confidence intervals.
  - Optionally compares the Monte Carlo estimate to the analytical Black–Scholes price via `scipy.stats.norm`.

- `monte_carlo_options.mojo`:
  - Implements the same algorithm in Mojo, with both single-threaded and `parallelize`-based variants.
  - Uses Mojo’s `random_float64`, `sqrt`, `log`, `cos`, and `exp` primitives to compute paths.
  - Benchmarks both variants with `perf_counter_ns`, reporting simulations per second and parallel speed-up.
  - Calls into Python (`math` and `scipy.stats`) to compute the analytical Black–Scholes price, then compares the Mojo Monte Carlo price to the Python result, demonstrating interop rather than reimplementing Black–Scholes in Mojo.

The design here is representative of a general pattern used across the repo: keep Python for ecosystem leverage, push inner numeric loops into Mojo, and wire them together with minimal interop boilerplate.
