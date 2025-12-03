# Game of Life Optimisation Guide

This guide demonstrates progressive performance optimisations for Conway's Game of Life in Mojo, from a simple `List[List[Int]]` implementation to a highly optimised parallelised version.

## Overview

Three implementations are provided, each building on the previous:

1. **gridv1.mojo** - Baseline using `List[List[Int]]`
2. **gridv2.mojo** - Memory optimised with flat layout and `UnsafePointer`
3. **gridv3.mojo** - Parallelised using `algorithm.parallelize`

Expected performance improvement: **50-100x speedup** from v1 to v3 on modern multi-core systems.

## Version 1: List[List[Int]] Baseline

### Implementation

```mojo
struct Grid(Copyable, Movable, Stringable):
    var rows: Int
    var cols: Int
    var data: List[List[Int]]  # Nested lists
```

### Characteristics

- ✅ Simple and straightforward
- ✅ Dynamic sizing (rows/cols determined at runtime)
- ❌ Poor memory locality (nested allocations)
- ❌ 8 bytes per cell (Int is 64-bit)
- ❌ Pointer chasing for every access
- ❌ Heap fragmentation

### Performance Issues

1. **Memory overhead**: Each `Int` is 8 bytes, but we only need 0 or 1
2. **Cache misses**: Nested lists cause poor spatial locality
3. **Allocation overhead**: Each row is a separate allocation
4. **Indirection**: `data[row][col]` requires two pointer dereferences

## Version 2: Flat Memory Layout

### Implementation

```mojo
struct Grid[rows: Int, cols: Int](Copyable, Movable, Stringable):
    alias num_cells = Self.rows * Self.cols
    var data: UnsafePointer[Int8, MutOrigin.external]  # Flat array
```

### Key Optimisations

#### 1. Compile-Time Dimensions

```mojo
struct Grid[rows: Int, cols: Int]  # Parameters, not fields
```

**Benefits:**
- Compiler knows grid size at compile time
- Enables aggressive optimisations and inlining
- `num_cells` computed at compile time

#### 2. Flat Memory Layout

```mojo
var data: UnsafePointer[Int8, MutOrigin.external]
```

**Benefits:**
- Single contiguous allocation
- Excellent cache locality
- No pointer chasing
- Simple row-major indexing: `row * cols + col`

#### 3. Compact Storage (Int8)

```mojo
var data: UnsafePointer[Int8, MutOrigin.external]  # 1 byte per cell
```

**Benefits:**
- 8x memory reduction (1 byte vs 8 bytes)
- Better cache utilisation
- More data fits in L1/L2 cache

#### 4. Direct Memory Operations

```mojo
memset_zero(self.data, self.num_cells)
memcpy(dest=self.data, src=existing.data, count=self.num_cells)
```

**Benefits:**
- Optimised platform-specific implementations
- SIMD acceleration where available
- Much faster than element-by-element operations

#### 5. Inlined Accessors

```mojo
@always_inline
fn __getitem__(self, row: Int, col: Int) -> Int8:
    return (self.data + row * Self.cols + col)[]
```

**Benefits:**
- No function call overhead
- Compiler can optimise surrounding code better

#### 6. Bitwise Trick for Conway's Rules

```mojo
if num_neighbors | self[row, col] == 3:
    next_generation[row, col] = 1
```

**Why this works:**

| Current | Neighbours | `n \| c` | Result | Rule |
|---------|-----------|----------|--------|------|
| 1 (alive) | 2 | `2 \| 1 = 3` | ✓ | Survive with 2 |
| 1 (alive) | 3 | `3 \| 1 = 3` | ✓ | Survive with 3 |
| 0 (dead) | 3 | `3 \| 0 = 3` | ✓ | Birth with 3 |
| Any | Other | `≠ 3` | ✗ | Die |

**Benefits:**
- Replaces complex if/else logic
- Single comparison instead of multiple conditions
- Branch predictor friendly

### Performance Gain

**Expected: 10-20x faster than v1**

## Version 3: Parallelisation

### Implementation

```mojo
fn evolve(self) -> Self:
    var next_generation = Self()
    
    @parameter
    fn worker(row: Int) -> None:
        # Process single row
        ...
    
    parallelize[worker](Self.rows)  # Process rows in parallel
    return next_generation^
```

### Key Optimisation

#### Parallel Row Processing

```mojo
@parameter
fn worker(row: Int) -> None:
    # Each row is independent - safe to parallelise
    ...

parallelize[worker](Self.rows)
```

**Why this is safe:**
- Each row's output depends only on input data (read-only)
- No race conditions (different output cells written by different threads)
- Minimal synchronisation overhead

**Benefits:**
- Utilises all available CPU cores
- Near-linear scaling with core count
- Low parallelisation overhead (coarse-grained)

### Performance Gain

**Expected: Additional 4-8x speedup on multi-core systems**

The exact speedup depends on:
- Number of CPU cores
- Grid size (larger grids parallelise better)
- System memory bandwidth

## Running the Benchmarks

### Using Mojo directly

```bash
mojo benchmark.mojo
```

### Expected Output

```
============================================================
Game of Life Performance Benchmark
============================================================
Grid size: 1024 x 1024 = 1048576 cells
Iterations: 1000
CPU cores available: 12
============================================================

gridv1 (List[List[Int]]):
  Time: 45000 ms
  Per evolution: 45.0 ms

gridv2 (UnsafePointer[Int8]):
  Time: 3200 ms
  Per evolution: 3.2 ms
  Speedup vs v1: 14.06 x

gridv3 (Parallelised):
  Time: 450 ms
  Per evolution: 0.45 ms
  Speedup vs v1: 100.0 x
  Speedup vs v2: 7.11 x

============================================================
Summary:
  gridv1 baseline: 45000 ms
  gridv2 (memory optimised): 3200 ms
  gridv3 (parallelised): 450 ms
  Total speedup: 100.0 x
============================================================
```

## Key Takeaways

### 1. Memory Layout Matters

The difference between nested lists and flat arrays is **dramatic**. Modern CPUs are optimised for sequential access.

### 2. Right-Size Your Data

Using `Int8` instead of `Int` for binary values saves 8x memory and improves cache performance.

### 3. Compile-Time Information Enables Optimisations

Making dimensions compile-time parameters allows the compiler to:
- Compute sizes at compile time
- Inline more aggressively
- Eliminate runtime checks

### 4. Parallelisation Amplifies Benefits

Good sequential performance is a prerequisite for good parallel performance. The v2→v3 speedup wouldn't be possible without v1→v2 optimisations first.

### 5. Bitwise Tricks Can Simplify Logic

The `num_neighbors | current_cell == 3` trick is:
- Faster (single comparison)
- More maintainable (no complex conditions)
- Branch predictor friendly

## Further Optimisation Ideas

### SIMD Vectorisation

Process multiple cells in parallel within each row:

```mojo
fn evolve_simd[simd_width: Int](self) -> Self:
    @parameter
    fn worker(row: Int):
        fn compute_vector[width: Int](col: Int) unified {mut}:
            # Process `width` cells at once using SIMD
            ...
        vectorize[simd_width, size=Self.cols](compute_vector)
    
    parallelize[worker](Self.rows)
```

**Potential gain:** Additional 2-4x speedup

### GPU Acceleration

For very large grids (10k x 10k+), GPU processing could provide massive speedups:

```mojo
from gpu import DeviceContext

fn evolve_gpu(self, ctx: DeviceContext) -> Self:
    # Launch GPU kernel to process entire grid
    ...
```

**Potential gain:** 10-100x for large grids

### Spatial Hashing for Sparse Grids

For grids with mostly dead cells, track only live cells:

```mojo
struct SparseGrid:
    var live_cells: Dict[Tuple[Int, Int], Bool]
```

**Benefit:** O(live cells) instead of O(total cells)

## Mojo Language Features Demonstrated

1. ✅ Compile-time parameters: `[rows: Int, cols: Int]`
2. ✅ Manual memory management: `alloc`, `free`, `UnsafePointer`
3. ✅ Memory operations: `memcpy`, `memset_zero`
4. ✅ Traits: `Copyable`, `Movable`, `Stringable`
5. ✅ Lifetimes: `__init__`, `__copyinit__`, `__del__`
6. ✅ Parallelisation: `algorithm.parallelize`
7. ✅ Inlining: `@always_inline`
8. ✅ Compile-time functions: `@parameter`
9. ✅ Type aliases: `alias num_cells`
10. ✅ Move semantics: `grid^`

## Conclusion

This progression demonstrates Mojo's **zero-cost abstractions** philosophy:
- High-level code (v1) is readable and maintainable
- Low-level optimisations (v2, v3) provide C/C++ level performance
- No runtime overhead for abstraction choices

The **50-100x speedup** from v1 to v3 comes from understanding how modern CPUs work and using Mojo's features to help the compiler generate optimal code.
