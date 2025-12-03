# Performance Analysis: Actual Benchmark Results

## Your Results (512×512, 1000 generations, macOS)

```
Implementation  Time (s)     Speedup vs Python
─────────────────────────────────────────────────
Pure Python     252.83s      1.0×    (baseline)
NumPy             0.29s      873.7×  ← FASTEST
Mojo v1           1.41s      178.7×
Mojo v2           0.74s      342.9×
Mojo v3           0.44s      577.0×
```

### Mojo Progressive Improvement
- Mojo v1 → v2: 1.9× (expected 10×) ❌
- Mojo v2 → v3: 1.7× (expected 4-8×) ❌  
- Total Mojo improvement: 3.2× (expected 40-100×) ❌

## Why NumPy is Winning

### NumPy's Secret Weapons

1. **Apple Accelerate Framework**
   - macOS NumPy is linked against Apple's Accelerate framework
   - Highly optimized for Apple Silicon (M1/M2/M3)
   - Uses AMX (Apple Matrix Extensions) and NEON SIMD
   - ~10-20× faster than generic NumPy on macOS

2. **Optimized Array Operations**
   ```python
   # This single line is heavily optimized:
   neighbors = (
       np.roll(data, 1, axis=0) + np.roll(data, -1, axis=0) +
       np.roll(data, 1, axis=1) + np.roll(data, -1, axis=1) +
       # ...
   )
   ```
   - Uses SIMD (128-bit or 256-bit vectors)
   - Minimal branching
   - Excellent cache utilization
   - Fused operations

3. **Memory Bandwidth Saturation**
   - NumPy operations are memory-bound
   - Near-optimal memory access patterns
   - Prefetching and streaming

## Why Mojo is Underperforming

### Issue 1: No SIMD Vectorization

Your current Mojo implementations process cells **one at a time**:

```mojo
for col in range(Self.cols):
    var num_neighbors = (
        self[row_above, col_left] +   # 1 byte at a time
        self[row_above, col] +
        # ...
    )
```

NumPy processes **16-32 bytes simultaneously** with SIMD.

### Issue 2: Excessive Modulo Operations

```mojo
# In the inner loop (executed cols × rows times):
var col_left = (col - 1) % Self.cols      # expensive!
var col_right = (col + 1) % Self.cols     # expensive!
```

Modulo operations are slow (20-40 cycles). With 512×512 grid, that's:
- 512 × 512 = 262,144 cells
- × 2 modulo ops per cell = 524,288 modulos per generation
- × 1000 generations = **524 million modulo operations**

### Issue 3: Indirect Indexing

```mojo
# gridv2/v3:
self[row, col]  →  (self.data + row * Self.cols + col)[]
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    Computed every access
```

While `@always_inline` helps, the multiplication `row * Self.cols` still happens for each access.

### Issue 4: Function Call Overhead in `__getitem__`

Even with `@always_inline`, there's still overhead compared to direct pointer arithmetic.

## Detailed Performance Breakdown

### Where Time is Spent (Estimated for Mojo v3)

| Operation | % of Time | Why Slow |
|-----------|-----------|----------|
| Modulo operations | 30-40% | Integer division is expensive |
| Memory access | 25-35% | Cache misses, indirection |
| Neighbour counting | 15-20% | 8 additions per cell |
| Control flow | 10-15% | Branches, conditionals |
| Other | 5-10% | Loop overhead, etc. |

### NumPy Advantage Breakdown

| Factor | Speedup Contribution |
|--------|---------------------|
| SIMD (16-32 way parallelism) | 8-16× |
| Optimized memory access | 2-3× |
| Fused operations | 1.5-2× |
| Apple Accelerate | 2-4× (macOS specific) |
| **Total** | **~48-384×** |

## Optimization Opportunities

### Quick Wins (gridv4)

1. **Pre-compute row pointers**
   ```mojo
   var curr_row = self.data + row * Self.cols
   var above_row = self.data + row_above * Self.cols
   var below_row = self.data + row_below * Self.cols
   ```
   Expected gain: 1.2-1.5×

2. **Reduce modulo operations**
   ```mojo
   # Pre-compute once per row instead of per cell
   var row_above = (row - 1 + Self.rows) % Self.rows
   var row_below = (row + 1) % Self.rows
   ```
   Expected gain: 1.3-1.8×

3. **Direct pointer arithmetic**
   ```mojo
   var num_neighbors = (
       above_row[col_left] +
       above_row[col] +
       // Direct array indexing
   )
   ```
   Expected gain: 1.1-1.2×

**Combined expected improvement: 1.7-3.2× → Mojo v4 should reach ~0.15-0.25s**

### Medium Optimization (gridv5)

4. **SIMD Vectorization**
   ```mojo
   @parameter
   fn compute_chunk[width: Int](col: Int):
       # Process `width` cells simultaneously
       var neighbors_vec = SIMD[DType.int8, width](0)
       # Load 8-16 cells at once, compute in parallel
   ```
   Expected gain: 4-8× → Could reach ~0.03-0.06s

5. **Loop Unrolling**
   ```mojo
   @parameter
   for i in range(8):
       # Unroll neighbour counting
   ```
   Expected gain: 1.2-1.5×

### Advanced Optimization (gridv6+)

6. **Temporal Blocking**
   - Process multiple generations before writing back
   - Improves cache reuse
   - Expected gain: 1.5-2×

7. **Spatial Tiling**
   - Process grid in cache-friendly tiles
   - Better L1/L2 cache utilization
   - Expected gain: 1.3-1.8×

8. **Prefetching**
   ```mojo
   __prefetch_read(next_row_ptr)
   ```
   Expected gain: 1.1-1.3×

## Realistic Performance Targets

| Version | Optimizations | Expected Time | vs NumPy |
|---------|---------------|---------------|----------|
| **Current v3** | Parallelization only | 0.44s | 1.5× slower |
| **v4** (pointer opt) | + Pointer arithmetic | 0.15-0.25s | Competitive |
| **v5** (SIMD) | + SIMD vectorization | 0.03-0.06s | **2-10× faster!** |
| **v6** (blocking) | + Temporal/spatial tiling | 0.01-0.03s | **10-30× faster!** |

## Why Your v2→v3 Gain Was Small (1.7× instead of 4-8×)

### Expected with Perfect Parallelization
- 8 cores (typical Mac) → 8× speedup
- But you got 1.7×

### Reasons for Poor Scaling

1. **Memory Bandwidth Bottleneck**
   - All cores compete for memory access
   - 512×512 × 1 byte = 256 KB per generation
   - × 2 (read + write) = 512 KB per generation
   - × 1000 generations = ~500 MB total
   - With poor cache use, this saturates memory bandwidth

2. **False Sharing** (possible)
   - Adjacent rows written by different threads
   - Cache line conflicts
   - Reduces parallel efficiency

3. **Overhead of Parallelization**
   - Thread creation/synchronization overhead
   - For 512 rows, overhead per row is significant

4. **Modulo Operations Dominate**
   - Even with 8 cores, if modulos take 40% of time
   - You only parallelize the non-modulo part (60%)
   - Theoretical max speedup: 1/((1-0.6) + 0.6/8) = 2.35×
   - Actual: 1.7× (73% efficiency)

## Immediate Action Plan

### Step 1: Test gridv4 (included in this project)

The gridv4 implementation includes:
- Pre-computed row pointers
- Reduced modulo operations
- Better memory access patterns

Expected performance: **~0.15-0.25s** (competitive with NumPy!)

### Step 2: Profile to Find Bottlenecks

```bash
# Use Mojo's profiler
mojo build --debug-info run_grid_bench_v4.mojo
instruments -t "Time Profiler" ./run_grid_bench_v4
```

Or use simple timing:
```mojo
var start = perf_counter_ns()
// Critical section
var end = perf_counter_ns()
print("Section took:", (end - start) / 1_000_000, "ms")
```

### Step 3: Implement SIMD (if needed)

If gridv4 doesn't match NumPy, the next step is proper SIMD vectorization.

## Why This is Educational

Your benchmarks reveal an important lesson: **Naive Mojo ≠ Fast Mojo**

Just like C++:
- ❌ Naive C++ can be slower than Python
- ✅ Optimized C++ is 10-100× faster than Python

The same applies to Mojo:
- ❌ Your current Mojo (v1-v3): 3× improvement over v1
- ✅ Optimized Mojo (v4-v6): 10-100× improvement potential

## NumPy's Unfair Advantages on macOS

It's worth noting that NumPy has some advantages that are hard to compete with:

1. **Decades of optimization** - Developed since 2006
2. **BLAS/LAPACK backends** - Highly tuned linear algebra
3. **Apple Accelerate** - macOS-specific optimizations
4. **Domain-specific tricks** - Array operations are very optimized

Mojo is **new** (2023) and still maturing. The fact that your v3 is already 577× faster than Python is impressive!

## Conclusion

Your benchmark results are **realistic and educational**. They show:

1. ✅ Mojo can be much faster than Python (178-577×)
2. ❌ Naive Mojo implementations don't automatically beat NumPy
3. ✅ There's significant room for optimization (v4-v6)
4. 💡 Understanding the hardware is crucial for performance

Try gridv4 and let me know the results! With the optimizations included, it should be much closer to NumPy performance (~0.15-0.25s range).

The next step would be implementing proper SIMD vectorization (v5), which could actually **beat NumPy by 2-10×**.
