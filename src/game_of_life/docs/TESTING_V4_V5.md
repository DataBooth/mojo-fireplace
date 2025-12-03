# Testing gridv4 and gridv5 Optimizations

This guide helps you test the newly created optimized versions (v4 and v5) against your current implementations.

## What's New

### gridv4 - Optimized Pointer Arithmetic
**Key improvements over v3:**
- Pre-computed row pointers (eliminate repeated `row * cols` multiplication)
- Reduced modulo operations (moved outside inner loop)
- Direct pointer arithmetic instead of `__getitem__` calls

**Expected improvement:** 2-3× faster than v3 → **~0.15-0.25s** (competitive with NumPy)

### gridv5 - SIMD Preparation
**Key improvements over v4:**
- Prepared for SIMD vectorization
- Optimized edge handling (left/right columns)
- Better memory access patterns
- Reduced branch mispredictions

**Expected improvement:** 1.5-2× faster than v4 → **~0.10-0.15s** (potentially beating NumPy)

## Quick Test

### Test Individual Versions

```bash
# Test v4 alone
mojo run run_grid_bench_v4.mojo

# Test v5 alone
mojo run run_grid_bench_v5.mojo
```

### Test All Versions Together

```bash
python benchmark_grid_all_versions.py
```

This will now run all 5 Mojo versions (v1-v5) plus Python implementations.

## Expected Results

Based on your system (512×512, 1000 generations, macOS):

### Current Results
```
Pure Python:  252.83s  (1.0×)
NumPy:          0.29s  (873.7×) ← Current champion
Mojo v1:        1.41s  (178.7×)
Mojo v2:        0.74s  (342.9×)
Mojo v3:        0.44s  (577.0×)
```

### Predicted with v4 and v5
```
Pure Python:  252.83s  (1.0×)
NumPy:          0.29s  (873.7×) ← Current champion
Mojo v1:        1.41s  (178.7×)
Mojo v2:        0.74s  (342.9×)
Mojo v3:        0.44s  (577.0×)
Mojo v4:     ~0.15s  (1,685×)  ← Should match/beat NumPy!
Mojo v5:     ~0.10s  (2,528×)  ← Should beat NumPy by 2-3×!
```

## What Each Optimization Does

### v3 → v4: Pointer Arithmetic

**Problem in v3:**
```mojo
for col in range(Self.cols):
    var col_left = (col - 1) % Self.cols      # 524 million modulos!
    var col_right = (col + 1) % Self.cols     # 524 million modulos!
    var num_neighbors = (
        self[row_above, col_left] +           # Multiply row*cols each time
        // ...
    )
```

**Solution in v4:**
```mojo
# Pre-compute once per row (not per cell)
var row_above = (row - 1 + Self.rows) % Self.rows
var row_below = (row + 1) % Self.rows

# Pre-compute row pointers (not per cell)
var curr_row = self.data + row * Self.cols
var above_row = self.data + row_above * Self.cols
var below_row = self.data + row_below * Self.cols

for col in range(Self.cols):
    var col_left = (col - 1 + Self.cols) % Self.cols
    var col_right = (col + 1) % Self.cols
    
    # Direct pointer indexing (no multiplication!)
    var num_neighbors = (
        above_row[col_left] + above_row[col] + above_row[col_right] +
        // ...
    )
```

**Savings:**
- 512 × 512 × 1000 = 262 million row multiplications eliminated
- Much better than 524 million modulos (still have column modulos)

### v4 → v5: Edge Optimization

**Problem in v4:**
Still computing modulo for every column:
```mojo
for col in range(Self.cols):
    var col_left = (col - 1 + Self.cols) % Self.cols
    var col_right = (col + 1) % Self.cols
```

**Solution in v5:**
Handle edges separately, no modulo for middle columns:
```mojo
# Left edge (col 0) - special case
var col_left = Self.cols - 1
var col_right = 1
// process col 0

# Middle columns - NO MODULO!
for col in range(1, Self.cols - 1):
    // col_left = col - 1, col_right = col + 1 (no modulo needed!)
    var neighbors = (
        above_row[col - 1] + above_row[col] + above_row[col + 1] +
        // ...
    )

# Right edge (col = cols-1) - special case
var col_left = Self.cols - 2
var col_right = 0
// process last col
```

**Savings:**
- 262 million modulo operations eliminated (only 2 per row for edges)
- Better branch prediction (straight-line code in inner loop)

## Debugging Tips

### If v4 is not faster than v3:

1. **Check compilation flags:**
   ```bash
   # Make sure you're not in debug mode
   mojo run run_grid_bench_v4.mojo  # Should use release mode by default
   ```

2. **Verify grid size:**
   - Smaller grids (< 128×128) might not show improvement
   - Try 512×512 or 1024×1024

3. **Check system load:**
   ```bash
   # Make sure no other heavy processes running
   top
   ```

### If v5 is not faster than v4:

This could happen if:
- Compiler is already optimizing away modulos in v4
- Edge case handling adds overhead
- Branch predictor is very good on your CPU

## Profiling Commands

### Simple Timing
Add to the evolve() function:
```mojo
var start = perf_counter_ns()
// critical section
var end = perf_counter_ns()
print("Modulo time:", (end - start) / 1_000_000, "ms")
```

### macOS Instruments
```bash
mojo build run_grid_bench_v4.mojo -o bench_v4
instruments -t "Time Profiler" ./bench_v4
```

## Next Steps After Testing

### If v4/v5 beat NumPy:
🎉 Success! You've beaten a decades-old optimized library!

Consider:
1. Writing a blog post about the optimization journey
2. Contributing the patterns back to Mojo examples
3. Trying even more optimizations (SIMD, temporal blocking)

### If v4/v5 still slower than NumPy:
📊 Profile to understand why:

1. Use Instruments (macOS) or perf (Linux)
2. Check for:
   - Cache misses
   - Branch mispredictions
   - Memory bandwidth saturation
3. Try SIMD vectorization (true SIMD, not just better scalar code)

## Summary

The v4 and v5 implementations focus on **reducing computational overhead**:

| Version | Key Optimization | Operations Saved |
|---------|------------------|------------------|
| v3 | Parallelization | Uses multiple cores |
| v4 | Pointer arithmetic | ~262M multiplications |
| v5 | Edge handling | ~262M modulo operations |

Combined, these should provide **3-6× improvement over v3**, bringing Mojo performance to **match or exceed NumPy**.

Run the benchmarks and let's see the real results! 🚀
