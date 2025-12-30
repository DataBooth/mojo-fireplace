# Tutorial 02: Elementwise Operations

Advanced patterns for GPU programming including activation functions, kernel fusion, and SIMD vectorisation.

## Files

- **`simple.mojo`** - Educational implementation
  - ReLU and GELU activation functions
  - Kernel fusion example (ReLU + Scale)
  - SIMD vectorised kernel
  - Validation for all variants
  - Runs at 10K and 1M elements

- **`bench.mojo`** - Performance comparison
  - Benchmarks 4 implementations: ReLU, GELU, Fused, SIMD
  - Tests at multiple scales (100K, 1M, 10M)
  - Shows memory vs compute tradeoffs
  - Demonstrates optimization strategies

## Concepts Covered

1. **Activation Functions**
   - **ReLU**: `max(0, x)` - Simple, fast, memory-bound
   - **GELU**: Gaussian Error Linear Unit - Complex, compute-bound
   - Performance/accuracy tradeoffs

2. **Kernel Fusion**
   - Combines multiple operations in one kernel
   - Example: ReLU + Scale instead of separate passes
   - Reduces memory bandwidth requirements
   - Maintains same computational complexity

3. **SIMD Vectorisation**
   - Process multiple elements per thread (4x in our examples)
   - Reduces thread count by factor of SIMD width
   - Better memory bandwidth utilisation
   - Compile-time parameter for flexibility

4. **Memory vs Compute Bound**
   - **Memory-bound**: Limited by data transfer (ReLU, Fused)
   - **Compute-bound**: Limited by math operations (GELU)
   - Different optimization strategies for each

## Running

```bash
# Simple version (educational)
pixi run mojo ../max/kernels/anatomy/02_elementwise/simple.mojo

# Benchmark version (performance comparison)
pixi run mojo ../max/kernels/anatomy/02_elementwise/bench.mojo
```

## Expected Output

### simple.mojo
- Demonstrates 4 different kernels
- Validates correctness for each
- Shows SIMD thread reduction (40 blocks → 10 blocks)
- Runs at 2 different scales

### bench.mojo
- Benchmarks each kernel at 3 sizes
- Reports timing and GFLOPS/s
- Clearly shows ReLU vs GELU performance difference
- Demonstrates SIMD benefits at scale

## Performance on Apple Silicon M1

### ReLU (simple, memory-bound)
- 100K: 0.3ms, 0.33 GFLOPS/s
- 1M: 0.4ms, 2.5 GFLOPS/s  
- 10M: 0.9ms, 11 GFLOPS/s

### GELU (complex, compute-bound)
- 100K: 0.3ms, 3.1 GFLOPS/s (10x more compute than ReLU!)
- 1M: 0.4ms, 27 GFLOPS/s
- 10M: 1.1ms, 90 GFLOPS/s (much higher throughput)

### Fused ReLU+Scale
- Similar timing to ReLU
- 2x GFLOPS (counts both max and multiply)
- Memory savings not visible in single-kernel benchmark

### SIMD ReLU (4-wide vectorisation)
- Similar or slightly better than scalar ReLU
- Fewer threads launched (4x reduction)
- Benefits more apparent at very large scales

## Key Observations

1. **GELU is ~8-10x higher GFLOPS** than ReLU despite similar timing
   - GELU does ~10x more compute per element
   - GELU is compute-bound, ReLU is memory-bound
   
2. **Fusion benefits are primarily memory savings**
   - Similar execution time to individual operations
   - Saves memory bandwidth by reducing passes
   
3. **SIMD reduces thread pressure**
   - Fewer blocks/threads needed
   - Better for GPU occupancy
   - Most beneficial for memory-bound operations

4. **Scale matters**
   - All kernels show improved GFLOPS at larger sizes
   - GPU utilisation increases with problem size
   - Sweet spot: 1M-100M elements

## Optimization Strategies

1. **Memory-bound kernels** (ReLU, simple ops)
   - Apply SIMD vectorisation
   - Use kernel fusion to reduce passes
   - Increase problem size when possible

2. **Compute-bound kernels** (GELU, complex ops)
   - Focus on mathematical optimizations
   - Consider approximations for speed
   - Profile math operations

3. **General**
   - Fuse operations to reduce memory traffic
   - Choose activation functions based on needs
   - Benchmark to identify bottlenecks

## Next Steps

- Explore reduction operations (Tutorial 03)
- Learn about shared memory and synchronisation
- Apply these patterns to your own kernels
