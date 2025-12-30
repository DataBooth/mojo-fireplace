# Tutorial 03: Reduction Operations

Advanced GPU programming with thread cooperation, shared memory, and synchronization.

## Files

- **`simple.mojo`** - Educational implementation
  - Sum reduction with atomic operations
  - Max reduction (two-stage pattern)
  - Clear explanation of shared memory
  - Detailed barrier synchronization examples
  - Runs at 1K, 100K, and 10K elements

- **`bench.mojo`** - Performance comparison
  - Sum with 256 vs 512 threads per block
  - Max reduction benchmarking
  - Mean reduction (sum + division)
  - Tests at 100K, 1M, 10M elements
  - Shows impact of block size choices

## Concepts Covered

1. **Shared Memory**
   - Fast on-chip memory shared by threads in a block
   - Allocated with `stack_allocation` and `AddressSpace.SHARED`
   - Limited size (typically 48-96KB)
   - Much faster than global memory

2. **Barrier Synchronization**
   - `barrier()` makes all threads in block wait
   - Ensures data consistency across reduction steps
   - Critical for correctness in parallel reductions
   - Must be called by all threads (no conditional barriers)

3. **Tree-Based Reduction**
   - Hierarchical combining: 256→128→64→32→16→8→4→2→1
   - O(log N) parallel steps instead of O(N) sequential
   - Each step: half the threads active, double the stride
   - Efficient use of parallelism

4. **Atomic Operations**
   - Thread-safe concurrent memory access
   - `Atomic.fetch_add()` for sum reductions
   - Enables single-stage reductions
   - Very fast on modern GPUs

5. **Two-Stage Reductions**
   - Stage 1: Reduce within each block to shared memory
   - Stage 2: Reduce block results (on CPU or second kernel)
   - Necessary for operations without atomic support (max, min)
   - Trade-off: More complex but handles any operation

## Running

```bash
# Simple version (educational)
pixi run mojo ../max/kernels/anatomy/03_reduction/simple.mojo

# Benchmark version (performance comparison)
pixi run mojo ../max/kernels/anatomy/03_reduction/bench.mojo
```

## Expected Output

### simple.mojo
- Demonstrates sum and max reductions
- Shows tree-based reduction in action
- Validates correctness at multiple scales
- Explains shared memory and barriers

### bench.mojo
- Compares 256 vs 512 threads per block
- Benchmarks sum, max, and mean operations
- Shows performance scaling with problem size
- Demonstrates optimization trade-offs

## Performance on Apple Silicon M1

### Sum Reduction
- **256 TPB**: 0.50 GElems/s at 10M elements
- **512 TPB**: 0.95 GElems/s at 10M elements (2x faster!)
- Atomic operations are efficient
- Larger blocks reduce overhead

### Max Reduction (Two-Stage)
- **256 TPB**: 12.6 GElems/s at 10M elements
- Surprisingly fast despite two stages
- Metal backend may optimize max operations heavily
- First stage (block-level) is very efficient

### Mean Reduction
- **256 TPB**: 0.50 GElems/s at 10M elements
- Similar to sum (division is cheap)
- Single atomic operation per block

### Block Size Analysis
- **256 TPB**: 
  - More blocks launched
  - Better occupancy on some workloads
  - 8 barrier synchronizations per reduction
  
- **512 TPB**:
  - Fewer blocks (half as many)
  - More work per block
  - 9 barrier synchronizations
  - **Generally faster for large arrays**

## Key Observations

1. **Block Size Matters**
   - 512 TPB is ~2x faster than 256 TPB for sum
   - Fewer blocks = less atomic contention
   - Extra barrier cost is outweighed by efficiency gains

2. **Max is Surprisingly Fast**
   - Outperforms sum by 25x at 10M elements
   - Two-stage pattern well-optimized by Metal
   - Block-level reductions are very efficient

3. **Atomic Operations**
   - Enable elegant single-stage reductions
   - Very fast on modern GPUs
   - Use whenever available (sum, count, etc.)

4. **Scaling Behaviour**
   - All operations scale well with problem size
   - Larger arrays → better GPU utilisation
   - Performance improves 10-20x from 100K to 10M elements

## Common Reduction Operations

| Operation | Atomic Support | Stages | Use Case |
|-----------|---------------|---------|----------|
| Sum | ✅ Yes | 1 | Total, accumulation |
| Max/Min | ❌ No | 2 | Finding extremes |
| Mean | ✅ Yes* | 1 | Average value |
| Variance | ❌ No | 2+ | Statistical analysis |
| Count | ✅ Yes | 1 | Histograms, filtering |

*Mean can use atomic by dividing in each block

## Optimization Strategies

1. **Choose Block Size Wisely**
   - Profile 256, 512, 1024 threads per block
   - Larger blocks often better for large arrays
   - Balance: more work per block vs more blocks

2. **Use Atomics When Available**
   - Single-stage reductions are simpler and often faster
   - Atomic operations are highly optimized on GPUs
   - For sum, count, bitwise operations

3. **Two-Stage for Everything Else**
   - Reduce in each block to shared memory
   - Collect block results and reduce on CPU
   - Or launch second kernel for large problems

4. **Optimize Shared Memory Access**
   - Avoid bank conflicts (use padding if needed)
   - Minimize shared memory usage for better occupancy
   - Tree reduction naturally avoids conflicts

5. **Consider Warp-Level Primitives**
   - `warp.sum()` for final 32 elements
   - Avoids shared memory and barriers
   - Example in production code

## Production Patterns

For production code, consider:
- Warp-level reductions for final step
- Template parameters for operation type
- Bounds checking optimizations
- Multiple elements per thread (grid-stride loops)
- Specialized kernels for common sizes

See `/mojo/stdlib/stdlib/algorithm/_gpu/reduction.mojo` for production examples.

## Next Steps

- Tutorial 04: Matrix multiplication (tiling, shared memory optimization)
- Tutorial 05: Softmax (combining reduction with other operations)
- Explore production reduction kernels in MAX source

## Common Pitfalls

1. **Forgetting barriers** → Race conditions, incorrect results
2. **Conditional barriers** → Deadlock (all threads must execute barrier)
3. **Too much shared memory** → Reduced occupancy
4. **Wrong initial value** → Incorrect results (use 0 for sum, -∞ for max)
5. **Bank conflicts** → Reduced shared memory bandwidth
