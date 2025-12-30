# Tutorial 01: Vector Addition

The foundational GPU kernel tutorial covering basic parallel programming concepts.

## Files

- **`simple.mojo`** - Clean, focused implementation
  - Core kernel pattern
  - Memory management
  - Device coordination
  - Validation
  - Runs at multiple scales (10K, 1M, 10M elements)

- **`bench.mojo`** - Performance benchmarking
  - Measures execution time
  - Reports GFLOPS/s throughput
  - Shows scaling behaviour across problem sizes
  - Explains benchmark metrics (iters, met)

## Concepts Covered

1. **GPU Kernel Structure**
   - Device functions that run in parallel
   - Thread indexing with `global_idx.x`
   - Bounds checking for safety

2. **Memory Management**
   - Host (CPU) memory allocation
   - Device (GPU) memory buffers
   - Data transfers: Host ↔ Device

3. **Kernel Launch Configuration**
   - Grid dimensions (number of blocks)
   - Block dimensions (threads per block)
   - `enqueue_function_checked` API

4. **Validation**
   - Correctness verification
   - Error reporting

## Running

```bash
# Simple version (educational)
pixi run mojo ../max/kernels/anatomy/01_vector_add/simple.mojo

# Benchmark version (performance)
pixi run mojo ../max/kernels/anatomy/01_vector_add/bench.mojo
```

## Expected Output

### simple.mojo
- Runs vector addition at 3 different sizes
- Shows platform detection (Apple Silicon GPU)
- Validates all results

### bench.mojo
- Benchmarks at 4 sizes (10K, 100K, 1M, 10M)
- Reports timing and throughput metrics
- Shows performance scaling

## Performance on Apple Silicon M1

- **10K elements**: ~0.3ms, ~0.03 GFLOPS/s
- **100K elements**: ~0.3ms, ~0.3 GFLOPS/s
- **1M elements**: ~0.4ms, ~2.5 GFLOPS/s
- **10M elements**: ~1.1ms, ~9 GFLOPS/s

Throughput increases with array size showing improved GPU utilisation.

## Key Takeaways

- GPU programming follows a predictable pattern
- Each thread processes one element independently
- Memory transfers can be overhead for small arrays
- Performance scales well with problem size
- Apple Silicon GPU works via Metal backend
