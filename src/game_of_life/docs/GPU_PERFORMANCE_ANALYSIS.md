# GPU Performance Analysis: Game of Life

## Current Results (512×512, 1000 generations)

| Implementation | Time (s) | Speedup | Notes |
|----------------|----------|---------|-------|
| Pure Python | 254.15 | 1.0× | Baseline |
| **NumPy** | **0.33** | **764× ← FASTEST** | Apple Accelerate framework |
| Mojo v1 | 1.23 | 207× | List baseline |
| Mojo v2 | 0.75 | 338× | Flat memory |
| Mojo v3 | 0.51 | 494× | Parallelised |
| Mojo v4 | 0.43 | 593× | Pointer optimisation |
| Mojo v5 | 0.47 | 546× | Edge optimisation |
| **Mojo v6 GPU** | **0.60** | **425×** | **Metal GPU (slower!)** |

## Why GPU Version is Slower

### 1. Memory Transfer Overhead (Dominant Factor)

**Current Implementation:**
```mojo
for generation in range(1000):
    # Copy CPU → GPU (262 KB)
    host_buffer = create_host_buffer()
    for i in range(cells): host_buffer[i] = data[i]  # 262,144 iterations!
    device_buffer = create_device_buffer()
    copy(device → device_buffer, host → host_buffer)
    
    # Compute on GPU (~1ms)
    launch_kernel()
    
    # Copy GPU → CPU (262 KB)
    result_buffer = create_host_buffer()
    copy(host → result_buffer, device → device_buffer)
    for i in range(cells): data[i] = result_buffer[i]  # 262,144 iterations!
```

**Cost per generation:**
- 2× memory copies (512 KB total)
- 524,288 Python-level loop iterations for data transfer
- Kernel launch overhead (~10-50μs)
- Actual GPU computation: ~1ms

**Total overhead for 1000 generations:**
- Data transfer: ~500-550ms
- Kernel launches: ~10-50ms
- **Actual computation: ~50ms**

The overhead is **10× larger** than the computation!

### 2. Small Computation per Thread

Each GPU thread performs:
- 8 memory reads (neighbors)
- 1 addition chain
- 1 bitwise OR
- 1 comparison
- 1 memory write

**Total: ~15 operations** → Too lightweight for GPU to shine.

### 3. Memory Access Pattern

Game of Life has **irregular memory access** (modulo wrapping for edges), which doesn't benefit from GPU's coalesced memory access optimisations.

## How to Make GPU Faster

### Approach 1: Keep Data on GPU (Essential Fix)

**Key idea:** Upload once, compute all generations on GPU, download once.

```mojo
struct GPUGrid[rows: Int, cols: Int]:
    var device_buffer_a: DeviceBuffer[DType.int8]
    var device_buffer_b: DeviceBuffer[DType.int8]
    var ctx: DeviceContext
    
    fn evolve_n_generations(mut self, n: Int) raises:
        """Evolve N generations entirely on GPU."""
        for gen in range(n):
            if gen % 2 == 0:
                launch_kernel(device_buffer_a, device_buffer_b)
            else:
                launch_kernel(device_buffer_b, device_buffer_a)
        # Ping-pong between two buffers
```

**Expected speedup:** 10-20× (eliminates transfer overhead)

**New estimated performance:**
- Current: 0.60s
- With this fix: **~0.03-0.06s** → Would beat all CPU versions!

### Approach 2: Batch Kernel Launches

Instead of launching kernel 1000 times, use a persistent kernel or batch generations.

### Approach 3: Use Shared Memory

Cache neighboring cells in GPU shared memory to reduce global memory accesses.

```mojo
fn evolve_kernel_optimized(...):
    # Load 18×18 tile into shared memory (16×16 data + 1-cell border)
    shared tile[18][18]
    
    # Each thread loads its cell + halo
    tile[local_y][local_x] = current[global_id]
    
    # Synchronise threads
    barrier()
    
    # Compute using shared memory (50× faster than global memory)
    var neighbors = tile[y-1][x-1] + tile[y-1][x] + ... 
```

**Expected speedup:** 2-5× on top of Approach 1

## Why NumPy is Still Fastest

NumPy + Apple Accelerate uses:

1. **AMX (Apple Matrix Coprocessor)**: Dedicated 512-bit wide matrix/vector hardware
2. **SIMD vectorisation**: Processes 16-64 cells simultaneously  
3. **Zero-copy operations**: Works directly on contiguous memory
4. **Hand-optimised kernels**: 10+ years of development
5. **CPU cache locality**: Data stays in L1/L2/L3 cache

For small grids (512×512), **CPU cache >> GPU memory bandwidth**.

## Breakeven Point

GPU becomes faster when:
- Grid size > 2048×2048 (4M+ cells)
- Generations > 10,000 (amortise transfer cost)
- Or: Keep data on GPU (Approach 1)

## Conclusion

**Current Status:**
- GPU v6 proves the concept works ✓
- But current architecture makes it slower due to overhead ✗

**To Make GPU Competitive:**
1. **Must implement:** Keep data on GPU between generations (Approach 1)
2. **Should implement:** Shared memory optimisation (Approach 3)
3. **Nice to have:** Batch kernel launches (Approach 2)

**Realistic expectations after fixes:**
- Best case: 0.02-0.03s (20× speedup over current GPU, 14× faster than NumPy)
- Realistic: 0.05-0.10s (6-12× speedup over current GPU, 3-6× faster than NumPy)

## Recommendation

For this workload (512×512 × 1000 gens):
- **Use Mojo v4** (0.43s) - best pure Mojo CPU performance
- **Use NumPy** (0.33s) - fastest overall

GPU only makes sense for:
- Much larger grids (>2048×2048)
- After implementing persistent GPU buffers
- When you need to chain multiple operations on GPU

## References

- Current implementation: `gridv6_gpu.mojo` (lines 82-187)
- GPU API docs: `examples/mojo/gpu-intro/vector_addition.mojo`
- Modular GPU documentation: https://docs.modular.com/mojo/manual/gpu/
