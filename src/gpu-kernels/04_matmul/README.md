# Tutorial 04: Matrix Multiplication

Matrix multiplication (matmul) is the cornerstone operation of deep learning and scientific computing. This tutorial explores three different implementations to understand the performance hierarchy from naive to production-optimised algorithms.

## What You'll Learn

- 2D thread indexing and grid configuration
- Shared memory and tiling strategies
- Memory bandwidth optimisation
- When to use production kernels vs custom implementations
- Performance characteristics on Apple Silicon Metal

## Files

- **`simple.mojo`**: Educational wrapper around a naive matmul kernel
- **`bench.mojo`**: Benchmarks comparing naive, tiled, and production algorithms

## Running the Code

```bash
# Simple example (demonstrates usage patterns)
pixi run mojo simple.mojo

# Benchmark (compares three algorithms at multiple sizes)
pixi run mojo bench.mojo
```

## Matrix Multiplication Basics

### The Operation

```
C[i,j] = sum(A[i,k] * B[k,j]) for all k
```

Where:
- A is M × K (M rows, K columns)
- B is K × N (K rows, N columns)  
- C is M × N (M rows, N columns)

### Computational Complexity

- **Operations**: O(M × N × K) = 2MNK FLOPs (multiply + add per element)
- **Example**: 512×512 @ 512×512 = 268M FLOPs

### Why Parallelisation Matters

Each output element C[i,j] can be computed independently, making matmul highly parallelisable. However, naive parallelisation has poor performance due to memory access patterns.

## Three Algorithm Implementations

### 1. Naive Algorithm

**Characteristics:**
- One thread per output element
- Each thread performs K multiply-adds
- No shared memory usage
- Poor memory reuse

**Performance:** ~1-10 GFLOPS/s (baseline)

**Good for:** Understanding GPU programming basics

**Code pattern:**
```mojo
fn matmul_naive[M, N, K, ...](a, b, c):
    var row = Int(global_idx.y)
    var col = Int(global_idx.x)
    
    if row < M and col < N:
        var sum = Float32(0.0)
        for k in range(K):
            sum += a[row, k] * b[k, col]
        c[row, col] = sum
```

### 2. Tiled Algorithm

**Characteristics:**
- Loads tiles into fast shared memory
- Reuses data across threads in the same block
- Better memory bandwidth utilisation
- Requires barrier synchronisation

**Performance:** ~10-50 GFLOPS/s (3-10x faster than naive)

**Good for:** Learning memory hierarchy optimisation

**Key concepts:**
```mojo
# Allocate shared memory tiles
var tile_a = LayoutTensor[..., address_space=AddressSpace.SHARED].stack_allocation()
var tile_b = LayoutTensor[..., address_space=AddressSpace.SHARED].stack_allocation()

# Load tiles cooperatively
for k_tile in range(0, K, TILE_SIZE):
    # Load A and B tiles into shared memory
    tile_a[thread_y, thread_x] = a[...]
    tile_b[thread_y, thread_x] = b[...]
    
    # Synchronise all threads
    barrier()
    
    # Compute using shared memory (fast!)
    for k in range(TILE_SIZE):
        acc += tile_a[thread_y, k] * tile_b[k, thread_x]
    
    # Synchronise before loading next tiles
    barrier()
```

### 3. Production Algorithm (Modular)

**Characteristics:**
- Hardware-specific optimisations
- Tensor cores (where available)
- Advanced register blocking
- Optimised instruction scheduling

**Performance:** ~50-500+ GFLOPS/s (50-500x faster than naive)

**Good for:** Real-world applications

**Usage:**
```mojo
from max.kernels.linalg.matmul import matmul

# Simply call the optimised implementation
matmul[target="gpu"](c, a, b, ctx)
```

## Memory Hierarchy

Understanding GPU memory hierarchy is crucial for matmul performance:

```
Register (per-thread):     ~1 cycle latency,    KB size
Shared/Local (per-block):  ~10 cycle latency,   KB-MB size  
Global (device):           ~400 cycle latency,  GB size
```

**Key insight:** Accessing global memory is ~400× slower than registers. Tiled algorithms exploit shared memory to reduce global memory accesses.

## Performance Factors

### Memory Bandwidth

- **Naive**: Each element of A and B loaded K times = O(MNK) memory ops
- **Tiled**: Each element loaded once per tile = O(MN + NK + MK) memory ops
- **Improvement**: K/TILE_SIZE reduction in memory traffic

### Computational Intensity

Ratio of computation to memory access:

- **Naive**: 2 FLOPs / 2 loads = 1.0 (memory-bound)
- **Tiled**: 2·TILE_SIZE FLOPs / 2 loads = TILE_SIZE (more compute-bound)
- **Production**: Maximises through register blocking and vectorisation

### Apple Silicon Specifics

On Metal (Apple Silicon):
- Unified memory architecture (CPU/GPU share memory)
- No explicit "tensor cores" like NVIDIA
- Different optimisation strategies than CUDA
- Still benefits from tiling and memory reuse

## When to Use Each Approach

### Use Naive When:
- Learning GPU programming fundamentals
- Prototyping or validating correctness
- Matrix sizes are tiny (<10×10)

### Use Tiled When:
- Learning memory optimisation techniques
- Need to understand performance bottlenecks
- Implementing custom fusion with matmul

### Use Production When:
- Building real applications
- Performance matters
- Matrix sizes are reasonable (>64×64)

**Recommendation:** For production code, always use Modular's optimised matmul kernel unless you have very specific fusion requirements.

## Benchmark Results (Typical)

Matrix Size | Naive | Tiled | Production
---|---|---|---
64×64 | 2 GFLOPS/s | 8 GFLOPS/s | 40 GFLOPS/s
128×128 | 3 GFLOPS/s | 15 GFLOPS/s | 100 GFLOPS/s
256×256 | 4 GFLOPS/s | 25 GFLOPS/s | 200 GFLOPS/s
512×512 | 5 GFLOPS/s | 35 GFLOPS/s | 350 GFLOPS/s

*Results vary by hardware. These are illustrative values for Apple Silicon M1/M2.*

## Key Takeaways

1. **Memory bandwidth is the bottleneck** for naive matmul
2. **Shared memory and tiling** dramatically improve performance by reusing data
3. **Production kernels** add register blocking, vectorisation, and hardware-specific optimisations
4. **Algorithmic improvements compound** as matrix size increases
5. **Don't reinvent the wheel** - use production kernels unless you have specific needs

## Further Optimisations (Production Kernels)

Advanced techniques not covered in this tutorial:

- **Register blocking**: Each thread computes multiple output elements
- **Double buffering**: Load next tile while computing current tile
- **Memory coalescing**: Ensure adjacent threads access adjacent memory
- **Warp-level operations**: SIMD within thread blocks
- **Tensor cores**: Specialised matrix hardware (NVIDIA)
- **Non-standard layouts**: Transposed, blocked, or strided matrices

## Where to Find More

- **Production kernels**: `/max/kernels/src/linalg/matmul/`
- **Tiled example**: `/examples/mojo/gpu-block-and-warp/tiled_matmul.mojo`
- **Custom ops**: `/examples/custom_ops/kernels/matrix_multiplication.mojo`
- **Documentation**: https://docs.modular.com/

## Related Tutorials

- Tutorial 02: Elementwise operations (kernel fusion)
- Tutorial 03: Reduction operations (shared memory, barriers)
- Tutorial 05: Softmax (combining techniques)

---

**Next:** Explore how production matmul kernels are used in operations like attention mechanisms and neural network layers!
