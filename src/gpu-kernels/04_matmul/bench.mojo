"""
Benchmark: Comparing Matrix Multiplication Algorithms

Compares two educational matmul implementations:
1. Naive: Simple per-element computation, no optimization
2. Tiled: Uses shared memory and tiling for better cache utilization

Key performance factors:
- Memory access patterns (coalescing)
- Shared memory usage (cache locality)
- Blocking and tiling strategies

Note: For production matmul, use: from linalg.matmul import matmul
"""

from time import perf_counter_ns
from math import ceildiv
from sys import has_accelerator
from gpu import global_idx, thread_idx, block_idx
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from gpu.sync import barrier
from layout import Layout, LayoutTensor
from builtin._closure import __ownership_keepalive


# ============================================================================
# KERNEL 1: NAIVE (no optimization)
# ============================================================================

fn matmul_naive[
    M: Int, N: Int, K: Int,
    a_layout: Layout, b_layout: Layout, c_layout: Layout,
](
    a: LayoutTensor[DType.float32, a_layout, MutAnyOrigin],
    b: LayoutTensor[DType.float32, b_layout, MutAnyOrigin],
    c: LayoutTensor[DType.float32, c_layout, MutAnyOrigin],
):
    """Naive: Each thread computes one output element."""
    var row = Int(global_idx.y)
    var col = Int(global_idx.x)
    
    if row < M and col < N:
        var sum: Float32 = 0.0
        for k in range(K):
            var a_val = a[row, k]
            var b_val = b[k, col]
            sum = sum + (a_val[0] * b_val[0])
        c[row, col] = sum


# ============================================================================
# KERNEL 2: TILED (shared memory optimization)
# ============================================================================

fn matmul_tiled[
    M: Int, N: Int, K: Int,
    TILE_SIZE: Int,
    a_layout: Layout, b_layout: Layout, c_layout: Layout,
](
    a: LayoutTensor[DType.float32, a_layout, MutAnyOrigin],
    b: LayoutTensor[DType.float32, b_layout, MutAnyOrigin],
    c: LayoutTensor[DType.float32, c_layout, MutAnyOrigin],
):
    """
    Tiled: Uses shared memory tiles for better cache locality.
    
    Key improvements over naive:
    - Loads tiles into fast shared memory
    - Reuses data across threads in same block
    - Better memory bandwidth utilization
    """
    # Thread and block indices
    var thread_x = Int(thread_idx.x)
    var thread_y = Int(thread_idx.y)
    var block_x = Int(block_idx.x)
    var block_y = Int(block_idx.y)
    
    # Global output position
    var global_row = block_y * TILE_SIZE + thread_y
    var global_col = block_x * TILE_SIZE + thread_x
    
    # Tile start positions
    var tile_row_start = block_y * TILE_SIZE
    var tile_col_start = block_x * TILE_SIZE
    
    # Allocate shared memory tiles
    comptime tile_layout = Layout.row_major(TILE_SIZE, TILE_SIZE)
    var tile_a = LayoutTensor[
        DType.float32,
        tile_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    
    var tile_b = LayoutTensor[
        DType.float32,
        tile_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    
    # Accumulator for output
    var acc: Float32 = 0.0
    
    # Iterate over tiles along K dimension
    @parameter
    for k_tile in range(0, K, TILE_SIZE):
        # Load A tile
        var a_row = tile_row_start + thread_y
        var a_col = k_tile + thread_x
        if a_row < M and a_col < K:
            var a_elem = a[a_row, a_col]
            tile_a[thread_y, thread_x] = a_elem[0]
        else:
            tile_a[thread_y, thread_x] = 0.0
        
        # Load B tile
        var b_row = k_tile + thread_y
        var b_col = tile_col_start + thread_x
        if b_row < K and b_col < N:
            var b_elem = b[b_row, b_col]
            tile_b[thread_y, thread_x] = b_elem[0]
        else:
            tile_b[thread_y, thread_x] = 0.0
        
        # Wait for all threads to finish loading
        barrier()
        
        # Compute partial dot product using shared memory
        @parameter
        for k in range(TILE_SIZE):
            var a_tile_elem = tile_a[thread_y, k]
            var b_tile_elem = tile_b[k, thread_x]
            acc = acc + (a_tile_elem[0] * b_tile_elem[0])
        
        # Wait before loading next tiles
        barrier()
    
    # Write result
    if global_row < M and global_col < N:
        c[global_row, global_col] = acc


# ============================================================================
# BENCHMARK HARNESS
# ============================================================================

fn benchmark_matmul[M: Int, N: Int, K: Int](warmup_iters: Int = 3, bench_iters: Int = 10) raises:
    """Benchmark all three matmul implementations."""
    
    print("\n" + "="*70)
    print("Benchmarking Matrix Multiplication:", M, "x", K, "@", K, "x", N)
    print("Platform:", "Apple Silicon GPU (Metal)" if has_accelerator() else "CPU")
    print("="*70)
    
    # Calculate operations
    var ops = Float64(2 * M * N * K)  # Each output: K multiplies + K adds
    
    # Define layouts
    comptime a_layout = Layout.row_major(M, K)
    comptime b_layout = Layout.row_major(K, N)
    comptime c_layout = Layout.row_major(M, N)
    
    with DeviceContext() as ctx:
        # Allocate buffers
        var a_buffer = ctx.enqueue_create_buffer[DType.float32](a_layout.size())
        var b_buffer = ctx.enqueue_create_buffer[DType.float32](b_layout.size())
        var c_buffer = ctx.enqueue_create_buffer[DType.float32](c_layout.size())
        
        # Initialize input matrices
        with a_buffer.map_to_host() as host_buffer:
            var a_tensor = LayoutTensor[DType.float32, a_layout](host_buffer)
            for i in range(M):
                for j in range(K):
                    a_tensor[i, j] = Float32(i + j + 1)
        
        with b_buffer.map_to_host() as host_buffer:
            var b_tensor = LayoutTensor[DType.float32, b_layout](host_buffer)
            for i in range(K):
                for j in range(N):
                    b_tensor[i, j] = Float32(i - j + 1)
        
        var a_dev = LayoutTensor[DType.float32, a_layout](a_buffer)
        var b_dev = LayoutTensor[DType.float32, b_layout](b_buffer)
        var c_dev = LayoutTensor[DType.float32, c_layout](c_buffer)
        
        # ====================================================================
        # BENCHMARK 1: NAIVE
        # ====================================================================
        
        print("\n[1/2] Naive Algorithm")
        print("  • Per-element computation")
        print("  • No shared memory")
        print("  • Poor memory reuse")
        
        comptime block_size = 16
        var grid_x = ceildiv(N, block_size)
        var grid_y = ceildiv(M, block_size)
        
        # Warmup
        for _ in range(warmup_iters):
            ctx.enqueue_function_checked[
                matmul_naive[M, N, K, a_layout, b_layout, c_layout],
                matmul_naive[M, N, K, a_layout, b_layout, c_layout]
            ](
                a_dev, b_dev, c_dev,
                grid_dim=(grid_x, grid_y),
                block_dim=(block_size, block_size),
            )
        ctx.synchronize()
        
        # Benchmark
        var start = perf_counter_ns()
        for _ in range(bench_iters):
            ctx.enqueue_function_checked[
                matmul_naive[M, N, K, a_layout, b_layout, c_layout],
                matmul_naive[M, N, K, a_layout, b_layout, c_layout]
            ](
                a_dev, b_dev, c_dev,
                grid_dim=(grid_x, grid_y),
                block_dim=(block_size, block_size),
            )
        ctx.synchronize()
        var end = perf_counter_ns()
        
        var naive_time_ms = Float64(end - start) / 1e6 / bench_iters
        var naive_gflops = ops / (naive_time_ms / 1000.0) / 1e9
        
        print("  Time:", naive_time_ms, "ms")
        print("  Performance:", naive_gflops, "GFLOPS/s")
        
        # ====================================================================
        # BENCHMARK 2: TILED
        # ====================================================================
        
        print("\n[2/2] Tiled Algorithm")
        print("  • Shared memory tiles")
        print("  • Better cache locality")
        print("  • Memory reuse within blocks")
        
        comptime tile_size = 16
        var tiled_grid_x = ceildiv(N, tile_size)
        var tiled_grid_y = ceildiv(M, tile_size)
        
        # Warmup
        for _ in range(warmup_iters):
            ctx.enqueue_function_checked[
                matmul_tiled[M, N, K, tile_size, a_layout, b_layout, c_layout],
                matmul_tiled[M, N, K, tile_size, a_layout, b_layout, c_layout]
            ](
                a_dev, b_dev, c_dev,
                grid_dim=(tiled_grid_x, tiled_grid_y),
                block_dim=(tile_size, tile_size),
            )
        ctx.synchronize()
        
        # Benchmark
        start = perf_counter_ns()
        for _ in range(bench_iters):
            ctx.enqueue_function_checked[
                matmul_tiled[M, N, K, tile_size, a_layout, b_layout, c_layout],
                matmul_tiled[M, N, K, tile_size, a_layout, b_layout, c_layout]
            ](
                a_dev, b_dev, c_dev,
                grid_dim=(tiled_grid_x, tiled_grid_y),
                block_dim=(tile_size, tile_size),
            )
        ctx.synchronize()
        end = perf_counter_ns()
        
        var tiled_time_ms = Float64(end - start) / 1e6 / bench_iters
        var tiled_gflops = ops / (tiled_time_ms / 1000.0) / 1e9
        
        print("  Time:", tiled_time_ms, "ms")
        print("  Performance:", tiled_gflops, "GFLOPS/s")
        print("  Speedup vs naive:", tiled_gflops / naive_gflops, "x")
        
        # ====================================================================
        # SUMMARY
        # ====================================================================
        
        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)
        print("\nAlgorithm          | Time (ms) | GFLOPS/s | Speedup")
        print("-" * 70)
        print("Naive              |", naive_time_ms, "|", naive_gflops, "| 1.00x")
        print("Tiled              |", tiled_time_ms, "|", tiled_gflops, "|", tiled_gflops / naive_gflops, "x")
        
        print("\nNote: For production performance, use: from linalg.matmul import matmul")
        print("      Production kernels are typically 10-100x faster than tiled.")
        print("\n✅ Benchmark completed!")
        
        __ownership_keepalive(a_buffer, b_buffer, c_buffer)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run benchmarks at various sizes."""
    
    print("\n" + "="*70)
    print("MATRIX MULTIPLICATION BENCHMARK")
    print("="*70)
    print("\nComparing two educational algorithms:")
    print("  1. Naive: Simple, no optimization")
    print("  2. Tiled: Shared memory optimization")
    print("\n(For production code, use: from linalg.matmul import matmul)")
    
    # Benchmark at different sizes
    benchmark_matmul[64, 64, 64]()      # Small
    benchmark_matmul[128, 128, 128]()   # Medium
    benchmark_matmul[256, 256, 256]()   # Large
    benchmark_matmul[512, 512, 512]()   # Very Large
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    print("\n1. Why Tiled > Naive:")
    print("   • Shared memory is ~100x faster than global memory")
    print("   • Tiles loaded once, reused by all threads in block")
    print("   • Better cache locality and memory bandwidth")
    
    print("\n2. Further Optimizations (Production Kernels):")
    print("   • Register blocking: Multiple outputs per thread")
    print("   • Double buffering: Load next tile while computing")
    print("   • Memory coalescing: Optimize access patterns")
    print("   • Vectorization: SIMD within threads")
    print("   • Hardware-specific: Tensor cores, warp operations")
    
    print("\n3. Scaling Behaviour:")
    print("   • Small matrices: Launch overhead dominates")
    print("   • Large matrices: Algorithmic efficiency dominates")
    print("   • Tiled improves with size (better amortization)")
    
    print("\n4. Apple Silicon Metal Specifics:")
    print("   • Unified memory (CPU/GPU share memory)")
    print("   • Different from NVIDIA CUDA")
    print("   • Thread groups vs CUDA blocks")
    print("   • Shared/threadgroup memory patterns")
    
    print("="*70 + "\n")
