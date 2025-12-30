"""
Reduction Operations Benchmark

Performance comparison of different reduction strategies:
- Sum reduction with atomics
- Max reduction (two-stage)
- Mean reduction
- Different block sizes for performance tuning
"""

from sys import has_accelerator
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from gpu import thread_idx, block_idx, block_dim, barrier, warp
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from memory import alloc, stack_allocation
from os.atomic import Atomic
from builtin._closure import __ownership_keepalive


# ============================================================================
# KERNELS
# ============================================================================

fn sum_kernel_256(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """Sum reduction with 256 threads per block."""
    comptime threads_per_block = 256
    var shared = stack_allocation[threads_per_block, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var global_id = Int(block_idx.x * block_dim.x + thread_idx.x)
    
    shared[tid] = input[global_id] if global_id < size else 0.0
    barrier()
    
    var stride = threads_per_block // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        barrier()
        stride //= 2
    
    if tid == 0:
        _ = Atomic.fetch_add(output, shared[0])


fn sum_kernel_512(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """Sum reduction with 512 threads per block."""
    comptime threads_per_block = 512
    var shared = stack_allocation[threads_per_block, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var global_id = Int(block_idx.x * block_dim.x + thread_idx.x)
    
    shared[tid] = input[global_id] if global_id < size else 0.0
    barrier()
    
    var stride = threads_per_block // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        barrier()
        stride //= 2
    
    if tid == 0:
        _ = Atomic.fetch_add(output, shared[0])


fn max_kernel_256(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """Max reduction with 256 threads per block."""
    comptime threads_per_block = 256
    var shared = stack_allocation[threads_per_block, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var global_id = Int(block_idx.x * block_dim.x + thread_idx.x)
    
    shared[tid] = input[global_id] if global_id < size else -1e38
    barrier()
    
    var stride = threads_per_block // 2
    while stride > 0:
        if tid < stride:
            shared[tid] = max(shared[tid], shared[tid + stride])
        barrier()
        stride //= 2
    
    if tid == 0:
        output[block_idx.x] = shared[0]


fn mean_kernel_256(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """Mean reduction (sum + divide) with 256 threads per block."""
    comptime threads_per_block = 256
    var shared = stack_allocation[threads_per_block, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var global_id = Int(block_idx.x * block_dim.x + thread_idx.x)
    
    shared[tid] = input[global_id] if global_id < size else 0.0
    barrier()
    
    var stride = threads_per_block // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        barrier()
        stride //= 2
    
    if tid == 0:
        _ = Atomic.fetch_add(output, shared[0] / Float32(size))


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

@no_inline
fn bench_sum_256(mut bench: Bench, size: Int, ctx: DeviceContext) raises:
    """Benchmark sum reduction with 256 TPB."""
    var input_host = alloc[Float32](size)
    var output_host = alloc[Float32](1)
    
    for i in range(size):
        input_host[i] = Float32(1.0)
    
    var input_dev = ctx.enqueue_create_buffer[DType.float32](size)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](1)
    
    output_host[0] = 0.0
    ctx.enqueue_copy(output_dev, output_host)
    ctx.enqueue_copy(input_dev, input_host)
    
    comptime threads_per_block = 256
    var num_blocks = (size + threads_per_block - 1) // threads_per_block
    
    @parameter
    @always_inline
    fn bench_fn(mut b: Bencher):
        @parameter
        @always_inline
        fn launch(ctx: DeviceContext) raises:
            ctx.enqueue_function_checked[sum_kernel_256, sum_kernel_256](
                input_dev, output_dev, size,
                grid_dim=(num_blocks,), block_dim=(threads_per_block,),
            )
        b.iter_custom[launch](ctx)
    
    bench.bench_function[bench_fn](
        BenchId("sum_256tpb", input_id=String("size=", size)),
        [ThroughputMeasure(BenchMetric.elements, size)],
    )
    
    __ownership_keepalive(input_dev, output_dev)
    input_host.free()
    output_host.free()


@no_inline
fn bench_sum_512(mut bench: Bench, size: Int, ctx: DeviceContext) raises:
    """Benchmark sum reduction with 512 TPB."""
    var input_host = alloc[Float32](size)
    var output_host = alloc[Float32](1)
    
    for i in range(size):
        input_host[i] = Float32(1.0)
    
    var input_dev = ctx.enqueue_create_buffer[DType.float32](size)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](1)
    
    output_host[0] = 0.0
    ctx.enqueue_copy(output_dev, output_host)
    ctx.enqueue_copy(input_dev, input_host)
    
    comptime threads_per_block = 512
    var num_blocks = (size + threads_per_block - 1) // threads_per_block
    
    @parameter
    @always_inline
    fn bench_fn(mut b: Bencher):
        @parameter
        @always_inline
        fn launch(ctx: DeviceContext) raises:
            ctx.enqueue_function_checked[sum_kernel_512, sum_kernel_512](
                input_dev, output_dev, size,
                grid_dim=(num_blocks,), block_dim=(threads_per_block,),
            )
        b.iter_custom[launch](ctx)
    
    bench.bench_function[bench_fn](
        BenchId("sum_512tpb", input_id=String("size=", size)),
        [ThroughputMeasure(BenchMetric.elements, size)],
    )
    
    __ownership_keepalive(input_dev, output_dev)
    input_host.free()
    output_host.free()


@no_inline
fn bench_max(mut bench: Bench, size: Int, ctx: DeviceContext) raises:
    """Benchmark max reduction."""
    var input_host = alloc[Float32](size)
    var output_host = alloc[Float32](1024)
    
    for i in range(size):
        input_host[i] = Float32(i % 1000)
    
    var input_dev = ctx.enqueue_create_buffer[DType.float32](size)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](1024)
    
    ctx.enqueue_copy(input_dev, input_host)
    
    comptime threads_per_block = 256
    var num_blocks = (size + threads_per_block - 1) // threads_per_block
    
    @parameter
    @always_inline
    fn bench_fn(mut b: Bencher):
        @parameter
        @always_inline
        fn launch(ctx: DeviceContext) raises:
            ctx.enqueue_function_checked[max_kernel_256, max_kernel_256](
                input_dev, output_dev, size,
                grid_dim=(num_blocks,), block_dim=(threads_per_block,),
            )
        b.iter_custom[launch](ctx)
    
    bench.bench_function[bench_fn](
        BenchId("max_256tpb", input_id=String("size=", size)),
        [ThroughputMeasure(BenchMetric.elements, size)],
    )
    
    __ownership_keepalive(input_dev, output_dev)
    input_host.free()
    output_host.free()


@no_inline
fn bench_mean(mut bench: Bench, size: Int, ctx: DeviceContext) raises:
    """Benchmark mean reduction."""
    var input_host = alloc[Float32](size)
    var output_host = alloc[Float32](1)
    
    for i in range(size):
        input_host[i] = Float32(i % 100)
    
    var input_dev = ctx.enqueue_create_buffer[DType.float32](size)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](1)
    
    output_host[0] = 0.0
    ctx.enqueue_copy(output_dev, output_host)
    ctx.enqueue_copy(input_dev, input_host)
    
    comptime threads_per_block = 256
    var num_blocks = (size + threads_per_block - 1) // threads_per_block
    
    @parameter
    @always_inline
    fn bench_fn(mut b: Bencher):
        @parameter
        @always_inline
        fn launch(ctx: DeviceContext) raises:
            ctx.enqueue_function_checked[mean_kernel_256, mean_kernel_256](
                input_dev, output_dev, size,
                grid_dim=(num_blocks,), block_dim=(threads_per_block,),
            )
        b.iter_custom[launch](ctx)
    
    bench.bench_function[bench_fn](
        BenchId("mean_256tpb", input_id=String("size=", size)),
        [ThroughputMeasure(BenchMetric.elements, size)],
    )
    
    __ownership_keepalive(input_dev, output_dev)
    input_host.free()
    output_host.free()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run comprehensive reduction benchmarks."""
    
    print("\n" + "="*70)
    print("REDUCTION OPERATIONS - PERFORMANCE BENCHMARK")
    print("="*70)
    print("\nPlatform:", "Apple Silicon GPU (Metal)" if has_accelerator() else "CPU")
    print("\nComparing:")
    print("  1. Sum (256 TPB) - Basic reduction with atomics")
    print("  2. Sum (512 TPB) - Larger blocks, fewer barriers")
    print("  3. Max (256 TPB) - Two-stage reduction pattern")
    print("  4. Mean (256 TPB) - Sum + division")
    print("\nTPB = Threads Per Block")
    print("="*70)
    
    var benchmark = Bench()
    
    with DeviceContext() as ctx:
        var sizes = [100_000, 1_000_000, 10_000_000]
        
        for i in range(len(sizes)):
            var size = sizes[i]
            print("\n[Size:", size, "elements]")
            
            bench_sum_256(benchmark, size, ctx)
            bench_sum_512(benchmark, size, ctx)
            bench_max(benchmark, size, ctx)
            bench_mean(benchmark, size, ctx)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    benchmark.dump_report()
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("\nKey Observations:")
    print("  • Sum reductions: Very fast, atomic operations efficient")
    print("  • Block size impact: 512 TPB may be faster (fewer blocks)")
    print("  • Max vs Sum: Max requires two stages, may be slower")
    print("  • Mean: Similar to sum (extra division is cheap)")
    
    print("\nPerformance Factors:")
    print("  1. Shared Memory Usage:")
    print("     - 256 TPB uses 1KB shared memory")
    print("     - 512 TPB uses 2KB shared memory")
    print("     - More shared memory = fewer blocks can run concurrently")
    
    print("\n  2. Barrier Overhead:")
    print("     - Each reduction step has a barrier()")
    print("     - 256 TPB: 8 barriers (log2(256))")
    print("     - 512 TPB: 9 barriers (log2(512))")
    print("     - Fewer blocks may compensate for extra barrier")
    
    print("\n  3. Atomic Operations:")
    print("     - Sum/Mean use atomic_add (very fast on modern GPUs)")
    print("     - Max requires second reduction stage (slower)")
    
    print("\n  4. Occupancy vs Thread Count:")
    print("     - 256 TPB: More blocks, better occupancy")
    print("     - 512 TPB: Fewer blocks, but more work per block")
    print("     - Optimal choice depends on GPU and problem size")
    
    print("\nOptimization Strategy:")
    print("  • Profile different block sizes for your workload")
    print("  • Use atomics when available (sum, add)")
    print("  • Two-stage for non-atomic ops (max, min)")
    print("  • Consider warp-level primitives for final reduction")
    print("  • Larger arrays benefit from larger blocks")
    print("="*70 + "\n")
