"""
Elementwise Operations Benchmark

Performance comparison of different activation functions and optimization techniques:
- ReLU vs GELU (computational complexity)
- Fused operations (memory efficiency)
- SIMD vectorization (throughput optimization)
"""

from sys import has_accelerator
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from gpu import global_idx
from gpu.host import DeviceContext
from memory import alloc
from builtin._closure import __ownership_keepalive
from math import tanh


# ============================================================================
# KERNELS
# ============================================================================

fn relu_kernel(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """ReLU: max(0, x)"""
    var idx = global_idx.x
    if idx >= UInt(size):
        return
    var value = input[idx]
    output[idx] = value if value > 0 else Float32(0)


fn gelu_kernel(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """GELU: Gaussian Error Linear Unit."""
    var idx = global_idx.x
    if idx >= UInt(size):
        return
    var x = input[idx]
    var x_cubed = x * x * x
    var inner = 0.7978845608 * (x + 0.044715 * x_cubed)
    var tanh_approx = tanh(inner)
    output[idx] = 0.5 * x * (1.0 + tanh_approx)


fn fused_relu_scale_kernel(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    scale: Float32,
    size: Int,
):
    """Fused: ReLU + Scale."""
    var idx = global_idx.x
    if idx >= UInt(size):
        return
    var value = input[idx]
    var relu_value = value if value > 0 else Float32(0)
    output[idx] = relu_value * scale


fn relu_kernel_simd[simd_width: Int](
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """SIMD vectorized ReLU."""
    var idx = Int(global_idx.x) * simd_width
    if idx >= size:
        return
    var remaining = size - idx
    var process_count = min(simd_width, remaining)
    for i in range(process_count):
        var value = input[idx + i]
        output[idx + i] = value if value > 0 else Float32(0)


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

@no_inline
fn bench_relu(mut bench: Bench, size: Int, ctx: DeviceContext) raises:
    """Benchmark standard ReLU."""
    var input_host = alloc[Float32](size)
    var output_host = alloc[Float32](size)
    
    for i in range(size):
        input_host[i] = Float32(i % 100 - 50) / 10.0
    
    var input_dev = ctx.enqueue_create_buffer[DType.float32](size)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_copy(input_dev, input_host)
    
    var threads_per_block = 256
    var num_blocks = (size + threads_per_block - 1) // threads_per_block
    
    @parameter
    @always_inline
    fn bench_fn(mut b: Bencher):
        @parameter
        @always_inline
        fn launch(ctx: DeviceContext) raises:
            ctx.enqueue_function_checked[relu_kernel, relu_kernel](
                input_dev, output_dev, size,
                grid_dim=(num_blocks,), block_dim=(threads_per_block,),
            )
        b.iter_custom[launch](ctx)
    
    bench.bench_function[bench_fn](
        BenchId("relu", input_id=String("size=", size)),
        [ThroughputMeasure(BenchMetric.flops, size)],
    )
    
    __ownership_keepalive(input_dev, output_dev)
    input_host.free()
    output_host.free()


@no_inline
fn bench_gelu(mut bench: Bench, size: Int, ctx: DeviceContext) raises:
    """Benchmark GELU."""
    var input_host = alloc[Float32](size)
    var output_host = alloc[Float32](size)
    
    for i in range(size):
        input_host[i] = Float32(i % 100 - 50) / 10.0
    
    var input_dev = ctx.enqueue_create_buffer[DType.float32](size)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_copy(input_dev, input_host)
    
    var threads_per_block = 256
    var num_blocks = (size + threads_per_block - 1) // threads_per_block
    
    @parameter
    @always_inline
    fn bench_fn(mut b: Bencher):
        @parameter
        @always_inline
        fn launch(ctx: DeviceContext) raises:
            ctx.enqueue_function_checked[gelu_kernel, gelu_kernel](
                input_dev, output_dev, size,
                grid_dim=(num_blocks,), block_dim=(threads_per_block,),
            )
        b.iter_custom[launch](ctx)
    
    bench.bench_function[bench_fn](
        BenchId("gelu", input_id=String("size=", size)),
        [ThroughputMeasure(BenchMetric.flops, size * 10)],  # ~10 ops per element
    )
    
    __ownership_keepalive(input_dev, output_dev)
    input_host.free()
    output_host.free()


@no_inline
fn bench_fused(mut bench: Bench, size: Int, ctx: DeviceContext) raises:
    """Benchmark fused ReLU+Scale."""
    var input_host = alloc[Float32](size)
    var output_host = alloc[Float32](size)
    
    for i in range(size):
        input_host[i] = Float32(i % 100 - 50) / 10.0
    
    var input_dev = ctx.enqueue_create_buffer[DType.float32](size)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_copy(input_dev, input_host)
    
    var threads_per_block = 256
    var num_blocks = (size + threads_per_block - 1) // threads_per_block
    var scale = Float32(2.5)
    
    @parameter
    @always_inline
    fn bench_fn(mut b: Bencher):
        @parameter
        @always_inline
        fn launch(ctx: DeviceContext) raises:
            ctx.enqueue_function_checked[fused_relu_scale_kernel, fused_relu_scale_kernel](
                input_dev, output_dev, scale, size,
                grid_dim=(num_blocks,), block_dim=(threads_per_block,),
            )
        b.iter_custom[launch](ctx)
    
    bench.bench_function[bench_fn](
        BenchId("relu_fused", input_id=String("size=", size)),
        [ThroughputMeasure(BenchMetric.flops, size * 2)],  # 2 ops: max and mul
    )
    
    __ownership_keepalive(input_dev, output_dev)
    input_host.free()
    output_host.free()


@no_inline
fn bench_simd_relu(mut bench: Bench, size: Int, ctx: DeviceContext) raises:
    """Benchmark SIMD vectorized ReLU."""
    var input_host = alloc[Float32](size)
    var output_host = alloc[Float32](size)
    
    for i in range(size):
        input_host[i] = Float32(i % 100 - 50) / 10.0
    
    var input_dev = ctx.enqueue_create_buffer[DType.float32](size)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_copy(input_dev, input_host)
    
    alias simd_width = 4
    var threads_per_block = 256
    var simd_threads = (size + simd_width - 1) // simd_width
    var num_blocks = (simd_threads + threads_per_block - 1) // threads_per_block
    
    @parameter
    @always_inline
    fn bench_fn(mut b: Bencher):
        @parameter
        @always_inline
        fn launch(ctx: DeviceContext) raises:
            ctx.enqueue_function_checked[relu_kernel_simd[simd_width], relu_kernel_simd[simd_width]](
                input_dev, output_dev, size,
                grid_dim=(num_blocks,), block_dim=(threads_per_block,),
            )
        b.iter_custom[launch](ctx)
    
    bench.bench_function[bench_fn](
        BenchId("relu_simd4", input_id=String("size=", size)),
        [ThroughputMeasure(BenchMetric.flops, size)],
    )
    
    __ownership_keepalive(input_dev, output_dev)
    input_host.free()
    output_host.free()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run comprehensive benchmarks."""
    
    print("\n" + "="*70)
    print("ELEMENTWISE OPERATIONS - PERFORMANCE BENCHMARK")
    print("="*70)
    print("\nPlatform:", "Apple Silicon GPU (Metal)" if has_accelerator() else "CPU")
    print("\nComparing:")
    print("  1. ReLU - Simple, fast activation")
    print("  2. GELU - Complex, transformer activation (~10x more compute)")
    print("  3. Fused ReLU+Scale - Memory-efficient fusion")
    print("  4. SIMD ReLU - Vectorized for throughput")
    print("="*70)
    
    var benchmark = Bench()
    
    with DeviceContext() as ctx:
        # Benchmark at multiple sizes
        var sizes = [100_000, 1_000_000, 10_000_000]
        
        for i in range(len(sizes)):
            var size = sizes[i]
            print("\n[Size:", size, "elements]")
            
            bench_relu(benchmark, size, ctx)
            bench_gelu(benchmark, size, ctx)
            bench_fused(benchmark, size, ctx)
            bench_simd_relu(benchmark, size, ctx)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    benchmark.dump_report()
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("\nKey Observations:")
    print("  • ReLU vs GELU: GELU is slower due to complex math (tanh, powers)")
    print("  • Fused ops: Similar to ReLU (fusion mainly saves memory, not compute)")
    print("  • SIMD: Should show improved throughput (GFLOPS/s) for large arrays")
    print("  • Scale effects: Larger arrays → better GPU utilization")
    print("\nMemory vs Compute Bound:")
    print("  • ReLU/Fused: Memory-bound (simple ops, lots of data movement)")
    print("  • GELU: More compute-bound (complex math per element)")
    print("  • SIMD: Better memory bandwidth utilization")
    print("\nOptimization Strategies:")
    print("  1. Use kernel fusion to reduce memory round-trips")
    print("  2. Apply SIMD when memory-bound")
    print("  3. Choose activation functions based on accuracy/speed tradeoff")
    print("  4. Profile to identify bottlenecks (memory vs compute)")
    print("="*70 + "\n")
