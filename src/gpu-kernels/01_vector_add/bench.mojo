"""
Vector Addition Benchmark

This file focuses on performance measurement using Mojo's benchmark infrastructure.
The kernel is the same as vector_add_simple.mojo, but with detailed timing analysis.
"""

from sys import has_accelerator
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from gpu import global_idx
from gpu.host import DeviceContext
from memory import alloc
from builtin._closure import __ownership_keepalive


# ============================================================================
# KERNEL (same as simple version)
# ============================================================================

fn vector_add_kernel(
    in0: UnsafePointer[Float32, MutAnyOrigin],
    in1: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """GPU kernel that adds two vectors element-wise."""
    var idx = global_idx.x
    if idx >= UInt(size):
        return
    output[idx] = in0[idx] + in1[idx]


# ============================================================================
# BENCHMARK HARNESS
# ============================================================================

@no_inline
fn benchmark_vector_add(
    mut bench: Bench,
    size: Int,
    ctx: DeviceContext,
) raises:
    """Benchmark vector addition kernel at a specific size."""
    
    # Allocate and initialize memory
    var a_host = alloc[Float32](size)
    var b_host = alloc[Float32](size)
    var c_host = alloc[Float32](size)
    
    for i in range(size):
        a_host[i] = Float32(i % 1000)
        b_host[i] = Float32(2.0)
    
    # Device memory
    var a_device = ctx.enqueue_create_buffer[DType.float32](size)
    var b_device = ctx.enqueue_create_buffer[DType.float32](size)
    var c_device = ctx.enqueue_create_buffer[DType.float32](size)
    
    # Copy to device
    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    
    # Launch configuration
    var threads_per_block = 256
    var num_blocks = (size + threads_per_block - 1) // threads_per_block
    
    # Define the benchmarked operation
    @parameter
    @always_inline
    fn bench_kernel(mut b: Bencher):
        @parameter
        @always_inline
        fn launch(ctx: DeviceContext) raises:
            ctx.enqueue_function_checked[vector_add_kernel, vector_add_kernel](
                a_device,
                b_device,
                c_device,
                size,
                grid_dim=(num_blocks,),
                block_dim=(threads_per_block,),
            )
        
        b.iter_custom[launch](ctx)
    
    # Run benchmark
    bench.bench_function[bench_kernel](
        BenchId("vector_add", input_id=String("size=", size)),
        [ThroughputMeasure(BenchMetric.flops, size)],
    )
    
    # Validate (quick check)
    ctx.synchronize()
    ctx.enqueue_copy(c_host, c_device)
    
    # Check: a[42] = 42 % 1000 = 42, b[42] = 2, so c[42] should be 44
    var expected = Float32((42 % 1000) + 2)
    var actual = c_host[42]
    if actual != expected:
        print("⚠️  Validation failed at index 42: got", actual, "expected", expected)
    
    # Cleanup
    __ownership_keepalive(a_device, b_device, c_device)
    a_host.free()
    b_host.free()
    c_host.free()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run benchmarks at multiple problem sizes."""
    
    print("\n" + "="*70)
    print("VECTOR ADDITION PERFORMANCE BENCHMARK")
    print("="*70)
    print("\nPlatform:", "Apple Silicon GPU (Metal)" if has_accelerator() else "CPU")
    print("\nBenchmark Metrics:")
    print("  • met (ms): Mean execution time per iteration")
    print("  • iters: Number of times kernel ran for accurate measurement")
    print("  • GFLOPS/s: Billions of floating-point operations per second")
    print("="*70)
    
    var benchmark = Bench()
    
    with DeviceContext() as ctx:
        # Benchmark at multiple scales
        print("\n[1/4] Benchmarking 10K elements...")
        benchmark_vector_add(benchmark, 10_000, ctx)
        
        print("[2/4] Benchmarking 100K elements...")
        benchmark_vector_add(benchmark, 100_000, ctx)
        
        print("[3/4] Benchmarking 1M elements...")
        benchmark_vector_add(benchmark, 1_000_000, ctx)
        
        print("[4/4] Benchmarking 10M elements...")
        benchmark_vector_add(benchmark, 10_000_000, ctx)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    benchmark.dump_report()
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("\nWhat 'iters' means:")
    print("  The benchmark framework automatically adjusts iteration count")
    print("  based on execution time to get statistically valid measurements.")
    print("  - Fast operations (small arrays): Run 100+ times")
    print("  - Slow operations (large arrays): Run fewer times")
    print("\nExpected patterns:")
    print("  • Time scales linearly with array size")
    print("  • GFLOPS/s increases with size (better GPU utilisation)")
    print("  • Iteration count decreases with size (longer operations)")
    print("\nOn Apple Silicon GPU:")
    print("  • Unified memory reduces copy overhead")
    print("  • Metal backend provides parallel execution")
    print("  • Peak performance at 1M-100M element range")
    print("="*70 + "\n")
