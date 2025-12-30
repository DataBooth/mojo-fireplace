"""
Tutorial 05: Softmax - Combining Reductions with Elementwise Operations

Softmax is fundamental to attention mechanisms in transformers and LLMs.
It combines reduction operations (max, sum) with elementwise operations (exp, divide).

Formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

Key concepts:
1. Numerical stability (subtract max before exp)
2. Multi-pass algorithm (3 passes: max, sum, normalize)
3. Online/fused algorithms (single-pass for memory efficiency)
4. Combining reduction + elementwise patterns

This is a real-world kernel used in every transformer model!
"""

from sys import has_accelerator
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from memory import alloc, stack_allocation
from math import exp
from builtin._closure import __ownership_keepalive


# ============================================================================
# KERNEL 1: Three-Pass Softmax (Naive but Clear)
# ============================================================================

fn softmax_pass1_max(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """Pass 1: Find maximum value in the row (for numerical stability)."""
    comptime threads_per_block = 256
    var shared = stack_allocation[threads_per_block, Float32, address_space=AddressSpace.SHARED]()
    
    var tid = Int(thread_idx.x)
    var global_id = Int(block_idx.x * block_dim.x + thread_idx.x)
    
    # Load data
    shared[tid] = input[global_id] if global_id < size else -1e38
    barrier()
    
    # Tree reduction to find max
    var stride = threads_per_block // 2
    while stride > 0:
        if tid < stride:
            shared[tid] = max(shared[tid], shared[tid + stride])
        barrier()
        stride //= 2
    
    # Thread 0 writes the max
    if tid == 0:
        output[block_idx.x] = shared[0]


fn softmax_pass2_exp_sum(
    input: UnsafePointer[Float32, MutAnyOrigin],
    max_val: Float32,
    output: UnsafePointer[Float32, MutAnyOrigin],
    sum_out: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """Pass 2: Compute exp(x - max) and sum them."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    
    if idx < size:
        var val = input[idx]
        var exp_val = exp(val - max_val)
        output[idx] = exp_val
        
        # Simple accumulation (in production, use reduction)
        # For tutorial simplicity, we'll handle sum on CPU


fn softmax_pass3_normalize(
    input: UnsafePointer[Float32, MutAnyOrigin],
    sum_val: Float32,
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """Pass 3: Divide by sum to normalize."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    
    if idx < size:
        output[idx] = input[idx] / sum_val


# ============================================================================
# KERNEL 2: Two-Pass Softmax (More Efficient)
# ============================================================================

fn softmax_two_pass_reduce(
    input: UnsafePointer[Float32, MutAnyOrigin],
    max_out: UnsafePointer[Float32, MutAnyOrigin],
    sum_out: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """Pass 1: Find max AND compute sum of exp with grid-stride loop."""
    comptime threads_per_block = 256
    var shared_max = stack_allocation[threads_per_block, Float32, address_space=AddressSpace.SHARED]()
    var shared_sum = stack_allocation[threads_per_block, Float32, address_space=AddressSpace.SHARED]()
    
    var tid = Int(thread_idx.x)
    var global_id = Int(block_idx.x * block_dim.x + thread_idx.x)
    var grid_stride = Int(block_dim.x)
    
    # Grid-stride loop: each thread processes multiple elements
    var thread_max = Float32(-1e38)
    var idx = global_id
    while idx < size:
        thread_max = max(thread_max, input[idx])
        idx += grid_stride
    
    shared_max[tid] = thread_max
    barrier()
    
    # Reduce max across block
    var stride = threads_per_block // 2
    while stride > 0:
        if tid < stride:
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride])
        barrier()
        stride //= 2
    
    var max_val = shared_max[0]
    barrier()
    
    # Grid-stride loop: compute sum of exp(x - max)
    var thread_sum = Float32(0.0)
    idx = global_id
    while idx < size:
        thread_sum += exp(input[idx] - max_val)
        idx += grid_stride
    
    shared_sum[tid] = thread_sum
    barrier()
    
    # Reduce sum across block
    stride = threads_per_block // 2
    while stride > 0:
        if tid < stride:
            shared_sum[tid] += shared_sum[tid + stride]
        barrier()
        stride //= 2
    
    # Thread 0 writes results
    if tid == 0:
        max_out[block_idx.x] = max_val
        sum_out[block_idx.x] = shared_sum[0]


fn softmax_two_pass_normalize(
    input: UnsafePointer[Float32, MutAnyOrigin],
    max_val: Float32,
    sum_val: Float32,
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """Pass 2: Normalize using precomputed max and sum."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    
    if idx < size:
        output[idx] = exp(input[idx] - max_val) / sum_val


# ============================================================================
# HOST CODE
# ============================================================================

fn run_softmax_three_pass(size: Int) raises:
    """Demonstrate three-pass softmax (clearest algorithm)."""
    
    print("\n" + "="*60)
    print("Three-Pass Softmax:", size, "elements")
    print("Platform:", "Apple Silicon GPU (Metal)" if has_accelerator() else "CPU")
    print("="*60)
    print("Algorithm: max → exp+sum → normalize")
    
    # Allocate memory
    var input_host = alloc[Float32](size)
    var temp_host = alloc[Float32](size)
    var output_host = alloc[Float32](size)
    
    # Initialize with random-ish values
    for i in range(size):
        input_host[i] = Float32(i % 10) - 5.0  # Range: -5 to +4
    
    with DeviceContext() as ctx:
        var input_dev = ctx.enqueue_create_buffer[DType.float32](size)
        var temp_dev = ctx.enqueue_create_buffer[DType.float32](size)
        var output_dev = ctx.enqueue_create_buffer[DType.float32](size)
        var max_dev = ctx.enqueue_create_buffer[DType.float32](1)
        
        ctx.enqueue_copy(input_dev, input_host)
        
        comptime threads_per_block = 256
        var num_blocks = (size + threads_per_block - 1) // threads_per_block
        
        # Pass 1: Find max
        print("\nPass 1: Finding maximum...")
        ctx.enqueue_function_checked[softmax_pass1_max, softmax_pass1_max](
            input_dev, max_dev, size,
            grid_dim=(1,), block_dim=(threads_per_block,),
        )
        
        var max_host = alloc[Float32](1)
        ctx.synchronize()
        ctx.enqueue_copy(max_host, max_dev)
        var max_val = max_host[0]
        print("  Max value:", max_val)
        
        # Pass 2: Compute exp(x - max) and sum on CPU
        print("\nPass 2: Computing exp(x - max)...")
        ctx.enqueue_function_checked[softmax_pass2_exp_sum, softmax_pass2_exp_sum](
            input_dev, max_val, temp_dev, output_dev, size,
            grid_dim=(num_blocks,), block_dim=(threads_per_block,),
        )
        
        ctx.synchronize()
        ctx.enqueue_copy(temp_host, temp_dev)
        
        # Compute sum on CPU
        var sum_val = Float32(0.0)
        for i in range(size):
            sum_val += temp_host[i]
        print("  Sum of exp:", sum_val)
        
        # Pass 3: Normalize
        print("\nPass 3: Normalizing...")
        ctx.enqueue_function_checked[softmax_pass3_normalize, softmax_pass3_normalize](
            temp_dev, sum_val, output_dev, size,
            grid_dim=(num_blocks,), block_dim=(threads_per_block,),
        )
        
        ctx.synchronize()
        ctx.enqueue_copy(output_host, output_dev)
        
        __ownership_keepalive(input_dev, temp_dev, output_dev, max_dev)
        max_host.free()
    
    # Validate: sum should be ~1.0
    var sum_check = Float32(0.0)
    for i in range(size):
        sum_check += output_host[i]
    
    print("\n✅ Results:")
    print("  Sum of softmax outputs:", sum_check, "(should be ≈1.0)")
    print("  First 5 values:", output_host[0], output_host[1], output_host[2], output_host[3], output_host[4])
    
    if abs(sum_check - 1.0) < 0.01:
        print("  ✅ Softmax validated!")
    else:
        print("  ❌ Validation failed")
    
    input_host.free()
    temp_host.free()
    output_host.free()


fn run_softmax_two_pass(size: Int) raises:
    """Demonstrate two-pass softmax (more efficient)."""
    
    print("\n" + "="*60)
    print("Two-Pass Softmax:", size, "elements")
    print("="*60)
    print("Algorithm: max+sum combined → normalize")
    
    var input_host = alloc[Float32](size)
    var output_host = alloc[Float32](size)
    
    for i in range(size):
        input_host[i] = Float32(i % 10) - 5.0
    
    with DeviceContext() as ctx:
        var input_dev = ctx.enqueue_create_buffer[DType.float32](size)
        var output_dev = ctx.enqueue_create_buffer[DType.float32](size)
        var max_dev = ctx.enqueue_create_buffer[DType.float32](1)
        var sum_dev = ctx.enqueue_create_buffer[DType.float32](1)
        
        ctx.enqueue_copy(input_dev, input_host)
        
        comptime threads_per_block = 256
        var num_blocks = (size + threads_per_block - 1) // threads_per_block
        
        # Pass 1: Find max AND sum in one kernel
        # Uses single block with grid-stride loop
        print("\nPass 1: Finding max and computing sum...")
        ctx.enqueue_function_checked[softmax_two_pass_reduce, softmax_two_pass_reduce](
            input_dev, max_dev, sum_dev, size,
            grid_dim=(1,), block_dim=(threads_per_block,),
        )
        
        var max_host = alloc[Float32](1)
        var sum_host = alloc[Float32](1)
        ctx.synchronize()
        ctx.enqueue_copy(max_host, max_dev)
        ctx.enqueue_copy(sum_host, sum_dev)
        
        print("  Max value:", max_host[0])
        print("  Sum of exp:", sum_host[0])
        
        # Pass 2: Normalize
        print("\nPass 2: Normalizing...")
        ctx.enqueue_function_checked[softmax_two_pass_normalize, softmax_two_pass_normalize](
            input_dev, max_host[0], sum_host[0], output_dev, size,
            grid_dim=(num_blocks,), block_dim=(threads_per_block,),
        )
        
        ctx.synchronize()
        ctx.enqueue_copy(output_host, output_dev)
        
        __ownership_keepalive(input_dev, output_dev, max_dev, sum_dev)
        max_host.free()
        sum_host.free()
    
    # Validate
    var sum_check = Float32(0.0)
    for i in range(size):
        sum_check += output_host[i]
    
    print("\n✅ Results:")
    print("  Sum of softmax outputs:", sum_check)
    
    if abs(sum_check - 1.0) < 0.01:
        print("  ✅ Two-pass softmax validated!")
    else:
        print("  ❌ Validation failed")
    
    input_host.free()
    output_host.free()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run softmax examples."""
    
    print("\n" + "="*60)
    print("TUTORIAL 05: SOFTMAX")
    print("="*60)
    print("\nSoftmax converts values to probabilities:")
    print("  • Output values are in range [0, 1]")
    print("  • Output values sum to 1.0")
    print("  • Larger inputs get higher probabilities")
    print("\nUsed in:")
    print("  • Attention mechanisms (transformers)")
    print("  • Classification layers (neural networks)")
    print("  • Reinforcement learning (policy gradients)")
    
    # Run examples
    run_softmax_three_pass(1000)
    run_softmax_two_pass(10_000)
    
    print("\n" + "="*60)
    print("KEY CONCEPTS")
    print("="*60)
    
    print("\n1. Numerical Stability:")
    print("   - Naive: exp(x_i) / sum(exp(x_j)) → can overflow!")
    print("   - Stable: exp(x_i - max) / sum(exp(x_j - max))")
    print("   - Subtracting max prevents overflow in exp()")
    print("   - Mathematically equivalent, numerically stable")
    
    print("\n2. Algorithm Passes:")
    print("   - Three-pass: max → exp+sum → normalize")
    print("     * Clearest to understand")
    print("     * 3 global memory passes")
    print("   - Two-pass: max+sum → normalize")
    print("     * More efficient")
    print("     * 2 global memory passes")
    print("   - One-pass: Online algorithm (advanced)")
    print("     * Most efficient")
    print("     * Complex implementation")
    
    print("\n3. Memory Access Pattern:")
    print("   - Each pass reads entire input")
    print("   - Memory bandwidth is often the bottleneck")
    print("   - Fusing passes reduces memory traffic")
    
    print("\n4. Combining Patterns:")
    print("   - Softmax = Reduction (max, sum) + Elementwise (exp, div)")
    print("   - Reuses reduction patterns from Tutorial 03")
    print("   - Reuses elementwise patterns from Tutorial 02")
    
    print("\n5. Production Optimizations:")
    print("   - Vectorized loads/stores (SIMD)")
    print("   - Warp-level reductions")
    print("   - Online algorithms (single pass)")
    print("   - Fused attention kernels")
    
    print("\n6. Attention Mechanism:")
    print("   - Softmax is core of transformer attention")
    print("   - Applied to query-key similarity scores")
    print("   - Output: attention weights for values")
    print("   - FlashAttention fuses softmax with matmul")
    print("="*60 + "\n")
