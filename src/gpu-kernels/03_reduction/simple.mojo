"""
Tutorial 03: Reduction Operations

Reductions combine many values into one using an operation like sum, max, or min.
Unlike elementwise operations, threads must cooperate and synchronize.

Key concepts:
1. Shared memory - Fast memory shared by threads in a block
2. Thread synchronization - Using barrier() to coordinate
3. Tree-based reduction - Hierarchical combining of values
4. Atomic operations - Safe concurrent writes

Common reductions: sum, max, min, mean, variance
"""

from sys import has_accelerator
from gpu import global_idx, thread_idx, block_idx, block_dim, barrier, warp
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from memory import alloc, stack_allocation
from os.atomic import Atomic
from builtin._closure import __ownership_keepalive


# ============================================================================
# KERNEL 1: Simple Block-Level Sum (Educational)
# ============================================================================

fn sum_kernel_simple(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """
    Simple reduction: Sum all elements in input array.
    
    Strategy:
    1. Each thread loads one element
    2. Threads cooperate within each block using shared memory
    3. Block-level results are combined atomically
    
    This demonstrates the basic reduction pattern.
    """
    comptime threads_per_block = 256
    
    # Allocate shared memory for this thread block
    # All threads in the block can access this
    var shared = stack_allocation[
        threads_per_block,
        Float32,
        address_space=AddressSpace.SHARED,
    ]()
    
    # Each thread's local ID within the block (0-255)
    var tid = Int(thread_idx.x)
    
    # Each thread's global ID across all blocks
    var global_id = Int(block_idx.x * block_dim.x + thread_idx.x)
    
    # Step 1: Load data into shared memory
    if global_id < size:
        shared[tid] = input[global_id]
    else:
        shared[tid] = 0.0  # Pad with zeros
    
    # Step 2: Synchronize - ensure all loads complete
    barrier()
    
    # Step 3: Tree-based reduction in shared memory
    # Reduce 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
    var stride = threads_per_block // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        barrier()  # Wait for all threads in this level
        stride //= 2
    
    # Step 4: Thread 0 writes the block's sum to global memory
    if tid == 0:
        # Use atomic add because multiple blocks write to same location
        _ = Atomic.fetch_add(output, shared[0])


# ============================================================================
# KERNEL 2: Max Reduction
# ============================================================================

fn max_kernel(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """Find the maximum value in the array."""
    comptime threads_per_block = 256
    
    var shared = stack_allocation[
        threads_per_block,
        Float32,
        address_space=AddressSpace.SHARED,
    ]()
    
    var tid = Int(thread_idx.x)
    var global_id = Int(block_idx.x * block_dim.x + thread_idx.x)
    
    # Load with minimum value as default
    if global_id < size:
        shared[tid] = input[global_id]
    else:
        shared[tid] = -1e38  # Very negative number
    
    barrier()
    
    # Tree reduction with max operation
    var stride = threads_per_block // 2
    while stride > 0:
        if tid < stride:
            shared[tid] = max(shared[tid], shared[tid + stride])
        barrier()
        stride //= 2
    
    # Write block maximum
    # Note: For max, we can't use atomic operations easily
    # In production, would use a two-stage reduction
    if tid == 0:
        output[block_idx.x] = shared[0]


# ============================================================================
# HOST CODE
# ============================================================================

fn run_sum_example(size: Int) raises:
    """Demonstrate sum reduction."""
    
    print("\n" + "="*60)
    print("Sum Reduction:", size, "elements")
    print("Platform:", "Apple Silicon GPU (Metal)" if has_accelerator() else "CPU")
    print("="*60)
    
    # Allocate memory
    var input_host = alloc[Float32](size)
    var output_host = alloc[Float32](1)
    
    # Initialize with simple values for easy verification
    for i in range(size):
        input_host[i] = 1.0  # Sum should equal size
    
    with DeviceContext() as ctx:
        var input_dev = ctx.enqueue_create_buffer[DType.float32](size)
        var output_dev = ctx.enqueue_create_buffer[DType.float32](1)
        
        # Initialize output to zero
        output_host[0] = 0.0
        ctx.enqueue_copy(output_dev, output_host)
        ctx.enqueue_copy(input_dev, input_host)
        
        # Launch kernel
        comptime threads_per_block = 256
        var num_blocks = (size + threads_per_block - 1) // threads_per_block
        
        print("Launching", num_blocks, "blocks ×", threads_per_block, "threads")
        
        ctx.enqueue_function_checked[sum_kernel_simple, sum_kernel_simple](
            input_dev,
            output_dev,
            size,
            grid_dim=(num_blocks,),
            block_dim=(threads_per_block,),
        )
        
        ctx.synchronize()
        ctx.enqueue_copy(output_host, output_dev)
        
        __ownership_keepalive(input_dev, output_dev)
    
    # Validate
    var expected = Float32(size)
    var actual = output_host[0]
    var error = abs(actual - expected)
    
    print("Expected sum:", expected)
    print("Actual sum:", actual)
    print("Error:", error)
    
    if error < 0.01:
        print("✅ Sum reduction validated!")
    else:
        print("❌ Validation failed")
        raise Error("Sum reduction incorrect")
    
    input_host.free()
    output_host.free()


fn run_max_example(size: Int) raises:
    """Demonstrate max reduction."""
    
    print("\n" + "="*60)
    print("Max Reduction:", size, "elements")
    print("="*60)
    
    var input_host = alloc[Float32](size)
    var output_host = alloc[Float32](1024)  # One per block temporarily
    
    # Initialize with values where max is easy to verify
    for i in range(size):
        input_host[i] = Float32(i % 100)
    input_host[size - 1] = 999.0  # This should be the max
    
    comptime threads_per_block = 256
    var num_blocks = (size + threads_per_block - 1) // threads_per_block
    
    with DeviceContext() as ctx:
        var input_dev = ctx.enqueue_create_buffer[DType.float32](size)
        var output_dev = ctx.enqueue_create_buffer[DType.float32](1024)
        
        ctx.enqueue_copy(input_dev, input_host)
        
        ctx.enqueue_function_checked[max_kernel, max_kernel](
            input_dev,
            output_dev,
            size,
            grid_dim=(num_blocks,),
            block_dim=(threads_per_block,),
        )
        
        ctx.synchronize()
        ctx.enqueue_copy(output_host, output_dev)
        
        __ownership_keepalive(input_dev, output_dev)
    
    # Find max from block results
    var final_max = output_host[0]
    for i in range(num_blocks):
        final_max = max(final_max, output_host[i])
    
    print("Expected max: 999.0")
    print("Actual max:", final_max)
    
    if abs(final_max - 999.0) < 0.01:
        print("✅ Max reduction validated!")
    else:
        print("❌ Validation failed")
    
    input_host.free()
    output_host.free()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run reduction examples."""
    
    print("\n" + "="*60)
    print("TUTORIAL 03: REDUCTION OPERATIONS")
    print("="*60)
    print("\nThis tutorial demonstrates:")
    print("  • Shared memory usage")
    print("  • Thread synchronization with barrier()")
    print("  • Tree-based reduction patterns")
    print("  • Atomic operations for cross-block coordination")
    
    # Run examples
    run_sum_example(1000)
    run_sum_example(100_000)
    run_max_example(10_000)
    
    print("\n" + "="*60)
    print("KEY CONCEPTS")
    print("="*60)
    print("\n1. Shared Memory:")
    print("   - Fast memory shared by threads in a block")
    print("   - Limited size (typically 48-96KB)")
    print("   - Requires explicit synchronization")
    
    print("\n2. Barrier Synchronization:")
    print("   - barrier() makes all threads in block wait")
    print("   - Ensures data is ready before next step")
    print("   - Critical for correctness in reductions")
    
    print("\n3. Tree Reduction:")
    print("   - Hierarchical combining: 256→128→64→...→1")
    print("   - O(log N) steps instead of O(N)")
    print("   - Each step: half the threads, double the stride")
    
    print("\n4. Atomic Operations:")
    print("   - Safe concurrent access to same memory")
    print("   - Used to combine results across blocks")
    print("   - Atomic.fetch_add() for sum reduction")
    
    print("\n5. Two-Stage Reductions:")
    print("   - Stage 1: Reduce within each block")
    print("   - Stage 2: Reduce block results (CPU or second kernel)")
    print("   - Necessary for operations without atomic support (max, min)")
    print("="*60 + "\n")
