"""
Simple Vector Addition - Clean Kernel Example

This demonstrates the core GPU kernel pattern without benchmarking complexity.
Focus: The essential GPU programming concepts.
"""

from sys import has_accelerator
from gpu import global_idx
from gpu.host import DeviceContext
from memory import alloc
from builtin._closure import __ownership_keepalive


# ============================================================================
# THE KERNEL - This runs on the GPU/accelerator
# ============================================================================

fn vector_add_kernel(
    in0: UnsafePointer[Float32, MutAnyOrigin],
    in1: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """
    GPU kernel that adds two vectors element-wise.
    
    Each thread processes one element:
    - Thread 0 computes: output[0] = in0[0] + in1[0]
    - Thread 1 computes: output[1] = in0[1] + in1[1]
    - And so on...
    """
    # Get this thread's unique index
    var idx = global_idx.x
    
    # Bounds check - threads beyond array size exit early
    if idx >= UInt(size):
        return
    
    # The actual computation - one addition per thread
    output[idx] = in0[idx] + in1[idx]


# ============================================================================
# HOST CODE - This runs on the CPU and coordinates the GPU
# ============================================================================

fn run_vector_add(size: Int) raises:
    """Run vector addition on the GPU/accelerator."""
    
    print("\n" + "="*60)
    print("Vector Addition:", size, "elements")
    print("Platform:", "Apple Silicon GPU (Metal)" if has_accelerator() else "CPU")
    print("="*60)
    
    # 1. Allocate and initialize host (CPU) memory
    var a_host = alloc[Float32](size)
    var b_host = alloc[Float32](size)
    var c_host = alloc[Float32](size)
    
    for i in range(size):
        a_host[i] = Float32(i)
        b_host[i] = Float32(2.0)
    
    # 2. Create device context and allocate device memory
    with DeviceContext() as ctx:
        var a_device = ctx.enqueue_create_buffer[DType.float32](size)
        var b_device = ctx.enqueue_create_buffer[DType.float32](size)
        var c_device = ctx.enqueue_create_buffer[DType.float32](size)
        
        # 3. Copy input data to device
        ctx.enqueue_copy(a_device, a_host)
        ctx.enqueue_copy(b_device, b_host)
        
        # 4. Launch the kernel
        var threads_per_block = 256
        var num_blocks = (size + threads_per_block - 1) // threads_per_block
        
        print("Launching", num_blocks, "blocks ×", threads_per_block, "threads")
        
        ctx.enqueue_function_checked[vector_add_kernel, vector_add_kernel](
            a_device,
            b_device,
            c_device,
            size,
            grid_dim=(num_blocks,),
            block_dim=(threads_per_block,),
        )
        
        # 5. Wait for completion and copy results back
        ctx.synchronize()
        ctx.enqueue_copy(c_host, c_device)
        
        # Keep device buffers alive until done
        __ownership_keepalive(a_device, b_device, c_device)
    
    # 6. Validate results
    var errors = 0
    for i in range(size):
        var expected = Float32(i + 2)
        if c_host[i] != expected:
            if errors < 3:
                print("Error at", i, ":", c_host[i], "!=", expected)
            errors += 1
    
    if errors == 0:
        print("✅ All", size, "elements correct!")
    else:
        print("❌", errors, "errors found")
        raise Error("Validation failed")
    
    # 7. Cleanup
    a_host.free()
    b_host.free()
    c_host.free()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run vector addition at different scales."""
    
    print("\n" + "="*60)
    print("SIMPLE VECTOR ADDITION - GPU KERNEL DEMO")
    print("="*60)
    
    # Run at multiple scales
    var sizes = [
        10_000,        # Small: 10K elements
        1_000_000,     # Medium: 1M elements  
        10_000_000,    # Large: 10M elements
    ]
    
    for i in range(len(sizes)):
        run_vector_add(sizes[i])
    
    print("\n" + "="*60)
    print("Key Concepts:")
    print("  • Kernel function runs in parallel on many threads")
    print("  • Each thread uses global_idx.x to find its work")
    print("  • Memory flows: CPU → GPU → Compute → CPU")
    print("  • DeviceContext manages GPU resources")
    print("="*60 + "\n")
