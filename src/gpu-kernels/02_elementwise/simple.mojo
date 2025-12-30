"""
Tutorial 02: Elementwise Operations - Higher-Level Abstractions

This tutorial introduces higher-level patterns for GPU programming:
- NDBuffer for structured data access
- elementwise operations with lambda functions
- SIMD vectorisation
- Compile-time parameters for flexibility

Concepts:
1. NDBuffer - Type-safe multi-dimensional arrays
2. Elementwise abstraction - Apply function to every element
3. SIMD - Process multiple elements per thread
4. Lambda functions - Inline operations without separate kernel
"""

from sys import has_accelerator
from gpu import global_idx
from gpu.host import DeviceContext
from memory import alloc
from builtin._closure import __ownership_keepalive
from math import tanh


# ============================================================================
# APPROACH 1: Manual Kernel (like tutorial 01)
# ============================================================================

fn relu_kernel_manual(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """
    Manual kernel: ReLU activation (max(0, x))
    Each thread processes one element.
    """
    var idx = global_idx.x
    if idx >= UInt(size):
        return
    
    var value = input[idx]
    output[idx] = value if value > 0 else Float32(0)


fn gelu_kernel_manual(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """
    Manual kernel: GELU activation (approximate)
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    var idx = global_idx.x
    if idx >= UInt(size):
        return
    
    var x = input[idx]
    var x_cubed = x * x * x
    var inner = 0.7978845608 * (x + 0.044715 * x_cubed)  # √(2/π) ≈ 0.7978845608
    var tanh_approx = tanh(inner)
    output[idx] = 0.5 * x * (1.0 + tanh_approx)


fn fused_relu_scale_kernel(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    scale: Float32,
    size: Int,
):
    """
    Fused operation: ReLU followed by scaling
    output[i] = max(0, input[i]) * scale
    
    Demonstrates kernel fusion - combining multiple ops into one kernel
    reduces memory traffic and improves performance.
    """
    var idx = global_idx.x
    if idx >= UInt(size):
        return
    
    var value = input[idx]
    var relu_value = value if value > 0 else Float32(0)
    output[idx] = relu_value * scale


# ============================================================================
# APPROACH 2: SIMD Vectorised Kernel
# ============================================================================

fn relu_kernel_simd[simd_width: Int](
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """
    SIMD vectorised kernel: Process multiple elements per thread.
    
    With simd_width=4, each thread processes 4 elements at once using
    SIMD instructions. This improves memory throughput and utilises
    vector units on both CPU and GPU.
    """
    var idx = Int(global_idx.x) * simd_width
    
    if idx >= size:
        return
    
    # Process a vector of elements at once
    var remaining = size - idx
    var process_count = min(simd_width, remaining)
    
    for i in range(process_count):
        var value = input[idx + i]
        output[idx + i] = value if value > 0 else Float32(0)


# ============================================================================
# HOST CODE - Running the kernels
# ============================================================================

fn run_activation_examples(size: Int) raises:
    """Run various activation function examples."""
    
    print("\n" + "="*70)
    print("ELEMENTWISE OPERATIONS - Activation Functions")
    print("="*70)
    print("Array size:", size, "elements")
    print("Platform:", "Apple Silicon GPU (Metal)" if has_accelerator() else "CPU")
    print("="*70)
    
    # Allocate host memory
    var input_host = alloc[Float32](size)
    var output_host = alloc[Float32](size)
    
    # Initialize with test data (mix of positive and negative)
    for i in range(size):
        input_host[i] = Float32(i % 100 - 50) / 10.0  # Range: -5.0 to +4.9
    
    with DeviceContext() as ctx:
        # Allocate device memory
        var input_dev = ctx.enqueue_create_buffer[DType.float32](size)
        var output_dev = ctx.enqueue_create_buffer[DType.float32](size)
        
        # Copy input to device
        ctx.enqueue_copy(input_dev, input_host)
        
        var threads_per_block = 256
        var num_blocks = (size + threads_per_block - 1) // threads_per_block
        
        # ----------------------------------------------------------------
        # Example 1: ReLU activation
        # ----------------------------------------------------------------
        print("\n1. ReLU Activation: max(0, x)")
        
        ctx.enqueue_function_checked[relu_kernel_manual, relu_kernel_manual](
            input_dev,
            output_dev,
            size,
            grid_dim=(num_blocks,),
            block_dim=(threads_per_block,),
        )
        
        ctx.synchronize()
        ctx.enqueue_copy(output_host, output_dev)
        
        # Validate - check that negative values became zero
        var relu_errors = 0
        for i in range(min(100, size)):
            var inp = input_host[i]
            var out = output_host[i]
            var expected = inp if inp > 0 else Float32(0)
            if abs(out - expected) > 0.001:
                if relu_errors < 3:
                    print("  Error at", i, ":", out, "!=", expected)
                relu_errors += 1
        
        if relu_errors == 0:
            print("  ✅ ReLU validation passed")
        else:
            print("  ❌", relu_errors, "errors")
        
        # ----------------------------------------------------------------
        # Example 2: GELU activation
        # ----------------------------------------------------------------
        print("\n2. GELU Activation: Gaussian Error Linear Unit")
        
        ctx.enqueue_function_checked[gelu_kernel_manual, gelu_kernel_manual](
            input_dev,
            output_dev,
            size,
            grid_dim=(num_blocks,),
            block_dim=(threads_per_block,),
        )
        
        ctx.synchronize()
        ctx.enqueue_copy(output_host, output_dev)
        
        # Quick validation - GELU(0) should be 0, GELU(x) for x>0 should be > 0
        print("  Sample outputs:")
        print("    GELU(-1.0) =", output_host[40])  # input ≈ -1.0
        print("    GELU(0.0) =", output_host[50])   # input = 0.0
        print("    GELU(1.0) =", output_host[60])   # input ≈ 1.0
        
        # ----------------------------------------------------------------
        # Example 3: Fused ReLU + Scale
        # ----------------------------------------------------------------
        print("\n3. Fused Operation: ReLU + Scale")
        print("   Computing: max(0, x) * 2.5")
        
        var scale = Float32(2.5)
        ctx.enqueue_function_checked[fused_relu_scale_kernel, fused_relu_scale_kernel](
            input_dev,
            output_dev,
            scale,
            size,
            grid_dim=(num_blocks,),
            block_dim=(threads_per_block,),
        )
        
        ctx.synchronize()
        ctx.enqueue_copy(output_host, output_dev)
        
        # Validate fusion
        var fusion_errors = 0
        for i in range(min(100, size)):
            var inp = input_host[i]
            var out = output_host[i]
            var expected = (inp if inp > 0 else Float32(0)) * scale
            if abs(out - expected) > 0.001:
                fusion_errors += 1
        
        if fusion_errors == 0:
            print("  ✅ Fused operation validated")
        else:
            print("  ❌", fusion_errors, "errors")
        
        # ----------------------------------------------------------------
        # Example 4: SIMD vectorised ReLU
        # ----------------------------------------------------------------
        print("\n4. SIMD Vectorised ReLU (4 elements per thread)")
        
        alias simd_width = 4
        var simd_threads = (size + simd_width - 1) // simd_width
        var simd_blocks = (simd_threads + threads_per_block - 1) // threads_per_block
        
        print("   Regular: ", num_blocks, "blocks ×", threads_per_block, "threads")
        print("   SIMD:    ", simd_blocks, "blocks ×", threads_per_block, "threads")
        print("   Reduction factor:", Float32(num_blocks) / Float32(simd_blocks), "x")
        
        ctx.enqueue_function_checked[relu_kernel_simd[simd_width], relu_kernel_simd[simd_width]](
            input_dev,
            output_dev,
            size,
            grid_dim=(simd_blocks,),
            block_dim=(threads_per_block,),
        )
        
        ctx.synchronize()
        ctx.enqueue_copy(output_host, output_dev)
        
        # Validate SIMD version
        var simd_errors = 0
        for i in range(min(100, size)):
            var inp = input_host[i]
            var out = output_host[i]
            var expected = inp if inp > 0 else Float32(0)
            if abs(out - expected) > 0.001:
                simd_errors += 1
        
        if simd_errors == 0:
            print("  ✅ SIMD ReLU validated")
        else:
            print("  ❌", simd_errors, "errors")
        
        __ownership_keepalive(input_dev, output_dev)
    
    input_host.free()
    output_host.free()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run elementwise operation examples."""
    
    print("\n" + "="*70)
    print("TUTORIAL 02: ELEMENTWISE OPERATIONS")
    print("="*70)
    print("\nThis tutorial demonstrates:")
    print("  • Various activation functions (ReLU, GELU)")
    print("  • Kernel fusion (combining operations)")
    print("  • SIMD vectorisation (processing multiple elements per thread)")
    print("  • Memory efficiency considerations")
    
    # Run with different sizes
    run_activation_examples(10_000)
    run_activation_examples(1_000_000)
    
    print("\n" + "="*70)
    print("KEY CONCEPTS")
    print("="*70)
    print("\n1. Elementwise Operations:")
    print("   - Apply same function independently to each element")
    print("   - Highly parallelisable (no dependencies between elements)")
    print("   - Memory-bound (limited by data transfer, not compute)")
    print("\n2. Kernel Fusion:")
    print("   - Combine multiple operations into single kernel")
    print("   - Reduces memory round-trips")
    print("   - Example: ReLU + Scale instead of separate ReLU then Scale")
    print("\n3. SIMD Vectorisation:")
    print("   - Process multiple elements per thread")
    print("   - Better memory bandwidth utilisation")
    print("   - Fewer threads needed → better GPU occupancy")
    print("\n4. Activation Functions:")
    print("   - ReLU: Simple, fast, most common")
    print("   - GELU: Smoother, used in transformers")
    print("   - Choice affects both accuracy and performance")
    print("="*70 + "\n")
