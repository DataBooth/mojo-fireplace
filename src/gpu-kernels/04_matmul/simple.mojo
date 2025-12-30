"""
Tutorial 04: Matrix Multiplication - Using Existing Kernels

Matrix multiplication (matmul) is fundamental to deep learning and scientific computing.
This tutorial wraps an existing naive matmul kernel to demonstrate usage patterns.

Formula: C[i,k] = sum(A[i,j] * B[j,k]) for all j

Key concepts:
1. 2D thread indexing (global_idx.x, global_idx.y)
2. LayoutTensor for structured matrix access
3. Nested loops in kernels (reduction over j dimension)
4. Grid and block dimensions for 2D problems

For production matmul, see: /max/kernels/src/linalg/matmul/
"""

from math import ceildiv
from sys import has_accelerator
from gpu import global_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from memory import alloc
from builtin._closure import __ownership_keepalive


# ============================================================================
# KERNEL: Naive Matrix Multiplication (from examples)
# ============================================================================

fn matmul_naive[
    M: Int, N: Int, K: Int,
    a_layout: Layout, b_layout: Layout, c_layout: Layout,
](
    a: LayoutTensor[DType.float32, a_layout, MutAnyOrigin],
    b: LayoutTensor[DType.float32, b_layout, MutAnyOrigin],
    c: LayoutTensor[DType.float32, c_layout, MutAnyOrigin],
):
    """
    Naive matrix multiplication: C = A @ B.
    
    A: M x K matrix
    B: K x N matrix
    C: M x N matrix
    
    Each thread computes one element of C:
    - Thread (i, j) computes C[i, j]
    - Performs dot product of row i of A with column j of B.
    """
    # 2D thread indexing
    var row = Int(global_idx.y)  # Which row of C
    var col = Int(global_idx.x)  # Which column of C
    
    # Bounds check
    if row < M and col < N:
        var sum: Float32 = 0.0
        
        # Dot product: sum over K dimension
        for k in range(K):
            var a_val = a[row, k]
            var b_val = b[k, col]
            # Scalar multiplication and accumulation
            sum = sum + (a_val[0] * b_val[0])
        
        c[row, col] = sum


# ============================================================================
# HOST CODE
# ============================================================================

fn run_matmul_example[M: Int, N: Int, K: Int]() raises:
    """Run matrix multiplication example."""
    
    print("\n" + "="*60)
    print("Matrix Multiplication:", M, "x", K, "@", K, "x", N, "=", M, "x", N)
    print("Platform:", "Apple Silicon GPU (Metal)" if has_accelerator() else "CPU")
    print("="*60)
    
    # Define layouts (row-major)
    comptime a_layout = Layout.row_major(M, K)
    comptime b_layout = Layout.row_major(K, N)
    comptime c_layout = Layout.row_major(M, N)
    
    with DeviceContext() as ctx:
        # Allocate device buffers
        var a_buffer = ctx.enqueue_create_buffer[DType.float32](a_layout.size())
        var b_buffer = ctx.enqueue_create_buffer[DType.float32](b_layout.size())
        var c_buffer = ctx.enqueue_create_buffer[DType.float32](c_layout.size())
        
        # Initialize matrices on host
        print("\nInitializing matrices...")
        with a_buffer.map_to_host() as host_buffer:
            var a_tensor = LayoutTensor[DType.float32, a_layout](host_buffer)
            for i in range(M):
                for j in range(K):
                    a_tensor[i, j] = Float32(i + j)  # Simple pattern
        
        with b_buffer.map_to_host() as host_buffer:
            var b_tensor = LayoutTensor[DType.float32, b_layout](host_buffer)
            for i in range(K):
                for j in range(N):
                    b_tensor[i, j] = Float32(i - j)  # Simple pattern
        
        # Wrap device buffers in LayoutTensor
        var a_dev = LayoutTensor[DType.float32, a_layout](a_buffer)
        var b_dev = LayoutTensor[DType.float32, b_layout](b_buffer)
        var c_dev = LayoutTensor[DType.float32, c_layout](c_buffer)
        
        # Launch configuration (2D grid)
        comptime block_size = 16
        var grid_x = ceildiv(N, block_size)  # Columns
        var grid_y = ceildiv(M, block_size)  # Rows
        
        print("Launching kernel:")
        print("  Grid:", grid_y, "x", grid_x, "blocks")
        print("  Block:", block_size, "x", block_size, "threads")
        print("  Total threads:", grid_y * block_size * grid_x * block_size)
        
        # Launch kernel
        ctx.enqueue_function_checked[
            matmul_naive[M, N, K, a_layout, b_layout, c_layout],
            matmul_naive[M, N, K, a_layout, b_layout, c_layout]
        ](
            a_dev, b_dev, c_dev,
            grid_dim=(grid_x, grid_y),
            block_dim=(block_size, block_size),
        )
        
        ctx.synchronize()
        
        # Read results
        with c_buffer.map_to_host() as host_buffer:
            var c_tensor = LayoutTensor[DType.float32, c_layout](host_buffer)
            
            # Show small matrices completely
            if M <= 5 and N <= 5:
                print("\nResult matrix C:")
                for i in range(M):
                    var row_str = String("  [")
                    for j in range(N):
                        row_str += String(c_tensor[i, j])
                        if j < N - 1:
                            row_str += ", "
                    row_str += "]"
                    print(row_str)
            else:
                print("\nResult matrix C (sample):")
                print("  C[0,0] =", c_tensor[0, 0])
                print("  C[0,", N-1, "] =", c_tensor[0, N-1])
                if M > 1:
                    print("  C[", M-1, ",0] =", c_tensor[M-1, 0])
                    print("  C[", M-1, ",", N-1, "] =", c_tensor[M-1, N-1])
        
        __ownership_keepalive(a_buffer, b_buffer, c_buffer)
    
    print("✅ Matrix multiplication completed!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run matrix multiplication examples."""
    
    print("\n" + "="*60)
    print("TUTORIAL 04: MATRIX MULTIPLICATION")
    print("="*60)
    print("\nThis tutorial demonstrates:")
    print("  • Using existing matmul kernels")
    print("  • 2D thread indexing (x, y)")
    print("  • LayoutTensor for structured access")
    print("  • Grid configuration for 2D problems")
    
    # Run examples at different sizes
    run_matmul_example[4, 4, 4]()     # Tiny: 4x4
    run_matmul_example[64, 64, 64]()   # Small: 64x64
    run_matmul_example[256, 256, 256]() # Medium: 256x256
    
    print("\n" + "="*60)
    print("KEY CONCEPTS")
    print("="*60)
    
    print("\n1. Matrix Multiplication:")
    print("   - C[i,j] = sum(A[i,k] * B[k,j]) for all k")
    print("   - O(M*N*K) operations")
    print("   - Highly parallelizable")
    
    print("\n2. 2D Thread Indexing:")
    print("   - global_idx.x: column index")
    print("   - global_idx.y: row index")
    print("   - Each thread computes one output element")
    
    print("\n3. Naive Algorithm:")
    print("   - Simple but not optimized")
    print("   - Each thread: K multiply-adds")
    print("   - No shared memory usage")
    print("   - No tiling or blocking")
    
    print("\n4. Performance Limitations:")
    print("   - No memory reuse (loads A and B multiple times)")
    print("   - No shared memory (slow global memory access)")
    print("   - No tiling (poor cache utilization)")
    print("   - Good starting point for learning!")
    
    print("\n5. Production Optimizations:")
    print("   - Tiling: Load blocks into shared memory")
    print("   - Memory coalescing: Ensure adjacent threads access adjacent memory")
    print("   - Register blocking: Accumulate multiple outputs per thread")
    print("   - Tensor cores: Use specialized hardware (NVIDIA)")
    
    print("\n6. Where to Find Better Implementations:")
    print("   - /max/kernels/src/linalg/matmul/ - Production kernels")
    print("   - /examples/custom_ops/kernels/matrix_multiplication.mojo - Tiled")
    print("   - /examples/mojo/gpu-block-and-warp/tiled_matmul.mojo - Educational")
    
    print("="*60 + "\n")
