import random
from collections import Optional
from memory import memcpy, memset_zero

from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx
from math import ceildiv
from layout import Layout, LayoutTensor

from sys import has_accelerator


struct Grid[rows: Int, cols: Int](Copyable, Movable, Stringable):
    """
    Game of Life grid with Apple Silicon GPU acceleration.

    Key features:
    1. Uses Mojo's native GPU API with Metal backend
    2. Entire grid computation happens on GPU
    3. Minimal CPU-GPU data transfers
    4. Should be 10-100× faster than CPU versions for large grids

    Requirements:
    - macOS 15+
    - Xcode 16+ with Metal toolchain
    - Mojo nightly build
    - Apple Silicon (M1/M2/M3/M4)
    """

    alias num_cells = Self.rows * Self.cols
    var data: UnsafePointer[Int8, MutOrigin.external]

    fn __init__(out self):
        """Allocate and zero-initialize grid memory on CPU."""
        self.data = alloc[Int8](self.num_cells)
        memset_zero(self.data, self.num_cells)

    fn __copyinit__(out self, existing: Self):
        """Create a deep copy of the grid."""
        self.data = alloc[Int8](self.num_cells)
        memcpy(dest=self.data, src=existing.data, count=self.num_cells)

    fn __del__(deinit self):
        """Free allocated memory."""
        self.data.free()

    @staticmethod
    fn random(seed: Optional[Int] = None) -> Self:
        """Create a grid with random initial state."""
        if seed:
            random.seed(seed.value())
        else:
            random.seed()

        var grid = Self()
        random.randint(grid.data, grid.num_cells, 0, 1)
        return grid^

    @always_inline
    fn __getitem__(self, row: Int, col: Int) -> Int8:
        """Get cell value at (row, col)."""
        return self.data[row * Self.cols + col]

    @always_inline
    fn __setitem__(mut self, row: Int, col: Int, value: Int8) -> None:
        """Set cell value at (row, col)."""
        self.data[row * Self.cols + col] = value

    fn __str__(self) -> String:
        """Return visual representation of the grid."""
        var result = String()
        for row in range(Self.rows):
            for col in range(Self.cols):
                if self[row, col] == 1:
                    result += "*"
                else:
                    result += " "
            if row != Self.rows - 1:
                result += "\n"
        return result

    fn evolve_gpu(self, ctx: DeviceContext) raises -> Self:
        """
        Compute the next generation using GPU acceleration.

        This version offloads the entire computation to the Apple Silicon GPU.
        Each cell's computation happens in parallel on the GPU.

        Args:
            ctx: DeviceContext for GPU operations.

        Returns:
            New Grid with evolved state.
        """
        var next_generation = Self()

        # Create host buffer and copy current grid data
        var host_current = ctx.enqueue_create_host_buffer[DType.int8](self.num_cells)
        ctx.synchronize()
        
        for i in range(self.num_cells):
            host_current[i] = self.data[i]

        # Create device buffers
        var device_current = ctx.enqueue_create_buffer[DType.int8](self.num_cells)
        var device_next = ctx.enqueue_create_buffer[DType.int8](self.num_cells)

        # Copy input data to GPU
        ctx.enqueue_copy(dst_buf=device_current, src_buf=host_current)

        # Wrap buffers in LayoutTensors
        alias layout = Layout.row_major(self.num_cells)
        var current_tensor = LayoutTensor[DType.int8, layout](device_current)
        var next_tensor = LayoutTensor[DType.int8, layout](device_next)

        # Define GPU kernel function
        fn evolve_kernel(
            current_t: LayoutTensor[DType.int8, layout, MutAnyOrigin],
            next_t: LayoutTensor[DType.int8, layout, MutAnyOrigin],
        ):
            """
            GPU kernel that processes cells in parallel.

            Each thread computes the next state of one cell.
            """
            # Get global thread ID
            var tid = Int(block_idx.x * block_dim.x + thread_idx.x)

            # Check bounds
            if tid >= Self.rows * Self.cols:
                return

            # Convert 1D thread ID to 2D grid coordinates
            var row = tid // Self.cols
            var col = tid % Self.cols

            # Calculate neighbor row/col indices with wrap-around
            var row_above = (row - 1 + Self.rows) % Self.rows
            var row_below = (row + 1) % Self.rows
            var col_left = (col - 1 + Self.cols) % Self.cols
            var col_right = (col + 1) % Self.cols

            # Count neighbors (8-connected)
            var num_neighbors = (
                Int(current_t[row_above * Self.cols + col_left])
                + Int(current_t[row_above * Self.cols + col])
                + Int(current_t[row_above * Self.cols + col_right])
                + Int(current_t[row * Self.cols + col_left])
                + Int(current_t[row * Self.cols + col_right])
                + Int(current_t[row_below * Self.cols + col_left])
                + Int(current_t[row_below * Self.cols + col])
                + Int(current_t[row_below * Self.cols + col_right])
            )

            # Apply Conway's rules using bitwise trick
            var cell_state = Int(current_t[tid])
            if num_neighbors | cell_state == 3:
                next_t[tid] = 1
            else:
                next_t[tid] = 0

        # Launch kernel with appropriate grid/block dimensions
        alias threads_per_block = 256
        var num_blocks = ceildiv(self.num_cells, threads_per_block)

        # Execute kernel on GPU
        ctx.enqueue_function_checked[evolve_kernel, evolve_kernel](
            current_tensor,
            next_tensor,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        # Create host buffer for result
        var host_result = ctx.enqueue_create_host_buffer[DType.int8](self.num_cells)

        # Copy result back to CPU
        ctx.enqueue_copy(dst_buf=host_result, src_buf=device_next)

        # Wait for GPU to finish
        ctx.synchronize()

        # Copy result to next generation
        for i in range(self.num_cells):
            next_generation.data[i] = host_result[i]

        return next_generation^

    fn evolve(self) raises -> Self:
        """
        CPU fallback version for compatibility.

        Note: For GPU acceleration, use evolve_gpu() with a DeviceContext.
        """
        var next_generation = Self()

        for row in range(Self.rows):
            var row_above = (row - 1 + Self.rows) % Self.rows
            var row_below = (row + 1) % Self.rows

            var curr_row = self.data + row * Self.cols
            var above_row = self.data + row_above * Self.cols
            var below_row = self.data + row_below * Self.cols
            var next_row = next_generation.data + row * Self.cols

            for col in range(Self.cols):
                var col_left = (col - 1 + Self.cols) % Self.cols
                var col_right = (col + 1) % Self.cols

                var num_neighbors = (
                    above_row[col_left]
                    + above_row[col]
                    + above_row[col_right]
                    + curr_row[col_left]
                    + curr_row[col_right]
                    + below_row[col_left]
                    + below_row[col]
                    + below_row[col_right]
                )

                if num_neighbors | curr_row[col] == 3:
                    next_row[col] = 1

        return next_generation^

    fn fingerprint_str(self) -> String:
        """
        Return the entire grid as a flat string of '0' and '1' characters.
        Used for cross-language correctness verification.
        """
        var s = String()
        for i in range(Self.num_cells):
            s += "1" if self.data[i] == 1 else "0"
        return s


fn check_gpu_available() raises -> Bool:
    """Check if GPU acceleration is available."""
    try:
        if has_accelerator():
            print("Found GPU")
            return True
    except:
        pass
    return False


fn main() raises:
    """
    Demonstration of GPU-accelerated Game of Life.

    This will try to use GPU acceleration if available,
    falling back to CPU if not.
    """
    alias test_rows = 256
    alias test_cols = 256
    alias test_gens = 100

    print("=== Mojo GPU-Accelerated Game of Life ===")
    print("Grid:", test_rows, "×", test_cols)
    print("Generations:", test_gens)
    print()

    # Check GPU availability
    var has_gpu = check_gpu_available()

    if not has_gpu:
        print("WARNING: No GPU detected or GPU support not available")
        print("Falling back to CPU implementation")
        print()
        print("Requirements for GPU support:")
        print("  - macOS 15+ (Sequoia)")
        print("  - Xcode 16+ with Metal toolchain")
        print("  - Mojo nightly build")
        print("  - Apple Silicon (M1/M2/M3/M4)")
        print()

    # Create initial random grid
    var grid = Grid[test_rows, test_cols].random(seed=42)

    if has_gpu:
        # GPU path
        print("Using GPU acceleration")
        var ctx = DeviceContext()

        from time import perf_counter_ns

        var start = perf_counter_ns()

        for _ in range(test_gens):
            grid = grid.evolve_gpu(ctx)

        var end = perf_counter_ns()
        var duration = Float64(end - start) / 1_000_000_000.0

        print("GPU time:", duration, "s")
        print("Per generation:", duration / test_gens * 1000, "ms")
    else:
        # CPU fallback path
        print("Using CPU (fallback)")

        from time import perf_counter_ns

        var start = perf_counter_ns()

        for _ in range(test_gens):
            grid = grid.evolve()

        var end = perf_counter_ns()
        var duration = Float64(end - start) / 1_000_000_000.0

        print("CPU time", duration, "s")
        print("Per generation:", duration / test_gens * 1000, "ms")

    print()
    print("Final grid fingerprint:", grid.fingerprint_str()[:32], "...")
