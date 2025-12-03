import random
from collections import Optional
from algorithm import parallelize
from memory import memcpy, memset_zero


struct Grid[rows: Int, cols: Int](Copyable, Movable, Stringable):
    """
    Game of Life grid optimised with parallelisation.
    
    Key optimisations over gridv2:
    1. All gridv2 optimisations (flat memory, Int8, compile-time dimensions)
    2. Parallel processing of rows using algorithm.parallelize
    3. Each row computed independently across available CPU cores
    
    Performance gains: Additional 4-8x speedup on multi-core systems
    (depends on number of cores and grid size)
    """
    
    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#
    
    alias num_cells = Self.rows * Self.cols
    var data: UnsafePointer[Int8, MutOrigin.external]
    
    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#
    
    fn __init__(out self):
        """Allocate and zero-initialise grid memory."""
        self.data = alloc[Int8](self.num_cells)
        memset_zero(self.data, self.num_cells)
    
    fn __copyinit__(out self, existing: Self):
        """Create a deep copy of the grid."""
        self.data = alloc[Int8](self.num_cells)
        memcpy(dest=self.data, src=existing.data, count=self.num_cells)
    
    fn __del__(deinit self):
        """Free allocated memory."""
        self.data.free()
    
    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#
    
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
    
    # ===-------------------------------------------------------------------===#
    # Indexing
    # ===-------------------------------------------------------------------===#
    
    @always_inline
    fn __getitem__(self, row: Int, col: Int) -> Int8:
        """Get cell value at (row, col)."""
        return (self.data + row * Self.cols + col)[]
    
    @always_inline
    fn __setitem__(mut self, row: Int, col: Int, value: Int8) -> None:
        """Set cell value at (row, col)."""
        (self.data + row * Self.cols + col)[] = value
    
    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#
    
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
    
    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#
    
    fn evolve(self) -> Self:
        """
        Compute the next generation of the Game of Life in parallel.
        
        Each row is processed independently, allowing the work to be
        distributed across available CPU cores using parallelize().
        """
        var next_generation = Self()
        
        @parameter
        fn worker(row: Int) -> None:
            """Process a single row - called in parallel for each row."""
            # Calculate neighbouring row indices with wrap-around
            var row_above = (row - 1) % Self.rows
            var row_below = (row + 1) % Self.rows
            
            for col in range(Self.cols):
                # Calculate neighbouring column indices with wrap-around
                var col_left = (col - 1) % Self.cols
                var col_right = (col + 1) % Self.cols
                
                # Count populated neighbours
                var num_neighbors = (
                    self[row_above, col_left]
                    + self[row_above, col]
                    + self[row_above, col_right]
                    + self[row, col_left]
                    + self[row, col_right]
                    + self[row_below, col_left]
                    + self[row_below, col]
                    + self[row_below, col_right]
                )
                
                # Apply Conway's rules using bitwise trick
                if num_neighbors | self[row, col] == 3:
                    next_generation[row, col] = 1
        
        # Parallelize the evolution of rows across available CPU cores
        # This is safe because each row's computation is independent
        parallelize[worker](Self.rows)
        
        return next_generation^
    
    fn fingerprint_str(self) -> String:
        """
        Return the entire grid as a flat string of '0' and '1' characters.
        Used for cross-language correctness verification.
        """
        var s = String()
        for r in range(Self.rows):
            for c in range(Self.cols):
                s += "1" if self[r, c] == 1 else "0"
        return s
