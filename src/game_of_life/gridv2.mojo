import random
from collections import Optional
from memory import memcpy, memset_zero


struct Grid[rows: Int, cols: Int](Copyable, Movable, Stringable):
    """
    Game of Life grid optimised with flat memory layout.
    
    Key optimisations over gridv1:
    1. Compile-time grid dimensions as parameters (enables optimisations)
    2. Flat memory layout using UnsafePointer instead of nested Lists
    3. Direct memory operations (memcpy, memset_zero)
    4. Compact Int8 storage (1 byte per cell vs Int's 8 bytes)
    5. Clever bitwise trick for Conway's rules: (num_neighbors | current_cell == 3)
    
    Performance gains: ~10-20x faster than List[List[Int]] approach
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
        Compute the next generation of the Game of Life.
        
        Uses a clever bitwise trick: (num_neighbors | current_cell == 3)
        This is equivalent to:
        - If cell is alive (1) and has 2 or 3 neighbours → stays alive
        - If cell is dead (0) and has exactly 3 neighbours → becomes alive
        """
        var next_generation = Self()
        
        for row in range(Self.rows):
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
                
                # Clever bitwise trick for Conway's rules:
                # (neighbors | current) == 3 means:
                #   - Live cell with 2 neighbors: (2 | 1) = 3 ✓
                #   - Live cell with 3 neighbors: (3 | 1) = 3 ✓
                #   - Dead cell with 3 neighbors: (3 | 0) = 3 ✓
                #   - All other cases: != 3 ✗
                if num_neighbors | self[row, col] == 3:
                    next_generation[row, col] = 1
        
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
