import random


struct Grid[rows: Int, cols: Int](Copyable, Movable, Stringable):
    var data: List[List[Int]]

    fn __str__(self) -> String:
        # Create an empty String
        str = String()

        # Iterate through rows 0 through rows-1
        for row in range(Self.rows):
            # Iterate through columns 0 through cols-1
            for col in range(Self.cols):
                if self.data[row][col] == 1:
                    str += "*"  # If cell is populated, append an asterisk
                else:
                    str += " "  # If cell is not populated, append a space
            if row != Self.rows - 1:
                str += "\n"  # Add a newline between rows, but not at the end
        return str

    fn __getitem__(self, row: Int, col: Int) -> Int:
        return self.data[row][col]

    fn __setitem__(mut self, row: Int, col: Int, value: Int) -> None:
        self.data[row][col] = value

    fn __init__(out self):
        """Initialize with empty grid."""
        self.data = List[List[Int]]()
        for _ in range(Self.rows):
            var row_data = List[Int]()
            for _ in range(Self.cols):
                row_data.append(0)
            self.data.append(row_data^)

    @staticmethod
    fn random() -> Self:
        # Seed the random number generator using the current time.
        random.seed()

        var result = Self()
        for row in range(Self.rows):
            for col in range(Self.cols):
                result[row, col] = Int(random.random_si64(0, 1))

        return result^

    fn evolve(self) -> Self:
        var next_generation = Self()

        for row in range(Self.rows):
            # Calculate neighboring row indices, handling "wrap-around"
            var row_above = (row - 1) % Self.rows
            var row_below = (row + 1) % Self.rows

            for col in range(Self.cols):
                # Calculate neighboring column indices, handling "wrap-around"
                var col_left = (col - 1) % Self.cols
                var col_right = (col + 1) % Self.cols

                # Determine number of populated cells around the current cell
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

                # Determine the state of the current cell for the next generation
                var new_state = 0
                if self[row, col] == 1 and (
                    num_neighbors == 2 or num_neighbors == 3
                ):
                    new_state = 1
                elif self[row, col] == 0 and num_neighbors == 3:
                    new_state = 1
                next_generation[row, col] = new_state

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

