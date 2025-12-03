import random
import hashlib
from typing import List


class Grid:
    def __init__(self, rows: int, cols: int, data: List[List[int]] | None = None):
        self.rows = rows
        self.cols = cols
        self.data = data or [[0] * cols for _ in range(rows)]

    @classmethod
    def random(cls, rows: int, cols: int):
        data = [[random.randint(0, 1) for _ in range(cols)] for _ in range(rows)]
        return cls(rows, cols, data)

    def evolve(self):
        next_gen = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                num_neighbors = sum(
                    self.data[(r + dr) % self.rows][(c + dc) % self.cols]
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if not (dr == 0 and dc == 0)
                )
                if self.data[r][c] == 1:
                    new_state = 1 if num_neighbors in (2, 3) else 0
                else:
                    new_state = 1 if num_neighbors == 3 else 0
                row.append(new_state)
            next_gen.append(row)
        return Grid(self.rows, self.cols, next_gen)

    def fingerprint(self) -> str:
        """Return SHA-256 hex digest of the grid as a flat string of 0s and 1s"""
        flat_str = "".join(
            "1" if cell == 1 else "0" for row in self.data for cell in row
        )
        return hashlib.sha256(flat_str.encode("utf-8")).hexdigest()

    def __str__(self) -> str:
        """Pretty-print the grid using █ and space (or ● and . if you prefer)"""
        lines = []
        for row in self.data:
            line = "".join("█" if cell == 1 else "░" for cell in row)
            lines.append(line)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Grid({self.rows}×{self.cols}, alive={sum(sum(row) for row in self.data)})"
        )


# ——— Example usage ———
if __name__ == "__main__":
    random.seed(42)
    grid = Grid.random(20, 40)
    print("Initial grid:")
    print(grid)
    print("\n" + "═" * 40 + "\n")

    grid = grid.evolve()
    print("After 1 generation:")
    print(grid)
    print(f"\nFingerprint: {grid.fingerprint()}")
    print(f"Alive cells: {sum(sum(row) for row in grid.data)}")
