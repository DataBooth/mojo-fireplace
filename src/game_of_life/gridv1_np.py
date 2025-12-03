import numpy as np
from typing import Self
import hashlib


class GridNP:
    def __init__(self, data: np.ndarray):
        self.data = data.astype(np.uint8)
        self.rows, self.cols = data.shape

    @classmethod
    def random(cls, rows: int, cols: int) -> Self:
        data = np.random.randint(0, 2, size=(rows, cols), dtype=np.uint8)
        return cls(data)

    def evolve(self) -> Self:
        # Correct wrap-around using np.roll (faster and correct)
        neighbors = sum(
            np.roll(np.roll(self.data, dr, 0), dc, 1)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        )

        new_state = (self.data == 1) & (neighbors == 2) | (neighbors == 3)
        return GridNP(new_state.astype(np.uint8))

    def fingerprint(self) -> str:
        flat_str = "".join("1" if cell else "0" for cell in self.data.flat)
        return hashlib.sha256(flat_str.encode("utf-8")).hexdigest()
