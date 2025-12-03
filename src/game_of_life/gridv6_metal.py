"""
gridv6_metal.py — GPU-Accelerated Game of Life using Apple Metal

Uses Metal Performance Shaders (MPS) via PyTorch's Metal backend.
This provides native M1/M2/M3 GPU acceleration.

Requirements:
    pip install torch torchvision

Expected performance: 5-10× faster than NumPy for large grids
"""

import hashlib
import torch


class GridMetal:
    """
    Game of Life grid using Apple Metal GPU acceleration.
    
    Uses PyTorch's MPS (Metal Performance Shaders) backend for
    native Apple Silicon GPU computation.
    """
    
    def __init__(self, rows: int, cols: int, data=None):
        self.rows = rows
        self.cols = cols
        
        # Check if Metal is available
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "Metal Performance Shaders (MPS) not available. "
                "Make sure you're on macOS 12.3+ with Apple Silicon."
            )
        
        self.device = torch.device("mps")
        
        if data is not None:
            # Convert input data to GPU tensor
            if isinstance(data, list):
                # Python list of lists
                self.data = torch.tensor(data, dtype=torch.uint8, device=self.device)
            else:
                # NumPy array or other
                self.data = torch.tensor(data, dtype=torch.uint8).to(self.device)
        else:
            # Create empty grid on GPU
            self.data = torch.zeros(
                (rows, cols), dtype=torch.uint8, device=self.device
            )
    
    def evolve(self) -> "GridMetal":
        """
        Evolve the grid by one generation using GPU acceleration.
        
        Uses PyTorch operations that are executed on Metal GPU.
        All computations happen on the GPU - no CPU-GPU transfers
        until the final result is needed.
        """
        # Pad with wrap-around (toroidal topology)
        # This handles the edge cases elegantly
        padded = torch.nn.functional.pad(
            self.data.float().unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1),
            mode='circular'
        ).squeeze()
        
        # Count neighbors using convolution
        # This is where the GPU shines - highly parallel operation
        neighbors = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]  # row above
            + padded[1:-1, :-2] + padded[1:-1, 2:]                  # current row (skip center)
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]    # row below
        )
        
        # Apply Conway's rules (vectorized on GPU)
        # Live cell with 2 or 3 neighbors survives
        # Dead cell with exactly 3 neighbors becomes alive
        current = self.data.float()
        alive = (current == 1) & ((neighbors == 2) | (neighbors == 3))
        born = (current == 0) & (neighbors == 3)
        
        next_data = (alive | born).to(torch.uint8)
        
        return GridMetal(self.rows, self.cols, next_data)
    
    def to_cpu(self):
        """Transfer grid data from GPU to CPU."""
        return self.data.cpu().numpy()
    
    def fingerprint(self) -> str:
        """
        Compute SHA-256 fingerprint for correctness verification.
        Transfers data to CPU only for fingerprinting.
        """
        flat_str = ''.join(str(int(x)) for x in self.data.flatten().cpu().tolist())
        return hashlib.sha256(flat_str.encode()).hexdigest()
    
    def __str__(self) -> str:
        """String representation of the grid."""
        # Transfer to CPU for display
        cpu_data = self.to_cpu()
        lines = []
        for row in cpu_data:
            line = ''.join('*' if cell else ' ' for cell in row)
            lines.append(line)
        return '\n'.join(lines)
    
    @staticmethod
    def random(rows: int, cols: int, seed: int = 42) -> "GridMetal":
        """Create a random grid on the GPU."""
        torch.manual_seed(seed)
        data = torch.randint(0, 2, (rows, cols), dtype=torch.uint8, device="mps")
        return GridMetal(rows, cols, data)


def benchmark_metal(rows: int, cols: int, generations: int, warmup: int = 5):
    """
    Benchmark the Metal GPU implementation.
    
    This is a standalone function for testing.
    """
    import time
    
    print(f"\nMetal GPU Benchmark")
    print(f"Grid: {rows}×{cols}, Generations: {generations}")
    print(f"Device: {torch.backends.mps.is_available() and 'Apple Silicon GPU (Metal)' or 'Not available'}")
    
    # Create random grid
    grid = GridMetal.random(rows, cols, seed=42)
    
    # Warmup (important for GPU - initializes kernels)
    print("Warming up GPU...", end="", flush=True)
    for _ in range(warmup):
        grid = grid.evolve()
    
    # Synchronize GPU (make sure warmup is done)
    torch.mps.synchronize()
    print(" done")
    
    # Benchmark
    print("Running benchmark...", end="", flush=True)
    start = time.perf_counter()
    
    for _ in range(generations):
        grid = grid.evolve()
    
    # Synchronize GPU before timing
    torch.mps.synchronize()
    
    duration = time.perf_counter() - start
    print(f" done")
    
    print(f"Time: {duration:.6f} s")
    print(f"Per generation: {duration/generations*1000:.3f} ms")
    print(f"Fingerprint: {grid.fingerprint()[:32]}...")
    
    return duration, grid


if __name__ == "__main__":
    # Test the Metal implementation
    try:
        benchmark_metal(512, 512, 1000, warmup=10)
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nMake sure you have PyTorch with Metal support installed:")
        print("  pip install torch torchvision")
