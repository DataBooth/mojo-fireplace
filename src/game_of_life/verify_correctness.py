#!/usr/bin/env python3
"""
Standalone Grid Implementation Correctness Verification

This script verifies that all grid implementations produce identical results
by comparing their final states after the same number of generations with
the same initial configuration.

Usage:
    python verify_correctness.py              # Verify all implementations
    python verify_correctness.py --verbose    # Show full fingerprints
    python verify_correctness.py --save-grids # Save final grids to files
"""

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path

import numpy as np

# Import all Python grid implementations
from gridv1 import Grid as PyGrid
from gridv1_np import GridNP

# Configuration
ROWS = 64
COLS = 64
GENERATIONS = 100
SEED = 42


def generate_test_grid():
    """Generate deterministic test grid."""
    random.seed(SEED)
    np.random.seed(SEED)
    return [[random.randint(0, 1) for _ in range(COLS)] for _ in range(ROWS)]


def grid_to_string(grid_data):
    """Convert 2D grid data to flat string."""
    result = ""
    for row in grid_data:
        for cell in row:
            result += "1" if cell == 1 else "0"
    return result


def save_grid_to_file(grid_data, filename):
    """Save grid to CSV file for manual inspection."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(grid_data)
    print(f"   Saved grid to: {filename}")


class VerificationRunner:
    def __init__(self, verbose=False, save_grids=False):
        self.verbose = verbose
        self.save_grids = save_grids
        self.fingerprints = {}
        self.raw_fingerprints = {}
        
    def verify_python(self, name, GridCls, initial_data):
        """Run and verify a Python implementation."""
        print(f"Testing {name}...", end="", flush=True)
        
        try:
            if "NumPy" in name:
                grid = GridCls(np.array(initial_data, dtype=np.uint8))
            else:
                grid = GridCls(ROWS, COLS, initial_data)
            
            # Evolve
            for _ in range(GENERATIONS):
                grid = grid.evolve()
            
            # Get fingerprint
            fp = grid.fingerprint()
            fp_hash = hashlib.sha256(fp.encode()).hexdigest()
            
            self.fingerprints[name] = fp_hash
            self.raw_fingerprints[name] = fp
            
            if self.verbose:
                print(f"\n   Fingerprint: {fp[:64]}... (length: {len(fp)})")
                print(f"   SHA256: {fp_hash}")
            else:
                print(" ✓")
            
            if self.save_grids:
                # Convert back to 2D for saving
                grid_2d = []
                for r in range(ROWS):
                    row = []
                    for c in range(COLS):
                        idx = r * COLS + c
                        row.append(int(fp[idx]))
                    grid_2d.append(row)
                save_grid_to_file(grid_2d, f"verify_{name.lower().replace(' ', '_')}.csv")
            
            return True
            
        except Exception as e:
            print(f" ✗ FAILED: {e}")
            return False
    
    def compare_all(self):
        """Compare all fingerprints against reference."""
        if not self.fingerprints:
            print("\n✗ No implementations to compare")
            return False
        
        print("\n" + "=" * 70)
        print("Correctness Verification Results")
        print("=" * 70)
        
        # Use first as reference
        ref_name = list(self.fingerprints.keys())[0]
        ref_fp = self.fingerprints[ref_name]
        ref_raw = self.raw_fingerprints[ref_name]
        
        print(f"\nReference: {ref_name}")
        print(f"  SHA256: {ref_fp}")
        
        if self.verbose:
            print(f"  Raw fingerprint (first 64 chars): {ref_raw[:64]}...")
            print(f"  Length: {len(ref_raw)} characters")
        
        print("\nComparison:")
        all_match = True
        
        for name, fp in self.fingerprints.items():
            if name == ref_name:
                continue
            
            if fp == ref_fp:
                print(f"  ✓ {name:<20} matches reference")
            else:
                print(f"  ✗ {name:<20} MISMATCH!")
                all_match = False
                
                if self.verbose:
                    # Find first difference
                    raw_ref = ref_raw
                    raw_test = self.raw_fingerprints[name]
                    
                    if len(raw_ref) != len(raw_test):
                        print(f"    Length mismatch: {len(raw_test)} vs {len(raw_ref)}")
                    else:
                        for i, (c1, c2) in enumerate(zip(raw_ref, raw_test)):
                            if c1 != c2:
                                row = i // COLS
                                col = i % COLS
                                print(f"    First difference at position {i} (row {row}, col {col})")
                                print(f"    Reference: '{c1}', Test: '{c2}'")
                                print(f"    Context: ...{raw_ref[max(0,i-5):i+6]}...")
                                break
        
        print("\n" + "=" * 70)
        
        if all_match:
            print("✓ ALL IMPLEMENTATIONS PRODUCE IDENTICAL RESULTS")
            print("=" * 70)
            return True
        else:
            print("✗ CORRECTNESS VERIFICATION FAILED")
            print("=" * 70)
            return False
    
    def run(self):
        """Run verification on all implementations."""
        print(f"Grid Correctness Verification")
        print(f"Grid size: {ROWS}×{COLS}")
        print(f"Generations: {GENERATIONS}")
        print(f"Seed: {SEED}\n")
        
        # Generate initial grid
        initial_data = generate_test_grid()
        
        # Test Python implementations
        self.verify_python("Pure Python", PyGrid, initial_data)
        self.verify_python("NumPy", GridNP, initial_data)
        
        # Compare results
        return self.compare_all()


def main():
    parser = argparse.ArgumentParser(
        description="Verify correctness of Grid implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed fingerprint information"
    )
    parser.add_argument(
        "--save-grids", "-s",
        action="store_true",
        help="Save final grids to CSV files for manual inspection"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=64,
        help="Grid rows (default: 64)"
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=64,
        help="Grid columns (default: 64)"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of generations (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Override globals if specified
    global ROWS, COLS, GENERATIONS
    ROWS = args.rows
    COLS = args.cols
    GENERATIONS = args.generations
    
    # Run verification
    runner = VerificationRunner(verbose=args.verbose, save_grids=args.save_grids)
    success = runner.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
