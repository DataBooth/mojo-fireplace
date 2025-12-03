# Advent of Code 2025 - Day 1: Secret Entrance

See [**Secret Entrance: Day 1 - Advent of Code 2025** _#AdventOfCode_](https://adventofcode.com/2025/day/1).

This folder contains solutions for Day 1 of Advent of Code 2025, a rotation/dial puzzle where a dial starts at position 50 on a 100-position circle. Instructions (L for left/negative, R for right/positive) rotate the dial, and the password is the count of times it lands on position 0.[1]

## File Contents

```
├── day1_initial.py     # Initial Python prototype with manual wrapping and revolution counting
├── day1_input_test.txt # Sample input data for testing (10 rotations)
├── day1.md             # This README with link to official puzzle page
├── day1.mojo           # Optimised Mojo implementation using modular arithmetic
└── day1.py             # Clean Python version with % operator for wrapping
```

## Implementations

- **day1_initial.py**: Early debug version (DEBUG=True hardcoded) that manually handles negative/positive wrapping and tracks full revolutions. Outputs both zero counts and total revolutions; uses verbose printing for every step.
- **day1.py**: Refined Python script with built-in DEBUG flag. Computes max/min rotations, uses `(current + rotation) % N_POSITION` for efficient wrapping, and prints the password (zero count).
- **day1.mojo**: Mojo port matching day1.py logic. Features `alias` constants, `List[Int]` for rotations, and a `load_rotations` function that parses L/R prefixes. Debug mode uses test input and expects password 3.

## Usage

Run `day1.py` or `day1.mojo` with test input for verification (test password: 3). Replace with `day1_input.txt` for full puzzle. Both languages handle wrapping identically via modulo, simplifying over the initial manual adjustments.