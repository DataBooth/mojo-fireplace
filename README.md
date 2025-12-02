# 🔥 `mojo-fireplace`

### Warm, practical Mojo examples by DataBooth

Welcome to the fireplace — a growing collection of clean, well-commented Mojo (🔥) examples that started with Advent of Code 2025 Day 1 and will keep growing all year round.

Every snippet here is deliberately kept close to Python so you can see exactly how little (or how much) changes when you move to Mojo. Think of this repo as the cosy spot where Python developers come to warm their hands on static typing, ownership, and blazing performance — without freezing in the cold.

## What you’ll find here

| Folder / File                | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `day01-dial/`                | Advent of Code 2025 Day 1 – the “sleigh ride” side-by-side Python ↔ Mojo example |
| `game-of-life/`              | (coming soon) Full Mojo implementation of Conway’s Game of Life with SIMD |
| `performance-showcase/`      | (planned) Real-world benchmarks and gradual optimisation paths            |
| …                            | More examples added regularly                                               |

Each example is designed to accompany a blog post on the DataBooth site, where we dive deeper into the why and the wow of Mojo.

## Why this repo exists

At [DataBooth](https://www.databooth.com.au) we help medium-to-large organisations run high-performance, private AI workloads without massive cloud bills or rare C++ talent. Mojo is rapidly becoming one of the most exciting tools in that toolbox, and we believe the best way to learn it is by starting from code you already understand.

So pull up a chair, grab a hot drink, and watch Python slowly turn into systems-level rocket fuel — one gentle example at a time.

## Quick start

```bash
# Clone and play with the Day 1 example
git clone https://github.com/databooth/mojo-fireplace.git
cd mojo-fireplace/day01-dial

# Python version (zero dependencies)
uv run day1.py                # or: python day1.py

# Mojo version
uv add modular                # one-time only, if you don’t have Mojo yet
mojo day1.mojo
