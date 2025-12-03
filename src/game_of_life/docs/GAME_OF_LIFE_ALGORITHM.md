# Conway's Game of Life: Algorithm & Mathematics

## Overview

Conway's Game of Life is a cellular automaton devised by mathematician John Conway in 1970. It's a zero-player game where the evolution is determined entirely by the initial state. Despite its simple rules, it exhibits complex emergent behaviour.

## The Grid

### Mathematical Definition

The Game of Life takes place on an infinite 2D orthogonal grid of square cells. In practice, we use a finite grid with boundary conditions.

**Grid representation:**
```
G[t] = {c[i,j] | i ∈ [0, rows), j ∈ [0, cols), c ∈ {0, 1}}
```

Where:
- `G[t]` is the grid state at generation `t`
- `c[i,j]` is the cell at row `i`, column `j`
- `0` = dead cell
- `1` = alive cell

### Boundary Conditions

We implement **toroidal topology** (wrap-around boundaries):
- The top edge wraps to the bottom edge
- The left edge wraps to the right edge
- Creates a "donut" topology with no edges

**Coordinate wrapping:**
```
i_wrapped = (i + rows) mod rows
j_wrapped = (j + cols) mod cols
```

This allows cells at edges to have exactly 8 neighbours like all other cells.

---

## The Rules

Conway's Game of Life has exactly **four rules** that determine the next state of each cell:

### Rule Set

For each cell `c[i,j]` at generation `t`, its state at `t+1` depends on:
1. Its current state: `c[i,j][t]`
2. The number of alive neighbours: `N[i,j]`

**The four rules:**

1. **Underpopulation (Death)**: Any live cell with fewer than 2 live neighbours dies
   ```
   c[i,j][t] = 1 AND N[i,j] < 2  →  c[i,j][t+1] = 0
   ```

2. **Survival**: Any live cell with 2 or 3 live neighbours survives
   ```
   c[i,j][t] = 1 AND N[i,j] ∈ {2, 3}  →  c[i,j][t+1] = 1
   ```

3. **Overpopulation (Death)**: Any live cell with more than 3 live neighbours dies
   ```
   c[i,j][t] = 1 AND N[i,j] > 3  →  c[i,j][t+1] = 0
   ```

4. **Reproduction (Birth)**: Any dead cell with exactly 3 live neighbours becomes alive
   ```
   c[i,j][t] = 0 AND N[i,j] = 3  →  c[i,j][t+1] = 1
   ```

### Condensed Formulation

The rules can be expressed more concisely:

```
c[i,j][t+1] = {
    1  if N[i,j] = 3, or (N[i,j] = 2 AND c[i,j][t] = 1)
    0  otherwise
}
```

Or as a single boolean expression:
```
c[i,j][t+1] = (N[i,j] = 3) OR (N[i,j] = 2 AND c[i,j][t] = 1)
```

---

## Neighbour Counting

### The Moore Neighbourhood

Each cell has exactly **8 neighbours** in the Moore neighbourhood:

```
     NW    N    NE
        ╲  │  ╱
    W ━━━ c ━━━ E
        ╱  │  ╲
     SW    S    SE
```

**Neighbour positions relative to cell `c[i,j]`:**

| Direction | Coordinates | Offset |
|-----------|-------------|--------|
| North-West (NW) | `[i-1, j-1]` | `(-1, -1)` |
| North (N) | `[i-1, j]` | `(-1, 0)` |
| North-East (NE) | `[i-1, j+1]` | `(-1, +1)` |
| West (W) | `[i, j-1]` | `(0, -1)` |
| East (E) | `[i, j+1]` | `(0, +1)` |
| South-West (SW) | `[i+1, j-1]` | `(+1, -1)` |
| South (S) | `[i+1, j]` | `(+1, 0)` |
| South-East (SE) | `[i+1, j+1]` | `(+1, +1)` |

### Neighbour Count Formula

The neighbour count `N[i,j]` is the sum of all 8 neighbour states:

```
N[i,j] = Σ c[(i+Δi) mod rows, (j+Δj) mod cols]
         (Δi,Δj) ∈ {-1,0,+1}² \ {(0,0)}
```

Expanded:
```
N[i,j] = c[i-1, j-1] + c[i-1, j] + c[i-1, j+1]
       + c[i,   j-1]             + c[i,   j+1]
       + c[i+1, j-1] + c[i+1, j] + c[i+1, j+1]
```

With boundary wrapping:
```
row_above = (i - 1 + rows) mod rows
row_below = (i + 1) mod rows
col_left  = (j - 1 + cols) mod cols
col_right = (j + 1) mod cols

N[i,j] = c[row_above, col_left]  + c[row_above, j]     + c[row_above, col_right]
       + c[i,         col_left]                         + c[i,         col_right]
       + c[row_below, col_left]  + c[row_below, j]     + c[row_below, col_right]
```

---

## The Bitwise Trick

### Standard Implementation

Using standard conditionals:
```python
if cell == 1 and (neighbors == 2 or neighbors == 3):
    next_cell = 1
elif cell == 0 and neighbors == 3:
    next_cell = 1
else:
    next_cell = 0
```

### Optimised Bitwise Version

We can collapse all four rules into a single expression using a bitwise trick:

```
next_cell = (neighbors | cell) == 3
```

Or equivalently:
```
next_cell = ((neighbors | cell) == 3) ? 1 : 0
```

### Why This Works

**Truth table analysis:**

| Cell State | Neighbors | `neighbors \| cell` | Result == 3 | Expected | Match? |
|------------|-----------|---------------------|-------------|----------|--------|
| 0 (dead) | 0 | 0 | False | Dead | ✓ |
| 0 (dead) | 1 | 1 | False | Dead | ✓ |
| 0 (dead) | 2 | 2 | False | Dead | ✓ |
| 0 (dead) | **3** | **3** | **True** | **Birth** | ✓ |
| 0 (dead) | 4+ | 4+ | False | Dead | ✓ |
| 1 (alive) | 0 | 1 | False | Death | ✓ |
| 1 (alive) | 1 | 1 | False | Death | ✓ |
| 1 (alive) | **2** | **3** | **True** | **Survive** | ✓ |
| 1 (alive) | **3** | **3** | **True** | **Survive** | ✓ |
| 1 (alive) | 4+ | 5+ | False | Death | ✓ |

**Key insight:**
- Dead cell + 3 neighbours → `0 | 3 = 3` → Birth ✓
- Live cell + 2 neighbours → `1 | 2 = 3` → Survive ✓  
- Live cell + 3 neighbours → `1 | 3 = 3` → Survive ✓
- All other cases → Not 3 → Death/stay dead ✓

### Performance Benefits

**Advantages:**
1. **Branch-free**: No conditional jumps → better CPU pipeline utilisation
2. **Vectorisable**: Can process multiple cells simultaneously with SIMD
3. **Simpler**: One operation instead of multiple conditionals
4. **Faster**: ~2-3× speedup in practice

**Cost:**
- One bitwise OR operation: 1 cycle
- One comparison: 1 cycle
- **Total: 2 cycles vs 5-10 cycles for branching logic**

---

## Algorithm Implementation

### Pseudocode

```
function evolve(grid[rows][cols]) -> grid[rows][cols]:
    next_grid = allocate(rows, cols)
    
    for i in 0 to rows-1:
        row_above = (i - 1 + rows) mod rows
        row_below = (i + 1) mod rows
        
        for j in 0 to cols-1:
            col_left  = (j - 1 + cols) mod cols
            col_right = (j + 1) mod cols
            
            # Count neighbours
            neighbors = grid[row_above][col_left]  + grid[row_above][j]     + grid[row_above][col_right]
                      + grid[i][col_left]                                    + grid[i][col_right]
                      + grid[row_below][col_left]  + grid[row_below][j]     + grid[row_below][col_right]
            
            # Apply rules (bitwise trick)
            cell = grid[i][j]
            next_grid[i][j] = (neighbors | cell) == 3 ? 1 : 0
    
    return next_grid
```

### Complexity Analysis

**Time Complexity:**
- Per cell: O(1) - constant 8 neighbour lookups
- Per generation: O(rows × cols)
- For G generations: **O(G × rows × cols)**

**Space Complexity:**
- Two grids needed (current and next): O(2 × rows × cols)
- Can be optimised to O(rows × cols) with triple-buffering or ping-pong buffers

**Operations per generation:**
- Neighbour lookups: `8 × rows × cols`
- Modulo operations: `4 × rows × cols` (with edge optimisation: much less)
- Comparisons: `1 × rows × cols`
- **Total: ~13 × rows × cols operations**

For a 512×512 grid over 1000 generations:
- Total operations: **~3.4 billion operations**

---

## Optimisation Strategies

### 1. Flat Memory Layout

**Instead of:** `grid[row][col]` (pointer indirection)
**Use:** `grid[row * cols + col]` (single array)

**Benefits:**
- Eliminates pointer chasing
- Improves cache locality
- Enables better compiler optimisations
- ~2× speedup

### 2. Pre-computed Row Pointers

**Instead of:** Computing `row * cols` every access
**Use:** Pre-compute row start pointers

```c
row_above_ptr = data + row_above * cols
row_curr_ptr  = data + row * cols
row_below_ptr = data + row_below * cols

neighbors = row_above_ptr[col_left] + row_above_ptr[col] + ...
```

**Benefits:**
- Eliminates multiplication in inner loop
- Saves ~262 million multiplications per 1000 generations (512×512)
- ~1.5× additional speedup

### 3. Edge-Specific Optimisation

**Key observation:** 99%+ of cells are not on edges!

**For middle columns** (not at j=0 or j=cols-1):
```c
col_left  = col - 1    // No modulo needed!
col_right = col + 1    // No modulo needed!
```

**Impact at 8192×8192:**
- Middle cells: 67,092,480 out of 67,108,864 (99.976%)
- Modulos eliminated: 537 billion over 1000 generations
- Savings: ~10 seconds (27% speedup)

### 4. Parallelisation

Process different rows on different CPU cores:

```
Thread 0: rows [0,    rows/8)
Thread 1: rows [rows/8, 2*rows/8)
...
Thread 7: rows [7*rows/8, rows)
```

**Benefits:**
- Near-linear scaling with cores (8 cores → 7-8× speedup)
- No data races (reading old grid, writing new grid)
- Essential for large grids (>4096×4096)

### 5. SIMD Vectorisation

Process multiple cells per instruction using SIMD:
- AVX2: 32 bytes → 32 cells at once
- AVX-512: 64 bytes → 64 cells at once

**Theoretical speedup:** 32-64× (in practice: 8-16× due to overhead)

---

## Memory Access Patterns

### Row-Major Order

Cells are stored sequentially by row:
```
[0,0] [0,1] [0,2] ... [0,cols-1] [1,0] [1,1] ... [rows-1,cols-1]
```

**Advantages:**
- Sequential access along rows → good cache locality
- Predictable prefetching by CPU

### Cache Behaviour

**Small grids (< 1 MB):**
- Entire grid fits in L2/L3 cache
- Nearly all accesses hit cache
- Access time: ~10 cycles

**Large grids (> 16 MB):**
- Grid exceeds cache capacity
- Frequent cache misses to main memory
- Access time: ~100-300 cycles
- **Performance dominated by memory bandwidth**

### Memory Bandwidth Requirements

For a 512×512 grid (256 KB):
- Reads per generation: 8 × 262,144 = 2,097,152 (2 MB)
- Writes per generation: 262,144 (256 KB)
- **Total: ~2.3 MB per generation**

For 1000 generations: **2.3 GB of memory traffic**

---

## Mathematical Properties

### Totalistic Rule

Game of Life is a **totalistic cellular automaton**: the next state depends only on:
1. The current state of the cell
2. The **total count** of alive neighbours

It does NOT depend on:
- Which specific neighbours are alive
- The arrangement/pattern of neighbours
- History beyond the previous generation

### Determinism

The Game of Life is **completely deterministic**:
- Given initial state S₀, the state at generation t (Sₜ) is uniquely determined
- `Sₜ = F(Sₜ₋₁) = Fᵗ(S₀)`
- No randomness or external input

### Reversibility

Game of Life is **NOT reversible**:
- Multiple different states can lead to the same next state
- Cannot uniquely determine Sₜ₋₁ from Sₜ
- Information is lost in each generation

**Example:** These two states both evolve to empty grid:
```
State A:  • •      State B:  • •
          • •                 
```

### State Space

For an m×n grid:
- Total possible states: 2^(m×n)
- For 512×512: 2^262,144 ≈ 10^78,913 states (vastly larger than atoms in universe!)
- Guaranteed to eventually cycle or stabilise (finite state space)

---

## Pattern Classes

### Still Lifes (Period 1)

Patterns that don't change:

**Block (2×2):**
```
••
••
```

**Beehive (3×4):**
```
 •• 
•  •
 •• 
```

### Oscillators (Period n)

Patterns that repeat every n generations:

**Blinker (Period 2):**
```
Generation 0:   •••
Generation 1:    •
                 •
                 •
```

**Toad (Period 2):**
```
Generation 0:   •••
                •••
```

### Spaceships

Patterns that translate across the grid:

**Glider (Period 4, moves diagonally):**
```
 • 
  •
•••
```

After 4 generations, returns to original shape but moved 1 cell diagonally.

### Methuselahs

Small patterns that take many generations to stabilise:

**R-pentomino (5 cells → stabilises after 1103 generations)**

---

## Computational Complexity

### Problem Class

- **Decision problem:** "Does cell [i,j] become alive at generation t?"
- **Complexity class:** **PSPACE-complete** (proven by Conway)
- This means Game of Life can simulate any computer program

### Universal Computation

Game of Life is **Turing complete**:
- Can simulate logic gates (AND, OR, NOT)
- Can build arbitrary circuits
- Can implement any algorithm
- Has been used to build:
  - A working calculator
  - A complete computer (Turing machine)
  - A copy of Game of Life inside Game of Life!

### Practical Limits

For our implementation:
- **Tractable:** Grids up to 10,000 × 10,000 for 1000s of generations
- **Challenging:** Grids > 100,000 × 100,000 (10 billion cells)
- **Infeasible:** Grids > 1,000,000 × 1,000,000 (1 trillion cells) on single machine

---

## Summary

**Algorithm characteristics:**
- Simple rules, complex behaviour
- Deterministic but not reversible  
- Turing complete (can compute anything)
- PSPACE-complete complexity class

**Implementation keys:**
- Toroidal boundary conditions (wrap-around)
- Moore neighbourhood (8 neighbours)
- Bitwise trick for rule evaluation
- Flat memory layout for performance
- Parallelisation essential for large grids

**Performance scaling:**
- O(rows × cols) per generation
- Well-suited for parallelisation (no data dependencies)
- Memory bandwidth becomes bottleneck at large scales
- Edge optimisations save billions of operations

**Mathematical beauty:**
- Emergent complexity from simple rules
- Universal computation capability
- Rich pattern behaviour (still lifes, oscillators, spaceships)
- Active research area since 1970

---

## References

1. Gardner, M. (1970). "Mathematical Games: The fantastic combinations of John Conway's new solitaire game 'life'". *Scientific American*.

2. Conway, J.H. (1970). "The Game of Life". *University of Cambridge*.

3. Berlekamp, E.R., Conway, J.H., Guy, R.K. (1982). *Winning Ways for your Mathematical Plays, Volume 2*. Academic Press.

4. Rendell, P. (2016). "Turing Machine in Conway's Game of Life". Available at: http://rendell-attic.org/gol/tm.htm

5. LifeWiki: https://conwaylife.com/wiki/
