# Mojo GPU/CPU Memory Structures: A Guide

Understanding memory structures in Mojo: this guide explains the hierarchy from low-level pointers to high-level tensors, with clear distinctions between CPU and GPU memory (with some focus on Apple Silicon as I use a MacBoook).

---

## Table of Contents

1. [Memory Location: CPU vs GPU](#memory-location-cpu-vs-gpu)
2. [Memory Structure Hierarchy](#memory-structure-hierarchy)
3. [Decision Tree](#decision-tree)
4. [GPU Puzzles Alignment](#alignment-with-modular-gpu-puzzles)
5. [Common Patterns and Anti-Patterns](#common-patterns-and-anti-patterns)



---

## Memory Location: CPU vs GPU

### CPU Memory (Host)

- **Location**: System RAM
- **Access**: Fast for CPU, slow for GPU (requires copy)
- **Size**: Typically gigabytes (8-128 GB)
- **Allocation**: `alloc[T](size)`, `stack_allocation[]()`
- **Use**: Initial data preparation, result validation

### GPU Memory (Device)

#### 1. Global Memory

- **Location**: GPU DRAM (HBM/GDDR)
- **Access**: Accessible by all threads, relatively slow (~400 cycle latency)
- **Size**: Gigabytes (8-80 GB on modern GPUs)
- **Allocation**: `ctx.enqueue_create_buffer[DType](size)`
- **Use**: Primary data storage, kernel inputs/outputs

#### 2. Shared/Local Memory

- **Location**: On-chip SRAM
- **Access**: Fast (~10 cycle latency), shared within thread block
- **Size**: Limited (32-48 KB per block typically)
- **Allocation**: `stack_allocation[..., address_space=AddressSpace.SHARED]()`
- **Use**: Tiling, caching, inter-thread communication

#### 3. Registers

- **Location**: On-chip, per-thread
- **Access**: Fastest (~1 cycle latency), private to thread
- **Size**: Very limited (typically 64K 32-bit registers per SM)
- **Allocation**: Automatic (local variables)
- **Use**: Thread-local computation

### Apple Silicon Unified Memory

Apple Silicon (M1/M2/M3/M4/Max/Ultra) uses **unified memory**:
- CPU and GPU share the same physical RAM
- No explicit host-to-device copies needed (but still queued for synchronisation)
- Metal backend handles memory coherency
- Still benefits from shared memory tiling (on-chip cache)

---

## Memory Structure Hierarchy

### 1. `UnsafePointer` - Lowest Level

```mojo
from memory import UnsafePointer # NO LONGER necessary

var ptr = UnsafePointer[Float32].alloc(100)
ptr[42] = 3.14
ptr.free()  # MUST manually free!
```

**Characteristics:**

- Raw memory address (like C pointers)
- No bounds checking
- No shape or layout information
- Manual memory management (alloc/free)
- Fastest but most dangerous

**When to use:**

- C library interop
- Extreme performance critical paths
- When you need precise control
- **Rarely needed in GPU kernels**

**Dangers:**

- Memory leaks if you forget `free()`
- Buffer overruns (no bounds checking)
- Use-after-free bugs
- Null pointer dereferences

**Location:** Can point to either CPU or GPU memory

---

### 2. `NDBuffer` - Mid-Level (Strided Arrays)

```mojo
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from utils.index import Index

var buf = NDBuffer[
    DType.float32,       # Element type
    2,                   # Rank (number of dimensions)
    MutAnyOrigin,       # Origin (mutable/immutable)
    DimList(128, 64)    # Shape (compile-time or runtime)
](ptr, Index(128, 64))   # Runtime shape

# Access
var value = buf[10, 5]   # Bounds-checked in debug mode
```

**Characteristics:**

- Wraps `UnsafePointer` with metadata
- Multi-dimensional indexing
- Stride information for non-contiguous layouts
- Shape can be dynamic (runtime)
- Bounds checking in debug builds

**When to use:**

- CPU-side tensor operations
- Dynamic shapes (not known at compile time)
- Working with Modular's production kernels
- Need flexible stride patterns
- Interfacing with linalg functions

**Trade-offs:**

- More overhead than raw pointers
- Runtime shape information
- Less compiler optimisation than `LayoutTensor`

**Location:** Can wrap either CPU or GPU memory (via `UnsafePointer`)

**Used in:** Production MAX kernels, linalg operations

---

### 3. `LayoutTensor` - High-Level (Structured Tensors)

```mojo
from layout import Layout, LayoutTensor

# Define layout at compile time
comptime matrix_layout = Layout.row_major(128, 64)

# Create tensor with this layout
var tensor = LayoutTensor[
    DType.float32,
    matrix_layout,       # Compile-time layout
    MutAnyOrigin
](buffer)  # buffer is a DeviceBuffer or UnsafePointer

# Clean access
var value = tensor[10, 5]
```

**Characteristics:**

- Layout defined at **compile time**
- Automatic index calculations
- Type-safe multi-dimensional access
- Compiler can optimise based on layout
- Supports custom memory layouts

**When to use:**

- **GPU kernels** (most common!)
- Shape known at compile time
- Want compiler optimisations
- Teaching/learning GPU programming
- Custom memory layouts (row-major, column-major, tiled)

**Benefits:**

- Compiler optimises index arithmetic
- Layout information helps memory coalescing
- Cleaner syntax than NDBuffer
- Type safety

**Location:** Can wrap either CPU or GPU memory

**Used in:** Educational tutorials, custom kernels, Modular internal kernels

---

### 4. `LayoutTensor` with Shared Memory

```mojo
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

# Allocate in GPU shared memory (fast, block-local)
comptime tile_layout = Layout.row_major(16, 16)

var shared_tile = LayoutTensor[
    DType.float32,
    tile_layout,
    MutAnyOrigin,
    address_space = AddressSpace.SHARED  # Key: SHARED memory!
].stack_allocation()
```

**Characteristics:**

- Allocated in GPU shared/local memory (SRAM)
- Shared across threads in a block
- Much faster than global memory (~100× on NVIDIA, ~10× on Metal)
- **Automatically freed** when scope ends
- Must be compile-time sized

**When to use:**

- **Tiling in GPU kernels**
- Temporary data shared within block
- Cache-like optimisations
- Reduction operations
- Data reuse patterns

**Limits:**

- Very limited size (32-48 KB per block typically)
- Must fit in shared memory budget
- Compile-time size only
- Only accessible within thread block

**Location:** GPU shared/local memory (on-chip SRAM)

**Used in:** All tutorials with tiling (matmul, reduction, softmax)

---

### 5. `DeviceBuffer` (Opaque Handle)

```mojo
from gpu.host import DeviceContext

with DeviceContext() as ctx:
    # Returns DeviceBuffer (opaque handle)
    var device_buf = ctx.enqueue_create_buffer[DType.float32](1024)
    
    # Wrap in LayoutTensor for kernel use
    comptime layout = Layout.row_major(32, 32)
    var tensor = LayoutTensor[DType.float32, layout](device_buf)
    
    # Or initialize on host
    with device_buf.map_to_host() as host_buf:
        var host_tensor = LayoutTensor[DType.float32, layout](host_buf)
        # Fill with data...
```

**Characteristics:**

- Opaque handle to GPU memory
- Managed by `DeviceContext`
- Can be mapped to host for initialisation
- Must wrap in `LayoutTensor`/`NDBuffer` for use

**When to use:**

- Allocating GPU memory from host
- Transferring data between CPU and GPU
- Kernel parameter passing

**Location:** GPU global memory

**Used in:** All GPU tutorials for device memory allocation

---

## Decision Tree

```
┌─────────────────────────────────────────┐
│ Are you writing a GPU kernel?          │
└─────────────┬───────────────────────────┘
              │
        ┌─────┴─────┐
        │           │
       YES         NO (CPU/Host code)
        │           │
        │           └──> Shape known at compile time?
        │                    ├─ YES → LayoutTensor (if in Modular repo)
        │                    ├─ NO → NDBuffer
        │                    └─ Raw perf/C interop → UnsafePointer
        │
        └──> GPU Kernel Memory Choice:
             │
             ├─ Input/Output matrices?
             │  └─> LayoutTensor[..., AddressSpace.GENERIC] (default)
             │      • Passed as kernel parameters
             │      • Lives in GPU global memory
             │      • Accessible by all threads
             │
             ├─ Shared tile/cache for block?
             │  └─> LayoutTensor[..., AddressSpace.SHARED].stack_allocation()
             │      • Created inside kernel
             │      • Fast block-local memory
             │      • Shared by threads in block
             │      • Requires barrier() synchronisation
             │
             └─ Thread-local temporary?
                └─> Local variables (registers)
                    • Fastest
                    • Private to each thread
                    • Automatic allocation
```

---

## Alignment with Modular GPU Puzzles

The [Mojo GPU Puzzles](https://puzzles.modular.com) uniquely challenge the status quo approach by first building understanding with low-level memory manipulation, then gradually transitioning to Mojo's `LayoutTensor` abstractions.

### Puzzle Progression

**Early Puzzles (1-3): Raw Memory**
```mojo
// Puzzle 1: Raw pointers
fn add_10(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    i = thread_idx.x
    output[i] = a[i] + 10.0
```
- **Memory**: UnsafePointer
- **Focus**: Basic thread indexing, parallel operations
- **Why**: Building understanding with low-level memory manipulation

**Mid Puzzles (4-10): LayoutTensor**

```mojo
// Puzzle 4: LayoutTensor introduction
fn add_10_2d(
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    a: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    size: UInt,
):
    row = thread_idx.y
    col = thread_idx.x
    if col < size and row < size:  // Bounds checking!
        output[row, col] = a[row, col] + 10.0
```
- **Memory**: LayoutTensor for structured access
- **Focus**: 2D indexing, bounds checking, memory safety
- **Why**: Transition to safer, more maintainable abstractions

**Advanced Puzzles (13, 16, 28): Shared Memory**

```mojo
// Puzzle 13: Convolution with shared memory
fn conv_1d_simple[...](
    output: LayoutTensor[mut=True, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    b: LayoutTensor[mut=False, dtype, conv_layout],
):
    // SHARED MEMORY for caching
    shared_a = LayoutTensor[
        dtype, Layout.row_major(SIZE), MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    
    shared_b = LayoutTensor[
        dtype, Layout.row_major(CONV), MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    
    // Load to shared, sync, compute
    if global_i < SIZE:
        shared_a[local_i] = a[global_i]
    barrier()
    
    // Compute using shared memory (fast!)
```
- **Memory**: LayoutTensor + AddressSpace.SHARED
- **Focus**: Data reuse, sliding windows, halo regions
- **Why**: Performance optimisation through memory hierarchy

**Expert Puzzles (28): Async Copy**

```mojo
// Puzzle 28: Async memory operations
copy_dram_to_sram_async()  // Launch background transfer
load_small_data()          // Useful work while waiting
async_copy_wait_all()      // Synchronize before using
```
- **Memory**: Async DMA between DRAM and SRAM
- **Focus**: Latency hiding capabilities - the key to high-performance memory-bound algorithms
- **Why**: Overlap memory transfers with computation

---

x## Common Patterns and Anti-Patterns

### ✅ Good Patterns

#### Pattern 1: Tiled Computation with Shared Memory

```mojo
fn tiled_operation[TILE_SIZE: Int](...):
    // Allocate shared memory once
    comptime tile_layout = Layout.row_major(TILE_SIZE, TILE_SIZE)
    var shared_tile = LayoutTensor[
        DType.float32, tile_layout, MutAnyOrigin,
        address_space = AddressSpace.SHARED
    ].stack_allocation()
    
    // Load tile from global → shared
    shared_tile[local_y, local_x] = global_data[global_y, global_x]
    barrier()  // Ensure all threads loaded
    
    // Compute using shared memory
    var result = compute_from_shared(shared_tile)
    barrier()  // Before loading next tile
    
    // Write result to global
    output[global_y, global_x] = result
```

**Why good:**

- Minimises slow global memory access
- Maximises fast shared memory reuse
- Proper synchronisation prevents race conditions

#### Pattern 2: Grid-Stride Loop for Large Data

```mojo
fn process_large_array(...):
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(gridDim.x * blockDim.x)
    
    // Each thread processes multiple elements
    while idx < size:
        output[idx] = process(input[idx])
        idx += stride
```

**Why good:**

- Handles arrays larger than grid size
- Better load balancing
- Kernel reusability

#### Pattern 3: Host Memory Initialisation

```mojo
with DeviceContext() as ctx:
    var device_buf = ctx.enqueue_create_buffer[DType.float32](size)
    
    // Map to host for initialisation (efficient!)
    with device_buf.map_to_host() as host_buf:
        var tensor = LayoutTensor[DType.float32, layout](host_buf)
        for i in range(size):
            tensor[i] = Float32(i)
    
    // device_buf now contains initialised data
```

**Why good:**

- Efficient initialisation on CPU
- No explicit copy needed (unified memory on Apple Silicon)
- Clear separation of init and compute

---

### ❌ Anti-Patterns

#### Anti-Pattern 1: Missing Barrier

```mojo
// ❌ BAD: Missing barrier after loading shared memory
fn buggy_reduction(...):
    var shared = stack_allocation[256, Float32, address_space=AddressSpace.SHARED]()
    shared[tid] = input[global_id]
    // Missing: barrier()
    
    // RACE CONDITION: Some threads may read before others write!
    if tid < 128:
        shared[tid] += shared[tid + 128]
```

**Why bad:**

- Race condition: undefined behavior
- May work sometimes, fail others
- A GPU program can produce "correct" results while simultaneously performing illegal memory accesses

**Fix:**

```mojo
shared[tid] = input[global_id]
barrier()  // ✅ Ensure all writes complete before reads
```

#### Anti-Pattern 2: Unbounded Shared Memory

```mojo
// ❌ BAD: Shared memory size from runtime parameter
fn bad_kernel(tile_size: Int):
    var shared = stack_allocation[
        tile_size,  // ❌ Runtime value!
        Float32,
        address_space=AddressSpace.SHARED
    ]()
```

**Why bad:**

- Shared memory must be compile-time sized
- Will not compile
- Can exceed hardware limits

**Fix:**

```mojo
fn good_kernel[TILE_SIZE: Int]():  // ✅ Compile-time parameter
    var shared = stack_allocation[
        TILE_SIZE,  // ✅ Compile-time constant
        Float32,
        address_space=AddressSpace.SHARED
    ]()
```

#### Anti-Pattern 3: No Bounds Checking

```mojo
// ❌ BAD: No bounds check
fn add_10_2d(output: LayoutTensor[...], a: LayoutTensor[...], size: UInt):
    row = thread_idx.y
    col = thread_idx.x
    output[row, col] = a[row, col] + 10.0  // ❌ May be out of bounds!
```

**Why bad:**

- Out-of-bounds memory access is a classic example of undefined behaviour
- Can corrupt memory, crash, or appear to work
- Silent data corruption

**Fix:**

```mojo
// ✅ GOOD: Bounds checking
fn add_10_2d(...):
    row = thread_idx.y
    col = thread_idx.x
    if col < size and row < size:  // ✅ Validate indices
        output[row, col] = a[row, col] + 10.0
```

#### Anti-Pattern 4: Memory Leaks

```mojo
// ❌ BAD: Forgot to free
fn process_data(size: Int):
    var data = alloc[Float32](size)
    // ... use data ...
    // Missing: data.free()
```

**Why bad:**

- Memory leak on CPU
- Will exhaust memory if called repeatedly

**Fix:**

```mojo
// ✅ GOOD: Always free
fn process_data(size: Int):
    var data = alloc[Float32](size)
    // ... use data ...
    data.free()  // ✅ Clean up
```

Or better yet, use RAII patterns where possible:

```mojo
with DeviceContext() as ctx:
    var device_buf = ctx.enqueue_create_buffer[DType.float32](size)
    // ... use device_buf ...
    // Automatically cleaned up when context exits
```

---

## Quick Reference Table

| Memory Type | Level | CPU/GPU | Size | Speed | When to Use |
|-------------|-------|---------|------|-------|-------------|
| **`UnsafePointer`** | Lowest | Both | Any | Fast* | C interop, extreme perf (avoid if possible) |
| **`alloc[T](n)`** | Low | CPU | Any | Fast | CPU host buffers, temporary data |
| **`NDBuffer`** | Mid | Both | Any | Medium | Dynamic shapes, CPU tensors |
| **`LayoutTensor` (Global)** | High | Both | Any | Medium | GPU kernels, compile-time layouts |
| **`LayoutTensor` (SHARED)** | High | GPU | ~48KB | Very Fast | Tiles, block-local cache |
| **`DeviceBuffer`** | Opaque | GPU | Any | N/A | GPU memory handles (wrap in LayoutTensor) |
| **Registers** | Automatic | GPU | ~256KB/SM | Fastest | Thread-local variables (automatic) |

*\*Speed depends on memory location (CPU RAM vs GPU RAM)*

---

## Summary: The Golden Rules

1. **For GPU kernels**: Use **`LayoutTensor`** (default) or **`LayoutTensor + AddressSpace.SHARED`** (for tiling)

2. **For CPU host code**: Use **`alloc[]`** for buffers or **`LayoutTensor`**

3. **For device memory**: Allocate with **`DeviceContext`**, wrap in **`LayoutTensor`**

4. **Shared memory**: Always use **`AddressSpace.SHARED`**, always use **`barrier()`**, always **compile-time sized**

5. **Avoid `UnsafePointer`** unless you have a specific reason (C interop, proven performance bottleneck)

6. **Always bounds check** in GPU kernels (avoid undefined behaviour)

7. **Always free** CPU memory (or use RAII patterns)

8. **Profile first**, optimise second (don't prematurely optimise)

---

## Further Reading

- **Mojo Documentation**: https://docs.modular.com/
- **GPU Puzzles**: https://puzzles.modular.com/
- **Production Kernels**: `/max/kernels/src/` for reference implementations

---

**Remember**: In GPU programming, data movement is often more expensive than computation: This inverts another common assumption in programming: computation is no longer the bottleneck: data movement is. Choose your memory structures wisely!
