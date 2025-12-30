# Tutorial 05: Softmax

Real-world application combining reductions and elementwise operations. Fundamental to transformers and LLMs.

## Files

- **`simple.mojo`** - Educational implementation
  - Three-pass algorithm (max → exp+sum → normalize)
  - Two-pass algorithm with grid-stride loop (max+sum → normalize)
  - Numerical stability techniques
  - Validates both implementations

## What is Softmax?

Softmax converts a vector of numbers into a probability distribution:

```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

**Properties:**
- All outputs in range [0, 1]
- All outputs sum to exactly 1.0
- Larger inputs get higher probabilities
- Differentiable (important for training)

## Why It Matters

Softmax is used in:
- **Attention mechanisms**: Core of transformer models (GPT, BERT, LLaMA)
- **Classification**: Final layer of neural networks
- **Policy gradients**: Reinforcement learning
- **Temperature scaling**: Controlling output sharpness

## Concepts Covered

### 1. Numerical Stability

**Naive approach** (WRONG):
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```
Problem: `exp(x)` overflows for large x!

**Stable approach** (CORRECT):
```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```
Solution: Subtracting max prevents overflow, mathematically equivalent

### 2. Multi-Pass Algorithms

**Three-Pass** (Clearest):
1. Find maximum value
2. Compute exp(x - max) and sum
3. Divide by sum to normalize

**Two-Pass** (More Efficient):
1. Find max AND compute sum of exp in one kernel (using grid-stride loop)
2. Normalize

**One-Pass** (Most Efficient, Advanced):
- Online algorithm using running statistics
- Single kernel pass
- Complex implementation (see production kernels)

### 3. Grid-Stride Loop Pattern

Enables a single thread block to process arrays larger than block size:

```mojo
var idx = global_id
while idx < size:
    # Process element at idx
    idx += grid_stride  # Move to next element this thread handles
```

Benefits:
- Single block = single shared memory space
- Simpler synchronization
- Good for reductions over entire array

### 4. Combining Patterns

Softmax = **Reduction** (max, sum) + **Elementwise** (exp, divide)

Reuses concepts from:
- Tutorial 03: Reduction patterns, shared memory, barriers
- Tutorial 02: Elementwise operations (exp, division)

## Running

```bash
pixi run mojo ../max/kernels/anatomy/05_softmax/simple.mojo
```

## Expected Output

- Three-pass softmax on 1,000 elements
- Two-pass softmax on 10,000 elements
- Both validate to sum ≈ 1.0
- Shows intermediate values (max, sum of exp)

## Performance Characteristics

### Memory Access Patterns

- **Three-pass**: 3 full array reads from global memory
- **Two-pass**: 2 full array reads from global memory
- **Bottleneck**: Memory bandwidth, not compute

### Optimization Opportunities

1. **Fuse passes**: Reduce memory traffic
2. **Vectorize**: SIMD loads/stores
3. **Warp reductions**: Use `warp.sum()` for final steps
4. **Online algorithm**: Single-pass implementation

## Real-World Usage

### In Attention Mechanisms

```
scores = Q @ K.T / sqrt(d_k)  # Query-key similarities
weights = softmax(scores)      # ← Softmax here!
output = weights @ V            # Weighted values
```

### FlashAttention

Modern implementations fuse softmax with matrix multiplication:
- Avoids materializing full attention matrix
- Dramatically reduces memory usage
- Enables longer context lengths

See `/max/kernels/src/nn/mha.mojo` for production implementations.

## Common Issues

1. **Numerical instability**: Always subtract max before exp()
2. **Sum not exactly 1.0**: Floating-point rounding (sum ≈ 1.0 is OK)
3. **Grid-stride complexity**: Single block limits parallelism for huge arrays
4. **Memory bandwidth**: Main bottleneck, not compute time

## Extensions

Try implementing:
- Log-softmax (numerically stable log(softmax(x)))
- Masked softmax (for causal attention)
- Grouped softmax (batch processing)
- Temperature-scaled softmax (softmax(x/T))

## Production Kernels

For production-quality implementations:
- `/max/kernels/src/nn/softmax.mojo` - Optimized softmax
- `/max/kernels/src/nn/mha.mojo` - Fused attention with softmax
- `/mojo/stdlib/stdlib/algorithm/_gpu/` - GPU primitives

## Key Takeaways

1. Softmax is fundamental to modern AI (transformers, attention)
2. Numerical stability requires care (subtract max)
3. Grid-stride loop is essential pattern for GPU programming
4. Real kernels combine multiple operations (reduction + elementwise)
5. Memory bandwidth is the primary bottleneck
6. Production implementations use advanced fusion techniques
