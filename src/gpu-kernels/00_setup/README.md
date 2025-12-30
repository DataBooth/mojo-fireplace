# Setup: Hardware Detection

Before diving into GPU programming, it's useful to understand what hardware you're running on.

## Files

- **`check_gpu.mojo`** - Comprehensive hardware detection utility
  - Platform information (macOS, Linux, Windows)
  - CPU architecture and core count
  - GPU accelerator detection (NVIDIA, AMD, Apple Silicon)
  - Apple Silicon specific details (M1-M5)
  - Current capabilities and limitations

## Running

```bash
pixi run mojo ../max/kernels/anatomy/00_setup/check_gpu.mojo
```

## Output

The tool provides detailed information about:

1. **Platform Information**
   - Operating system
   - CPU architecture
   - Physical and logical core count
   - Apple Silicon generation (if applicable)

2. **Accelerator Status**
   - Whether any GPU/accelerator is detected
   - Specific GPU type (NVIDIA CUDA, AMD ROCm, Apple Metal)

3. **Apple Silicon GPU Details** (if detected)
   - Metal Shading Language support
   - Supported M-series chips (M1-M5)
   - Requirements (macOS 15+, Xcode 16+)
   - Current capabilities:
     * GPU functions (vector ops, matrix ops)
     * Mojo GPU puzzles compatibility
     * MAX graphs status
     * AI model serving status
   - Performance characteristics:
     * Unified memory architecture
     * Zero-copy CPU-GPU data sharing
     * Memory capacity (shared with system)

## Example Output (Apple M1)

```
=== Hardware Accelerator Detection ===

Platform Information:
  - Operating System: macOS
  - CPU Architecture: apple-m1
  - Physical Cores: 8
  - Logical Cores: 8
  - Apple Silicon: M1 series

Accelerator Status:
  - Has any accelerator: True
  - NVIDIA GPU (CUDA): False
  - AMD GPU (ROCm): False
  - Apple Silicon GPU (Metal): True

✓ APPLE SILICON GPU DETECTED
  Acceleration: Metal Shading Language (MSL)
  Supported chips: M1, M2, M3, M4, M5 series
  Requirements: macOS 15+, Xcode 16+
  Features: Unified memory architecture, Neural Engine integration

  Current capabilities:
    ✓ GPU functions (vector ops, basic matrix ops)
    ✓ Most Mojo GPU puzzles (1-15 and many higher)
    ✓ Basic MAX graphs with custom ops
    ⚠ Full AI model serving: In development

  Performance notes:
    - Unified memory allows zero-copy CPU-GPU data sharing
    - Best for: Local development, prototyping, small-to-medium models
    - Memory: Shared with system (8GB-192GB depending on Mac model)
```

## Why This Matters

- **Confirms GPU availability**: Know if you'll get GPU acceleration or CPU fallback
- **Understands capabilities**: Different GPUs have different features
- **Sets expectations**: Apple Silicon support is newer, some features still in development
- **Platform-specific details**: Unified memory on Apple Silicon vs discrete GPUs

## Key Concepts

### Unified Memory (Apple Silicon)
- CPU and GPU share the same physical memory
- No expensive data copies between CPU and GPU
- Simpler programming model
- Trade-off: Memory shared with system (no dedicated VRAM)

### Metal Backend
- Apple's GPU programming framework
- Used by Mojo for Apple Silicon GPU support
- Compiles Mojo kernels to Metal Shading Language (MSL)
- Mature, optimised by Apple for their hardware

### Current Limitations (Apple Silicon)
- Full MAX graph support: In progress
- AI model serving: In development  
- Some advanced GPU features: Being implemented

These will improve with nightly Mojo releases!
