# Apple Silicon GPU Setup Guide

This guide will help you set up and test GPU-accelerated Game of Life on your M1 MacBook Pro.

## Requirements

Based on the Modular forum post, you need:

1. ✅ **Apple Silicon Mac** (M1/M2/M3/M4) - You have M1 Pro
2. ❓ **macOS 15+ (Sequoia)** - Need to check
3. ❓ **Xcode 16+** - Need to check
4. ❓ **Mojo nightly build** - Need to check
5. ❓ **Metal toolchain** - Need to install

## Step 1: Check Your System

### Check macOS Version

```bash
sw_vers
```

You need **macOS 15.0 (Sequoia) or newer**. If you're on macOS 14 or earlier, you'll need to upgrade.

### Check Xcode Version

```bash
xcodebuild -version
```

You need **Xcode 16.0 or newer**. 

If you don't have Xcode 16:
```bash
# Download from App Store or developer.apple.com
# Then accept the license:
sudo xcodebuild -license accept
```

### Check Xcode Command Line Tools

```bash
xcode-select -p
```

Should output: `/Applications/Xcode.app/Contents/Developer`

If different, fix it:
```bash
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
```

## Step 2: Install Metal Toolchain

Check if Metal toolchain is available:

```bash
xcrun -sdk macosx metal
```

**Expected output:** `metal: error: no input files`

**If you see:** `error: cannot execute tool 'metal' due to missing Metal Toolchain`

**Then install it:**
```bash
xcodebuild -downloadComponent MetalToolchain
```

This may take a while (several GB download).

## Step 3: Verify Mojo Version

```bash
mojo --version
```

You need a **nightly build from September 2025 or later** for Apple Silicon GPU support.

If you're on an older version:
```bash
# Update to nightly
modular update mojo
```

Or follow: https://docs.modular.com/mojo/manual/get-started/

## Step 4: Test GPU Examples

The Modular repository includes GPU examples. Let's test one:

```bash
# Navigate to examples
cd /Users/mjboothaus/code/github/databooth/mojo/modular/examples/mojo/gpu-functions

# Try vector addition example
mojo vector_addition.mojo
```

**Expected output:**
```
Found GPU: Apple M1 Pro
Resulting vector: 3.75 3.75 3.75 3.75 ...
```

**If you see errors**, check the troubleshooting section below.

## Step 5: Test Game of Life GPU Version

Once GPU examples work, try our GPU implementation:

```bash
cd /Users/mjboothaus/code/github/databooth/mojo/modular/my_mojo_experiments

# Run GPU version
mojo run gridv6_gpu.mojo
```

**Expected output:**
```
=== Mojo GPU-Accelerated Game of Life ===
Grid: 256×256
Generations: 100

Found GPU: Apple M1 Pro
Using GPU acceleration
GPU time: 0.XXXXX s
Per generation: X.XXX ms

Final grid fingerprint: ...
```

## Troubleshooting

### Error: "sh -c 'xcodebuild -sdk ... -find metallib' failed"

This means the Metal toolchain isn't properly installed.

**Solution:**
1. Check Xcode path: `xcode-select -p`
2. Fix if needed: `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer`
3. Install Metal: `xcodebuild -downloadComponent MetalToolchain`

### Error: "cannot execute tool 'metal'"

**Solution:**
```bash
# Install Metal toolchain
xcodebuild -downloadComponent MetalToolchain

# Verify
xcrun -sdk macosx metal
# Should output: "metal: error: no input files"
```

### Error: "incompatible bitcode versions"

You're running macOS 14 or earlier, or Xcode 15 or earlier.

**Solution:** Upgrade to macOS 15 and Xcode 16.

### Error: "No GPU detected"

The GPU API might not be available in your Mojo version.

**Solution:**
1. Check Mojo version: `mojo --version`
2. Update if needed: `modular update mojo`
3. Make sure you have the **nightly build**, not stable

### GPU Example Works, But Game of Life Doesn't

The GPU API in Mojo is still experimental. Some features might not work yet.

**Fallback:** The code includes CPU fallback, so it should still run (just slower).

## Expected Performance

Based on your current results:

| Version | Platform | Time (512×512, 1000 gens) |
|---------|----------|---------------------------|
| NumPy | CPU (Accelerate) | 0.29s |
| Mojo v3 | CPU (parallel) | 0.44s |
| **Mojo v6 | GPU (Metal) | **0.02-0.05s** ← Target! |

GPU should be **6-20× faster than NumPy** for large grids!

## Performance Notes

### GPU Speedup Factors

- **Small grids (< 256×256)**: GPU overhead dominates, may be slower
- **Medium grids (512×512)**: GPU starts to shine, 5-10× faster
- **Large grids (1024×1024+)**: GPU excels, 10-50× faster

### Why GPU is Fast

1. **Massive parallelism**: All 262,144 cells computed simultaneously
2. **No CPU-GPU sync**: Data stays on GPU between generations
3. **Memory bandwidth**: M1 unified memory = fast transfers
4. **Hardware optimization**: Metal shaders highly optimized

### Why CPU Held Up So Far

Your M1 Pro has:
- **Apple Accelerate** framework (NumPy uses this)
- **10 CPU cores** (8 performance + 2 efficiency)
- **Unified memory** (fast CPU-RAM access)

This is why NumPy was so competitive!

## Next Steps

### If GPU Works

🎉 Congratulations! You're running Game of Life on the GPU!

Try scaling up:
```bash
# Edit gridv6_gpu.mojo, change:
alias test_rows = 1024
alias test_cols = 1024
alias test_gens = 1000

# Rerun
mojo run gridv6_gpu.mojo
```

### If GPU Doesn't Work Yet

The API is very new (September 2025). Some features might not be implemented:

1. ✅ Keep the PyTorch Metal version (`gridv6_metal.py`) as backup
2. ✅ Use the CPU optimized versions (v4, v5) - still very fast!
3. ✅ Watch for Mojo updates: https://forum.modular.com/c/gpu-programming/

### Contribute Back

If you get it working:
1. Share results on Modular forum
2. Help improve the GPU examples
3. Write about your optimization journey!

## Quick Test Script

Save this to test your setup:

```bash
#!/bin/bash
echo "=== Apple Silicon GPU Setup Check ==="
echo ""

echo "1. macOS version:"
sw_vers | grep ProductVersion

echo ""
echo "2. Xcode version:"
xcodebuild -version | head -1

echo ""
echo "3. Xcode path:"
xcode-select -p

echo ""
echo "4. Metal toolchain:"
xcrun -sdk macosx metal 2>&1 | head -1

echo ""
echo "5. Mojo version:"
mojo --version 2>&1 | head -1

echo ""
echo "=== Status ==="
if [[ $(sw_vers -productVersion | cut -d. -f1) -ge 15 ]]; then
    echo "✅ macOS 15+"
else
    echo "❌ Need macOS 15+ (you have $(sw_vers -productVersion))"
fi

if xcrun -sdk macosx metal 2>&1 | grep -q "no input files"; then
    echo "✅ Metal toolchain installed"
else
    echo "❌ Metal toolchain missing"
fi

echo ""
echo "If all checks pass, try:"
echo "  cd /path/to/modular/examples/mojo/gpu-functions"
echo "  mojo vector_addition.mojo"
```

Save as `check_gpu_setup.sh`, make executable, and run:
```bash
chmod +x check_gpu_setup.sh
./check_gpu_setup.sh
```

## Summary

Apple Silicon GPU support in Mojo is **brand new** (September 2025). It's exciting but experimental!

**Best path forward:**
1. Check system requirements (macOS 15, Xcode 16)
2. Test official GPU examples first
3. Try `gridv6_gpu.mojo`
4. If issues, use CPU versions (v4, v5) which are already very fast
5. Stay tuned for Mojo updates!

Good luck! 🚀
