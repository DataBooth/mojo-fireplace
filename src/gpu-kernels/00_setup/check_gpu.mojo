"""Hardware Detection Utility

This script checks what hardware accelerators are available on the system.
Useful for understanding what GPU/accelerator capabilities are present before
running MAX kernels.

Supported accelerators:
- NVIDIA GPUs (CUDA)
- AMD GPUs (ROCm/HIP)
- Apple Silicon GPUs (Metal) - M1 through M5 series on macOS 15+
"""

from sys import (
    has_accelerator,
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    CompilationTarget,
    num_logical_cores,
    num_physical_cores,
)


fn main():
    """Check and display available hardware accelerators with platform details."""
    print("=== Hardware Accelerator Detection ===")
    print()
    
    # Platform detection
    print("Platform Information:")
    if CompilationTarget.is_macos():
        print("  - Operating System: macOS")
    elif CompilationTarget.is_linux():
        print("  - Operating System: Linux")
    else:
        print("  - Operating System: Windows/Other")
    
    var cpu = CompilationTarget._arch()
    print("  - CPU Architecture:", cpu)
    print("  - Physical Cores:", num_physical_cores())
    print("  - Logical Cores:", num_logical_cores())
    
    # Check for Apple Silicon specifics
    if CompilationTarget.is_apple_silicon():
        if CompilationTarget.is_apple_m1():
            print("  - Apple Silicon: M1 series")
        elif CompilationTarget.is_apple_m2():
            print("  - Apple Silicon: M2 series")
        elif CompilationTarget.is_apple_m3():
            print("  - Apple Silicon: M3 series")
        elif CompilationTarget.is_apple_m4():
            print("  - Apple Silicon: M4 series")
        else:
            print("  - Apple Silicon: Detected (M5 or newer)")
    print()
    
    # Accelerator detection
    var has_any = has_accelerator()
    print("Accelerator Status:")
    print("  - Has any accelerator:", has_any)
    
    if has_any:
        var has_nvidia = has_nvidia_gpu_accelerator()
        var has_amd = has_amd_gpu_accelerator()
        var has_apple = has_apple_gpu_accelerator()
        
        print("  - NVIDIA GPU (CUDA):", has_nvidia)
        print("  - AMD GPU (ROCm):", has_amd)
        print("  - Apple Silicon GPU (Metal):", has_apple)
        print()
        
        if has_nvidia:
            print("✓ NVIDIA GPU DETECTED")
            print("  Acceleration: CUDA kernels")
            print("  Suitable for: Deep learning, HPC, general GPU compute")
        elif has_amd:
            print("✓ AMD GPU DETECTED")
            print("  Acceleration: ROCm/HIP kernels")
            print("  Suitable for: Deep learning, HPC, general GPU compute")
        elif has_apple:
            print("✓ APPLE SILICON GPU DETECTED")
            print("  Acceleration: Metal Shading Language (MSL)")
            print("  Supported chips: M1, M2, M3, M4, M5 series")
            print("  Requirements: macOS 15+, Xcode 16+")
            print("  Features: Unified memory architecture, Neural Engine integration")
            print()
            print("  Current capabilities:")
            print("    ✓ GPU functions (vector ops, basic matrix ops)")
            print("    ✓ Most Mojo GPU puzzles (1-15 and many higher)")
            print("    ✓ Basic MAX graphs with custom ops")
            print("    ⚠ Full AI model serving: In development")
            print()
            print("  Performance notes:")
            print("    - Unified memory allows zero-copy CPU-GPU data sharing")
            print("    - Best for: Local development, prototyping, small-to-medium models")
            print("    - Memory: Shared with system (8GB-192GB depending on Mac model)")
        else:
            print("✓ OTHER ACCELERATOR DETECTED")
            print("  An accelerator is present but type could not be determined")
    else:
        print()
        print("⚠ NO GPU ACCELERATOR DETECTED")
        print("  Execution will fall back to CPU")
        print("  CPU execution is still functional but will be slower for large workloads")
    
    print()
