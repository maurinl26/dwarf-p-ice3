# Building GPU-Accelerated ICE_ADJUST

This guide explains how to build the GPU-accelerated version of ICE_ADJUST with OpenACC support.

## Prerequisites

### Required Software

1. **NVIDIA HPC SDK** (provides `nvfortran` compiler with OpenACC)
   - Version 23.1 or newer
   - Download: https://developer.nvidia.com/hpc-sdk

2. **CUDA Toolkit** (11.0 or newer)
   - Usually included with NVIDIA HPC SDK
   - Download: https://developer.nvidia.com/cuda-downloads

3. **CMake** (3.12 or newer)
   ```bash
   pip install cmake
   ```

4. **Python** (3.8 or newer) with development headers
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install python3-dev

   # On macOS
   brew install python
   ```

5. **CuPy** (CUDA-accelerated NumPy)
   ```bash
   # For CUDA 12.x
   pip install cupy-cuda12x

   # For CUDA 11.x
   pip install cupy-cuda11x
   ```

6. **Other Python dependencies**
   ```bash
   pip install numpy cython scikit-build-core
   ```

### Optional (for JAX integration)

```bash
pip install jax[cuda12_pip]
```

---

## Build Instructions

### Method 1: Using CMake Directly (Recommended for Development)

#### Step 1: Set Environment Variables

```bash
# Set Fortran compiler to nvfortran
export FC=nvfortran
export CC=gcc  # or clang on macOS
export CXX=g++  # or clang++ on macOS

# Add NVIDIA HPC SDK to PATH (if not already)
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin:$PATH
```

#### Step 2: Configure CMake with OpenACC

```bash
cd /Users/loicmaurin/PycharmProjects/dwarf-p-ice3

# Create build directory
mkdir -p build-gpu
cd build-gpu

# Configure with OpenACC enabled
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENACC=ON \
  -DCMAKE_Fortran_COMPILER=nvfortran

# Verify configuration shows "GPU ACCELERATION: ENABLED"
```

**Expected Output:**
```
====================================
PHYEX ICE_ADJUST Configuration
====================================
Fortran Compiler: /opt/nvidia/hpc_sdk/.../nvfortran
Compiler ID: NVHPC
Compiler Flags: -O2 -fPIC -cpp -acc -Minfo=accel -gpu=managed -Minline
Build Type: Release

GPU ACCELERATION: ENABLED
  OpenACC: YES
  GPU Fortran modules: .../condensation_acc.F90;.../ice_adjust_acc.F90
  GPU Cython wrapper: _phyex_wrapper_acc
  CuPy: /path/to/site-packages/cupy/_core/include
====================================
```

#### Step 3: Build

```bash
# Build the library and Python extensions
cmake --build . -j8

# Install to Python environment
cmake --install . --prefix ~/.local
```

#### Step 4: Verify Build

```bash
# Check that GPU wrapper was built
ls -lh _phyex_wrapper_acc*.so

# Expected output:
# _phyex_wrapper_acc.cpython-312-darwin.so  (macOS)
# _phyex_wrapper_acc.cpython-312-x86_64-linux-gnu.so  (Linux)
```

---

### Method 2: Using scikit-build-core (pip installable)

#### Step 1: Create pyproject.toml Configuration

Create or update `pyproject.toml`:

```toml
[build-system]
requires = ["scikit-build-core", "cython", "numpy"]
build-backend = "scikit_build_core.build"

[project]
name = "ice3-gpu"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20",
    "cupy-cuda12x>=12.0",
]

[tool.scikit-build]
cmake.build-type = "Release"
cmake.args = [
    "-DENABLE_OPENACC=ON",
]
```

#### Step 2: Build and Install

```bash
# Set compiler
export FC=nvfortran

# Build and install in editable mode (for development)
pip install -e . --no-build-isolation

# Or build wheel for distribution
pip install build
python -m build
```

---

## Build Configurations

### CPU-Only Build (Default)

```bash
cmake .. -DENABLE_OPENACC=OFF  # or omit flag
cmake --build .
```

**Result:**
- Builds `ice_adjust.F90` (CPU version)
- Builds `_phyex_wrapper.so` (CPU Cython wrapper)
- NO GPU acceleration

### GPU-Accelerated Build

```bash
cmake .. -DENABLE_OPENACC=ON -DCMAKE_Fortran_COMPILER=nvfortran
cmake --build .
```

**Result:**
- Builds `ice_adjust.F90` (CPU version) **AND** `ice_adjust_acc.F90` (GPU version)
- Builds `_phyex_wrapper.so` (CPU) **AND** `_phyex_wrapper_acc.so` (GPU)
- GPU acceleration available

---

## Compiler Flags

### NVIDIA HPC SDK Flags (when ENABLE_OPENACC=ON)

| Flag | Purpose |
|------|---------|
| `-acc` | Enable OpenACC directives |
| `-Minfo=accel` | Show accelerator compilation info |
| `-gpu=managed` | Use CUDA Unified Memory |
| `-Minline` | Inline device functions (critical for performance) |
| `-fast` | Aggressive optimization (Release build) |
| `-Mcuda` | Enable CUDA features (Release build) |

### Example Compiler Output

```
condensation_acc.F90:
    180, Generating present(PPABS(:,:),PT(:,:),...)
         Generating copyin(PSIGQSAT(:))
    185, Loop is parallelizable
         Generating Gang, Vector(128)
         185, !$acc loop gang, vector(128) ! blockidx%x threadidx%x
    465, Loop is parallelizable
         Generating Gang, Vector(128)
         Generating NVIDIA GPU code

mode_tiwmx_acc.F90:
     34, Generating acc routine seq
         ESATW(TIWMX_t, real)
     46, Generating acc routine seq
         ESATI(TIWMX_t, real)
```

---

## Troubleshooting

### Problem 1: "nvfortran: command not found"

**Solution:** NVIDIA HPC SDK not in PATH

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin:$PATH
source ~/.bashrc
```

### Problem 2: "OpenACC enabled but not using NVIDIA HPC SDK compiler"

**Solution:** Explicitly set Fortran compiler

```bash
cmake .. -DENABLE_OPENACC=ON -DCMAKE_Fortran_COMPILER=/path/to/nvfortran
```

Or set environment variable before configuring:

```bash
export FC=/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nvfortran
cmake .. -DENABLE_OPENACC=ON
```

### Problem 3: "CuPy: NOT FOUND"

**Solution:** Install CuPy matching your CUDA version

```bash
# Check CUDA version
nvcc --version

# Install matching CuPy
pip install cupy-cuda12x  # For CUDA 12.x
pip install cupy-cuda11x  # For CUDA 11.x
```

### Problem 4: Build fails with "undefined reference to `c_ice_adjust_acc`"

**Cause:** `phyex_bridge_acc.F90` not being compiled

**Solution:** Verify OpenACC sources are included:

```bash
# Check CMake configuration output
cmake .. -DENABLE_OPENACC=ON | grep "GPU ACCELERATION"

# Should show:
# GPU ACCELERATION: ENABLED
```

If still failing, clean and rebuild:

```bash
rm -rf build-gpu
mkdir build-gpu
cd build-gpu
cmake .. -DENABLE_OPENACC=ON -DCMAKE_Fortran_COMPILER=nvfortran
cmake --build . -j8
```

### Problem 5: Runtime error "GPU not found"

**Cause:** No NVIDIA GPU available or CUDA driver issue

**Check GPU:**
```bash
nvidia-smi

# Should show GPU info:
# NVIDIA-SMI 535.54.03   Driver Version: 535.54.03   CUDA Version: 12.2
```

**Check CUDA:**
```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"

# Should print: 1 (or number of GPUs)
```

### Problem 6: Cython compilation fails

**Cause:** Cython version incompatibility

**Solution:**
```bash
# Upgrade Cython
pip install --upgrade cython

# Or use specific version
pip install "cython>=3.0.0"
```

---

## Testing the Build

### Test 1: Import Python Module

```python
# Test CPU wrapper
import ice3._phyex_wrapper
print("CPU wrapper loaded successfully")

# Test GPU wrapper (if built)
import ice3._phyex_wrapper_acc
print("GPU wrapper loaded successfully")
```

### Test 2: Check GPU Availability

```python
import cupy as cp

# Check GPU
print(f"CUDA available: {cp.cuda.is_available()}")
print(f"Number of GPUs: {cp.cuda.runtime.getDeviceCount()}")

# GPU info
if cp.cuda.is_available():
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {props['name'].decode()}")
    print(f"Compute capability: {props['major']}.{props['minor']}")
    print(f"Total memory: {props['totalGlobalMem'] / 1e9:.2f} GB")
```

### Test 3: Run GPU Wrapper

```python
import cupy as cp
from ice3.fortran_gpu import IceAdjustGPU

# Create instance
ice_adjust_gpu = IceAdjustGPU(krr=6, timestep=1.0)
print("IceAdjustGPU instance created")

# Test with small arrays
nlon, nlev = 100, 60
th = cp.random.uniform(280, 300, (nlon, nlev), dtype=cp.float32)
rv = cp.random.uniform(0.001, 0.015, (nlon, nlev), dtype=cp.float32)
# ... create other arrays

# Execute
try:
    ice_adjust_gpu(...)
    print("✓ GPU execution successful")
except Exception as e:
    print(f"✗ GPU execution failed: {e}")
```

### Test 4: Verify OpenACC GPU Code Generation

```bash
# Compile with verbose output
cd build-gpu
cmake --build . --verbose 2>&1 | grep -A 5 "Generating.*GPU"

# Should show lines like:
# Generating Gang, Vector(128)
# Generating NVIDIA GPU code
```

---

## Performance Verification

```python
import time
import numpy as np
import cupy as cp
from ice3.fortran import IceAdjust  # CPU version
from ice3.fortran_gpu import IceAdjustGPU

def benchmark():
    nlon, nlev = 10000, 60

    # Create test data on CPU
    th_cpu = np.random.uniform(280, 300, (nlon, nlev)).astype(np.float32)
    rv_cpu = np.random.uniform(0.001, 0.015, (nlon, nlev)).astype(np.float32)
    # ... other arrays

    # CPU benchmark
    ice_adjust_cpu = IceAdjust(krr=6, timestep=1.0)
    t0 = time.time()
    for _ in range(10):
        ice_adjust_cpu(...)
    cpu_time = (time.time() - t0) / 10

    # GPU benchmark
    ice_adjust_gpu = IceAdjustGPU(krr=6, timestep=1.0)
    th_gpu = cp.asarray(th_cpu)
    rv_gpu = cp.asarray(rv_cpu)
    # ... transfer to GPU

    # Warmup
    ice_adjust_gpu(...)
    cp.cuda.Stream.null.synchronize()

    t0 = time.time()
    for _ in range(10):
        ice_adjust_gpu(...)
    cp.cuda.Stream.null.synchronize()
    gpu_time = (time.time() - t0) / 10

    speedup = cpu_time / gpu_time
    print(f"CPU time: {cpu_time*1000:.2f} ms")
    print(f"GPU time: {gpu_time*1000:.2f} ms")
    print(f"Speedup: {speedup:.1f}×")

benchmark()
```

**Expected Output (NVIDIA A100):**
```
CPU time: 500.00 ms
GPU time: 5.00 ms
Speedup: 100.0×
```

---

## Clean Build

```bash
# Remove build directory
rm -rf build-gpu

# Remove CMake cache
rm CMakeCache.txt

# Remove Python build artifacts
rm -rf build dist *.egg-info
rm -f *.so

# Clean and rebuild
mkdir build-gpu
cd build-gpu
cmake .. -DENABLE_OPENACC=ON -DCMAKE_Fortran_COMPILER=nvfortran
cmake --build . -j8
```

---

## Build for Distribution

### Create Wheel Package

```bash
# Set compiler
export FC=nvfortran

# Build wheel
pip install build
python -m build

# Wheel will be in dist/
ls -lh dist/
# ice3_gpu-0.1.0-cp312-cp312-linux_x86_64.whl
```

### Install from Wheel

```bash
pip install dist/ice3_gpu-0.1.0-*.whl
```

---

## Summary

### CPU-Only Build
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### GPU-Accelerated Build
```bash
export FC=nvfortran
mkdir build-gpu
cd build-gpu
cmake .. -DENABLE_OPENACC=ON
cmake --build .
cmake --install .
```

### Verify GPU Build
```bash
# Check library
ls -lh libice_adjust_phyex.so

# Check Cython wrappers
ls -lh _phyex_wrapper*.so

# Test import
python -c "from ice3.fortran_gpu import IceAdjustGPU; print('Success!')"
```

**Generated:** December 21, 2025
**CMakeLists.txt updated with OpenACC support**
