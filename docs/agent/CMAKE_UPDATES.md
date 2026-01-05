# CMakeLists.txt Updates for GPU Acceleration

## Overview

The `CMakeLists.txt` has been updated to support conditional compilation of OpenACC GPU-accelerated modules alongside the standard CPU implementation.

**Date:** December 21, 2025

---

## Key Changes

### 1. Added OpenACC Option

**Lines 4-30:** New `ENABLE_OPENACC` CMake option

```cmake
option(ENABLE_OPENACC "Enable OpenACC GPU acceleration (requires NVIDIA HPC SDK)" OFF)
```

- **Default:** OFF (CPU-only build)
- **Enable:** `cmake -DENABLE_OPENACC=ON`
- **Requires:** NVIDIA HPC SDK compiler (`nvfortran`)

### 2. Compiler Detection

Automatically detects NVIDIA HPC SDK and sets appropriate flags:

```cmake
if(CMAKE_Fortran_COMPILER_ID MATCHES "^(NVHPC|PGI)$")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -acc -Minfo=accel -gpu=managed -Minline")
    set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -fast -Mcuda")
endif()
```

**Flags Added:**
- `-acc` - Enable OpenACC directives
- `-Minfo=accel` - Show GPU code generation info
- `-gpu=managed` - Use CUDA Unified Memory
- `-Minline` - Inline device functions
- `-fast` - Aggressive optimization (Release)
- `-Mcuda` - Enable CUDA features (Release)

### 3. OpenACC Source Files

**Lines 97-123:** New source variables for GPU modules

```cmake
if(ENABLE_OPENACC)
    set(MICRO_MODD_SOURCES_ACC
        ${PHYEX_DIR}/micro/modd_tiwmx_acc.F90
        ${PHYEX_DIR}/micro/modd_nebn_acc.F90
    )

    set(MICRO_MODE_SOURCES_ACC
        ${PHYEX_DIR}/micro/mode_tiwmx_acc.F90
    )

    set(MAIN_SOURCES_ACC
        ${PHYEX_DIR}/micro/condensation_acc.F90
        ${PHYEX_DIR}/micro/ice_adjust_acc.F90
    )

    set(BRIDGE_SOURCES_ACC
        ${PHYEX_DIR}/bridge/phyex_bridge_acc.F90
    )
endif()
```

**GPU Modules Added:**
- `modd_tiwmx_acc.F90` - Lookup table data with `!$acc declare`
- `modd_nebn_acc.F90` - Nebulosity parameters with `!$acc declare`
- `mode_tiwmx_acc.F90` - GPU-callable lookup functions (`!$acc routine seq`)
- `condensation_acc.F90` - GPU-parallelized condensation kernel
- `ice_adjust_acc.F90` - GPU-parallelized ice adjustment
- `phyex_bridge_acc.F90` - C-binding bridge with `!$acc data deviceptr()`

### 4. Updated Build Order

**Lines 282-329:** Modified `ALL_SOURCES` to include GPU modules

```cmake
set(ALL_SOURCES
    ${STUB_SOURCES}
    ${AUX_MODD_SOURCES}
    ${MICRO_MODD_SOURCES}
    ${TURB_MODD_SOURCES}
    ${CONV_MODD_SOURCES}

    # NEW: GPU data modules
    ${MICRO_MODD_SOURCES_ACC}

    ${PHYEX_SOURCES}
    ${MODI_SOURCES}
    ${AUX_MODE_SOURCES}
    ${AUX_FUNC_SOURCES}
    ${MICRO_MODE_SOURCES}

    # NEW: GPU mode modules
    ${MICRO_MODE_SOURCES_ACC}

    ${MICRO_FUNC_SOURCES}
    ${MAIN_SOURCES}

    # NEW: GPU main routines
    ${MAIN_SOURCES_ACC}

    ${BRIDGE_SOURCES}

    # NEW: GPU bridge
    ${BRIDGE_SOURCES_ACC}
)
```

**Build Order:**
1. Stubs (external dependencies)
2. CPU data modules (`modd_*.F90`)
3. GPU data modules (`modd_*_acc.F90`) ← **NEW**
4. PHYEX aggregation
5. MODI interfaces
6. CPU mode modules (`mode_*.F90`)
7. GPU mode modules (`mode_*_acc.F90`) ← **NEW**
8. Microphysics functions
9. CPU main routines
10. GPU main routines (`*_acc.F90`) ← **NEW**
11. CPU bridge
12. GPU bridge (`phyex_bridge_acc.F90`) ← **NEW**

### 5. GPU Cython Wrapper

**Lines 384-433:** New GPU Cython extension module

```cmake
if(ENABLE_OPENACC)
    set(CYTHON_SOURCE_ACC ${PHYEX_DIR}/bridge/_phyex_wrapper_acc.pyx)
    set(CYTHON_OUTPUT_ACC ${CMAKE_CURRENT_BINARY_DIR}/_phyex_wrapper_acc.c)

    add_custom_command(
        OUTPUT ${CYTHON_OUTPUT_ACC}
        COMMAND ${Python_EXECUTABLE} -m cython
            ${CYTHON_SOURCE_ACC}
            -o ${CYTHON_OUTPUT_ACC}
        DEPENDS ${CYTHON_SOURCE_ACC}
        COMMENT "Cythonizing _phyex_wrapper_acc.pyx to C (GPU version)"
    )

    Python_add_library(_phyex_wrapper_acc MODULE ${CYTHON_OUTPUT_ACC})
    target_link_libraries(_phyex_wrapper_acc PRIVATE ice_adjust_phyex)

    # Include CuPy headers if available
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import cupy; print(cupy.get_include())"
        OUTPUT_VARIABLE CUPY_INCLUDE_DIR
        ...
    )
endif()
```

**Builds:**
- `_phyex_wrapper_acc.so` (Python module)
- Links with `libice_adjust_phyex.so` containing GPU code
- Automatically finds CuPy headers

### 6. Updated Installation

**Lines 451-456, 475-479:** Install GPU wrapper if built

```cmake
if(ENABLE_OPENACC)
    install(TARGETS _phyex_wrapper_acc
        LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/ice3
    )
endif()
```

### 7. Enhanced Configuration Summary

**Lines 512-528:** Shows GPU build status

```
====================================
PHYEX ICE_ADJUST Configuration
====================================
Fortran Compiler: /opt/nvidia/hpc_sdk/.../nvfortran
Compiler ID: NVHPC
...

GPU ACCELERATION: ENABLED
  OpenACC: YES
  GPU Fortran modules: .../condensation_acc.F90;.../ice_adjust_acc.F90
  GPU Cython wrapper: _phyex_wrapper_acc
  CuPy: /path/to/cupy/_core/include
====================================
```

Or if disabled:

```
GPU ACCELERATION: DISABLED
  Use -DENABLE_OPENACC=ON to enable GPU support
  Requires: NVIDIA HPC SDK (nvfortran compiler)
```

---

## Build Targets

### CPU-Only Build (Default)

```bash
cmake ..
cmake --build .
```

**Creates:**
- `libice_adjust_phyex.so` - Fortran library with CPU code
- `_phyex_wrapper.so` - CPU Cython wrapper

**Does NOT create:**
- GPU modules (`*_acc.F90`)
- GPU wrapper (`_phyex_wrapper_acc.so`)

### GPU-Accelerated Build

```bash
cmake .. -DENABLE_OPENACC=ON -DCMAKE_Fortran_COMPILER=nvfortran
cmake --build .
```

**Creates:**
- `libice_adjust_phyex.so` - Fortran library with **CPU AND GPU** code
- `_phyex_wrapper.so` - CPU Cython wrapper
- `_phyex_wrapper_acc.so` - **GPU Cython wrapper** ← NEW

**Additional modules compiled:**
- `modd_tiwmx_acc.F90`
- `modd_nebn_acc.F90`
- `mode_tiwmx_acc.F90`
- `condensation_acc.F90`
- `ice_adjust_acc.F90`
- `phyex_bridge_acc.F90`

---

## Usage Examples

### Example 1: Default CPU Build

```bash
mkdir build
cd build
cmake ..
make -j8
```

**Result:** CPU-only library

### Example 2: GPU Build with NVIDIA HPC SDK

```bash
export FC=nvfortran
mkdir build-gpu
cd build-gpu
cmake .. -DENABLE_OPENACC=ON
make -j8
```

**Result:** Library with both CPU and GPU implementations

### Example 3: Using Build Script

```bash
# GPU build
./build_gpu.sh

# CPU-only build
./build_gpu.sh --cpu-only

# Clean rebuild
./build_gpu.sh --clean
```

### Example 4: Install to Python Environment

```bash
cd build-gpu
cmake --install . --prefix ~/.local
```

**Installs:**
- Libraries → `~/.local/lib/python3.X/site-packages/ice3/`
- Modules → `~/.local/lib/python3.X/site-packages/ice3/include/`

### Example 5: scikit-build-core (pip)

```bash
export FC=nvfortran
pip install -e . --no-build-isolation -C cmake.args="-DENABLE_OPENACC=ON"
```

---

## File Structure After Build

### CPU-Only Build

```
build/
├── libice_adjust_phyex.so          # Fortran library (CPU)
├── _phyex_wrapper.cpython-*.so     # Python wrapper (CPU)
└── modules/
    └── *.mod                        # Fortran module files
```

### GPU-Accelerated Build

```
build-gpu/
├── libice_adjust_phyex.so          # Fortran library (CPU + GPU)
├── _phyex_wrapper.cpython-*.so     # Python wrapper (CPU)
├── _phyex_wrapper_acc.cpython-*.so # Python wrapper (GPU) ← NEW
└── modules/
    ├── modd_tiwmx_acc.mod          # GPU modules ← NEW
    ├── mode_tiwmx_acc.mod          # ← NEW
    ├── condensation_acc.mod        # ← NEW
    ├── ice_adjust_acc.mod          # ← NEW
    └── ... (other .mod files)
```

---

## Verification

### Check GPU Modules Were Compiled

```bash
cd build-gpu
grep -r "Generating.*GPU" . --include="*.lst"

# Should show:
# Generating Gang, Vector(128)
# Generating NVIDIA GPU code
```

### Check Python Wrappers

```bash
ls -lh _phyex_wrapper*.so

# CPU build:
# _phyex_wrapper.cpython-312-darwin.so

# GPU build:
# _phyex_wrapper.cpython-312-darwin.so
# _phyex_wrapper_acc.cpython-312-darwin.so  ← NEW
```

### Test Import

```python
# CPU wrapper (always available)
from ice3 import _phyex_wrapper
print("CPU wrapper OK")

# GPU wrapper (only if ENABLE_OPENACC=ON)
from ice3 import _phyex_wrapper_acc
print("GPU wrapper OK")
```

---

## Troubleshooting

### Issue: "OpenACC enabled but not using NVIDIA HPC SDK compiler"

**Cause:** Using gfortran or other compiler instead of nvfortran

**Fix:**
```bash
export FC=nvfortran
cmake .. -DENABLE_OPENACC=ON -DCMAKE_Fortran_COMPILER=nvfortran
```

### Issue: "_phyex_wrapper_acc.so not created"

**Cause:** ENABLE_OPENACC not set or set to OFF

**Fix:**
```bash
cmake .. -DENABLE_OPENACC=ON
```

Check configuration output shows:
```
GPU ACCELERATION: ENABLED
```

### Issue: "CuPy: NOT FOUND" warning

**Non-critical:** Build will succeed, but GPU wrapper won't work at runtime

**Fix (optional):**
```bash
pip install cupy-cuda12x
cmake ..  # Reconfigure to detect CuPy
```

---

## Migration from Old Build System

### Old Approach (Manual)

```bash
nvfortran -acc -fPIC -shared \
  modd_*.F90 mode_*.F90 condensation.F90 ice_adjust.F90 \
  -o libphyex.so

python setup.py build_ext --inplace
```

### New Approach (CMake)

```bash
cmake .. -DENABLE_OPENACC=ON
cmake --build .
```

**Benefits:**
- Automatic dependency resolution
- Parallel compilation
- Proper module compilation order
- Integrated Cython build
- Easy CPU/GPU switching
- scikit-build-core compatible

---

## Summary of Changes

| Component | Before | After |
|-----------|--------|-------|
| **OpenACC Option** | ❌ Not available | ✅ `-DENABLE_OPENACC=ON/OFF` |
| **Compiler Detection** | Manual | ✅ Automatic NVHPC detection |
| **GPU Modules** | ❌ Not included | ✅ Conditional compilation |
| **GPU Wrapper** | ❌ Manual build | ✅ Automatic with CMake |
| **Build Targets** | CPU only | ✅ CPU + GPU (conditional) |
| **Installation** | Manual | ✅ `cmake --install` |
| **Configuration Summary** | Basic | ✅ Shows GPU status |

**Total Lines Changed:** ~120 lines added to CMakeLists.txt

**Backward Compatible:** ✅ Yes - CPU-only build works as before

**Generated:** December 21, 2025
