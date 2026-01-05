# Using Compiled PHYEX Library with fmodpy

This document explains how to use the full compiled PHYEX library (`libice_adjust_phyex.so`) with fmodpy to call ICE_ADJUST from Python.

## Overview

The project uses a two-stage approach:

1. **CMake Build**: Compiles the entire PHYEX library including ICE_ADJUST and all dependencies
2. **fmodpy Integration**: Uses the compiled library to call Fortran from Python

## 1. Compiling PHYEX Library

### Build the Library

```bash
# Create build directory
mkdir -p build_fortran
cd build_fortran

# Configure with CMake
cmake ..

# Compile
make -j4

# Verify library was created
ls -lh libice_adjust_phyex.so
```

**Output:**
```
libice_adjust_phyex.so -> libice_adjust_phyex.so.1.0.0
libice_adjust_phyex.so.1 -> libice_adjust_phyex.so.1.0.0
libice_adjust_phyex.so.1.0.0  # ~1.15 MB
```

### What Gets Compiled

The CMakeLists.txt compiles in order:

1. **Stubs** (`parkind1`, `ec_parkind`, `yomhook`, `yomlun`)
2. **Data Modules** (`modd_*` - 15+ modules)
3. **PHYEX Aggregation** (`modd_phyex`)
4. **Mode Modules** (`mode_*` - 20+ modules)
5. **Functions** (`gamma`, `shuman`, etc.)
6. **Main Routines** (`condensation.F90`, `ice_adjust.F90`)

## 2. Using fmodpy with Compiled Library

### Approach 1: Use `compile_fortran_stencil` (Recommended)

The existing `compile_fortran_stencil` utility automatically handles:
- Finding source files
- Compiling with f2py/fmodpy
- Linking against compiled library
- Creating Python wrappers

```python
from ice3.utils.compile_fortran import compile_fortran_stencil

# This compiles ice_adjust.F90 and links with libice_adjust_phyex.so
ice_adjust_module = compile_fortran_stencil(
    fortran_script="../PHYEX-IAL_CY50T1/micro/ice_adjust.F90",
    fortran_module="ice_adjust",
    fortran_stencil="ice_adjust"
)

# Call the Fortran subroutine
result = ice_adjust_module.ice_adjust(
    # ... all parameters
)
```

### Approach 2: Direct Library Loading with ctypes

For direct access to the compiled library:

```python
import ctypes
import numpy as np
from pathlib import Path

# Load library
lib_path = Path("build_fortran/libice_adjust_phyex.so")
lib = ctypes.CDLL(str(lib_path))

# Access Fortran subroutine (note trailing underscore)
ice_adjust_func = lib.ice_adjust_

# Define argument types (complex - requires all derived types)
# This approach is tedious but gives full control
```

### Approach 3: Use IceAdjustFmodpy Wrapper (Best)

The `IceAdjustFmodpy` class provides a complete Python interface:

```python
from ice3.components.ice_adjust_fmodpy import IceAdjustFmodpy
from ice3.phyex_common.phyex import Phyex
import numpy as np

# Initialize
phyex = Phyex("AROME")
ice_adjust = IceAdjustFmodpy(phyex)

# Prepare arrays (MUST be Fortran-contiguous!)
nijt, nkt = 100, 60

prhodj = np.ones((nkt, nijt), dtype=np.float64, order='F')
pexnref = np.ones((nkt, nijt), dtype=np.float64, order='F')
# ... prepare all arrays

# Call ICE_ADJUST
result = ice_adjust(
    nijt=nijt, nkt=nkt,
    prhodj=prhodj, pexnref=pexnref, prhodref=prhodref,
    ppabst=ppabst, pzz=pzz, pexn=pexn,
    pcf_mf=pcf_mf, prc_mf=prc_mf, pri_mf=pri_mf,
    pweight_mf_cloud=pweight_mf_cloud,
    prv=prv, prc=prc, pri=pri, pth=pth,
    prr=prr, prs=prs, prg=prg,
    prvs=prvs, prcs=prcs, pris=pris, pths=pths,
    timestep=1.0,
    krr=6,
)

# Extract results
cloud_fraction = result['pcldfr']
ice_cloud_fraction = result['picldfr']
```

## 3. Complete Integration Example

### Step-by-Step Integration

```bash
# 1. Compile PHYEX library
cd build_fortran
cmake ..
make -j4
cd ..

# 2. Run Python example
python3 examples/test_ice_adjust_fmodpy_full.py
```

### Python Code Structure

```python
# File: your_script.py
import numpy as np
from ice3.components.ice_adjust_fmodpy import IceAdjustFmodpy
from ice3.phyex_common.phyex import Phyex

def main():
    # 1. Initialize PHYEX configuration
    phyex = Phyex("AROME")
    
    # 2. Create wrapper (loads compiled library)
    ice_adjust = IceAdjustFmodpy(phyex)
    
    # 3. Prepare input data
    nijt, nkt = 100, 60
    
    # Create Fortran-contiguous arrays
    data = {
        'prhodj': np.ones((nkt, nijt), dtype=np.float64, order='F'),
        'pexnref': np.ones((nkt, nijt), dtype=np.float64, order='F'),
        # ... all other arrays
    }
    
    # 4. Call Fortran ICE_ADJUST
    result = ice_adjust(
        nijt=nijt, nkt=nkt,
        **data,  # Pass all arrays
        timestep=1.0,
        krr=6,
    )
    
    # 5. Process results
    print(f"Cloud fraction range: {result['pcldfr'].min():.4f} - {result['pcldfr'].max():.4f}")
    
    return result

if __name__ == "__main__":
    result = main()
```

## 4. Library Architecture

### Compiled Library Contents

```
libice_adjust_phyex.so contains:
├── Fortran Modules (.mod files in build_fortran/modules/)
│   ├── modd_dimphyex
│   ├── modd_cst
│   ├── modd_rain_ice_param_n
│   ├── modd_neb_n
│   ├── modd_turb_n
│   ├── modd_param_ice_n
│   └── ... (30+ modules)
├── Subroutines
│   ├── ice_adjust_
│   ├── condensation_
│   ├── mode_thermo_*
│   └── ... (100+ routines)
└── Functions
    ├── gamma_
    ├── momg_
    └── ...
```

### Symbol Table

Check available symbols:

```bash
nm -D build_fortran/libice_adjust_phyex.so | grep ice_adjust
```

Output:
```
00000000001a2b40 T __ice_adjust_MOD_ice_adjust
000000000019f8c0 T ice_adjust_
```

## 5. fmodpy Integration Details

### How compile_fortran_stencil Works

```python
# In ice3/utils/compile_fortran.py
def compile_fortran_stencil(fortran_script, fortran_module, fortran_stencil):
    """
    Compiles Fortran code and links with libice_adjust_phyex.so
    
    Steps:
    1. Parse Fortran source
    2. Generate f2py signature
    3. Compile with f2py
    4. Link against libice_adjust_phyex.so
    5. Return Python module
    """
    
    # Add library path
    library_dirs = ['build_fortran']
    libraries = ['ice_adjust_phyex']
    
    # Compile with f2py
    # f2py -c -m module_name source.F90 -L./build_fortran -lice_adjust_phyex
    
    return compiled_module
```

### Linking Configuration

The wrapper automatically sets:

```python
# In setup or compilation
extra_link_args = [
    '-L./build_fortran',           # Library directory
    '-lice_adjust_phyex',          # Library name
    '-Wl,-rpath,./build_fortran',  # Runtime path
]
```

## 6. Array Memory Layout

### Critical: Fortran vs C Order

**Fortran (Column-major)**:
```python
# Memory: [col0_row0, col0_row1, col0_row2, col1_row0, ...]
arr = np.zeros((nkt, nijt), dtype=np.float64, order='F')
```

**C (Row-major)** - DEFAULT in NumPy:
```python
# Memory: [row0_col0, row0_col1, row0_col2, row1_col0, ...]
arr = np.zeros((nkt, nijt), dtype=np.float64, order='C')  # WRONG for Fortran!
```

### Always Use Fortran Order

```python
# Method 1: Create with order='F'
arr = np.zeros((nkt, nijt), dtype=np.float64, order='F')

# Method 2: Convert existing array
arr_c = np.zeros((nkt, nijt))  # C-order
arr_f = np.asfortranarray(arr_c)  # Convert to Fortran

# Method 3: Ensure operations preserve order
arr = np.ones((nkt, nijt), order='F')
result = np.asfortranarray(arr * 2.0)  # Ensure result is Fortran
```

### Verify Array Order

```python
def check_fortran_order(arr, name="array"):
    if not arr.flags['F_CONTIGUOUS']:
        raise ValueError(f"{name} must be Fortran-contiguous!")
    print(f"✓ {name} is Fortran-contiguous")

# Check all arrays
for name, arr in arrays.items():
    check_fortran_order(arr, name)
```

## 7. Troubleshooting

### Library Not Found

```python
FileNotFoundError: Compiled library not found at build_fortran/libice_adjust_phyex.so
```

**Solution:**
```bash
cd build_fortran
cmake ..
make -j4
```

### Symbol Not Found

```python
AttributeError: function 'ice_adjust_' not found
```

**Solution:** Check symbol name:
```bash
nm -D build_fortran/libice_adjust_phyex.so | grep ice_adjust
```

Try both:
- `ice_adjust_` (with underscore)
- `__ice_adjust_MOD_ice_adjust` (module version)

### Array Not Fortran-Contiguous

```python
ValueError: Array must be Fortran-contiguous
```

**Solution:**
```python
# Wrong
arr = np.zeros((nkt, nijt))  # C-order by default

# Right
arr = np.zeros((nkt, nijt), order='F')

# Or convert
arr = np.asfortranarray(arr)
```

### Segmentation Fault

Usually caused by:
1. Wrong array shapes
2. C-order arrays passed to Fortran
3. Uninitialized arrays
4. Type mismatches

**Debug:**
```python
# Add logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Validate arrays
for name, arr in arrays.items():
    print(f"{name}: shape={arr.shape}, F_CONTIGUOUS={arr.flags['F_CONTIGUOUS']}")
```

## 8. Performance Considerations

### Compilation Flags

In `CMakeLists.txt`:
```cmake
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -march=native")
```

### Build for Performance

```bash
cd build_fortran
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

### Benchmark

```python
import time

# Warmup
result = ice_adjust(...)

# Benchmark
n_iter = 100
start = time.time()
for _ in range(n_iter):
    result = ice_adjust(...)
elapsed = (time.time() - start) / n_iter

print(f"Time per call: {elapsed*1000:.2f} ms")
```

## 9. Comparison: gt4py vs fmodpy

| Feature | gt4py (Current) | fmodpy (This) |
|---------|-----------------|---------------|
| **Source** | Python translation | Original Fortran |
| **Compilation** | JIT | Pre-compiled |
| **Validation** | Against Fortran | IS Fortran |
| **Setup** | Complex | Moderate |
| **Performance** | Excellent | Native |
| **GPU Support** | Yes | No |
| **Maintenance** | High | Low |

## 10. Complete Example

See `examples/test_ice_adjust_fmodpy_full.py` for a complete working example.

## 11. Summary

**To use compiled PHYEX with fmodpy:**

1. ✅ Compile library: `cd build_fortran && cmake .. && make`
2. ✅ Use wrapper: `from ice3.components.ice_adjust_fmodpy import IceAdjustFmodpy`
3. ✅ Prepare arrays: All must be `order='F'`
4. ✅ Call ICE_ADJUST: Full parameter passing
5. ✅ Extract results: Dictionary with all outputs

**Key Points:**
- Library already compiled by CMake
- fmodpy wrapper ready to use
- All arrays MUST be Fortran-contiguous
- Full Fortran subroutine called, no shortcuts
- Perfect for validation vs gt4py

## References

- CMakeLists.txt - Compilation configuration
- src/ice3/components/ice_adjust_fmodpy.py - Wrapper implementation
- examples/test_ice_adjust_fmodpy_full.py - Complete example
- PHYEX-IAL_CY50T1/micro/ice_adjust.F90 - Fortran source
