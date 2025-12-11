# Fortran-Python Bindings Guide

This document explains how to use the Fortran library from Python in the dwarf-p-ice3 project.

## Overview

The project provides multiple approaches for calling Fortran code from Python:

1. **fmodpy** (Recommended) - Already integrated, automatic binding generation
2. **ctypes** - Manual bindings for simpler cases
3. **Cython** - High-performance custom bindings (advanced)

## Method 1: Using fmodpy (Recommended)

The project already uses `fmodpy` which automatically generates Python bindings from Fortran source code.

### Example Usage

```python
from ice3.utils.compile_fortran import compile_fortran_stencil

# Compile and import a Fortran stencil
condensation = compile_fortran_stencil(
    fortran_script="mode_condensation.F90",
    fortran_module="mode_condensation", 
    fortran_stencil="condensation"
)

# Now call the Fortran function directly
result = condensation(
    d=d_params,
    cst=cst_params,
    # ... other parameters
)
```

### How it Works

1. `fmodpy.fimport()` compiles the Fortran source on-the-fly
2. Creates Python wrappers that handle type conversion automatically
3. Numpy arrays are automatically converted to/from Fortran arrays
4. Fortran derived types are mapped to Python objects

### Advantages

- ✅ Automatic type conversion
- ✅ Handles complex Fortran features (modules, derived types)
- ✅ No manual wrapper code needed
- ✅ Good performance

## Method 2: Using the Compiled Library with ctypes

For the compiled `libice_adjust_phyex.so` library, you can use ctypes:

```python
from ice3.fortran_bindings import FortranArray, find_fortran_library
import ctypes
import numpy as np

# Load library
lib_path = find_fortran_library()
lib = ctypes.CDLL(str(lib_path))

# Prepare Fortran-ordered arrays
nijt, nkt = 100, 60
temperature = FortranArray.prepare_array((nijt, nkt))
pressure = FortranArray.prepare_array((nijt, nkt))

# Convert to Fortran pointers
temp_ptr = FortranArray.to_fortran(temperature)
press_ptr = FortranArray.to_fortran(pressure)

# Call Fortran function (see notes below about complexity)
```

### Challenges with ctypes

- ⚠️ Fortran derived types (TYPE) are difficult to map
- ⚠️ Manual type conversion required
- ⚠️ Character strings need special handling  
- ⚠️ Array indexing differences (1-based vs 0-based)

## Method 3: Cython Bindings (Advanced)

For maximum performance and control, use Cython. See [Cython Setup](##cython-setup) below.

## Cython Setup

### Prerequisites

Add Cython to your dependencies:

```bash
pip install cython numpy
```

### Create Cython Wrapper

1. **Define Fortran interface** (`phyex_wrappers.pxd`):

```cython
# cython: language_level=3
cdef extern from *:
    """
    void condensation_(/* parameters */);
    """
    void condensation_(...)
```

2. **Create Python wrapper** (`phyex_wrappers.pyx`):

```cython
# cython: language_level=3
import numpy as np
cimport numpy as cnp

def call_condensation(cnp.ndarray[double, ndim=2, mode='fortran'] temperature,
                      cnp.ndarray[double, ndim=2, mode='fortran'] pressure,
                      # ... more parameters
                      ):
    """
    Python wrapper for Fortran CONDENSATION subroutine.
    """
    cdef int nijt = temperature.shape[0]
    cdef int nkt = temperature.shape[1]
    
    # Prepare output arrays  
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] rv_out = np.zeros((nijt, nkt), order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] rc_out = np.zeros((nijt, nkt), order='F')
    
    # Call Fortran
    # Note: This requires handling Fortran derived types properly
    # which is complex - see full implementation
    
    return {
        'rv_out': rv_out,
        'rc_out': rc_out,
    }
```

3. **Setup compilation** (`setup.py`):

```python
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "ice3.cython_bindings.phyex_wrappers",
        sources=["src/ice3/cython_bindings/phyex_wrappers.pyx"],
        include_dirs=[numpy.get_include()],
        library_dirs=["build_fortran"],
        libraries=["ice_adjust_phyex"],
        extra_compile_args=['-O3'],
    )
]

setup(
    name="dwarf-p-ice3",
    ext_modules=cythonize(extensions, language_level="3"),
)
```

4. **Build**:

```bash
python setup.py build_ext --inplace
```

### Using Cython Bindings

```python
from ice3.cython_bindings.phyex_wrappers import call_condensation
import numpy as np

# Prepare input (Fortran-ordered!)
temp = np.asfortranarray(np.random.rand(100, 60))
press = np.asfortranarray(np.random.rand(100, 60))

# Call
result = call_condensation(temp, press, ...)

# Use
results
print(result['rv_out'])
```

### Advantages of Cython

- ✅ Maximum performance (C-level speed)
- ✅ Fine control over memory and types
- ✅ Can inline small Fortran calls
- ✅ Type checking at compile time

### Disadvantages

- ❌ More complex setup
- ❌ Requires compilation step
- ❌ Still challenging with Fortran derived types
- ❌ Maintenance burden

## Recommendations

**For most users**: Use the existing `fmodpy` approach via `compile_fortran.py`. It handles the complexity automatically.

**For simple performance-critical loops**: Use Cython with the compiled library.

**For quick prototypes**: Use the ctypes wrapper in `fortran_bindings`.

## Handling Fortran Derived Types

Fortran `TYPE` structures (like `DIMPHYEX_t`, `CST_t`) are challenging to bind. Options:

1. **fmodpy**: Handles this automatically
2. **ctypes**: Create equivalent C structs (tedious)
3. **Cython**: Similar to ctypes, requires manual struct definitions
4. **Best approach**: Use Fortran helper subroutines that take simple arrays

Example helper subroutine in Fortran:

```fortran
SUBROUTINE CONDENSATION_SIMPLE(NIJT, NKT, PPABS, PZZ, PT, PRV_OUT, ...)
  INTEGER, INTENT(IN) :: NIJT, NKT
  REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: PPABS, PZZ, PT
  REAL, DIMENSION(NIJT,NKT), INTENT(OUT) :: PRV_OUT
  ! ... 
  ! Internal: create derived types from simple parameters
  ! Call main CONDENSATION
END SUBROUTINE
```

Then bind the simple helper from Python.

## Performance Comparison

| Method | Setup | Performance | Ease of Use |
|--------|-------|-------------|-------------|
| fmodpy | Easy | Good | ⭐⭐⭐⭐⭐ |
| ctypes | Medium | Good | ⭐⭐⭐ |
| Cython | Hard | Excellent | ⭐⭐ |

## See Also

- [fmodpy documentation](https://github.com/tchlux/fmodpy)
- [Cython with Fortran](https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html)
- `src/ice3/utils/compile_fortran.py` - Current implementation
