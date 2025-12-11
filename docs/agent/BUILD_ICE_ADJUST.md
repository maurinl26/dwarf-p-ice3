# Building ICE_ADJUST from PHYEX-IAL_CY50T1 for Python Integration

This document explains how to compile the `ice_adjust.F90` routine from PHYEX-IAL_CY50T1 and use it in Python.

## Overview

The `ice_adjust.F90` routine performs microphysical adjustments for mixed-phase clouds. This build system compiles it along with all its dependencies into a shared library that can be called from Python.

## Prerequisites

### Required Software
- **CMake** (>= 3.12): Build system generator
- **gfortran**: GNU Fortran compiler
- **Python 3**: For creating Python bindings (if needed)
- **numpy**: For f2py (Python bindings)

### Installation on Linux
```bash
# Ubuntu/Debian
sudo apt-get install cmake gfortran python3-numpy

# Fedora/RHEL
sudo dnf install cmake gcc-gfortran python3-numpy
```

## Directory Structure

```
dwarf-p-ice3/
├── PHYEX-IAL_CY50T1/          # PHYEX source code
│   ├── aux/                    # Auxiliary modules
│   └── micro/                  # Microphysics routines
├── CMakeLists.txt              # CMake configuration (THIS FILE)
├── build_ice_adjust.sh         # Build script
└── BUILD_ICE_ADJUST.md         # This documentation
```

## Building the Library

### Method 1: Using the Build Script (Recommended)

```bash
./build_ice_adjust.sh
```

This script will:
1. Check for required dependencies
2. Create a `build_fortran` directory
3. Run CMake configuration
4. Compile all sources
5. Create `libice_adjust_phyex.so`

### Method 2: Manual Build

```bash
# Create build directory
mkdir -p build_fortran
cd build_fortran

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compile
make -j$(nproc)

# Optional: Install system-wide
sudo make install
```

### Build Outputs

After successful compilation:
- **Shared Library**: `build_fortran/libice_adjust_phyex.so`
- **Fortran Modules**: `build_fortran/modules/*.mod`

## Using in Python

### Option 1: Using f2py (Recommended for Array Operations)

Create a Python wrapper using f2py:

```bash
# Generate Python module from the Fortran source
cd build_fortran
f2py -c -m ice_adjust_py \
    --include-paths=modules \
    -L. -lice_adjust_phyex \
    ../PHYEX-IAL_CY50T1/micro/ice_adjust.F90
```

Then use in Python:
```python
import numpy as np
import ice_adjust_py

# Prepare input arrays (example dimensions)
nijt = 100  # Horizontal points
nkt = 50    # Vertical levels

# Call the Fortran routine
ice_adjust_py.ice_adjust(...)
```

### Option 2: Using ctypes (Direct Library Loading)

```python
import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./build_fortran/libice_adjust_phyex.so')

# Define the function signature
# Note: You'll need to match the exact Fortran interface
lib.ice_adjust_.argtypes = [
    # Define argument types matching Fortran signature
    ctypes.POINTER(ctypes.c_double),  # PRHODJ
    # ... add all other arguments
]

# Call the function
# ...
```

### Option 3: Using numpy.f2py Directly in Python

```python
import numpy.f2py as f2py

# Compile and import in one step
with open('ice_adjust_wrapper.f90', 'r') as f:
    source = f.read()

f2py.compile(source, modulename='ice_adjust_module',
             extra_args=['-L./build_fortran', '-lice_adjust_phyex'])

import ice_adjust_module
```

## ICE_ADJUST Routine Interface

The main subroutine `ICE_ADJUST` has the following signature:

```fortran
SUBROUTINE ICE_ADJUST (D, CST, ICEP, NEBN, TURBN, PARAMI, BUCONF, KRR,
                      HBUNAME, PTSTEP, PSIGQSAT,
                      PRHODJ, PEXNREF, PRHODREF, PSIGS, LMFCONV, PMFCONV,
                      PPABST, PZZ, PEXN, PCF_MF, PRC_MF, PRI_MF, PWEIGHT_MF_CLOUD,
                      PICLDFR, PWCLDFR, PSSIO, PSSIU, PIFR,
                      PRV, PRC, PRVS, PRCS, PTH, PTHS,
                      OCOMPUTE_SRC, PSRCS, PCLDFR,
                      PRR, PRI, PRIS, PRS, PRG, TBUDGETS, KBUDGETS,
                      PICE_CLD_WGT, PRH,
                      POUT_RV, POUT_RC, POUT_RI, POUT_TH,
                      PHLC_HRC, PHLC_HCF, PHLI_HRI, PHLI_HCF,
                      PHLC_HRC_MF, PHLC_HCF_MF, PHLI_HRI_MF, PHLI_HCF_MF)
```

### Key Input Parameters
- `D`: Dimension parameters (NIJT, NKT grid sizes)
- `CST`: Physical constants
- `ICEP`: Ice parameters
- `NEBN`: Nebulosity parameters
- `TURBN`: Turbulence parameters
- `PRHODJ`: Dry density × Jacobian
- `PRV, PRC, PRI`: Water vapor, cloud liquid, cloud ice mixing ratios
- `PTH`: Potential temperature

### Key Output Parameters
- `PRVS, PRCS, PRIS`: Sources/tendencies for vapor, liquid, ice
- `PTHS`: Temperature source
- `PCLDFR`: Cloud fraction
- `POUT_RV, POUT_RC, POUT_RI, POUT_TH`: Adjusted states (optional)

## Dependencies Included

The CMakeLists.txt compiles the following dependencies in order:

1. **Basic Modules** (modd_*): Constants, dimensions, parameters
2. **Microphysics Modules**: Rain/ice parameters, nebulosity
3. **Interfaces** (modi_*): Module interfaces
4. **Mode Modules** (mode_*): Thermodynamics, utilities
5. **Functions**: Gamma functions, interpolation
6. **Main Routines**: condensation.F90, ice_adjust.F90

## Troubleshooting

### Missing Source Files
If you get errors about missing files, ensure PHYEX-IAL_CY50T1 is properly extracted:
```bash
# If you have the tar.gz file
tar xf IAL_CY50T1.tar.gz
```

### Compiler Errors
- Check gfortran version: `gfortran --version` (recommend >= 7.0)
- Try Debug build for more info: `cmake .. -DCMAKE_BUILD_TYPE=Debug`

### Module Dependencies
If you get "module not found" errors, the compilation order may need adjustment in CMakeLists.txt.

## Advanced Options

### Custom Compiler Flags
```bash
cmake .. -DCMAKE_Fortran_FLAGS="-O3 -march=native -ffast-math"
```

### Debug Build
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

### Specify Compiler
```bash
cmake .. -DCMAKE_Fortran_COMPILER=ifort  # Use Intel Fortran
```

## Integration with Existing Python Code

If you're integrating with the existing gt4py-based ice3 code in this repository:

```python
# Example wrapper to call Fortran from Python
import numpy as np
from ice3.components.ice_adjust import IceAdjust  # GT4Py version
import ice_adjust_fortran  # Compiled Fortran version

# You can compare results or use Fortran as reference
def compare_implementations():
    # ... setup input data ...
    
    # Run GT4Py version
    result_gt4py = ice_adjust.run(...)
    
    # Run Fortran version
    result_fortran = ice_adjust_fortran.ice_adjust(...)
    
    # Compare
    diff = np.abs(result_gt4py - result_fortran)
    print(f"Max difference: {np.max(diff)}")
```

## References

- PHYEX Repository: https://github.com/UMR-CNRM/PHYEX
- PHYEX-IAL_CY50T1 Release: https://github.com/UMR-CNRM/PHYEX/releases/tag/IAL_CY50T1
- f2py Documentation: https://numpy.org/doc/stable/f2py/

## Support

For issues related to:
- **CMake configuration**: Check this BUILD_ICE_ADJUST.md
- **PHYEX source code**: Refer to PHYEX repository
- **Python integration**: See Python wrapper examples above
