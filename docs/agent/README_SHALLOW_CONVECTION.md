# PHYEX Shallow Convection Cython Wrapper

This directory contains Cython wrappers for PHYEX Fortran routines, including the shallow convection scheme.

## Files

### Fortran Bridge
- **[phyex_bridge.F90](phyex_bridge.F90)**: Fortran module with ISO C bindings
  - `c_shallow_convection_wrap`: C-callable wrapper for `SHALLOW_CONVECTION`
  - `c_ice_adjust_wrap`: C-callable wrapper for `ICE_ADJUST`
  - `c_rain_ice_wrap`: C-callable wrapper for `RAIN_ICE`

### Cython Wrapper
- **[_phyex_wrapper.pyx](_phyex_wrapper.pyx)**: Cython module exposing Fortran routines to Python
  - `shallow_convection()`: Python function for shallow convection
  - `ice_adjust()`: Python function for saturation adjustment
  - `rain_ice()`: Python function for microphysics
  - `init_rain_ice()`: Initialization function for rain_ice

### Test Scripts
- **[test_shallow_convection_wrapper.py](test_shallow_convection_wrapper.py)**: Example usage of the shallow convection wrapper

## Compilation

### Prerequisites

You need:
- Python 3.x with NumPy
- Cython (`pip install cython`)
- Fortran compiler (gfortran recommended)
- C compiler (gcc)

### Building the Extension

```bash
cd PHYEX-IAL_CY50T1/bridge
python setup.py build_ext --inplace
```

This will compile:
1. `phyex_bridge.F90` → Fortran object files
2. `_phyex_wrapper.pyx` → C code → shared library (`.so` or `.pyd`)

After compilation, you should have a file like `_phyex_wrapper.cpython-*.so` (Linux) or `_phyex_wrapper.cpython-*.pyd` (Windows).

## Usage

### Basic Example

```python
import numpy as np
from _phyex_wrapper import shallow_convection

# Set up dimensions
nlon = 100  # horizontal points
nlev = 60   # vertical levels
kch1 = 1    # chemical species

# Create input arrays (Fortran order, single precision)
ptkecls = np.ones(nlon, dtype=np.float32, order='F') * 0.5
ppabst = np.ones((nlon, nlev), dtype=np.float32, order='F') * 85000.0
pzz = np.linspace(0, 15000, nlev, dtype=np.float32)
pzz = np.tile(pzz, (nlon, 1)).astype(np.float32, order='F')
# ... initialize other input arrays ...

# Initialize output arrays
ptten = np.zeros((nlon, nlev), dtype=np.float32, order='F')
prvten = np.zeros((nlon, nlev), dtype=np.float32, order='F')
prcten = np.zeros((nlon, nlev), dtype=np.float32, order='F')
priten = np.zeros((nlon, nlev), dtype=np.float32, order='F')
kcltop = np.zeros(nlon, dtype=np.int32, order='F')
kclbas = np.zeros(nlon, dtype=np.int32, order='F')
pumf = np.zeros((nlon, nlev), dtype=np.float32, order='F')
pch1 = np.zeros((nlon, nlev, kch1), dtype=np.float32, order='F')
pch1ten = np.zeros((nlon, nlev, kch1), dtype=np.float32, order='F')

# Call shallow convection
shallow_convection(
    kice=1,           # Include ice
    kbdia=1,          # Start at level 1
    ktdia=1,          # End at top
    osettadj=False,   # Use default adjustment time
    ptadjs=10800.0,   # Adjustment time (s)
    och1conv=False,   # No chemical tracers
    kch1=kch1,
    ptkecls=ptkecls,
    ppabst=ppabst,
    pzz=pzz,
    ptt=ptt,
    prvt=prvt,
    prct=prct,
    prit=prit,
    pwt=pwt,
    ptten=ptten,
    prvten=prvten,
    prcten=prcten,
    priten=priten,
    kcltop=kcltop,
    kclbas=kclbas,
    pumf=pumf,
    pch1=pch1,
    pch1ten=pch1ten
)

# Access results
n_triggered = (kcltop > 0).sum()
print(f"Convective columns: {n_triggered}")
print(f"Temperature tendency range: {ptten.min():.8f} - {ptten.max():.8f} K/s")
```

### Running the Test

```bash
cd PHYEX-IAL_CY50T1/bridge
python test_shallow_convection_wrapper.py
```

## API Reference

### `shallow_convection()`

Computes shallow convective tendencies for temperature, moisture, and clouds.

#### Parameters

**Scalar Parameters:**
- `kice` (int): Ice flag (1=include ice, 0=no ice)
- `kbdia` (int): Vertical computations start level (≥1)
- `ktdia` (int): Vertical computations end level (≥1)
- `osettadj` (bool): Use user-defined adjustment time
- `ptadjs` (float32): Adjustment time in seconds
- `och1conv` (bool): Include chemical tracer transport
- `kch1` (int): Number of chemical species

**1D Input Arrays** (float32, shape: `(nlon,)`, Fortran-contiguous):
- `ptkecls`: TKE in cloud layer (m²/s²)

**2D Input Arrays** (float32, shape: `(nlon, nlev)`, Fortran-contiguous):
- `ppabst`: Grid-scale pressure (Pa)
- `pzz`: Height of model layers (m)
- `ptt`: Grid-scale temperature (K)
- `prvt`: Water vapor mixing ratio (kg/kg)
- `prct`: Cloud water mixing ratio (kg/kg)
- `prit`: Cloud ice mixing ratio (kg/kg)
- `pwt`: Vertical velocity (m/s)

**2D Input/Output Arrays** (float32, shape: `(nlon, nlev)`):
- `ptten`: Temperature tendency (K/s)
- `prvten`: Water vapor tendency (1/s)
- `prcten`: Cloud water tendency (1/s)
- `priten`: Cloud ice tendency (1/s)
- `pumf`: Updraft mass flux (kg/s·m²)

**1D Input/Output Arrays** (int32, shape: `(nlon,)`):
- `kcltop`: Cloud top level (0 if no convection)
- `kclbas`: Cloud base level (0 if no convection)

**3D Input/Output Arrays** (float32, shape: `(nlon, nlev, kch1)`):
- `pch1`: Grid-scale chemical species
- `pch1ten`: Chemical species tendency (1/s)

#### Returns

None - all output arrays are modified in-place.

#### Important Notes

1. **Array Order**: All arrays **must** be Fortran-contiguous (use `order='F'` when creating)
2. **Data Type**: All float arrays **must** be `np.float32` (single precision)
3. **Integer Type**: Integer arrays **must** be `np.int32`
4. **In-Place Modification**: Output arrays are modified in-place, no return value
5. **Initialization**: Output arrays should be pre-allocated with zeros

## Implementation Details

### Data Flow

```
Python (NumPy arrays)
    ↓
Cython wrapper (_phyex_wrapper.pyx)
    ↓ (converts to C pointers)
C bridge (phyex_bridge.F90::c_shallow_convection_wrap)
    ↓ (converts pointers to Fortran arrays)
Fortran routine (SHALLOW_CONVECTION)
    ↓ (modifies arrays in-place)
C bridge
    ↓
Cython wrapper
    ↓
Python (modified NumPy arrays)
```

### Memory Layout

- **Python/Cython**: NumPy arrays with Fortran memory layout (column-major)
- **Fortran**: Native Fortran arrays
- **C Bridge**: Uses `ISO_C_BINDING` and `C_F_POINTER` for zero-copy conversion

The Fortran-contiguous requirement ensures that the memory layout matches between Python and Fortran, allowing efficient zero-copy data transfer.

### Parameter Structures

The Fortran bridge initializes several PHYEX structure types:

1. **DIMPHYEX_t**: Dimension parameters
2. **NSV_t**: Tracer indices
3. **CONVPAR_t**: Convection parameters (defaults used)
4. **CONVPAR_SHAL**: Shallow convection parameters
5. **CST_t**: Physical constants (initialized via `INI_CST()`)

## Troubleshooting

### Import Error

**Error**: `ImportError: No module named '_phyex_wrapper'`

**Solution**: Compile the extension first:
```bash
cd PHYEX-IAL_CY50T1/bridge
python setup.py build_ext --inplace
```

### Wrong Array Order

**Error**: Arrays not modified or strange results

**Solution**: Ensure arrays are Fortran-contiguous:
```python
# Correct
arr = np.zeros((nlon, nlev), dtype=np.float32, order='F')

# Incorrect
arr = np.zeros((nlon, nlev), dtype=np.float32)  # C-contiguous by default
```

### Wrong Data Type

**Error**: `ValueError: Buffer dtype mismatch`

**Solution**: Use `np.float32` for floats and `np.int32` for integers:
```python
# Correct
ptten = np.zeros((nlon, nlev), dtype=np.float32, order='F')

# Incorrect
ptten = np.zeros((nlon, nlev), order='F')  # defaults to float64
```

## References

- **PHYEX Documentation**: [Meso-NH scientific documentation](http://mesonh.aero.obs-mip.fr/)
- **Physics References**:
  - Bechtold (1997): Meso-NH scientific documentation
  - Fritsch and Chappell (1980), J. Atmos. Sci., Vol. 37, 1722-1761
  - Kain and Fritsch (1990), J. Atmos. Sci., Vol. 47, 2784-2801

## See Also

- [JAX Implementation](../../src/ice3/jax/convection/README.md): Pure JAX implementation of shallow convection
- [Fortran Source](../conv/shallow_convection.F90): Original Fortran implementation
