# DaCe Conversion Summary

## Overview

This document describes the conversion of ICE3/ICE4 microphysics stencils from GT4Py to DaCe (Data-Centric Parallel Programming) format for GPU acceleration.

## Converted Stencils

### Already Existed (3 stencils)
1. ✅ **ice4_fast_rg.py** - Graupel processes (rain contact freezing, collection)
2. ✅ **ice4_fast_rs.py** - Snow processes (collection, aggregation)
3. ✅ **sigrc_computation.py** - Subgrid condensate variance computation

### Newly Converted (8 stencils)

#### Cloud Fraction and Condensation
4. ✅ **cloud_fraction.py** - Cloud fraction computation with 3 stencils:
   - `thermodynamic_fields`: Compute T, Lv, Ls, Cph
   - `cloud_fraction_1`: Microphysical sources with conservation
   - `cloud_fraction_2`: Final cloud fraction and autoconversion

5. ✅ **condensation.py** - CB02 statistical condensation scheme with 2 stencils:
   - `condensation`: Main CB02 scheme (Chaboureau & Bechtold 2002)
   - `sigrc_computation`: Lookup table for subgrid variance (duplicate from #3)

#### ICE4 Fast Processes
6. ✅ **ice4_fast_ri.py** - Bergeron-Findeisen effect (ice crystal growth)
7. ✅ **ice4_compute_pdf.py** - PDF-based cloud partitioning
8. ✅ **ice4_correct_negativities.py** - Negativity correction with conservation
9. ✅ **ice4_nucleation.py** - Heterogeneous ice nucleation (HENI)
10. ✅ **ice4_rimltc.py** - Ice crystal melting above 0°C

### Total: 11 DaCe Stencils Implemented

## Component Translation

### IceAdjustModularDaCe

A new component `IceAdjustModularDaCe` has been created that:
- Uses DaCe stencils instead of GT4Py
- Maintains the same interface as the original `IceAdjustModular`
- Supports GPU acceleration through DaCe
- Implements the complete ICE_ADJUST microphysical adjustment sequence

Location: `src/ice3/components/ice_adjust_modular_dace.py`

## DaCe Conversion Pattern

All stencils follow a consistent conversion pattern:

### From GT4Py:
```python
from gt4py.cartesian.gtscript import PARALLEL, Field, computation, interval

def stencil_name(
    field1: Field["float"],
    field2: Field["float"],
):
    with computation(PARALLEL), interval(...):
        # computation logic
```

### To DaCe:
```python
import dace

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")

@dace.program
def stencil_name(
    field1: dace.float32[I, J, K],
    field2: dace.float32[I, J, K],
    param1: dace.float32,
):
    @dace.map
    def compute_something(i: _[0:I], j: _[0:J], k: _[0:K]):
        # computation logic using field[i, j, k]
```

### Key Differences:
1. **Symbols**: Use `dace.symbol()` for dimensions instead of implicit ranges
2. **Arrays**: Explicit shape `[I, J, K]` instead of `Field["float"]`
3. **Scalars**: Pass as `dace.float32` parameters instead of externals
4. **Iteration**: Use `@dace.map` instead of `computation(PARALLEL), interval(...)`
5. **Indexing**: Explicit `[i, j, k]` indexing instead of implicit

## Usage Example

### Using DaCe Component

```python
from ice3.components import IceAdjustModularDaCe
from ice3.phyex_common.phyex import Phyex
import numpy as np

# Initialize with AROME configuration
phyex = Phyex("AROME")
ice_adjust = IceAdjustModularDaCe(phyex, backend="gpu")

# Prepare input fields (all numpy arrays)
ni, nj, nk = 100, 100, 90  # domain size
domain = (ni, nj, nk)

# ... initialize all input arrays ...

# Execute the full ICE_ADJUST sequence
ice_adjust(
    sigqsat, exn, exnref, rhodref, pabs, sigs,
    cf_mf, rc_mf, ri_mf, th, rv, rc, rr, ri, rs, rg,
    cldfr, hlc_hrc, hlc_hcf, hli_hri, hli_hcf, sigrc,
    ths, rvs, rcs, ris, timestep, domain
)
```

### Using Individual DaCe Stencils

```python
from ice3.stencils_dace.cloud_fraction import thermodynamic_fields
import numpy as np

# Initialize fields
ni, nj, nk = 100, 100, 90
th = np.random.rand(ni, nj, nk).astype(np.float32)
exn = np.random.rand(ni, nj, nk).astype(np.float32)
# ... initialize other fields ...

# Call DaCe stencil directly
thermodynamic_fields(
    th, exn, rv, rc, rr, ri, rs, rg,
    lv, ls, cph, t,
    NRR=6, CPD=1004.0, CPV=1846.0, CL=4218.0, CI=2106.0
)
```

## Benefits of DaCe

1. **GPU Acceleration**: Automatic GPU code generation
2. **Performance**: Optimized data movement and kernel fusion
3. **Flexibility**: Can target different architectures (CPU, GPU, FPGA)
4. **Integration**: Works with NumPy arrays, no special storage needed
5. **Portability**: Same code runs on different hardware platforms

## File Structure

```
src/ice3/
├── components/
│   ├── ice_adjust_modular.py          # Original GT4Py component
│   ├── ice_adjust_modular_dace.py     # New DaCe component
│   └── __init__.py                     # Exports both components
├── stencils/                           # Original GT4Py stencils
│   ├── cloud_fraction.py
│   ├── condensation.py
│   ├── ice4_*.py
│   └── ...
└── stencils_dace/                      # New DaCe stencils
    ├── cloud_fraction.py
    ├── condensation.py
    ├── ice4_compute_pdf.py
    ├── ice4_correct_negativities.py
    ├── ice4_fast_rg.py
    ├── ice4_fast_ri.py
    ├── ice4_fast_rs.py
    ├── ice4_nucleation.py
    ├── ice4_rimltc.py
    └── sigrc_computation.py
```

## Remaining Work

### Not Yet Converted (9 stencils)
- ice4_rrhong.py - Homogeneous freezing of rain
- ice4_slow.py - Slow microphysical processes
- ice4_stepping.py - Time stepping routines
- ice4_tendencies.py - Tendency computation
- ice4_warm.py - Warm rain processes
- ice_adjust.py - Monolithic ice adjustment
- precipitation_fraction_liquid_content.py - Precipitation fraction
- rain_ice.py - Rain-ice interactions
- sedimentation.py - Precipitation sedimentation

These can be converted following the same pattern when needed.

## Testing

To test the DaCe component:

```python
# Run existing tests with DaCe backend
pytest tests/repro_dace/

# Compare with GT4Py version
pytest tests/components/test_ice_adjust_modular.py
```

## Performance Considerations

1. **First Call Overhead**: DaCe compiles code on first execution (JIT)
2. **Memory Management**: Use contiguous NumPy arrays for best performance
3. **Data Transfer**: Minimize CPU-GPU transfers by processing in batches
4. **Kernel Fusion**: DaCe automatically fuses compatible operations

## References

- **DaCe Documentation**: https://spcldace.readthedocs.io/
- **PHYEX Reference**: PHYEX-IAL_CY50T1/micro/
- **Original Paper**: Chaboureau & Bechtold (2002) for CB02 scheme

## Authors

- Original Fortran: PHYEX team (Météo-France)
- GT4Py Translation: Previous contributors
- DaCe Translation: Cline AI Assistant (2025)

## License

Same as the original dwarf-p-ice3 project.
