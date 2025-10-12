# Shallow Convection JAX Translation

This directory contains the JAX translation of PHYEX shallow convection routines from Fortran to Python/JAX.

## Translation Status

### ✅ Completed Components

1. **convpar_shal.py** - Shallow convection parameters
   - Translated from `MODD_CONVPAR_SHAL.F90` and `INI_CONVPAR_SHAL.F90`
   - Python dataclass with default values
   - All 21 convection parameters included

2. **satmixratio.py** - Saturation mixing ratio computation
   - Translated from `convect_satmixratio.h`
   - Computes vapor saturation mixing ratio over liquid water
   - Returns latent heats (L_v, L_s) and specific heat (C_ph)
   - Fully functional JAX implementation

3. **trigger_shal.py** - Convection trigger function
   - Translated from `convect_trigger_shal.F90` (~350 lines)
   - Determines convective columns and LCL properties
   - Includes mixed layer computation, Bolton LCL formula, CAPE estimation
   - **Note**: Simplified CAPE computation for JAX compatibility
   - Uses Python loops (not yet JIT-optimized with `lax.scan`)

4. **condens.py** - Condensation routine
   - Translated from `convect_condens.F90` (~110 lines)
   - Computes temperature, cloud water, and ice from enthalpy and total water
   - Iterative solution (6 iterations) for thermodynamic consistency
   - Handles freezing transition between XTFRZ1 and XTFRZ2

5. **mixing_funct.py** - Entrainment/detrainment mixing function
   - Translated from `convect_mixing_funct.F90` (~90 lines)
   - Gaussian distribution function (KMF=1) for normalized rates
   - Uses error function approximation (Abramowitz & Stegun)
   - Returns normalized entrainment and detrainment rates

6. **updraft_shal.py** - Updraft computations
   - Translated from `convect_updraft_shal.F90` (~400 lines → ~280 lines JAX)
   - Computes updraft properties from DPL to CTL
   - Includes: mass flux, entrainment, detrainment, thermodynamics
   - Buoyancy calculations and vertical velocity
   - CAPE computation along updraft trajectory
   - **Note**: Simplified from Fortran (removed some loop optimizations)

### 🚧 Not Yet Implemented

The following components from `shallow_convection.F90` are **not yet translated**:

1. **convect_updraft_shal.F90** - Updraft computations
   - Cloud properties along updraft trajectory
   - Entrainment and detrainment calculations
   - Required for full convection scheme

2. **convect_closure_shal.F90** - Closure scheme
   - Mass flux determination
   - Convective adjustment time
   - Required for tendencies computation

3. **convect_condens.F90** - Condensation in updraft
   - Used within updraft calculations
   - Cloud water/ice formation

4. **SHALLOW_CONVECTION_PART1/PART2/PART2_SELECT** - Main orchestrators
   - These routines were referenced but not found in PHYEX/conv/
   - Likely optimization variants or may be in a different location
   - The main `shallow_convection.F90` calls these

5. **shallow_convection.F90** main wrapper
   - Top-level orchestration
   - Calls PART1, PART2, or PART2_SELECT based on conditions
   - Requires all sub-components to be completed first

## Architecture

```
convection/
├── __init__.py          # Module exports
├── README.md            # This file
├── convpar_shal.py      # ✅ Parameters dataclass
├── satmixratio.py       # ✅ Helper function
├── trigger_shal.py      # ✅ Trigger function (initial)
├── updraft_shal.py      # ⏳ TODO: Updraft computations
├── closure_shal.py      # ⏳ TODO: Closure scheme
└── shallow_convection.py # ⏳ TODO: Main wrapper

```

## Usage Example

```python
from ice3.convection import (
    CONVPAR_SHAL,
    init_convpar_shal,
    convect_satmixratio,
    convect_trigger_shal,
)
from ice3.phyex_common import Constants, PHYEX
import jax.numpy as jnp

# Initialize parameters
cvp_shal = init_convpar_shal()
cst = Constants()
phyex = PHYEX(...)

# Prepare input arrays (nit, nkt)
ppres = ...  # Pressure
pth = ...    # Potential temperature
pthv = ...   # Virtual potential temperature
# ... other inputs

# Call trigger function
pthlcl, ptlcl, prvlcl, pwlcl, pzlcl, pthvelcl, klcl, kdpl, kpbl, otrig = \
    convect_trigger_shal(
        cvp_shal, cst, phyex,
        ppres, pth, pthv, pthes, prv, pw, pz, ptkecls,
        kdpl, kpbl, klcl
    )

# otrig is boolean mask indicating which columns have triggered convection
```

## Implementation Notes

### JAX Compatibility Challenges

1. **Python Loops vs JAX Loops**
   - Current implementation uses Python `for` loops for clarity
   - For JIT compilation, these should be refactored using `jax.lax.scan` or `jax.lax.fori_loop`
   - Trade-off: readability vs performance

2. **Dynamic Array Indexing**
   - Fortran code uses computed indices (e.g., `ilcl(ji)`) for array access
   - JAX requires static or carefully managed dynamic indexing
   - Current solution uses `jnp.arange(nit)` for gathering

3. **Mask-based Computation**
   - Fortran uses `WHERE` statements and logical masks
   - JAX equivalent: `jnp.where()` for conditional updates
   - Maintains functional purity required by JAX

4. **Nested Loops**
   - Original Fortran has complex nested loops with early exits
   - Simplified in JAX version for initial translation
   - Full optimization would use `lax.scan` with carry state

### Differences from Fortran

1. **CAPE Computation**: Simplified compared to the highly optimized Fortran version with loop tiling
2. **Loop Optimization**: Fortran uses stride-4 loop optimization; JAX version is more straightforward
3. **Memory Layout**: Fortran is column-major, JAX/NumPy is row-major
4. **Reproducibility**: May have slight numerical differences due to loop ordering

## Next Steps

### Priority 1: Complete Core Components

1. **Translate convect_updraft_shal.F90**
   - Read and analyze the updraft routine
   - Implement cloud properties calculation
   - Handle entrainment/detrainment

2. **Translate convect_closure_shal.F90**
   - Implement mass flux closure
   - Add convective adjustment logic

### Priority 2: Integration

3. **Create main shallow_convection wrapper**
   - Integrate trigger, updraft, and closure
   - Handle PART1/PART2 logic (if needed)
   - Compute final tendencies

4. **Add Tests**
   - Unit tests for each component
   - Integration test with synthetic data
   - Comparison with Fortran output (if test data available)

### Priority 3: Optimization

5. **JIT Optimization**
   - Refactor loops using `lax.scan`
   - Add `@jax.jit` decorators
   - Profile performance

6. **Vectorization**
   - Ensure operations are fully vectorized
   - Remove remaining Python loops where possible

## References

- **Original Source**: `PHYEX/conv/shallow_convection.F90`
- **PHYEX Documentation**: Book2 (routine TRIGGER_FUNCT, CONVECT_UPDRAFT_SHAL)
- **Scientific References**:
  - Fritsch and Chappell (1980), J. Atm. Sci., Vol. 37, 1722-1761
  - Kain and Fritsch (1990), J. Atm. Sci., Vol. 47, 2784-2801
- **Original Author**: P. BECHTOLD, Laboratoire d'Aerologie

## Contributing

When adding new components:
1. Follow the existing pattern (functional, type-annotated)
2. Include comprehensive docstrings
3. Add references to original Fortran routines
4. Document any simplifications or differences
5. Include usage examples

## Known Limitations

1. **No JIT compilation yet**: Functions use Python loops
2. **Simplified CAPE**: Not as optimized as Fortran version
3. **Missing components**: Updraft, closure, main wrapper not yet implemented
4. **No chemical transport**: OCH1CONV feature deferred
5. **Test coverage**: No automated tests yet
