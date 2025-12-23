# JAX Shallow Convection Implementation

This directory contains the JAX implementation of the shallow convection scheme, translated from the Fortran PHYEX code.

## Files

### Main Entry Point (Recommended)

- **`shallow_convection.py`**: **Complete shallow convection scheme** ⭐
  - Single function call for the entire convection workflow
  - Automatically optimizes computation based on triggered columns
  - Easiest to use - recommended for most applications

### Component Functions (Advanced)

- **`shallow_convection_part1.py`**: First part of the shallow convection scheme
  - Prepares grid-scale thermodynamic variables (θ, θ_v, θ_es)
  - Tests for convective columns using the trigger function
  - Returns all convective properties at the LCL (Lifting Condensation Level)

- **`shallow_convection_part2.py`**: Second part of the shallow convection scheme
  - Computes updraft properties (mass flux, entrainment, detrainment)
  - Applies Fritsch-Chappell closure to adjust environmental values
  - Computes grid-scale convective tendencies
  - Applies conservation corrections

- **`shallow_convection_part2_select.py`**: Optimized version of part 2
  - Packs only triggered columns before computation
  - Significant performance improvement when convection is sparse
  - Same physics as part2, but more efficient

### Examples

- **`example_simple_shallow_convection.py`**: Simplest usage with main wrapper
- **`example_shallow_convection.py`**: Part 1 usage demonstration
- **`example_full_shallow_convection.py`**: Complete workflow using part 1 + part 2

- **`__init__.py`**: Package initialization exporting all functions

## Dependencies

This implementation relies on existing JAX stencils from `../stencils/`:
- `convect_trigger_shal.py`: Determines convective columns and LCL properties
- `convect_updraft_shal.py`: Computes updraft properties from DPL to CTL
- `convect_closure_shal.py`: Fritsch-Chappell closure scheme
- `convect_condens.py`: Condensation calculations
- `satmixratio.py`: Computes saturation mixing ratio
- `constants.py`: Physical constants

## Usage

### Simple usage (Recommended) - Main wrapper

The easiest way to use the shallow convection scheme:

```python
from ice3.jax.convection import shallow_convection, ConvectionParameters

# Single function call - automatic optimization!
outputs = shallow_convection(
    ppabst=pressure,        # (nit, nkt)
    pzz=height,            # (nit, nkt)
    ptkecls=tke,           # (nit,)
    ptt=temperature,       # (nit, nkt)
    prvt=water_vapor,      # (nit, nkt)
    prct=cloud_water,      # (nit, nkt)
    prit=ice,              # (nit, nkt)
    pwt=vertical_velocity, # (nit, nkt)
    ptten=temp_tend,       # (nit, nkt) - will be overwritten
    prvten=vapor_tend,     # (nit, nkt) - will be overwritten
    prcten=cloud_tend,     # (nit, nkt) - will be overwritten
    priten=ice_tend,       # (nit, nkt) - will be overwritten
    kcltop=cloud_top,      # (nit,) - will be overwritten
    kclbas=cloud_base,     # (nit,) - will be overwritten
    pumf=mass_flux,        # (nit, nkt) - will be overwritten
    pch1=chemicals,        # (nit, nkt, kch1)
    pch1ten=chem_tend,     # (nit, nkt, kch1) - will be overwritten
    convection_params=ConvectionParameters(),  # Use defaults
)

# Access results
temp_tendency = outputs.ptten
vapor_tendency = outputs.prvten
cloud_tops = outputs.kcltop
mass_flux = outputs.pumf
```

**Advantages**:
- ✅ Single function call - simplest API
- ✅ Automatic optimization (selects best computation method)
- ✅ No need to manage intermediate variables
- ✅ Perfect for production use

### Advanced usage - Complete workflow (Part 1 + Part 2)

```python
from ice3.jax.convection import (
    shallow_convection_part1,
    shallow_convection_part2,
    ConvectionParameters
)
from ice3.jax.stencils.constants import PHYS_CONSTANTS

# Set up your atmospheric data arrays
# ppabst: pressure (Pa), shape (nit, nkt)
# ptt: temperature (K), shape (nit, nkt)
# prvt: water vapor mixing ratio, shape (nit, nkt)
# ... etc

# PART 1: Trigger and prepare
part1_outputs = shallow_convection_part1(
    ppabst=ppabst,
    pzz=pzz,
    ptkecls=ptkecls,
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
    pch1ten=pch1ten,
    jcvexb=0,
    jcvext=0,
    convection_params=ConvectionParameters(),
    och1conv=False,
)

# PART 2: Updraft, closure, and tendencies
cst = PHYS_CONSTANTS
prdocp = cst.rd / cst.cpd

part2_outputs = shallow_convection_part2(
    ppabst=ppabst,
    pzz=pzz,
    ptt=ptt,
    prvt=prvt,
    prct=prct,
    prit=prit,
    pch1=pch1,
    prdocp=prdocp,
    ptht=part1_outputs.ptht,
    psthv=part1_outputs.psthv,
    psthes=part1_outputs.psthes,
    isdpl=part1_outputs.ksdpl,
    ispbl=part1_outputs.kspbl,
    islcl=part1_outputs.kslcl,
    psthlcl=part1_outputs.psthlcl,
    pstlcl=part1_outputs.pstlcl,
    psrvlcl=part1_outputs.psrvlcl,
    pswlcl=part1_outputs.pswlcl,
    pszlcl=part1_outputs.pszlcl,
    psthvelcl=part1_outputs.psthvelcl,
    gtrig1=part1_outputs.otrig1,
    kice=1,
    jcvexb=0,
    jcvext=0,
    convection_params=ConvectionParameters(),
    osettadj=False,
    ptadjs=10800.0,
    och1conv=False,
)

# Access outputs
print(f"Number of triggered columns: {part1_outputs.otrig1.sum()}")
print(f"Cloud top levels: {part2_outputs.ictl}")
print(f"Mass flux: {part2_outputs.pumf}")
print(f"Temperature tendency: {part2_outputs.pthc}")
```

### Optimized version (part2_select) - Recommended for production

For better performance when only a subset of columns have active convection:

```python
from ice3.jax.convection import (
    shallow_convection_part1,
    shallow_convection_part2_select,  # Use the select version
    ConvectionParameters
)

# ... Part 1 as above ...

# PART 2: Use select version (automatically handles packing/unpacking)
part2_outputs = shallow_convection_part2_select(
    # Same arguments as shallow_convection_part2
    ppabst=ppabst,
    pzz=pzz,
    ptt=ptt,
    prvt=prvt,
    prct=prct,
    prit=prit,
    pch1=pch1,
    prdocp=prdocp,
    ptht=part1_outputs.ptht,
    psthv=part1_outputs.psthv,
    psthes=part1_outputs.psthes,
    isdpl=part1_outputs.ksdpl,
    ispbl=part1_outputs.kspbl,
    islcl=part1_outputs.kslcl,
    psthlcl=part1_outputs.psthlcl,
    pstlcl=part1_outputs.pstlcl,
    psrvlcl=part1_outputs.psrvlcl,
    pswlcl=part1_outputs.pswlcl,
    pszlcl=part1_outputs.pszlcl,
    psthvelcl=part1_outputs.psthvelcl,
    gtrig1=part1_outputs.otrig1,
    kice=1,
    jcvexb=0,
    jcvext=0,
    convection_params=ConvectionParameters(),
)

# Output format is identical to part2
# But computation is faster when convection is sparse
```

**Performance benefit**: When only 10% of columns have active convection,
`shallow_convection_part2_select` can be up to 10x faster than `shallow_convection_part2`.

```

## Output Structures

### Part 1: `ShallowConvectionOutputs`

#### Tendencies (reset to zero in part 1)
- `ptten`: Temperature tendency (K/s)
- `prvten`: Water vapor tendency (1/s)
- `prcten`: Cloud water tendency (1/s)
- `priten`: Ice tendency (1/s)
- `pumf`: Updraft mass flux (kg/s m²)
- `pch1ten`: Chemical species tendencies (1/s)

#### Cloud Properties
- `kcltop`: Cloud top level indices
- `kclbas`: Cloud base level indices

#### Grid-scale Thermodynamic Variables
- `ptht`: Potential temperature (K)
- `psthv`: Virtual potential temperature (K)
- `psthes`: Equivalent potential temperature (K)

#### LCL Properties
- `kslcl`: LCL vertical index
- `ksdpl`: Departure level index
- `kspbl`: PBL top level index
- `psthlcl`: Updraft θ at LCL (K)
- `pstlcl`: Updraft temperature at LCL (K)
- `psrvlcl`: Updraft water vapor at LCL (kg/kg)
- `pswlcl`: Updraft vertical velocity at LCL (m/s)
- `pszlcl`: LCL height (m)
- `psthvelcl`: Environmental θ_v at LCL (K)
- `otrig1`: Trigger mask (boolean) indicating which columns have active convection

### Part 2: `ShallowConvectionPart2Outputs`

- `pumf`: Updraft mass flux per unit area (kg/s m²), shape (nit, nkt)
- `pthc`: Convective temperature tendency (K/s), shape (nit, nkt)
- `prvc`: Convective water vapor tendency (1/s), shape (nit, nkt)
- `prcc`: Convective cloud water tendency (1/s), shape (nit, nkt)
- `pric`: Convective ice tendency (1/s), shape (nit, nkt)
- `ictl`: Cloud top level indices, shape (nit,)
- `iminctl`: Minimum of cloud top and LCL indices, shape (nit,)
- `ppch1ten`: Chemical species convective tendency (1/s), shape (nit, nkt, kch1)

## Physics

The implementation follows the shallow convection scheme described in:
- Bechtold (1997): Meso-NH scientific documentation
- Fritsch and Chappell (1980), J. Atmos. Sci., Vol. 37, 1722-1761
- Kain and Fritsch (1990), J. Atmos. Sci., Vol. 47, 2784-2801

### Main Steps

#### Part 1: Trigger and Initialization

1. **Compute potential temperature**: θ = T × (P₀/P)^(Rd/Cp)

2. **Compute virtual potential temperature**: θ_v = θ × (1 + ε_a × r_v) / (1 + r_v + r_c + r_i)

3. **Compute equivalent potential temperature** using Bolton (1980) formula:
   - θ_es = T × (θ/T)^(1 - 0.28×e_s) × exp[(3374.6525/T - 2.5403) × e_s × (1 + 0.81×e_s)]

4. **Trigger shallow convection** by:
   - Constructing a mixed layer of at least XZPBL depth
   - Computing LCL using Bolton (1980) formula
   - Estimating cloud top via CAPE calculation
   - Checking if cloud depth exceeds threshold

#### Part 2: Updraft, Closure, and Tendencies

1. **Prepare environmental variables**:
   - Compute pressure differences: ΔP = P(k-1) - P(k)
   - Compute total water: r_w = r_v + r_c + r_i
   - Compute enthalpy: h = C_p×T + (1+r_w)×g×z - L_v×r_c - L_s×r_i

2. **Updraft calculation** (CONVECT_UPDRAFT_SHAL):
   - Set initial mass flux at LCL
   - Compute entrainment and detrainment rates
   - Calculate updraft thermodynamics level by level
   - Determine condensation and ice formation
   - Accumulate CAPE and find cloud top (CTL)

3. **Closure scheme** (CONVECT_CLOSURE_SHAL):
   - Compute environmental subsidence from mass continuity
   - Adjust mass flux iteratively (4 iterations) to remove CAPE
   - Update environment via mass flux convergence
   - Ensure numerical stability with fractional time steps

4. **Compute tendencies**:
   - Convert adjusted values to tendencies: dX/dt = (X_adj - X_init) / Δt
   - Apply smoothing at cloud top for PBL inversions
   - Apply conservation corrections (zero column integrals)

## JAX Features

This implementation is fully compatible with JAX:
- ✅ Pure functional (no side effects)
- ✅ JIT-compilable with `jax.jit()`
- ✅ Vectorizable with `jax.vmap()`
- ✅ Differentiable (gradients can be computed)
- ✅ GPU/TPU compatible

## Examples

Run the examples from the convection directory:

### Simple usage (Recommended)
```bash
cd src/ice3/jax/convection
python3 example_simple_shallow_convection.py
```
**Simplest example** using the main `shallow_convection` wrapper with automatic optimization.

### Part 1 only (trigger)
```bash
python3 example_shallow_convection.py
```
Demonstrates trigger function and LCL computation (advanced usage).

### Complete workflow (Part 1 + Part 2)
```bash
python3 example_full_shallow_convection.py
```
Demonstrates the full shallow convection scheme with explicit part1 and part2 calls (advanced usage).

## Summary

The JAX shallow convection implementation provides three levels of API:

1. **`shallow_convection()`** - ⭐ **Recommended** - Single function, automatic optimization
2. **`shallow_convection_part1()` + `shallow_convection_part2_select()`** - Advanced control with optimization
3. **`shallow_convection_part1()` + `shallow_convection_part2()`** - Full manual control

All implementations use the same validated physics from PHYEX and are fully JAX-compatible.
