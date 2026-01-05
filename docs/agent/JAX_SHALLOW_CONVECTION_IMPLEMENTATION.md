# JAX Translation of Shallow Convection - Implementation Guide

## Status

**Cython Wrapper**: ✅ Complete
**JAX Translation**: 🚧 In Progress - Foundation Ready

## What Has Been Completed

### 1. Cython/Fortran Bridge (100% Complete)

All files have been created and integrated to enable calling `shallow_convection.F90` from Python:

#### Files Modified/Created:
- **`PHYEX-IAL_CY50T1/conv/modi_shallow_convection.F90`** - Fortran interface module
- **`PHYEX-IAL_CY50T1/bridge/phyex_bridge.F90`** - C-callable wrapper function `c_shallow_convection_wrap()`
- **`PHYEX-IAL_CY50T1/bridge/_phyex_wrapper.pyx`** - Python/Cython wrapper `shallow_convection()`
- **`CMakeLists.txt`** - Build configuration with all dependencies

#### Key Features:
- Full parameter passing for shallow convection (15 2D arrays, 3 1D arrays, 6 scalars)
- Automatic Fortran structure initialization (DIMPHYEX, CONVPAR, CONVPAR_SHAL, NSV)
- AROME default parameters for operational use
- Proper array memory management (Fortran-contiguous, float32/int32)
- Complete documentation with examples

### 2. JAX Infrastructure (Foundation Complete)

#### Created:
- **`src/ice3/jax/stencils/convect_constants.py`** - Convection-specific constants
- **`src/ice3/jax/stencils/satmixratio.py`** - Already exists (saturation mixing ratio)

#### Constants Available:
```python
{
    "xdtpert": 1.0e-4,    # Temperature perturbation
    "xnhgam": 1.3333,     # Buoyancy gamma exponent
    "xtfrz1": 268.15,     # Freezing interval start
    "xtfrz2": 248.15,     # Freezing interval end
    "xstabt": 0.90,       # Stability threshold
    "xstabc": 0.95,       # Cloud stability threshold
    "xentr": 0.03,        # Entrainment constant
    "xdtadj": 3600.0,     # Adjustment timescale
    # ... and more
}
```

## JAX Translation Roadmap

The shallow convection scheme consists of approximately **7,500 lines** of Fortran code across 25 files. For numerical reproducibility, the following approach is recommended:

### Phase 1: Core Stencils (Priority: HIGH)

These are the fundamental building blocks needed for shallow convection:

#### 1.1 Trigger Function (`convect_trigger_shal.F90` - ~517 lines)
**Purpose**: Determine where convection is triggered

**Key Computations**:
- Compute mixed layer properties (averaging over PBL)
- Calculate lifting condensation level (LCL)
- Evaluate buoyancy and stability criteria
- Set parcel departure level

**JAX Translation Patterns**:
```python
# Bolton (1980) LCL formula
def compute_lcl(t, p, rv):
    """Compute lifting condensation level."""
    tdew = compute_dewpoint(rv, p)
    zlcl = 125.0 * (t - tdew)  # meters
    return zlcl

# Buoyancy check with jnp.where
def check_trigger(thv_parcel, thv_env, xtrig_thresh):
    """Check if convection is triggered."""
    buoyancy = thv_parcel - thv_env
    triggered = buoyancy > xtrig_thresh
    return jnp.where(triggered, 1, 0)
```

**Numerical Considerations**:
- Use exact same saturation vapor pressure formula as Fortran
- Preserve averaging order for mixed layer calculations
- Match Fortran's conditional logic with `jnp.where()`

#### 1.2 Updraft Calculation (`convect_updraft_shal.F90` - ~592 lines)
**Purpose**: Compute updraft properties (mass flux, temperature, moisture)

**Key Computations**:
- Entrainment/detrainment rates
- Condensation in updraft
- Vertical integration of mass flux
- Buoyancy-driven vertical velocity

**JAX Translation Patterns**:
```python
# Vertical loop with jax.lax.scan for efficiency
def updraft_vertical_loop(carry, level_inputs):
    """Process one vertical level of updraft."""
    # carry: (mass_flux, theta_u, rv_u, w_u, ...)
    # level_inputs: (theta_env, rv_env, p, z, ...)

    # Entrainment
    epsilon = xentr / dz
    mass_flux_new = mass_flux * (1.0 - epsilon)

    # Mixing
    theta_u_new = (theta_u + epsilon * theta_env) / (1.0 + epsilon)

    # Condensation
    rvsat = saturation_mixing_ratio(p, theta_u_new)
    rc_u = jnp.maximum(0.0, rv_u - rvsat)

    # Buoyancy
    thv_u = theta_u_new * (1.0 + 0.61 * rv_u - rc_u)
    thv_env = theta_env * (1.0 + 0.61 * rv_env)
    buoyancy = (thv_u - thv_env) / thv_env

    # Vertical velocity
    w_u_new = jnp.sqrt(jnp.maximum(0.0, w_u**2 + 2.0 * g * dz * buoyancy))

    return (mass_flux_new, theta_u_new, rv_u_new, w_u_new, ...), outputs

# Use scan for efficiency
init = (mass_flux0, theta_u0, rv_u0, w_u0, ...)
final, outputs = jax.lax.scan(updraft_vertical_loop, init, level_inputs)
```

**Numerical Considerations**:
- Vertical integration must match Fortran's bottom-up or top-down order
- Use `jax.lax.scan` instead of Python loops for JIT compatibility
- Preserve exact condensation calculation order

#### 1.3 Closure Scheme (`convect_closure_shal.F90` - ~720 lines)
**Purpose**: Determine mass flux magnitude from CAPE removal

**Key Computations**:
- Calculate CAPE (Convective Available Potential Energy)
- Compute adjustment timescale
- Determine mass flux to remove specified fraction of CAPE
- Apply stability constraints

**JAX Translation Patterns**:
```python
def compute_cape(thv_parcel, thv_env, dz):
    """Compute CAPE by vertical integration."""
    buoyancy = (thv_parcel - thv_env) / thv_env
    cape = jnp.sum(jnp.maximum(0.0, buoyancy * g * dz))
    return cape

def closure_mass_flux(cape, ptadjs, xdcape, dz):
    """Determine mass flux from CAPE closure."""
    # Remove xdcape fraction of CAPE over timescale ptadjs
    mass_flux = cape * xdcape / (g * dz * ptadjs)
    return mass_flux
```

**Numerical Considerations**:
- CAPE integral must use same vertical sum order as Fortran
- Match Fortran's min/max clipping exactly
- Preserve adjustment timescale formula

### Phase 2: Supporting Functions (Priority: MEDIUM)

#### 2.1 Condensation (`convect_condens.F90`)
- Already have saturation mixing ratio
- Need to add condensation rate calculation

#### 2.2 Mixing Function (`convect_mixing_funct.F90`)
- Entrainment/detrainment formulas
- Conservative mixing

#### 2.3 Closure Utilities (`convect_closure_thrvlcl.F90`)
- Virtual temperature at LCL
- Thermodynamic adjustments

### Phase 3: Integration (Priority: HIGH)

#### 3.1 Main Shallow Convection Routine
```python
def shallow_convection_jax(
    state: dict,
    constants: dict,
    dt: float,
    osettadj: bool = False,
    ptadjs: float = 3600.0,
) -> tuple[dict, dict]:
    """
    JAX implementation of shallow convection scheme.

    Parameters
    ----------
    state : dict
        Atmospheric state variables:
        - ptt: temperature (nlon, nlev) [K]
        - prvt: water vapor mixing ratio [kg/kg]
        - prct: cloud water mixing ratio [kg/kg]
        - prit: cloud ice mixing ratio [kg/kg]
        - ppabst: pressure [Pa]
        - pzz: height [m]
        - pwt: vertical velocity [m/s]
        - ptkecls: TKE in cloud layer [m²/s²]

    constants : dict
        Physical and convection constants

    dt : float
        Timestep [s]

    osettadj : bool
        Use user-defined adjustment time

    ptadjs : float
        User-defined adjustment timescale [s]

    Returns
    -------
    state_out : dict
        Updated state with convective tendencies applied

    diagnostics : dict
        - kcltop: cloud top level
        - kclbas: cloud base level
        - pumf: updraft mass flux [kg/m²/s]
        - cape: Convective available potential energy [J/kg]
    """

    # 1. Trigger: Determine convective columns
    triggered, lcl_level, pbl_top = convect_trigger_shal(
        state["ptt"], state["prvt"], state["ppabst"],
        state["pzz"], state["ptkecls"], constants
    )

    # 2. Updraft: Calculate updraft properties
    mass_flux, theta_up, rv_up, rc_up = convect_updraft_shal(
        state, lcl_level, pbl_top, triggered, constants
    )

    # 3. Closure: Determine mass flux magnitude
    mass_flux_final, cape = convect_closure_shal(
        mass_flux, theta_up, state["ptt"], state["ppabst"],
        state["pzz"], dt, ptadjs if osettadj else constants["xdtadj"],
        constants
    )

    # 4. Apply convective tendencies
    tendencies = compute_tendencies(
        mass_flux_final, theta_up, rv_up, rc_up,
        state, constants
    )

    # 5. Update state
    state_out = {
        "ptt": state["ptt"] + tendencies["ptten"] * dt,
        "prvt": state["prvt"] + tendencies["prvten"] * dt,
        "prct": state["prct"] + tendencies["prcten"] * dt,
        "prit": state["prit"] + tendencies["priten"] * dt,
    }

    diagnostics = {
        "kcltop": jnp.where(triggered, find_cloud_top(mass_flux_final), 0),
        "kclbas": jnp.where(triggered, lcl_level, 0),
        "pumf": mass_flux_final,
        "cape": cape,
    }

    return state_out, diagnostics
```

## Numerical Reproducibility Strategy

### 1. Match Fortran Precision
```python
# Use same precision as Fortran (single precision)
jax.config.update('jax_enable_x64', False)  # float32
```

### 2. Preserve Calculation Order
```python
# Fortran: mixed_layer = SUM(field * dz) / SUM(dz)
# JAX: Must use same summation order
mixed_layer = jnp.sum(field * dz, axis=0) / jnp.sum(dz, axis=0)
```

### 3. Match Conditionals Exactly
```python
# Fortran: IF (condition) THEN output = calc() ELSE output = 0.0
# JAX: Use jnp.where with same condition
output = jnp.where(condition, calc(), 0.0)
```

### 4. Preserve Vertical Loop Order
```python
# Fortran BACKWARD loop: k = nz-1, nz-2, ..., 0
# JAX: Use jax.lax.scan with reverse=True or explicit range
for k in range(nz-1, -1, -1):  # Python loop
    # OR
jax.lax.scan(fn, init, xs, reverse=True)  # JIT-friendly
```

### 5. Test Against Fortran
```python
def test_reproducibility():
    """Test JAX against Fortran implementation."""
    # Same input
    state_fortran = run_fortran_shallow_convection(inputs)
    state_jax, diag_jax = shallow_convection_jax(inputs_jax, constants, dt)

    # Compare outputs
    rtol = 1e-6  # Relative tolerance
    atol = 1e-8  # Absolute tolerance

    assert jnp.allclose(state_jax["ptten"], state_fortran["ptten"], rtol=rtol, atol=atol)
    assert jnp.allclose(state_jax["prvten"], state_fortran["prvten"], rtol=rtol, atol=atol)

    print(f"Max temperature tendency diff: {jnp.max(jnp.abs(state_jax['ptten'] - state_fortran['ptten']))}")
```

## Usage Example

### Cython/Fortran (Available Now)
```python
import numpy as np
from ice3._phyex_wrapper import shallow_convection

# Initialize arrays (Fortran order, single precision)
nlon, nlev = 100, 60
ppabst = np.ones((nlon, nlev), dtype=np.float32, order='F') * 85000.0
pzz = np.linspace(0, 12000, nlev, dtype=np.float32).reshape(1, -1).repeat(nlon, axis=0)
pzz = np.asfortranarray(pzz)
ptkecls = np.ones(nlon, dtype=np.float32, order='F') * 0.1
ptt = np.ones((nlon, nlev), dtype=np.float32, order='F') * 280.0
prvt = np.ones((nlon, nlev), dtype=np.float32, order='F') * 0.01
prct = np.zeros((nlon, nlev), dtype=np.float32, order='F')
prit = np.zeros((nlon, nlev), dtype=np.float32, order='F')
pwt = np.zeros((nlon, nlev), dtype=np.float32, order='F')

# Tendency arrays (initialized to zero)
ptten = np.zeros((nlon, nlev), dtype=np.float32, order='F')
prvten = np.zeros((nlon, nlev), dtype=np.float32, order='F')
prcten = np.zeros((nlon, nlev), dtype=np.float32, order='F')
priten = np.zeros((nlon, nlev), dtype=np.float32, order='F')

# Cloud levels
kcltop = np.zeros(nlon, dtype=np.int32, order='F')
kclbas = np.zeros(nlon, dtype=np.int32, order='F')

# Mass flux
pumf = np.zeros((nlon, nlev), dtype=np.float32, order='F')

# Call shallow convection
shallow_convection(
    timestep=60.0,
    kice=1,
    osettadj=False,
    ptadjs=3600.0,
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
    pumf=pumf
)

print(f"Number of convective columns: {np.sum(kcltop > 0)}")
print(f"Max temperature tendency: {np.max(np.abs(ptten))} K/s")
print(f"Max mass flux: {np.max(pumf)} kg/m²/s")
```

### JAX (When Complete)
```python
import jax.numpy as jnp
from ice3.jax.stencils import get_convection_constants
from ice3.jax.functions import shallow_convection_jax

# Get constants
constants = get_convection_constants()

# Prepare state (JAX arrays, float32)
state = {
    "ptt": jnp.ones((100, 60), dtype=jnp.float32) * 280.0,
    "prvt": jnp.ones((100, 60), dtype=jnp.float32) * 0.01,
    "prct": jnp.zeros((100, 60), dtype=jnp.float32),
    "prit": jnp.zeros((100, 60), dtype=jnp.float32),
    "ppabst": jnp.ones((100, 60), dtype=jnp.float32) * 85000.0,
    "pzz": jnp.linspace(0, 12000, 60, dtype=jnp.float32)[None, :].repeat(100, axis=0),
    "pwt": jnp.zeros((100, 60), dtype=jnp.float32),
    "ptkecls": jnp.ones(100, dtype=jnp.float32) * 0.1,
}

# Run shallow convection
state_out, diagnostics = shallow_convection_jax(
    state, constants, dt=60.0, osettadj=False, ptadjs=3600.0
)

# JIT compile for performance
from jax import jit
shallow_convection_jit = jit(shallow_convection_jax, static_argnames=("osettadj",))
state_out, diagnostics = shallow_convection_jit(state, constants, 60.0, False, 3600.0)
```

## Next Steps

### Immediate (1-2 weeks)
1. ✅ Complete Cython wrapper (DONE)
2. ✅ Create convection constants (DONE)
3. ⏳ Implement `convect_trigger_shal` JAX stencil
4. ⏳ Implement `convect_updraft_shal` JAX stencil
5. ⏳ Implement `convect_closure_shal` JAX stencil

### Short Term (2-4 weeks)
6. Integrate stencils into `shallow_convection_jax`
7. Create comprehensive test suite
8. Validate against Fortran reference
9. Optimize with `jax.lax.scan` for vertical loops
10. Profile and benchmark performance

### Medium Term (1-2 months)
11. Add deep convection translation
12. Implement full convection chemistry transport
13. Add gradient/AD support for optimization
14. Create documentation and tutorials

## File Structure

```
src/ice3/
├── jax/
│   ├── stencils/
│   │   ├── convect_constants.py        # ✅ DONE
│   │   ├── satmixratio.py             # ✅ EXISTS
│   │   ├── convect_trigger_shal.py    # TODO
│   │   ├── convect_updraft_shal.py    # TODO
│   │   ├── convect_closure_shal.py    # TODO
│   │   ├── convect_condens.py         # TODO
│   │   └── convect_mixing_funct.py    # TODO
│   ├── functions/
│   │   └── shallow_convection.py      # TODO - Main routine
│   └── examples/
│       └── simple_shallow_conv.py     # TODO - Usage example
│
PHYEX-IAL_CY50T1/
├── conv/
│   ├── modi_shallow_convection.F90    # ✅ DONE
│   └── shallow_convection.F90         # Reference
└── bridge/
    ├── phyex_bridge.F90               # ✅ DONE
    └── _phyex_wrapper.pyx             # ✅ DONE
```

## References

- **Fortran Source**: `PHYEX-IAL_CY50T1/conv/shallow_convection.F90`
- **JAX Documentation**: https://jax.readthedocs.io/
- **Existing JAX Stencils**: `src/ice3/jax/stencils/`
- **AROME Configuration**: Default parameters in convect_constants.py

## Contact & Support

For questions on the JAX translation:
1. Review existing JAX stencils in `src/ice3/jax/stencils/`
2. Check JAX README at `src/ice3/jax/README.md`
3. Refer to this implementation guide

---

**Last Updated**: December 20, 2025
**Status**: Cython wrapper complete, JAX foundation ready
**Priority**: Complete Phase 1 stencils for full JAX translation
