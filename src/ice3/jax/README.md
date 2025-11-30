# JAX Implementation of ICE3/ICE4 Microphysics

This directory contains JAX translations of the GT4Py ice adjustment stencils and related components from the PHYEX atmospheric physics package.

## Overview

The JAX implementation provides:
- **Automatic differentiation**: Compute gradients of microphysical processes
- **JIT compilation**: Improved performance through XLA compilation
- **GPU/TPU support**: Run on accelerators with minimal code changes
- **Functional paradigm**: Pure functions enable composition and parallelization

## Directory Structure

```
src/ice3/jax/
├── __init__.py                 # Package initialization
├── README.md                   # This file
├── components/                 # High-level components
│   ├── __init__.py
│   └── ice_adjust.py          # IceAdjustJAX component
├── stencils/                   # Core computation kernels
│   ├── __init__.py
│   └── ice_adjust.py          # ice_adjust stencil
└── functions/                  # Helper functions
    ├── __init__.py
    ├── ice_adjust.py          # Thermodynamic functions
    └── tiwmx.py               # Saturation vapor pressure
```

## Translation from GT4Py

### Key Differences

1. **No GT4Py decorators**: JAX uses standard Python functions
2. **Explicit constants**: Constants passed as dictionary instead of externals
3. **Pure functions**: No in-place modifications; all outputs returned
4. **Array operations**: Uses `jax.numpy` instead of GT4Py field operations
5. **Control flow**: Uses `jnp.where` for conditionals instead of GT4Py's `if`

### Translated Components

#### Functions (`functions/`)
- `ice_adjust.py`: Thermodynamic helper functions
  - `vaporisation_latent_heat()`: L_v(T) calculation
  - `sublimation_latent_heat()`: L_s(T) calculation
  - `constant_pressure_heat_capacity()`: c_p calculation

- `tiwmx.py`: Saturation vapor pressure
  - `e_sat_w()`: Saturation over liquid water
  - `e_sat_i()`: Saturation over ice

#### Stencils (`stencils/`)
- `ice_adjust.py`: Main saturation adjustment algorithm
  - CB02 subgrid condensation scheme
  - Ice fraction computation (FRAC_ICE_ADJUST modes 0, 3)
  - Subgrid autoconversion (None and Triangle PDF)
  - Cloud fraction calculation

#### Components (`components/`)
- `ice_adjust.py`: High-level `IceAdjustJAX` component
  - Wraps stencil with convenient interface
  - Manages physics configuration (Phyex)
  - Optional JIT compilation
  - Logging and diagnostics

## Usage

### Basic Example

```python
import jax.numpy as jnp
from ice3.jax.components.ice_adjust import IceAdjustJAX
from ice3.phyex_common.phyex import Phyex

# Initialize physics configuration for AROME
phyex = Phyex(program="AROME", TSTEP=60.0)

# Create ice adjustment component (JIT-compiled by default)
ice_adjust = IceAdjustJAX(phyex=phyex, jit=True)

# Prepare input fields
shape = (10, 10, 20)  # (nx, ny, nz)

# Thermodynamic state
sigqsat = jnp.ones(shape) * 0.01      # Subgrid saturation variability
pabs = jnp.ones(shape) * 85000.0      # Pressure (Pa)
sigs = jnp.ones(shape) * 0.1          # Subgrid mixing parameter
th = jnp.ones(shape) * 285.0          # Potential temperature (K)
exn = jnp.ones(shape) * 0.95          # Exner function
rho_dry_ref = jnp.ones(shape) * 1.0   # Reference density (kg/m³)

# Mixing ratios (kg/kg)
rv = jnp.ones(shape) * 0.005          # Water vapor
rc = jnp.zeros(shape)                 # Cloud liquid
ri = jnp.zeros(shape)                 # Cloud ice
rr = jnp.zeros(shape)                 # Rain
rs = jnp.zeros(shape)                 # Snow
rg = jnp.zeros(shape)                 # Graupel

# Mass flux contributions
cf_mf = jnp.zeros(shape)              # Cloud fraction from mass flux
rc_mf = jnp.zeros(shape)              # Liquid from mass flux
ri_mf = jnp.zeros(shape)              # Ice from mass flux

# Tendencies (initialized to zero)
rvs = jnp.zeros(shape)
rcs = jnp.zeros(shape)
ris = jnp.zeros(shape)
ths = jnp.zeros(shape)

# Run ice adjustment
results = ice_adjust(
    sigqsat=sigqsat, pabs=pabs, sigs=sigs, th=th,
    exn=exn, exn_ref=exn, rho_dry_ref=rho_dry_ref,
    rv=rv, rc=rc, ri=ri, rr=rr, rs=rs, rg=rg,
    cf_mf=cf_mf, rc_mf=rc_mf, ri_mf=ri_mf,
    rvs=rvs, rcs=rcs, ris=ris, ths=ths,
    timestep=60.0
)

# Unpack results
(t, rv_out, rc_out, ri_out, cldfr, 
 hlc_hrc, hlc_hcf, hli_hri, hli_hcf,
 cph, lv, ls, rvs_out, rcs_out, ris_out, ths_out) = results

print(f"Cloud fraction: {cldfr.mean():.4f}")
print(f"Cloud liquid: {rc_out.mean():.6f} kg/kg")
print(f"Cloud ice: {ri_out.mean():.6f} kg/kg")
```

### Using Custom Physics Configuration

```python
# Meso-NH configuration
phyex_mnh = Phyex(program="MESO-NH", TSTEP=2.0)
ice_adjust_mnh = IceAdjustJAX(phyex=phyex_mnh)

# Custom configuration
phyex_custom = Phyex(program="AROME", TSTEP=30.0)
phyex_custom.param_icen.SUBG_MF_PDF = 1  # Triangle PDF
phyex_custom.param_icen.FRAC_ICE_ADJUST = 3  # Statistical mode
ice_adjust_custom = IceAdjustJAX(phyex=phyex_custom)
```

### Automatic Differentiation

```python
import jax

# Define loss function
def loss_fn(rv_init, other_inputs):
    """Example: Minimize difference from target cloud fraction."""
    results = ice_adjust(rv=rv_init, **other_inputs)
    cldfr = results[4]
    target_cldfr = jnp.ones_like(cldfr) * 0.5
    return jnp.mean((cldfr - target_cldfr)**2)

# Compute gradient
grad_fn = jax.grad(loss_fn)
gradient = grad_fn(rv, other_inputs)
```

### GPU Acceleration

```python
# JAX automatically uses GPU if available
# Check device
import jax
print(f"Default backend: {jax.default_backend()}")

# Force CPU (for debugging)
with jax.default_device(jax.devices('cpu')[0]):
    results = ice_adjust(...)

# Force GPU
with jax.default_device(jax.devices('gpu')[0]):
    results = ice_adjust(...)
```

## Features

### Implemented

✅ Ice adjustment stencil (ice_adjust.F90 translation)
✅ Thermodynamic helper functions
✅ Saturation vapor pressure functions
✅ CB02 subgrid condensation scheme
✅ Cloud fraction computation
✅ Subgrid autoconversion (None and Triangle PDF)
✅ Ice fraction modes (0: temperature-based, 3: statistical)
✅ JIT compilation support
✅ Component wrapper with Phyex integration

### Configuration Options

The implementation supports the following configuration flags (set via Phyex):

- `LSUBG_COND`: Enable subgrid condensation scheme (default: True)
- `LSIGMAS`: Use sigma_s formulation (default: True)
- `LSTATNW`: Statistical cloud scheme (default: False)
- `FRAC_ICE_ADJUST`: Ice fraction mode (0: T-based, 3: statistical)
- `CONDENS`: Condensation scheme (0: CB02)
- `SUBG_MF_PDF`: Subgrid PDF type (0: None, 1: Triangle)
- `NRR`: Number of rain categories (2, 4, 5, 6)

## Performance Considerations

### JIT Compilation

First call includes compilation overhead:
```python
# First call: compilation + execution
results = ice_adjust(...)  # ~1-2 seconds

# Subsequent calls: execution only
results = ice_adjust(...)  # ~microseconds
```

### Memory Efficiency

JAX uses lazy evaluation and trace-based compilation:
- Avoid Python loops; use `jax.vmap` for vectorization
- Minimize data transfers between CPU/GPU
- Reuse compiled functions when possible

### Batch Processing

```python
# Process multiple atmospheric columns efficiently
batched_ice_adjust = jax.vmap(ice_adjust, in_axes=(0,) * 20)
batch_results = batched_ice_adjust(rv_batch, rc_batch, ...)
```

## Validation

The JAX implementation has been translated to match the GT4Py version, which itself
matches the original Fortran PHYEX code. Key validation points:

1. **Algorithm preservation**: All computation steps from ice_adjust.F90 retained
2. **Numerical equivalence**: Same numerical methods and thresholds
3. **Configuration compatibility**: Uses same Phyex parameter system
4. **Physical constraints**: Maintains phase equilibrium and conservation

## Limitations

- No 3D stencil neighborhood operations (not needed for ice_adjust)
- No MPI/distributed computing (use JAX's built-in parallelization instead)
- Different handling of externals (passed as dictionary)

## Dependencies

Required packages:
- `jax` >= 0.4.0
- `jaxlib` >= 0.4.0
- `numpy` >= 1.20.0

The implementation reuses the existing PHYEX configuration system:
- `ice3.phyex_common.phyex.Phyex`
- `ice3.phyex_common.constants.Constants`
- `ice3.phyex_common.rain_ice_parameters.IceParameters`

## References

- **Source Fortran**: `PHYEX/src/common/micro/ice_adjust.F90`
- **GT4Py version**: `src/ice3/stencils/ice_adjust.py`
- **JAX documentation**: https://jax.readthedocs.io/
- **PHYEX package**: Physics components from AROME/Meso-NH models

## Contributing

When extending the JAX implementation:

1. Maintain functional purity (no side effects)
2. Use `jax.numpy` for array operations
3. Document all constants and parameters
4. Include type hints (`Array`, `Dict[str, Any]`)
5. Add examples and tests
6. Preserve numerical equivalence with GT4Py version

## License

Same as parent project (PHYEX atmospheric physics package).
