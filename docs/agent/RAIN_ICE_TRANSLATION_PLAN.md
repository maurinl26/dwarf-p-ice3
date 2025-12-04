# Rain-Ice JAX Translation Plan

## Overview

This document tracks the translation of the complete rain_ice component from GT4Py to JAX. The rain_ice scheme is significantly more complex than ice_adjust, involving multiple microphysical processes, nested time-stepping loops, and sedimentation.

## Translation Status

### ✅ Completed (Ice Adjust Foundation)

**Helper Functions**
- [x] `functions/ice_adjust.py` - Thermodynamic functions
- [x] `functions/tiwmx.py` - Saturation vapor pressure

**Stencils**
- [x] `stencils/ice_adjust.py` - Saturation adjustment

**Components**
- [x] `components/ice_adjust.py` - IceAdjustJAX wrapper

**Rain-Ice Orchestration**
- [x] `stencils/rain_ice.py` - Helper stencils (partial)
  - rain_ice_total_tendencies
  - rain_ice_thermo
  - rain_ice_mask
  - initial_values_saving
  - ice4_precipitation_fraction_sigma
  - rain_fraction_sedimentation
  - ice4_rainfr_vert
  - fog_deposition

### 🔄 In Progress / To Be Translated

**Core Process Stencils** (from `src/ice3/stencils/`)
- [ ] `ice4_correct_negativities.py` - Negative value corrections
- [ ] `ice4_compute_pdf.py` - PDF computations for subgrid
- [ ] `precipitation_fraction_liquid_content.py` - Precipitation fraction

**Ice4 Microphysical Process Stencils**
- [ ] `ice4_warm.py` - Warm processes (autoconversion, accretion, etc.)
- [ ] `ice4_nucleation.py` - Ice nucleation processes
- [ ] `ice4_fast_rs.py` - Fast snow processes
- [ ] `ice4_fast_rg.py` - Fast graupel processes
- [ ] `ice4_fast_ri.py` - Fast ice processes
- [ ] `ice4_slow.py` - Slow processes (aggregation, rimming, etc.)
- [ ] `ice4_rimltc.py` - Rime splintering
- [ ] `ice4_rrhong.py` - Graupel growth

**Time-Stepping & Integration**
- [ ] `ice4_stepping.py` - Nested time-stepping loops
  - ice4_stepping_heat
  - ice4_step_limiter
  - ice4_mixing_ratio_step_limiter
  - ice4_state_update
  - external_tendencies_update
  - tmicro_init, tsoft_init, ldcompute_init

**Ice4 Tendencies Component**
- [ ] `components/ice4_tendencies.py` - Orchestrates all ice4 processes
  - Calls fast_rs, fast_rg, fast_ri, slow, warm, nucleation, rimltc, rrhong

**Sedimentation**
- [ ] `sedimentation.py` - Fall speed and flux computations
  - sedimentation_stat (statistical method)
  - upwind_sedimentation (split method)

**Cloud Fraction & Condensation**
- [ ] `cloud_fraction.py` - Cloud fraction schemes
- [ ] `condensation.py` - Condensation stencils

**Main Component**
- [ ] `components/rain_ice.py` - High-level RainIceJAX wrapper

## Dependency Tree

```
RainIce (main component)
├── rain_ice helper stencils [✓]
│   ├── rain_ice_thermo
│   ├── rain_ice_mask
│   ├── initial_values_saving
│   ├── rain_ice_total_tendencies
│   └── fog_deposition
│
├── Sedimentation [ ]
│   ├── sedimentation_stat
│   └── upwind_sedimentation
│
├── Ice4Stepping [ ]
│   ├── ice4_stepping_heat
│   ├── ice4_step_limiter
│   ├── ice4_mixing_ratio_step_limiter
│   ├── ice4_state_update
│   └── external_tendencies_update
│
├── Ice4Tendencies component [ ]
│   ├── ice4_warm [ ]
│   ├── ice4_nucleation [ ]
│   ├── ice4_fast_rs [ ]
│   ├── ice4_fast_rg [ ]
│   ├── ice4_fast_ri [ ]
│   ├── ice4_slow [ ]
│   ├── ice4_rimltc [ ]
│   └── ice4_rrhong [ ]
│
├── Auxiliary stencils [ ]
│   ├── ice4_correct_negativities
│   ├── ice4_compute_pdf
│   └── precipitation_fraction_liquid_content
│
└── IceAdjust [✓]
    └── (already translated)
```

## Translation Priority

### Phase 1: Core Utilities (Days 1-2)
1. ice4_correct_negativities
2. ice4_compute_pdf
3. precipitation_fraction_liquid_content
4. cloud_fraction
5. condensation

### Phase 2: Sedimentation (Days 3-4)
1. sedimentation_stat
2. upwind_sedimentation
3. Helper functions for fall speeds

### Phase 3: Ice4 Processes (Days 5-10)
1. ice4_warm (largest, most complex)
2. ice4_slow (second largest)
3. ice4_fast_rs
4. ice4_fast_rg
5. ice4_fast_ri
6. ice4_nucleation
7. ice4_rimltc
8. ice4_rrhong

### Phase 4: Stepping & Integration (Days 11-12)
1. ice4_stepping stencils
2. Ice4Tendencies component

### Phase 5: Main Component (Days 13-14)
1. RainIceJAX wrapper
2. Integration testing
3. Example scripts

## Complexity Estimates

| Component | Lines | Complexity | Time Estimate |
|-----------|-------|------------|---------------|
| ice4_warm | ~800 | High | 2 days |
| ice4_slow | ~600 | High | 1.5 days |
| ice4_fast_rs | ~400 | Medium | 1 day |
| ice4_fast_rg | ~400 | Medium | 1 day |
| ice4_fast_ri | ~400 | Medium | 1 day |
| ice4_nucleation | ~300 | Medium | 1 day |
| Sedimentation | ~400 | Medium | 1 day |
| ice4_stepping | ~300 | Medium | 1 day |
| ice4_tendencies | ~500 | Medium | 1 day |
| Auxiliary stencils | ~200 each | Low | 0.5 days each |
| Main component | ~400 | Medium | 1 day |
| **Total** | **~6000** | **-** | **~14 days** |

## Key Translation Challenges

### 1. Nested Time-Stepping Loops
The stepping component has complex nested while loops that need careful translation:
```python
# Outer loop: while t_micro < dt
#   Inner loop: while ldcompute.any()
#     - Heat computation
#     - Tendency updates
#     - Step limiters
#     - State updates
```

**JAX Solution**: Use `jax.lax.while_loop` or `jax.lax.scan` for JIT-compatible loops.

### 2. Conditional Process Activation
Many processes are conditionally activated based on:
- Temperature thresholds
- Mixing ratio thresholds
- Configuration flags

**JAX Solution**: Use `jnp.where` for element-wise conditions, static if/else for config flags.

### 3. Multiple Hydrometeor Species
Ice4 scheme handles 6-7 species with complex interactions:
- Cloud liquid (rc)
- Cloud ice (ri)
- Rain (rr)
- Snow (rs)
- Graupel (rg)
- Hail (rh) - optional

**JAX Solution**: Maintain separate arrays for each species, use structured dictionaries.

### 4. Sedimentation Schemes
Two different sedimentation methods:
- Statistical (moments-based)
- Split explicit (upwind scheme)

**JAX Solution**: Implement both, select based on configuration.

### 5. Subgrid Variability
PDF-based schemes for:
- Cloud fraction
- Autoconversion
- Precipitation

**JAX Solution**: Port PDF calculations, maintain sigma calculations.

## Translation Guidelines

### Function Signatures
All JAX functions should follow this pattern:
```python
def process_name(
    # Input fields (Arrays)
    field1: Array,
    field2: Array,
    # Constants dictionary
    constants: Dict[str, Any],
) -> Union[Array, Tuple[Array, ...]]:
    """Docstring with algorithm description."""
    # Extract constants
    PARAM1 = constants["PARAM1"]
    
    # Computations
    result = ...
    
    return result
```

### Pure Functions
- No in-place modifications
- All outputs returned explicitly
- No global state

### Configuration Handling
- Static config (NRR, SEDIM mode): Python if/elif
- Dynamic conditions: `jnp.where`

### Array Operations
- Use `jnp.*` for all array ops
- Avoid Python loops where possible
- Use `jax.lax.scan` for required loops

### Documentation
- Port all docstrings from GT4Py version
- Add JAX-specific notes
- Include references to Fortran source lines

## Testing Strategy

### Unit Tests
For each translated stencil:
1. Compare outputs with GT4Py version
2. Test edge cases (zeros, extremes)
3. Verify conservation properties

### Integration Tests
1. Full rain_ice call with synthetic data
2. Compare with GT4Py rain_ice component
3. Verify energy/mass conservation

### Performance Tests
1. Benchmark JIT compilation time
2. Compare execution speed with GT4Py
3. Test on GPU if available

## File Organization

```
src/ice3/jax/
├── functions/
│   ├── __init__.py
│   ├── ice_adjust.py [✓]
│   ├── tiwmx.py [✓]
│   └── sedimentation_utils.py [ ]
│
├── stencils/
│   ├── __init__.py
│   ├── ice_adjust.py [✓]
│   ├── rain_ice.py [✓ partial]
│   ├── ice4_warm.py [ ]
│   ├── ice4_slow.py [ ]
│   ├── ice4_fast_rs.py [ ]
│   ├── ice4_fast_rg.py [ ]
│   ├── ice4_fast_ri.py [ ]
│   ├── ice4_nucleation.py [ ]
│   ├── ice4_rimltc.py [ ]
│   ├── ice4_rrhong.py [ ]
│   ├── ice4_stepping.py [ ]
│   ├── ice4_correct_negativities.py [ ]
│   ├── ice4_compute_pdf.py [ ]
│   ├── sedimentation.py [ ]
│   ├── cloud_fraction.py [ ]
│   └── condensation.py [ ]
│
├── components/
│   ├── __init__.py
│   ├── ice_adjust.py [✓]
│   ├── ice4_tendencies.py [ ]
│   └── rain_ice.py [ ]
│
├── examples/
│   ├── __init__.py
│   ├── simple_ice_adjust.py [✓]
│   └── simple_rain_ice.py [ ]
│
├── README.md [✓]
├── TRANSLATION_SUMMARY.md [✓]
└── RAIN_ICE_TRANSLATION_PLAN.md [✓ this file]
```

## Dependencies & Constants

### Required from Phyex
- All physical constants (CPD, CPV, CL, CI, RD, RV, etc.)
- Ice parameters (CRIAUTC, CRIAUTI, etc.)
- Rain-ice descriptors (fall speed parameters)
- Rain-ice parameters (collision efficiencies, etc.)

### New JAX Dependencies
None beyond base JAX (already required for ice_adjust).

## Validation Approach

### 1. Algorithm Preservation
- Port all mathematical operations exactly
- Maintain same physical thresholds
- Preserve process interactions

### 2. Numerical Verification
Create test cases from GT4Py outputs:
```python
# For each stencil
gt4py_output = gt4py_stencil(inputs)
jax_output = jax_stencil(inputs, constants)
assert jnp.allclose(jax_output, gt4py_output, rtol=1e-10)
```

### 3. Physical Consistency
- Energy conservation
- Mass conservation
- Positive definiteness
- Phase equilibrium

## Next Steps

1. **Immediate**: Continue with ice4_correct_negativities
2. **Short-term**: Complete Phase 1 stencils (utilities)
3. **Medium-term**: Implement Phase 2-3 (sedimentation & processes)
4. **Long-term**: Complete integration and testing

## Notes for Developers

### Common Patterns

**GT4Py Iteration to JAX**
```python
# GT4Py
with computation(PARALLEL), interval(...):
    result = f(x)

# JAX
result = f(x)  # Vectorized automatically
```

**GT4Py Conditional to JAX**
```python
# GT4Py
if __INLINED(FLAG):
    result = branch1
else:
    result = branch2

# JAX (static)
if FLAG:
    result = branch1
else:
    result = branch2

# JAX (dynamic)
result = jnp.where(condition, branch1, branch2)
```

**GT4Py Backward to JAX**
```python
# GT4Py
with computation(BACKWARD), interval(0, -1):
    field[0,0,0] = f(field[0,0,1])

# JAX
for k in range(nz-2, -1, -1):
    field = field.at[:,:,k].set(f(field[:,:,k+1]))
```

### Performance Tips
1. JIT compile entire functions, not just kernels
2. Use static arguments for config flags
3. Avoid Python loops - use `jax.lax.scan`
4. Batch operations when possible
5. Profile before optimizing

## References

- **Fortran Source**: PHYEX/src/common/micro/
- **GT4Py Source**: src/ice3/stencils/
- **JAX Documentation**: https://jax.readthedocs.io/
- **Ice Adjust Translation**: src/ice3/jax/TRANSLATION_SUMMARY.md

## Conclusion

The complete rain_ice translation represents approximately 6000 lines of code across 20+ stencils and components. With systematic approach and the foundation from ice_adjust, this is achievable in ~2 weeks of focused development.

**Current Status**: Foundation complete with ice_adjust and rain_ice helpers. Ready to proceed with Phase 1 utilities.
