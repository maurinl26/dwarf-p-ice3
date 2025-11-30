# ICE4 Tendencies JAX Translation Status

## Overview

The ice4_tendencies component is the core orchestrator of all microphysical processes in the ICE4 scheme. It requires ~20 process stencils and aggregates their contributions into total tendencies.

## ✅ Completed: Utility & Management Stencils

**File**: `src/ice3/jax/stencils/ice4_tendencies.py`

Translated 8 key management functions:

1. **ice4_nucleation_post_processing** - Apply heterogeneous nucleation changes
2. **ice4_rrhong_post_processing** - Apply homogeneous rain freezing
3. **ice4_rimltc_post_processing** - Apply ice crystal melting
4. **ice4_fast_rg_pre_post_processing** - Aggregate graupel sources
5. **ice4_increment_update** - Update increments with phase changes
6. **ice4_derived_fields** - Compute supersaturation, conductivity, diffusivity, ventilation
7. **ice4_slope_parameters** - Compute size distribution slopes (lambda) for rain/snow/graupel
8. **ice4_total_tendencies_update** - Master aggregation of 30+ process rates

These functions:
- Handle tendency initialization and aggregation
- Manage thermodynamic consistency (latent heat accounting)
- Compute derived microphysical fields
- Aggregate contributions from all processes

## 🔄 Remaining: Process Stencils

These are the large microphysical process stencils that must be translated to complete ice4_tendencies:

### Critical Process Stencils (Required)

| Stencil | LOC | Complexity | Priority | Description |
|---------|-----|------------|----------|-------------|
| **ice4_nucleation** | ~150 | Medium | HIGH | Heterogeneous ice nucleation (HENI) |
| **ice4_rrhong** | ~100 | Low | HIGH | Homogeneous rain freezing (HONG) |
| **ice4_rimltc** | ~100 | Low | HIGH | Ice crystal melting (IMLTC) |
| **ice4_warm** | ~800 | High | HIGH | Warm processes (autoconversion, accretion, evaporation) |
| **ice4_slow** | ~600 | High | HIGH | Slow processes (aggregation, deposition, Bergeron) |
| **ice4_fast_rs** | ~400 | Medium | HIGH | Fast snow processes (riming, accretion, melting) |
| **ice4_fast_rg** | ~400 | Medium | HIGH | Fast graupel processes (wet/dry growth) |
| **ice4_fast_ri** | ~200 | Medium | MEDIUM | Fast ice processes (Bergeron-Findeisen) |
| **ice4_compute_pdf** | ~100 | Low | MEDIUM | PDF computations for subgrid variance |

**Total**: ~2,850 lines for process stencils

### Supporting Stencils

Also needed:
- `ice4_correct_negativities` - Ensure positive definiteness
- `precipitation_fraction_liquid_content` - Precipitation fraction
- Lookup table operations (KER_RACCS, KER_RDRYG, KER_SDRYG, GAMINC_RIM*)

## Process Stencil Details

### 1. ice4_nucleation (~150 lines)
**Purpose**: Heterogeneous ice nucleation (vapor → ice crystals)

**Key Physics**:
- Contact nucleation (IN in contact with droplets)
- Immersion freezing (IN within droplets)
- Deposition nucleation (vapor directly to ice)
- Depends on: temperature, supersaturation, aerosol concentration

**Inputs**: th_t, rhodref, exn, ls_fact, t, rv_t, ci_t, ssi, pabs_t, ldcompute
**Outputs**: rvheni_mr (vapor consumed)

### 2. ice4_rrhong (~100 lines)
**Purpose**: Homogeneous freezing of rain (rain → graupel)

**Key Physics**:
- Instantaneous freezing at T < -40°C
- All rain freezes to graupel
- Pure phase change, no collection

**Inputs**: ldcompute, t, exn, lv_fact, ls_fact, th_t, rr_t
**Outputs**: rrhong_mr (rain frozen)

### 3. ice4_rimltc (~100 lines)
**Purpose**: Ice crystal melting (ice → cloud liquid)

**Key Physics**:
- Occurs at T > 0°C
- Small crystals melt to droplets
- Absorbs latent heat (cooling)

**Inputs**: ldcompute, t, exn, lv_fact, ls_fact, th_t, ri_t
**Outputs**: rimltc_mr (ice melted)

### 4. ice4_warm (~800 lines) - LARGEST
**Purpose**: Warm microphysical processes

**Key Physics**:
- **Autoconversion**: Cloud → Rain (dr_auto)
- **Accretion**: Cloud + Rain → Rain (dr_accr)
- **Evaporation**: Rain → Vapor (dr_evap)
- Ventilation effects
- PDF-based subgrid scheme
- Collection efficiencies

**Inputs**: ldcompute, rhodref, lv_fact, t, th_t, pres, ka, dv, cj, hlc_*, rv_t, rc_t, rr_t, cf, rf, lbdar, lbdar_rf
**Outputs**: rcautr, rcaccr, rrevav

**Complexity**: Most complex stencil, handles:
- Multiple sub-processes
- Subgrid variability (PDF methods)
- Cloud fraction interactions
- Ventilation calculations
- Saturation adjustments

### 5. ice4_slow (~600 lines) - SECOND LARGEST
**Purpose**: Slow cold cloud processes

**Key Physics**:
- **Deposition**: Vapor → Snow/Ice (DEPT, DEPI)
- **Aggregation**: Ice → Snow (AGGS)
- **Autoconversion**: Ice → Snow (AUTS)
- **Bergeron-Findeisen**: Cloud → Ice (via vapor) (BERFI)
- **Homogeneous freezing**: Cloud → Ice (CFRZ)

**Inputs**: ldcompute, rhodref, t, ssi, lv_fact, ls_fact, rv_t, rc_t, ri_t, rs_t, rg_t, ai, cj, hli_hcf, hli_hri, lbdas, lbdag
**Outputs**: rc_honi_tnd, rv_deps_tnd, ri_aggs_tnd, ri_auts_tnd, rv_depg_tnd

**Complexity**:
- Multiple deposition calculations
- Aggregation with size distributions
- Temperature-dependent thresholds
- Subgrid cloud interactions

### 6. ice4_fast_rs (~400 lines)
**Purpose**: Fast snow processes

**Key Physics**:
- **Riming**: Cloud + Snow → Snow/Graupel (RIMS)
- **Accretion**: Rain + Snow → Snow/Graupel (ACCS)
- **Collection**: Various combinations
- **Melting**: Snow → Rain (SMLT)
- **Conversion**: Snow → Graupel threshold

**Inputs**: ldcompute, rhodref, lv_fact, ls_fact, pres, dv, ka, cj, t, rv_t, rc_t, rr_t, rs_t, lbdar, lbdar_rf
**Outputs**: 10+ tendency fields (rs_mltg_tnd, rc_mltsr_tnd, rs_rcrims_tnd, etc.)

**Uses**: Lookup tables (KER_RACCS, KER_RACCSS, KER_SACCRG, GAMINC_RIM*)

### 7. ice4_fast_rg (~400 lines)
**Purpose**: Fast graupel processes

**Key Physics**:
- **Wet growth**: T > 0°C, produces water coat
- **Dry growth**: T < 0°C, all freezes
- **Collection**: Cloud/Ice/Snow/Rain → Graupel
- **Melting**: Graupel → Rain
- **Shedding**: Excess water → Rain

**Inputs**: ldcompute, t, rhodref, pres, rv_t, rr_t, ri_t, rg_t, rc_t, rs_t, ci_t, ka, dv, cj, lbdar, lbdas, lbdag
**Outputs**: 8+ tendency fields (rg_rcdry_tnd, rg_ridry_tnd, rg_freez*_tnd, etc.)

**Uses**: Lookup tables (KER_SDRYG, KER_RDRYG)

### 8. ice4_fast_ri (~200 lines)
**Purpose**: Fast ice crystal processes

**Key Physics**:
- **Bergeron-Findeisen effect**: Enhanced growth in mixed-phase
- **Ice multiplication**: Rime splintering (Hallett-Mossop)

**Inputs**: ldcompute, rhodref, lv_fact, ls_fact, ai, cj, ci_t, ssi, rc_t, ri_t
**Outputs**: rc_beri_tnd

### 9. ice4_compute_pdf (~100 lines)
**Purpose**: PDF-based subgrid computations

**Key Physics**:
- Cloud fraction from PDF assumptions
- Liquid/ice content distribution
- High/low content separation
- Autoconversion variance

**Inputs**: ldcompute, rhodref, rc_t, ri_t, cf, t, sigma_rc, hlc_*, hli_*, fr
**Outputs**: Updated hlc_*, hli_* fields

## Translation Strategy

### Phase 1: Core Processes (Days 1-2)
1. ice4_nucleation
2. ice4_rrhong  
3. ice4_rimltc
4. ice4_compute_pdf

### Phase 2: Major Processes (Days 3-6)
1. ice4_warm (~2 days - most complex)
2. ice4_slow (~1.5 days)
3. ice4_fast_rs (~1 day)
4. ice4_fast_rg (~1 day)

### Phase 3: Supporting (Day 7)
1. ice4_fast_ri
2. ice4_correct_negativities
3. Lookup table utilities

### Phase 4: Component Wrapper (Day 8)
1. Ice4TendenciesJAX class
2. Orchestration logic
3. Integration testing

## JAX Translation Patterns

### Pattern 1: Conditional Process Activation
```python
# GT4Py
if __INLINED(LWARM):
    process_rate = compute_warm_process(...)

# JAX (static config)
if LWARM:
    process_rate = compute_warm_process(...)
else:
    process_rate = jnp.zeros_like(...)
```

### Pattern 2: Temperature Thresholds
```python
# GT4Py
if t > TT:
    rate = melting_formula(...)
else:
    rate = 0

# JAX
rate = jnp.where(t > TT, melting_formula(...), 0.0)
```

### Pattern 3: Lookup Tables
```python
# GT4Py
value = lookup_table[index]

# JAX
value = lookup_table[index.astype(int)]
# Or use jax.numpy interp/interpolation
```

### Pattern 4: Size Distribution Integration
```python
# GT4Py
integral = integrate_over_distribution(lambda_, params)

# JAX
# Precompute/vectorize
integral = vectorized_integration(lambda_, params)
```

## Testing Strategy

### Unit Tests for Each Process
```python
def test_ice4_nucleation():
    # Setup
    constants = phyex.to_externals()
    t = jnp.array([250, 260, 270])  # Below freezing
    ssi = jnp.array([0.1, 0.2, 0.3])  # Supersaturated
    
    # Execute
    rvheni_mr = ice4_nucleation(t, ssi, ..., constants)
    
    # Verify
    assert (rvheni_mr >= 0).all()  # Positive definite
    assert (rvheni_mr < 0.001).all()  # Reasonable magnitude
```

### Integration Test
```python
def test_ice4_tendencies_full():
    # Run full tendency calculation
    tendencies = ice4_tendencies_jax(state, constants)
    
    # Conservation checks
    total_water_before = sum(state.r_*)
    total_water_after = sum(state.r_* + tendencies * dt)
    assert jnp.isclose(total_water_before, total_water_after)
    
    # Energy check
    energy_before = compute_energy(state)
    energy_after = compute_energy(apply_tendencies(state, tendencies, dt))
    assert jnp.isclose(energy_before, energy_after, atol=1e-6)
```

## Dependencies

### From Phyex
All physical constants and parameters already available via `phyex.to_externals()`:
- Thermodynamic constants (CPD, CPV, CL, CI, RD, RV, etc.)
- Microphysical parameters (CRIAUTC, CRIAUTI, collection efficiencies, etc.)
- Size distribution parameters (LBR, LBS, LBG, LBEXR, etc.)

### Lookup Tables
Need to provide as JAX arrays:
- `KER_RACCS`: Rain-snow collection efficiency (12x10 table)
- `KER_RACCSS`: Rain-snow accretion (12x10 table)
- `KER_SACCRG`: Snow-graupel accretion (20x20 table)
- `KER_RDRYG`: Rain-graupel dry growth (31 entries)
- `KER_SDRYG`: Snow-graupel dry growth (31 entries)
- `GAMINC_RIM1/2/4`: Incomplete gamma functions

These are already available in:
- `src/ice3/phyex_common/xker_raccs.py`
- `src/ice3/phyex_common/xker_rdryg.py`
- `src/ice3/phyex_common/xker_sdryg.py`

## Current Status Summary

### ✅ Complete
- Foundation (ice_adjust): 100%
- Rain-ice helpers: 100%
- Ice4 tendencies utilities: 100%

### 🔄 In Progress
- Ice4 process stencils: 0%

### 📊 Overall Progress
- **Lines translated**: ~1,500 / ~6,000 (25%)
- **Components complete**: 2 / 3 major components
- **Estimated remaining**: 6-8 days of focused work

## Next Steps

1. **Immediate**: Translate ice4_nucleation (simplest process, ~150 lines)
2. **Short-term**: Complete Phase 1 processes (nucleation family)
3. **Medium-term**: Tackle ice4_warm and ice4_slow (bulk of work)
4. **Long-term**: Complete integration and testing

## File Locations

```
src/ice3/jax/
├── stencils/
│   ├── ice_adjust.py ✓
│   ├── rain_ice.py ✓
│   ├── ice4_tendencies.py ✓ (utilities only)
│   ├── ice4_nucleation.py [ ] TODO
│   ├── ice4_rrhong.py [ ] TODO
│   ├── ice4_rimltc.py [ ] TODO
│   ├── ice4_warm.py [ ] TODO
│   ├── ice4_slow.py [ ] TODO
│   ├── ice4_fast_rs.py [ ] TODO
│   ├── ice4_fast_rg.py [ ] TODO
│   ├── ice4_fast_ri.py [ ] TODO
│   ├── ice4_compute_pdf.py [ ] TODO
│   └── ice4_correct_negativities.py [ ] TODO
├── components/
│   ├── ice_adjust.py ✓
│   ├── ice4_tendencies.py [ ] TODO
│   └── rain_ice.py [ ] TODO
└── ICE4_TENDENCIES_STATUS.md ✓ (this file)
```

## Conclusion

The ice4_tendencies utility infrastructure is complete and ready. The remaining work focuses on translating the 9 major process stencils (~2,850 lines). Each process stencil is self-contained and can be translated independently, making this a parallelizable task for future development.
