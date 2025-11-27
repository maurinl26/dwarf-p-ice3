# Stencil File Documentation Template

This template provides a standardized format for documenting stencil files in `src/ice3/stencils/`.

## Module-Level Docstring Template

```python
# -*- coding: utf-8 -*-
"""
[Brief title describing the stencil's purpose]

This module implements [describe what microphysical processes or computations
are handled]. [Add 1-2 sentences about the scientific context or role in 
the overall scheme].

Source: PHYEX/src/common/micro/[original Fortran file name]
"""
from __future__ import annotations

# imports...
```

## Function-Level Docstring Template

```python
def function_name(
    param1: Field["float"],
    param2: Field["float"],
    # ... more parameters
):
    """
    [One-line summary of what the function does]
    
    [2-3 sentences providing more detail about the physical process,
    algorithm, or computation being performed. Include scientific context
    where relevant.]
    
    Parameters
    ----------
    param1 : Field[float]
        Description of parameter including physical meaning and units.
        E.g., "Temperature (K)" or "Mixing ratio (kg/kg)".
    param2 : Field[float]
        Description with units.
    output_param : Field[float]
        Output: Description of what is computed/updated.
        
    Returns
    -------
    Field[float] (if applicable)
        Description of return value.
        
    Notes
    -----
    [Optional section for:]
    - Algorithm overview or steps
    - Physical interpretation
    - Key equations or formulas
    - Assumptions or limitations
    - Related processes
    - Special cases handled
    
    [If relevant, add subsections like:]
    
    Process Overview:
    1. Step one
    2. Step two
    3. Step three
    
    Physical Basis:
    - Key physics principle
    - Governing equations
    
    External Parameters:
    - PARAM_NAME: Description (units)
    - CONSTANT_NAME: Description (value/units)
    
    Examples
    --------
    [Optional: Show typical values or usage]
    
    References
    ----------
    [Optional: Cite scientific papers if relevant]
    Author, Year: Title. Journal.
    
    Source Reference:
    PHYEX/src/common/micro/[file.F90], lines [start-end]
    """
    # Implementation...
```

## Guidelines for Good Documentation

### 1. Physical Units
Always specify units for physical quantities:
- Temperature: K
- Pressure: Pa or hPa
- Mixing ratios: kg/kg
- Density: kg/m³
- Length scales: m
- Time: s
- Rates/tendencies: (quantity)/s

### 2. Process Descriptions
For microphysical processes, explain:
- **What**: What physical process (e.g., "rain evaporation")
- **When**: Conditions for occurrence (e.g., "T > 273.15 K", "r_r > R_RTMIN")
- **How**: Mechanism (e.g., "droplets fall through subsaturated air")
- **Energy**: Latent heat effects (L_v, L_s, L_f)

### 3. Parameter Naming Conventions
Be consistent with naming:
- `t`: Temperature (K)
- `th` or `tht`: Potential temperature (K)
- `rv`, `rc`, `rr`, `ri`, `rs`, `rg`: Vapor, cloud, rain, ice, snow, graupel mixing ratios (kg/kg)
- `rhodref`: Reference density (kg/m³)
- `pres`: Pressure (Pa)
- `exn`: Exner function (dimensionless)
- `lsfact`, `lvfact`: Latent heat factors L_s/c_p, L_v/c_p (K)
- `lbdar`, `lbdas`, `lbdag`: Slope parameters for size distributions (m⁻¹)
- `_tnd`: Tendency (rate of change per second)
- `_mr`: Mixing ratio increment (kg/kg)

### 4. External Parameters
Document external compile-time constants:
```python
External Parameters:
- TT: Triple point temperature (K), typically 273.15
- LVTT: Latent heat of vaporization at triple point (J/kg)
- LSTT: Latent heat of sublimation at triple point (J/kg)
- R_RTMIN: Minimum rain mixing ratio threshold (kg/kg)
- LBEXR: Exponent for rain distribution slope
```

### 5. Translation Notes
Include helpful comments about Fortran→Python translation:
```python
# Translation note: l100 to l150 kept (AROME configuration)
# Translation note: l151 to l200 omitted (LLRFR = False)
# Translation note: #ifdef REPRO48 version retained
```

### 6. Scientific References
When appropriate, cite key papers:
```python
References
----------
Cohard, J.-M., and J.-P. Pinty, 2000: A comprehensive two-moment
warm microphysical bulk scheme. Q. J. R. Meteorol. Soc., 126, 1815-1842.
```

## Common Stencil Patterns

### Initialization Stencil
```python
def init_fields(...):
    """
    Initialize [what fields] to [what values].
    
    [Why initialization is needed - e.g., "Resets accumulators before
    microphysics calculations" or "Sets boundary conditions"]
    """
```

### Process Rate Stencil
```python
def compute_process_rate(...):
    """
    Compute [process name] rate.
    
    Calculates the rate of [physical process] based on [conditions].
    [Physical interpretation - what happens and why].
    
    Parameters
    ----------
    [inputs with units]
    
    output : Field[float]
        Output: Process rate (kg/kg/s).
        
    Notes
    -----
    Process occurs when:
    - Condition 1  
    - Condition 2
    
    Rate depends on:
    - Factor 1
    - Factor 2
    
    Formula: rate = [mathematical expression if simple]
    ```

### Post-Processing Stencil
```python
def apply_changes(...):
    """
    Apply [what changes] to prognostic variables.
    
    Updates [which variables] after [which process] with proper
    accounting for [thermodynamics/mass conservation].
    
    Notes
    -----
    Process: r_x → r_y (describe phase change)
    Energy: Releases/absorbs [which latent heat]
    Temperature change: ΔT = [formula]
    """
```

### Aggregation Stencil
```python
def aggregate_tendencies(...):
    """
    Aggregate contributions from multiple processes.
    
    Sums [what] from [which processes] to compute [what result].
    Ensures [conservation law] is maintained.
    """
```

## Checklist for Complete Documentation

- [ ] Module docstring with title, description, source reference
- [ ] All function parameters documented with types and units
- [ ] Physical interpretation provided for key processes
- [ ] Algorithm steps outlined where relevant
- [ ] External parameters documented
- [ ] Conditions/thresholds explained
- [ ] Energy/mass conservation noted where applicable
- [ ] Source line references included
- [ ] Translation notes for Fortran→Python where helpful
- [ ] Special cases or edge cases documented

## Example: Complete Function Documentation

```python
def ice4_rain_evaporation(
    t: Field["float"],
    rhodref: Field["float"],
    pres: Field["float"],
    rvt: Field["float"],
    rrt: Field["float"],
    ka: Field["float"],
    dv: Field["float"],
    cj: Field["float"],
    lbdar: Field["float"],
    rrevav: Field["float"],
    ldsoft: "bool",
):
    """
    Compute rain evaporation rate below cloud base.
    
    Calculates the rate at which rain droplets evaporate when falling
    through subsaturated air. The evaporation cools the air and moistens
    it, which can trigger convection or influence cloud development.
    
    Parameters
    ----------
    t : Field[float]
        Temperature (K).
    rhodref : Field[float]
        Reference air density (kg/m³).
    pres : Field[float]
        Pressure (Pa).
    rvt : Field[float]
        Water vapor mixing ratio (kg/kg).
    rrt : Field[float]
        Rain mixing ratio (kg/kg).
    ka : Field[float]
        Thermal conductivity of air (W/(m·K)).
    dv : Field[float]
        Diffusivity of water vapor in air (m²/s).
    cj : Field[float]
        Ventilation coefficient (dimensionless).
    lbdar : Field[float]
        Rain drop size distribution slope parameter (m⁻¹).
    rrevav : Field[float]
        Output: Rain evaporation rate (kg/kg/s). Negative values.
    ldsoft : bool
        If True, use previously computed value without recalculation.
        
    Notes
    -----
    Physical Process:
    Rain droplets falling through subsaturated air (RH < 100%) evaporate,
    absorbing latent heat from the environment and cooling the air. This
    process is most important in:
    - Convective downdrafts
    - Below cloud base in stratiform precipitation
    - Dry subcloud layers
    
    The evaporation rate depends on:
    - Subsaturation: (e_sat - e), larger deficit → faster evaporation
    - Drop size distribution: via lambda parameter
    - Ventilation: falling drops enhance mass transfer (cj factor)
    - Air properties: ka (heat), dv (moisture diffusion)
    
    Formula:
    rate = -K × (e_sat(T) - e) × f(λ, cj) × ventilation_correction
    
    External Parameters:
    - R_RTMIN: Minimum rain threshold (kg/kg), typically 1e-8
    - LEVLIMIT: Flag to limit saturation vapor pressure
    - ALPW, BETAW, GAMW: Saturation vapor pressure coefficients
    - EX0EVAP, EX1EVAP: Exponents for evaporation formula
    - O0EVAP, O1EVAP: Coefficients for evaporation formula
    
    The rate is computed only when:
    - rrt > R_RTMIN (sufficient rain present)
    - Air is subsaturated (checked internally)
    - ldsoft = False (recalculation mode)
    
    Source Reference:
    PHYEX/src/common/micro/mode_ice4_warm.F90, lines 250-290
    """
    from __externals__ import (ALPW, BETAW, EX0EVAP, EX1EVAP, GAMW, LEVLIMIT,
                               O0EVAP, O1EVAP, R_RTMIN, RV)
    
    with computation(PARALLEL), interval(...):
        if rrt > R_RTMIN and ldcompute:
            if not ldsoft:
                # Compute saturation vapor pressure
                esat = exp(ALPW - BETAW / t - GAMW * log(t))
                if LEVLIMIT:
                    esat = min(esat, 0.99 * pres)
                
                # Compute subsaturation
                e = rvt * pres / (EPSILO + rvt)
                subsaturation = esat - e
                
                if subsaturation > 0:  # Only evaporate if subsaturated
                    # Compute evaporation rate with ventilation
                    rrevav = (subsaturation / (rhodref * RV * t)) * (
                        ka * (O0EVAP * lbdar ** EX0EVAP + 
                              O1EVAP * cj * lbdar ** EX1EVAP)
                    )
                    rrevav = -min(rrt / TSTEP, rrevav)  # Don't evaporate more than exists
                else:
                    rrevav = 0
        else:
            rrevav = 0
```

## File-Specific Notes

### For remaining stencil files to document:

1. **ice4_stepping.py**: Time stepping and iteration control
2. **ice4_fast_rs.py**: Rain-snow fast processes (already read, 344 lines - complex)
3. **ice4_warm.py**: Warm cloud processes (autoconversion, accretion, evaporation)
4. **ice4_slow.py**: Slow depositional growth processes
5. **ice4_nucleation.py**: Ice nucleation parameterizations
6. **sedimentation.py**: Vertical transport of precipitation
7. **ice4_rimltc.py**: Ice crystal melting processes
8. **ice4_rrhong.py**: Homogeneous freezing of rain/hail
9. **cloud_fraction.py**: Cloud fraction diagnostics
10. **condensation.py**: Condensation/evaporation calculations
11. **ice4_compute_pdf.py**: PDF calculations for subgrid processes
12. **ice4_correct_negativities.py**: Numerical stability corrections
13. **ice4_nucleation_processing.py**: Nucleation post-processing
14. **precipitation_fraction_liquid_content.py**: Precipitation diagnostics

Priority order reflects scientific importance and usage frequency in the scheme.
