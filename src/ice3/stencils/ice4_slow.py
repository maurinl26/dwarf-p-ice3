# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp

# "PHYEX/src/common/micro/mode_ice4_slow.F90"
def ice4_slow(
    ldcompute: Field["bool"],
    rhodref: Field["float"],
    t: Field["float"],
    ssi: Field["float"],
    rvt: Field["float"],
    rct: Field["float"],
    rit: Field["float"],
    rst: Field["float"],
    rgt: Field["float"],
    lbdas: Field["float"],
    lbdag: Field["float"],
    ai: Field["float"],
    cj: Field["float"],
    hli_hcf: Field["float"],
    hli_hri: Field["float"],
    rc_honi_tnd: Field["float"],
    rv_deps_tnd: Field["float"],
    ri_aggs_tnd: Field["float"],
    ri_auts_tnd: Field["float"],
    rv_depg_tnd: Field["float"],
    ldsoft: "bool",
):
    """
    Compute slow cold microphysical processes for ICE4 scheme.
    
    This stencil computes five slow (temperature-dependent) microphysical
    processes that occur over longer timescales than collision processes:
    homogeneous nucleation, vapor deposition on snow and graupel, ice crystal
    aggregation, and ice-to-snow autoconversion. These processes are
    "slow" because they involve phase changes or growth by diffusion rather
    than fast collision-coalescence.
    
    Parameters
    ----------
    ldcompute : Field[bool]
        Computation mask to activate processes (True = compute, False = skip).
        Used to optimize performance by skipping columns without clouds.
    rhodref : Field[float]
        Reference air density (kg/m³). Input field.
    t : Field[float]
        Temperature (K). Input field.
    ssi : Field[float]
        Supersaturation with respect to ice, S_i = (r_v - r_v,sat,i)/r_v,sat,i
        (dimensionless). Input field.
    rvt : Field[float]
        Water vapor mixing ratio (kg/kg). Input field.
    rct : Field[float]
        Cloud droplet mixing ratio (kg/kg). Input field.
    rit : Field[float]
        Ice crystal mixing ratio (kg/kg). Input field.
    rst : Field[float]
        Snow/aggregate mixing ratio (kg/kg). Input field.
    rgt : Field[float]
        Graupel mixing ratio (kg/kg). Input field.
    lbdas : Field[float]
        Snow size distribution slope parameter λ_s (m⁻¹). Input field.
    lbdag : Field[float]
        Graupel size distribution slope parameter λ_g (m⁻¹). Input field.
    ai : Field[float]
        Thermodynamic diffusion coefficient A_i (s·m⁻²). Input field.
        Accounts for heat and vapor diffusion limitations.
    cj : Field[float]
        Ventilation coefficient C_j (m^((1-D)/2)). Input field.
        Accounts for particle motion enhancement of diffusion.
    hli_hcf : Field[float]
        Cloud fraction for subgrid ice autoconversion (0-1). Input field.
        From subgrid condensation scheme.
    hli_hri : Field[float]
        Ice mixing ratio for subgrid autoconversion (kg/kg). Input field.
        From subgrid condensation scheme.
    rc_honi_tnd : Field[float]
        Homogeneous nucleation tendency: r_c → r_i (kg/kg/s). Output field.
    rv_deps_tnd : Field[float]
        Vapor deposition on snow tendency: r_v → r_s (kg/kg/s). Output field.
    ri_aggs_tnd : Field[float]
        Ice aggregation to snow tendency: r_i → r_s (kg/kg/s). Output field.
    ri_auts_tnd : Field[float]
        Ice autoconversion to snow tendency: r_i → r_s (kg/kg/s). Output field.
    rv_depg_tnd : Field[float]
        Vapor deposition on graupel tendency: r_v → r_g (kg/kg/s). Output field.
    ldsoft : bool
        Soft threshold mode (scalar). If True, sets all tendencies to zero.
        Used for special diagnostic modes.
    
    Returns
    -------
    None
        Modifies tendency fields in place.
    
    Notes
    -----
    **Process 1: Homogeneous Nucleation (RCHONI)**
    
    Condition: T < -35°C and r_c > threshold
    
    At very cold temperatures, cloud droplets freeze spontaneously:
    
    dr_c/dt = -HON × ρ × r_c × exp(α₃×(T - T_t) - β₃)
    
    where:
    - HON: Homogeneous nucleation rate constant (m³·kg⁻¹·s⁻¹)
    - α₃, β₃: Temperature coefficients
    - Rate limited to max 1000 kg/kg/s for numerical stability
    
    Physical basis: Fletcher's nucleation theory, all droplets freeze
    below approximately -38°C in nature.
    
    **Process 2: Vapor Deposition on Snow (RVDEPS)**
    
    Condition: r_v > threshold, r_s > threshold, supersaturated over ice
    
    Water vapor deposits directly onto snow crystals:
    
    dr_v/dt = -(S_i / (ρ × A_i)) × [O0DEPS × λ_s^EX0DEPS + O1DEPS × C_j × λ_s^EX1DEPS]
    
    Two terms:
    - First term (O0DEPS): Non-ventilated diffusion growth
    - Second term (O1DEPS): Ventilated growth (enhanced by falling)
    
    where:
    - S_i: Ice supersaturation
    - A_i: Thermodynamic diffusion coefficient
    - λ_s: Snow slope parameter (∝ concentration/size)
    - C_j: Ventilation factor
    - EX0DEPS, EX1DEPS: Size distribution exponents
    
    **Process 3: Ice Aggregation to Snow (RIAGGS)**
    
    Condition: r_i > threshold, r_s > threshold
    
    Ice crystals collide and stick to form larger snow aggregates:
    
    dr_i/dt = -F_IAGGS × exp(COLEXIS × (T - T_t)) × r_i × λ_s^EXIAGGS × ρ^(-CEXVT)
    
    where:
    - F_IAGGS: Aggregation efficiency pre-factor
    - COLEXIS: Temperature factor (sticking efficiency increases near 0°C)
    - EXIAGGS: Size distribution exponent
    - ρ^(-CEXVT): Air density correction for fall speed
    
    Physical basis: Collision-coalescence between ice crystals and snow,
    temperature-dependent sticking efficiency (dendrites stick better).
    
    **Process 4: Ice Autoconversion to Snow (RIAUTS)**
    
    Condition: hli_hri > threshold (subgrid ice amount)
    
    Ice crystals grow large enough to be reclassified as snow:
    
    dr_i/dt = -(T_IMAUTI × exp(T_EXAUTI × (T - T_t))) × max(0, hli_hri - r_crit × hli_hcf)
    
    where:
    - T_IMAUTI: Time constant at T_t (s⁻¹)
    - T_EXAUTI: Temperature factor (K⁻¹)
    - r_crit(T): Temperature-dependent threshold
      r_crit = min(CRIAUTI, 10^(ACRIAUTI×(T-T_t) + BCRIAUTI))
    - hli_hri: Subgrid ice mixing ratio
    - hli_hcf: Subgrid cloud fraction
    
    Threshold increases with warming (more ice needed at warmer T).
    Uses subgrid values to account for cloud heterogeneity.
    
    **Process 5: Vapor Deposition on Graupel (RVDEPG)**
    
    Condition: r_v > threshold, r_g > threshold, supersaturated over ice
    
    Similar to snow deposition but for graupel:
    
    dr_v/dt = -(S_i / (ρ × A_i)) × [O0DEPG × λ_g^EX0DEPG + O1DEPG × C_j × λ_g^EX1DEPG]
    
    Same form as RVDEPS but with graupel parameters (O0DEPG, etc.).
    Graupel has higher terminal velocities → stronger ventilation effects.
    
    **Threshold Logic:**
    
    All processes check minimum mixing ratios (V_RTMIN, C_RTMIN, etc.)
    to prevent:
    - Numerical underflow
    - Spurious tendencies from trace amounts
    - Division by zero in rate calculations
    
    Typical thresholds: 10⁻²⁰ kg/kg for vapor/cloud, 10⁻¹⁵ kg/kg for snow/graupel
    
    **ldsoft Mode:**
    
    When ldsoft=True, all tendencies set to zero regardless of conditions.
    Used for:
    - Sensitivity tests
    - Process isolation experiments
    - Diagnostic runs
    
    **Time Scales:**
    
    These are "slow" processes compared to collision processes:
    - Homogeneous nucleation: τ ~ 1-10 s (very fast once initiated)
    - Deposition: τ ~ 100-1000 s (diffusion-limited)
    - Aggregation: τ ~ 100-1000 s (collection + temperature dependent)
    - Autoconversion: τ ~ 100-1000 s (size threshold based)
    
    **Physical Interpretation:**
    
    This stencil handles the "background" ice processes:
    - Nucleation: Creates initial ice at cold temperatures
    - Deposition: Grows existing ice/snow by vapor uptake
    - Aggregation: Builds larger particles from crystals
    - Autoconversion: Transitions ice to snow category
    
    These processes set up the ice phase structure that fast processes
    (riming, accretion) then modify through collisions.
    
    **Ventilation Effects:**
    
    Both deposition processes include ventilation terms (C_j):
    - Stationary particles: diffusion only (O0 term)
    - Falling particles: enhanced diffusion (O1 × C_j term)
    - Enhancement factor ∝ √(Re) from boundary layer thinning
    
    Source Reference
    ----------------
    PHYEX/src/common/micro/mode_ice4_slow.F90
    
    See Also
    --------
    ice4_fast_rg : Fast riming processes (rain-graupel)
    ice4_fast_rs : Fast snow processes
    ice4_warm : Warm rain processes
    ice4_stepping : Time integration of all tendencies
    
    Examples
    --------
    >>> # Cold cloud with ice and snow
    >>> # T = -20°C, r_i = 1e-4 kg/kg, r_s = 5e-4 kg/kg
    >>> # → Aggregation active (RIAGGS > 0)
    >>> # → Deposition active if supersaturated (RVDEPS > 0)
    >>> # → Autoconversion possible if r_i exceeds threshold
    
    >>> # Very cold cloud with supercooled droplets
    >>> # T = -40°C, r_c = 1e-4 kg/kg
    >>> # → Homogeneous nucleation active (RCHONI ~ 10-100 kg/kg/s)
    >>> # → All droplets freeze rapidly
    """

    from __externals__ import (ACRIAUTI, ALPHA3, BCRIAUTI, BETA3, C_RTMIN,
                               CEXVT, COLEXIS, CRIAUTI, EX0DEPG, EX0DEPS,
                               EX1DEPG, EX1DEPS, EXIAGGS, FIAGGS, G_RTMIN, HON,
                               I_RTMIN, O0DEPG, O0DEPS, O1DEPG, O1DEPS,
                               S_RTMIN, TEXAUTI, TIMAUTI, TT, V_RTMIN)

    # 3.2 compute the homogeneous nucleation source : RCHONI
    with computation(PARALLEL), interval(...):
        if t < TT - 35.0 and rct > C_RTMIN and ldcompute:
            if not ldsoft:
                rc_honi_tnd = min(1000, HON * rhodref * rct * exp(ALPHA3 * (t - TT) - BETA3))
        else:
            rc_honi_tnd = 0

    # 3.4 compute the deposition, aggregation and autoconversion sources
    # 3.4.3 compute the deposition on r_s : RVDEPS
    with computation(PARALLEL), interval(...):
        if rvt > V_RTMIN and rst > S_RTMIN and ldcompute:
            # Translation note : #ifdef REPRO48 l118 to 120 kept
            # Translation note : #else REPRO48  l121 to 126 omitted
            if not ldsoft:
                rv_deps_tnd = (ssi / (rhodref * ai)) * (O0DEPS * lbdas**EX0DEPS + O1DEPS * cj * lbdas**EX1DEPS)
        else:
            rv_deps_tnd = 0

    # 3.4.4 compute the aggregation on r_s: RIAGGS
    with computation(PARALLEL), interval(...):
        if rit > I_RTMIN and rst > S_RTMIN and ldcompute:
            # Translation note : #ifdef REPRO48 l138 to 142 kept
            # Translation note : #else REPRO48 l143 to 150 omitted
            if not ldsoft:
                ri_aggs_tnd = FIAGGS * exp(COLEXIS * (t - TT)) * rit * lbdas**EXIAGGS * rhodref ** (-CEXVT)
        # Translation note : OELEC = False l151 omitted
        else:
            ri_aggs_tnd = 0

    # 3.4.5 compute the autoconversion of r_i for r_s production: RIAUTS
    with computation(PARALLEL), interval(...):
        if hli_hri > I_RTMIN and ldcompute:
            if not ldsoft:
                criauti_tmp = min(CRIAUTI, 10 ** (ACRIAUTI * (t - TT) + BCRIAUTI))
                ri_auts_tnd = (
                    TIMAUTI
                    * exp(TEXAUTI * (t - TT))
                    * max(0, hli_hri - criauti_tmp * hli_hcf)
                )

        else:
            ri_auts_tnd = 0

    # 3.4.6 compute the depsoition on r_g: RVDEPG
    with computation(PARALLEL), interval(...):
        if rvt > V_RTMIN and rgt > G_RTMIN and ldcompute:
            if not ldsoft:
                rv_depg_tnd = (ssi / (rhodref * ai)) * (O0DEPG * lbdag**EX0DEPG + O1DEPG * cj * lbdag**EX1DEPG)
        else:
            rv_depg_tnd = 0
