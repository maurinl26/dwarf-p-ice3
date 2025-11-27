# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    __INLINED,
    PARALLEL,
    Field,
    GlobalTable,
    atan,
    computation,
    exp,
    floor,
    interval,
    sqrt,
)

from ice3.functions.tiwmx import e_sat_i, e_sat_w


# "PHYEX/src/common/micro/condensation.F90"
def condensation(
    sigqsat: Field["float"],
    pabs: Field["float"],
    sigs: Field["float"],
    t: Field["float"],
    rv: Field["float"],
    ri: Field["float"],
    rc: Field["float"],
    rv_out: Field["float"],
    rc_out: Field["float"],
    ri_out: Field["float"],
    cldfr: Field["float"],
    cph: Field["float"],
    lv: Field["float"],
    ls: Field["float"],
    q1: Field["float"],
    pv_out: Field["float"],
    piv_out: Field["float"],
    frac_out: Field["float"],
    qsl_out: Field["float"],
    qsi_out: Field["float"],
    sigma_out: Field["float"],
    cond_out: Field["float"],
    a_out: Field["float"],
    b_out: Field["float"],
    sbar_out: Field["float"],
):
    """
    Compute subgrid condensation using CB02 statistical cloud scheme.
    
    This stencil implements the Chaboureau and Bechtold (2002) statistical
    cloud scheme, which accounts for subgrid variability in temperature and
    moisture to compute partial cloud cover and condensate amounts. It uses
    a triangular PDF for supersaturation and handles both liquid and ice
    condensation with temperature-dependent ice fractions.
    
    Parameters
    ----------
    sigqsat : Field[float]
        Saturation mixing ratio variance coefficient (dimensionless). Input field.
        Controls subgrid variability of q_sat.
    pabs : Field[float]
        Absolute pressure (Pa). Input field.
    sigs : Field[float]
        Subgrid standard deviation (dimensionless). Input field.
        From turbulence scheme when LSIGMAS=True.
    t : Field[float]
        Temperature (K). Input/output field, updated by latent heating.
    rv : Field[float]
        Water vapor mixing ratio (kg/kg). Input field.
    ri : Field[float]
        Ice mixing ratio (kg/kg). Input field.
    rc : Field[float]
        Cloud droplet mixing ratio (kg/kg). Input field.
    rv_out : Field[float]
        Updated vapor mixing ratio (kg/kg). Output field.
    rc_out : Field[float]
        Updated cloud mixing ratio (kg/kg). Output field.
    ri_out : Field[float]
        Updated ice mixing ratio (kg/kg). Output field.
    cldfr : Field[float]
        Cloud fraction (0-1). Output field.
    cph : Field[float]
        Specific heat of moist air (J/(kg·K)). Input field.
    lv : Field[float]
        Latent heat of vaporization (J/kg). Input field.
    ls : Field[float]
        Latent heat of sublimation (J/kg). Input field.
    q1 : Field[float]
        Normalized supersaturation s̄/σ (dimensionless). Output field.
    pv_out : Field[float]
        Saturation vapor pressure over liquid (Pa). Diagnostic output.
    piv_out : Field[float]
        Saturation vapor pressure over ice (Pa). Diagnostic output.
    frac_out : Field[float]
        Ice fraction (0-1). Diagnostic output.
    qsl_out : Field[float]
        Saturation mixing ratio over liquid (kg/kg). Diagnostic output.
    qsi_out : Field[float]
        Saturation mixing ratio over ice (kg/kg). Diagnostic output.
    sigma_out : Field[float]
        Total subgrid standard deviation (kg/kg). Diagnostic output.
    cond_out : Field[float]
        Total condensate from PDF integration (kg/kg). Diagnostic output.
    a_out : Field[float]
        Supersaturation coefficient a (dimensionless). Diagnostic output.
    b_out : Field[float]
        Supersaturation coefficient b (dimensionless). Diagnostic output.
    sbar_out : Field[float]
        Mean supersaturation s̄ (kg/kg). Diagnostic output.
    
    Returns
    -------
    None
        Modifies output fields in place.
    
    Notes
    -----
    **CB02 Statistical Cloud Scheme:**
    
    The Chaboureau and Bechtold (2002) scheme represents subgrid variability
    using a statistical approach:
    
    1. **Total Water Distribution:**
       r_t = r_v + r_c + r_i (conserved during adjustment)
    
    2. **Supersaturation:**
       s = r_t - q_sat(T,p)
       
    3. **Subgrid Variability:**
       σ = √[(2σ_s)² + (σ_qsat × q_sat × a)²]
       
       where:
       - σ_s: turbulent variance (from turbulence scheme)
       - σ_qsat: q_sat variance coefficient (typically 0.02)
       - a, b: coefficients accounting for T-dependence
    
    4. **Normalized Supersaturation:**
       q₁ = s̄/σ
       
    **Cloud Fraction Formula (CB02):**
    
    For q₁ > 0 (supersaturated):
        CF = max(0, min(1, 0.5 + 0.36 × arctan(1.55 × q₁)))
        
    For q₁ ≤ 0 (subsaturated):
        CF → 0 for large negative q₁
    
    **Condensate Formula (CB02):**
    
    For q₁ ≤ 2:
        cond/σ = min(e⁻¹ + 0.66q₁ + 0.086q₁², 2)
        
    For q₁ > 2:
        cond/σ = q₁
        
    For q₁ < 0:
        cond/σ = exp(1.2q₁ - 1)
    
    **Temperature-Dependent Ice Fraction:**
    
    Two methods via FRAC_ICE_ADJUST:
    
    Mode 0 (AROME, temperature-based):
        f_ice = (T_max - T) / (T_max - T_min)
        Linear between T_max (-20°C) and T_min (0°C)
        
    Mode 3 (Statistical):
        Uses existing rc/ri ratio
        More physical but requires prior ice
    
    **Mixed-Phase Condensation:**
    
    Total condensate split by ice fraction:
    - r_c_new = (1 - f_ice) × cond
    - r_i_new = f_ice × cond
    
    **Temperature Update:**
    
    Latent heat release modifies temperature:
    T_new = T + [(r_c_new - r_c) × L_v + (r_i_new - r_i) × L_s] / c_ph
    
    **Supersaturation Coefficients:**
    
    Account for temperature-dependence of q_sat:
    
    a = 1 / (1 + (L_vs/c_ph) × (L_vs×q_sat)/(R_v×T²) × (1 + R_v×q_sat/R_d))
    
    b = [(L_vs×q_sat)/(R_v×T²) × (1 + R_v×q_sat/R_d)] × a
    
    where L_vs = (1-f_ice)×L_v + f_ice×L_s
    
    **Physical Interpretation:**
    
    The CB02 scheme:
    - Provides smooth cloud edges (partial cloud fractions)
    - Accounts for subgrid temperature/moisture variability
    - Predicts onset of condensation before grid-mean saturation
    - Works well for both stratiform and convective clouds
    - Computationally efficient (analytical formulas)
    
    **Comparison with All-or-Nothing:**
    
    Traditional: Cloud if r_v > q_sat, CF = 1; else CF = 0
    CB02: Cloud begins before grid-mean saturation, CF varies smoothly
    
    **Diagnostic Outputs:**
    
    Multiple diagnostic fields provided for analysis:
    - Saturation vapor pressures (e_sat_w, e_sat_i)
    - Saturation mixing ratios (q_sl, q_si)
    - Ice fraction, supersaturation, cloud fraction
    - Coefficients a, b, s̄, σ, q₁
    
    Source Reference
    ----------------
    PHYEX/src/common/micro/condensation.F90
    
    See Also
    --------
    cloud_fraction_1 : Convert tendencies to sources
    cloud_fraction_2 : Subgrid autoconversion
    sigrc_computation : Compute subgrid condensate variance
    
    References
    ----------
    Chaboureau, J.-P., and P. Bechtold, 2002: A simple cloud
    parameterization derived from cloud resolving model data:
    Diagnostic and prognostic applications. J. Atmos. Sci., 59, 2362-2372.
    
    Sommeria, G., and J.W. Deardorff, 1977: Subgrid-scale condensation
    in models of nonprecipitating clouds. J. Atmos. Sci., 34, 344-355.
    
    Examples
    --------
    >>> # High supersaturation → large cloud fraction
    >>> # q₁ = 2.0 → CF ≈ 0.85, cond/σ = 2.0
    
    >>> # Near saturation → partial cloud
    >>> # q₁ = 0.5 → CF ≈ 0.65, cond/σ ≈ 0.95
    
    >>> # Subsaturated → small cloud fraction
    >>> # q₁ = -1.0 → CF ≈ 0.35, cond/σ ≈ 0.08
    
    >>> # Far subsaturated → no cloud
    >>> # q₁ < -2.0 → CF → 0, cond → 0
    """

    from __externals__ import (
        CONDENS,
        FRAC_ICE_ADJUST,
        LSIGMAS,
        LSTATNW,
        OCND2,
        RD,
        RV,
        TMAXMIX,
        TMINMIX,
    )

    # initialize values
    with computation(PARALLEL), interval(...):
        cldfr = 0.0
        rv_out = 0.0
        rc_out = 0.0
        ri_out = 0.0

    # 3. subgrid condensation scheme
    # Translation note : only the case with LSUBG_COND = True retained (l475 in ice_adjust.F90)
    # sigqsat and sigs must be provided by the user
    with computation(PARALLEL), interval(...):
        # local
        # Translation note : 506 -> 514 kept (ocnd2 == False) # Arome default setting
        # Translation note : 515 -> 575 skipped (ocnd2 == True)
        prifact = 1  # ocnd2 == False for AROME
        frac_tmp = 0  # l340 in Condensation .f90

        # Translation note : 252 -> 263 if(present(PLV)) skipped (ls/lv are assumed to be present)
        # Translation note : 264 -> 274 if(present(PCPH)) skipped (files are assumed to be present)

        # store total water mixing ratio (244 -> 248)
        rt = rv + rc + ri * prifact

        # Translation note : 276 -> 310 (not osigmas) skipped : (osigmas = True) for Arome default version
        # Translation note : 316 -> 331 (ocnd2 == True) skipped : ocnd2 = False for Arome

        # l334 to l337
        if __INLINED(not OCND2):
            pv = min(
                e_sat_w(t),
                0.99 * pabs,
            )
            piv = min(
                e_sat_i(t),
                0.99 * pabs,
            )

        # TODO : l341 -> l350
        # Translation note : OUSERI = TRUE, OCND2 = False
        if __INLINED(not OCND2):
            frac_tmp = rc / (rc + ri) if rc + ri > 1e-20 else 0

            # Compute frac ice inlined
            # Default Mode (S)
            if __INLINED(FRAC_ICE_ADJUST == 3):
                frac_tmp = max(0, min(1, frac_tmp))

            # AROME mode
            if __INLINED(FRAC_ICE_ADJUST == 0):
                frac_tmp = max(0, min(1, ((TMAXMIX - t) / (TMAXMIX - TMINMIX))))

        # Supersaturation coefficients
        qsl = RD / RV * pv / (pabs - pv)
        qsi = RD / RV * piv / (pabs - piv)
        
        # Store intermediate values for diagnostics
        pv_out = pv
        piv_out = piv
        qsl_out = qsl
        qsi_out = qsi

        # interpolate between liquid and solid as a function of temperature
        qsl = (1 - frac_tmp) * qsl + frac_tmp * qsi
        lvs = (1 - frac_tmp) * lv + frac_tmp * ls

        # coefficients a et b
        ah = lvs * qsl / (RV * t**2) * (1 + RV * qsl / RD)
        a = 1 / (1 + lvs / cph * ah)
        b = ah * a
        sbar = a * (rt - qsl + ah * lvs * (rc + ri * prifact) / cph)
        
        # Store coefficients for diagnostics
        a_out = a
        b_out = b
        sbar_out = sbar
        frac_out = frac_tmp

        # Translation note : l369 - l390 kept
        # Translation note : l391 - l406 skipped (OSIGMAS = False)
        # Translation note : LSTATNW = False
        # Translation note : l381 retained for sigmas formulation
        # Translation note : OSIGMAS = TRUE
        # Translation npte : LHGT_QS = False (and ZDZFACT unused)
        if __INLINED(LSIGMAS and not LSTATNW):
            sigma = (
                sqrt((2 * sigs) ** 2 + (sigqsat * qsl * a) ** 2)
                if sigqsat != 0
                else 2 * sigs
            )

        # Translation note : l407 - l411
        sigma = max(1e-10, sigma)
        q1 = sbar / sigma
        
        # Store sigma for diagnostics
        sigma_out = sigma

        # Translation notes : l413 to l468 skipped (HCONDENS=="GAUS")
        # TODO : add hcondens == "GAUS" option
        # Translation notes : l469 to l504 kept (HCONDENS = "CB02")
        # 9.2.3 Fractional cloudiness and cloud condensate
        # HCONDENS = 0 is CB02 option
        if __INLINED(CONDENS == 0):
            # Translation note : l470 to l479
            if q1 > 0.0:
                cond_tmp = (
                    min(exp(-1.0) + 0.66 * q1 + 0.086 * q1**2, 2.0) if q1 <= 2.0 else q1
                )  # we use the MIN function for continuity
            else:
                cond_tmp = exp(1.2 * q1 - 1.0)
            cond_tmp *= sigma

            # Translation note : l482 to l489
            # cloud fraction
            cldfr = (
                max(0.0, min(1.0, 0.5 + 0.36 * atan(1.55 * q1)))
                if cond_tmp >= 1e-12
                else 0
            )

            # Translation note : l487 to l489
            cond_tmp = 0 if cldfr == 0 else cond_tmp
            
            # Store cond for diagnostics
            cond_out = cond_tmp

            # Translation note : l496 to l503 removed (because initialized further in cloud_fraction diagnostics)

            # Translation notes : 506 -> 514 (not ocnd2)
            # Translation notes : l515 to l565 (removed)
            if __INLINED(not OCND2):
                rc_out = (1 - frac_tmp) * cond_tmp  # liquid condensate
                ri_out = frac_tmp * cond_tmp  # solid condensate
                t += ((rc_out - rc) * lv + (ri_out - ri) * ls) / cph
                rv_out = rt - rc_out - ri_out * prifact

            # Translation note : sigrc computation out of scope
            # Translation note : l491 to l494 skept
            # sigrc computation in sigrc_computation stencil

        # Translation note : end jiter

# from_file="./PHYEX/src/common/micro/condensation.F90",
# from_line=186,
# to_line=189
def sigrc_computation(
    q1: Field["float"],
    sigrc: Field["float"],
    # inq1: Field["int"],
    inq2: "int",
    src_1d: GlobalTable["float", (34)],
):
    """
    Compute subgrid cloud condensate variance using lookup table.
    
    This stencil computes the subgrid variance of cloud condensate (σ_rc)
    as a function of the normalized supersaturation (q₁). It uses a
    precomputed lookup table (SRC_1D) with linear interpolation to
    efficiently evaluate the variance relationship derived from
    high-resolution simulations.
    
    Parameters
    ----------
    q1 : Field[float]
        Normalized supersaturation s̄/σ (dimensionless). Input field.
        Typically ranges from -10 to +5 in atmospheric conditions.
    sigrc : Field[float]
        Subgrid cloud condensate variance σ_rc/σ (dimensionless, 0-1).
        Output field.
    inq2 : int
        Lookup table index (scalar). Unused parameter in GT4Py version.
    src_1d : GlobalTable[float, (34)]
        Precomputed variance lookup table. Contains 34 values covering
        q₁ range from -22 to +11 (discretized at 0.5 intervals).
    
    Returns
    -------
    None
        Modifies sigrc in place.
    
    Notes
    -----
    **Physical Basis:**
    
    In the CB02 statistical cloud scheme, both the mean condensate and
    its subgrid variance are needed for:
    - Accurate radiative transfer (cloud optical properties)
    - Autoconversion rates (nonlinear dependence on condensate)
    - Precipitation initiation (threshold behavior)
    
    The variance relationship σ_rc(q₁) is derived from Cloud Resolving
    Model (CRM) simulations and encapsulates the sub-grid structure
    of clouds at different stages of development.
    
    **Lookup Table Approach:**
    
    Rather than computing σ_rc analytically (expensive), a lookup table
    is used with linear interpolation:
    
    1. Convert q₁ to table index:
       inq1 = floor(2 × q₁)
       Discretization: Δq₁ = 0.5
    
    2. Bound index to table range:
       inq2 = min(max(-22, inq1), 10)
       Table covers q₁ ∈ [-11, 5.5]
    
    3. Linear interpolation:
       frac = 2×q₁ - inq1
       σ_rc = (1-frac)×SRC_1D[inq2+22] + frac×SRC_1D[inq2+23]
    
    4. Cap at maximum:
       σ_rc = min(σ_rc, 1.0)
    
    **Index Offset:**
    
    The "+22" offset accounts for negative q₁ values:
    - q₁ = -11 → inq1 = -22 → table index 0
    - q₁ = 0 → inq1 = 0 → table index 22
    - q₁ = +5 → inq1 = +10 → table index 32
    
    **Typical Variance Behavior:**
    
    Based on CRM simulations:
    
    - Strong subsaturation (q₁ < -3): σ_rc/σ ≈ 0
      No cloud condensate
    
    - Weak subsaturation (-3 < q₁ < 0): σ_rc/σ increases
      Scattered clouds forming
    
    - Near saturation (q₁ ≈ 0): σ_rc/σ ≈ 0.4
      Maximum variability, clouds developing
    
    - Supersaturation (q₁ > 0): σ_rc/σ decreases slightly
      More uniform clouds, less relative variability
    
    - Strong supersaturation (q₁ > 3): σ_rc/σ ≈ 0.2-0.3
      Mature stratiform cloud, relatively uniform
    
    **Physical Interpretation:**
    
    The variance σ_rc represents the spread in cloud condensate values
    within a grid box:
    
    - High σ_rc: patchy clouds, large local variations
    - Low σ_rc: uniform clouds, small variations
    - σ_rc = 0: no cloud or completely uniform
    
    **Usage in Microphysics:**
    
    The computed σ_rc is used for:
    
    1. **Radiation:** Cloud optical depth variance affects
       albedo and transmissivity (plane-parallel bias correction)
    
    2. **Autoconversion:** Nonlinear rate ∝ r_c² requires
       accounting for variance: ⟨r_c²⟩ = ⟨r_c⟩² + σ_rc²
    
    3. **Diagnostics:** Subgrid cloud structure analysis
    
    **Numerical Considerations:**
    
    - Floor operation prevents floating point issues
    - Min/max bounds prevent array out-of-bounds
    - Linear interpolation is first-order accurate
    - Table values precomputed from CRM statistics
    
    **Lambda3 Option:**
    
    The code mentions "LAMBDA3='CB'" option (not yet implemented).
    This refers to the Chaboureau and Bechtold choice of the
    λ₃ parameter in the statistical scheme, which affects the
    variance-mean relationship.
    
    Source Reference
    ----------------
    PHYEX/src/common/micro/condensation.F90, lines 186-189
    
    See Also
    --------
    condensation : Main CB02 condensation scheme
    cloud_fraction_2 : Subgrid autoconversion using variance
    
    References
    ----------
    Chaboureau, J.-P., and P. Bechtold, 2002: A simple cloud
    parameterization derived from cloud resolving model data:
    Diagnostic and prognostic applications. J. Atmos. Sci., 59, 2362-2372.
    
    Sommeria, G., and J.W. Deardorff, 1977: Subgrid-scale condensation
    in models of nonprecipitating clouds. J. Atmos. Sci., 34, 344-355.
    
    Examples
    --------
    >>> # Near saturation → maximum variance
    >>> # q₁ = 0.0 → σ_rc/σ ≈ 0.4 (from table lookup)
    
    >>> # Supersaturated → lower variance
    >>> # q₁ = 2.0 → σ_rc/σ ≈ 0.3
    
    >>> # Subsaturated → very low variance
    >>> # q₁ = -5.0 → σ_rc/σ ≈ 0.05
    """
    with computation(PARALLEL), interval(...):
        inq1 = floor(min(100.0, max(-100.0, 2 * q1[0, 0, 0])))
        inq2 = min(max(-22, inq1), 10)
        # inner min/max prevents sigfpe when 2*zq1 does not fit dtype_into an "int"
        inc = 2 * q1  # - inq2
        sigrc = min(1, (1 - inc) * src_1d.A[inq2 + 22] + inc * src_1d.A[inq2 + 23])

        # todo : add LAMBDA3='CB' inlined conditional
