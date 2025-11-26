# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import __INLINED, PARALLEL, Field, computation, interval
from ice3.functions.ice_adjust import sublimation_latent_heat, vaporisation_latent_heat


# from_file="PHYEX/src/common/micro/ice_adjust.F90",
# from_line=450,
# to_line=473
def thermodynamic_fields(
    th: Field["float"],
    exn: Field["float"],
    rv: Field["float"],
    rc: Field["float"],
    rr: Field["float"],
    ri: Field["float"],
    rs: Field["float"],
    rg: Field["float"],
    lv: Field["float"],
    ls: Field["float"],
    cph: Field["float"],
    t: Field["float"],
):
    """
    Compute temperature, latent heats, and moist air specific heat.
    
    This stencil computes fundamental thermodynamic fields needed for
    saturation adjustment and cloud fraction calculations. It converts
    potential temperature to actual temperature, calculates temperature-
    dependent latent heats, and computes the specific heat of moist air
    including all hydrometeor species.
    
    Parameters
    ----------
    th : Field[float]
        Potential temperature (K). Input field.
    exn : Field[float]
        Exner function π = (p/p₀)^(R_d/c_pd) (dimensionless). Input field.
    rv : Field[float]
        Water vapor mixing ratio (kg/kg). Input field.
    rc : Field[float]
        Cloud droplet mixing ratio (kg/kg). Input field.
    rr : Field[float]
        Rain mixing ratio (kg/kg). Input field.
    ri : Field[float]
        Ice crystal mixing ratio (kg/kg). Input field.
    rs : Field[float]
        Snow mixing ratio (kg/kg). Input field.
    rg : Field[float]
        Graupel mixing ratio (kg/kg). Input field.
    lv : Field[float]
        Latent heat of vaporization (J/kg). Output field.
    ls : Field[float]
        Latent heat of sublimation (J/kg). Output field.
    cph : Field[float]
        Specific heat of moist air at constant pressure (J/(kg·K)). Output field.
    t : Field[float]
        Actual temperature (K). Output field.
    
    Returns
    -------
    None
        Modifies lv, ls, cph, and t in place.
    
    Notes
    -----
    **Temperature Calculation:**
    
    T = θ × π
    
    where θ is potential temperature and π is the Exner function.
    
    **Latent Heats (Temperature Dependent):**
    
    The latent heats vary with temperature to account for the temperature
    dependence of specific heats:
    
    - L_v(T): Vaporization latent heat (liquid → vapor)
    - L_s(T): Sublimation latent heat (ice → vapor)
    - L_m(T) = L_s(T) - L_v(T): Melting latent heat (ice → liquid)
    
    **Moist Air Specific Heat:**
    
    c_ph = c_pd + c_pv×r_v + c_l×(r_c + r_r) + c_i×(r_i + r_s + r_g)
    
    Includes contributions from:
    - Dry air (c_pd)
    - Water vapor (c_pv × r_v)
    - Liquid water (c_l × (r_c + r_r))
    - Ice (c_i × (r_i + r_s + r_g))
    
    The specific heat is used in temperature tendency calculations when
    phase changes occur: dT/dt = L_x × (dr_x/dt) / c_ph
    
    **Hydrometeor Categories (NRR):**
    
    - NRR=2: Cloud + Ice (minimal scheme)
    - NRR=4: Cloud + Rain (warm rain only)
    - NRR=5: Cloud + Rain + Ice + Snow (no graupel)
    - NRR=6: Cloud + Rain + Ice + Snow + Graupel (full ICE3)
    - NRR=7: Full ICE4 with hail (not implemented in GT4Py version)
    
    Source Reference
    ----------------
    PHYEX/src/common/micro/ice_adjust.F90, lines 450-473
    
    See Also
    --------
    cloud_fraction_1 : Cloud fraction computation after condensation
    cloud_fraction_2 : Subgrid autoconversion in cloud fraction
    """
    from __externals__ import CI, CL, CPD, CPV, NRR

    # 2.3 Compute the variation of mixing ratio
    with computation(PARALLEL), interval(...):
        t = th * exn
        lv = vaporisation_latent_heat(t)
        ls = sublimation_latent_heat(t)

    # Translation note : in Fortran, ITERMAX = 1, DO JITER =1,ITERMAX
    # Translation note : version without iteration is kept (1 iteration)
    #                   IF jiter = 1; CALL ITERATION()
    # jiter > 0

    # numer of moist variables fixed to 6 (without hail)

    # Translation note :
    # 2.4 specific heat for moist air at t+1
    with computation(PARALLEL), interval(...):
        # Translation note : case(7) removed because hail is not taken into account
        # Translation note : l453 to l456 removed
        if __INLINED(NRR == 6):
            cph = CPD + CPV * rv + CL * (rc + rr) + CI * (ri + rs + rg)
        if __INLINED(NRR == 5):
            cph = CPD + CPV * rv + CL * (rc + rr) + CI * (ri + rs)
        if __INLINED(NRR == 4):
            cph = CPD + CPV * rv + CL * (rc + rr)
        if __INLINED(NRR == 2):
            cph = CPD + CPV * rv + CL * rc + CI * ri


# from_file="PHYEX/src/common/micro/ice_adjust.F90",
# from_line=278,
# to_line=312
def cloud_fraction_1(
    lv: Field["float"],
    ls: Field["float"],
    cph: Field["float"],
    exnref: Field["float"],
    rc: Field["float"],
    ri: Field["float"],
    ths: Field["float"],
    rvs: Field["float"],
    rcs: Field["float"],
    ris: Field["float"],
    rc_tmp: Field["float"],
    ri_tmp: Field["float"],
    dt: "float",
):
    """
    Compute mixing ratio sources from condensation/evaporation tendencies.
    
    This stencil converts condensation tendencies into actual source terms
    for the prognostic equations, applying conservative constraints and
    updating potential temperature accordingly. It handles both liquid
    (cloud droplets) and ice (crystals) phase changes.
    
    Parameters
    ----------
    lv : Field[float]
        Latent heat of vaporization (J/kg). Input field.
    ls : Field[float]
        Latent heat of sublimation (J/kg). Input field.
    cph : Field[float]
        Specific heat of moist air (J/(kg·K)). Input field.
    exnref : Field[float]
        Reference Exner function (dimensionless). Input field.
    rc : Field[float]
        Initial cloud droplet mixing ratio (kg/kg). Input field.
    ri : Field[float]
        Initial ice crystal mixing ratio (kg/kg). Input field.
    ths : Field[float]
        Potential temperature source term (K/s). Modified in place.
    rvs : Field[float]
        Water vapor source term (kg/kg/s). Modified in place.
    rcs : Field[float]
        Cloud droplet source term (kg/kg/s). Modified in place.
    ris : Field[float]
        Ice crystal source term (kg/kg/s). Modified in place.
    rc_tmp : Field[float]
        Temporary cloud mixing ratio after condensation (kg/kg). Input field.
    ri_tmp : Field[float]
        Temporary ice mixing ratio after deposition (kg/kg). Input field.
    dt : float
        Physics time step (s). Scalar parameter.
    
    Returns
    -------
    None
        Modifies ths, rvs, rcs, and ris in place.
    
    Notes
    -----
    **Tendency Calculation:**
    
    The condensation/deposition tendencies are computed as:
    
    w₁ = (rc_tmp - rc) / Δt  (liquid condensation rate, kg/kg/s)
    w₂ = (ri_tmp - ri) / Δt  (ice deposition rate, kg/kg/s)
    
    **Conservative Constraints:**
    
    For condensation (w > 0):
    - Limited by available vapor: w ≤ rv_s
    
    For evaporation (w < 0):
    - Limited by available condensate: |w| ≤ rc_s or ri_s
    
    **Mass Conservation:**
    
    The vapor source is reduced by condensation/deposition:
    rv_s → rv_s - w₁ - w₂
    
    The condensate sources are increased:
    rc_s → rc_s + w₁
    ri_s → ri_s + w₂
    
    **Temperature Tendency:**
    
    Phase changes produce/consume latent heat:
    
    dθ/dt = (w₁×L_v + w₂×L_s) / (c_ph × π)
    
    - Condensation/deposition: dθ/dt > 0 (warming)
    - Evaporation/sublimation: dθ/dt < 0 (cooling)
    
    **Physical Interpretation:**
    
    This stencil ensures:
    1. Mass conservation (total water remains constant)
    2. Energy conservation (latent heat properly accounted)
    3. Physical realizability (no negative mixing ratios)
    4. Numerical stability (limited changes per time step)
    
    Source Reference
    ----------------
    PHYEX/src/common/micro/ice_adjust.F90, lines 278-312
    
    See Also
    --------
    thermodynamic_fields : Computes temperature and latent heats
    cloud_fraction_2 : Subgrid cloud fraction and autoconversion
    """
    # l274 in ice_adjust.F90
    ##### 5.     COMPUTE THE SOURCES AND STORES THE CLOUD FRACTION #####
    with computation(PARALLEL), interval(...):
        # 5.0 compute the variation of mixing ratio
        w1 = (rc_tmp - rc) / dt
        w2 = (ri_tmp - ri) / dt

        # 5.1 compute the sources
        w1 = max(w1, -rcs) if w1 < 0.0 else min(w1, rvs)
        rvs -= w1
        rcs += w1
        ths += w1 * lv / (cph * exnref)

        w2 = max(w2, -ris) if w2 < 0.0 else min(w2, rvs)
        rvs -= w2
        ris += w2
        ths += w2 * ls / (cph * exnref)

        #### split


# from_file="PHYEX/src/common/micro/ice_adjust.F90",
# from_line=313,
# to_line=419
def cloud_fraction_2(
    rhodref: Field["float"],
    exnref: Field["float"],
    t: Field["float"],
    cph: Field["float"],
    lv: Field["float"],
    ls: Field["float"],
    ths: Field["float"],
    rvs: Field["float"],
    rcs: Field["float"],
    ris: Field["float"],
    rc_mf: Field["float"],
    ri_mf: Field["float"],
    cf_mf: Field["float"],
    cldfr: Field["float"],
    hlc_hrc: Field["float"],
    hlc_hcf: Field["float"],
    hli_hri: Field["float"],
    hli_hcf: Field["float"],
    dt: "float",
):
    """
    Compute cloud fraction and subgrid autoconversion from mass flux condensate.
    
    This stencil handles subgrid condensation from shallow convection mass fluxes
    and computes the resulting cloud fraction. When enabled (LSUBG_COND=True), it
    also calculates subgrid autoconversion using PDF-based methods, accounting for
    sub-grid variability in condensate fields.
    
    Parameters
    ----------
    rhodref : Field[float]
        Reference air density (kg/m³). Input field.
    exnref : Field[float]
        Reference Exner function (dimensionless). Input field.
    t : Field[float]
        Temperature (K). Input field.
    cph : Field[float]
        Specific heat of moist air (J/(kg·K)). Input field.
    lv : Field[float]
        Latent heat of vaporization (J/kg). Input field.
    ls : Field[float]
        Latent heat of sublimation (J/kg). Input field.
    ths : Field[float]
        Potential temperature source (K/s). Modified in place.
    rvs : Field[float]
        Water vapor source (kg/kg/s). Modified in place.
    rcs : Field[float]
        Cloud droplet source (kg/kg/s). Modified in place.
    ris : Field[float]
        Ice crystal source (kg/kg/s). Modified in place.
    rc_mf : Field[float]
        Cloud condensate from mass flux (kg/kg). Input field.
    ri_mf : Field[float]
        Ice condensate from mass flux (kg/kg). Input field.
    cf_mf : Field[float]
        Cloud fraction from mass flux (dimensionless, 0-1). Input field.
    cldfr : Field[float]
        Total cloud fraction (dimensionless, 0-1). Modified in place.
    hlc_hrc : Field[float]
        Liquid autoconversion amount (kg/kg). Modified in place.
    hlc_hcf : Field[float]
        Liquid autoconversion cloud fraction (dimensionless). Modified in place.
    hli_hri : Field[float]
        Ice autoconversion amount (kg/kg). Modified in place.
    hli_hcf : Field[float]
        Ice autoconversion cloud fraction (dimensionless). Modified in place.
    dt : float
        Physics time step (s). Scalar parameter.
    
    Returns
    -------
    None
        Modifies cldfr, ths, rvs, rcs, ris, hlc_hrc, hlc_hcf, hli_hri, hli_hcf in place.
    
    Notes
    -----
    **Two Operating Modes:**
    
    1. **Without Subgrid Condensation (LSUBG_COND=False):**
       - Binary cloud fraction: 0 or 1
       - Cloud present if (r_c + r_i) × Δt > 10⁻¹²
       - Simple all-or-nothing approach
    
    2. **With Subgrid Condensation (LSUBG_COND=True, AROME default):**
       - Partial cloud fractions from mass flux scheme
       - Subgrid autoconversion using PDF methods
       - Accounts for sub-grid variability
    
    **Mass Flux Condensate Processing:**
    
    When LSUBG_COND is enabled:
    
    1. Convert mass flux condensate to rates:
       w₁ = rc_mf / Δt  (liquid rate, kg/kg/s)
       w₂ = ri_mf / Δt  (ice rate, kg/kg/s)
    
    2. Conservatively limit to available vapor:
       If w₁ + w₂ > rv_s:
           w₁ → w₁ × rv_s/(w₁ + w₂)
           w₂ → rv_s - w₁
    
    3. Update cloud fraction additively:
       cf_total = min(1, cf_existing + cf_mf)
    
    4. Update mixing ratios and temperature
    
    **Subgrid Autoconversion:**
    
    Two PDF methods available via SUBG_MF_PDF:
    
    **Method 0 (NONE - Step Function):**
    - Autoconversion occurs if condensate exceeds threshold
    - Liquid: w₁×Δt > cf_mf × (CRIAUTC/ρ)
    - Ice: w₂×Δt > cf_mf × CRIAUTI(T)
    - All excess converts → rain/snow
    - Cloud fraction set to mass flux value
    
    **Method 1 (TRIANGLE - Triangular PDF):**
    - Assumes triangular sub-grid condensate distribution
    - Calculates partial autoconversion
    - More physical than step function
    
    For triangular PDF, three regimes:
    
    A. High condensate (q > cf×r_crit):
       hcf = 1 - 0.5×(cf×r_crit/q)²
       hr = q - (cf×r_crit)³/(3q²)
       Large conversion, most of cloud converts
    
    B. Low condensate (q ≤ 0.5×cf×r_crit):
       hcf = 0, hr = 0
       No autoconversion
    
    C. Intermediate (0.5×cf×r_crit < q ≤ cf×r_crit):
       hcf = (2q - cf×r_crit)²/(2q²)
       hr = [4q³ - 3q(cf×r_crit)² + (cf×r_crit)³]/(3q²)
       Partial conversion
    
    **Autoconversion Thresholds:**
    
    Liquid (cloud → rain):
    r_crit = CRIAUTC / ρ_ref
    Typical: ~5×10⁻⁴ kg/kg at surface
    
    Ice (ice → snow):
    r_crit(T) = min(CRIAUTI, 10^(A×(T-273.16) + B))
    Temperature-dependent, increases with warming
    Typical: ~2×10⁻⁵ kg/kg at -20°C
    
    **Physical Interpretation:**
    
    This stencil bridges the gap between:
    - Resolved-scale condensation (explicit grid-box saturation)
    - Unresolved shallow convection (mass flux parameterization)
    - Sub-grid variability (PDF-based autoconversion)
    
    It ensures smooth transition from small cumulus (mass flux)
    to stratiform clouds (resolved) by additive cloud fractions.
    
    **Diagnostic Outputs:**
    
    hlc_hrc, hlc_hcf: Liquid autoconversion amount and cloud fraction
    hli_hri, hli_hcf: Ice autoconversion amount and cloud fraction
    
    These are used later for:
    - Computing rain/snow production rates
    - Diagnosing precipitation initiation
    - Radiation calculations
    
    Source Reference
    ----------------
    PHYEX/src/common/micro/ice_adjust.F90, lines 313-419
    
    See Also
    --------
    thermodynamic_fields : Temperature and latent heat computation
    cloud_fraction_1 : Condensation source term computation
    condensation : Main condensation scheme with CB02
    
    References
    ----------
    Bougeault, P., 1981: Modeling the trade-wind cumulus boundary layer.
    Part I: Testing the ensemble cloud relations against numerical data.
    J. Atmos. Sci., 38, 2414-2428.
    
    Examples
    --------
    >>> # Binary cloud fraction mode (LSUBG_COND=False)
    >>> # If total condensate > threshold → cloud fraction = 1
    >>> # Otherwise → cloud fraction = 0
    
    >>> # PDF-based mode (LSUBG_COND=True, SUBG_MF_PDF=1)
    >>> # Partial cloud fractions from mass flux
    >>> # Subgrid autoconversion with triangular PDF
    >>> # Smooth transitions, more realistic variability
    """
    from __externals__ import (
        ACRIAUTI,
        BCRIAUTI,
        CRIAUTC,
        CRIAUTI,
        LSUBG_COND,
        SUBG_MF_PDF,
        TT,
    )

    # 5.2  compute the cloud fraction cldfr
    with computation(PARALLEL), interval(...):
        if __INLINED(not LSUBG_COND):
            cldfr = 1.0 if ((rcs + ris) * dt > 1e-12) else 0.0
        # Translation note : OCOMPUTE_SRC is taken False
        # Translation note : l320 to l322 removed

        # Translation note : LSUBG_COND = TRUE for Arome
        else:
            w1 = rc_mf / dt
            w2 = ri_mf / dt

            if w1 + w2 > rvs:
                w1 *= rvs / (w1 + w2)
                w2 = rvs - w1

            cldfr = min(1, cldfr + cf_mf)
            rcs += w1
            ris += w2
            rvs -= w1 + w2
            ths += (w1 * lv + w2 * ls) / (cph * exnref)

            # Droplets subgrid autoconversion
            # LLHLC_H is True (AROME like) phlc_hrc and phlc_hcf are present
            # LLHLI_H is True (AROME like) phli_hri and phli_hcf are present
            criaut = CRIAUTC / rhodref

            # ice_adjust.F90 IF LLNONE; IF CSUBG_MF_PDF is None
            if __INLINED(SUBG_MF_PDF == 0):
                if w1 * dt > cf_mf * criaut:
                    hlc_hrc += w1 * dt
                    hlc_hcf = min(1.0, hlc_hcf + cf_mf)

            # Translation note : if LLTRIANGLE in .F90
            if __INLINED(SUBG_MF_PDF == 1):
                if w1 * dt > cf_mf * criaut:
                    hcf = 1.0 - 0.5 * (criaut * cf_mf / max(1e-20, w1 * dt)) ** 2
                    hr = w1 * dt - (criaut * cf_mf) ** 3 / (
                        3 * max(1e-20, w1 * dt) ** 2
                    )

                elif 2.0 * w1 * dt <= cf_mf * criaut:
                    hcf = 0.0
                    hr = 0.0

                else:
                    hcf = (2.0 * w1 * dt - criaut * cf_mf) ** 2 / (
                        2.0 * max(1.0e-20, w1 * dt) ** 2
                    )
                    hr = (
                        4.0 * (w1 * dt) ** 3
                        - 3.0 * w1 * dt * (criaut * cf_mf) ** 2
                        + (criaut * cf_mf**3)
                    ) / (3 * max(1.0e-20, w1 * dt) ** 2)

                hcf *= cf_mf
                hlc_hcf = min(1.0, hlc_hcf + hcf)
                hlc_hrc += hr

            # Ice subgrid autoconversion
            criaut = min(
                CRIAUTI,
                10 ** (ACRIAUTI * (t - TT) + BCRIAUTI),
            )

            # LLNONE in ice_adjust.F90
            if __INLINED(SUBG_MF_PDF == 0):
                if w2 * dt > cf_mf * criaut:
                    hli_hri += w2 * dt
                    hli_hcf = min(1.0, hli_hcf + cf_mf)

            # LLTRIANGLE in ice_adjust.F90
            if __INLINED(SUBG_MF_PDF == 1):
                if w2 * dt > cf_mf * criaut:
                    hcf = 1.0 - 0.5 * ((criaut * cf_mf) / (w2 * dt)) ** 2
                    hri = w2 * dt - (criaut * cf_mf) ** 3 / (3 * (w2 * dt) ** 2)

                elif 2 * w2 * dt <= cf_mf * criaut:
                    hcf = 0.0
                    hri = 0.0

                else:
                    hcf = (2.0 * w2 * dt - criaut * cf_mf) ** 2 / (2.0 * (w2 * dt) ** 2)
                    hri = (
                        4.0 * (w2 * dt) ** 3
                        - 3.0 * w2 * dt * (criaut * cf_mf) ** 2
                        + (criaut * cf_mf) ** 3
                    ) / (3.0 * (w2 * dt) ** 2)

                hcf *= cf_mf
                hli_hcf = min(1.0, hli_hcf + hcf)
                hli_hri += hri
    # Translation note : 402 -> 427 (removed pout_x not present )
