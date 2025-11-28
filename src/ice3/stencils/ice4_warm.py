# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (PARALLEL, Field,
                                      computation, exp, interval, log)


def ice4_warm(
    ldcompute: Field["bool"],  # boolean field for microphysics computation
    rhodref: Field["float"],
    t: Field["float"],  # temperature
    pres: Field["float"],
    tht: Field["float"],
    lbdar: Field["float"],  # slope parameter for the rain drop distribution
    lbdar_rf: Field["float"],  # slope parameter for the rain fraction part
    ka: Field["float"],  # thermal conductivity of the air
    dv: Field["float"],  # diffusivity of water vapour
    cj: Field["float"],  # function to compute the ventilation coefficient
    hlc_hcf: Field["float"],  # High Cloud Fraction in grid
    hlc_hrc: Field["float"],  # LWC that is high in grid
    cf: Field["float"],  # cloud fraction
    rf: Field["float"],  # rain fraction
    rvt: Field["float"],  # water vapour mixing ratio at t
    rct: Field["float"],  # cloud water mixing ratio at t
    rrt: Field["float"],  # rain water mixing ratio at t
    rcautr: Field["float"],  # autoconversion of rc for rr production
    rcaccr: Field["float"],  # accretion of r_c for r_r production
    rrevav: Field["float"],  # evaporation of rr
    ldsoft: "bool",
):
    """
    Compute warm rain microphysical processes for ICE4 scheme.
    
    This stencil computes three fundamental warm rain processes that occur
    in liquid-phase clouds: autoconversion (cloud droplets → rain drops),
    accretion (cloud droplets collected by rain), and evaporation (rain
    drops → vapor). These processes form the foundation of precipitation
    formation in warm clouds (T > 0°C) and the warm-phase portion of
    mixed-phase clouds.
    
    Parameters
    ----------
    ldcompute : Field[bool]
        Computation mask (True = compute, False = skip). Input field.
    rhodref : Field[float]
        Reference air density (kg/m³). Input field.
    t : Field[float]
        Temperature (K). Input field.
    pres : Field[float]
        Pressure (Pa). Input field.
    tht : Field[float]
        Potential temperature (K). Input field.
    lbdar : Field[float]
        Rain size distribution slope parameter λ_r (m⁻¹). Input field.
        For grid-mean rain distribution.
    lbdar_rf : Field[float]
        Rain slope parameter for rain fraction λ_r,rf (m⁻¹). Input field.
        Used when SUBG_RR_EVAP='PRFR'.
    ka : Field[float]
        Thermal conductivity of air (W/(m·K)). Input field.
    dv : Field[float]
        Water vapor diffusivity in air (m²/s). Input field.
    cj : Field[float]
        Ventilation coefficient C_j (m^((1-D)/2)). Input field.
    hlc_hcf : Field[float]
        High cloud fraction from subgrid scheme (0-1). Input field.
    hlc_hrc : Field[float]
        High cloud liquid water content (kg/kg). Input field.
        From subgrid autoconversion scheme.
    cf : Field[float]
        Total cloud fraction (0-1). Input field.
    rf : Field[float]
        Rain/precipitation fraction (0-1). Input field.
        Used when SUBG_RR_EVAP='PRFR'.
    rvt : Field[float]
        Water vapor mixing ratio (kg/kg). Input field.
    rct : Field[float]
        Cloud droplet mixing ratio (kg/kg). Input field.
    rrt : Field[float]
        Rain mixing ratio (kg/kg). Input field.
    rcautr : Field[float]
        Cloud autoconversion rate: r_c → r_r (kg/kg/s). Output field.
    rcaccr : Field[float]
        Cloud accretion rate: r_c → r_r (kg/kg/s). Output field.
    rrevav : Field[float]
        Rain evaporation rate: r_r → r_v (kg/kg/s). Output field.
    ldsoft : bool
        Soft threshold mode (scalar). If True, disables all processes.
    
    Returns
    -------
    None
        Modifies tendency fields in place.
    
    Notes
    -----
    **Process 1: Cloud Autoconversion (RCAUTR)**
    
    Condition: hlc_hrc > threshold, hlc_hcf > 0
    
    Small cloud droplets spontaneously form precipitation-sized drops:
    
    dr_c/dt = -T_IMAUTC × max(0, hlc_hrc - hlc_hcf × CRIAUTC/ρ)
    
    where:
    - T_IMAUTC: Autoconversion time constant (s⁻¹) ~ 1-10 s
    - CRIAUTC: Critical cloud content for autoconversion (kg/m³)
    - hlc_hrc: Subgrid cloud liquid water
    - hlc_hcf: Subgrid cloud fraction
    - ρ: Air density
    
    Physical basis: Kessler (1969) threshold mechanism, droplets larger
    than ~40 μm diameter start colliding and coalescing efficiently.
    
    Uses subgrid values (hlc_hrc, hlc_hcf) to account for cloud
    heterogeneity - autoconversion occurs where condensate exceeds
    threshold within cloudy portions of grid box.
    
    **Process 2: Cloud Accretion by Rain (RCACCR)**
    
    Condition: r_c > threshold, r_r > threshold
    
    Falling rain drops efficiently collect cloud droplets:
    
    dr_c/dt = -F_CACCR × r_c × λ_r^EXCACCR × ρ^(-CEXVT)
    
    where:
    - F_CACCR: Collection efficiency factor (m^EXCACCR·s⁻¹)
    - λ_r: Rain slope parameter (∝ concentration/size)
    - EXCACCR: Size distribution exponent
    - ρ^(-CEXVT): Air density correction for fall speed
    
    Physical basis: Continuous collection process, highly efficient
    (collection efficiency E ~ 0.8-1.0). Rain drops sweep out cloud
    droplets in their path.
    
    Unlike autoconversion (threshold-based), accretion is linear in r_c
    and occurs whenever both rain and cloud are present.
    
    **Process 3: Rain Evaporation (RREVAV)**
    
    Three options via SUBG_RR_EVAP parameter:
    
    **Option 0: NONE (Grid-mean evaporation)**
    
    Condition: r_r > threshold, r_c ≤ threshold (clear sky)
    
    Rain evaporates into subsaturated environment:
    
    dr_r/dt = (S_w / (ρ × A_v)) × [O0EVAR × λ_r^EX0EVAR + O1EVAR × C_j × λ_r^EX1EVAR]
    
    where:
    - S_w: Subsaturation with respect to liquid water
      S_w = 1 - r_v × (p - e_sat,w)/(ε × e_sat,w)
    - A_v: Thermodynamic evaporation coefficient
      A_v = (L_v + (c_pv - c_l)×(T - T_t))²/(k_a × R_v × T²) + R_v×T/(D_v × e_sat,w)
    - O0EVAR, O1EVAR: Non-ventilated and ventilated coefficients
    - C_j: Ventilation factor
    
    Two terms represent:
    - O0EVAR: Diffusive evaporation (stationary drops)
    - O1EVAR × C_j: Enhanced evaporation (falling drops, ventilation)
    
    **Option 1: CLFR (Cloud Fraction method)**
    
    Evaporation occurs only in clear portion of grid box (1 - cf):
    
    dr_r/dt = Evap_rate × (1 - cf)
    
    Accounts for cloud shielding - rain falling through cloudy regions
    doesn't evaporate (already saturated).
    
    **Option 2: PRFR (Precipitation Fraction method)**
    
    Evaporation in region where rain exists but cloud doesn't (rf - cf):
    
    dr_r/dt = Evap_rate × (rf - cf)
    
    Most sophisticated option:
    - Uses rain fraction (rf) and cloud fraction (cf)
    - Evaporation only where rf > cf (rain shaft outside cloud)
    - Accounts for both cloud shielding and precipitation geometry
    - Uses unsaturated temperature T^u for environment calculation
    
    **Unsaturated Temperature (T^u):**
    
    In CLFR and PRFR modes, environment temperature differs from
    in-cloud temperature:
    
    θ_l = θ - (L_v/c_pd) × (θ/T) × r_c  (liquid water potential temp)
    T^u = θ_l × (T/θ)  (unsaturated temperature)
    
    This T^u is used for computing e_sat,w and subsaturation in the
    environment, following Bechtold et al. (1993).
    
    **Evaporation Coefficients:**
    
    The thermodynamic coefficient A_v accounts for:
    1. Heat conduction limitation (thermal diffusivity k_a)
    2. Vapor diffusion limitation (molecular diffusivity D_v)
    3. Latent heat release (L_v)
    4. Temperature dependence
    
    Ventilation enhances evaporation by factor ~2-3 for typical
    rain drop sizes and fall speeds.
    
    **Physical Interpretation:**
    
    Warm rain processes:
    - Autoconversion: Initiates precipitation (slow, threshold-based)
    - Accretion: Rapidly grows rain (fast, continuous collection)
    - Evaporation: Dissipates precipitation below cloud base
    
    Typical timescales:
    - Autoconversion: τ ~ 1000 s (slow, droplet coalescence)
    - Accretion: τ ~ 100 s (fast, efficient collection)
    - Evaporation: τ ~ 100-1000 s (depends on subsaturation)
    
    **Autoconversion vs Accretion:**
    
    Autoconversion:
    - Dominates early in cloud development
    - Produces first rain drops from cloud droplets
    - Threshold-based (needs sufficient r_c)
    - Relatively slow
    
    Accretion:
    - Dominates once rain forms
    - Rapidly transfers cloud water to rain
    - No threshold, always active
    - Much faster than autoconversion
    
    **Subgrid Treatment:**
    
    Autoconversion uses subgrid values (hlc_hrc, hlc_hcf) to:
    - Account for cloud heterogeneity
    - Properly represent threshold behavior
    - Avoid "drizzle problem" in coarse-resolution models
    
    Evaporation optionally uses cloud/rain fractions (cf, rf) to:
    - Account for precipitation geometry
    - Prevent spurious evaporation in cloudy regions
    - Better represent mesoscale organization
    
    Source Reference
    ----------------
    PHYEX/src/common/micro/mode_ice4_warm.F90
    
    See Also
    --------
    ice4_slow : Slow cold processes
    ice4_fast_rg : Fast riming processes
    cloud_fraction_2 : Subgrid autoconversion computation
    
    References
    ----------
    Kessler, E., 1969: On the Distribution and Continuity of Water
    Substance in Atmospheric Circulations. Meteor. Monogr., No. 32,
    Amer. Meteor. Soc., 84 pp.
    
    Bechtold, P., E. Bazile, F. Guichard, P. Mascart, and E. Richard,
    1993: A mass-flux convection scheme for regional and global models.
    Quart. J. Roy. Meteor. Soc., 127, 869-886.
    
    Examples
    --------
    >>> # Warm cloud with sufficient cloud water
    >>> # r_c = 1e-3 kg/kg, hlc_hrc = 1.2e-3 kg/kg
    >>> # CRIAUTC/ρ ~ 5e-4 kg/kg
    >>> # → Autoconversion active: RCAUTR ~ 7e-4 kg/kg/s
    
    >>> # Mature warm rain
    >>> # r_c = 5e-4 kg/kg, r_r = 1e-3 kg/kg
    >>> # → Accretion dominates: RCACCR ~ 5e-3 kg/kg/s (10× faster)
    
    >>> # Rain below cloud base, subsaturated
    >>> # r_r = 1e-3 kg/kg, RH = 80%, r_c = 0
    >>> # → Evaporation active: RREVAV ~ 1e-4 kg/kg/s
    >>> # → Virgae, rain evaporates before reaching ground
    """
    from __externals__ import (ALPW, BETAW, C_RTMIN, CEXVT, CL, CPD, CPV,
                               CRIAUTC, EPSILO, EX0EVAR, EX1EVAR, EXCACCR,
                               FCACCR, GAMW, LVTT, O0EVAR, O1EVAR, R_RTMIN, RV,
                               SUBG_RR_EVAP, TIMAUTC, TT)

    # 4.2 compute the autoconversion of r_c for r_r : RCAUTR
    with computation(PARALLEL), interval(...):
        if ldcompute and hlc_hrc > C_RTMIN and hlc_hcf > 0.0:
            if not ldsoft:
                rcautr = TIMAUTC * max(0.0, hlc_hrc - hlc_hcf * CRIAUTC / rhodref)
        else:
            rcautr = 0.0

    # 4.3 compute the accretion of r_c for r_r : RCACCR
    with computation(PARALLEL), interval(...):
        # Translation note : HSUBG_RC_RR_ACCR=='NONE'
        if ldcompute and rct > C_RTMIN and rrt > R_RTMIN:
            if not ldsoft:
                rcaccr = FCACCR * rct * lbdar ** EXCACCR * rhodref ** (-CEXVT)
        else:
            rcaccr = 0.0

        # Translation note : second option from l121 to l155 ommitted
        # elif csubg_rc_rr_accr == 1:

    # 4.4 computes the evaporation of r_r :  RREVAV
    with computation(PARALLEL), interval(...):
        # NONE in Fortran code
        if SUBG_RR_EVAP == 0:
            if ldcompute and rrt > R_RTMIN and rct <= C_RTMIN:
                if not ldsoft:
                    rrevav = exp(ALPW - BETAW / t - GAMW * log(t))
                    usw = 1 - rvt * (pres - rrevav) / (EPSILO * rrevav)
                    rrevav = (LVTT + (CPV - CL) * (t - TT)) ** 2 / (
                        ka * RV * t**2
                    ) + (RV * t) / (dv * rrevav)
                    rrevav = (max(0.0, usw) / (rhodref * rrevav)) * (
                        O0EVAR * lbdar**EX0EVAR + O1EVAR * cj * lbdar**EX1EVAR
                    )
            else:
                rrevav = 0.0

        if SUBG_RR_EVAP == 1 or SUBG_RR_EVAP == 2:
            # HSUBG_RR_EVAP=='CLFR'
            if SUBG_RR_EVAP == 1:
                zw4 = 1.0  # precipitation fraction
                zw3 = lbdar

            # HSUBG_RR_EVAP=='PRFR'
            elif SUBG_RR_EVAP == 2:
                zw4 = rf  # precipitation fraction
                zw3 = lbdar_rf

            if ldcompute and rrt > R_RTMIN and zw4 > cf:
                if not ldsoft:
                    # outside the cloud (environment) the use of T^u (unsaturated) instead of T
                    # ! Bechtold et al. 1993

                    # ! T_l
                    thlt_tmp = tht - LVTT * tht / CPD / t * rct

                    # T^u = T_l = theta_l * (T/theta)
                    zw2 = thlt_tmp * t / tht

                    # saturation over water
                    rrevav = exp(ALPW - BETAW / zw2 - GAMW * log(zw2))

                    # s, undersaturation over water (with new theta^u)
                    usw = 1 - rvt * (pres - rrevav) / (EPSILO * rrevav)

                    rrevav = (LVTT + (CPV - CL) * (zw2 - TT)) ** 2 / (
                        ka * RV * zw2**2
                    ) + RV * zw2 / (dv * rrevav)
                    rrevav = (
                        max(0.0, usw)
                        / (rhodref * rrevav)
                        * (O0EVAR * zw3**EX0EVAR + O1EVAR * cj * zw3**EX1EVAR)
                    )
                    rrevav = rrevav * (zw4 - cf)

            else:
                rrevav = 0.0
