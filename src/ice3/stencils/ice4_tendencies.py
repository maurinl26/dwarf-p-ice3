# -*- coding: utf-8 -*-
"""
ICE4 tendency management and aggregation stencils.

This module handles initialization, update, and post-processing of
microphysical tendencies in the ICE4 scheme. It aggregates contributions
from different microphysical processes (nucleation, riming, melting, etc.)
and applies them to prognostic variables with proper thermodynamic
consistency.

Source: PHYEX/src/common/micro/mode_ice4_tendencies.F90
"""
from __future__ import annotations

from gt4py.cartesian.gtscript import (__INLINED, PARALLEL, Field, computation,
                                      exp, interval, log, sqrt)


def init_tendencies(
    rv_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    rv_inst: Field["float"],
    rc_inst: Field["float"],
    rr_inst: Field["float"],
    ri_inst: Field["float"],
    rs_inst: Field["float"],
    rg_inst: Field["float"],
    rv_tnd: Field["float"],
    rc_tnd: Field["float"],
    rr_tnd: Field["float"],
    ri_tnd: Field["float"],
    rs_tnd: Field["float"],
    rg_tnd: Field["float"]    
):
    """
    Initialize all tendency and instantaneous change fields to zero.
    
    This function resets all tendency accumulators before the microphysics
    calculations begin. Essential for proper tendency tracking through
    the various microphysical processes.
    
    Parameters
    ----------
    rv_t, rc_t, rr_t, ri_t, rs_t, rg_t : Field[float]
        Tendency storage fields for vapor, cloud, rain, ice, snow, graupel.
    rv_inst, rc_inst, rr_inst, ri_inst, rs_inst, rg_inst : Field[float]
        Instantaneous change fields (set to 0).
    rv_tnd, rc_tnd, rr_tnd, ri_tnd, rs_tnd, rg_tnd : Field[float]
        Total tendency fields (set to 0).
        
    Source Reference:
    PHYEX/src/common/micro/mode_ice4_tendencies.F90, lines 129-134
    """
    with computation(PARALLEL), interval(...):
        rv_t = 0.0
        rc_t = 0.0
        rr_t = 0.0
        ri_t = 0.0
        rs_t = 0.0
        rg_t = 0.0
        rv_inst = 0.0
        rc_inst = 0.0
        rr_inst = 0.0
        ri_inst = 0.0
        rs_inst = 0.0
        rg_inst = 0.0
        rv_tnd = 0.0
        rc_tnd = 0.0
        rr_tnd = 0.0
        ri_tnd = 0.0
        rs_tnd = 0.0
        rg_tnd = 0.0


def mixing_ratio_init(
    rvheni_mr: Field["float"],
    rrhong_mr: Field["float"],
    rimltc_mr: Field["float"],
    rsrimcg_mr: Field["float"],
    ldsoft: "bool"
):
    """
    Initialize mixing ratio increments for specific processes.
    
    Resets the mixing ratio changes from heterogeneous nucleation (HENI),
    homogeneous freezing (HONG), ice crystal melting (IMLTC), and
    snow riming conversion (RSRIMCG) when in soft mode.
    
    Parameters
    ----------
    rvheni_mr : Field[float]
        Vapor consumed by heterogeneous nucleation (kg/kg).
    rrhong_mr : Field[float]
        Rain frozen by homogeneous nucleation (kg/kg).
    rimltc_mr : Field[float]
        Ice melted to cloud liquid (kg/kg).
    rsrimcg_mr : Field[float]
        Snow converted to graupel by riming (kg/kg).
    ldsoft : bool
        If True, reset these fields.
        
    Source Reference:
    PHYEX/src/common/micro/mode_ice4_tendencies.F90, lines 136-140
    """
    with computation(PARALLEL), interval(...):
        if ldsoft:
            rvheni_mr = 0.0
            rrhong_mr = 0.0
            rimltc_mr = 0.0
            rsrimcg_mr = 0.0


def ice4_nucleation_post_processing(
    t: Field["float"],
    exn: Field["float"],
    lsfact: Field["float"],
    tht: Field["float"],
    rvt: Field["float"],
    rit: Field["float"],
    rvheni_mr: Field["float"],
):
    """
    Apply heterogeneous ice nucleation changes to prognostic variables.
    
    Updates potential temperature, temperature, vapor, and ice after
    heterogeneous nucleation (HENI process) where water vapor deposits
    directly onto ice nuclei to form ice crystals.
    
    Parameters
    ----------
    t : Field[float]
        Temperature (K), updated with latent heat release.
    exn : Field[float]
        Exner function (dimensionless).
    lsfact : Field[float]
        Latent heat of sublimation over heat capacity, L_s/c_p (K).
    tht : Field[float]
        Potential temperature (K), updated.
    rvt : Field[float]
        Water vapor mixing ratio (kg/kg), decreased.
    rit : Field[float]
        Ice mixing ratio (kg/kg), increased.
    rvheni_mr : Field[float]
        Vapor consumed by heterogeneous nucleation (kg/kg).
        
    Notes
    -----
    Process: r_v → r_i (vapor deposits as ice)
    Energy: Releases latent heat of sublimation L_s
    Temperature change: ΔT = rvheni_mr × L_s/c_p
    
    Source Reference:
    PHYEX/src/common/micro/mode_ice4_tendencies.F90, lines 152-157
    """
    with computation(PARALLEL), interval(...):
        tht += rvheni_mr * lsfact
        t = tht * exn
        rvt -= rvheni_mr
        rit += rvheni_mr


def ice4_rrhong_post_processing(
    t: Field["float"],
    exn: Field["float"],
    lsfact: Field["float"],
    lvfact: Field["float"],
    tht: Field["float"],
    rrt: Field["float"],
    rgt: Field["float"],
    rrhong_mr: Field["float"],
):
    """
    Apply homogeneous freezing of rain to prognostic variables.
    
    Updates variables after homogeneous nucleation (HONG process) where
    rain droplets freeze instantly at very cold temperatures (T < -40°C)
    to form graupel particles.
    
    Parameters
    ----------
    t : Field[float]
        Temperature (K), updated.
    exn : Field[float]
        Exner function (dimensionless).
    lsfact : Field[float]
        L_s/c_p (K).
    lvfact : Field[float]
        L_v/c_p (K).
    tht : Field[float]
        Potential temperature (K), updated.
    rrt : Field[float]
        Rain mixing ratio (kg/kg), decreased.
    rgt : Field[float]
        Graupel mixing ratio (kg/kg), increased.
    rrhong_mr : Field[float]
        Rain frozen by homogeneous nucleation (kg/kg).
        
    Notes
    -----
    Process: r_r → r_g (rain freezes to graupel)
    Energy: Releases latent heat of fusion L_f = L_s - L_v
    Temperature change: ΔT = rrhong_mr × (L_s - L_v)/c_p
    
    Occurs at T < approximately -40°C where droplets cannot remain liquid.
    
    Source Reference:
    PHYEX/src/common/micro/mode_ice4_tendencies.F90, lines 166-171
    """
    with computation(PARALLEL), interval(...):
        tht += rrhong_mr * (lsfact - lvfact)
        t = tht * exn
        rrt -= rrhong_mr
        rgt += rrhong_mr


def ice4_rimltc_post_processing(
    t: Field["float"],
    exn: Field["float"],
    lsfact: Field["float"],
    lvfact: Field["float"],
    rimltc_mr: Field["float"],
    tht: Field["float"],
    rct: Field["float"],
    rit: Field["float"],
):
    """
    Apply ice crystal melting to prognostic variables.
    
    Updates variables after ice crystal melting (IMLTC process) where
    small ice crystals melt to form cloud droplets at T > 0°C.
    
    Parameters
    ----------
    t : Field[float]
        Temperature (K), updated.
    exn : Field[float]
        Exner function (dimensionless).
    lsfact : Field[float]
        L_s/c_p (K).
    lvfact : Field[float]
        L_v/c_p (K).
    rimltc_mr : Field[float]
        Ice melted to cloud water (kg/kg).
    tht : Field[float]
        Potential temperature (K), updated.
    rct : Field[float]
        Cloud liquid mixing ratio (kg/kg), increased.
    rit : Field[float]
        Ice crystal mixing ratio (kg/kg), decreased.
        
    Notes
    -----
    Process: r_i → r_c (ice melts to cloud liquid)
    Energy: Absorbs latent heat of fusion L_f = L_s - L_v
    Temperature change: ΔT = -rimltc_mr × (L_s - L_v)/c_p (cooling)
    
    Source Reference:
    PHYEX/src/common/micro/mode_ice4_tendencies.F90, lines 180-185
    """
    with computation(PARALLEL), interval(...):
        tht -= rimltc_mr * (lsfact - lvfact)
        t = tht * exn
        rct += rimltc_mr
        rit -= rimltc_mr


def ice4_fast_rg_pre_post_processing(
    rgsi: Field["float"],
    rgsi_mr: Field["float"],
    rvdepg: Field["float"],
    rsmltg: Field["float"],
    rraccsg: Field["float"],
    rsaccrg: Field["float"],
    rcrimsg: Field["float"],
    rsrimcg: Field["float"],
    rrhong_mr: Field["float"],
    rsrimcg_mr: Field["float"],
):
    """
    Aggregate graupel source terms from various processes.
    
    Sums all contributions to graupel production for the fast processes,
    separating direct sources from mixing ratio conversions.
    
    Parameters
    ----------
    rgsi : Field[float]
        Output: Total graupel source from collection/deposition/melting.
    rgsi_mr : Field[float]
        Output: Total graupel source from phase conversions.
    rvdepg : Field[float]
        Vapor deposition on graupel.
    rsmltg : Field[float]
        Snow melting producing graupel.
    rraccsg, rsaccrg : Field[float]
        Accretion processes.
    rcrimsg, rsrimcg : Field[float]
        Riming processes.
    rrhong_mr, rsrimcg_mr : Field[float]
        Conversion processes.
        
    Source Reference:
    PHYEX/src/common/micro/mode_ice4_tendencies.F90, lines 386-390
    """
    with computation(PARALLEL), interval(...):
        rgsi = rvdepg + rsmltg + rraccsg + rsaccrg + rcrimsg + rsrimcg
        rgsi_mr = rrhong_mr + rsrimcg_mr


def ice4_increment_update(
    lsfact: Field["float"],
    lvfact: Field["float"],
    theta_increment: Field["float"],
    rv_increment: Field["float"],
    rc_increment: Field["float"],
    rr_increment: Field["float"],
    ri_increment: Field["float"],
    rs_increment: Field["float"],
    rg_increment: Field["float"],
    rvheni_mr: Field["float"],
    rimltc_mr: Field["float"],
    rrhong_mr: Field["float"],
    rsrimcg_mr: Field["float"],
):
    """
    Update increment fields with nucleation and phase change processes.
    
    Adds the contributions from heterogeneous nucleation (HENI),
    homogeneous freezing (HONG), ice melting (IMLTC), and snow riming
    conversion (RSRIMCG) to the increment fields, with proper accounting
    for latent heat effects on potential temperature.
    
    Parameters
    ----------
    lsfact, lvfact : Field[float]
        Latent heat factors L_s/c_p and L_v/c_p (K).
    theta_increment : Field[float]
        Potential temperature increment, updated.
    rv_increment, rc_increment, rr_increment : Field[float]
        Vapor, cloud, rain increments, updated.
    ri_increment, rs_increment, rg_increment : Field[float]
        Ice, snow, graupel increments, updated.
    rvheni_mr : Field[float]
        Vapor consumed by heterogeneous nucleation.
    rimltc_mr : Field[float]
        Ice melted to cloud liquid.
    rrhong_mr : Field[float]
        Rain frozen homogeneously.
    rsrimcg_mr : Field[float]
        Snow converted to graupel by riming.
        
    Notes
    -----
    Energy accounting:
    - HENI: vapor → ice, releases L_s
    - HONG: rain → graupel, releases L_f = L_s - L_v
    - IMLTC: ice → cloud, absorbs L_f = L_s - L_v
    - RSRIMCG: snow → graupel, no phase change, no heat
    
    Source Reference:
    PHYEX/src/common/micro/mode_ice4_tendencies.F90, lines 220-238
    """
    # 5.1.6 riming-conversion of the large sized aggregates into graupel
    # Translation note : l189 to l215 omitted (since CSNOWRIMING = M90 in AROME)
    with computation(PARALLEL), interval(...):
        theta_increment += (
            rvheni_mr * lsfact
            + rrhong_mr * (lsfact - lvfact)
            - rimltc_mr * (lsfact - lvfact)
        )

        rv_increment -= rvheni_mr
        rc_increment += rimltc_mr
        rr_increment -= rrhong_mr
        ri_increment += rvheni_mr - rimltc_mr
        rs_increment -= rsrimcg_mr
        rg_increment += rrhong_mr + rsrimcg_mr


def ice4_derived_fields(
    t: Field["float"],
    rhodref: Field["float"],
    pres: Field["float"],
    ssi: Field["float"],
    ka: Field["float"],
    dv: Field["float"],
    ai: Field["float"],
    cj: Field["float"],
    rvt: Field["float"],
    zw: Field["float"]
):
    """
    Compute derived microphysical fields for process calculations.
    
    Calculates auxiliary fields needed by various microphysical parameterizations:
    - Supersaturation with respect to ice
    - Thermal conductivity of air
    - Diffusivity of water vapor
    - Thermodynamic function for deposition
    - Ventilation coefficient
    
    Parameters
    ----------
    t : Field[float]
        Temperature (K).
    rhodref : Field[float]
        Reference air density (kg/m³).
    pres : Field[float]
        Pressure (Pa).
    ssi : Field[float]
        Output: Supersaturation w.r.t. ice (dimensionless).
    ka : Field[float]
        Output: Thermal conductivity of air (W/(m·K)).
    dv : Field[float]
        Output: Diffusivity of water vapor in air (m²/s).
    ai : Field[float]
        Output: Thermodynamic function for vapor diffusion.
    cj : Field[float]
        Output: Ventilation coefficient (dimensionless).
    rvt : Field[float]
        Water vapor mixing ratio (kg/kg).
    zw : Field[float]
        Workspace for saturation vapor pressure (Pa).
        
    Notes
    -----
    These derived fields are used in:
    - Deposition/sublimation rates (ai, ssi, ka, dv)
    - Ventilation effects (cj)
    - Particle growth calculations
    
    The ventilation coefficient cj accounts for enhanced mass transfer
    due to particle motion relative to air.
    
    External Parameters:
    - ALPI, BETAI, GAMI: Saturation vapor pressure over ice coefficients
    - SCFAC: Schmidt number factor for ventilation
    - P00: Reference pressure (Pa)
    
    Source Reference:
    PHYEX/src/common/micro/mode_ice4_tendencies.F90, lines 220-238
    """
    from __externals__ import (ALPI, BETAI, CI, CPV, EPSILO, GAMI, LSTT, P00,
                               RV, SCFAC, TT)

    with computation(PARALLEL), interval(...):
        zw = exp(ALPI - BETAI / t - GAMI * log(t))
        ssi = rvt * (pres - zw) / (EPSILO * zw) - 1.0  # Supersaturation over ice
        ka = 2.38e-2 + 7.1e-5 * (t - TT)
        dv = 2.11e-5 * (t / TT) ** 1.94 * (P00 / pres)
        ai = (LSTT + (CPV - CI) * (t - TT)) ** 2 / (ka * RV * t**2) + (
            (RV * t) / (dv * zw)
        )
        cj = SCFAC * rhodref**0.3 / sqrt(1.718e-5 + 4.9e-8 * (t - TT))


def ice4_slope_parameters(
    rhodref: Field["float"],
    t: Field["float"],
    rrt: Field["float"],
    rst: Field["float"],
    rgt: Field["float"],
    lbdar: Field["float"],
    lbdar_rf: Field["float"],
    lbdas: Field["float"],
    lbdag: Field["float"],
):
    """
    Compute slope parameters (lambda) for hydrometeor size distributions.
    
    Calculates the exponential distribution slope parameters for rain, snow,
    and graupel. These lambda values characterize the particle size distribution
    N(D) = N₀ exp(-λD) and are fundamental to all microphysical rate calculations.
    
    Parameters
    ----------
    rhodref : Field[float]
        Reference air density (kg/m³).
    t : Field[float]
        Temperature (K).
    rrt, rst, rgt : Field[float]
        Rain, snow, graupel mixing ratios (kg/kg).
    lbdar : Field[float]
        Output: Rain slope parameter (m⁻¹).
    lbdar_rf : Field[float]
        Output: Rain slope for rainfall (m⁻¹), equals lbdar for AROME.
    lbdas : Field[float]
        Output: Snow slope parameter (m⁻¹).
    lbdag : Field[float]
        Output: Graupel slope parameter (m⁻¹).
        
    Notes
    -----
    Standard formula: λ = C × (ρ × r)^β
    where C and β are empirical constants (LBR/LBEXR for rain, etc.)
    
    Snow has temperature-dependent option (LSNOW_T):
    - At T > 263.15 K: Uses warmer temperature formula
    - At T ≤ 263.15 K: Uses colder temperature formula
    - Clamped between LBDAS_MIN and LBDAS_MAX
    
    All slopes are set to 0 when the corresponding mixing ratio is 0
    to avoid undefined behavior in subsequent calculations.
    
    External Parameters:
    - LBR, LBS, LBG: Intercept constants
    - LBEXR, LBEXS, LBEXG: Exponent constants
    - LSNOW_T: Temperature-dependent snow flag
    - TRANS_MP_GAMMAS: Transformation factor for snow
    
    Source Reference:
    PHYEX/src/common/micro/mode_ice4_tendencies.F90, lines 285-329
    """
    from __externals__ import (G_RTMIN, LBDAG_MAX, LBDAS_MAX, LBDAS_MIN, LBEXG,
                               LBEXR, LBEXS, LBG, LBR, LBS, LSNOW_T, R_RTMIN,
                               S_RTMIN, TRANS_MP_GAMMAS)

    with computation(PARALLEL), interval(...):
        lbdar = LBR * (rhodref * max(rrt, R_RTMIN)) ** LBEXR if rrt > 0 else 0
        # Translation note : l293 to l298 omitted LLRFR = True (not used in AROME)
        # Translation note : l299 to l301 kept (used in AROME)
        lbdar_rf = lbdar

        if __INLINED(LSNOW_T):
            if rst > 0 and t > 263.15:
                lbdas = (
                    max(min(LBDAS_MAX, 10 ** (14.554 - 0.0423 * t)), LBDAS_MIN)
                    * TRANS_MP_GAMMAS
                )
            elif rst > 0 and t <= 263.15:
                lbdas = (
                    max(min(LBDAS_MAX, 10 ** (6.226 - 0.0106 * t)), LBDAS_MIN)
                    * TRANS_MP_GAMMAS
                )
            else:
                lbdas = 0
        else:
            lbdas = (
                min(LBDAS_MAX, LBS * (rhodref * max(rst, S_RTMIN)) ** LBEXS)
                if rst > 0
                else 0
            )

        lbdag = (
            LBG * (rhodref * max(rgt, G_RTMIN)) ** LBEXG
            if rgt > 0.0
            else 0
        )


def ice4_total_tendencies_update(
    lsfact: Field["float"],
    lvfact: Field["float"],
    th_tnd: Field["float"],
    rv_tnd: Field["float"],
    rc_tnd: Field["float"],
    rr_tnd: Field["float"],
    ri_tnd: Field["float"],
    rs_tnd: Field["float"],
    rg_tnd: Field["float"],
    rchoni: Field["float"],
    rvdeps: Field["float"],
    riaggs: Field["float"],
    riauts: Field["float"],
    rvdepg: Field["float"],
    rcautr: Field["float"],
    rcaccr: Field["float"],
    rrevav: Field["float"],
    rcberi: Field["float"],
    rsmltg: Field["float"],
    rcmltsr: Field["float"],
    rraccss: Field["float"],
    rraccsg: Field["float"],
    rsaccrg: Field["float"],
    rcrimss: Field["float"],
    rcrimsg: Field["float"],
    rsrimcg: Field["float"],
    ricfrrg: Field["float"],
    rrcfrig: Field["float"],
    ricfrr: Field["float"],
    rcwetg: Field["float"],
    riwetg: Field["float"],
    rrwetg: Field["float"],
    rswetg: Field["float"],
    rcdryg: Field["float"],
    ridryg: Field["float"],
    rrdryg: Field["float"],
    rsdryg: Field["float"],
    rgmltr: Field["float"],
    rwetgh: Field["float"]
):
    """
    Aggregate all microphysical process contributions to total tendencies.
    
    This is the master aggregation function that sums all individual
    microphysical process rates to compute the total tendencies for
    potential temperature and all hydrometeor species. Ensures mass
    and energy conservation across all processes.
    
    Parameters
    ----------
    lsfact, lvfact : Field[float]
        Latent heat factors (K).
    th_tnd, rv_tnd, rc_tnd, rr_tnd, ri_tnd, rs_tnd, rg_tnd : Field[float]
        Total tendency fields to be updated (various units/s).
    
    Process rates (all in kg/kg/s unless noted):
    rchoni : Cloud droplet homogeneous freezing
    rvdeps : Vapor deposition on snow
    riaggs : Ice aggregation to snow
    riauts : Ice autoconversion to snow
    rvdepg : Vapor deposition on graupel
    rcautr : Cloud autoconversion to rain
    rcaccr : Cloud accretion by rain
    rrevav : Rain evaporation
    rcberi : Bergeron-Findeisen effect
    rsmltg : Snow melting to graupel
    rcmltsr : Cloud collection by melting snow
    rraccss, rraccsg : Rain accretion processes
    rsaccrg : Snow accretion by graupel
    rcrimss, rcrimsg : Cloud riming processes
    rsrimcg : Snow riming conversion
    ricfrrg, rrcfrig, ricfrr : Contact freezing processes
    rcwetg, riwetg, rrwetg, rswetg : Wet growth collection
    rcdryg, ridryg, rrdryg, rsdryg : Dry growth collection
    rgmltr : Graupel melting
    rwetgh : Wet growth to hail (not used in 6-category scheme)
    
    Notes
    -----
    The potential temperature tendency accounts for latent heat from:
    - Phase changes (L_v for liquid, L_s for ice)
    - Difference L_f = L_s - L_v for freezing/melting
    
    Each mixing ratio tendency is the algebraic sum of all processes
    that produce (+) or consume (-) that species.
    
    Mass conservation: Total water is conserved
    Energy conservation: Latent heat properly accounted
    
    Source Reference:
    PHYEX/src/common/micro/mode_ice4_tendencies.F90, lines 454-559
    """
    with computation(PARALLEL), interval(...):
        th_tnd += (
            rvdepg * lsfact
            + rchoni * (lsfact - lvfact)
            + rvdeps * lsfact
            - rrevav * lvfact
            + rcrimss * (lsfact - lvfact)
            + rcrimsg * (lsfact - lvfact)
            + rraccss * (lsfact - lvfact)
            + rraccsg * (lsfact - lvfact)
            + (rrcfrig - ricfrr) * (lsfact - lvfact)
            + (rcwetg + rrwetg) * (lsfact - lvfact)
            + (rcdryg + rrdryg) * (lsfact - lvfact)
            - rgmltr * (lsfact - lvfact)
            + rcberi * (lsfact - lvfact)
        )

        # (v) Vapor tendency
        rv_tnd += -rvdepg - rvdeps + rrevav

        # (c) Cloud tendency
        rc_tnd += (
            -rchoni
            - rcautr
            - rcaccr
            - rcrimss
            - rcrimsg
            - rcmltsr
            - rcwetg
            - rcdryg
            - rcberi
        )

        # (r) Rain tendency
        rr_tnd += (
            rcautr
            + rcaccr
            - rrevav
            - rraccss
            - rraccsg
            + rcmltsr
            - rrcfrig
            + ricfrr
            - rrwetg
            - rrdryg
            + rgmltr
        )

        # (i) Ice tendency
        ri_tnd += rchoni - riaggs - riauts - ricfrrg - ricfrr - riwetg - ridryg + rcberi

        # (s) Snow tendency
        rs_tnd += (
            rvdeps
            + riaggs
            + riauts
            + rcrimss
            - rsrimcg
            + rraccss
            - rsaccrg
            - rsmltg
            - rswetg
            - rsdryg
        )

        # (g) Graupel tendency
        rg_tnd += (
            rvdepg
            + rcrimsg
            + rsrimcg
            + rraccsg
            + rsaccrg
            + rsmltg
            + ricfrrg
            + rrcfrig
            + rcwetg
            + riwetg
            + rswetg
            + rrwetg
            + rcdryg
            + ridryg
            + rsdryg
            + rrdryg
            - rgmltr
            - rwetgh
        )
