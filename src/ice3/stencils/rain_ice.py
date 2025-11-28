# -*- coding: utf-8 -*-
"""
Rain-ice scheme orchestration stencils.

This module contains utility stencils that support the main rain-ice
microphysics scheme. These functions handle tendency aggregation,
thermodynamic computations, masking, and auxiliary processes like
fog deposition and precipitation fraction calculations.

Source: PHYEX/src/common/micro/rain_ice.F90
"""
from __future__ import annotations

from gt4py.cartesian.gtscript import IJ, PARALLEL, Field, computation, interval


def rain_ice_total_tendencies(
    wr_th: Field["float"],
    wr_v: Field["float"],
    wr_c: Field["float"],
    wr_r: Field["float"],
    wr_i: Field["float"],
    wr_s: Field["float"],
    wr_g: Field["float"],
    ls_fact: Field["float"],
    lv_fact: Field["float"],
    exnref: Field["float"],
    ths: Field["float"],
    rvs: Field["float"],
    rcs: Field["float"],
    rrs: Field["float"],
    ris: Field["float"],
    rss: Field["float"],
    rgs: Field["float"],
    rvheni: Field["float"],
    rv_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
):
    """
    Compute total tendencies and update source terms.
    
    This function calculates the total change in mixing ratios over the
    time step, converts these to tendencies, and adds them to the source
    terms. It also computes the corresponding potential temperature tendency
    accounting for latent heat release/absorption.
    
    Parameters
    ----------
    wr_th, wr_v, wr_c, wr_r, wr_i, wr_s, wr_g : Field[float]
        Initial values for potential temperature and mixing ratios (kg/kg).
        Input as initial values, output as tendencies (kg/kg/s).
    ls_fact : Field[float]
        Latent heat of sublimation divided by heat capacity, L_s/c_p (K).
    lv_fact : Field[float]
        Latent heat of vaporization divided by heat capacity, L_v/c_p (K).
    exnref : Field[float]
        Reference Exner function (dimensionless).
    ths, rvs, rcs, rrs, ris, rss, rgs : Field[float]
        Source terms (tendencies) to be updated (units/s).
    rvheni : Field[float]
        Vapor consumed by heterogeneous nucleation (kg/kg/s).
    rv_t, rc_t, rr_t, ri_t, rs_t, rg_t : Field[float]
        Current mixing ratios at time t (kg/kg).
        
    Notes
    -----
    Process Steps:
    1. Compute mixing ratio changes: Δr = r_initial - r_current
    2. Convert to tendencies: tendency = Δr / Δt
    3. Calculate theta tendency from latent heat:
       Δθ = (Δr_liquid * L_v + Δr_ice * L_s) / c_p
    4. Add nucleation contribution to ice and vapor tendencies
    5. Update all source terms
    
    The potential temperature change accounts for:
    - Condensation/evaporation of liquid (using L_v)
    - Deposition/sublimation of ice (using L_s)
    - Heterogeneous nucleation (vapor → ice)
    
    External Parameters:
    - INV_TSTEP: Inverse of time step (1/s)
    
    Source Reference:
    PHYEX/src/common/micro/rain_ice.F90, lines 693-728
    """
    from __externals__ import INV_TSTEP

    with computation(PARALLEL), interval(...):
        # Translation note ls, lv replaced by ls_fact, lv_fact

        # Hydrometeor tendency
        wr_v = (wr_v - rv_t) * INV_TSTEP
        wr_c = (wr_c - rc_t) * INV_TSTEP
        wr_r = (wr_r - rr_t) * INV_TSTEP
        wr_i = (wr_i - ri_t) * INV_TSTEP
        wr_s = (wr_s - rs_t) * INV_TSTEP
        wr_g = (wr_g - rg_t) * INV_TSTEP

        # Theta tendency
        wr_th = (wr_c + wr_r) * lv_fact + (wr_i + wr_s + wr_g) * ls_fact

        # Tendencies to sources, taking nucleation into account (rv_heni)
        ths += wr_th + rvheni * ls_fact
        rvs += wr_v - rvheni
        rcs += wr_c
        rrs += wr_r
        ris += wr_i + rvheni
        rss += wr_s
        rgs += wr_g


def rain_ice_thermo(
    exn: Field["float"],
    ls_fact: Field["float"],
    lv_fact: Field["float"],
    th_t: Field["float"],
    rv_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
):
    """
    Compute thermodynamic functions for the microphysics scheme.
    
    This function calculates the latent heat factors (L/c_p) accounting for
    temperature-dependent variations in latent heat and the specific heat
    capacity of the moist air parcel.
    
    Parameters
    ----------
    exn : Field[float]
        Exner function π = (p/p₀)^(R_d/c_p) (dimensionless).
    ls_fact : Field[float]
        Output: Latent heat of sublimation over heat capacity, L_s/c_p (K).
    lv_fact : Field[float]
        Output: Latent heat of vaporization over heat capacity, L_v/c_p (K).
    th_t : Field[float]
        Potential temperature at time t (K).
    rv_t, rc_t, rr_t, ri_t, rs_t, rg_t : Field[float]
        Mixing ratios at time t for vapor, cloud, rain, ice, snow, graupel (kg/kg).
        
    Notes
    -----
    The effective heat capacity of moist air depends on the mixing ratios:
    c_p,moist = c_pd + c_pv·r_v + c_l·(r_c + r_r) + c_i·(r_i + r_s + r_g)
    
    The latent heats vary with temperature:
    L_v(T) = L_v(T_t) + (c_pv - c_l)·(T - T_t)
    L_s(T) = L_s(T_t) + (c_pv - c_i)·(T - T_t)
    
    These factors are used throughout the microphysics to convert
    mixing ratio changes to temperature changes and vice versa.
    
    External Parameters:
    - CPD, CPV, CL, CI: Specific heats (J/(kg·K))
    - LVTT, LSTT: Latent heats at triple point (J/kg)
    - TT: Triple point temperature (K)
    
    Source Reference:
    PHYEX/src/common/micro/rain_ice.F90, lines 367-396
    """
    from __externals__ import CI, CL, CPD, CPV, LSTT, LVTT, TT

    with computation(PARALLEL), interval(...):
        divider = CPD + CPV * rv_t + CL * (rc_t + rr_t) + CI * (ri_t + rs_t + rg_t)
        t = th_t * exn
        ls_fact = (LSTT + (CPV - CI) * (t - TT)) / divider
        lv_fact = (LVTT + (CPV - CL) * (t - TT)) / divider


def rain_ice_mask(
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    ldmicro: Field["bool"],
):
    """
    Determine which grid points require microphysical computations.
    
    This function creates a boolean mask indicating grid points with
    sufficient hydrometeor content to warrant microphysical calculations.
    This optimization avoids unnecessary computations in clear air.
    
    Parameters
    ----------
    rc_t, rr_t, ri_t, rs_t, rg_t : Field[float]
        Mixing ratios for cloud, rain, ice, snow, graupel at time t (kg/kg).
    ldmicro : Field[bool]
        Output: Mask indicating grid points needing microphysics (True/False).
        
    Notes
    -----
    A grid point requires microphysical computation if ANY hydrometeor
    species exceeds its threshold:
    - rc_t > C_RTMIN (cloud droplets)
    - rr_t > R_RTMIN (rain)
    - ri_t > I_RTMIN (ice crystals)
    - rs_t > S_RTMIN (snow)
    - rg_t > G_RTMIN (graupel)
    
    Typical threshold values are O(10⁻⁸) kg/kg.
    
    External Parameters:
    - C_RTMIN, R_RTMIN, I_RTMIN, S_RTMIN, G_RTMIN: Minimum thresholds (kg/kg)
    """
    from __externals__ import C_RTMIN, G_RTMIN, I_RTMIN, R_RTMIN, S_RTMIN

    with computation(PARALLEL), interval(...):
        ldmicro = (
            rc_t > C_RTMIN
            or rr_t > R_RTMIN
            or ri_t > I_RTMIN
            or rs_t > S_RTMIN
            or rg_t > G_RTMIN
        )


def initial_values_saving(
    wr_th: Field["float"],
    wr_v: Field["float"],
    wr_c: Field["float"],
    wr_r: Field["float"],
    wr_i: Field["float"],
    wr_s: Field["float"],
    wr_g: Field["float"],
    th_t: Field["float"],
    rv_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    evap3d: Field["float"],
    rainfr: Field["float"],
):
    """
    Save initial values before microphysical processing.
    
    This function stores the initial state of all prognostic variables
    before the microphysical scheme modifies them. These saved values
    are later used to compute total tendencies.
    
    Parameters
    ----------
    wr_th, wr_v, wr_c, wr_r, wr_i, wr_s, wr_g : Field[float]
        Output: Storage fields for initial values of theta and mixing ratios.
    th_t, rv_t, rc_t, rr_t, ri_t, rs_t, rg_t : Field[float]
        Current values to be saved.
    evap3d : Field[float]
        Output: 3D evaporation diagnostic (initialized to 0 if LWARM=True).
    rainfr : Field[float]
        Output: Rain fraction diagnostic (initialized to 0).
        
    Notes
    -----
    This simple copy operation creates a snapshot of the atmosphere state.
    The difference between these saved values and the final values after
    microphysics determines the tendencies.
    
    The LWARM flag (True for AROME) controls whether evap3d is computed.
    
    External Parameters:
    - LWARM: Flag for warm processes (True for AROME)
    
    Source Reference:
    PHYEX/src/common/micro/rain_ice.F90, lines 424-444
    """
    from __externals__ import LWARM

    with computation(PARALLEL), interval(...):
        wr_th = th_t
        wr_v = rv_t
        wr_c = rc_t
        wr_r = rr_t
        wr_i = ri_t
        wr_s = rs_t
        wr_g = rg_t

        # LWARM is True for AROME
        if __INLINED(LWARM):
            evap3d = 0
        rainfr = 0


def ice4_precipitation_fraction_sigma(
    sigs: Field["float"],
    sigma_rc: Field["float"]
):
    """
    Compute cloud water variance for subgrid precipitation scheme.
    
    This function calculates the variance of cloud water mixing ratio
    from the standard deviation of saturation, used in PDF-based
    precipitation parameterizations.
    
    Parameters
    ----------
    sigs : Field[float]
        Standard deviation of saturation (dimensionless).
    sigma_rc : Field[float]
        Output: Variance of cloud water mixing ratio (dimensionless).
        
    Notes
    -----
    This is used when:
    - CSUBG_AUCV_RC = 'PDF' (PDF method for autoconversion)
    - CSUBG_PR_PDF = 'SIGM' (Sigma method for precipitation)
    
    The variance σ²_rc = σ²_s relates subgrid variability in saturation
    to variability in cloud water for subgrid precipitation calculations.
    
    Source Reference:
    PHYEX/src/common/micro/rain_ice.F90, lines 492-498
    """
    with computation(PARALLEL), interval(...):
        sigma_rc = sigs**2


def rain_fraction_sedimentation(
    wr_r: Field["float"],
    wr_s: Field["float"],
    wr_g: Field["float"],
    rrs: Field["float"],
    rss: Field["float"],
    rgs: Field["float"],
):
    """
    Initialize precipitation mixing ratios for sedimentation.
    
    This function prepares the precipitation fields at the top of the
    domain for the sedimentation calculation by computing the amount
    produced during the time step.
    
    Parameters
    ----------
    wr_r, wr_s, wr_g : Field[float]
        Output: Rain, snow, graupel mixing ratios at top boundary (kg/kg).
    rrs, rss, rgs : Field[float]
        Source terms (tendencies) for rain, snow, graupel (kg/kg/s).
        
    Notes
    -----
    Applied only at the top level (k=0) of the domain.
    The amount is computed as: r = tendency × Δt
    
    This sets the upper boundary condition for the sedimentation scheme,
    which then propagates precipitation downward through the column.
    
    External Parameters:
    - TSTEP: Time step (s)
    
    Source Reference:
    PHYEX/src/common/micro/rain_ice.F90, lines 792-801
    """
    from __externals__ import TSTEP

    with computation(PARALLEL), interval(0, 1):
        wr_r = rrs * TSTEP
        wr_s = rss * TSTEP
        wr_g = rgs * TSTEP


def ice4_rainfr_vert(
    prfr: Field["float"], rr: Field["float"], rs: Field["float"], rg: Field["float"]
):
    """
    Compute vertical rain fraction for diagnostics.
    
    This function determines the vertical extent of precipitation by
    propagating the rain fraction upward from levels with significant
    precipitation content. Used for diagnostic purposes and analysis.
    
    Parameters
    ----------
    prfr : Field[float]
        Input/Output: Rain fraction (0-1). Updated by backward sweep.
    rr, rs, rg : Field[float]
        Rain, snow, graupel mixing ratios (kg/kg).
        
    Notes
    -----
    Algorithm (backward/upward sweep):
    1. Start from bottom of domain (highest index)
    2. At each level, if precipitation exists:
       - Take maximum of current and level below
       - If no rain fraction, set to 1
    3. If no precipitation, set to 0
    
    This creates a rain fraction field that:
    - Is 1 where precipitation is present
    - Extends upward through the cloud column
    - Is 0 in clear air
    
    External Parameters:
    - R_RTMIN, S_RTMIN, G_RTMIN: Minimum thresholds (kg/kg)
    
    Source Reference:
    PHYEX/src/common/micro/rain_ice.F90, lines 792-801
    """
    from __externals__ import G_RTMIN, R_RTMIN, S_RTMIN

    with computation(BACKWARD), interval(0, -1):
        if rr > R_RTMIN or rs > S_RTMIN or rg > G_RTMIN:
            prfr[0, 0, 0] = max(prfr[0, 0, 0], prfr[0, 0, 1])
            if prfr == 0:
                prfr = 1
        else:
            prfr = 0


def fog_deposition(
    rcs: Field["float"],
    rc_t: Field["float"],
    rhodref: Field["float"],
    dzz: Field["float"],
    inprc: Field[IJ, "float"],
):
    """
    Compute fog deposition on vegetation (surface process).
    
    This function calculates the removal of cloud droplets (fog) by
    deposition on vegetation and other surface features. Not activated
    in standard AROME configuration.
    
    Parameters
    ----------
    rcs : Field[float]
        Input/Output: Cloud droplet source term, reduced by deposition (kg/kg/s).
    rc_t : Field[float]
        Cloud droplet mixing ratio (kg/kg).
    rhodref : Field[float]
        Reference air density (kg/m³).
    dzz : Field[float]
        Vertical grid spacing (m).
    inprc : Field[IJ, float]
        Output: Accumulated deposition on surface (m).
        Liquid water equivalent depth.
        
    Notes
    -----
    The deposition rate is parameterized using a deposition velocity:
    - Removal rate = V_dep × r_c / Δz
    
    Applied only at the surface level (k=0) in a forward computation.
    
    The accumulated deposition is converted to liquid water equivalent
    depth (meters) for diagnostic purposes.
    
    Physical Basis:
    - Fog droplets impact and stick to vegetation
    - Parameterized by empirical deposition velocity
    - Important for visibility forecasts near surface
    
    External Parameters:
    - VDEPOSC: Deposition velocity (m/s)
    - RHOLW: Liquid water density (kg/m³)
    - Activated if LDEPOSC=True in rain_ice.F90
    
    Source Reference:
    PHYEX/src/common/micro/rain_ice.F90.func.h, lines 816-830
    """
    from __externals__ import RHOLW, VDEPOSC

    # Note : activated if LDEPOSC is True in rain_ice.F90
    with computation(FORWARD), interval(0, 1):
        rcs -= VDEPOSC * rc_t / dzz
        inprc[0, 0] += VDEPOSC * rc_t[0, 0, 0] * rhodref[0, 0, 0] / RHOLW
