# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    BACKWARD,
    FORWARD,
    IJ,
    PARALLEL,
    Field,
    computation,
    function,
    interval,
    log,
    max,
    min,
)

from ..functions.upwind_sedimentation import (
    instant_precipitation,
    maximum_time_step,
    mixing_ratio_update,
    upper_air_flux,
)


# Helper functions matching Fortran subroutines
@function
def ray(sea: Field[IJ, "float"]):
    """Compute cloud mean radius parameter based on sea fraction"""
    # Simplified: returns 1.0 (would need gamma function ratios for full implementation)
    return 1.0


@function 
def lbc(sea: Field[IJ, "float"]):
    """Compute LBC parameter weighted by sea fraction"""
    from __externals__ import LBC_LAND, LBC_SEA
    
    return (1.0 - sea) * LBC_LAND + sea * LBC_SEA


@function
def fsedc(sea: Field[IJ, "float"]):
    """Compute FSEDC parameter weighted by sea fraction"""
    from __externals__ import FSEDC_LAND, FSEDC_SEA
    
    return (1.0 - sea) * FSEDC_LAND + sea * FSEDC_SEA


@function
def conc3d(town: Field[IJ, "float"], sea: Field[IJ, "float"]):
    """Compute 3D concentration based on town and sea fractions"""
    from __externals__ import CONC_LAND, CONC_SEA, CONC_URBAN
    
    return (1.0 - town) * ((1.0 - sea) * CONC_LAND + sea * CONC_SEA) + town * CONC_URBAN


@function
def other_species_velocity(
    fsed: "float",
    exsed: "float", 
    content: "float",
    rhodref: "float"
):
    """Compute terminal velocity for rain, snow, graupel, hail"""
    from __externals__ import CEXVT
    
    return fsed * content ** (exsed - 1.0) * rhodref ** (exsed - CEXVT - 1.0)


@function
def pristine_ice_velocity(content: "float", rhodref: "float"):
    """Compute terminal velocity for pristine ice"""
    from __externals__ import FSEDI, EXCSEDI, CEXVT
    
    return FSEDI * rhodref ** (-CEXVT) * max(0.05e6, -0.15319e6 - 0.021454e6 * log(rhodref * content)) ** EXCSEDI


@function
def fwsed1(wsedw: "float", tstep: "float", dzz: "float", rhodref: "float", content: "float", invtstep: "float"):
    """Compute weighted sedimentation flux (first component)"""
    return min(rhodref * dzz * content * invtstep, wsedw * rhodref * content)


@function 
def fwsed2(wsedw: "float", tstep: "float", dzz: "float", wsedw_above: "float"):
    """Compute weighted sedimentation flux (second component from level above)"""
    return max(0.0, 1.0 - dzz / (tstep * wsedw)) * wsedw_above


# "PHYEX/src/common/micro/mode_ice4_sedimentation_stat.F90"
def sedimentation_stat(
    rhodref: Field["float"],
    dzz: Field["float"],
    pabs_t: Field["float"],
    th_t: Field["float"],
    rcs: Field["float"],
    rrs: Field["float"],
    ris: Field["float"],
    rss: Field["float"],
    rgs: Field["float"],
    sea: Field[IJ, "float"],
    town: Field[IJ, "float"],
    fpr_c: Field["float"],
    fpr_r: Field["float"],
    fpr_i: Field["float"],
    fpr_s: Field["float"],
    fpr_g: Field["float"],
    inprr: Field[IJ, "float"],
    inprc: Field[IJ, "float"],
    inpri: Field[IJ, "float"],
    inprs: Field[IJ, "float"],
    inprg: Field[IJ, "float"],
):
    """
    Compute gravitational sedimentation using statistical method.
    
    This stencil implements the statistical sedimentation scheme for ICE4
    microphysics, computing the vertical transport of hydrometeors due to
    gravitational settling. It uses size-distribution-averaged terminal
    velocities and backward-in-time flux calculations to ensure numerical
    stability and mass conservation.
    
    Parameters
    ----------
    rhodref : Field[float]
        Reference air density (kg/m³). Input field.
    dzz : Field[float]
        Vertical grid spacing (m). Input field.
    pabs_t : Field[float]
        Absolute pressure (Pa). Input field.
    th_t : Field[float]
        Potential temperature (K). Input field.
    rcs : Field[float]
        Cloud droplet tendency (kg/kg/s). Modified in place.
    rrs : Field[float]
        Rain tendency (kg/kg/s). Modified in place.
    ris : Field[float]
        Ice crystal tendency (kg/kg/s). Modified in place.
    rss : Field[float]
        Snow tendency (kg/kg/s). Modified in place.
    rgs : Field[float]
        Graupel tendency (kg/kg/s). Modified in place.
    sea : Field[IJ, float]
        Sea fraction (0-1). Input field (2D).
        Used to weight microphysical parameters.
    town : Field[IJ, float]
        Urban fraction (0-1). Input field (2D).
        Used to weight cloud droplet concentration.
    fpr_c, fpr_r, fpr_i, fpr_s, fpr_g : Field[float]
        Sedimentation fluxes (kg/(m²·s)). Output fields.
    inprr, inprc, inpri, inprs, inprg : Field[IJ, float]
        Instantaneous surface precipitation rates (m/s). Output fields (2D).
    
    Returns
    -------
    None
        Modifies tendency and flux fields in place.
    
    Notes
    -----
    **Statistical Sedimentation Method:**
    
    The statistical approach assumes particles follow a gamma size
    distribution and computes mean terminal velocities based on:
    - Size distribution parameters (λ, N)
    - Power-law velocity-diameter relationships
    - Environmental corrections (air density, pressure)
    
    **Terminal Velocity Formulations:**
    
    1. **Cloud Droplets:**
       V_c = ρ^(-CEXVT) × λ_c^(-D_c) × C_c × F_SEDC
       
       where:
       - λ_c: Slope parameter (m⁻¹)
       - C_c: Correction factor C_c = CC × (1 + 1.26λ_mfp/r_mean)
       - λ_mfp: Mean free path ~6.6×10⁻⁸ m at surface
       - Accounts for slip-flow regime (Cunningham correction)
    
    2. **Rain:**
       V_r = F_SEDR × r_r^(EXSEDR-1) × ρ^(EXSEDR-CEXVT-1)
       
       Power law with exponent EXSEDR ≈ 0.25
    
    3. **Pristine Ice:**
       V_i = F_SEDI × ρ^(-CEXVT) × D_max^EXCSEDI
       
       where D_max = max(50 μm, f(ρ×r_i))
       Special formulation for plate/columnar crystals
    
    4. **Snow:**
       V_s = F_SEDS × r_s^(EXSEDS-1) × ρ^(EXSEDS-CEXVT-1)
       
       Aggregate fall speed, EXSEDS ≈ 0.125
    
    5. **Graupel:**
       V_g = F_SEDG × r_g^(EXSEDG-1) × ρ^(EXSEDG-CEXVT-1)
       
       Dense rimed particles, EXSEDG ≈ 0.33
    
    **Flux Computation (Backward Sweep):**
    
    Fluxes computed from top to bottom (BACKWARD computation):
    
    F_x(k) = F_wsed1 + F_wsed2
    
    where:
    - F_wsed1 = min(ρ×Δz×r_x/Δt, V_x×ρ×r_x)  [local contribution]
    - F_wsed2 = max(0, 1 - Δz/(Δt×V_x)) × F_x(k+1)  [from above]
    
    The first term ensures CFL stability, the second propagates flux
    from the level above.
    
    **Tendency Update:**
    
    Sedimentation tendency:
    
    dr_x/dt = (1/(ρ×Δz)) × [F_x(k+1) - F_x(k)]
    
    Positive: convergence (flux in > flux out), mass increases
    Negative: divergence (flux out > flux in), mass decreases
    
    **Surface Precipitation:**
    
    Instantaneous surface precipitation rate:
    
    P_x = F_x(k=1) / ρ_water  (m/s or mm/s)
    
    Converted from mass flux to depth rate using ρ_water = 1000 kg/m³
    
    **Parameter Dependence on Surface Type:**
    
    Parameters vary with sea/land/urban fractions:
    
    - Cloud: λ_bc, F_SEDC, N (concentration)
      * Sea: Higher concentration, smaller drops
      * Land: Lower concentration, larger drops
      * Urban: Highest concentration (pollution)
    
    - Rain/snow/graupel: Independent of surface type
      (large particles, surface effects negligible)
    
    **Weighting Formula:**
    
    param = (1 - f_town) × [(1 - f_sea)×param_land + f_sea×param_sea]
            + f_town × param_urban
    
    **Numerical Stability:**
    
    CFL condition automatically satisfied by min() in F_wsed1:
    
    V × Δt / Δz ≤ 1
    
    This prevents flux from exceeding available mass in grid cell.
    
    **Mass Conservation:**
    
    Total column mass conserved:
    
    ∫ρ×Δz×(dr/dt) dz = F_top - F_surface
    
    With F_top = 0 at model top:
    ∫ρ×Δz×(dr/dt) dz = -F_surface
    
    All mass lost from column appears as surface precipitation.
    
    **Typical Terminal Velocities:**
    
    Cloud droplets: ~1 cm/s (very slow)
    Ice crystals: ~10-50 cm/s (slow)
    Rain: ~1-5 m/s (moderate to fast)
    Snow: ~0.5-1.5 m/s (moderate)
    Graupel: ~2-5 m/s (fast)
    
    **Computational Flow:**
    
    1. Convert tendencies to mixing ratios (×TSTEP)
    2. Initialize fluxes to zero
    3. Loop from top to bottom (BACKWARD):
       a. Compute terminal velocity
       b. Calculate flux from local content
       c. Add flux from level above
    4. Update tendencies from flux divergence
    5. Extract surface precipitation from bottom flux
    
    **Advantages of Statistical Method:**
    
    - Computationally efficient (one pass)
    - Unconditionally stable (CFL built-in)
    - Mass conservative
    - Smooth fields (no numerical dispersion)
    
    **Comparison with Upwind Method:**
    
    Statistical (this function):
    - Mean velocities from size distribution
    - Backward sweep, single time step
    - Faster, simpler
    - Used in AROME operational
    
    Upwind (separate function):
    - Explicit particle tracking
    - Multiple substeps possible
    - More accurate for resolution
    - Used in research mode
    
    **Physical Process:**
    
    Sedimentation removes hydrometeors from upper levels and deposits
    them at lower levels or surface:
    
    - Cloud → slow removal, minimal surface impact
    - Rain → fast removal, surface precipitation
    - Snow/graupel → moderate removal, surface accumulation
    - Ice → very slow removal, redistributes in cloud
    
    Source Reference
    ----------------
    PHYEX/src/common/micro/mode_ice4_sedimentation_stat.F90
    
    See Also
    --------
    upwind_sedimentation : Alternative sedimentation method
    ice4_stepping : Time integration framework
    
    Examples
    --------
    >>> # Warm rain sedimentation
    >>> # r_r = 1e-3 kg/kg at z=500m
    >>> # V_r ≈ 5 m/s, precip at surface in ~100 s
    
    >>> # Snow sedimentation
    >>> # r_s = 5e-4 kg/kg at z=2000m  
    >>> # V_s ≈ 1 m/s, precip at surface in ~2000 s
    
    >>> # Cloud droplets (negligible sedimentation)
    >>> # r_c = 5e-4 kg/kg, V_c ≈ 0.01 m/s
    >>> # Removed mainly by collection, not sedimentation
    """
    from __externals__ import (
        C_RTMIN,
        CC,
        CEXVT,
        DC,
        EXSEDG,
        EXSEDR,
        EXSEDS,
        FSEDG,
        FSEDR,
        FSEDS,
        G_RTMIN,
        I_RTMIN,
        LBEXC,
        R_RTMIN,
        RHOLW,
        S_RTMIN,
        TSTEP,
    )

    # Convert tendencies to mixing ratios
    with computation(PARALLEL), interval(...):
        rc_t = rcs * TSTEP
        rr_t = rrs * TSTEP
        ri_t = ris * TSTEP
        rs_t = rss * TSTEP
        rg_t = rgs * TSTEP

    # Initialize fluxes
    with computation(PARALLEL), interval(...):
        fpr_c = 0.0
        fpr_r = 0.0
        fpr_i = 0.0
        fpr_s = 0.0
        fpr_g = 0.0

    # Compute the sedimentation fluxes from top to bottom
    with computation(BACKWARD), interval(...):
        TSTEP__rho_dz = TSTEP / (rhodref * dzz)
        zinvtstep = 1.0 / TSTEP
        
        # Compute parameters for cloud (depend on sea/land/town)
        _ray = ray(sea)
        _lbc = lbc(sea)
        _fsedc = fsedc(sea)
        _conc3d = conc3d(town, sea)

        # 2.1 Cloud droplets
        qp = fpr_c[0, 0, 1] * TSTEP__rho_dz
        
        # Compute terminal velocities
        if rc_t > C_RTMIN:
            wlbda = 6.6e-8 * (101325.0 / pabs_t) * (th_t / 293.15)
            wlbdc = (_lbc * _conc3d / (rhodref * rc_t)) ** LBEXC
            cc = CC * (1.0 + 1.26 * wlbda * wlbdc / _ray)
            wsedw1 = rhodref ** (-CEXVT) * wlbdc ** (-DC) * cc * _fsedc
        else:
            wsedw1 = 0.0
            
        if qp > C_RTMIN:
            wlbda = 6.6e-8 * (101325.0 / pabs_t) * (th_t / 293.15)
            wlbdc = (_lbc * _conc3d / (rhodref * qp)) ** LBEXC
            cc = CC * (1.0 + 1.26 * wlbda * wlbdc / _ray)
            wsedw2 = rhodref ** (-CEXVT) * wlbdc ** (-DC) * cc * _fsedc
        else:
            wsedw2 = 0.0
        
        fpr_c = fwsed1(wsedw1, TSTEP, dzz, rhodref, rc_t, zinvtstep)
        if wsedw2 != 0.0:
            fpr_c = fpr_c + fwsed2(wsedw2, TSTEP, dzz, fpr_c[0, 0, 1])

        # 2.2 Rain
        qp = fpr_r[0, 0, 1] * TSTEP__rho_dz
        
        if rr_t > R_RTMIN:
            wsedw1 = other_species_velocity(FSEDR, EXSEDR, rr_t, rhodref)
        else:
            wsedw1 = 0.0
            
        if qp > R_RTMIN:
            wsedw2 = other_species_velocity(FSEDR, EXSEDR, qp, rhodref)
        else:
            wsedw2 = 0.0
        
        fpr_r = fwsed1(wsedw1, TSTEP, dzz, rhodref, rr_t, zinvtstep)
        if wsedw2 != 0.0:
            fpr_r = fpr_r + fwsed2(wsedw2, TSTEP, dzz, fpr_r[0, 0, 1])

        # 2.3 Pristine ice
        qp = fpr_i[0, 0, 1] * TSTEP__rho_dz
        
        if ri_t > max(I_RTMIN, 1.0e-7):
            wsedw1 = pristine_ice_velocity(ri_t, rhodref)
        else:
            wsedw1 = 0.0
            
        if qp > max(I_RTMIN, 1.0e-7):
            wsedw2 = pristine_ice_velocity(qp, rhodref)
        else:
            wsedw2 = 0.0
        
        fpr_i = fwsed1(wsedw1, TSTEP, dzz, rhodref, ri_t, zinvtstep)
        if wsedw2 != 0.0:
            fpr_i = fpr_i + fwsed2(wsedw2, TSTEP, dzz, fpr_i[0, 0, 1])

        # 2.4 Snow
        qp = fpr_s[0, 0, 1] * TSTEP__rho_dz
        
        if rs_t > S_RTMIN:
            wsedw1 = other_species_velocity(FSEDS, EXSEDS, rs_t, rhodref)
        else:
            wsedw1 = 0.0
            
        if qp > S_RTMIN:
            wsedw2 = other_species_velocity(FSEDS, EXSEDS, qp, rhodref)
        else:
            wsedw2 = 0.0
        
        fpr_s = fwsed1(wsedw1, TSTEP, dzz, rhodref, rs_t, zinvtstep)
        if wsedw2 != 0.0:
            fpr_s = fpr_s + fwsed2(wsedw2, TSTEP, dzz, fpr_s[0, 0, 1])

        # 2.5 Graupel
        qp = fpr_g[0, 0, 1] * TSTEP__rho_dz
        
        if rg_t > G_RTMIN:
            wsedw1 = other_species_velocity(FSEDG, EXSEDG, rg_t, rhodref)
        else:
            wsedw1 = 0.0
            
        if qp > G_RTMIN:
            wsedw2 = other_species_velocity(FSEDG, EXSEDG, qp, rhodref)
        else:
            wsedw2 = 0.0
        
        fpr_g = fwsed1(wsedw1, TSTEP, dzz, rhodref, rg_t, zinvtstep)
        if wsedw2 != 0.0:
            fpr_g = fpr_g + fwsed2(wsedw2, TSTEP, dzz, fpr_g[0, 0, 1])

    # 3. Source - Calculate tendencies
    with computation(PARALLEL), interval(...):
        rcs = rcs + TSTEP__rho_dz * (fpr_c[0, 0, 1] - fpr_c[0, 0, 0]) / TSTEP
        rrs = rrs + TSTEP__rho_dz * (fpr_r[0, 0, 1] - fpr_r[0, 0, 0]) / TSTEP
        ris = ris + TSTEP__rho_dz * (fpr_i[0, 0, 1] - fpr_i[0, 0, 0]) / TSTEP
        rss = rss + TSTEP__rho_dz * (fpr_s[0, 0, 1] - fpr_s[0, 0, 0]) / TSTEP
        rgs = rgs + TSTEP__rho_dz * (fpr_g[0, 0, 1] - fpr_g[0, 0, 0]) / TSTEP

    # Instantaneous fluxes at ground level
    with computation(FORWARD), interval(0, 1):
        inprc = fpr_c / RHOLW
        inprr = fpr_r / RHOLW
        inpri = fpr_i / RHOLW
        inprs = fpr_s / RHOLW
        inprg = fpr_g / RHOLW


def upwind_sedimentation(
    rhodref: Field["float"],
    dzz: Field["float"],
    pabs_t: Field["float"],
    th_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    rcs: Field["float"],
    rrs: Field["float"],
    ris: Field["float"],
    rss: Field["float"],
    rgs: Field["float"],
    inst_rr: Field[IJ, "float"],
    inst_rc: Field[IJ, "float"],
    inst_ri: Field[IJ, "float"],
    inst_rs: Field[IJ, "float"],
    inst_rg: Field[IJ, "float"],
    fpr_c: Field["float"],
    fpr_r: Field["float"],
    fpr_s: Field["float"],
    fpr_i: Field["float"],
    fpr_g: Field["float"],
    sea: Field[IJ, "float"],
    town: Field[IJ, "float"],
    remaining_time: Field[IJ, "float"],
):
    """Compute sedimentation of contents (rx_t) with piecewise
    constant method.

    Args:
        rhodref (Field[float]): dry density of air
        dzz (Field[float]): spacing between cell centers
        pabs_t (Field[float]): absolute pressure at t
        th_t (Field[float]): potential temperature at t
        rc_t (Field[float]): cloud droplets m.r. at t
        rr_t (Field[float]): rain m.r. at t
        ri_t (Field[float]): ice m.r. at t
        rs_t (Field[float]): snow m.r. at t
        rg_t (Field[float]): graupel m.r. at t
        rcs (Field[float]): cloud droplets m.r. tendency
        rrs (Field[float]): rain m.r. tendency
        ris (Field[float]): ice m.r. tendency
        rss (Field[float]): snow m.r. tendency
        rgs (Field[float]): graupel m.r. tendency
        inst_rr (Field[IJ, float]): instant precip of rain
        inst_rc (Field[IJ, float]): instant precip of cloud droplets
        inst_ri (Field[IJ, float]): instant precip of ice
        inst_rs (Field[IJ, float]): instant precip of snow
        inst_rg (Field[IJ, float]): instant precipe of graupel
        fpr_c (Field[float]): _description_
        fpr_r (Field[float]): _description_
        fpr_s (Field[float]): _description_
        fpr_i (Field[float]): _description_
        fpr_g (Field[float]): _description_
        sea (Field[float]): mask for presence of sea
        town (Field[float]): mask for presence of town
        remaining_time (Field[IJ, float]): _description_
    """

    from __externals__ import (
        C_RTMIN,
        CC,
        CEXVT,
        CPD,
        G_RTMIN,
        I_RTMIN,
        LBEXC,
        P00,
        R_RTMIN,
        RD,
        S_RTMIN,
        TSTEP,
        TT,
    )

    with computation(PARALLEL), interval(...):
        dt__rho_dz = TSTEP / (rhodref * dzz)
        oorhodz = 1.0 / (rhodref * dzz)

    # 2. Compute the fluxes
    with computation(PARALLEL), interval(...):
        rcs -= rc_t / TSTEP
        ris -= ri_t / TSTEP
        rrs -= rr_t / TSTEP
        rss -= rs_t / TSTEP
        rgs -= rg_t / TSTEP

        wsed_c = 0.0
        wsed_r = 0.0
        wsed_i = 0.0
        wsed_s = 0.0
        wsed_g = 0.0

        remaining_time = TSTEP

    # Compute parameters
    with computation(PARALLEL), interval(...):
        _ray = ray(sea)
        _lbc = lbc(sea)
        _fsedc = fsedc(sea)
        _conc3d = conc3d(town, sea)

    ## 2.1 For cloud droplets
    with computation(PARALLEL), interval(...):
        wlbdc = (_lbc * _conc3d / (rhodref * rc_t)) ** LBEXC
        _ray /= wlbdc
        t = th_t * (pabs_t / P00) ** (RD / CPD)
        wlbda = 6.6e-8 * (P00 / pabs_t) * (t / TT)
        cc = CC * (1.0 + 1.26 * wlbda / _ray)
        wsed = rhodref ** (-CEXVT + 1.0) * wlbdc * cc * _fsedc

    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            C_RTMIN, rhodref, max_tstep, rc_t, dzz, wsed_c, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_rc[0, 0] += instant_precipitation(wsed_c, max_tstep, TSTEP)

    with computation(PARALLEL), interval(...):
        rcs = mixing_ratio_update(max_tstep, oorhodz, wsed_s, rcs, rc_t, TSTEP)
        fpr_c += upper_air_flux(wsed_s, max_tstep, TSTEP)

    ## 2.2 for ice
    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            I_RTMIN, rhodref, max_tstep, ri_t, dzz, wsed_i, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_ri[0, 0] += instant_precipitation(wsed_i, max_tstep, TSTEP)

    with computation(PARALLEL), interval(...):
        rcs = mixing_ratio_update(max_tstep, oorhodz, wsed_i, ris, ri_t, TSTEP)
        fpr_i += upper_air_flux(wsed_i, max_tstep, TSTEP)

    ## 2.3 for rain
    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            R_RTMIN, rhodref, max_tstep, rr_t, dzz, wsed_r, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_rr[0, 0] += instant_precipitation(wsed, max_tstep, TSTEP)

    with computation(PARALLEL), interval(...):
        rrs = mixing_ratio_update(max_tstep, oorhodz, wsed, rrs, rr_t, TSTEP)
        fpr_r += upper_air_flux(wsed, max_tstep, TSTEP)

    ## 2.4. for snow
    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            S_RTMIN, rhodref, max_tstep, rs_t, dzz, wsed, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_rs[0, 0] += instant_precipitation(wsed_s, max_tstep, TSTEP)

    with computation(PARALLEL), interval(...):
        rcs = mixing_ratio_update(max_tstep, oorhodz, wsed_s, rss, rs_t, TSTEP)
        fpr_s += upper_air_flux(wsed_s, max_tstep, TSTEP)

    # 2.5. for graupel
    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            G_RTMIN, rhodref, max_tstep, rg_t, dzz, wsed_g, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_rg[0, 0] += instant_precipitation(wsed_g, max_tstep, TSTEP)

    with computation(PARALLEL), interval(...):
        rcs = mixing_ratio_update(max_tstep, oorhodz, wsed_g, rgs, rg_t, TSTEP)
        fpr_g += upper_air_flux(wsed_g, max_tstep, TSTEP)
