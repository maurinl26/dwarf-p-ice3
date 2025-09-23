# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    Field,
    function,
    computation,
    PARALLEL,
    BACKWARD,
    FORWARD,
    interval,
    IJ,
    log,
)
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method

from ice3_gt4py.functions.sedimentation_flux import (
    other_species,
    pristine_ice,
    weighted_sedimentation_flux_1,
    weighted_sedimentation_flux_2,
)
from ice3_gt4py.functions.sea_town_masks import lbc, conc3d, ray, fsedc
from ice3_gt4py.functions.upwind_sedimentation import instant_precipitation, maximum_time_step, mixing_ratio_update, upper_air_flux


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_sedimentation_stat.F90")
@stencil_collection("statistical_sedimentation")
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
    """Compute sedimentation sources for statistical sedimentation

    Args:
        TSTEP (float): physical time step
        rhodref (Field[float]): density of dry air
        dzz (Field[float]): vertical spacing of cells
        pabs_t (Field[float]): absolute pressure at t
        th_t (Field[float]): potential temperature at t
        rcs (Field[float]): cloud droplets m.r. tendency
        rrs (Field[float]): rain m.r. tendency
        ris (Field[float]): ice m.r. tendency
        rss (Field[float]): snow m.r. tendency
        rgs (Field[float]): graupel m.r. tendency
        sea (Field[IJ, float]): mask for sea
        town (Field[IJ, float]): mask for town
        fpr (Field[float]): upper-air precipitation fluxes
        inst_rr (Field[float]): instant prepicipitations of rain
        inst_rc (Field[float]): instant prepicipitations of cloud droplets
        inst_ri (Field[float]): instant prepicipitations of ice
        inst_rs (Field[float]): instant prepicipitations of snow
        inst_rg (Field[float]): instant prepicipitations of graupel
    """
    from __externals__ import (
        C_RTMIN,
        RHOLW,
        EXSEDR,
        FSEDR,
        R_RTMIN,
        TSTEP,
        FSEDS,
        EXSEDS,
        S_RTMIN,
        FSEDG,
        G_RTMIN,
        EXSEDG,
    )

    # Note Hail is omitted
    # Note : lsedic = True in Arome
    # Note : frp is sed

    # "PHYEX/src/common/micro/mode_ice4_sedimentation.F90", from_line=169, to_line=178
    with computation(PARALLEL), interval(...):
        rc_t = rcs * TSTEP
        rr_t = rrs * TSTEP
        ri_t = ris * TSTEP
        rs_t = rss * TSTEP
        rg_t = rgs * TSTEP

    # FRPR present for AROME config
    # 1. Compute the fluxes
    # Gamma computations shifted in RainIceDescr
    # Warning : call shift

    # 2. Fluxes

    # Initialize vertical loop
    with computation(PARALLEL), interval(...):
        fpr_c = 0
        fpr_r = 0
        fpr_i = 0
        fpr_s = 0
        fpr_g = 0

    # l253 to l258
    with computation(PARALLEL), interval(...):
        _ray = ray(sea)
        _lbc = lbc(sea)
        # _fsedc = fsedc(sea)
        _conc3d = conc3d(town, sea)

    # Compute the sedimentation fluxes
    with computation(BACKWARD), interval(...):
        TSTEP__rho_dz = TSTEP / (rhodref * dzz)

        # 2.1 cloud
        # Translation note : LSEDIC is assumed to be True
        # Translation note : PSEA and PTOWN are assumed to be present as in AROME

    # TODO  compute ray, lbc, fsedc, conc3d
    with computation(PARALLEL), interval(...):

        # 2.1 cloud
        qp = fpr_c[0, 0, 1] * TSTEP__rho_dz[0, 0, 0]
        wsedw1 = (
            terminal_velocity(rc_t, th_t, pabs_t, rhodref, _lbc, _ray, _conc3d)
            if rc_t > C_RTMIN
            else 0
        )
        wsedw2 = (
            terminal_velocity(qp, th_t, pabs_t, rhodref, _lbc, _ray, _conc3d)
            if qp > C_RTMIN
            else 0
        )

        fpr_c = weighted_sedimentation_flux_1(wsedw1, dzz, rhodref, rc_t, TSTEP)
        fpr_c += (
            weighted_sedimentation_flux_2(wsedw2, fpr_c, dzz, TSTEP)
            if wsedw2 != 0
            else 0
        )

        # 2.2 rain
        # Other species
        qp[0, 0, 0] = fpr_r[0, 0, 1] * TSTEP__rho_dz[0, 0, 0]
        wsedw1 = other_species(FSEDR, EXSEDR, rr_t, rhodref) if rr_t > R_RTMIN else 0
        wsedw2 = other_species(FSEDR, EXSEDR, qp, rhodref) if qp > R_RTMIN else 0

        fpr_r = weighted_sedimentation_flux_1(wsedw1, dzz, rhodref, rr_t, TSTEP)
        fpr_r += (
            weighted_sedimentation_flux_2(wsedw2, fpr_r, dzz, TSTEP)
            if wsedw2 != 0
            else 0
        )

        # 2.3 ice
        qp[0, 0, 0] = fpr_i[0, 0, 1] * TSTEP__rho_dz[0, 0, 0]
        wsedw1 = pristine_ice(ri_t, rhodref)
        wsedw2 = pristine_ice(qp, rhodref)

        fpr_i = weighted_sedimentation_flux_1(wsedw1, dzz, rhodref, ri_t, TSTEP)
        fpr_i += (
            weighted_sedimentation_flux_2(wsedw2, fpr_i, dzz, TSTEP) if qp != 0 else 0
        )

        # 2.4 snow
        # Translation note : REPRO48 set to True
        qp[0, 0, 0] = fpr_s[0, 0, 1] * TSTEP__rho_dz[0, 0, 0]
        wsedw1 = other_species(FSEDS, EXSEDS, rs_t, rhodref) if rs_t > S_RTMIN else 0
        wsedw2 = other_species(FSEDS, EXSEDS, qp, rhodref) if qp > S_RTMIN else 0

        fpr_s = weighted_sedimentation_flux_1(wsedw1, dzz, rhodref, rs_t, TSTEP)
        fpr_s += (
            weighted_sedimentation_flux_2(wsedw2, fpr_s, dzz, TSTEP)
            if wsedw2 != 0
            else 0
        )

        # 2.5 graupel
        qp[0, 0, 0] = fpr_g[0, 0, 1] * TSTEP__rho_dz[0, 0, 0]
        wsedw1 = other_species(FSEDG, EXSEDG, rg_t, rhodref) if rg_t > G_RTMIN else 0
        wsedw2 = other_species(FSEDG, EXSEDG, qp, rhodref) if qp > G_RTMIN else 0

        fpr_g = weighted_sedimentation_flux_1(wsedw1, dzz, rhodref, rc_t, TSTEP)
        fpr_g += (
            weighted_sedimentation_flux_2(wsedw2, fpr_g, dzz, TSTEP)
            if wsedw2 != 0
            else 0
        )

    # 3. Source
    # Calcul des tendances
    with computation(PARALLEL), interval(...):
        rcs = rcs + TSTEP__rho_dz * (fpr_c[0, 0, 1] - fpr_c[0, 0, 0]) / TSTEP
        ris = ris + TSTEP__rho_dz * (fpr_i[0, 0, 1] - fpr_i[0, 0, 0]) / TSTEP
        rss = rss + TSTEP__rho_dz * (fpr_s[0, 0, 1] - fpr_s[0, 0, 0]) / TSTEP
        rgs = rgs + TSTEP__rho_dz * (fpr_g[0, 0, 1] - fpr_g[0, 0, 0]) / TSTEP
        rrs = rrs + TSTEP__rho_dz * (fpr_r[0, 0, 1] - fpr_r[0, 0, 0]) / TSTEP

    # Instantaneous fluxes
    with computation(FORWARD), interval(0, 1):
        inprc = fpr_c / RHOLW
        inprr = fpr_r / RHOLW
        inpri = fpr_i / RHOLW
        inprs = fpr_s / RHOLW
        inprg = fpr_g / RHOLW


@function
def terminal_velocity(
    content: Field["float"],
    th_t: Field["float"],
    pabs_t: Field["float"],
    rhodref: Field["float"],
    lbc: Field["float"],
    ray: Field["float"],
    conc3d: Field["float"],
):
    from __externals__ import CC, CEXVT, DC, FSEDC_1, LBEXC

    wlbda = 6.6e-8 * (101325 / pabs_t[0, 0, 0]) * (th_t[0, 0, 0] / 293.15)
    wlbdc = (lbc * conc3d / (rhodref * content)) ** LBEXC
    cc = CC * (1 + 1.26 * wlbda * wlbdc / ray)
    wsedw1 = rhodref ** (-CEXVT) * wlbdc * (-DC) * cc * FSEDC_1

    return wsedw1


@stencil_collection("upwind_sedimentation")
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
        TT,
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
    )

    with computation(PARALLEL), interval(...):
        dt__rho_dz = TSTEP / (rhodref * dzz)
        oorhodz = 1 / (rhodref * dzz)

    # TODO
    # remaining time to be initialized
    # 2. Compute the fluxes
    # l219 to l262
    with computation(PARALLEL), interval(...):
        rcs -= rc_t / TSTEP
        ris -= ri_t / TSTEP
        rrs -= rr_t / TSTEP
        rss -= rs_t / TSTEP
        rgs -= rg_t / TSTEP

        wsed_c = 0
        wsed_r = 0
        wsed_i = 0
        wsed_s = 0
        wsed_g = 0

        remaining_time = TSTEP

    # in internal_sedim_split
    with computation(PARALLEL), interval(...):
        _ray = ray(sea)
        _lbc = lbc(sea)
        _fsedc = fsedc(sea)
        _conc3d = conc3d(town, sea)

    ## 2.1 For cloud droplets

    # TODO : share function with statistical sedimentation
    with computation(PARALLEL), interval(...):
        wlbdc = (_lbc * _conc3d / (rhodref * rc_t)) ** LBEXC
        _ray /= wlbdc
        t = th_t * (pabs_t / P00) ** (RD / CPD)
        wlbda = 6.6e-8 * (P00 / pabs_t) * (t / TT)
        cc = CC * (1 + 1.26 * wlbda / _ray)
        wsed = rhodref ** (-CEXVT + 1) * wlbdc * cc * _fsedc

    # Translation note : l723 in mode_ice4_sedimentation_split.F90
    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            C_RTMIN, rhodref, max_tstep, rc_t, dzz, wsed_c, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_rc[0, 0] += instant_precipitation(wsed_c, max_tstep, TSTEP)

    # Translation note : l738 in mode_ice4_sedimentation_split.F90
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

