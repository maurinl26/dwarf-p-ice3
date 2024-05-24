# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    Field,
    function,
    computation,
    PARALLEL,
    BACKWARD,
    interval,
    IJ,
)
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method

from ice3_gt4py.functions.sedimentation_flux import (
    other_species,
    weighted_sedimentation_flux_1,
    weighted_sedimentation_flux_2,
)


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_sedimentation_stat.F90")
@stencil_collection("statistical_sedimentation")
def sedimentation_stat(
    dt: "float",
    rhodref: Field["float"],
    dzz: Field["float"],
    pabst: Field["float"],
    tht: Field["float"],
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
    sea: Field["bool"],
    town: Field["float"],
    fpr: Field["float"],
    inst_rr: Field[IJ, "float"],
    inst_rc: Field[IJ, "float"],
    inst_ri: Field[IJ, "float"],
    inst_rs: Field[IJ, "float"],
    inst_rg: Field[IJ, "float"],
):
    """Compute sedimentation sources for statistical sedimentation

    Args:
        dt (float): _description_
        rhodref (Field[float]): _description_
        dzz (Field[float]): _description_
        pabst (Field[float]): _description_
        tht (Field[float]): _description_
        rc_t (Field[float]): _description_
        rr_t (Field[float]): _description_
        ri_t (Field[float]): _description_
        rs_t (Field[float]): _description_
        rg_t (Field[float]): _description_
        rcs (Field[float]): _description_
        rrs (Field[float]): _description_
        ris (Field[float]): _description_
        rss (Field[float]): _description_
        rgs (Field[float]): _description_
        sea (Field[bool]): _description_
        town (Field[float]): _description_
        fpr (Field[float]): _description_
        inst_rr (Field[float]): _description_
        inst_rc (Field[float]): _description_
        inst_ri (Field[float]): _description_
        inst_rs (Field[float]): _description_
        inst_rg (Field[float]): _description_
    """
    from __externals__ import (
        C_RTMIN,
        RHOLW,
        CC,
        CEXVT,
        DC,
        EXSEDR,
        FSEDC,
        FSEDR,
        LBEXC,
        LSEDIC,
        R_RTMIN,
        LBC,
        CONC_SEA,
        CONC_LAND,
        CONC_URBAN,
        GAC,
        GAC2,
        GC,
        GC2,
        RAYDEF0,
        TSTEP,
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
        c_sed = 0
        r_sed = 0
        i_sed = 0
        s_sed = 0
        g_sed = 0

    # l253 to l258
    with computation(PARALLEL), interval(...):
        ray = max(1, 0.5 * ((1 - sea) * GAC / GC + sea * GAC2 / GC2))
        lbc = max(min(LBC[0], LBC[1]), sea * LBC[0] + (1 - sea * LBC[1]))
        fsedc = max(min(FSEDC[0], FSEDC[1]), sea * FSEDC[0] + (1 - sea) * FSEDC[1])
        conc3d = (1 - town) * (
            sea * CONC_SEA + (1 - sea) * CONC_LAND
        ) + town * CONC_URBAN

    # Compute the sedimentation fluxes
    with computation(BACKWARD), interval(...):
        dt__rho_dz = dt / (rhodref * dzz)

        # 2.1 cloud
        # Translation note : LSEDIC is assumed to be True
        # Translation note : PSEA and PTOWN are assumed to be present as in AROME

    # TODO  compute ray, lbc, fsedc, conc3d
    with computation(PARALLEL), interval(...):

        # 2.1 cloud
        qp = c_sed[0, 0, 1] * dt__rho_dz[0, 0, 0]
        wsedw1 = (
            terminal_velocity(rc_t, tht, pabst, rhodref, lbc, ray, conc3d)
            if rc_t > C_RTMIN
            else 0
        )
        wsedw2 = (
            terminal_velocity(qp, tht, pabst, rhodref, lbc, ray, conc3d)
            if qp > C_RTMIN
            else 0
        )

        c_sed = weighted_sedimentation_flux_1(wsedw1, dzz, rhodref, rc_t, dt)
        c_sed += (
            weighted_sedimentation_flux_2(wsedw2, dt, dzz, c_sed) if wsedw2 != 0 else 0
        )

        # 2.2 rain
        # Other species
        qp[0, 0, 0] = r_sed[0, 0, 1] * dt__rho_dz[0, 0, 0]
        wsedw1 = other_species(FSEDR, EXSEDR, rr_t, rhodref) if rr_t > R_RTMIN else 0
        wsedw2 = other_species(FSEDR, EXSEDR, qp, rhodref) if qp > R_RTMIN else 0

        r_sed = weighted_sedimentation_flux_1(wsedw1, dzz, rhodref, rc_t, dt)
        r_sed += (
            weighted_sedimentation_flux_2(wsedw2, dt, dzz, r_sed) if wsedw2 != 0 else 0
        )

        # 2.3 ice

        # 2.4 snow
        qp[0, 0, 0] = r_sed[0, 0, 1] * dt__rho_dz[0, 0, 0]
        wsedw1 = other_species(FSEDR, EXSEDR, rr_t, rhodref) if rr_t > R_RTMIN else 0
        wsedw2 = other_species(FSEDR, EXSEDR, qp, rhodref) if qp > R_RTMIN else 0

        s_sed = weighted_sedimentation_flux_1(wsedw1, dzz, rhodref, rc_t, dt)
        s_sed += (
            weighted_sedimentation_flux_2(wsedw2, dt, dzz, s_sed) if wsedw2 != 0 else 0
        )

        # 2.5 graupel
        qp[0, 0, 0] = r_sed[0, 0, 1] * dt__rho_dz[0, 0, 0]
        wsedw1 = other_species(FSEDR, EXSEDR, rr_t, rhodref) if rr_t > R_RTMIN else 0
        wsedw2 = other_species(FSEDR, EXSEDR, qp, rhodref) if qp > R_RTMIN else 0

        g_sed = weighted_sedimentation_flux_1(wsedw1, dzz, rhodref, rc_t, dt)
        g_sed += (
            weighted_sedimentation_flux_2(wsedw2, dt, dzz, g_sed) if wsedw2 != 0 else 0
        )

    # 3. Sources
    # Calcul des tendances
    with computation(PARALLEL), interval(...):
        rcs = rcs + dt__rho_dz * (c_sed[0, 0, 1] - c_sed[0, 0, 0]) / dt
        ris = ris + dt__rho_dz * (i_sed[0, 0, 1] - i_sed[0, 0, 0]) / dt
        rss = rss + dt__rho_dz * (s_sed[0, 0, 1] - s_sed[0, 0, 0]) / dt
        rgs = rgs + dt__rho_dz * (g_sed[0, 0, 1] - g_sed[0, 0, 0]) / dt
        rrs = rrs + dt__rho_dz * (r_sed[0, 0, 1] - r_sed[0, 0, 0]) / dt

    # Instantaneous fluxes
    with computation(PARALLEL), interval(0, 1):
        inst_rc = c_sed / RHOLW
        inst_rr = r_sed / RHOLW
        inst_ri = i_sed / RHOLW
        inst_rs = s_sed / RHOLW
        inst_rg = g_sed / RHOLW


@function
def terminal_velocity(
    content: Field["float"],
    tht: Field["float"],
    pabst: Field["float"],
    rhodref: Field["float"],
    lbc: Field["float"],
    ray: Field["float"],
    conc3d: Field["float"],
):
    from __externals__ import CC, CEXVT, DC, FSEDC, LBEXC

    wlbda = 6.6e-8 * (101325 / pabst[0, 0, 0]) * (tht[0, 0, 0] / 293.15)
    wlbdc = (lbc * conc3d / (rhodref * content)) ** LBEXC
    cc = CC * (1 + 1.26 * wlbda * wlbdc / ray)
    wsedw1 = rhodref ** (-CEXVT) * wlbdc * (-DC) * cc * FSEDC

    return wsedw1
