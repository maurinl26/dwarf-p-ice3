# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import IJ, PARALLEL, Field, computation, interval

# 8. Total tendencies
# 8.1 Total tendencies limited by available species
# from_file="PHYEX/src/common/micro/rain_ice.F90",
# from_line=693,
# to_line=728
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
    """Update tendencies

    Args:
        wr_th (Field[float]): potential temperature initial value
        wr_v (Field[float]): vapour initial value
        wr_c (Field[float]): cloud droplets initial value
        wr_r (Field[float]): rain initial value
        wr_i (Field[float]): ice initial value
        wr_s (Field[float]): snow initial value
        wr_g (Field[float]): graupel initial value
        ls_fact (Field[float]): sublimation latent heat over heat capacity
        lv_fact (Field[float]): vapourisation latent heat over heat capacity
        exnref (Field[float]): reference exner pressure
        ths (Field[float]): source (tendency) of potential temperature
        rvs (Field[float]): source (tendency) of vapour
        rcs (Field[float]): source (tendency) of cloud droplets
        rrs (Field[float]): source (tendency) of rain
        ris (Field[float]): source (tendency) of ice
        rss (Field[float]): source (tendency) of snow
        rgs (Field[float]): source (tendency) of graupel
        rvheni (Field[float]): _description_
        rv_t (Field[float]): vapour m.r. at t
        rc_t (Field[float]): droplets m.r. at t
        rr_t (Field[float]): rain m.r. at t
        ri_t (Field[float]): ice m.r. at t
        rs_t (Field[float]): snow m.r. at t
        rg_t (Field[float]): graupel m.r. at t
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


# from_file="PHYEX/src/common/micro/rain_ice.F90",
# from_line=367,
# to_line=396
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
    """_summary_

    Args:
        ldmicro (Field[bool]): mask for microphysical computations
        exn (Field[float]): exner pressure
        ls_fact (Field[float]): sublimation latent heat over capacity
        lv_fact (Field[float]): vaporisation latent heat over capacity
        th_t (Field[float]): potential temperature at t
        rv_t (Field[float]): vapour m.r. at t
        rc_t (Field[float]): cloud droplet m.r. at t
        rr_t (Field[float]): rain m.r. at t
        ri_t (Field[float]): ice m.r. at t
        rs_t (Field[float]): snow m.r.
        rg_t (Field[float]): graupel m.r.
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
    from __externals__ import C_RTMIN, G_RTMIN, I_RTMIN, R_RTMIN, S_RTMIN

    with computation(PARALLEL), interval(...):
        ldmicro = (
            rc_t > C_RTMIN
            or rr_t > R_RTMIN
            or ri_t > I_RTMIN
            or rs_t > S_RTMIN
            or rg_t > G_RTMIN
        )


# 3. Initial values saving
#    from_file="PHYEX/src/common/micro/rain_ice.F90",
#    from_line=424,
#    to_line=444
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


# from_file="PHYEX/src/common/micro/rain_ice.F90",
# from_line=492,
# to_line=498
def ice4_precipitation_fraction_sigma(
        sigs: Field["float"],
        sigma_rc: Field["float"]
):
    """Compute supersaturation variance with supersaturation standard deviation.

    In rain_ice.F90
    IF (PARAMI%CSUBG_AUCV_RC=='PDF ' .AND. PARAMI%CSUBG_PR_PDF=='SIGM')

    Args:
        sigs (Field[float]): _description_
        sigma_rc (Field[float]):
    """
    with computation(PARALLEL), interval(...):
        sigma_rc = sigs**2


# from_file="PHYEX/src/common/micro/rain_ice.F90",
# from_line=792,
# to_line=801
def rain_fraction_sedimentation(
    wr_r: Field["float"],
    wr_s: Field["float"],
    wr_g: Field["float"],
    rrs: Field["float"],
    rss: Field["float"],
    rgs: Field["float"],
):
    """Computes vertical rain fraction

    Args:
        wr_r (Field[float]): initial value for rain m.r.
        wr_s (Field[float]): initial value for snow m.r.
        wr_g (Field[float]): initial value for graupel m.r.
        rrs (Field[float]): tendency (source) for rain
        rss (Field[float]): tendency (source) for snow
        rgs (Field[float]): tendency (source) for graupel
    """

    from __externals__ import TSTEP

    with computation(PARALLEL), interval(0, 1):
        wr_r = rrs * TSTEP
        wr_s = rss * TSTEP
        wr_g = rgs * TSTEP


# from_file="PHYEX/src/common/micro/rain_ice.F90",
# from_line=792,
# to_line=801
def ice4_rainfr_vert(
    prfr: Field["float"], rr: Field["float"], rs: Field["float"], rg: Field["float"]
):
    from __externals__ import G_RTMIN, R_RTMIN, S_RTMIN

    with computation(BACKWARD), interval(0, -1):
        if rr > R_RTMIN or rs > S_RTMIN or rg > G_RTMIN:
            prfr[0, 0, 0] = max(prfr[0, 0, 0], prfr[0, 0, 1])
            if prfr == 0:
                prfr = 1
        else:
            prfr = 0


#  from_file="PHYEX/src/common/micro/rain_ice.F90.func.h",
#  from_line=816,
#  to_line=830
def fog_deposition(
    rcs: Field["float"],
    rc_t: Field["float"],
    rhodref: Field["float"],
    dzz: Field["float"],
    inprc: Field[IJ, "float"],
):
    """Compute fog deposition on vegetation.
    Not activated in AROME.

    Args:
        rcs (Field[float]): source of cloud droplets
        rc_t (Field[float]): cloud droplets m.r.
        rhodref (Field[float]): dry density of air
        dzz (Field[float]): vertical spacing of cells
        inprc (Field[IJ, float]): deposition on vegetation
    """

    from __externals__ import RHOLW, VDEPOSC

    # Note : activated if LDEPOSC is True in rain_ice.F90
    with computation(FORWARD), interval(0, 1):
        rcs -= VDEPOSC * rc_t / dzz
        inprc[0, 0] += VDEPOSC * rc_t[0, 0, 0] * rhodref[0, 0, 0] / RHOLW
