# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp

from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_fast_rg.F90")
@stencil_collection("ice4_fast_rg")
def ice4_fast_rg(
    ldcompute: Field["int"],
    t: Field["float"],
    rhodref: Field["float"],
    ri_t: Field["float"],
    rg_t: Field["float"],
    rc_t: Field["float"],
    rs_t: Field["float"],
    ci_t: Field["float"],
    lbdar: Field["float"],
    lbdas: Field["float"],
    lbdag: Field["float"],
    ricfrrg: Field["float"],
    rrcfrig: Field["float"],
    ricfrr: Field["float"],
    rg_rcdry_tnd: Field["float"],
    rg_ridry_tnd: Field["float"],
    rg_riwet_tnd: Field["float"],
    rg_rsdry_tnd: Field["float"],
    rg_rswet_tnd: Field["float"],
    gdry: Field["int"],
):
    """Compute fast graupel sources

    Args:
        ldcompute (Field[int]): switch to compute microphysical processes on column
        t (Field[float]): temperature
        rhodref (Field[float]): reference density
        ri_t (Field[float]): ice mixing ratio at t
        rg_t (Field[float]): graupel m.r. at t
        rc_t (Field[float]): cloud droplets m.r. at t
        rs_t (Field[float]): snow m.r. at t
        ci_t (Field[float]): _description_
        lbdar (Field[float]): slope parameter for rain
        lbdas (Field[float]): slope parameter for snow
        lbdag (Field[float]): slope parameter for graupel
        ricfrrg (Field[float]): rain contact freezing
        rrcfrig (Field[float]): rain contact freezing
        ricfrr (Field[float]): rain contact freezing
        rg_rcdry_tnd (Field[float]): Graupel wet growth
        rg_ridry_tnd (Field[float]): Graupel wet growth
        rg_riwet_tnd (Field[float]): Graupel wet growth
        rg_rsdry_tnd (Field[float]): Graupel wet growth
        rg_rswet_tnd (Field[float]): Graupel wet growth
        gdry (Field[int]): _description_
    """

    from __externals__ import (
        Ci,
        Cl,
        tt,
        lvtt,
        i_rtmin,
        r_rtmin,
        g_rtmin,
        s_rtmin,
        icfrr,
        rcfri,
        exicfrr,
        exrcfri,
        cexvt,
        crflimit,  # True to limit rain contact freezing to possible heat exchange
        cxg,
        dg,
        fcdryg,
        fidryg,
        colexig,
        colig,
        ldsoft,
    )

    # 6.1 rain contact freezing
    with computation(PARALLEL), interval(...):

        if ri_t > i_rtmin and rr_t > r_rtmin and ldcompute == 1:

            # not LDSOFT : compute the tendencies
            if ldsoft == 0:

                ricfrrg = icfrr * ri_t * lbdar**exicfrr * rhodref ** (-cexvt)
                rrcfrig = rcfri * ci_t * lbdar**exrcfri * rhodref ** (-cexvt)

                if crflimit:
                    zw0d = max(
                        0,
                        min(
                            1,
                            (ricfrrg * Ci + rrcfrig * Cl)
                            * (tt - t)
                            / max(1e-20, lvtt * rrcfrig),
                        ),
                    )
                    rrcfrig = zw0d * rrcfrig
                    ricffr = (1 - zw0d) * rrcfrig
                    ricfrrg = zw0d * ricfrrg

                else:
                    ricfrr = 0

        else:
            ricfrrg = 0
            rrcfrig = 0
            ricfrr = 0

    # 6.3 compute graupel growth
    with computation(PARALLEL), interval(...):

        if rg_t > g_rtmin and rc_t > r_rtmin and ldcompute == 1:

            if ldsoft == 0:
                rg_rcdry_tnd = lbdag ** (cxg - dg - 2.0) * rhodref ** (-cexvt)
                rg_rcdry_tnd = rg_rcdry_tnd * fcdryg * rc_t

        else:
            rg_rcdry_tnd = 0

        if rg_t > g_rtmin and ri_t > i_rtmin and ldcompute == 1:

            if ldsoft == 0:
                rg_ridry_tnd = lbdag ** (cxg - dg - 2.0) * rhodref ** (-cexvt)
                rg_ridry_tnd = fidryg * exp(colexig * (t - tt)) * ri_t * rg_ridry_tnd
                rg_riwet_tnd = rg_ridry_tnd / (colig * exp(colexig * (t - tt)))

        else:
            rg_ridry_tnd = 0
            rg_riwet_tnd = 0

    # 6.2.1 wet and dry collection of rs on graupel
    # Translation note : l171 in mode_ice4_fast_rg.F90
    with computation(PARALLEL), interval(...):

        if rs_t > s_rtmin and rg_t > g_rtmin and ldcompute == 1:
            gdry = 1  # GDRY is a boolean field in f90

        else:
            gdry = 0
            rg_rsdry_tnd = 0
            rg_rswet_tnd = 0

    # TODO: l182 to 212
    # if ldsoft == 0:
    # Call interp micro

    # Translation note : #ifdef REPRO48 l191 to l198 kept in mode_ice4_fast_rg.F90
    # Translation note : #else REPRO49  l200 to l207 omitted in mode_ice4_fast_rg.F90
    # with computation(PARALLEL), interval(...):
