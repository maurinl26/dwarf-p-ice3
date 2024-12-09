# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    __INLINED,
    PARALLEL,
    atan,
    computation,
    exp,
    interval,
    sqrt,
)
from gt4py.cartesian import gtscript
from ifs_physics_common.framework.stencil import stencil_collection

from ice3_gt4py.functions.temperature import update_temperature
from ice3_gt4py.functions.tiwmx import e_sat_i, e_sat_w
from ifs_physics_common.utils.f2py import ported_method


@ported_method(
    from_file="PHYEX/src/common/micro/condensation.F90",
)
@stencil_collection("condensation")
def condensation(
    sigqsat: gtscript.Field["float"],
    pabs: gtscript.Field["float"],
    sigs: gtscript.Field["float"],
    t: gtscript.Field["float"],
    rv_in: gtscript.Field["float"],
    ri_in: gtscript.Field["float"],
    rc_in: gtscript.Field["float"],
    rv_out: gtscript.Field["float"],
    rc_out: gtscript.Field["float"],
    ri_out: gtscript.Field["float"],
    t_out: gtscript.Field["float"],
    cldfr: gtscript.Field["float"],
    cph: gtscript.Field["float"],
    lv: gtscript.Field["float"],
    ls: gtscript.Field["float"],
):
    """Microphysical adjustments for specific contents due to condensation."""

    from __externals__ import (
        CPD,
        RD,
        RV,
    )

    # 3. subgrid condensation scheme
    # Translation note : only the case with LSUBG_COND = True retained (l475 in ice_adjust.F90)
    # sigqsat and sigs must be provided by the user
    with computation(PARALLEL), interval(...):
        # local gtscript.Fields
        # Translation note : 506 -> 514 kept (ocnd2 == False) # Arome default setting
        # Translation note : 515 -> 575 skipped (ocnd2 == True)
        prifact = 1  # ocnd2 == False for AROME
        frac_tmp = 0  # l340 in Condensation .f90

        # Translation note : 252 -> 263 if(present(PLV)) skipped (ls/lv are assumed to be present)
        # Translation note : 264 -> 274 if(present(PCPH)) skipped (files are assumed to be present)

        # store total water mixing ratio (244 -> 248)
        rt = rv_in + rc_in + ri_in * prifact

        # Translation note : 276 -> 310 (not osigmas) skipped (osigmas = True) for Arome default version
        # Translation note : 316 -> 331 (ocnd2 == True) skipped

        # l334 to l337
        pv = min(
            e_sat_w(t),
            0.99 * pabs,
        )
        piv = min(
            e_sat_i(t),
            0.99 * pabs,
        )

        # Translation note : OUSERI = False, OCND2 = False
        # Supersaturation coefficients
        qsl = RD / RV * pv / (pabs - pv)
        qsi = RD / RV * piv / (pabs - piv)

        # # dtype_interpolate bewteen liquid and solid as a function of temperature
        qsl = (1 - frac_tmp) * qsl + frac_tmp * qsi
        lvs = (1 - frac_tmp) * lv + frac_tmp * ls

        # # coefficients a et b
        ah = lvs * qsl / (RV * t**2) * (1 + RV * qsl / RD)
        a = 1 / (1 + lvs / cph * ah)
        # # b = ah * a
        sbar = a * (rt - qsl + ah * lvs * (rc_in + ri_in * prifact) / CPD)

        # l369 - l390
        # Translation note : LSTATNW = False
        # Translation note : l381 retained for sigmas formulation
        sigma = (
            sqrt((2 * sigs) ** 2 + (sigqsat * qsl * a) ** 2) if sigqsat != 0 else sigs
        )

        # Translation note : l407 - l411
        sigma = max(1e-10, sigma)
        q1 = sbar / sigma

        # Translation notes : l413 to l468 skipped (HCONDENS=="GAUS")
        # Translation notes : l469 to l504 kept (HCONDENS = "CB02")
        # 9.2.3 Fractional cloudiness and cloud condensate

        # Translation note : l470 to l479
        if q1 > 0:
            cond_tmp = (
                min(exp(-1) + 0.66 * q1 + 0.086 * q1**2, 2) if q1 <= 2 else q1
            )  # we use the MIN function for continuity
        else:
            cond_tmp = exp(1.2 * q1 - 1)
        cond_tmp *= sigma

        # Translation note : l482 to l489
        # cloud fraction
        cldfr = max(0, min(1, 0.5 + 0.36 * atan(1.55 * q1))) if cond_tmp > 1e-12 else 0

        # Translation note : l487 to l489
        cond_tmp = 0 if cldfr == 0 else cond_tmp

        # Translation notes : 506 -> 514 (not ocnd2)
        rc_out = (1 - frac_tmp) * cond_tmp  # liquid condensate
        ri_out = frac_tmp * cond_tmp  # solid condensate
        t_out = update_temperature(t, rc_out, rc_in, ri_out, ri_in, lv, ls)
        rv_out = rt - rc_in - ri_in * prifact

        # Translation note : sigrc computation out of scope
        # sigrc computation in sigrc_computation stencil

    # Translation note : end jiter
