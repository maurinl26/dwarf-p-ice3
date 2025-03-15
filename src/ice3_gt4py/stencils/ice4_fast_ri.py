# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp, computation, interval, PARALLEL
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_fast_ri.F90")
@stencil_collection("ice4_fast_ri")
def ice4_fast_ri(
    ldcompute: Field["bool"],
    rhodref: Field["float"],
    ai: Field["float"],
    cj: Field["float"],
    cit: Field["float"],
    ssi: Field["float"],
    rct: Field["float"],
    rit: Field["float"],
    rc_beri_tnd: Field["float"],
    ldsoft: "bool",
):
    """Computes Bergeron-Findeisen effect RCBERI.

    Evaporation of cloud droplets for deposition over ice-crystals.

    Args:
        lcompute (Field[bool]): switch to compute microphysical processes
        lv_fact (Field[float]): latent heat of vaporisation
        ls_fact (Field[float]): latent heat of sublimation
        ai (Field[float]): thermodynamical function
        cj (Field[float]): function to compute ventilation factor
        cit (Field[float]): concentration of ice at t
        ssi (Field[float]): supersaturation over ice
        rct (Field[float]): cloud droplets mixing ratio at t
        rit (Field[float]): pristine ice mixing ratio at t
        rc_beritnd (Field[float]): tendency for Bergeron Findeisen effect
    """

    from __externals__ import C_RTMIN, DI, I_RTMIN, LBEXI, LBI, O0DEPI, O2DEPI

    # 7.2 Bergeron-Findeisen effect: RCBERI
    with computation(PARALLEL), interval(...):
        

            if (
                ssi > 0
                and rct > C_RTMIN
                and rit > I_RTMIN
                and cit > 1e-20
                and ldcompute
            ):
            
                if not ldsoft:

                    rc_beri_tnd = min(
                    1e8, LBI * (rhodref * rit / cit) ** LBEXI
                )  # lambda_i
                    rc_beri_tnd = (
                    (ssi / (rhodref * ai))
                    * cit
                    * (O0DEPI / rc_beri_tnd + O2DEPI * cj ** 2 / rc_beri_tnd ** (DI + 2.0))
                )

            else:
                rc_beri_tnd = 0
