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
    lv_fact: Field["float"],
    ls_fact: Field["float"],
    ai: Field["float"],
    cj: Field["float"],
    ci_in: Field["float"],
    ssi: Field["float"],
    rc_in: Field["float"],
    ri_in: Field["float"],
    rc_beri_tnd: Field["float"],
    ldsoft: Field["bool"],
):
    """Computes Bergeron-Findeisen effect RCBERI.

    Evaporation of cloud droplets for deposition over ice-crystals.

    Args:
        lcompute (Field[bool]): switch to compute microphysical processes
        lv_fact (Field[float]): latent heat of vaporisation
        ls_fact (Field[float]): latent heat of sublimation
        ai (Field[float]): thermodynamical function
        cj (Field[float]): function to compute ventilation factor
        ci_t (Field[float]): _description_
        ssi (Field[float]): supersaturation over ice
        rc_in (Field[float]): cloud droplets mixing ratio at t
        ri_in (Field[float]): pristine ice mixing ratio at t
        rc_beri_tnd (Field[float]): tendency for Bergeron Findeisen effect
    """

    from __externals__ import C_RTMIN, DI, I_RTMIN, LBEXI, LBI, O0DEPI, O2DEPI

    # 7.2 Bergeron-Findeisen effect: RCBERI
    with computation(PARALLEL), interval(...):
        if not ldsoft:
            if (
                ssi > 0
                and rc_in > C_RTMIN
                and ri_in > I_RTMIN
                and ci_in > 1e-20
                and ldcompute
            ):
                rc_beri_tnd = min(
                    1e-8, LBI * (rhodref * ri_in / ci_in) ** LBEXI
                )  # lambda_i
                rc_beri_tnd = (
                    (ssi / (rhodref * ai))
                    * ci_in
                    * (O0DEPI / rc_beri_tnd + O2DEPI * cj / rc_beri_tnd ** (DI + 2.0))
                )

            else:
                rc_beri_tnd = 0
