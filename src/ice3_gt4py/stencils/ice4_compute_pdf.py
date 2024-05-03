# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    Field,
    computation,
    PARALLEL,
    interval,
    __externals__,
)
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_compute_pdf.F90")
@stencil_collection("ice4_compute_pdf")
def ice4_compute_pdf(
    ldmicro: Field["bool"],
    rhodref: Field["float"],
    rc_t: Field["float"],
    ri_t: Field["float"],
    cf: Field["float"],
    t: Field["float"],
    sigma_rc: Field["float"],
    hlc_hcf: Field["float"],
    hlc_lcf: Field["float"],
    hlc_hrc: Field["float"],
    hlc_lrc: Field["float"],
    hli_hcf: Field["float"],
    hli_lcf: Field["float"],
    hli_hri: Field["float"],
    hli_lri: Field["float"],
    rf: Field["float"],
    rcautc_tmp: Field["float"],
):
    """PDF used to split clouds into high and low content parts

    Args:
        ldmicro (Field[bool]): mask for microphysics computation
        rc_t (Field[float]): cloud droplet m.r. estimate at t
        ri_t (Field[float]): ice m.r. estimate at t
        cf (Field[float]): cloud fraction
        t (Field[float]): temperature
        sigma_rc (Field[float]): standard dev of cloud droplets m.r. over the cell
        hlc_hcf (Field[float]): _description_
        hlc_lcf (Field[float]): _description_
        hlc_hrc (Field[float]): _description_
        hlc_lrc (Field[float]): _description_
        hli_hcf (Field[float]): _description_
        hli_lcf (Field[float]): _description_
        hli_hri (Field[float]): _description_
        hli_lri (Field[float]): _description_
        rf (Field[float]): _description_
    """

    from __externals__ import CRIAUTC

    with computation(PARALLEL), interval(...):
        rcautc_tmp = CRIAUTC / rhodref if ldmicro else 0

    # HSUBG_AUCV_RC = NONE (0)
