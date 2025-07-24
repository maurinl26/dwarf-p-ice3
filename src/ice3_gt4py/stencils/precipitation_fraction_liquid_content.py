# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, computation, interval, PARALLEL, IJ
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method

# stencil introduced in cy49
@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=492, to_line=498
)
@stencil_collection("ice4_precipitation_fraction_liquid_content")
def ice4_precipitation_fraction_liquid_content(
    hlc_lrc: Field["float"],
    hlc_hrc: Field["float"],
    hli_lri: Field["float"],
    hli_hri: Field["float"],
    hlc_lcf: Field["float"],
    hlc_hcf: Field["float"],
    hli_hcf: Field["float"],
    hli_lcf: Field["float"],
    rc_t: Field["float"],
    ri_t: Field["float"],
    cldfr: Field["float"],
):
    """Compute supersaturation variance with supersaturation standard deviation.

    In rain_ice.F90
    IF (PARAMI%CSUBG_AUCV_RC=='ADJU' .OR. PARAMI%CSUBG_AUCV_RI=='ADJU') THEN

    Args:
        hlc_lrc: Field["float"],
        hlc_hrc: Field["float"],
        hli_lri: Field["float"],
        hli_hri: Field["float"],
        hlc_lcf: Field["float"],
        hlc_hcf: Field["float"],
        hli_hcf: Field["float"],
        hli_lcf: Field["float"],
        rc_t: Field["float"],
        ri_t: Field["float"],
        cldfr: Field["float"]
    """
    with computation(PARALLEL), interval(...):
        hlc_lrc = rc_t - hlc_hrc
        hli_lri = ri_t - hli_hri
        hlc_lcf = cldfr - hlc_hcf if rc_t > 0 else 0
        hli_lcf = cldfr - hli_hcf if ri_t > 0 else 0