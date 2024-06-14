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


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=492, to_line=498
)
@stencil_collection("ice4_precipitation_fraction_sigma")
def ice4_precipitation_fraction_sigma(sigs: Field["float"], sigma_rc: Field["float"]):
    """Compute supersaturation variance with supersaturation standard deviation.

    In rain_ice.F90
    IF (PARAMI%CSUBG_AUCV_RC=='PDF ' .AND. PARAMI%CSUBG_PR_PDF=='SIGM')

    Args:
        sigs (Field[float]): _description_
        sigma_rc (Field[float]):
    """
    with computation(PARALLEL), interval(...):
        sigma_rc = sigs**2


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

    from __externals__ import (
        CRIAUTC,
        C_RTMIN,
        SUBG_AUCV_RC,
        SUBG_PR_PDF,
        CRIAUTI,
        ACRIAUTI,
        BCRIAUTI,
        TT,
        SUBG_AUCV_RI,
        I_RTMIN,
    )

    with computation(PARALLEL), interval(...):
        rcrautc_tmp = CRIAUTC / rhodref if ldmicro else 0

    # HSUBG_AUCV_RC = NONE (0)
    with computation(PARALLEL), interval(...):

        # TODO: inline this choice
        if SUBG_AUCV_RC == 0:
            if rc_t > rcrautc_tmp and ldmicro:
                hlc_hcf = 1
                hlc_lcf = 0
                hlc_hrc = rc_t
                hlc_lrc = 0

            elif rc_t > C_RTMIN and ldmicro:
                hlc_hcf = 0
                hlc_lcf = 1
                hlc_hrc = 0
                hlc_lrc = rc_t

            else:
                hlc_hcf = 0
                hlc_lcf = 0
                hlc_hrc = 0
                hlc_lrc = 0

        # HSUBG_AUCV_RC = CLFR (1)
        elif SUBG_AUCV_RC == 1:
            if cf > 0 and rc_t > rcrautc_tmp * cf and ldmicro:
                hlc_hcf = cf
                hlc_lcf = 0
                hlc_hrc = rc_t
                hlc_lrc = 0

            elif cf > 0 and rc_t > C_RTMIN and ldmicro:
                hlc_hcf = 0
                hlc_lcf = cf
                hlc_hrc = 0
                hlc_lrc = rc_t

            else:
                hlc_hcf = 0
                hlc_lcf = 0
                hlc_hrc = 0
                hlc_lrc = 0

        # HSUBG_AUCV_RC = ADJU (2)
        elif SUBG_AUCV_RC == 2:
            sumrc_tmp = hlc_lrc + hlc_hrc if ldmicro else 0

            if sumrc_tmp > 0 and ldmicro:
                hlc_lrc *= rc_t / sumrc_tmp
                hlc_hrc *= rc_t / sumrc_tmp

            else:
                hlc_lrc = 0
                hlc_hrc = 0

        # HSUBG_AUCV_RC = PDF (3)
        elif SUBG_AUCV_RC == 3:

            # HSUBG_PR_PDF = SIGM (0)
            if SUBG_PR_PDF == 0:
                if rc_t > rcrautc_tmp + sigma_rc and ldmicro:
                    hlc_hcf = 1
                    hlc_lcf = 0
                    hlc_hrc = rc_t
                    hlc_lrc = 0

                elif (
                    rc_t > (rcrautc_tmp - sigma_rc)
                    and rc_t >= (rcrautc_tmp + sigma_rc)
                    and ldmicro
                ):
                    hlc_hcf = (rc_t + sigma_rc - rcrautc_tmp) / (2.0 * sigma_rc)
                    hlc_lcf = max(0.0, cf - hlc_hcf)
                    hlc_hrc = (
                        (rc_t + sigma_rc - rcrautc_tmp)
                        * (rc_t + sigma_rc + rcrautc_tmp)
                        / (4.0 * sigma_rc)
                    )
                    hlc_lrc = max(0.0, rc_t - hlc_hrc)

                elif rc_t > C_RTMIN and cf > 0 and ldmicro:
                    hlc_hcf = 0
                    hlc_lcf = cf
                    hlc_hrc = 0
                    hlc_lrc = rc_t

                else:
                    hlc_hcf = 0.0
                    hlc_lcf = 0.0
                    hlc_hrc = 0.0
                    hlc_lrc = 0.0

            # Translation note : l187 to l296 omitted since options are not used in AROME

    with computation(PARALLEL), interval(...):
        criauti_tmp = (
            min(CRIAUTI, 10 ** (ACRIAUTI * (t - TT) + BCRIAUTI)) if ldmicro else 0
        )

        # TODO: inline this code
        # HSUBG_AUCV_RI = NONE (0)
        if SUBG_AUCV_RI == 0:
            if ri_t > criauti_tmp and ldmicro:
                hli_hcf = 1
                hli_lcf = 0
                hli_hri = ri_t
                hli_lri = 0

            elif ri_t > I_RTMIN and ldmicro:
                hli_hcf = 0
                hli_lcf = 1
                hli_hri = 0
                hli_lri = ri_t

            else:
                hli_hcf = 0
                hli_lcf = 0
                hli_hri = 0
                hli_lri = 0

        # HSUBG_AUCV_RI = CLFR (1)
        elif SUBG_AUCV_RI == 1:
            if cf > 0 and ri_t > criauti_tmp * cf and ldmicro:
                hli_hcf = cf
                hli_hri = 0
                hli_hri = ri_t
                hli_lri = 0

            elif cf > 0 and ri_t > I_RTMIN and ldmicro:
                hli_hcf = 0
                hli_lcf = cf
                hli_hri = 0
                hli_lri = ri_t

            else:
                hli_hcf = 0
                hli_lcf = 0
                hli_hri = 0
                hli_lri = 0

        # HSUBG_AUCV_RI == 2
        elif SUBG_AUCV_RI == 2:
            sumri_tmp = hli_lri + hli_hri if ldmicro else 0

            if sumri_tmp > 0 and ldmicro:
                hli_lri *= ri_t / sumri_tmp
                hli_hri *= ri_t / sumri_tmp
            else:
                hli_lri = 0
                hli_hri = 0

    with computation(PARALLEL), interval(...):
        rf = max(hlc_hcf, hli_hcf) if ldmicro else 0
