# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import PARALLEL, Field, computation, interval


# "PHYEX/src/common/micro/mode_ice4_compute_pdf.F90
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
    """Compute probability density function to split clouds into high and low content parts.
    
    This stencil uses a PDF-based approach to partition cloud water and ice content into
    high-content (prone to autoconversion) and low-content regions. The splitting is based
    on comparison with autoconversion thresholds and can use different subgrid schemes
    (NONE, CLFR, ADJU, PDF) controlled by SUBG_AUCV_RC and SUBG_AUCV_RI externals.
    
    The function computes separate partitions for liquid (hlc_*) and ice (hli_*) phases,
    determining both the cloud fractions and mixing ratios in high and low content regions.
    Finally, it calculates the precipitation fraction (rf) as the maximum of the high
    content cloud fractions.

    Args:
        ldmicro (Field[bool]): Mask for microphysics computation - true where microphysics is active
        rhodref (Field[float]): Reference air density (kg/m³)
        rc_t (Field[float]): Cloud droplet mixing ratio estimate at time t (kg/kg)
        ri_t (Field[float]): Ice crystal mixing ratio estimate at time t (kg/kg)
        cf (Field[float]): Total cloud fraction (dimensionless, 0-1)
        t (Field[float]): Temperature (K)
        sigma_rc (Field[float]): Standard deviation of cloud droplet mixing ratio within the grid cell (kg/kg)
        hlc_hcf (Field[float]): Output - High liquid content cloud fraction (dimensionless, 0-1)
        hlc_lcf (Field[float]): Output - Low liquid content cloud fraction (dimensionless, 0-1)
        hlc_hrc (Field[float]): Output - Cloud droplet mixing ratio in high content region (kg/kg)
        hlc_lrc (Field[float]): Output - Cloud droplet mixing ratio in low content region (kg/kg)
        hli_hcf (Field[float]): Output - High ice content cloud fraction (dimensionless, 0-1)
        hli_lcf (Field[float]): Output - Low ice content cloud fraction (dimensionless, 0-1)
        hli_hri (Field[float]): Output - Ice crystal mixing ratio in high content region (kg/kg)
        hli_lri (Field[float]): Output - Ice crystal mixing ratio in low content region (kg/kg)
        rf (Field[float]): Output - Precipitation fraction, max of high content fractions (dimensionless, 0-1)
    """

    from __externals__ import (ACRIAUTI, BCRIAUTI, C_RTMIN, CRIAUTC, CRIAUTI,
                               I_RTMIN, SUBG_AUCV_RC, SUBG_AUCV_RI,
                               SUBG_PR_PDF, TT)

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
