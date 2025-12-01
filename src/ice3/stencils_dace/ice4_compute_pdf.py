"""
ICE4 Compute PDF - DaCe Implementation

This module implements PDF-based cloud partitioning for the ICE4
microphysics scheme, translated from GT4Py to DaCe.

Process implemented:
- Split clouds into high and low content regions based on PDF
- Compute precipitation fraction

Reference:
    PHYEX/src/common/micro/mode_ice4_compute_pdf.F90
    
Author:
    Translated to Python/DaCe from GT4Py by Cline AI Assistant
"""

import dace

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")


@dace.program
def ice4_compute_pdf(
    ldmicro: dace.bool[I, J, K],
    rhodref: dace.float32[I, J, K],
    rc_t: dace.float32[I, J, K],
    ri_t: dace.float32[I, J, K],
    cf: dace.float32[I, J, K],
    t: dace.float32[I, J, K],
    sigma_rc: dace.float32[I, J, K],
    hlc_hcf: dace.float32[I, J, K],
    hlc_lcf: dace.float32[I, J, K],
    hlc_hrc: dace.float32[I, J, K],
    hlc_lrc: dace.float32[I, J, K],
    hli_hcf: dace.float32[I, J, K],
    hli_lcf: dace.float32[I, J, K],
    hli_hri: dace.float32[I, J, K],
    hli_lri: dace.float32[I, J, K],
    rf: dace.float32[I, J, K],
    SUBG_AUCV_RC: dace.int32,
    SUBG_AUCV_RI: dace.int32,
    SUBG_PR_PDF: dace.int32,
    CRIAUTC: dace.float32,
    CRIAUTI: dace.float32,
    ACRIAUTI: dace.float32,
    BCRIAUTI: dace.float32,
    C_RTMIN: dace.float32,
    I_RTMIN: dace.float32,
    TT: dace.float32,
):
    """
    Compute PDF for cloud partitioning into high and low content regions.
    
    Parameters
    ----------
    ldmicro : dace.bool[I, J, K]
        Microphysics computation mask
    rhodref : dace.float32[I, J, K]
        Reference air density (kg/m³)
    rc_t, ri_t : dace.float32[I, J, K]
        Cloud and ice mixing ratios (kg/kg)
    cf : dace.float32[I, J, K]
        Total cloud fraction (0-1)
    t : dace.float32[I, J, K]
        Temperature (K)
    sigma_rc : dace.float32[I, J, K]
        Cloud mixing ratio standard deviation (kg/kg)
    hlc_hcf, hlc_lcf : dace.float32[I, J, K]
        High/low liquid content cloud fractions (output)
    hlc_hrc, hlc_lrc : dace.float32[I, J, K]
        High/low liquid content mixing ratios (output)
    hli_hcf, hli_lcf : dace.float32[I, J, K]
        High/low ice content cloud fractions (output)
    hli_hri, hli_lri : dace.float32[I, J, K]
        High/low ice content mixing ratios (output)
    rf : dace.float32[I, J, K]
        Precipitation fraction (output)
    SUBG_AUCV_RC, SUBG_AUCV_RI : dace.int32
        Subgrid autoconversion schemes (0=NONE, 1=CLFR, 2=ADJU, 3=PDF)
    SUBG_PR_PDF : dace.int32
        PDF type for precipitation (0=SIGM)
    CRIAUTC, CRIAUTI : dace.float32
        Autoconversion thresholds
    ACRIAUTI, BCRIAUTI : dace.float32
        Ice autoconversion parameters
    C_RTMIN, I_RTMIN : dace.float32
        Minimum thresholds
    TT : dace.float32
        Triple point temperature (K)
    """
    @dace.map
    def compute_pdf_partition(i: _[0:I], j: _[0:J], k: _[0:K]):
        # Compute autoconversion thresholds
        if ldmicro[i, j, k]:
            rcrautc_tmp = CRIAUTC / rhodref[i, j, k]
            criauti_tmp = min(CRIAUTI, 10.0 ** (ACRIAUTI * (t[i, j, k] - TT) + BCRIAUTI))
        else:
            rcrautc_tmp = 0.0
            criauti_tmp = 0.0
        
        # Process liquid cloud (RC)
        if SUBG_AUCV_RC == 0:
            # NONE: Binary partition based on threshold
            if rc_t[i, j, k] > rcrautc_tmp and ldmicro[i, j, k]:
                hlc_hcf[i, j, k] = 1.0
                hlc_lcf[i, j, k] = 0.0
                hlc_hrc[i, j, k] = rc_t[i, j, k]
                hlc_lrc[i, j, k] = 0.0
            elif rc_t[i, j, k] > C_RTMIN and ldmicro[i, j, k]:
                hlc_hcf[i, j, k] = 0.0
                hlc_lcf[i, j, k] = 1.0
                hlc_hrc[i, j, k] = 0.0
                hlc_lrc[i, j, k] = rc_t[i, j, k]
            else:
                hlc_hcf[i, j, k] = 0.0
                hlc_lcf[i, j, k] = 0.0
                hlc_hrc[i, j, k] = 0.0
                hlc_lrc[i, j, k] = 0.0
        
        elif SUBG_AUCV_RC == 1:
            # CLFR: Use cloud fraction
            if cf[i, j, k] > 0.0 and rc_t[i, j, k] > rcrautc_tmp * cf[i, j, k] and ldmicro[i, j, k]:
                hlc_hcf[i, j, k] = cf[i, j, k]
                hlc_lcf[i, j, k] = 0.0
                hlc_hrc[i, j, k] = rc_t[i, j, k]
                hlc_lrc[i, j, k] = 0.0
            elif cf[i, j, k] > 0.0 and rc_t[i, j, k] > C_RTMIN and ldmicro[i, j, k]:
                hlc_hcf[i, j, k] = 0.0
                hlc_lcf[i, j, k] = cf[i, j, k]
                hlc_hrc[i, j, k] = 0.0
                hlc_lrc[i, j, k] = rc_t[i, j, k]
            else:
                hlc_hcf[i, j, k] = 0.0
                hlc_lcf[i, j, k] = 0.0
                hlc_hrc[i, j, k] = 0.0
                hlc_lrc[i, j, k] = 0.0
        
        elif SUBG_AUCV_RC == 2:
            # ADJU: Adjust existing partition
            if ldmicro[i, j, k]:
                sumrc_tmp = hlc_lrc[i, j, k] + hlc_hrc[i, j, k]
                if sumrc_tmp > 0.0:
                    hlc_lrc[i, j, k] *= rc_t[i, j, k] / sumrc_tmp
                    hlc_hrc[i, j, k] *= rc_t[i, j, k] / sumrc_tmp
                else:
                    hlc_lrc[i, j, k] = 0.0
                    hlc_hrc[i, j, k] = 0.0
        
        elif SUBG_AUCV_RC == 3:
            # PDF: Use probability density function
            if SUBG_PR_PDF == 0:
                # SIGM: Sigma-based PDF
                if rc_t[i, j, k] > rcrautc_tmp + sigma_rc[i, j, k] and ldmicro[i, j, k]:
                    hlc_hcf[i, j, k] = 1.0
                    hlc_lcf[i, j, k] = 0.0
                    hlc_hrc[i, j, k] = rc_t[i, j, k]
                    hlc_lrc[i, j, k] = 0.0
                elif (rc_t[i, j, k] > (rcrautc_tmp - sigma_rc[i, j, k]) and 
                      rc_t[i, j, k] <= (rcrautc_tmp + sigma_rc[i, j, k]) and 
                      ldmicro[i, j, k]):
                    hlc_hcf[i, j, k] = (rc_t[i, j, k] + sigma_rc[i, j, k] - rcrautc_tmp) / (2.0 * sigma_rc[i, j, k])
                    hlc_lcf[i, j, k] = max(0.0, cf[i, j, k] - hlc_hcf[i, j, k])
                    hlc_hrc[i, j, k] = ((rc_t[i, j, k] + sigma_rc[i, j, k] - rcrautc_tmp) * 
                                       (rc_t[i, j, k] + sigma_rc[i, j, k] + rcrautc_tmp) / 
                                       (4.0 * sigma_rc[i, j, k]))
                    hlc_lrc[i, j, k] = max(0.0, rc_t[i, j, k] - hlc_hrc[i, j, k])
                elif rc_t[i, j, k] > C_RTMIN and cf[i, j, k] > 0.0 and ldmicro[i, j, k]:
                    hlc_hcf[i, j, k] = 0.0
                    hlc_lcf[i, j, k] = cf[i, j, k]
                    hlc_hrc[i, j, k] = 0.0
                    hlc_lrc[i, j, k] = rc_t[i, j, k]
                else:
                    hlc_hcf[i, j, k] = 0.0
                    hlc_lcf[i, j, k] = 0.0
                    hlc_hrc[i, j, k] = 0.0
                    hlc_lrc[i, j, k] = 0.0
        
        # Process ice (RI)
        if SUBG_AUCV_RI == 0:
            # NONE: Binary partition based on threshold
            if ri_t[i, j, k] > criauti_tmp and ldmicro[i, j, k]:
                hli_hcf[i, j, k] = 1.0
                hli_lcf[i, j, k] = 0.0
                hli_hri[i, j, k] = ri_t[i, j, k]
                hli_lri[i, j, k] = 0.0
            elif ri_t[i, j, k] > I_RTMIN and ldmicro[i, j, k]:
                hli_hcf[i, j, k] = 0.0
                hli_lcf[i, j, k] = 1.0
                hli_hri[i, j, k] = 0.0
                hli_lri[i, j, k] = ri_t[i, j, k]
            else:
                hli_hcf[i, j, k] = 0.0
                hli_lcf[i, j, k] = 0.0
                hli_hri[i, j, k] = 0.0
                hli_lri[i, j, k] = 0.0
        
        elif SUBG_AUCV_RI == 1:
            # CLFR: Use cloud fraction
            if cf[i, j, k] > 0.0 and ri_t[i, j, k] > criauti_tmp * cf[i, j, k] and ldmicro[i, j, k]:
                hli_hcf[i, j, k] = cf[i, j, k]
                hli_lcf[i, j, k] = 0.0
                hli_hri[i, j, k] = ri_t[i, j, k]
                hli_lri[i, j, k] = 0.0
            elif cf[i, j, k] > 0.0 and ri_t[i, j, k] > I_RTMIN and ldmicro[i, j, k]:
                hli_hcf[i, j, k] = 0.0
                hli_lcf[i, j, k] = cf[i, j, k]
                hli_hri[i, j, k] = 0.0
                hli_lri[i, j, k] = ri_t[i, j, k]
            else:
                hli_hcf[i, j, k] = 0.0
                hli_lcf[i, j, k] = 0.0
                hli_hri[i, j, k] = 0.0
                hli_lri[i, j, k] = 0.0
        
        elif SUBG_AUCV_RI == 2:
            # ADJU: Adjust existing partition
            if ldmicro[i, j, k]:
                sumri_tmp = hli_lri[i, j, k] + hli_hri[i, j, k]
                if sumri_tmp > 0.0:
                    hli_lri[i, j, k] *= ri_t[i, j, k] / sumri_tmp
                    hli_hri[i, j, k] *= ri_t[i, j, k] / sumri_tmp
                else:
                    hli_lri[i, j, k] = 0.0
                    hli_hri[i, j, k] = 0.0
        
        # Compute precipitation fraction
        if ldmicro[i, j, k]:
            rf[i, j, k] = max(hlc_hcf[i, j, k], hli_hcf[i, j, k])
        else:
            rf[i, j, k] = 0.0
