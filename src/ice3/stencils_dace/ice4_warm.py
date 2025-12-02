"""
ICE4 Warm Processes - DaCe Implementation

This module implements warm rain microphysical processes for the ICE4
microphysics scheme, translated from GT4Py to DaCe.

Processes implemented:
- Cloud autoconversion to rain (RCAUTR)
- Cloud accretion by rain (RCACCR)
- Rain evaporation (RREVAV)

Reference:
    PHYEX/src/common/micro/mode_ice4_warm.F90
    
Author:
    Translated to Python/DaCe from GT4Py by Cline AI Assistant
"""

import dace

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")


@dace.program
def ice4_warm(
    ldcompute: dace.bool[I, J, K],
    rhodref: dace.float32[I, J, K],
    t: dace.float32[I, J, K],
    pres: dace.float32[I, J, K],
    th_t: dace.float32[I, J, K],
    lbdar: dace.float32[I, J, K],
    lbdar_rf: dace.float32[I, J, K],
    ka: dace.float32[I, J, K],
    dv: dace.float32[I, J, K],
    cj: dace.float32[I, J, K],
    hlc_hcf: dace.float32[I, J, K],
    hlc_lcf: dace.float32[I, J, K],
    hlc_hrc: dace.float32[I, J, K],
    hlc_lrc: dace.float32[I, J, K],
    cf: dace.float32[I, J, K],
    rf: dace.float32[I, J, K],
    rv_t: dace.float32[I, J, K],
    rc_t: dace.float32[I, J, K],
    rr_t: dace.float32[I, J, K],
    rcautr: dace.float32[I, J, K],
    rcaccr: dace.float32[I, J, K],
    rrevav: dace.float32[I, J, K],
    ldsoft: dace.bool,
    SUBG_RR_EVAP: dace.int32,
    C_RTMIN: dace.float32,
    R_RTMIN: dace.float32,
    TIMAUTC: dace.float32,
    CRIAUTC: dace.float32,
    FCACCR: dace.float32,
    EXCACCR: dace.float32,
    CEXVT: dace.float32,
    ALPW: dace.float32,
    BETAW: dace.float32,
    GAMW: dace.float32,
    EPSILO: dace.float32,
    LVTT: dace.float32,
    CPV: dace.float32,
    CL: dace.float32,
    CPD: dace.float32,
    RV: dace.float32,
    TT: dace.float32,
    O0EVAR: dace.float32,
    O1EVAR: dace.float32,
    EX0EVAR: dace.float32,
    EX1EVAR: dace.float32,
):
    """
    Compute warm rain microphysical processes for ICE4 scheme.
    
    Parameters
    ----------
    ldcompute : dace.bool[I, J, K]
        Computation mask
    rhodref : dace.float32[I, J, K]
        Reference air density (kg/m³)
    t : dace.float32[I, J, K]
        Temperature (K)
    pres : dace.float32[I, J, K]
        Pressure (Pa)
    th_t : dace.float32[I, J, K]
        Potential temperature (K)
    lbdar : dace.float32[I, J, K]
        Rain slope parameter (m⁻¹)
    lbdar_rf : dace.float32[I, J, K]
        Rain slope for rain fraction (m⁻¹)
    ka : dace.float32[I, J, K]
        Thermal conductivity of air (W/(m·K))
    dv : dace.float32[I, J, K]
        Water vapor diffusivity (m²/s)
    cj : dace.float32[I, J, K]
        Ventilation coefficient
    hlc_hcf, hlc_lcf : dace.float32[I, J, K]
        High/low cloud fraction
    hlc_hrc, hlc_lrc : dace.float32[I, J, K]
        High/low cloud liquid water
    cf : dace.float32[I, J, K]
        Cloud fraction
    rf : dace.float32[I, J, K]
        Rain fraction
    rv_t, rc_t, rr_t : dace.float32[I, J, K]
        Mixing ratios (kg/kg)
    rcautr, rcaccr, rrevav : dace.float32[I, J, K]
        Output tendencies (kg/kg/s)
    ldsoft : dace.bool
        Soft threshold mode
    SUBG_RR_EVAP : dace.int32
        Evaporation scheme (0=NONE, 1=CLFR, 2=PRFR)
    C_RTMIN, R_RTMIN : dace.float32
        Minimum thresholds
    TIMAUTC, CRIAUTC : dace.float32
        Autoconversion parameters
    FCACCR, EXCACCR, CEXVT : dace.float32
        Accretion parameters
    ALPW, BETAW, GAMW, EPSILO : dace.float32
        Saturation vapor pressure parameters
    LVTT, CPV, CL, CPD, RV, TT : dace.float32
        Thermodynamic constants
    O0EVAR, O1EVAR, EX0EVAR, EX1EVAR : dace.float32
        Evaporation parameters
    """
    @dace.map
    def compute_warm_processes(i: _[0:I], j: _[0:J], k: _[0:K]):
        # Autoconversion of cloud to rain (RCAUTR)
        if ldcompute[i, j, k] and hlc_hrc[i, j, k] > C_RTMIN and hlc_hcf[i, j, k] > 0.0:
            if not ldsoft:
                rcautr[i, j, k] = TIMAUTC * max(0.0, hlc_hrc[i, j, k] - 
                                                hlc_hcf[i, j, k] * CRIAUTC / rhodref[i, j, k])
            else:
                rcautr[i, j, k] = 0.0
        else:
            rcautr[i, j, k] = 0.0
        
        # Accretion of cloud by rain (RCACCR)
        if ldcompute[i, j, k] and rc_t[i, j, k] > C_RTMIN and rr_t[i, j, k] > R_RTMIN:
            if not ldsoft:
                rcaccr[i, j, k] = FCACCR * rc_t[i, j, k] * lbdar[i, j, k] ** EXCACCR * \
                                 rhodref[i, j, k] ** (-CEXVT)
            else:
                rcaccr[i, j, k] = 0.0
        else:
            rcaccr[i, j, k] = 0.0
        
        # Rain evaporation (RREVAV)
        if SUBG_RR_EVAP == 0:
            # NONE: Grid-mean evaporation
            if ldcompute[i, j, k] and rr_t[i, j, k] > R_RTMIN and rc_t[i, j, k] <= C_RTMIN:
                if not ldsoft:
                    # Saturation vapor pressure over water
                    esat_w = exp(ALPW - BETAW / t[i, j, k] - GAMW * log(t[i, j, k]))
                    
                    # Undersaturation
                    usw = 1.0 - rv_t[i, j, k] * (pres[i, j, k] - esat_w) / (EPSILO * esat_w)
                    
                    # Thermodynamic evaporation coefficient
                    av = ((LVTT + (CPV - CL) * (t[i, j, k] - TT)) ** 2 / 
                         (ka[i, j, k] * RV * t[i, j, k] ** 2) + 
                         (RV * t[i, j, k]) / (dv[i, j, k] * esat_w))
                    
                    # Evaporation rate
                    rrevav[i, j, k] = (max(0.0, usw) / (rhodref[i, j, k] * av)) * \
                                     (O0EVAR * lbdar[i, j, k] ** EX0EVAR + 
                                      O1EVAR * cj[i, j, k] * lbdar[i, j, k] ** EX1EVAR)
                else:
                    rrevav[i, j, k] = 0.0
            else:
                rrevav[i, j, k] = 0.0
                
        elif SUBG_RR_EVAP == 1 or SUBG_RR_EVAP == 2:
            # CLFR or PRFR: Subgrid evaporation
            if SUBG_RR_EVAP == 1:
                # CLFR: precipitation fraction = 1
                zw4 = 1.0
                zw3 = lbdar[i, j, k]
            else:
                # PRFR: use rain fraction
                zw4 = rf[i, j, k]
                zw3 = lbdar_rf[i, j, k]
            
            if ldcompute[i, j, k] and rr_t[i, j, k] > R_RTMIN and zw4 > cf[i, j, k]:
                if not ldsoft:
                    # Liquid water potential temperature
                    thlt_tmp = th_t[i, j, k] - LVTT * th_t[i, j, k] / CPD / t[i, j, k] * rc_t[i, j, k]
                    
                    # Unsaturated temperature
                    tu = thlt_tmp * t[i, j, k] / th_t[i, j, k]
                    
                    # Saturation vapor pressure over water (at Tu)
                    esat_w = exp(ALPW - BETAW / tu - GAMW * log(tu))
                    
                    # Undersaturation
                    usw = 1.0 - rv_t[i, j, k] * (pres[i, j, k] - esat_w) / (EPSILO * esat_w)
                    
                    # Thermodynamic evaporation coefficient (at Tu)
                    av = ((LVTT + (CPV - CL) * (tu - TT)) ** 2 / 
                         (ka[i, j, k] * RV * tu ** 2) + 
                         RV * tu / (dv[i, j, k] * esat_w))
                    
                    # Evaporation rate in clear region
                    rrevav[i, j, k] = (max(0.0, usw) / (rhodref[i, j, k] * av)) * \
                                     (O0EVAR * zw3 ** EX0EVAR + 
                                      O1EVAR * cj[i, j, k] * zw3 ** EX1EVAR) * \
                                     (zw4 - cf[i, j, k])
                else:
                    rrevav[i, j, k] = 0.0
            else:
                rrevav[i, j, k] = 0.0
