"""
ICE4 RRHONG - DaCe Implementation

This module implements homogeneous freezing of rain (RRHONG) for the ICE4
microphysics scheme, translated from GT4Py to DaCe.

Process implemented:
- Spontaneous freezing of supercooled rain drops at very cold temperatures

Reference:
    PHYEX/src/common/micro/mode_ice4_rrhong.F90
    
Author:
    Translated to Python/DaCe from GT4Py by Cline AI Assistant
"""

import dace

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")


@dace.program
def ice4_rrhong(
    ldcompute: dace.bool[I, J, K],
    t: dace.float32[I, J, K],
    exn: dace.float32[I, J, K],
    lv_fact: dace.float32[I, J, K],
    ls_fact: dace.float32[I, J, K],
    th_t: dace.float32[I, J, K],
    rr_t: dace.float32[I, J, K],
    rrhong_mr: dace.float32[I, J, K],
    LFEEDBACKT: dace.bool,
    R_RTMIN: dace.float32,
    TT: dace.float32,
):
    """
    Compute spontaneous (homogeneous) freezing of supercooled rain drops.
    
    Parameters
    ----------
    ldcompute : dace.bool[I, J, K]
        Computation mask
    t : dace.float32[I, J, K]
        Temperature (K)
    exn : dace.float32[I, J, K]
        Exner function
    lv_fact : dace.float32[I, J, K]
        Vaporization latent heat factor L_v/(c_ph×π) (K)
    ls_fact : dace.float32[I, J, K]
        Sublimation latent heat factor L_s/(c_ph×π) (K)
    th_t : dace.float32[I, J, K]
        Potential temperature θ (K)
    rr_t : dace.float32[I, J, K]
        Rain mixing ratio (kg/kg)
    rrhong_mr : dace.float32[I, J, K]
        Output: Rain homogeneous freezing rate (kg/kg/s)
    LFEEDBACKT : dace.bool
        Enable temperature feedback
    R_RTMIN : dace.float32
        Minimum rain threshold
    TT : dace.float32
        Triple point temperature (K)
    """
    @dace.map
    def compute_freezing(i: _[0:I], j: _[0:J], k: _[0:K]):
        if (t[i, j, k] < TT - 35.0 and 
            rr_t[i, j, k] > R_RTMIN and 
            ldcompute[i, j, k]):
            
            rrhong_mr[i, j, k] = rr_t[i, j, k]
            
            # Limit freezing to prevent temperature from rising above -35°C
            if LFEEDBACKT:
                rrhong_mr[i, j, k] = min(
                    rrhong_mr[i, j, k],
                    max(0.0, ((TT - 35.0) / exn[i, j, k] - th_t[i, j, k]) / 
                        (ls_fact[i, j, k] - lv_fact[i, j, k]))
                )
        else:
            rrhong_mr[i, j, k] = 0.0
