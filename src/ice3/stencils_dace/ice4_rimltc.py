# -*- coding: utf-8 -*-
"""
ICE4 RIMLTC - DaCe Implementation

This module implements ice crystal melting (RIMLTC) for the ICE4
microphysics scheme, translated from GT4Py to DaCe.

Process implemented:
- Ice crystal melting above freezing temperature

Reference:
    PHYEX/src/common/micro/mode_ice4_rimltc.F90
    
Author:
    Translated to Python/DaCe from GT4Py by Cline AI Assistant
"""

import dace

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")


@dace.program
def ice4_rimltc(
    ldcompute: dace.bool[I, J, K],
    t: dace.float32[I, J, K],
    exn: dace.float32[I, J, K],
    lvfact: dace.float32[I, J, K],
    lsfact: dace.float32[I, J, K],
    tht: dace.float32[I, J, K],
    rit: dace.float32[I, J, K],
    rimltc_mr: dace.float32[I, J, K],
    LFEEDBACKT: dace.bool,
    TT: dace.float32,
):
    """
    Compute melting of cloud ice crystals above freezing temperature.
    
    Parameters
    ----------
    ldcompute : dace.bool[I, J, K]
        Computation mask
    t : dace.float32[I, J, K]
        Temperature (K)
    exn : dace.float32[I, J, K]
        Exner function
    lvfact : dace.float32[I, J, K]
        Vaporization latent heat factor L_v/(c_ph×π) (K)
    lsfact : dace.float32[I, J, K]
        Sublimation latent heat factor L_s/(c_ph×π) (K)
    tht : dace.float32[I, J, K]
        Potential temperature θ (K)
    rit : dace.float32[I, J, K]
        Ice crystal mixing ratio (kg/kg)
    rimltc_mr : dace.float32[I, J, K]
        Output: Ice melting rate (kg/kg/s)
    LFEEDBACKT : dace.bool
        Enable temperature feedback
    TT : dace.float32
        Triple point temperature (K)
    """
    @dace.map
    def compute_melting(i: _[0:I], j: _[0:J], k: _[0:K]):
        if rit[i, j, k] > 0.0 and t[i, j, k] > TT and ldcompute[i, j, k]:
            rimltc_mr[i, j, k] = rit[i, j, k]
            
            # Limit melting to prevent temperature from dropping below freezing
            if LFEEDBACKT:
                rimltc_mr[i, j, k] = min(
                    rimltc_mr[i, j, k], 
                    max(0.0, (tht[i, j, k] - TT / exn[i, j, k]) / (lsfact[i, j, k] - lvfact[i, j, k]))
                )
        else:
            rimltc_mr[i, j, k] = 0.0
