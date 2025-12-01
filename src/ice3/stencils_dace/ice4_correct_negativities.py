"""
ICE4 Correct Negativities - DaCe Implementation

This module implements negativity correction with conservation for the ICE4
microphysics scheme, translated from GT4Py to DaCe.

Process implemented:
- Correct negative mixing ratios
- Conserve mass and energy

Reference:
    PHYEX/src/common/micro/mode_ice4_correct_negativities.F90
    
Author:
    Translated to Python/DaCe from GT4Py by Cline AI Assistant
"""

import dace

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")


@dace.program
def ice4_correct_negativities(
    th_t: dace.float32[I, J, K],
    rv_t: dace.float32[I, J, K],
    rc_t: dace.float32[I, J, K],
    rr_t: dace.float32[I, J, K],
    ri_t: dace.float32[I, J, K],
    rs_t: dace.float32[I, J, K],
    rg_t: dace.float32[I, J, K],
    lv_fact: dace.float32[I, J, K],
    ls_fact: dace.float32[I, J, K],
    S_RTMIN: dace.float32,
    G_RTMIN: dace.float32,
):
    """
    Correct negative mixing ratios with conservation.
    
    Parameters
    ----------
    th_t : dace.float32[I, J, K]
        Potential temperature (K), modified
    rv_t : dace.float32[I, J, K]
        Water vapor mixing ratio (kg/kg), modified
    rc_t : dace.float32[I, J, K]
        Cloud droplet mixing ratio (kg/kg), modified
    rr_t : dace.float32[I, J, K]
        Rain mixing ratio (kg/kg), modified
    ri_t : dace.float32[I, J, K]
        Ice crystal mixing ratio (kg/kg), modified
    rs_t : dace.float32[I, J, K]
        Snow mixing ratio (kg/kg), modified
    rg_t : dace.float32[I, J, K]
        Graupel mixing ratio (kg/kg), modified
    lv_fact : dace.float32[I, J, K]
        Latent heat of vaporization (over cph)
    ls_fact : dace.float32[I, J, K]
        Latent heat of sublimation (over cph)
    S_RTMIN : dace.float32
        Minimum snow mixing ratio threshold
    G_RTMIN : dace.float32
        Minimum graupel mixing ratio threshold
    """
    @dace.map
    def correct_negatives(i: _[0:I], j: _[0:J], k: _[0:K]):
        # Correct negative cloud droplets
        w = rc_t[i, j, k] - max(rc_t[i, j, k], 0.0)
        rv_t[i, j, k] += w
        th_t[i, j, k] -= w * lv_fact[i, j, k]
        rc_t[i, j, k] -= w
        
        # Correct negative rain
        w = rr_t[i, j, k] - max(rr_t[i, j, k], 0.0)
        rv_t[i, j, k] += w
        th_t[i, j, k] -= w * lv_fact[i, j, k]
        rr_t[i, j, k] -= w
        
        # Correct negative ice
        w = ri_t[i, j, k] - max(ri_t[i, j, k], 0.0)
        rv_t[i, j, k] += w
        th_t[i, j, k] -= w * ls_fact[i, j, k]
        ri_t[i, j, k] -= w
        
        # Correct negative snow
        w = rs_t[i, j, k] - max(rs_t[i, j, k], 0.0)
        rv_t[i, j, k] += w
        th_t[i, j, k] -= w * ls_fact[i, j, k]
        rs_t[i, j, k] -= w
        
        # Correct negative graupel
        w = rg_t[i, j, k] - max(rg_t[i, j, k], 0.0)
        rv_t[i, j, k] += w
        th_t[i, j, k] -= w * ls_fact[i, j, k]
        rg_t[i, j, k] -= w
        
        # Correct negative vapor by evaporating snow
        w = min(max(rs_t[i, j, k], 0.0), max(S_RTMIN - rv_t[i, j, k], 0.0))
        rv_t[i, j, k] += w
        rs_t[i, j, k] -= w
        th_t[i, j, k] -= w * ls_fact[i, j, k]
        
        # Correct negative vapor by evaporating graupel
        w = min(max(rg_t[i, j, k], 0.0), max(G_RTMIN - rv_t[i, j, k], 0.0))
        rv_t[i, j, k] += w
        rg_t[i, j, k] -= w
        th_t[i, j, k] -= w * ls_fact[i, j, k]
