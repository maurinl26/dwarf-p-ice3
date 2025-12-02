# -*- coding: utf-8 -*-
"""
ICE4 Fast RI (Ice Crystals) - DaCe Implementation

This module implements the Bergeron-Findeisen effect for the ICE4
microphysics scheme, translated from GT4Py to DaCe.

Process implemented:
- Bergeron-Findeisen effect: Cloud droplets evaporate to deposit on ice crystals

Reference:
    PHYEX/src/common/micro/mode_ice4_fast_ri.F90
    
Author:
    Translated to Python/DaCe from GT4Py by Cline AI Assistant
"""

import dace

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")


@dace.program
def ice4_fast_ri(
    ldcompute: dace.bool[I, J, K],
    rhodref: dace.float32[I, J, K],
    ai: dace.float32[I, J, K],
    cj: dace.float32[I, J, K],
    cit: dace.float32[I, J, K],
    ssi: dace.float32[I, J, K],
    rct: dace.float32[I, J, K],
    rit: dace.float32[I, J, K],
    rc_beri_tnd: dace.float32[I, J, K],
    ldsoft: dace.bool,
    C_RTMIN: dace.float32,
    I_RTMIN: dace.float32,
    LBI: dace.float32,
    LBEXI: dace.float32,
    O0DEPI: dace.float32,
    O2DEPI: dace.float32,
    DI: dace.float32,
):
    """
    Compute the Bergeron-Findeisen effect (RCBERI tendency).
    
    Parameters
    ----------
    ldcompute : dace.bool[I, J, K]
        Computation mask
    rhodref : dace.float32[I, J, K]
        Reference air density (kg/m³)
    ai : dace.float32[I, J, K]
        Thermodynamical function for vapor diffusion
    cj : dace.float32[I, J, K]
        Ventilation coefficient
    cit : dace.float32[I, J, K]
        Ice crystal number concentration (m⁻³)
    ssi : dace.float32[I, J, K]
        Supersaturation with respect to ice
    rct : dace.float32[I, J, K]
        Cloud liquid water mixing ratio (kg/kg)
    rit : dace.float32[I, J, K]
        Ice crystal mixing ratio (kg/kg)
    rc_beri_tnd : dace.float32[I, J, K]
        Output: Cloud water tendency due to Bergeron-Findeisen (kg/kg/s)
    ldsoft : dace.bool
        Flag for using precomputed tendencies
    C_RTMIN, I_RTMIN : dace.float32
        Minimum thresholds for cloud and ice
    LBI, LBEXI : dace.float32
        Ice crystal size distribution parameters
    O0DEPI, O2DEPI : dace.float32
        Deposition coefficients
    DI : dace.float32
        Ice crystal size distribution exponent
    """
    @dace.map
    def compute_bergeron(i: _[0:I], j: _[0:J], k: _[0:K]):
        if (ssi[i, j, k] > 0.0 and 
            rct[i, j, k] > C_RTMIN and 
            rit[i, j, k] > I_RTMIN and 
            cit[i, j, k] > 1e-20 and 
            ldcompute[i, j, k]):
            
            if not ldsoft:
                # Compute ice crystal slope parameter lambda_i
                lambda_i = min(1e8, LBI * (rhodref[i, j, k] * rit[i, j, k] / cit[i, j, k]) ** LBEXI)
                
                # Compute deposition rate with ventilation correction
                rc_beri_tnd[i, j, k] = (ssi[i, j, k] / (rhodref[i, j, k] * ai[i, j, k])) * \
                    cit[i, j, k] * (O0DEPI / lambda_i + O2DEPI * cj[i, j, k] ** 2 / lambda_i ** (DI + 2.0))
        else:
            rc_beri_tnd[i, j, k] = 0.0
