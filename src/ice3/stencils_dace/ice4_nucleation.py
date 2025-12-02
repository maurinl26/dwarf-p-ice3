# -*- coding: utf-8 -*-
"""
ICE4 Nucleation - DaCe Implementation

This module implements heterogeneous ice nucleation for the ICE4
microphysics scheme, translated from GT4Py to DaCe.

Process implemented:
- Heterogeneous ice nucleation on ice nuclei (HENI process)

Reference:
    PHYEX/src/common/micro/ice4_nucleation.func.h
    
Author:
    Translated to Python/DaCe from GT4Py by Cline AI Assistant
"""

import dace

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")


@dace.program
def ice4_nucleation(
    ldcompute: dace.bool[I, J, K],
    tht: dace.float32[I, J, K],
    pabst: dace.float32[I, J, K],
    rhodref: dace.float32[I, J, K],
    exn: dace.float32[I, J, K],
    lsfact: dace.float32[I, J, K],
    t: dace.float32[I, J, K],
    rvt: dace.float32[I, J, K],
    cit: dace.float32[I, J, K],
    rvheni_mr: dace.float32[I, J, K],
    ssi: dace.float32[I, J, K],
    LFEEDBACKT: dace.bool,
    ALPHA1: dace.float32,
    ALPHA2: dace.float32,
    BETA1: dace.float32,
    BETA2: dace.float32,
    NU10: dace.float32,
    NU20: dace.float32,
    MNU0: dace.float32,
    TT: dace.float32,
    V_RTMIN: dace.float32,
    EPSILO: dace.float32,
    ALPW: dace.float32,
    BETAW: dace.float32,
    GAMW: dace.float32,
    ALPI: dace.float32,
    BETAI: dace.float32,
    GAMI: dace.float32,
):
    """
    Compute heterogeneous ice nucleation.
    
    Parameters
    ----------
    ldcompute : dace.bool[I, J, K]
        Computation mask
    tht : dace.float32[I, J, K]
        Potential temperature (K)
    pabst : dace.float32[I, J, K]
        Absolute pressure (Pa)
    rhodref : dace.float32[I, J, K]
        Reference air density (kg/m³)
    exn : dace.float32[I, J, K]
        Exner function
    lsfact : dace.float32[I, J, K]
        Latent heat factor for sublimation (K·kg/kg)
    t : dace.float32[I, J, K]
        Temperature (K)
    rvt : dace.float32[I, J, K]
        Water vapor mixing ratio (kg/kg)
    cit : dace.float32[I, J, K]
        Ice crystal number concentration (1/kg), modified
    rvheni_mr : dace.float32[I, J, K]
        Output: Vapor change due to nucleation (kg/kg)
    ssi : dace.float32[I, J, K]
        Output: Supersaturation over ice
    LFEEDBACKT : dace.bool
        Enable temperature feedback
    ALPHA1, ALPHA2, BETA1, BETA2 : dace.float32
        Nucleation parameterization constants
    NU10, NU20 : dace.float32
        Nucleation rate constants (1/kg)
    MNU0 : dace.float32
        Mean initial ice crystal mass (kg)
    TT : dace.float32
        Triple point temperature (K)
    V_RTMIN : dace.float32
        Minimum vapor mixing ratio threshold
    EPSILO : dace.float32
        Ratio of gas constants (Rd/Rv)
    ALPW, BETAW, GAMW : dace.float32
        Saturation vapor pressure constants (liquid)
    ALPI, BETAI, GAMI : dace.float32
        Saturation vapor pressure constants (ice)
    """
    @dace.map
    def compute_nucleation(i: _[0:I], j: _[0:J], k: _[0:K]):
        # Initialize outputs
        ssi[i, j, k] = 0.0
        rvheni_mr[i, j, k] = 0.0
        
        if t[i, j, k] < TT and rvt[i, j, k] > V_RTMIN and ldcompute[i, j, k]:
            # Compute saturation vapor pressures
            zw_log = log(t[i, j, k])
            usw = exp(ALPW - BETAW / t[i, j, k] - GAMW * zw_log)
            zw = exp(ALPI - BETAI / t[i, j, k] - GAMI * zw_log)
            
            # Limit to half of ambient pressure
            zw = min(pabst[i, j, k] / 2.0, zw)
            
            # Compute supersaturation over ice
            ssi[i, j, k] = rvt[i, j, k] * (pabst[i, j, k] - zw) / (EPSILO * zw) - 1.0
            
            # Compute supersaturation of water-saturated air over ice
            usw = min(pabst[i, j, k] / 2.0, usw)
            usw = (usw / zw) * ((pabst[i, j, k] - zw) / (pabst[i, j, k] - usw)) - 1.0
            
            # Limit ssi by water saturation
            ssi[i, j, k] = min(ssi[i, j, k], usw)
            
            # Compute nucleation rate based on temperature regime
            zw_nuc = 0.0
            if t[i, j, k] < TT - 5.0 and ssi[i, j, k] > 0.0:
                # Cold regime (T < -5°C): NU20 parameterization
                zw_nuc = NU20 * exp(ALPHA2 * ssi[i, j, k] - BETA2)
            elif t[i, j, k] <= TT - 2.0 and t[i, j, k] >= TT - 5.0 and ssi[i, j, k] > 0.0:
                # Transition regime (-5°C ≤ T < -2°C): max of two parameterizations
                zw_nuc = max(
                    NU20 * exp(-BETA2),
                    NU10 * exp(-BETA1 * (t[i, j, k] - TT)) * (ssi[i, j, k] / usw) ** ALPHA1
                )
            
            # Compute net nucleation (subtract existing ice crystals)
            zw_nuc = zw_nuc - cit[i, j, k]
            zw_nuc = min(zw_nuc, 5e4)
            
            # Convert to vapor mass change
            rvheni_mr[i, j, k] = max(zw_nuc, 0.0) * MNU0 / rhodref[i, j, k]
            rvheni_mr[i, j, k] = min(rvt[i, j, k], rvheni_mr[i, j, k])
            
            # Temperature feedback limiter (optional)
            if LFEEDBACKT:
                w1 = min(rvheni_mr[i, j, k], 
                        max(0.0, (TT / exn[i, j, k] - tht[i, j, k])) / lsfact[i, j, k]) / \
                     max(rvheni_mr[i, j, k], 1e-20)
                rvheni_mr[i, j, k] *= w1
                zw_nuc *= w1
            
            # Update ice crystal concentration
            cit[i, j, k] = max(zw_nuc + cit[i, j, k], cit[i, j, k])
