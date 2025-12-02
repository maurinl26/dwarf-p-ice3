"""
ICE4 Slow Processes - DaCe Implementation

This module implements slow cold microphysical processes for the ICE4
microphysics scheme, translated from GT4Py to DaCe.

Processes implemented:
- Homogeneous nucleation (RCHONI)
- Vapor deposition on snow (RVDEPS)
- Ice aggregation to snow (RIAGGS)
- Ice autoconversion to snow (RIAUTS)
- Vapor deposition on graupel (RVDEPG)

Reference:
    PHYEX/src/common/micro/mode_ice4_slow.F90
    
Author:
    Translated to Python/DaCe from GT4Py by Cline AI Assistant
"""

import dace

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")


@dace.program
def ice4_slow(
    ldcompute: dace.bool[I, J, K],
    rhodref: dace.float32[I, J, K],
    t: dace.float32[I, J, K],
    ssi: dace.float32[I, J, K],
    rv_t: dace.float32[I, J, K],
    rc_t: dace.float32[I, J, K],
    ri_t: dace.float32[I, J, K],
    rs_t: dace.float32[I, J, K],
    rg_t: dace.float32[I, J, K],
    lbdas: dace.float32[I, J, K],
    lbdag: dace.float32[I, J, K],
    ai: dace.float32[I, J, K],
    cj: dace.float32[I, J, K],
    hli_hcf: dace.float32[I, J, K],
    hli_hri: dace.float32[I, J, K],
    rc_honi_tnd: dace.float32[I, J, K],
    rv_deps_tnd: dace.float32[I, J, K],
    ri_aggs_tnd: dace.float32[I, J, K],
    ri_auts_tnd: dace.float32[I, J, K],
    rv_depg_tnd: dace.float32[I, J, K],
    ldsoft: dace.bool,
    V_RTMIN: dace.float32,
    C_RTMIN: dace.float32,
    I_RTMIN: dace.float32,
    S_RTMIN: dace.float32,
    G_RTMIN: dace.float32,
    TT: dace.float32,
    HON: dace.float32,
    ALPHA3: dace.float32,
    BETA3: dace.float32,
    O0DEPS: dace.float32,
    O1DEPS: dace.float32,
    EX0DEPS: dace.float32,
    EX1DEPS: dace.float32,
    FIAGGS: dace.float32,
    COLEXIS: dace.float32,
    EXIAGGS: dace.float32,
    CEXVT: dace.float32,
    TIMAUTI: dace.float32,
    TEXAUTI: dace.float32,
    CRIAUTI: dace.float32,
    ACRIAUTI: dace.float32,
    BCRIAUTI: dace.float32,
    O0DEPG: dace.float32,
    O1DEPG: dace.float32,
    EX0DEPG: dace.float32,
    EX1DEPG: dace.float32,
):
    """
    Compute slow cold microphysical processes for ICE4 scheme.
    
    Parameters
    ----------
    ldcompute : dace.bool[I, J, K]
        Computation mask
    rhodref : dace.float32[I, J, K]
        Reference air density (kg/m³)
    t : dace.float32[I, J, K]
        Temperature (K)
    ssi : dace.float32[I, J, K]
        Supersaturation over ice
    rv_t, rc_t, ri_t, rs_t, rg_t : dace.float32[I, J, K]
        Mixing ratios (kg/kg)
    lbdas, lbdag : dace.float32[I, J, K]
        Slope parameters (m⁻¹)
    ai : dace.float32[I, J, K]
        Thermodynamic diffusion coefficient
    cj : dace.float32[I, J, K]
        Ventilation coefficient
    hli_hcf, hli_hri : dace.float32[I, J, K]
        Subgrid cloud fraction and ice
    rc_honi_tnd, rv_deps_tnd, ri_aggs_tnd, ri_auts_tnd, rv_depg_tnd : dace.float32[I, J, K]
        Output tendencies (kg/kg/s)
    ldsoft : dace.bool
        Soft threshold mode
    V_RTMIN, C_RTMIN, I_RTMIN, S_RTMIN, G_RTMIN : dace.float32
        Minimum thresholds
    TT : dace.float32
        Triple point temperature
    HON, ALPHA3, BETA3 : dace.float32
        Homogeneous nucleation parameters
    O0DEPS, O1DEPS, EX0DEPS, EX1DEPS : dace.float32
        Snow deposition parameters
    FIAGGS, COLEXIS, EXIAGGS, CEXVT : dace.float32
        Aggregation parameters
    TIMAUTI, TEXAUTI, CRIAUTI, ACRIAUTI, BCRIAUTI : dace.float32
        Autoconversion parameters
    O0DEPG, O1DEPG, EX0DEPG, EX1DEPG : dace.float32
        Graupel deposition parameters
    """
    @dace.map
    def compute_slow_processes(i: _[0:I], j: _[0:J], k: _[0:K]):
        # Homogeneous nucleation (RCHONI)
        if t[i, j, k] < TT - 35.0 and rc_t[i, j, k] > C_RTMIN and ldcompute[i, j, k]:
            if not ldsoft:
                rc_honi_tnd[i, j, k] = min(1000.0, HON * rhodref[i, j, k] * rc_t[i, j, k] * 
                                          exp(ALPHA3 * (t[i, j, k] - TT) - BETA3))
            else:
                rc_honi_tnd[i, j, k] = 0.0
        else:
            rc_honi_tnd[i, j, k] = 0.0
        
        # Vapor deposition on snow (RVDEPS)
        if rv_t[i, j, k] > V_RTMIN and rs_t[i, j, k] > S_RTMIN and ldcompute[i, j, k]:
            if not ldsoft:
                rv_deps_tnd[i, j, k] = (ssi[i, j, k] / (rhodref[i, j, k] * ai[i, j, k])) * (
                    O0DEPS * lbdas[i, j, k] ** EX0DEPS + 
                    O1DEPS * cj[i, j, k] * lbdas[i, j, k] ** EX1DEPS
                )
            else:
                rv_deps_tnd[i, j, k] = 0.0
        else:
            rv_deps_tnd[i, j, k] = 0.0
        
        # Ice aggregation to snow (RIAGGS)
        if ri_t[i, j, k] > I_RTMIN and rs_t[i, j, k] > S_RTMIN and ldcompute[i, j, k]:
            if not ldsoft:
                ri_aggs_tnd[i, j, k] = (FIAGGS * exp(COLEXIS * (t[i, j, k] - TT)) * 
                                       ri_t[i, j, k] * lbdas[i, j, k] ** EXIAGGS * 
                                       rhodref[i, j, k] ** (-CEXVT))
            else:
                ri_aggs_tnd[i, j, k] = 0.0
        else:
            ri_aggs_tnd[i, j, k] = 0.0
        
        # Ice autoconversion to snow (RIAUTS)
        if hli_hri[i, j, k] > I_RTMIN and ldcompute[i, j, k]:
            if not ldsoft:
                criauti_tmp = min(CRIAUTI, 10.0 ** (ACRIAUTI * (t[i, j, k] - TT) + BCRIAUTI))
                ri_auts_tnd[i, j, k] = (TIMAUTI * exp(TEXAUTI * (t[i, j, k] - TT)) * 
                                       max(0.0, hli_hri[i, j, k] - criauti_tmp * hli_hcf[i, j, k]))
            else:
                ri_auts_tnd[i, j, k] = 0.0
        else:
            ri_auts_tnd[i, j, k] = 0.0
        
        # Vapor deposition on graupel (RVDEPG)
        if rv_t[i, j, k] > V_RTMIN and rg_t[i, j, k] > G_RTMIN and ldcompute[i, j, k]:
            if not ldsoft:
                rv_depg_tnd[i, j, k] = (ssi[i, j, k] / (rhodref[i, j, k] * ai[i, j, k])) * (
                    O0DEPG * lbdag[i, j, k] ** EX0DEPG + 
                    O1DEPG * cj[i, j, k] * lbdag[i, j, k] ** EX1DEPG
                )
            else:
                rv_depg_tnd[i, j, k] = 0.0
        else:
            rv_depg_tnd[i, j, k] = 0.0
