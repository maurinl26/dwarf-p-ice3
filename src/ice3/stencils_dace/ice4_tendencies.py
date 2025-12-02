"""
ICE4 Tendency Management - DaCe Implementation

This module handles initialization, update, and post-processing of
microphysical tendencies in the ICE4 scheme, translated from GT4Py to DaCe.

Reference:
    PHYEX/src/common/micro/mode_ice4_tendencies.F90
    
Author:
    Translated to Python/DaCe from GT4Py by Cline AI Assistant
"""

import dace

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")


@dace.program
def ice4_nucleation_post_processing(
    t: dace.float32[I, J, K],
    exn: dace.float32[I, J, K],
    ls_fact: dace.float32[I, J, K],
    th_t: dace.float32[I, J, K],
    rv_t: dace.float32[I, J, K],
    ri_t: dace.float32[I, J, K],
    rvheni_mr: dace.float32[I, J, K],
):
    """Apply heterogeneous ice nucleation changes to prognostic variables."""
    @dace.map
    def apply_nucleation(i: _[0:I], j: _[0:J], k: _[0:K]):
        th_t[i, j, k] += rvheni_mr[i, j, k] * ls_fact[i, j, k]
        t[i, j, k] = th_t[i, j, k] * exn[i, j, k]
        rv_t[i, j, k] -= rvheni_mr[i, j, k]
        ri_t[i, j, k] += rvheni_mr[i, j, k]


@dace.program
def ice4_rrhong_post_processing(
    t: dace.float32[I, J, K],
    exn: dace.float32[I, J, K],
    ls_fact: dace.float32[I, J, K],
    lv_fact: dace.float32[I, J, K],
    th_t: dace.float32[I, J, K],
    rr_t: dace.float32[I, J, K],
    rg_t: dace.float32[I, J, K],
    rrhong_mr: dace.float32[I, J, K],
):
    """Apply homogeneous freezing of rain to prognostic variables."""
    @dace.map
    def apply_freezing(i: _[0:I], j: _[0:J], k: _[0:K]):
        th_t[i, j, k] += rrhong_mr[i, j, k] * (ls_fact[i, j, k] - lv_fact[i, j, k])
        t[i, j, k] = th_t[i, j, k] * exn[i, j, k]
        rr_t[i, j, k] -= rrhong_mr[i, j, k]
        rg_t[i, j, k] += rrhong_mr[i, j, k]


@dace.program
def ice4_rimltc_post_processing(
    t: dace.float32[I, J, K],
    exn: dace.float32[I, J, K],
    ls_fact: dace.float32[I, J, K],
    lv_fact: dace.float32[I, J, K],
    rimltc_mr: dace.float32[I, J, K],
    th_t: dace.float32[I, J, K],
    rc_t: dace.float32[I, J, K],
    ri_t: dace.float32[I, J, K],
):
    """Apply ice crystal melting to prognostic variables."""
    @dace.map
    def apply_melting(i: _[0:I], j: _[0:J], k: _[0:K]):
        th_t[i, j, k] -= rimltc_mr[i, j, k] * (ls_fact[i, j, k] - lv_fact[i, j, k])
        t[i, j, k] = th_t[i, j, k] * exn[i, j, k]
        rc_t[i, j, k] += rimltc_mr[i, j, k]
        ri_t[i, j, k] -= rimltc_mr[i, j, k]


@dace.program
def ice4_fast_rg_pre_post_processing(
    rgsi: dace.float32[I, J, K],
    rgsi_mr: dace.float32[I, J, K],
    rvdepg: dace.float32[I, J, K],
    rsmltg: dace.float32[I, J, K],
    rraccsg: dace.float32[I, J, K],
    rsaccrg: dace.float32[I, J, K],
    rcrimsg: dace.float32[I, J, K],
    rsrimcg: dace.float32[I, J, K],
    rrhong_mr: dace.float32[I, J, K],
    rsrimcg_mr: dace.float32[I, J, K],
):
    """Aggregate graupel source terms from various processes."""
    @dace.map
    def aggregate_sources(i: _[0:I], j: _[0:J], k: _[0:K]):
        rgsi[i, j, k] = (rvdepg[i, j, k] + rsmltg[i, j, k] + rraccsg[i, j, k] + 
                        rsaccrg[i, j, k] + rcrimsg[i, j, k] + rsrimcg[i, j, k])
        rgsi_mr[i, j, k] = rrhong_mr[i, j, k] + rsrimcg_mr[i, j, k]


@dace.program
def ice4_increment_update(
    ls_fact: dace.float32[I, J, K],
    lv_fact: dace.float32[I, J, K],
    theta_increment: dace.float32[I, J, K],
    rv_increment: dace.float32[I, J, K],
    rc_increment: dace.float32[I, J, K],
    rr_increment: dace.float32[I, J, K],
    ri_increment: dace.float32[I, J, K],
    rs_increment: dace.float32[I, J, K],
    rg_increment: dace.float32[I, J, K],
    rvheni_mr: dace.float32[I, J, K],
    rimltc_mr: dace.float32[I, J, K],
    rrhong_mr: dace.float32[I, J, K],
    rsrimcg_mr: dace.float32[I, J, K],
):
    """Update increment fields with nucleation and phase change processes."""
    @dace.map
    def update_increments(i: _[0:I], j: _[0:J], k: _[0:K]):
        theta_increment[i, j, k] += (
            rvheni_mr[i, j, k] * ls_fact[i, j, k] +
            rrhong_mr[i, j, k] * (ls_fact[i, j, k] - lv_fact[i, j, k]) -
            rimltc_mr[i, j, k] * (ls_fact[i, j, k] - lv_fact[i, j, k])
        )
        
        rv_increment[i, j, k] -= rvheni_mr[i, j, k]
        rc_increment[i, j, k] += rimltc_mr[i, j, k]
        rr_increment[i, j, k] -= rrhong_mr[i, j, k]
        ri_increment[i, j, k] += rvheni_mr[i, j, k] - rimltc_mr[i, j, k]
        rs_increment[i, j, k] -= rsrimcg_mr[i, j, k]
        rg_increment[i, j, k] += rrhong_mr[i, j, k] + rsrimcg_mr[i, j, k]


@dace.program
def ice4_derived_fields(
    t: dace.float32[I, J, K],
    rhodref: dace.float32[I, J, K],
    rv_t: dace.float32[I, J, K],
    pres: dace.float32[I, J, K],
    ssi: dace.float32[I, J, K],
    ka: dace.float32[I, J, K],
    dv: dace.float32[I, J, K],
    ai: dace.float32[I, J, K],
    cj: dace.float32[I, J, K],
    ALPI: dace.float32,
    BETAI: dace.float32,
    GAMI: dace.float32,
    EPSILO: dace.float32,
    TT: dace.float32,
    P00: dace.float32,
    RV: dace.float32,
    LSTT: dace.float32,
    CPV: dace.float32,
    CI: dace.float32,
    SCFAC: dace.float32,
):
    """Compute derived microphysical fields for process calculations."""
    @dace.map
    def compute_derived(i: _[0:I], j: _[0:J], k: _[0:K]):
        # Saturation vapor pressure over ice
        zw = exp(ALPI - BETAI / t[i, j, k] - GAMI * log(t[i, j, k]))
        
        # Supersaturation over ice
        ssi[i, j, k] = rv_t[i, j, k] * (pres[i, j, k] - zw) / (EPSILO * zw) - 1.0
        
        # Thermal conductivity of air
        ka[i, j, k] = 2.38e-2 + 7.1e-5 * (t[i, j, k] - TT)
        
        # Diffusivity of water vapor
        dv[i, j, k] = 2.11e-5 * (t[i, j, k] / TT) ** 1.94 * (P00 / pres[i, j, k])
        
        # Thermodynamic function for deposition
        ai[i, j, k] = ((LSTT + (CPV - CI) * (t[i, j, k] - TT)) ** 2 / 
                      (ka[i, j, k] * RV * t[i, j, k] ** 2) + 
                      (RV * t[i, j, k]) / (dv[i, j, k] * zw))
        
        # Ventilation coefficient
        cj[i, j, k] = SCFAC * rhodref[i, j, k] ** 0.3 / sqrt(1.718e-5 + 4.9e-8 * (t[i, j, k] - TT))


@dace.program
def ice4_slope_parameters(
    rhodref: dace.float32[I, J, K],
    t: dace.float32[I, J, K],
    rr_t: dace.float32[I, J, K],
    rs_t: dace.float32[I, J, K],
    rg_t: dace.float32[I, J, K],
    lbdar: dace.float32[I, J, K],
    lbdar_rf: dace.float32[I, J, K],
    lbdas: dace.float32[I, J, K],
    lbdag: dace.float32[I, J, K],
    LBR: dace.float32,
    LBEXR: dace.float32,
    LBS: dace.float32,
    LBEXS: dace.float32,
    LBG: dace.float32,
    LBEXG: dace.float32,
    R_RTMIN: dace.float32,
    S_RTMIN: dace.float32,
    G_RTMIN: dace.float32,
    LSNOW_T: dace.bool,
    LBDAS_MIN: dace.float32,
    LBDAS_MAX: dace.float32,
    TRANS_MP_GAMMAS: dace.float32,
):
    """Compute slope parameters for hydrometeor size distributions."""
    @dace.map
    def compute_slopes(i: _[0:I], j: _[0:J], k: _[0:K]):
        # Rain slope parameter
        if rr_t[i, j, k] > 0.0:
            lbdar[i, j, k] = LBR * (rhodref[i, j, k] * max(rr_t[i, j, k], R_RTMIN)) ** LBEXR
        else:
            lbdar[i, j, k] = 0.0
        
        lbdar_rf[i, j, k] = lbdar[i, j, k]
        
        # Snow slope parameter
        if LSNOW_T:
            if rs_t[i, j, k] > 0.0 and t[i, j, k] > 263.15:
                lbdas[i, j, k] = (max(min(LBDAS_MAX, 10.0 ** (14.554 - 0.0423 * t[i, j, k])), LBDAS_MIN) * 
                                 TRANS_MP_GAMMAS)
            elif rs_t[i, j, k] > 0.0 and t[i, j, k] <= 263.15:
                lbdas[i, j, k] = (max(min(LBDAS_MAX, 10.0 ** (6.226 - 0.0106 * t[i, j, k])), LBDAS_MIN) * 
                                 TRANS_MP_GAMMAS)
            else:
                lbdas[i, j, k] = 0.0
        else:
            if rs_t[i, j, k] > 0.0:
                lbdas[i, j, k] = min(LBDAS_MAX, LBS * (rhodref[i, j, k] * max(rs_t[i, j, k], S_RTMIN)) ** LBEXS)
            else:
                lbdas[i, j, k] = 0.0
        
        # Graupel slope parameter
        if rg_t[i, j, k] > 0.0:
            lbdag[i, j, k] = LBG * (rhodref[i, j, k] * max(rg_t[i, j, k], G_RTMIN)) ** LBEXG
        else:
            lbdag[i, j, k] = 0.0


@dace.program
def ice4_total_tendencies_update(
    ls_fact: dace.float32[I, J, K],
    lv_fact: dace.float32[I, J, K],
    theta_tnd: dace.float32[I, J, K],
    rv_tnd: dace.float32[I, J, K],
    rc_tnd: dace.float32[I, J, K],
    rr_tnd: dace.float32[I, J, K],
    ri_tnd: dace.float32[I, J, K],
    rs_tnd: dace.float32[I, J, K],
    rg_tnd: dace.float32[I, J, K],
    rchoni: dace.float32[I, J, K],
    rvdeps: dace.float32[I, J, K],
    riaggs: dace.float32[I, J, K],
    riauts: dace.float32[I, J, K],
    rvdepg: dace.float32[I, J, K],
    rcautr: dace.float32[I, J, K],
    rcaccr: dace.float32[I, J, K],
    rrevav: dace.float32[I, J, K],
    rcberi: dace.float32[I, J, K],
    rsmltg: dace.float32[I, J, K],
    rcmltsr: dace.float32[I, J, K],
    rraccss: dace.float32[I, J, K],
    rraccsg: dace.float32[I, J, K],
    rsaccrg: dace.float32[I, J, K],
    rcrimss: dace.float32[I, J, K],
    rcrimsg: dace.float32[I, J, K],
    rsrimcg: dace.float32[I, J, K],
    ricfrrg: dace.float32[I, J, K],
    rrcfrig: dace.float32[I, J, K],
    ricfrr: dace.float32[I, J, K],
    rcwetg: dace.float32[I, J, K],
    riwetg: dace.float32[I, J, K],
    rrwetg: dace.float32[I, J, K],
    rswetg: dace.float32[I, J, K],
    rcdryg: dace.float32[I, J, K],
    ridryg: dace.float32[I, J, K],
    rrdryg: dace.float32[I, J, K],
    rsdryg: dace.float32[I, J, K],
    rgmltr: dace.float32[I, J, K],
    rwetgh: dace.float32,
):
    """Aggregate all microphysical process contributions to total tendencies."""
    @dace.map
    def aggregate_tendencies(i: _[0:I], j: _[0:J], k: _[0:K]):
        # Potential temperature tendency
        theta_tnd[i, j, k] += (
            rvdepg[i, j, k] * ls_fact[i, j, k] +
            rchoni[i, j, k] * (ls_fact[i, j, k] - lv_fact[i, j, k]) +
            rvdeps[i, j, k] * ls_fact[i, j, k] -
            rrevav[i, j, k] * lv_fact[i, j, k] +
            rcrimss[i, j, k] * (ls_fact[i, j, k] - lv_fact[i, j, k]) +
            rcrimsg[i, j, k] * (ls_fact[i, j, k] - lv_fact[i, j, k]) +
            rraccss[i, j, k] * (ls_fact[i, j, k] - lv_fact[i, j, k]) +
            rraccsg[i, j, k] * (ls_fact[i, j, k] - lv_fact[i, j, k]) +
            (rrcfrig[i, j, k] - ricfrr[i, j, k]) * (ls_fact[i, j, k] - lv_fact[i, j, k]) +
            (rcwetg[i, j, k] + rrwetg[i, j, k]) * (ls_fact[i, j, k] - lv_fact[i, j, k]) +
            (rcdryg[i, j, k] + rrdryg[i, j, k]) * (ls_fact[i, j, k] - lv_fact[i, j, k]) -
            rgmltr[i, j, k] * (ls_fact[i, j, k] - lv_fact[i, j, k]) +
            rcberi[i, j, k] * (ls_fact[i, j, k] - lv_fact[i, j, k])
        )
        
        # Vapor tendency
        rv_tnd[i, j, k] += -rvdepg[i, j, k] - rvdeps[i, j, k] + rrevav[i, j, k]
        
        # Cloud tendency
        rc_tnd[i, j, k] += (-rchoni[i, j, k] - rcautr[i, j, k] - rcaccr[i, j, k] -
                           rcrimss[i, j, k] - rcrimsg[i, j, k] - rcmltsr[i, j, k] -
                           rcwetg[i, j, k] - rcdryg[i, j, k] - rcberi[i, j, k])
        
        # Rain tendency
        rr_tnd[i, j, k] += (rcautr[i, j, k] + rcaccr[i, j, k] - rrevav[i, j, k] -
                           rraccss[i, j, k] - rraccsg[i, j, k] + rcmltsr[i, j, k] -
                           rrcfrig[i, j, k] + ricfrr[i, j, k] - rrwetg[i, j, k] -
                           rrdryg[i, j, k] + rgmltr[i, j, k])
        
        # Ice tendency
        ri_tnd[i, j, k] += (rchoni[i, j, k] - riaggs[i, j, k] - riauts[i, j, k] -
                           ricfrrg[i, j, k] - ricfrr[i, j, k] - riwetg[i, j, k] -
                           ridryg[i, j, k] + rcberi[i, j, k])
        
        # Snow tendency
        rs_tnd[i, j, k] += (rvdeps[i, j, k] + riaggs[i, j, k] + riauts[i, j, k] +
                           rcrimss[i, j, k] - rsrimcg[i, j, k] + rraccss[i, j, k] -
                           rsaccrg[i, j, k] - rsmltg[i, j, k] - rswetg[i, j, k] -
                           rsdryg[i, j, k])
        
        # Graupel tendency
        rg_tnd[i, j, k] += (rvdepg[i, j, k] + rcrimsg[i, j, k] + rsrimcg[i, j, k] +
                           rraccsg[i, j, k] + rsaccrg[i, j, k] + rsmltg[i, j, k] +
                           ricfrrg[i, j, k] + rrcfrig[i, j, k] + rcwetg[i, j, k] +
                           riwetg[i, j, k] + rswetg[i, j, k] + rrwetg[i, j, k] +
                           rcdryg[i, j, k] + ridryg[i, j, k] + rsdryg[i, j, k] +
                           rrdryg[i, j, k] - rgmltr[i, j, k] - rwetgh)
