"""
Cloud Fraction Computation - DaCe Implementation

This module implements cloud fraction computation and subgrid autoconversion
for the ICE3/ICE4 microphysics scheme, translated from GT4Py to DaCe.

Stencils implemented:
- thermodynamic_fields: Compute temperature, latent heats, and moist air specific heat
- cloud_fraction_1: Compute mixing ratio sources from condensation/evaporation
- cloud_fraction_2: Compute cloud fraction and subgrid autoconversion

Reference:
    PHYEX/src/common/micro/ice_adjust.F90
    
Author:
    Translated to Python/DaCe from GT4Py by Cline AI Assistant
"""

import dace

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")


def vaporisation_latent_heat(t: dace.float32) -> dace.float32:
    """Compute temperature-dependent latent heat of vaporization"""
    # Constants from PHYEX
    XLVTT = 2.5008e6  # Latent heat of vaporization at triple point (J/kg)
    XCPV = 1846.0     # Specific heat of water vapor (J/(kg·K))
    XCL = 4218.0      # Specific heat of liquid water (J/(kg·K))
    XTT = 273.16      # Triple point temperature (K)
    
    return XLVTT + (XCPV - XCL) * (t - XTT)


def sublimation_latent_heat(t: dace.float32) -> dace.float32:
    """Compute temperature-dependent latent heat of sublimation"""
    # Constants from PHYEX
    XLSTT = 2.8345e6  # Latent heat of sublimation at triple point (J/kg)
    XCPV = 1846.0     # Specific heat of water vapor (J/(kg·K))
    XCI = 2106.0      # Specific heat of ice (J/(kg·K))
    XTT = 273.16      # Triple point temperature (K)
    
    return XLSTT + (XCPV - XCI) * (t - XTT)


@dace.program
def thermodynamic_fields(
    th: dace.float32[I, J, K],
    exn: dace.float32[I, J, K],
    rv: dace.float32[I, J, K],
    rc: dace.float32[I, J, K],
    rr: dace.float32[I, J, K],
    ri: dace.float32[I, J, K],
    rs: dace.float32[I, J, K],
    rg: dace.float32[I, J, K],
    lv: dace.float32[I, J, K],
    ls: dace.float32[I, J, K],
    cph: dace.float32[I, J, K],
    t: dace.float32[I, J, K],
    NRR: dace.int32,
    CPD: dace.float32,
    CPV: dace.float32,
    CL: dace.float32,
    CI: dace.float32,
):
    """
    Compute temperature, latent heats, and moist air specific heat.
    
    Parameters
    ----------
    th : dace.float32[I, J, K]
        Potential temperature (K)
    exn : dace.float32[I, J, K]
        Exner function (dimensionless)
    rv, rc, rr, ri, rs, rg : dace.float32[I, J, K]
        Mixing ratios for vapor, cloud, rain, ice, snow, graupel (kg/kg)
    lv, ls : dace.float32[I, J, K]
        Output: Latent heats of vaporization and sublimation (J/kg)
    cph : dace.float32[I, J, K]
        Output: Specific heat of moist air (J/(kg·K))
    t : dace.float32[I, J, K]
        Output: Temperature (K)
    NRR : dace.int32
        Number of hydrometeor species (2, 4, 5, or 6)
    CPD, CPV, CL, CI : dace.float32
        Specific heats (dry air, vapor, liquid, ice)
    """
    @dace.map
    def compute_thermo(i: _[0:I], j: _[0:J], k: _[0:K]):
        # Compute temperature
        t[i, j, k] = th[i, j, k] * exn[i, j, k]
        
        # Compute latent heats
        lv[i, j, k] = vaporisation_latent_heat(t[i, j, k])
        ls[i, j, k] = sublimation_latent_heat(t[i, j, k])
        
        # Compute specific heat based on NRR
        if NRR == 6:
            cph[i, j, k] = CPD + CPV * rv[i, j, k] + CL * (rc[i, j, k] + rr[i, j, k]) + \
                          CI * (ri[i, j, k] + rs[i, j, k] + rg[i, j, k])
        elif NRR == 5:
            cph[i, j, k] = CPD + CPV * rv[i, j, k] + CL * (rc[i, j, k] + rr[i, j, k]) + \
                          CI * (ri[i, j, k] + rs[i, j, k])
        elif NRR == 4:
            cph[i, j, k] = CPD + CPV * rv[i, j, k] + CL * (rc[i, j, k] + rr[i, j, k])
        elif NRR == 2:
            cph[i, j, k] = CPD + CPV * rv[i, j, k] + CL * rc[i, j, k] + CI * ri[i, j, k]


@dace.program
def cloud_fraction_1(
    lv: dace.float32[I, J, K],
    ls: dace.float32[I, J, K],
    cph: dace.float32[I, J, K],
    exnref: dace.float32[I, J, K],
    rc: dace.float32[I, J, K],
    ri: dace.float32[I, J, K],
    ths: dace.float32[I, J, K],
    rvs: dace.float32[I, J, K],
    rcs: dace.float32[I, J, K],
    ris: dace.float32[I, J, K],
    rc_tmp: dace.float32[I, J, K],
    ri_tmp: dace.float32[I, J, K],
    dt: dace.float32,
):
    """
    Compute mixing ratio sources from condensation/evaporation tendencies.
    
    Parameters
    ----------
    lv, ls : dace.float32[I, J, K]
        Latent heats of vaporization and sublimation (J/kg)
    cph : dace.float32[I, J, K]
        Specific heat of moist air (J/(kg·K))
    exnref : dace.float32[I, J, K]
        Reference Exner function
    rc, ri : dace.float32[I, J, K]
        Initial cloud and ice mixing ratios (kg/kg)
    ths, rvs, rcs, ris : dace.float32[I, J, K]
        Source terms (modified in place)
    rc_tmp, ri_tmp : dace.float32[I, J, K]
        Temporary mixing ratios after condensation
    dt : dace.float32
        Time step (s)
    """
    @dace.map
    def compute_sources(i: _[0:I], j: _[0:J], k: _[0:K]):
        # Compute tendencies
        w1 = (rc_tmp[i, j, k] - rc[i, j, k]) / dt
        w2 = (ri_tmp[i, j, k] - ri[i, j, k]) / dt
        
        # Limit w1 conservatively
        if w1 < 0.0:
            w1 = max(w1, -rcs[i, j, k])
        else:
            w1 = min(w1, rvs[i, j, k])
        
        # Update sources for liquid
        rvs[i, j, k] -= w1
        rcs[i, j, k] += w1
        ths[i, j, k] += w1 * lv[i, j, k] / (cph[i, j, k] * exnref[i, j, k])
        
        # Limit w2 conservatively
        if w2 < 0.0:
            w2 = max(w2, -ris[i, j, k])
        else:
            w2 = min(w2, rvs[i, j, k])
        
        # Update sources for ice
        rvs[i, j, k] -= w2
        ris[i, j, k] += w2
        ths[i, j, k] += w2 * ls[i, j, k] / (cph[i, j, k] * exnref[i, j, k])


@dace.program
def cloud_fraction_2(
    rhodref: dace.float32[I, J, K],
    exnref: dace.float32[I, J, K],
    t: dace.float32[I, J, K],
    cph: dace.float32[I, J, K],
    lv: dace.float32[I, J, K],
    ls: dace.float32[I, J, K],
    ths: dace.float32[I, J, K],
    rvs: dace.float32[I, J, K],
    rcs: dace.float32[I, J, K],
    ris: dace.float32[I, J, K],
    rc_mf: dace.float32[I, J, K],
    ri_mf: dace.float32[I, J, K],
    cf_mf: dace.float32[I, J, K],
    cldfr: dace.float32[I, J, K],
    hlc_hrc: dace.float32[I, J, K],
    hlc_hcf: dace.float32[I, J, K],
    hli_hri: dace.float32[I, J, K],
    hli_hcf: dace.float32[I, J, K],
    dt: dace.float32,
    LSUBG_COND: dace.bool,
    SUBG_MF_PDF: dace.int32,
    CRIAUTC: dace.float32,
    CRIAUTI: dace.float32,
    ACRIAUTI: dace.float32,
    BCRIAUTI: dace.float32,
    TT: dace.float32,
):
    """
    Compute cloud fraction and subgrid autoconversion.
    
    Parameters
    ----------
    rhodref : dace.float32[I, J, K]
        Reference air density (kg/m³)
    exnref : dace.float32[I, J, K]
        Reference Exner function
    t : dace.float32[I, J, K]
        Temperature (K)
    cph, lv, ls : dace.float32[I, J, K]
        Thermodynamic fields
    ths, rvs, rcs, ris : dace.float32[I, J, K]
        Source terms (modified)
    rc_mf, ri_mf, cf_mf : dace.float32[I, J, K]
        Mass flux condensate and cloud fraction
    cldfr : dace.float32[I, J, K]
        Total cloud fraction (modified)
    hlc_hrc, hlc_hcf, hli_hri, hli_hcf : dace.float32[I, J, K]
        Autoconversion diagnostics (modified)
    dt : dace.float32
        Time step (s)
    LSUBG_COND : dace.bool
        Enable subgrid condensation
    SUBG_MF_PDF : dace.int32
        PDF method (0=none, 1=triangle)
    CRIAUTC, CRIAUTI : dace.float32
        Autoconversion thresholds
    ACRIAUTI, BCRIAUTI : dace.float32
        Ice autoconversion parameters
    TT : dace.float32
        Triple point temperature (K)
    """
    @dace.map
    def compute_cloud_fraction(i: _[0:I], j: _[0:J], k: _[0:K]):
        if not LSUBG_COND:
            # Binary cloud fraction
            if (rcs[i, j, k] + ris[i, j, k]) * dt > 1e-12:
                cldfr[i, j, k] = 1.0
            else:
                cldfr[i, j, k] = 0.0
        else:
            # Subgrid condensation mode
            w1 = rc_mf[i, j, k] / dt
            w2 = ri_mf[i, j, k] / dt
            
            # Limit by available vapor
            if w1 + w2 > rvs[i, j, k]:
                w1_limited = w1 * rvs[i, j, k] / (w1 + w2)
                w2 = rvs[i, j, k] - w1_limited
                w1 = w1_limited
            
            # Update cloud fraction
            cldfr[i, j, k] = min(1.0, cldfr[i, j, k] + cf_mf[i, j, k])
            
            # Update sources
            rcs[i, j, k] += w1
            ris[i, j, k] += w2
            rvs[i, j, k] -= (w1 + w2)
            ths[i, j, k] += (w1 * lv[i, j, k] + w2 * ls[i, j, k]) / (cph[i, j, k] * exnref[i, j, k])
            
            # Liquid autoconversion
            criaut = CRIAUTC / rhodref[i, j, k]
            
            if SUBG_MF_PDF == 0:
                # Step function
                if w1 * dt > cf_mf[i, j, k] * criaut:
                    hlc_hrc[i, j, k] += w1 * dt
                    hlc_hcf[i, j, k] = min(1.0, hlc_hcf[i, j, k] + cf_mf[i, j, k])
            
            elif SUBG_MF_PDF == 1:
                # Triangular PDF
                if w1 * dt > cf_mf[i, j, k] * criaut:
                    hcf = 1.0 - 0.5 * (criaut * cf_mf[i, j, k] / max(1e-20, w1 * dt)) ** 2
                    hr = w1 * dt - (criaut * cf_mf[i, j, k]) ** 3 / (3 * max(1e-20, w1 * dt) ** 2)
                elif 2.0 * w1 * dt <= cf_mf[i, j, k] * criaut:
                    hcf = 0.0
                    hr = 0.0
                else:
                    hcf = (2.0 * w1 * dt - criaut * cf_mf[i, j, k]) ** 2 / (2.0 * max(1.0e-20, w1 * dt) ** 2)
                    hr = (4.0 * (w1 * dt) ** 3 - 3.0 * w1 * dt * (criaut * cf_mf[i, j, k]) ** 2 + 
                          (criaut * cf_mf[i, j, k]) ** 3) / (3 * max(1.0e-20, w1 * dt) ** 2)
                
                hcf *= cf_mf[i, j, k]
                hlc_hcf[i, j, k] = min(1.0, hlc_hcf[i, j, k] + hcf)
                hlc_hrc[i, j, k] += hr
            
            # Ice autoconversion
            criaut_ice = min(CRIAUTI, 10.0 ** (ACRIAUTI * (t[i, j, k] - TT) + BCRIAUTI))
            
            if SUBG_MF_PDF == 0:
                # Step function
                if w2 * dt > cf_mf[i, j, k] * criaut_ice:
                    hli_hri[i, j, k] += w2 * dt
                    hli_hcf[i, j, k] = min(1.0, hli_hcf[i, j, k] + cf_mf[i, j, k])
            
            elif SUBG_MF_PDF == 1:
                # Triangular PDF
                if w2 * dt > cf_mf[i, j, k] * criaut_ice:
                    hcf = 1.0 - 0.5 * ((criaut_ice * cf_mf[i, j, k]) / (w2 * dt)) ** 2
                    hri = w2 * dt - (criaut_ice * cf_mf[i, j, k]) ** 3 / (3 * (w2 * dt) ** 2)
                elif 2.0 * w2 * dt <= cf_mf[i, j, k] * criaut_ice:
                    hcf = 0.0
                    hri = 0.0
                else:
                    hcf = (2.0 * w2 * dt - criaut_ice * cf_mf[i, j, k]) ** 2 / (2.0 * (w2 * dt) ** 2)
                    hri = (4.0 * (w2 * dt) ** 3 - 3.0 * w2 * dt * (criaut_ice * cf_mf[i, j, k]) ** 2 + 
                          (criaut_ice * cf_mf[i, j, k]) ** 3) / (3.0 * (w2 * dt) ** 2)
                
                hcf *= cf_mf[i, j, k]
                hli_hcf[i, j, k] = min(1.0, hli_hcf[i, j, k] + hcf)
                hli_hri[i, j, k] += hri
