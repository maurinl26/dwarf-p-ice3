"""
ICE4 Slow Processes - JAX Implementation

This module computes slow cold microphysical processes: homogeneous nucleation,
vapor deposition, ice aggregation, and ice autoconversion.

Reference:
    PHYEX/src/common/micro/mode_ice4_slow.F90
"""
import jax
import jax.numpy as jnp
from typing import Tuple


def ice4_slow(
    rhodref: jnp.ndarray,
    t: jnp.ndarray,
    ssi: jnp.ndarray,
    rvt: jnp.ndarray,
    rct: jnp.ndarray,
    rit: jnp.ndarray,
    rst: jnp.ndarray,
    rgt: jnp.ndarray,
    lbdas: jnp.ndarray,
    lbdag: jnp.ndarray,
    ai: jnp.ndarray,
    cj: jnp.ndarray,
    hli_hcf: jnp.ndarray,
    hli_hri: jnp.ndarray,
    ldcompute: jnp.ndarray,
    ldsoft: bool,
    constants: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute slow cold microphysical processes.
    
    This function computes five slow (temperature-dependent) processes:
    1. Homogeneous nucleation: cloud droplets → ice crystals
    2. Vapor deposition on snow: water vapor → snow
    3. Ice aggregation to snow: ice crystals → snow (collection)
    4. Ice autoconversion to snow: ice crystals → snow (size threshold)
    5. Vapor deposition on graupel: water vapor → graupel
    
    Args:
        rhodref: Reference air density (kg/m³)
        t: Temperature (K)
        ssi: Supersaturation with respect to ice (dimensionless)
        rvt: Water vapor mixing ratio (kg/kg)
        rct: Cloud droplet mixing ratio (kg/kg)
        rit: Ice crystal mixing ratio (kg/kg)
        rst: Snow mixing ratio (kg/kg)
        rgt: Graupel mixing ratio (kg/kg)
        lbdas: Snow slope parameter (m⁻¹)
        lbdag: Graupel slope parameter (m⁻¹)
        ai: Thermodynamic diffusion coefficient (s·m⁻²)
        cj: Ventilation coefficient
        hli_hcf: Subgrid cloud fraction for ice (0-1)
        hli_hri: Subgrid ice mixing ratio (kg/kg)
        ldcompute: Computation mask
        ldsoft: Soft threshold mode flag
        constants: Dictionary of physical constants
        
    Returns:
        Tuple of (rc_honi_tnd, rv_deps_tnd, ri_aggs_tnd, ri_auts_tnd, rv_depg_tnd)
    """
    # Extract constants
    V_RTMIN = constants["V_RTMIN"]
    C_RTMIN = constants["C_RTMIN"]
    I_RTMIN = constants["I_RTMIN"]
    S_RTMIN = constants["S_RTMIN"]
    G_RTMIN = constants["G_RTMIN"]
    TT = constants["TT"]
    HON = constants["HON"]
    ALPHA3 = constants["ALPHA3"]
    BETA3 = constants["BETA3"]
    O0DEPS = constants["O0DEPS"]
    O1DEPS = constants["O1DEPS"]
    EX0DEPS = constants["EX0DEPS"]
    EX1DEPS = constants["EX1DEPS"]
    FIAGGS = constants["FIAGGS"]
    COLEXIS = constants["COLEXIS"]
    EXIAGGS = constants["EXIAGGS"]
    CEXVT = constants["CEXVT"]
    TIMAUTI = constants["TIMAUTI"]
    TEXAUTI = constants["TEXAUTI"]
    CRIAUTI = constants["CRIAUTI"]
    ACRIAUTI = constants["ACRIAUTI"]
    BCRIAUTI = constants["BCRIAUTI"]
    O0DEPG = constants["O0DEPG"]
    O1DEPG = constants["O1DEPG"]
    EX0DEPG = constants["EX0DEPG"]
    EX1DEPG = constants["EX1DEPG"]
    
    # Initialize outputs
    rc_honi_tnd = jnp.zeros_like(rhodref)
    rv_deps_tnd = jnp.zeros_like(rhodref)
    ri_aggs_tnd = jnp.zeros_like(rhodref)
    ri_auts_tnd = jnp.zeros_like(rhodref)
    rv_depg_tnd = jnp.zeros_like(rhodref)
    
    if not ldsoft:
        # =============================
        # 1. Homogeneous Nucleation (RCHONI)
        # =============================
        # Cloud droplets freeze at very cold temperatures (T < -35°C)
        mask_honi = (t < TT - 35.0) & (rct > C_RTMIN) & ldcompute
        
        rc_honi_tnd = jnp.where(
            mask_honi,
            jnp.minimum(
                1000.0,
                HON * rhodref * rct * jnp.exp(ALPHA3 * (t - TT) - BETA3)
            ),
            0.0
        )
        
        # =============================
        # 2. Vapor Deposition on Snow (RVDEPS)
        # =============================
        # Water vapor deposits directly onto snow crystals
        mask_deps = (rvt > V_RTMIN) & (rst > S_RTMIN) & ldcompute
        
        rv_deps_tnd = jnp.where(
            mask_deps,
            (ssi / (rhodref * ai)) * (
                O0DEPS * jnp.power(lbdas, EX0DEPS) +
                O1DEPS * cj * jnp.power(lbdas, EX1DEPS)
            ),
            0.0
        )
        
        # =============================
        # 3. Ice Aggregation to Snow (RIAGGS)
        # =============================
        # Ice crystals collide and stick to form snow
        mask_aggs = (rit > I_RTMIN) & (rst > S_RTMIN) & ldcompute
        
        ri_aggs_tnd = jnp.where(
            mask_aggs,
            FIAGGS * jnp.exp(COLEXIS * (t - TT)) * rit * 
            jnp.power(lbdas, EXIAGGS) * jnp.power(rhodref, -CEXVT),
            0.0
        )
        
        # =============================
        # 4. Ice Autoconversion to Snow (RIAUTS)
        # =============================
        # Ice crystals grow large enough to become snow
        mask_auts = (hli_hri > I_RTMIN) & ldcompute
        
        # Temperature-dependent threshold
        criauti_tmp = jnp.minimum(
            CRIAUTI,
            jnp.power(10.0, ACRIAUTI * (t - TT) + BCRIAUTI)
        )
        
        ri_auts_tnd = jnp.where(
            mask_auts,
            TIMAUTI * jnp.exp(TEXAUTI * (t - TT)) * 
            jnp.maximum(0.0, hli_hri - criauti_tmp * hli_hcf),
            0.0
        )
        
        # =============================
        # 5. Vapor Deposition on Graupel (RVDEPG)
        # =============================
        # Water vapor deposits onto graupel
        mask_depg = (rvt > V_RTMIN) & (rgt > G_RTMIN) & ldcompute
        
        rv_depg_tnd = jnp.where(
            mask_depg,
            (ssi / (rhodref * ai)) * (
                O0DEPG * jnp.power(lbdag, EX0DEPG) +
                O1DEPG * cj * jnp.power(lbdag, EX1DEPG)
            ),
            0.0
        )
    
    return rc_honi_tnd, rv_deps_tnd, ri_aggs_tnd, ri_auts_tnd, rv_depg_tnd
