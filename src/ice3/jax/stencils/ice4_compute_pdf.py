"""
ICE4 Compute PDF - JAX Implementation

This module computes probability density functions to split clouds into high and low content parts.

Reference:
    PHYEX/src/common/micro/mode_ice4_compute_pdf.F90
"""
import jax
import jax.numpy as jnp
from typing import Tuple


def ice4_compute_pdf(
    rhodref: jnp.ndarray,
    rc_t: jnp.ndarray,
    ri_t: jnp.ndarray,
    cf: jnp.ndarray,
    t: jnp.ndarray,
    sigma_rc: jnp.ndarray,
    ldmicro: jnp.ndarray,
    subg_aucv_rc: int,
    subg_aucv_ri: int,
    subg_pr_pdf: int,
    constants: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, 
           jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute PDF to partition clouds into high and low content regions.
    
    This function uses a PDF-based approach to partition cloud water and ice
    content into high-content (prone to autoconversion) and low-content regions.
    
    Args:
        rhodref: Reference air density (kg/m³)
        rc_t: Cloud droplet mixing ratio (kg/kg)
        ri_t: Ice crystal mixing ratio (kg/kg)
        cf: Total cloud fraction (0-1)
        t: Temperature (K)
        sigma_rc: Standard deviation of cloud droplet mixing ratio (kg/kg)
        ldmicro: Mask for microphysics computation
        subg_aucv_rc: Subgrid scheme for liquid (0=NONE, 1=CLFR, 2=ADJU, 3=PDF)
        subg_aucv_ri: Subgrid scheme for ice (0=NONE, 1=CLFR, 2=ADJU)
        subg_pr_pdf: PDF scheme (0=SIGM)
        constants: Dictionary of physical constants
        
    Returns:
        Tuple of (hlc_hcf, hlc_lcf, hlc_hrc, hlc_lrc,
                  hli_hcf, hli_lcf, hli_hri, hli_lri, rf)
    """
    # Extract constants
    C_RTMIN = constants["C_RTMIN"]
    I_RTMIN = constants["I_RTMIN"]
    CRIAUTC = constants["CRIAUTC"]
    CRIAUTI = constants["CRIAUTI"]
    ACRIAUTI = constants["ACRIAUTI"]
    BCRIAUTI = constants["BCRIAUTI"]
    TT = constants["TT"]
    
    # Initialize outputs
    hlc_hcf = jnp.zeros_like(rhodref)
    hlc_lcf = jnp.zeros_like(rhodref)
    hlc_hrc = jnp.zeros_like(rhodref)
    hlc_lrc = jnp.zeros_like(rhodref)
    hli_hcf = jnp.zeros_like(rhodref)
    hli_lcf = jnp.zeros_like(rhodref)
    hli_hri = jnp.zeros_like(rhodref)
    hli_lri = jnp.zeros_like(rhodref)
    
    # Compute autoconversion thresholds
    rcrautc_tmp = jnp.where(ldmicro, CRIAUTC / rhodref, 0.0)
    criauti_tmp = jnp.where(
        ldmicro,
        jnp.minimum(CRIAUTI, 10 ** (ACRIAUTI * (t - TT) + BCRIAUTI)),
        0.0
    )
    
    # ==========================
    # Liquid water partitioning
    # ==========================
    
    if subg_aucv_rc == 0:  # NONE
        # High content condition
        mask_liq_high = (rc_t > rcrautc_tmp) & ldmicro
        hlc_hcf = jnp.where(mask_liq_high, 1.0, hlc_hcf)
        hlc_lcf = jnp.where(mask_liq_high, 0.0, hlc_lcf)
        hlc_hrc = jnp.where(mask_liq_high, rc_t, hlc_hrc)
        hlc_lrc = jnp.where(mask_liq_high, 0.0, hlc_lrc)
        
        # Low content condition
        mask_liq_low = (rc_t > C_RTMIN) & (~mask_liq_high) & ldmicro
        hlc_hcf = jnp.where(mask_liq_low, 0.0, hlc_hcf)
        hlc_lcf = jnp.where(mask_liq_low, 1.0, hlc_lcf)
        hlc_hrc = jnp.where(mask_liq_low, 0.0, hlc_hrc)
        hlc_lrc = jnp.where(mask_liq_low, rc_t, hlc_lrc)
        
    elif subg_aucv_rc == 1:  # CLFR
        # High content condition
        mask_liq_high = (cf > 0) & (rc_t > rcrautc_tmp * cf) & ldmicro
        hlc_hcf = jnp.where(mask_liq_high, cf, hlc_hcf)
        hlc_lcf = jnp.where(mask_liq_high, 0.0, hlc_lcf)
        hlc_hrc = jnp.where(mask_liq_high, rc_t, hlc_hrc)
        hlc_lrc = jnp.where(mask_liq_high, 0.0, hlc_lrc)
        
        # Low content condition
        mask_liq_low = (cf > 0) & (rc_t > C_RTMIN) & (~mask_liq_high) & ldmicro
        hlc_hcf = jnp.where(mask_liq_low, 0.0, hlc_hcf)
        hlc_lcf = jnp.where(mask_liq_low, cf, hlc_lcf)
        hlc_hrc = jnp.where(mask_liq_low, 0.0, hlc_hrc)
        hlc_lrc = jnp.where(mask_liq_low, rc_t, hlc_lrc)
        
    elif subg_aucv_rc == 2:  # ADJU
        # Adjust based on current partition
        sumrc_tmp = jnp.where(ldmicro, hlc_lrc + hlc_hrc, 0.0)
        mask_adjust = (sumrc_tmp > 0) & ldmicro
        
        hlc_lrc = jnp.where(mask_adjust, hlc_lrc * rc_t / sumrc_tmp, 0.0)
        hlc_hrc = jnp.where(mask_adjust, hlc_hrc * rc_t / sumrc_tmp, 0.0)
        
    elif subg_aucv_rc == 3:  # PDF
        if subg_pr_pdf == 0:  # SIGM
            # Very high content
            mask_very_high = (rc_t > rcrautc_tmp + sigma_rc) & ldmicro
            hlc_hcf = jnp.where(mask_very_high, 1.0, hlc_hcf)
            hlc_lcf = jnp.where(mask_very_high, 0.0, hlc_lcf)
            hlc_hrc = jnp.where(mask_very_high, rc_t, hlc_hrc)
            hlc_lrc = jnp.where(mask_very_high, 0.0, hlc_lrc)
            
            # Intermediate content (PDF split)
            mask_intermediate = (
                (rc_t > (rcrautc_tmp - sigma_rc)) & 
                (rc_t <= (rcrautc_tmp + sigma_rc)) & 
                (~mask_very_high) & 
                ldmicro
            )
            hcf_inter = (rc_t + sigma_rc - rcrautc_tmp) / (2.0 * sigma_rc)
            hrc_inter = (
                (rc_t + sigma_rc - rcrautc_tmp) * 
                (rc_t + sigma_rc + rcrautc_tmp) / 
                (4.0 * sigma_rc)
            )
            
            hlc_hcf = jnp.where(mask_intermediate, hcf_inter, hlc_hcf)
            hlc_lcf = jnp.where(mask_intermediate, jnp.maximum(0.0, cf - hcf_inter), hlc_lcf)
            hlc_hrc = jnp.where(mask_intermediate, hrc_inter, hlc_hrc)
            hlc_lrc = jnp.where(mask_intermediate, jnp.maximum(0.0, rc_t - hrc_inter), hlc_lrc)
            
            # Low content
            mask_low = (
                (rc_t > C_RTMIN) & 
                (cf > 0) & 
                (~mask_very_high) & 
                (~mask_intermediate) & 
                ldmicro
            )
            hlc_hcf = jnp.where(mask_low, 0.0, hlc_hcf)
            hlc_lcf = jnp.where(mask_low, cf, hlc_lcf)
            hlc_hrc = jnp.where(mask_low, 0.0, hlc_hrc)
            hlc_lrc = jnp.where(mask_low, rc_t, hlc_lrc)
    
    # ==========================
    # Ice partitioning
    # ==========================
    
    if subg_aucv_ri == 0:  # NONE
        # High content condition
        mask_ice_high = (ri_t > criauti_tmp) & ldmicro
        hli_hcf = jnp.where(mask_ice_high, 1.0, hli_hcf)
        hli_lcf = jnp.where(mask_ice_high, 0.0, hli_lcf)
        hli_hri = jnp.where(mask_ice_high, ri_t, hli_hri)
        hli_lri = jnp.where(mask_ice_high, 0.0, hli_lri)
        
        # Low content condition
        mask_ice_low = (ri_t > I_RTMIN) & (~mask_ice_high) & ldmicro
        hli_hcf = jnp.where(mask_ice_low, 0.0, hli_hcf)
        hli_lcf = jnp.where(mask_ice_low, 1.0, hli_lcf)
        hli_hri = jnp.where(mask_ice_low, 0.0, hli_hri)
        hli_lri = jnp.where(mask_ice_low, ri_t, hli_lri)
        
    elif subg_aucv_ri == 1:  # CLFR
        # High content condition
        mask_ice_high = (cf > 0) & (ri_t > criauti_tmp * cf) & ldmicro
        hli_hcf = jnp.where(mask_ice_high, cf, hli_hcf)
        hli_lcf = jnp.where(mask_ice_high, 0.0, hli_lcf)
        hli_hri = jnp.where(mask_ice_high, ri_t, hli_hri)
        hli_lri = jnp.where(mask_ice_high, 0.0, hli_lri)
        
        # Low content condition
        mask_ice_low = (cf > 0) & (ri_t > I_RTMIN) & (~mask_ice_high) & ldmicro
        hli_hcf = jnp.where(mask_ice_low, 0.0, hli_hcf)
        hli_lcf = jnp.where(mask_ice_low, cf, hli_lcf)
        hli_hri = jnp.where(mask_ice_low, 0.0, hli_hri)
        hli_lri = jnp.where(mask_ice_low, ri_t, hli_lri)
        
    elif subg_aucv_ri == 2:  # ADJU
        # Adjust based on current partition
        sumri_tmp = jnp.where(ldmicro, hli_lri + hli_hri, 0.0)
        mask_adjust = (sumri_tmp > 0) & ldmicro
        
        hli_lri = jnp.where(mask_adjust, hli_lri * ri_t / sumri_tmp, 0.0)
        hli_hri = jnp.where(mask_adjust, hli_hri * ri_t / sumri_tmp, 0.0)
    
    # Compute precipitation fraction
    rf = jnp.where(ldmicro, jnp.maximum(hlc_hcf, hli_hcf), 0.0)
    
    return hlc_hcf, hlc_lcf, hlc_hrc, hlc_lrc, hli_hcf, hli_lcf, hli_hri, hli_lri, rf
