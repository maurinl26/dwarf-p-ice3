"""
Sedimentation - JAX Implementation

This module computes gravitational sedimentation using the statistical method.

Reference:
    PHYEX/src/common/micro/mode_ice4_sedimentation_stat.F90
"""
import jax
import jax.numpy as jnp
from typing import Tuple


def compute_lbc(sea: jnp.ndarray, lbc_land: float, lbc_sea: float) -> jnp.ndarray:
    """Compute LBC parameter weighted by sea fraction"""
    return (1.0 - sea) * lbc_land + sea * lbc_sea


def compute_fsedc(sea: jnp.ndarray, fsedc_land: float, fsedc_sea: float) -> jnp.ndarray:
    """Compute FSEDC parameter weighted by sea fraction"""
    return (1.0 - sea) * fsedc_land + sea * fsedc_sea


def compute_conc3d(
    town: jnp.ndarray, 
    sea: jnp.ndarray, 
    conc_land: float, 
    conc_sea: float, 
    conc_urban: float
) -> jnp.ndarray:
    """Compute 3D concentration based on town and sea fractions"""
    return (1.0 - town) * ((1.0 - sea) * conc_land + sea * conc_sea) + town * conc_urban


def other_species_velocity(
    fsed: float,
    exsed: float,
    content: jnp.ndarray,
    rhodref: jnp.ndarray,
    cexvt: float,
) -> jnp.ndarray:
    """Compute terminal velocity for rain, snow, graupel"""
    return fsed * jnp.power(content, exsed - 1.0) * jnp.power(rhodref, exsed - cexvt - 1.0)


def pristine_ice_velocity(
    content: jnp.ndarray, 
    rhodref: jnp.ndarray,
    fsedi: float,
    excsedi: float,
    cexvt: float,
) -> jnp.ndarray:
    """Compute terminal velocity for pristine ice"""
    dmax = jnp.maximum(0.05e6, -0.15319e6 - 0.021454e6 * jnp.log(rhodref * content))
    return fsedi * jnp.power(rhodref, -cexvt) * jnp.power(dmax, excsedi)


def sedimentation_stat(
    rhodref: jnp.ndarray,
    dzz: jnp.ndarray,
    pabs_t: jnp.ndarray,
    th_t: jnp.ndarray,
    rcs: jnp.ndarray,
    rrs: jnp.ndarray,
    ris: jnp.ndarray,
    rss: jnp.ndarray,
    rgs: jnp.ndarray,
    sea: jnp.ndarray,
    town: jnp.ndarray,
    constants: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
           jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
           jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute gravitational sedimentation using statistical method.
    
    This function implements the statistical sedimentation scheme for ICE4
    microphysics, computing vertical transport of hydrometeors due to
    gravitational settling.
    
    Args:
        rhodref: Reference air density (kg/m³)
        dzz: Vertical grid spacing (m)
        pabs_t: Absolute pressure (Pa)
        th_t: Potential temperature (K)
        rcs: Cloud droplet tendency (kg/kg/s)
        rrs: Rain tendency (kg/kg/s)
        ris: Ice crystal tendency (kg/kg/s)
        rss: Snow tendency (kg/kg/s)
        rgs: Graupel tendency (kg/kg/s)
        sea: Sea fraction (0-1), 2D field
        town: Urban fraction (0-1), 2D field
        constants: Dictionary of physical constants
        
    Returns:
        Tuple of (rcs_new, rrs_new, ris_new, rss_new, rgs_new,
                  fpr_c, fpr_r, fpr_i, fpr_s, fpr_g,
                  inprc, inprr, inpri, inprs, inprg)
    """
    # Extract constants
    TSTEP = constants["TSTEP"]
    C_RTMIN = constants["C_RTMIN"]
    R_RTMIN = constants["R_RTMIN"]
    I_RTMIN = constants["I_RTMIN"]
    S_RTMIN = constants["S_RTMIN"]
    G_RTMIN = constants["G_RTMIN"]
    CEXVT = constants["CEXVT"]
    CC = constants["CC"]
    DC = constants["DC"]
    LBEXC = constants["LBEXC"]
    LBC_LAND = constants["LBC_LAND"]
    LBC_SEA = constants["LBC_SEA"]
    FSEDC_LAND = constants["FSEDC_LAND"]
    FSEDC_SEA = constants["FSEDC_SEA"]
    CONC_LAND = constants["CONC_LAND"]
    CONC_SEA = constants["CONC_SEA"]
    CONC_URBAN = constants["CONC_URBAN"]
    FSEDR = constants["FSEDR"]
    EXSEDR = constants["EXSEDR"]
    FSEDI = constants["FSEDI"]
    EXCSEDI = constants["EXCSEDI"]
    FSEDS = constants["FSEDS"]
    EXSEDS = constants["EXSEDS"]
    FSEDG = constants["FSEDG"]
    EXSEDG = constants["EXSEDG"]
    RHOLW = constants["RHOLW"]
    
    # Get shape (Assuming layout is (ngpblks, nproma, nflevg))
    ng = rhodref.shape[0]     # Number of blocks
    nproma = rhodref.shape[1] # Number of points
    nz = rhodref.shape[2]     # Number of vertical levels
    
    # Convert tendencies to mixing ratios
    rc_t = rcs * TSTEP
    rr_t = rrs * TSTEP
    ri_t = ris * TSTEP
    rs_t = rss * TSTEP
    rg_t = rgs * TSTEP
    
    # Initialize fluxes (one extra level for top boundary)
    fpr_c = jnp.zeros_like(rhodref)
    fpr_r = jnp.zeros_like(rhodref)
    fpr_i = jnp.zeros_like(rhodref)
    fpr_s = jnp.zeros_like(rhodref)
    fpr_g = jnp.zeros_like(rhodref)
    
    # Precompute common terms
    tstep_rho_dz = TSTEP / (rhodref * dzz)
    zinvtstep = 1.0 / TSTEP
    zinvtstep = 1.0 / TSTEP
    
    # Compute surface-dependent parameters (broadcast 2D/3D to 3D)
    _lbc = compute_lbc(sea, LBC_LAND, LBC_SEA)
    _fsedc = compute_fsedc(sea, FSEDC_LAND, FSEDC_SEA)
    _conc3d = compute_conc3d(town, sea, CONC_LAND, CONC_SEA, CONC_URBAN)
    _ray = 1.0  # Simplified
    
    # Backward sweep from top to bottom (k = nz-1 down to 0)
    for k in range(nz - 1, -1, -1):
        # Get flux from level above (k+1)
        if k < nz - 1:
            fpr_c_above = fpr_c[:, :, k + 1]
            fpr_r_above = fpr_r[:, :, k + 1]
            fpr_i_above = fpr_i[:, :, k + 1]
            fpr_s_above = fpr_s[:, :, k + 1]
            fpr_g_above = fpr_g[:, :, k + 1]
            
            qp_c = fpr_c_above * tstep_rho_dz[:, :, k]
            qp_r = fpr_r_above * tstep_rho_dz[:, :, k]
            qp_i = fpr_i_above * tstep_rho_dz[:, :, k]
            qp_s = fpr_s_above * tstep_rho_dz[:, :, k]
            qp_g = fpr_g_above * tstep_rho_dz[:, :, k]
        else:
            fpr_c_above = 0.0
            fpr_r_above = 0.0
            fpr_i_above = 0.0
            fpr_s_above = 0.0
            fpr_g_above = 0.0
            
            qp_c = 0.0
            qp_r = 0.0
            qp_i = 0.0
            qp_s = 0.0
            qp_g = 0.0
        
        # Get level-specific air properties
        curr_pabs_t = pabs_t[:, :, k]
        curr_th_t = th_t[:, :, k]
        curr_rhodref = rhodref[:, :, k]
        curr_dzz = dzz[:, :, k]
        
        # Surface parameters are already 2D (ng, nproma), no need to squeeze
        # Actually our compute functions return what we pass. 
        # If we passed 3D sea, they return 3D.
        # If they come in as (ngpblks, nproma, 1), we select the first level
        def get_surf(x):
            return x[:, :, 0] if x.ndim == 3 else x

        lbc_2d = get_surf(_lbc)
        fsedc_2d = get_surf(_fsedc)
        conc3d_2d = get_surf(_conc3d)

        # Cloud droplets
        curr_rc_t = rc_t[:, :, k]
        mask_rc = curr_rc_t > C_RTMIN
        wlbda = 6.6e-8 * (101325.0 / curr_pabs_t) * (curr_th_t / 293.15)
        wlbdc = jnp.power(lbc_2d * conc3d_2d / (curr_rhodref * curr_rc_t), LBEXC)
        cc = CC * (1.0 + 1.26 * wlbda * wlbdc / _ray)
        wsedw1 = jnp.where(mask_rc, jnp.power(curr_rhodref, -CEXVT) * jnp.power(wlbdc, -DC) * cc * fsedc_2d, 0.0)
        
        mask_qp_c = qp_c > C_RTMIN
        wlbdc_qp = jnp.power(lbc_2d * conc3d_2d / (curr_rhodref * qp_c), LBEXC)
        cc_qp = CC * (1.0 + 1.26 * wlbda * wlbdc_qp / _ray)
        wsedw2 = jnp.where(mask_qp_c, jnp.power(curr_rhodref, -CEXVT) * jnp.power(wlbdc_qp, -DC) * cc_qp * fsedc_2d, 0.0)
        
        fpr_c_local = jnp.minimum(
            curr_rhodref * curr_dzz * curr_rc_t * zinvtstep,
            wsedw1 * curr_rhodref * curr_rc_t
        )
        
        fpr_c_from_above = jnp.where(wsedw2 != 0.0, jnp.maximum(0.0, 1.0 - curr_dzz / (TSTEP * wsedw2)) * fpr_c_above, 0.0)
        
        fpr_c = fpr_c.at[:, :, k].set(fpr_c_local + fpr_c_from_above)
        
        # Rain
        curr_rr_t = rr_t[:, :, k]
        mask_rr = curr_rr_t > R_RTMIN
        wsedw1 = jnp.where(mask_rr, other_species_velocity(FSEDR, EXSEDR, curr_rr_t, curr_rhodref, CEXVT), 0.0)
        
        mask_qp_r = qp_r > R_RTMIN
        wsedw2 = jnp.where(mask_qp_r, other_species_velocity(FSEDR, EXSEDR, qp_r, curr_rhodref, CEXVT), 0.0)
        
        fpr_r_local = jnp.minimum(
            curr_rhodref * curr_dzz * curr_rr_t * zinvtstep,
            wsedw1 * curr_rhodref * curr_rr_t
        )
        
        fpr_r_from_above = jnp.where(wsedw2 != 0.0, jnp.maximum(0.0, 1.0 - curr_dzz / (TSTEP * wsedw2)) * fpr_r_above, 0.0)
        
        fpr_r = fpr_r.at[:, :, k].set(fpr_r_local + fpr_r_from_above)
        
        # Pristine ice
        curr_ri_t = ri_t[:, :, k]
        mask_ri = curr_ri_t > jnp.maximum(I_RTMIN, 1.0e-7)
        wsedw1 = jnp.where(mask_ri, pristine_ice_velocity(curr_ri_t, curr_rhodref, FSEDI, EXCSEDI, CEXVT), 0.0)
        
        mask_qp_i = qp_i > jnp.maximum(I_RTMIN, 1.0e-7)
        wsedw2 = jnp.where(mask_qp_i, pristine_ice_velocity(qp_i, curr_rhodref, FSEDI, EXCSEDI, CEXVT), 0.0)
        
        fpr_i_local = jnp.minimum(
            curr_rhodref * curr_dzz * curr_ri_t * zinvtstep,
            wsedw1 * curr_rhodref * curr_ri_t
        )
        
        fpr_i_from_above = jnp.where(wsedw2 != 0.0, jnp.maximum(0.0, 1.0 - curr_dzz / (TSTEP * wsedw2)) * fpr_i_above, 0.0)
        
        fpr_i = fpr_i.at[:, :, k].set(fpr_i_local + fpr_i_from_above)
        
        # Snow
        curr_rs_t = rs_t[:, :, k]
        mask_rs = curr_rs_t > S_RTMIN
        wsedw1 = jnp.where(mask_rs, other_species_velocity(FSEDS, EXSEDS, curr_rs_t, curr_rhodref, CEXVT), 0.0)
        
        mask_qp_s = qp_s > S_RTMIN
        wsedw2 = jnp.where(mask_qp_s, other_species_velocity(FSEDS, EXSEDS, qp_s, curr_rhodref, CEXVT), 0.0)
        
        fpr_s_local = jnp.minimum(
            curr_rhodref * curr_dzz * curr_rs_t * zinvtstep,
            wsedw1 * curr_rhodref * curr_rs_t
        )
        
        fpr_s_from_above = jnp.where(wsedw2 != 0.0, jnp.maximum(0.0, 1.0 - curr_dzz / (TSTEP * wsedw2)) * fpr_s_above, 0.0)
        
        fpr_s = fpr_s.at[:, :, k].set(fpr_s_local + fpr_s_from_above)
        
        # Graupel
        curr_rg_t = rg_t[:, :, k]
        mask_rg = curr_rg_t > G_RTMIN
        wsedw1 = jnp.where(mask_rg, other_species_velocity(FSEDG, EXSEDG, curr_rg_t, curr_rhodref, CEXVT), 0.0)
        
        mask_qp_g = qp_g > G_RTMIN
        wsedw2 = jnp.where(mask_qp_g, other_species_velocity(FSEDG, EXSEDG, qp_g, curr_rhodref, CEXVT), 0.0)
        
        fpr_g_local = jnp.minimum(
            curr_rhodref * curr_dzz * curr_rg_t * zinvtstep,
            wsedw1 * curr_rhodref * curr_rg_t
        )
        
        fpr_g_from_above = jnp.where(wsedw2 != 0.0, jnp.maximum(0.0, 1.0 - curr_dzz / (TSTEP * wsedw2)) * fpr_g_above, 0.0)
        
        fpr_g = fpr_g.at[:, :, k].set(fpr_g_local + fpr_g_from_above)
        
        fpr_i = fpr_i.at[:, :, k].set(fpr_i_local + fpr_i_from_above)
        
        # Snow
        mask_rs = curr_rs_t > S_RTMIN
        wsedw1 = jnp.where(mask_rs, other_species_velocity(FSEDS, EXSEDS, curr_rs_t, curr_rhodref, CEXVT), 0.0)
        
        mask_qp_s = qp_s > S_RTMIN
        wsedw2 = jnp.where(mask_qp_s, other_species_velocity(FSEDS, EXSEDS, qp_s, curr_rhodref, CEXVT), 0.0)
        
        fpr_s_local = jnp.minimum(
            curr_rhodref * curr_dzz * curr_rs_t * zinvtstep,
            wsedw1 * curr_rhodref * curr_rs_t
        )
        
        fpr_s_from_above = jnp.where(wsedw2 != 0.0, jnp.maximum(0.0, 1.0 - curr_dzz / (TSTEP * wsedw2)) * fpr_s_above, 0.0)
        
        fpr_s = fpr_s.at[:, :, k].set(fpr_s_local + fpr_s_from_above)
        
        # Graupel
        mask_rg = curr_rg_t > G_RTMIN
        wsedw1 = jnp.where(mask_rg, other_species_velocity(FSEDG, EXSEDG, curr_rg_t, curr_rhodref, CEXVT), 0.0)
        
        mask_qp_g = qp_g > G_RTMIN
        wsedw2 = jnp.where(mask_qp_g, other_species_velocity(FSEDG, EXSEDG, qp_g, curr_rhodref, CEXVT), 0.0)
        
        fpr_g_local = jnp.minimum(
            curr_rhodref * curr_dzz * curr_rg_t * zinvtstep,
            wsedw1 * curr_rhodref * curr_rg_t
        )
        
        fpr_g_from_above = jnp.where(wsedw2 != 0.0, jnp.maximum(0.0, 1.0 - curr_dzz / (TSTEP * wsedw2)) * fpr_g_above, 0.0)
        
        fpr_g = fpr_g.at[:, :, k].set(fpr_g_local + fpr_g_from_above)
    
    # Compute flux divergence and update tendencies
    # Pad fluxes with zeros at top (level nz)
    fpr_c_pad = jnp.pad(fpr_c, ((0, 0), (0, 0), (0, 1)), constant_values=0.0)
    fpr_r_pad = jnp.pad(fpr_r, ((0, 0), (0, 0), (0, 1)), constant_values=0.0)
    fpr_i_pad = jnp.pad(fpr_i, ((0, 0), (0, 0), (0, 1)), constant_values=0.0)
    fpr_s_pad = jnp.pad(fpr_s, ((0, 0), (0, 0), (0, 1)), constant_values=0.0)
    fpr_g_pad = jnp.pad(fpr_g, ((0, 0), (0, 0), (0, 1)), constant_values=0.0)
    
    # Flux divergence: (flux_in - flux_out) / (rho * dz * dt)
    # flux_in is at index k+1, flux_out is at index k
    rcs_new = rcs + (fpr_c_pad[:, :, 1:] - fpr_c_pad[:, :, :-1]) / (rhodref * dzz * TSTEP)
    rrs_new = rrs + (fpr_r_pad[:, :, 1:] - fpr_r_pad[:, :, :-1]) / (rhodref * dzz * TSTEP)
    ris_new = ris + (fpr_i_pad[:, :, 1:] - fpr_i_pad[:, :, :-1]) / (rhodref * dzz * TSTEP)
    rss_new = rss + (fpr_s_pad[:, :, 1:] - fpr_s_pad[:, :, :-1]) / (rhodref * dzz * TSTEP)
    rgs_new = rgs + (fpr_g_pad[:, :, 1:] - fpr_g_pad[:, :, :-1]) / (rhodref * dzz * TSTEP)
    
    # Surface precipitation (instantaneous flux at ground level, index 0)
    inprc = fpr_c[:, :, 0] / RHOLW
    inprr = fpr_r[:, :, 0] / RHOLW
    inpri = fpr_i[:, :, 0] / RHOLW
    inprs = fpr_s[:, :, 0] / RHOLW
    inprg = fpr_g[:, :, 0] / RHOLW
    
    return (rcs_new, rrs_new, ris_new, rss_new, rgs_new,
            fpr_c, fpr_r, fpr_i, fpr_s, fpr_g,
            inprc, inprr, inpri, inprs, inprg)
