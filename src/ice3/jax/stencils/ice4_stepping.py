"""
ICE4 Stepping - JAX Implementation

This module implements adaptive time stepping for ICE4 microphysics.

Reference:
    PHYEX/src/common/micro/mode_ice4_stepping.F90
"""
import jax
import jax.numpy as jnp
from typing import Tuple


def compute_latent_heat_factors(
    rv_t: jnp.ndarray,
    rc_t: jnp.ndarray,
    rr_t: jnp.ndarray,
    ri_t: jnp.ndarray,
    rs_t: jnp.ndarray,
    rg_t: jnp.ndarray,
    exn: jnp.ndarray,
    t: jnp.ndarray,
    constants: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute latent heat factors for microphysics.
    
    Args:
        rv_t: Water vapor mixing ratio (kg/kg)
        rc_t: Cloud mixing ratio (kg/kg)
        rr_t: Rain mixing ratio (kg/kg)
        ri_t: Ice mixing ratio (kg/kg)
        rs_t: Snow mixing ratio (kg/kg)
        rg_t: Graupel mixing ratio (kg/kg)
        exn: Exner function
        t: Temperature (K)
        constants: Physical constants
        
    Returns:
        Tuple of (lv_fact, ls_fact)
    """
    CPD = constants["CPD"]
    CPV = constants["CPV"]
    CL = constants["CL"]
    CI = constants["CI"]
    LVTT = constants["LVTT"]
    LSTT = constants["LSTT"]
    
    # Compute specific heat capacity (simplified)
    cph = CPD + CPV * rv_t + CL * (rc_t + rr_t) + CI * (ri_t + rs_t + rg_t)
    
    # Latent heat factors
    lv_fact = LVTT / (cph * exn)
    ls_fact = LSTT / (cph * exn)
    
    return lv_fact, ls_fact


def mixing_ratio_step_limiter(
    r_tnd: jnp.ndarray,
    r_b: jnp.ndarray,
    r_t: jnp.ndarray,
    delta_t: jnp.ndarray,
    r_min: float,
    tiny: float,
) -> jnp.ndarray:
    """
    Limit time step based on species depletion.
    
    Args:
        r_tnd: Tendency of mixing ratio (kg/kg/s)
        r_b: Instantaneous change (kg/kg)
        r_t: Current mixing ratio (kg/kg)
        delta_t: Current time step (s)
        r_min: Minimum threshold (kg/kg)
        tiny: Small value for comparisons
        
    Returns:
        Limited time step (s)
    """
    # Calculate time to reach minimum threshold
    # r_t + r_tnd * dt + r_b = r_min
    # dt = (r_min - r_t - r_b) / r_tnd
    
    # Only compute where tendency is significant
    mask_significant = jnp.abs(r_tnd) > tiny
    
    time_threshold = jnp.where(
        mask_significant,
        (r_min - r_t - r_b) / r_tnd,
        -1.0  # Negative value means no limit
    )
    
    # Only limit if threshold is positive and conditions met
    mask_limit = (
        (time_threshold > 0.0) & 
        ((r_t > r_min) | (r_tnd > 0.0))
    )
    
    delta_t_new = jnp.where(
        mask_limit,
        jnp.minimum(delta_t, time_threshold),
        delta_t
    )
    
    return delta_t_new


def ice4_step_limiter(
    exn: jnp.ndarray,
    th_t: jnp.ndarray,
    theta_tnd: jnp.ndarray,
    theta_b: jnp.ndarray,
    rc_t: jnp.ndarray,
    rr_t: jnp.ndarray,
    ri_t: jnp.ndarray,
    rs_t: jnp.ndarray,
    rg_t: jnp.ndarray,
    rc_tnd: jnp.ndarray,
    rr_tnd: jnp.ndarray,
    ri_tnd: jnp.ndarray,
    rs_tnd: jnp.ndarray,
    rg_tnd: jnp.ndarray,
    rc_b: jnp.ndarray,
    rr_b: jnp.ndarray,
    ri_b: jnp.ndarray,
    rs_b: jnp.ndarray,
    rg_b: jnp.ndarray,
    t_micro: jnp.ndarray,
    ldcompute: jnp.ndarray,
    tstep: float,
    constants: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute adaptive time step with limiters.
    
    This function prevents numerical instabilities by limiting the time
    step based on species depletion and temperature crossing 0°C.
    
    Args:
        exn: Exner function
        th_t: Potential temperature (K)
        theta_tnd: Temperature tendency (K/s)
        theta_b: Instantaneous temperature change (K)
        rc_t, rr_t, ri_t, rs_t, rg_t: Mixing ratios (kg/kg)
        rc_tnd, rr_tnd, ri_tnd, rs_tnd, rg_tnd: Tendencies (kg/kg/s)
        rc_b, rr_b, ri_b, rs_b, rg_b: Instantaneous changes (kg/kg)
        t_micro: Elapsed time (s)
        ldcompute: Computation mask
        tstep: Total physics time step (s)
        constants: Physical constants
        
    Returns:
        Tuple of (delta_t_micro, ldcompute_new)
    """
    TT = constants["TT"]
    C_RTMIN = constants["C_RTMIN"]
    R_RTMIN = constants["R_RTMIN"]
    I_RTMIN = constants["I_RTMIN"]
    S_RTMIN = constants["S_RTMIN"]
    G_RTMIN = constants["G_RTMIN"]
    MNH_TINY = constants.get("MNH_TINY", 1e-20)
    
    # Initialize with remaining time
    delta_t_micro = jnp.where(ldcompute, tstep - t_micro, 0.0)
    
    # =============================
    # 1. Temperature Crossing 0°C
    # =============================
    th_tt = TT / exn  # Threshold potential temperature
    
    # Check if temperature crosses 0°C
    crosses_freezing = (th_t - th_tt) * (th_t + theta_b - th_tt) < 0.0
    delta_t_micro = jnp.where(crosses_freezing, 0.0, delta_t_micro)
    
    # If tendency would cross threshold, limit time step
    mask_theta = jnp.abs(theta_tnd) > MNH_TINY
    time_to_freezing = jnp.where(
        mask_theta,
        (th_tt - theta_b - th_t) / theta_tnd,
        -1.0
    )
    
    mask_limit_theta = (time_to_freezing > 0.0) & mask_theta
    delta_t_micro = jnp.where(
        mask_limit_theta,
        jnp.minimum(delta_t_micro, time_to_freezing),
        delta_t_micro
    )
    
    # =============================
    # 2. Species Depletion Limiters
    # =============================
    
    # Cloud
    delta_t_micro = mixing_ratio_step_limiter(
        rc_tnd, rc_b, rc_t, delta_t_micro, C_RTMIN, MNH_TINY
    )
    
    # Rain
    delta_t_micro = mixing_ratio_step_limiter(
        rr_tnd, rr_b, rr_t, delta_t_micro, R_RTMIN, MNH_TINY
    )
    
    # Ice
    delta_t_micro = mixing_ratio_step_limiter(
        ri_tnd, ri_b, ri_t, delta_t_micro, I_RTMIN, MNH_TINY
    )
    
    # Snow
    delta_t_micro = mixing_ratio_step_limiter(
        rs_tnd, rs_b, rs_t, delta_t_micro, S_RTMIN, MNH_TINY
    )
    
    # Graupel
    delta_t_micro = mixing_ratio_step_limiter(
        rg_tnd, rg_b, rg_t, delta_t_micro, G_RTMIN, MNH_TINY
    )
    
    # =============================
    # 3. Stop at End of Time Step
    # =============================
    ldcompute_new = jnp.where(
        t_micro + delta_t_micro >= tstep,
        False,
        ldcompute
    )
    
    return delta_t_micro, ldcompute_new


def state_update(
    th_t: jnp.ndarray,
    theta_tnd: jnp.ndarray,
    theta_b: jnp.ndarray,
    rc_t: jnp.ndarray,
    rr_t: jnp.ndarray,
    ri_t: jnp.ndarray,
    rs_t: jnp.ndarray,
    rg_t: jnp.ndarray,
    rc_tnd: jnp.ndarray,
    rr_tnd: jnp.ndarray,
    ri_tnd: jnp.ndarray,
    rs_tnd: jnp.ndarray,
    rg_tnd: jnp.ndarray,
    rc_b: jnp.ndarray,
    rr_b: jnp.ndarray,
    ri_b: jnp.ndarray,
    rs_b: jnp.ndarray,
    rg_b: jnp.ndarray,
    ci_t: jnp.ndarray,
    delta_t_micro: jnp.ndarray,
    ldmicro: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, 
           jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Update state variables after time step.
    
    Uses explicit forward Euler integration:
    x(t + Δt) = x(t) + (dx/dt) × Δt + Δx_instant
    
    Args:
        th_t: Potential temperature (K)
        theta_tnd: Temperature tendency (K/s)
        theta_b: Instantaneous temperature change (K)
        rc_t, rr_t, ri_t, rs_t, rg_t: Mixing ratios (kg/kg)
        rc_tnd, rr_tnd, ri_tnd, rs_tnd, rg_tnd: Tendencies (kg/kg/s)
        rc_b, rr_b, ri_b, rs_b, rg_b: Instantaneous changes (kg/kg)
        ci_t: Ice crystal concentration (#/kg)
        delta_t_micro: Time step (s)
        ldmicro: Microphysics mask
        
    Returns:
        Tuple of updated (th_t, rc_t, rr_t, ri_t, rs_t, rg_t, ci_t)
    """
    # Update state variables
    th_t_new = th_t + theta_tnd * delta_t_micro + theta_b
    rc_t_new = rc_t + rc_tnd * delta_t_micro + rc_b
    rr_t_new = rr_t + rr_tnd * delta_t_micro + rr_b
    ri_t_new = ri_t + ri_tnd * delta_t_micro + ri_b
    rs_t_new = rs_t + rs_tnd * delta_t_micro + rs_b
    rg_t_new = rg_t + rg_tnd * delta_t_micro + rg_b
    
    # Handle ice crystal concentration when ice depletes
    ci_t_new = jnp.where(
        (ri_t_new <= 0.0) & ldmicro,
        0.0,
        ci_t
    )
    
    return th_t_new, rc_t_new, rr_t_new, ri_t_new, rs_t_new, rg_t_new, ci_t_new


def external_tendencies_update(
    th_t: jnp.ndarray,
    rc_t: jnp.ndarray,
    rr_t: jnp.ndarray,
    ri_t: jnp.ndarray,
    rs_t: jnp.ndarray,
    rg_t: jnp.ndarray,
    theta_tnd_ext: jnp.ndarray,
    rc_tnd_ext: jnp.ndarray,
    rr_tnd_ext: jnp.ndarray,
    ri_tnd_ext: jnp.ndarray,
    rs_tnd_ext: jnp.ndarray,
    rg_tnd_ext: jnp.ndarray,
    ldmicro: jnp.ndarray,
    dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Apply external tendencies (from turbulence, radiation, etc.).
    
    Args:
        th_t: Potential temperature (K)
        rc_t, rr_t, ri_t, rs_t, rg_t: Mixing ratios (kg/kg)
        theta_tnd_ext: External temperature tendency (K/s)
        rc_tnd_ext, rr_tnd_ext, ri_tnd_ext, rs_tnd_ext, rg_tnd_ext: External tendencies (kg/kg/s)
        ldmicro: Microphysics mask
        dt: Time step (s)
        
    Returns:
        Tuple of updated (th_t, rc_t, rr_t, ri_t, rs_t, rg_t)
    """
    # Apply external tendencies where microphysics is active
    th_t_new = jnp.where(ldmicro, th_t - theta_tnd_ext * dt, th_t)
    rc_t_new = jnp.where(ldmicro, rc_t - rc_tnd_ext * dt, rc_t)
    rr_t_new = jnp.where(ldmicro, rr_t - rr_tnd_ext * dt, rr_t)
    ri_t_new = jnp.where(ldmicro, ri_t - ri_tnd_ext * dt, ri_t)
    rs_t_new = jnp.where(ldmicro, rs_t - rs_tnd_ext * dt, rs_t)
    rg_t_new = jnp.where(ldmicro, rg_t - rg_tnd_ext * dt, rg_t)
    
    return th_t_new, rc_t_new, rr_t_new, ri_t_new, rs_t_new, rg_t_new
