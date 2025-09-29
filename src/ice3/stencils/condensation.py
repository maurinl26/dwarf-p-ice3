"""
Condensation module - GT4Py translation of condensation.F90.

This module implements the condensation scheme for mixed-phase clouds,
including subgrid condensation and all-or-nothing schemes.
"""

import numpy as np
import gt4py.next as gtx
from gt4py.next import Field, float64, int32
from typing import Tuple, Optional

from ice3.common.constants import CST
from ice3.common.dimensions import DimPhyex
from ice3.utils.next_tools import (
    compute_saturation_vapor_pressure_liquid,
    compute_saturation_vapor_pressure_ice,
    compute_latent_heat_vaporization,
    compute_latent_heat_sublimation,
    compute_specific_heat_moist_air,
    compute_saturation_mixing_ratio,
    compute_ice_fraction_simple,
    gaussian_cdf,
    gaussian_pdf,
    safe_divide,
    limit_to_range
)


@gtx.field_operator
def compute_condensation_coefficients(
    temperature: Field[[I, J, K], float64],
    pressure: Field[[I, J, K], float64],
    qsl: Field[[I, J, K], float64],
    lvs: Field[[I, J, K], float64],
    cph: Field[[I, J, K], float64]
) -> Tuple[
    Field[[I, J, K], float64],  # ah
    Field[[I, J, K], float64],  # a
    Field[[I, J, K], float64]   # b
]:
    """
    Compute condensation coefficients a, b, and ah.
    
    These coefficients are used in the statistical condensation scheme.
    """
    # Compute ah coefficient
    ah = lvs * qsl / (CST.XRV * temperature * temperature) * (CST.XRV * qsl / CST.XRD + 1.0)
    
    # Compute a and b coefficients
    a = 1.0 / (1.0 + lvs / cph * ah)
    b = ah * a
    
    return ah, a, b


@gtx.field_operator
def compute_sbar(
    rt: Field[[I, J, K], float64],
    qsl: Field[[I, J, K], float64],
    ah: Field[[I, J, K], float64],
    lvs: Field[[I, J, K], float64],
    cph: Field[[I, J, K], float64],
    rc_in: Field[[I, J, K], float64],
    ri_in: Field[[I, J, K], float64],
    a: Field[[I, J, K], float64],
    zprifact: Field[[I, J, K], float64]
) -> Field[[I, J, K], float64]:
    """
    Compute normalized saturation deficit (sbar).
    """
    return a * (rt - qsl + ah * lvs * (rc_in + ri_in * zprifact) / cph)


@gtx.field_operator
def compute_sigma_turbulent(
    ll: Field[[I, J, K], float64],
    dzz: Field[[I, J, K], float64],
    a: Field[[I, J, K], float64],
    b: Field[[I, J, K], float64],
    drw: Field[[I, J, K], float64],
    dtl: Field[[I, J, K], float64],
    sig_conv: Field[[I, J, K], float64]
) -> Field[[I, J, K], float64]:
    """
    Compute turbulent part of sigma_s using first-order closure.
    """
    csigma = 0.2
    
    # Turbulent variance
    sigma_turb_sq = (csigma * csigma * ll * ll / (dzz * dzz) * 
                     (a * a * drw * drw - 2.0 * a * b * drw * dtl + b * b * dtl * dtl))
    
    # Total variance including convective part
    sigma_sq = gtx.maximum(1.0e-25, sigma_turb_sq + sig_conv * sig_conv)
    
    return gtx.sqrt(sigma_sq)


@gtx.field_operator
def compute_gaussian_condensation(
    sbar: Field[[I, J, K], float64],
    sigma: Field[[I, J, K], float64]
) -> Tuple[
    Field[[I, J, K], float64],  # cloud_fraction
    Field[[I, J, K], float64],  # condensate
    Field[[I, J, K], float64]   # sigrc
]:
    """
    Compute cloud fraction and condensate using Gaussian PDF.
    """
    # Normalized saturation deficit
    q1 = sbar / sigma
    
    # Gaussian computation
    gcond = -q1 / gtx.sqrt(2.0)
    gauv = 1.0 + gtx.erf(-gcond)
    
    # Cloud fraction
    cloud_fraction = gtx.maximum(0.0, gtx.minimum(1.0, 0.5 * gauv))
    
    # Condensate
    condensate = ((gtx.exp(-gcond * gcond) - gcond * gtx.sqrt(CST.XPI) * gauv) * 
                  sigma / gtx.sqrt(2.0 * CST.XPI))
    condensate = gtx.maximum(condensate, 0.0)
    
    # Set to zero if very small
    condensate = gtx.where(
        (condensate < 1.0e-12) | (cloud_fraction == 0.0),
        0.0,
        condensate
    )
    cloud_fraction = gtx.where(condensate == 0.0, 0.0, cloud_fraction)
    
    # sigrc
    sigrc = cloud_fraction
    
    return cloud_fraction, condensate, sigrc


@gtx.field_operator
def compute_cb02_condensation(
    sbar: Field[[I, J, K], float64],
    sigma: Field[[I, J, K], float64]
) -> Tuple[
    Field[[I, J, K], float64],  # cloud_fraction
    Field[[I, J, K], float64],  # condensate
    Field[[I, J, K], float64]   # sigrc
]:
    """
    Compute cloud fraction and condensate using CB02 scheme.
    """
    # Normalized saturation deficit
    q1 = sbar / sigma
    
    # Total condensate using CB02 formulation
    condensate = gtx.where(
        (q1 > 0.0) & (q1 <= 2.0),
        gtx.minimum(gtx.exp(-1.0) + 0.66 * q1 + 0.086 * q1 * q1, 2.0),
        gtx.where(
            q1 > 2.0,
            q1,
            gtx.exp(1.2 * q1 - 1.0)
        )
    )
    condensate = condensate * sigma
    condensate = gtx.maximum(condensate, 0.0)
    
    # Cloud fraction
    cloud_fraction = gtx.where(
        condensate < 1.0e-12,
        0.0,
        gtx.maximum(0.0, gtx.minimum(1.0, 0.5 + 0.36 * gtx.atan(1.55 * q1)))
    )
    
    # Set condensate to zero if cloud fraction is zero
    condensate = gtx.where(cloud_fraction == 0.0, 0.0, condensate)
    
    # Compute sigrc using lookup table approximation
    # This is a simplified version - the original uses a lookup table
    inq1 = gtx.floor(gtx.minimum(gtx.maximum(-22.0, 2.0 * q1), 10.0))
    zinc = 2.0 * q1 - inq1
    
    # Simplified sigrc computation (would need full lookup table for exact match)
    sigrc = gtx.minimum(1.0, gtx.maximum(0.0, 0.5 * (1.0 + gtx.erf(q1 / gtx.sqrt(2.0)))))
    
    return cloud_fraction, condensate, sigrc


@gtx.field_operator
def split_condensate_by_phase(
    condensate: Field[[I, J, K], float64],
    ice_fraction: Field[[I, J, K], float64]
) -> Tuple[
    Field[[I, J, K], float64],  # liquid condensate
    Field[[I, J, K], float64]   # ice condensate
]:
    """
    Split total condensate into liquid and ice components based on ice fraction.
    """
    rc_out = (1.0 - ice_fraction) * condensate
    ri_out = ice_fraction * condensate
    
    return rc_out, ri_out


@gtx.field_operator
def update_temperature_from_condensation(
    temperature: Field[[I, J, K], float64],
    rc_change: Field[[I, J, K], float64],
    ri_change: Field[[I, J, K], float64],
    lv: Field[[I, J, K], float64],
    ls: Field[[I, J, K], float64],
    cph: Field[[I, J, K], float64]
) -> Field[[I, J, K], float64]:
    """
    Update temperature due to latent heat release from condensation.
    """
    return temperature + (rc_change * lv + ri_change * ls) / cph


def condensation(
    dim: DimPhyex,
    # Input fields
    pressure: Field[[I, J, K], float64],
    height: Field[[I, J, K], float64],
    rho_ref: Field[[I, J, K], float64],
    temperature: Field[[I, J, K], float64],
    rv_in: Field[[I, J, K], float64],
    rc_in: Field[[I, J, K], float64],
    ri_in: Field[[I, J, K], float64],
    rr: Field[[I, J, K], float64],
    rs: Field[[I, J, K], float64],
    rg: Field[[I, J, K], float64],
    # Configuration
    hfrac_ice: str = "T",
    hcondens: str = "GAUS",
    hlambda3: str = "CB",
    use_subgrid: bool = True,
    sigma_s: Optional[Field[[I, J, K], float64]] = None,
    # Optional inputs
    lv: Optional[Field[[I, J, K], float64]] = None,
    ls: Optional[Field[[I, J, K], float64]] = None,
    cph: Optional[Field[[I, J, K], float64]] = None,
) -> Tuple[
    Field[[I, J, K], float64],  # rv_out
    Field[[I, J, K], float64],  # rc_out
    Field[[I, J, K], float64],  # ri_out
    Field[[I, J, K], float64],  # temperature_out
    Field[[I, J, K], float64],  # cloud_fraction
    Field[[I, J, K], float64],  # sigrc
]:
    """
    Main condensation routine.
    
    This function implements the condensation scheme for mixed-phase clouds,
    including both subgrid condensation and all-or-nothing schemes.
    
    Args:
        dim: Dimension structure
        pressure: Pressure field [Pa]
        height: Height field [m]
        rho_ref: Reference density [kg/m³]
        temperature: Temperature field [K] (modified in-place)
        rv_in: Input water vapor mixing ratio [kg/kg]
        rc_in: Input cloud water mixing ratio [kg/kg]
        ri_in: Input cloud ice mixing ratio [kg/kg]
        rr: Rain mixing ratio [kg/kg]
        rs: Snow mixing ratio [kg/kg]
        rg: Graupel mixing ratio [kg/kg]
        hfrac_ice: Ice fraction method
        hcondens: Condensation scheme ("GAUS" or "CB02")
        hlambda3: Lambda3 formulation
        use_subgrid: Whether to use subgrid condensation
        sigma_s: Sigma_s from turbulence scheme (if provided)
        lv: Latent heat of vaporization (computed if not provided)
        ls: Latent heat of sublimation (computed if not provided)
        cph: Specific heat of moist air (computed if not provided)
        
    Returns:
        Tuple of output fields: (rv_out, rc_out, ri_out, temperature_out, cloud_fraction, sigrc)
    """
    
    # Compute thermodynamic quantities if not provided
    if lv is None:
        lv = compute_latent_heat_vaporization(temperature)
    if ls is None:
        ls = compute_latent_heat_sublimation(temperature)
    if cph is None:
        cph = compute_specific_heat_moist_air(rv_in, rc_in, ri_in, rr, rs, rg)
    
    # Initialize zprifact (for compatibility with original code)
    zprifact = gtx.ones_like(rv_in)
    
    # Compute total water mixing ratio
    rt = rv_in + rc_in + ri_in * zprifact
    
    # Compute saturation vapor pressures
    pv_liquid = compute_saturation_vapor_pressure_liquid(temperature)
    pv_ice = compute_saturation_vapor_pressure_ice(temperature)
    
    # Limit to avoid numerical issues
    pv_liquid = gtx.minimum(pv_liquid, 0.99 * pressure)
    pv_ice = gtx.minimum(pv_ice, 0.99 * pressure)
    
    # Compute saturation mixing ratios
    qsl_liquid = compute_saturation_mixing_ratio(pressure, pv_liquid)
    qsi_ice = compute_saturation_mixing_ratio(pressure, pv_ice)
    
    # Compute ice fraction
    ice_fraction = compute_ice_fraction_simple(temperature)
    
    # Interpolate between liquid and ice saturation
    qsl = (1.0 - ice_fraction) * qsl_liquid + ice_fraction * qsi_ice
    lvs = (1.0 - ice_fraction) * lv + ice_fraction * ls
    
    # Compute condensation coefficients
    ah, a, b = compute_condensation_coefficients(temperature, pressure, qsl, lvs, cph)
    
    # Compute normalized saturation deficit
    sbar = compute_sbar(rt, qsl, ah, lvs, cph, rc_in, ri_in, a, zprifact)
    
    if use_subgrid and sigma_s is not None:
        # Use provided sigma_s (subgrid condensation)
        sigma = sigma_s
        # Apply height-dependent scaling if needed
        sigma = gtx.maximum(1.0e-10, sigma)
    else:
        # Compute sigma using first-order closure (simplified version)
        # This would need more complex implementation for full turbulent closure
        sigma = gtx.ones_like(sbar) * 1.0e-4  # Simplified constant value
        sigma = gtx.maximum(1.0e-10, sigma)
    
    # Apply condensation scheme
    if hcondens == "GAUS":
        cloud_fraction, condensate, sigrc = compute_gaussian_condensation(sbar, sigma)
    elif hcondens == "CB02":
        cloud_fraction, condensate, sigrc = compute_cb02_condensation(sbar, sigma)
    else:
        raise ValueError(f"Unknown condensation scheme: {hcondens}")
    
    # Split condensate into liquid and ice
    rc_out, ri_out = split_condensate_by_phase(condensate, ice_fraction)
    
    # Update temperature due to latent heat release
    rc_change = rc_out - rc_in
    ri_change = ri_out - ri_in
    temperature_out = update_temperature_from_condensation(
        temperature, rc_change, ri_change, lv, ls, cph
    )
    
    # Compute output water vapor
    rv_out = rt - rc_out - ri_out * zprifact
    
    # Apply lambda3 coefficient if requested
    if hlambda3 == "CB":
        q1 = sbar / sigma
        lambda3 = gtx.minimum(3.0, gtx.maximum(1.0, 1.0 - q1))
        sigrc = sigrc * lambda3
    
    return rv_out, rc_out, ri_out, temperature_out, cloud_fraction, sigrc
