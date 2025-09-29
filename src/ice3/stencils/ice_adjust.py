"""
Ice adjustment module - GT4Py translation of ice_adjust.F90.

This module implements the ice adjustment scheme for mixed-phase clouds,
which computes the fast microphysical sources through a saturation adjustment
procedure.
"""

import numpy as np
import gt4py.next as gtx
from gt4py.next import Field, float64, int32
from typing import Tuple, Optional

from ice3.common.constants import CST
from ice3.common.dimensions import DimPhyex
from ice3.stencils.condensation import condensation
from ice3.utils.next_tools import (
    compute_latent_heat_vaporization,
    compute_latent_heat_sublimation,
    compute_specific_heat_moist_air,
    limit_to_range
)


@gtx.field_operator
def compute_source_terms(
    rc_new: Field[[I, J, K], float64],
    rc_old: Field[[I, J, K], float64],
    ri_new: Field[[I, J, K], float64],
    ri_old: Field[[I, J, K], float64],
    timestep: float
) -> Tuple[
    Field[[I, J, K], float64],  # rc_source
    Field[[I, J, K], float64]   # ri_source
]:
    """
    Compute source terms from the difference between new and old mixing ratios.
    
    Args:
        rc_new: New cloud water mixing ratio
        rc_old: Old cloud water mixing ratio  
        ri_new: New cloud ice mixing ratio
        ri_old: Old cloud ice mixing ratio
        timestep: Time step
        
    Returns:
        Tuple of (rc_source, ri_source) in kg/kg/s
    """
    rc_source = (rc_new - rc_old) / timestep
    ri_source = (ri_new - ri_old) / timestep
    
    return rc_source, ri_source


@gtx.field_operator
def apply_mass_flux_weighting(
    field: Field[[I, J, K], float64],
    weight_mf_cloud: Field[[I, J, K], float64]
) -> Field[[I, J, K], float64]:
    """
    Apply mass flux cloud weighting to a field.
    
    Args:
        field: Input field
        weight_mf_cloud: Mass flux cloud weight coefficient
        
    Returns:
        Weighted field
    """
    return field * (1.0 - weight_mf_cloud)


@gtx.field_operator
def compute_theta_source(
    rv_source: Field[[I, J, K], float64],
    rc_source: Field[[I, J, K], float64],
    ri_source: Field[[I, J, K], float64],
    lv: Field[[I, J, K], float64],
    ls: Field[[I, J, K], float64],
    cph: Field[[I, J, K], float64],
    exnref: Field[[I, J, K], float64]
) -> Field[[I, J, K], float64]:
    """
    Compute potential temperature source term from latent heat release.
    
    Args:
        rv_source: Water vapor source (negative for condensation)
        rc_source: Cloud water source
        ri_source: Cloud ice source
        lv: Latent heat of vaporization
        ls: Latent heat of sublimation
        cph: Specific heat of moist air
        exnref: Reference Exner function
        
    Returns:
        Potential temperature source in K/s
    """
    # Note: rv_source is negative when condensation occurs, so we use -rv_source
    # to get positive heating from condensation
    return (-rv_source * lv + ri_source * (ls - lv)) / (cph * exnref)


@gtx.field_operator
def limit_source_terms(
    rv_source: Field[[I, J, K], float64],
    rc_source: Field[[I, J, K], float64],
    ri_source: Field[[I, J, K], float64],
    rv_current: Field[[I, J, K], float64],
    rc_current: Field[[I, J, K], float64],
    ri_current: Field[[I, J, K], float64]
) -> Tuple[
    Field[[I, J, K], float64],  # limited rv_source
    Field[[I, J, K], float64],  # limited rc_source
    Field[[I, J, K], float64]   # limited ri_source
]:
    """
    Limit source terms to prevent negative mixing ratios.
    
    Args:
        rv_source: Water vapor source
        rc_source: Cloud water source
        ri_source: Cloud ice source
        rv_current: Current water vapor mixing ratio
        rc_current: Current cloud water mixing ratio
        ri_current: Current cloud ice mixing ratio
        
    Returns:
        Tuple of limited source terms
    """
    # Limit cloud water source
    rc_source_limited = gtx.where(
        rc_source < 0.0,
        gtx.maximum(rc_source, -rc_current),
        gtx.minimum(rc_source, rv_current)
    )
    
    # Limit cloud ice source
    ri_source_limited = gtx.where(
        ri_source < 0.0,
        gtx.maximum(ri_source, -ri_current),
        gtx.minimum(ri_source, rv_current)
    )
    
    # Update water vapor source to maintain conservation
    rv_source_limited = rv_source - (rc_source_limited - rc_source) - (ri_source_limited - ri_source)
    
    return rv_source_limited, rc_source_limited, ri_source_limited


@gtx.field_operator
def compute_all_or_nothing_cloud_fraction(
    rc_source: Field[[I, J, K], float64],
    ri_source: Field[[I, J, K], float64],
    timestep: float
) -> Field[[I, J, K], float64]:
    """
    Compute cloud fraction for all-or-nothing condensation scheme.
    
    Args:
        rc_source: Cloud water source
        ri_source: Cloud ice source
        timestep: Time step
        
    Returns:
        Cloud fraction (0 or 1)
    """
    total_condensate_source = (rc_source + ri_source) * timestep
    return gtx.where(total_condensate_source > 1.0e-12, 1.0, 0.0)


def ice_adjust(
    dim: DimPhyex,
    # Input fields - state variables
    pressure: Field[[I, J, K], float64],
    height: Field[[I, J, K], float64],
    rho_dry: Field[[I, J, K], float64],
    exner: Field[[I, J, K], float64],
    exner_ref: Field[[I, J, K], float64],
    rho_ref: Field[[I, J, K], float64],
    theta: Field[[I, J, K], float64],
    rv: Field[[I, J, K], float64],
    rc: Field[[I, J, K], float64],
    ri: Field[[I, J, K], float64],
    rr: Field[[I, J, K], float64],
    rs: Field[[I, J, K], float64],
    rg: Field[[I, J, K], float64],
    # Input/output fields - source terms
    rv_source: Field[[I, J, K], float64],
    rc_source: Field[[I, J, K], float64],
    ri_source: Field[[I, J, K], float64],
    theta_source: Field[[I, J, K], float64],
    # Configuration parameters
    timestep: float,
    krr: int = 6,  # Number of moist variables
    use_subgrid_condensation: bool = True,
    hfrac_ice: str = "T",
    hcondens: str = "GAUS",
    hlambda3: str = "CB",
    # Optional mass flux fields
    weight_mf_cloud: Optional[Field[[I, J, K], float64]] = None,
    cf_mf: Optional[Field[[I, J, K], float64]] = None,
    rc_mf: Optional[Field[[I, J, K], float64]] = None,
    ri_mf: Optional[Field[[I, J, K], float64]] = None,
    # Optional turbulence fields
    sigma_s: Optional[Field[[I, J, K], float64]] = None,
    # Optional hail field
    rh: Optional[Field[[I, J, K], float64]] = None,
) -> Tuple[
    Field[[I, J, K], float64],  # rv_source_out
    Field[[I, J, K], float64],  # rc_source_out
    Field[[I, J, K], float64],  # ri_source_out
    Field[[I, J, K], float64],  # theta_source_out
    Field[[I, J, K], float64],  # cloud_fraction
    Field[[I, J, K], float64],  # ice_cloud_fraction
    Field[[I, J, K], float64],  # water_cloud_fraction
]:
    """
    Ice adjustment scheme for mixed-phase clouds.
    
    This function computes the fast microphysical sources through a saturation
    adjustment procedure in case of mixed-phase clouds.
    
    Args:
        dim: Dimension structure
        pressure: Pressure field [Pa]
        height: Height field [m]
        rho_dry: Dry density * Jacobian [kg/m³]
        exner: Exner function
        exner_ref: Reference Exner function
        rho_ref: Reference density [kg/m³]
        theta: Potential temperature [K]
        rv: Water vapor mixing ratio [kg/kg]
        rc: Cloud water mixing ratio [kg/kg]
        ri: Cloud ice mixing ratio [kg/kg]
        rr: Rain mixing ratio [kg/kg]
        rs: Snow mixing ratio [kg/kg]
        rg: Graupel mixing ratio [kg/kg]
        rv_source: Water vapor source [kg/kg/s] (input/output)
        rc_source: Cloud water source [kg/kg/s] (input/output)
        ri_source: Cloud ice source [kg/kg/s] (input/output)
        theta_source: Potential temperature source [K/s] (input/output)
        timestep: Time step [s]
        krr: Number of moist variables
        use_subgrid_condensation: Whether to use subgrid condensation
        hfrac_ice: Ice fraction method
        hcondens: Condensation scheme
        hlambda3: Lambda3 formulation
        weight_mf_cloud: Mass flux cloud weight coefficient
        cf_mf: Convective mass flux cloud fraction
        rc_mf: Convective mass flux liquid mixing ratio
        ri_mf: Convective mass flux ice mixing ratio
        sigma_s: Sigma_s from turbulence scheme
        rh: Hail mixing ratio [kg/kg] (optional)
        
    Returns:
        Tuple of updated source terms and cloud fractions
    """
    
    # Compute temperature from potential temperature
    temperature = theta * exner
    
    # Compute thermodynamic quantities
    lv = compute_latent_heat_vaporization(temperature)
    ls = compute_latent_heat_sublimation(temperature)
    
    # Include hail in specific heat calculation if provided
    if rh is not None:
        cph = compute_specific_heat_moist_air(rv, rc, ri, rr, rs, rg, rh)
    else:
        cph = compute_specific_heat_moist_air(rv, rc, ri, rr, rs, rg)
    
    # Set default mass flux weight if not provided
    if weight_mf_cloud is None:
        weight_mf_cloud = gtx.zeros_like(rv)
    
    # Perform iterative adjustment (simplified to 1 iteration for now)
    max_iterations = 1
    
    # Initialize adjusted state with current values
    rv_adjusted = rv
    rc_adjusted = rc
    ri_adjusted = ri
    temperature_adjusted = temperature
    
    for iteration in range(max_iterations):
        # Call condensation scheme
        rv_new, rc_new, ri_new, temperature_new, cloud_fraction, sigrc = condensation(
            dim=dim,
            pressure=pressure,
            height=height,
            rho_ref=rho_ref,
            temperature=temperature_adjusted,
            rv_in=rv_adjusted,
            rc_in=rc_adjusted,
            ri_in=ri_adjusted,
            rr=rr,
            rs=rs,
            rg=rg,
            hfrac_ice=hfrac_ice,
            hcondens=hcondens,
            hlambda3=hlambda3,
            use_subgrid=use_subgrid_condensation,
            sigma_s=sigma_s,
            lv=lv,
            ls=ls,
            cph=cph
        )
        
        # Update adjusted state
        rv_adjusted = rv_new
        rc_adjusted = rc_new
        ri_adjusted = ri_new
        temperature_adjusted = temperature_new
    
    # Apply mass flux weighting to condensation results
    rc_adjusted = apply_mass_flux_weighting(rc_adjusted, weight_mf_cloud)
    ri_adjusted = apply_mass_flux_weighting(ri_adjusted, weight_mf_cloud)
    cloud_fraction = apply_mass_flux_weighting(cloud_fraction, weight_mf_cloud)
    sigrc = apply_mass_flux_weighting(sigrc, weight_mf_cloud)
    
    # Add mass flux contributions if provided
    if cf_mf is not None and rc_mf is not None and ri_mf is not None:
        # Limit mass flux contributions to available water vapor
        rc_mf_limited = rc_mf / timestep
        ri_mf_limited = ri_mf / timestep
        total_mf = rc_mf_limited + ri_mf_limited
        
        # Scale if total exceeds available water vapor
        scale_factor = gtx.where(
            total_mf > rv_source,
            rv_source / gtx.maximum(total_mf, 1.0e-20),
            1.0
        )
        
        rc_mf_limited = rc_mf_limited * scale_factor
        ri_mf_limited = ri_mf_limited * scale_factor
        
        # Add mass flux contributions
        cloud_fraction = gtx.minimum(1.0, cloud_fraction + cf_mf)
        rc_source = rc_source + rc_mf_limited
        ri_source = ri_source + ri_mf_limited
        rv_source = rv_source - (rc_mf_limited + ri_mf_limited)
        
        # Update theta source
        theta_source = theta_source + (rc_mf_limited * lv + ri_mf_limited * ls) / (cph * exner_ref)
    
    # Compute source terms from adjustment
    rc_source_adj, ri_source_adj = compute_source_terms(
        rc_adjusted, rc, ri_adjusted, ri, timestep
    )
    
    # Update source terms
    rv_source_new = rv_source - rc_source_adj - ri_source_adj
    rc_source_new = rc_source + rc_source_adj
    ri_source_new = ri_source + ri_source_adj
    
    # Limit source terms to prevent negative mixing ratios
    rv_source_limited, rc_source_limited, ri_source_limited = limit_source_terms(
        rv_source_new, rc_source_new, ri_source_new, rv, rc, ri
    )
    
    # Compute theta source from latent heat release
    theta_source_adj = compute_theta_source(
        rv_source_limited - rv_source, rc_source_limited - rc_source, 
        ri_source_limited - ri_source, lv, ls, cph, exner_ref
    )
    theta_source_new = theta_source + theta_source_adj
    
    # Compute cloud fractions
    if not use_subgrid_condensation:
        # All-or-nothing scheme
        cloud_fraction = compute_all_or_nothing_cloud_fraction(
            rc_source_limited, ri_source_limited, timestep
        )
        sigrc = cloud_fraction
    
    # For now, set ice and water cloud fractions equal to total cloud fraction
    # In a more complete implementation, these would be computed separately
    ice_cloud_fraction = cloud_fraction
    water_cloud_fraction = cloud_fraction
    
    return (rv_source_limited, rc_source_limited, ri_source_limited, 
            theta_source_new, cloud_fraction, ice_cloud_fraction, water_cloud_fraction)
