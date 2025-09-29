"""
Utility functions for the GT4Py ice adjustment implementation.

This module contains helper functions for thermodynamic calculations,
saturation vapor pressure computations, and other utilities.
"""

import numpy as np
import gt4py.next as gtx
from gt4py.next import Field, float64, int32

from ice3.common.constants import CST
from ice3.common.dims import I, J, K


def compute_saturation_vapor_pressure_liquid(temperature: Field[[I, J, K], float64]) -> Field[[I, J, K], float64]:
    """
    Compute saturation vapor pressure over liquid water.
    
    Equivalent to the Fortran expression:
    EXP(XALPW - XBETAW / T - XGAMW * LOG(T))
    
    Args:
        temperature: Temperature field in Kelvin
        
    Returns:
        Saturation vapor pressure over liquid water in Pa
    """
    return gtx.exp(CST.XALPW - CST.XBETAW / temperature - CST.XGAMW * gtx.log(temperature))


def compute_saturation_vapor_pressure_ice(temperature: Field[[I, J, K], float64]) -> Field[[I, J, K], float64]:
    """
    Compute saturation vapor pressure over ice.
    
    Equivalent to the Fortran expression:
    EXP(XALPI - XBETAI / T - XGAMI * LOG(T))
    
    Args:
        temperature: Temperature field in Kelvin
        
    Returns:
        Saturation vapor pressure over ice in Pa
    """
    return gtx.exp(CST.XALPI - CST.XBETAI / temperature - CST.XGAMI * gtx.log(temperature))


def compute_latent_heat_vaporization(temperature: Field[[I, J, K], float64]) -> Field[[I, J, K], float64]:
    """
    Compute latent heat of vaporization as function of temperature.
    
    Equivalent to the Fortran expression:
    XLVTT + (XCPV - XCL) * (T - XTT)
    
    Args:
        temperature: Temperature field in Kelvin
        
    Returns:
        Latent heat of vaporization in J/kg
    """
    return CST.XLVTT + (CST.XCPV - CST.XCL) * (temperature - CST.XTT)


def compute_latent_heat_sublimation(temperature: Field[[I, J, K], float64]) -> Field[[I, J, K], float64]:
    """
    Compute latent heat of sublimation as function of temperature.
    
    Equivalent to the Fortran expression:
    XLSTT + (XCPV - XCI) * (T - XTT)
    
    Args:
        temperature: Temperature field in Kelvin
        
    Returns:
        Latent heat of sublimation in J/kg
    """
    return CST.XLSTT + (CST.XCPV - CST.XCI) * (temperature - CST.XTT)


def compute_specific_heat_moist_air(
    rv: Field[[I, J, K], float64],
    rc: Field[[I, J, K], float64],
    ri: Field[[I, J, K], float64],
    rr: Field[[I, J, K], float64],
    rs: Field[[I, J, K], float64],
    rg: Field[[I, J, K], float64],
    rh: Field[[I, J, K], float64] = None
) -> Field[[I, J, K], float64]:
    """
    Compute specific heat of moist air including all hydrometeor species.
    
    Args:
        rv: Water vapor mixing ratio
        rc: Cloud water mixing ratio
        ri: Cloud ice mixing ratio
        rr: Rain mixing ratio
        rs: Snow mixing ratio
        rg: Graupel mixing ratio
        rh: Hail mixing ratio (optional)
        
    Returns:
        Specific heat of moist air in J/(kg·K)
    """
    cph = (CST.XCPD + CST.XCPV * rv + 
           CST.XCL * (rc + rr) + 
           CST.XCI * (ri + rs + rg))
    
    if rh is not None:
        cph = cph + CST.XCI * rh
        
    return cph


def compute_saturation_mixing_ratio(
    pressure: Field[[I, J, K], float64],
    saturation_pressure: Field[[I, J, K], float64]
) -> Field[[I, J, K], float64]:
    """
    Compute saturation mixing ratio from pressure and saturation vapor pressure.
    
    Equivalent to the Fortran expression:
    XRD / XRV * es / (p - es)
    
    Args:
        pressure: Total pressure in Pa
        saturation_pressure: Saturation vapor pressure in Pa
        
    Returns:
        Saturation mixing ratio in kg/kg
    """
    return CST.XRD / CST.XRV * saturation_pressure / (pressure - saturation_pressure)


@gtx.field_operator
def compute_ice_fraction_simple(temperature: Field[[I, J, K], float64]) -> Field[[I, J, K], float64]:
    """
    Compute ice fraction based on temperature (simple linear transition).
    
    This is a simplified version of the COMPUTE_FRAC_ICE function.
    The ice fraction varies linearly between 0°C and -40°C.
    
    Args:
        temperature: Temperature field in Kelvin
        
    Returns:
        Ice fraction (0 = all liquid, 1 = all ice)
    """
    t_celsius = temperature - CST.XTT
    
    # Linear transition between 0°C and -40°C
    ice_frac = gtx.where(
        t_celsius >= 0.0,
        0.0,  # All liquid above 0°C
        gtx.where(
            t_celsius <= -40.0,
            1.0,  # All ice below -40°C
            -t_celsius / 40.0  # Linear transition
        )
    )
    
    return ice_frac


@gtx.field_operator  
def gaussian_cdf(x: Field[[I, J, K], float64]) -> Field[[I, J, K], float64]:
    """
    Compute the cumulative distribution function of the standard normal distribution.
    
    This uses the error function approximation: CDF(x) = 0.5 * (1 + erf(x/sqrt(2)))
    
    Args:
        x: Input values
        
    Returns:
        CDF values
    """
    return 0.5 * (1.0 + gtx.erf(x / gtx.sqrt(2.0)))


@gtx.field_operator
def gaussian_pdf(x: Field[[I, J, K], float64]) -> Field[[I, J, K], float64]:
    """
    Compute the probability density function of the standard normal distribution.
    
    PDF(x) = exp(-x²/2) / sqrt(2π)
    
    Args:
        x: Input values
        
    Returns:
        PDF values
    """
    return gtx.exp(-0.5 * x * x) / gtx.sqrt(2.0 * CST.XPI)


def safe_divide(
    numerator: Field[[I, J, K], float64],
    denominator: Field[[I, J, K], float64],
    default_value: float = 0.0
) -> Field[[I, J, K], float64]:
    """
    Perform safe division with protection against division by zero.
    
    Args:
        numerator: Numerator field
        denominator: Denominator field
        default_value: Value to use when denominator is zero
        
    Returns:
        Result of division with protection against division by zero
    """
    return gtx.where(
        gtx.abs(denominator) > CST.XMNH_TINY,
        numerator / denominator,
        default_value
    )


def limit_to_range(
    field: Field[[I, J, K], float64],
    min_val: float,
    max_val: float
) -> Field[[I, J, K], float64]:
    """
    Limit field values to a specified range.
    
    Args:
        field: Input field
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Field with values limited to [min_val, max_val]
    """
    return gtx.maximum(min_val, gtx.minimum(max_val, field))
