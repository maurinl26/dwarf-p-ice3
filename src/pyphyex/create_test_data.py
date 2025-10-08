import numpy as np
from typing import Dict, Any

def create_test_data(nijt: int = 100, nkt: int = 50) -> Dict[str, Any]:
    """
    Create test data for the ice_adjust wrapper.
    
    Parameters:
    -----------
    nijt : int
        Horizontal dimension size
    nkt : int 
        Vertical dimension size
        
    Returns:
    --------
    dict
        Dictionary with test input data
    """
    
    # Create realistic atmospheric profiles
    pressure = np.linspace(101325, 10000, nkt)  # Pa, surface to ~16km
    height = -np.log(pressure / 101325.0) * 8000  # Approximate height [m]
    
    # Broadcast to full grid
    p3d = np.broadcast_to(pressure[None, :], (nijt, nkt))
    z3d = np.broadcast_to(height[None, :], (nijt, nkt))
    
    # Temperature profile (realistic troposphere)
    temp_profile = 288.15 - 0.0065 * height  # Linear lapse rate
    temp_profile = np.maximum(temp_profile, 216.65)  # Stratosphere floor
    temp3d = np.broadcast_to(temp_profile[None, :], (nijt, nkt)).copy()
    
    # Add some horizontal variability
    temp3d += np.random.normal(0, 2, (nijt, nkt))
    
    # Compute Exner function and potential temperature
    pexnref = (p3d / 100000.0) ** (287.0 / 1004.0)  # R_d / c_pd
    pth = temp3d / pexnref
    
    # Density
    prhodref = p3d / (287.0 * temp3d)  # Ideal gas law
    prhodj = prhodref  # Assume Jacobian = 1
    
    # Mixing ratios
    # Realistic water vapor profile
    qs_surface = 0.622 * 611.2 * np.exp(17.67 * 15 / (288.15 - 29.65)) / (101325 - 0.378 * 611.2 * np.exp(17.67 * 15 / (288.15 - 29.65)))
    prv = qs_surface * np.exp(-height[:, None] / 2000) * 0.8  # 80% relative humidity
    prv = np.broadcast_to(prv.T, (nijt, nkt)).copy()
    
    # Small amount of cloud water
    prc = np.where(temp3d < 280, 1e-6, 0.0)
    
    # Add some noise to create variability
    prv += np.random.normal(0, prv * 0.1)
    prv = np.maximum(prv, 1e-8)
    
    return {
        'nijt': nijt,
        'nkt': nkt,
        'krr': 6,  # vapor, cloud, rain, ice, snow, graupel
        'ptstep': 300.0,  # 5 minutes
        'psigqsat': np.ones(nijt),
        'prhodj': prhodj,
        'pexnref': pexnref,
        'prhodref': prhodref,
        'ppabst': p3d,
        'pzz': z3d,
        'pexn': pexnref,
        'prv': prv,
        'prc': prc,
        'prvs': np.zeros_like(prv),
        'prcs': np.zeros_like(prc),
        'pth': pth,
        'pths': np.zeros_like(pth),
        'prr': np.zeros_like(prc),
        'pri': np.zeros_like(prc),
        'pris': np.zeros_like(prc),
        'prs': np.zeros_like(prc),
        'prg': np.zeros_like(prc)
    }
