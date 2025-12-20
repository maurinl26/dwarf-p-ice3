"""
Example usage of the PHYEX ICE_ADJUST Cython wrapper.

This demonstrates how to call the Fortran ICE_ADJUST routine from Python
using the Cython bridge defined in _phyex_wrapper.pyx.
"""
import sys
import os
# Add build directory to path to find the compiled extension
build_dir = os.path.join(os.path.dirname(__file__), '../../../build/cp312-cp312-macosx_26_0_arm64')
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import numpy as np
from _phyex_wrapper import ice_adjust

def example_ice_adjust():
    """
    Example demonstrating ICE_ADJUST call with realistic atmospheric data.
    """
    # Domain dimensions
    nlon = 10   # Number of horizontal points
    nlev = 50   # Number of vertical levels
    krr = 6     # Number of moist variables (vapor, cloud, rain, ice, snow, graupel)
    
    # Time step (seconds) - must be float32 to match Cython wrapper
    timestep = np.float32(60.0)
    
    print("="*70)
    print("PHYEX ICE_ADJUST Cython Wrapper Example")
    print("="*70)
    print(f"Domain: {nlon} × {nlev} (nlon × nlev)")
    print(f"Time step: {timestep} s")
    print(f"Moist variables: {krr}")
    print()
    
    # IMPORTANT: All arrays must be:
    # 1. float32 (np.float32) - PHYEX uses single precision
    # 2. Fortran-contiguous (order='F') - Fortran column-major layout
    
    # 1D arrays (nlon,)
    sigqsat = np.ones(nlon, dtype=np.float32, order='F') * 0.01
    
    # 2D arrays (nlon, nlev) - Atmospheric state
    # IMPORTANT: Create arrays as Fortran-contiguous from the start
    pabs = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    th = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    exn = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    
    # Standard atmosphere approximation
    p0 = 101325.0  # Surface pressure (Pa)
    T0 = 288.15    # Surface temperature (K)
    gamma = 0.0065 # Lapse rate (K/m)
    
    for k in range(nlev):
        # Height (simplified, 0-15 km)
        z = k * 300.0
        
        # Pressure profile
        p = p0 * (1 - gamma * z / T0) ** 5.26
        pabs[:, k] = p
        
        # Temperature profile
        T = T0 - gamma * z
        
        # Exner function
        p00 = 100000.0
        Rd = 287.0
        cp = 1004.0
        exn_val = (p / p00) ** (Rd / cp)
        exn[:, k] = exn_val
        
        # Potential temperature
        th[:, k] = T / exn_val
    
    # Reference values (ensure Fortran contiguity)
    exn_ref = np.asfortranarray(exn.copy(), dtype=np.float32)
    # Compute density and ensure Fortran contiguity
    rho_dry_ref = np.asfortranarray(pabs / (287.0 * th * exn), dtype=np.float32)
    
    # Subgrid turbulence parameter
    sigs = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    
    # Mixing ratios (kg/kg)
    rv = np.ones((nlon, nlev), dtype=np.float32, order='F') * 0.010  # 10 g/kg water vapor
    rc = np.zeros((nlon, nlev), dtype=np.float32, order='F')         # Cloud water
    ri = np.zeros((nlon, nlev), dtype=np.float32, order='F')         # Cloud ice
    rr = np.zeros((nlon, nlev), dtype=np.float32, order='F')         # Rain
    rs = np.zeros((nlon, nlev), dtype=np.float32, order='F')         # Snow
    rg = np.zeros((nlon, nlev), dtype=np.float32, order='F')         # Graupel
    
    # Add some cloud water at mid-levels
    rc[:, 20:30] = 0.001  # 1 g/kg
    
    # Mass flux scheme variables (set to zero for this example)
    cf_mf = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    rc_mf = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    ri_mf = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    
    # Tendency arrays (input/output)
    rvs = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    rcs = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    ris = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    ths = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    
    # Output arrays (cloud fractions)
    cldfr = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    icldfr = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    wcldfr = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    
    # Print initial state
    print("Initial state:")
    print(f"  Temperature range: {(th * exn).min():.1f} - {(th * exn).max():.1f} K")
    print(f"  Pressure range: {pabs.min():.0f} - {pabs.max():.0f} Pa")
    print(f"  Water vapor: {rv.mean()*1000:.3f} g/kg (mean)")
    print(f"  Cloud water: {rc.mean()*1000:.3f} g/kg (mean)")
    print()
    
    # Call ICE_ADJUST
    print("Calling ICE_ADJUST...")
    ice_adjust(
        timestep=timestep,
        krr=krr,
        sigqsat=sigqsat,
        pabs=pabs,
        sigs=sigs,
        th=th,
        exn=exn,
        exn_ref=exn_ref,
        rho_dry_ref=rho_dry_ref,
        rv=rv,
        rc=rc,
        ri=ri,
        rr=rr,
        rs=rs,
        rg=rg,
        cf_mf=cf_mf,
        rc_mf=rc_mf,
        ri_mf=ri_mf,
        rvs=rvs,
        rcs=rcs,
        ris=ris,
        ths=ths,
        cldfr=cldfr,
        icldfr=icldfr,
        wcldfr=wcldfr
    )
    print("✓ ICE_ADJUST completed")
    print()
    
    # Print results
    print("Results:")
    print(f"  Cloud fraction: {cldfr.mean():.4f} (mean), max: {cldfr.max():.4f}")
    print(f"  Ice cloud fraction: {icldfr.mean():.4f} (mean)")
    print(f"  Water cloud fraction: {wcldfr.mean():.4f} (mean)")
    print(f"  Cloudy points: {(cldfr > 0.01).sum()} / {cldfr.size}")
    print()
    
    # Check tendencies
    print("Tendencies (per timestep):")
    print(f"  Water vapor: {rvs.mean()*1000:.6f} g/kg (mean)")
    print(f"  Cloud water: {rcs.mean()*1000:.6f} g/kg (mean)")
    print(f"  Cloud ice: {ris.mean()*1000:.6f} g/kg (mean)")
    print(f"  Temperature: {ths.mean():.6f} K (mean)")
    print()
    
    # Verify conservation
    total_water_change = (rvs + rcs + ris).sum()
    print(f"Total water conservation check: {total_water_change:.10e}")
    print("  (should be close to zero)")
    print()
    
    print("="*70)
    
    return {
        'cldfr': cldfr,
        'icldfr': icldfr,
        'wcldfr': wcldfr,
        'rvs': rvs,
        'rcs': rcs,
        'ris': ris,
        'ths': ths
    }


if __name__ == "__main__":
    results = example_ice_adjust()