"""
Example usage of the PHYEX RAIN_ICE Cython wrapper.

This demonstrates how to call the Fortran RAIN_ICE routine from Python
using the Cython bridge defined in _phyex_wrapper.pyx.
"""
import sys
import os
# Add build directory to path to find the compiled extension
build_dir = os.path.join(os.path.dirname(__file__), '../../../build/cp312-cp312-macosx_26_0_arm64')
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import numpy as np
from _phyex_wrapper import rain_ice

def example_rain_ice():
    """
    Example demonstrating RAIN_ICE call with realistic atmospheric data.
    """
    # Domain dimensions
    nlon = 10   # Number of horizontal points
    nlev = 50   # Number of vertical levels
    krr = 6     # Number of moist variables (vapor, cloud, rain, ice, snow, graupel)
    
    # Time step (seconds) - must be float32 to match Cython wrapper
    timestep = np.float32(60.0)
    
    print("="*70)
    print("PHYEX RAIN_ICE Cython Wrapper Example")
    print("="*70)
    print(f"Domain: {nlon} × {nlev} (nlon × nlev)")
    print(f"Time step: {timestep} s")
    print(f"Moist variables: {krr}")
    print()
    
    # IMPORTANT: All arrays must be:
    # 1. float32 (np.float32) - PHYEX uses single precision
    # 2. Fortran-contiguous (order='F') - Fortran column-major layout
    
    # 2D arrays (nlon, nlev) - Atmospheric state
    # IMPORTANT: Create arrays as Fortran-contiguous from the start
    exn = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    dzz = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    rhodj = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    rhodref = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    exnref = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    pabs = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    
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
        
        # Layer thickness (simplified)
        dzz[:, k] = 300.0
        
        # Density (ideal gas law)
        rho = p / (Rd * T)
        rhodref[:, k] = rho
        rhodj[:, k] = rho * dzz[:, k]  # Dry density * Jacobian
    
    # Reference Exner function
    exnref = np.asfortranarray(exn.copy(), dtype=np.float32)
    
    # Potential temperature
    tht = np.asfortranarray(pabs / (rhodref * Rd * exn), dtype=np.float32)
    
    # Cloud fractions and supersaturation
    cldfr = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    icldfr = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    ssio = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    ssiu = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    ifr = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    
    # Mixing ratios at time t (kg/kg)
    rvt = np.ones((nlon, nlev), dtype=np.float32, order='F') * 0.010  # 10 g/kg water vapor
    rct = np.zeros((nlon, nlev), dtype=np.float32, order='F')         # Cloud water
    rrt = np.zeros((nlon, nlev), dtype=np.float32, order='F')         # Rain
    rit = np.zeros((nlon, nlev), dtype=np.float32, order='F')         # Cloud ice
    rst = np.zeros((nlon, nlev), dtype=np.float32, order='F')         # Snow
    rgt = np.zeros((nlon, nlev), dtype=np.float32, order='F')         # Graupel
    
    # Add some cloud water and rain at mid-levels
    rct[:, 20:30] = 0.001  # 1 g/kg cloud water
    rrt[:, 15:25] = 0.0005 # 0.5 g/kg rain
    
    # Subgrid turbulence parameter
    sigs = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    
    # Input/output arrays
    cit = np.zeros((nlon, nlev), dtype=np.float32, order='F')  # Ice number concentration
    hlc_hrc = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    hlc_hcf = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    hli_hri = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    hli_hcf = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    
    # Tendency arrays (input/output)
    ths = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    rvs = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    rcs = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    rrs = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    ris = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    rss = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    rgs = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    
    # Output arrays
    evap3d = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    rainfr = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    
    # 1D output arrays (precipitation rates)
    inprc = np.zeros(nlon, dtype=np.float32, order='F')
    inprr = np.zeros(nlon, dtype=np.float32, order='F')
    inprs = np.zeros(nlon, dtype=np.float32, order='F')
    inprg = np.zeros(nlon, dtype=np.float32, order='F')
    indep = np.zeros(nlon, dtype=np.float32, order='F')
    
    # Print initial state
    print("Initial state:")
    print(f"  Temperature range: {(tht * exn).min():.1f} - {(tht * exn).max():.1f} K")
    print(f"  Pressure range: {pabs.min():.0f} - {pabs.max():.0f} Pa")
    print(f"  Water vapor: {rvt.mean()*1000:.3f} g/kg (mean)")
    print(f"  Cloud water: {rct.mean()*1000:.3f} g/kg (mean)")
    print(f"  Rain: {rrt.mean()*1000:.3f} g/kg (mean)")
    print()
    
    # Call RAIN_ICE
    print("Calling RAIN_ICE...")
    try:
        rain_ice(
            timestep=timestep,
            krr=krr,
            exn=exn,
            dzz=dzz,
            rhodj=rhodj,
            rhodref=rhodref,
            exnref=exnref,
            pabs=pabs,
            cldfr=cldfr,
            icldfr=icldfr,
            ssio=ssio,
            ssiu=ssiu,
            ifr=ifr,
            tht=tht,
            rvt=rvt,
            rct=rct,
            rrt=rrt,
            rit=rit,
            rst=rst,
            rgt=rgt,
            sigs=sigs,
            cit=cit,
            hlc_hrc=hlc_hrc,
            hlc_hcf=hlc_hcf,
            hli_hri=hli_hri,
            hli_hcf=hli_hcf,
            ths=ths,
            rvs=rvs,
            rcs=rcs,
            rrs=rrs,
            ris=ris,
            rss=rss,
            rgs=rgs,
            evap3d=evap3d,
            rainfr=rainfr,
            inprc=inprc,
            inprr=inprr,
            inprs=inprs,
            inprg=inprg,
            indep=indep
        )
        print("✓ RAIN_ICE completed")
        print()
        
        # Print results
        print("Results:")
        print(f"  Precipitation fraction: {rainfr.mean():.4f} (mean), max: {rainfr.max():.4f}")
        print(f"  Rain evaporation: {evap3d.mean()*1000:.6f} g/kg (mean)")
        print()
        
        # Precipitation rates
        print("Instantaneous precipitation rates:")
        print(f"  Cloud precip: {inprc.mean():.6f} mm/s (mean)")
        print(f"  Rain precip: {inprr.mean():.6f} mm/s (mean)")
        print(f"  Snow precip: {inprs.mean():.6f} mm/s (mean)")
        print(f"  Graupel precip: {inprg.mean():.6f} mm/s (mean)")
        print(f"  Deposition: {indep.mean():.6f} mm/s (mean)")
        print()
        
        # Check tendencies
        print("Tendencies (per timestep):")
        print(f"  Temperature: {ths.mean():.6f} K (mean)")
        print(f"  Water vapor: {rvs.mean()*1000:.6f} g/kg (mean)")
        print(f"  Cloud water: {rcs.mean()*1000:.6f} g/kg (mean)")
        print(f"  Rain: {rrs.mean()*1000:.6f} g/kg (mean)")
        print(f"  Cloud ice: {ris.mean()*1000:.6f} g/kg (mean)")
        print(f"  Snow: {rss.mean()*1000:.6f} g/kg (mean)")
        print(f"  Graupel: {rgs.mean()*1000:.6f} g/kg (mean)")
        print()
        
        print("="*70)
        
        return {
            'ths': ths, 'rvs': rvs, 'rcs': rcs, 'rrs': rrs,
            'ris': ris, 'rss': rss, 'rgs': rgs,
            'evap3d': evap3d, 'rainfr': rainfr,
            'inprc': inprc, 'inprr': inprr, 'inprs': inprs,
            'inprg': inprg, 'indep': indep
        }
        
    except Exception as e:
        print(f"✗ RAIN_ICE failed with error: {e}")
        print()
        print("Note: RAIN_ICE may require proper initialization of ICEP and ICED structures.")
        print("This is a known issue being investigated.")
        print("="*70)
        return None


if __name__ == "__main__":
    results = example_rain_ice()
