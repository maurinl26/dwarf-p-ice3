#!/usr/bin/env python3
"""
Simple example demonstrating the GT4Py ice adjustment scheme.

This example creates a simple atmospheric column with supersaturated conditions
and demonstrates how the ice adjustment scheme handles condensation and cloud formation.
"""

import numpy as np
import matplotlib.pyplot as plt
import gt4py.next as gtx

from ice3.stencils.ice_adjust import ice_adjust
from ice3.common.constants import CST
from ice3.common.dimensions import DimPhyex


def create_atmospheric_profile(nk=50):
    """
    Create a realistic atmospheric profile for testing.
    
    Args:
        nk: Number of vertical levels
        
    Returns:
        Dictionary containing atmospheric fields
    """
    # Vertical coordinate (height in meters)
    height = np.linspace(0, 15000, nk)  # 0 to 15 km
    
    # Standard atmosphere pressure profile
    pressure_surface = 101325.0  # Pa
    scale_height = 8000.0  # m
    pressure = pressure_surface * np.exp(-height / scale_height)
    
    # Temperature profile with standard lapse rate
    temperature_surface = 288.0  # K
    lapse_rate = 0.0065  # K/m
    temperature = temperature_surface - lapse_rate * height
    
    # Exner function
    exner = (pressure / CST.XP00) ** (CST.XRD / CST.XCPD)
    
    # Potential temperature
    theta = temperature / exner
    
    # Reference density
    rho_ref = pressure / (CST.XRD * temperature)
    
    # Water vapor mixing ratio (decreases with height)
    # Create supersaturated conditions in the middle troposphere
    rv = np.zeros_like(height)
    for i, h in enumerate(height):
        if h < 2000:
            rv[i] = 0.012 * np.exp(-h / 2000)  # High humidity near surface
        elif 2000 <= h <= 8000:
            rv[i] = 0.008 * np.exp(-(h - 2000) / 3000)  # Supersaturated layer
        else:
            rv[i] = 0.001 * np.exp(-(h - 8000) / 5000)  # Dry upper levels
    
    # Initial cloud water and ice (small background values)
    rc = np.full_like(height, 1.0e-6)
    ri = np.full_like(height, 1.0e-6)
    
    # Precipitation species (minimal values)
    rr = np.full_like(height, 1.0e-8)
    rs = np.full_like(height, 1.0e-8)
    rg = np.full_like(height, 1.0e-8)
    
    return {
        'height': height,
        'pressure': pressure,
        'temperature': temperature,
        'theta': theta,
        'exner': exner,
        'rho_ref': rho_ref,
        'rv': rv,
        'rc': rc,
        'ri': ri,
        'rr': rr,
        'rs': rs,
        'rg': rg
    }


def run_ice_adjust_example():
    """Run the ice adjustment example."""
    print("GT4Py Ice Adjustment Example")
    print("=" * 40)
    
    # Create domain (single column)
    dim = DimPhyex.create_simple(ni=1, nj=1, nk=50)
    print(f"Domain shape: {dim.get_total_shape()}")
    
    # Create atmospheric profile
    profile = create_atmospheric_profile(nk=50)
    
    # Expand to 3D arrays for GT4Py
    shape = dim.get_total_shape()
    fields = {}
    for name, data in profile.items():
        # Expand 1D profile to 3D
        array_3d = np.broadcast_to(data[np.newaxis, np.newaxis, :], shape)
        fields[name] = gtx.as_field([gtx.Dims.I, gtx.Dims.J, gtx.Dims.K], array_3d)
    
    # Initialize source terms to zero
    rv_source = gtx.as_field([gtx.Dims.I, gtx.Dims.J, gtx.Dims.K], np.zeros(shape))
    rc_source = gtx.as_field([gtx.Dims.I, gtx.Dims.J, gtx.Dims.K], np.zeros(shape))
    ri_source = gtx.as_field([gtx.Dims.I, gtx.Dims.J, gtx.Dims.K], np.zeros(shape))
    theta_source = gtx.as_field([gtx.Dims.I, gtx.Dims.J, gtx.Dims.K], np.zeros(shape))
    
    print("\nRunning ice adjustment scheme...")
    
    # Run ice adjustment with Gaussian condensation scheme
    result = ice_adjust(
        dim=dim,
        pressure=fields['pressure'],
        height=fields['height'],
        rho_dry=fields['rho_ref'],
        exner=fields['exner'],
        exner_ref=fields['exner'],
        rho_ref=fields['rho_ref'],
        theta=fields['theta'],
        rv=fields['rv'],
        rc=fields['rc'],
        ri=fields['ri'],
        rr=fields['rr'],
        rs=fields['rs'],
        rg=fields['rg'],
        rv_source=rv_source,
        rc_source=rc_source,
        ri_source=ri_source,
        theta_source=theta_source,
        timestep=300.0,  # 5 minutes
        use_subgrid_condensation=False,  # All-or-nothing for simplicity
        hcondens="GAUS",
        hfrac_ice="T"
    )
    
    # Unpack results
    (rv_source_out, rc_source_out, ri_source_out, theta_source_out,
     cloud_fraction, ice_cloud_fraction, water_cloud_fraction) = result
    
    # Convert back to numpy arrays for analysis
    height_1d = profile['height']
    rv_src = gtx.as_array(rv_source_out)[0, 0, :]
    rc_src = gtx.as_array(rc_source_out)[0, 0, :]
    ri_src = gtx.as_array(ri_source_out)[0, 0, :]
    theta_src = gtx.as_array(theta_source_out)[0, 0, :]
    cf = gtx.as_array(cloud_fraction)[0, 0, :]
    
    # Print summary statistics
    print(f"\nResults Summary:")
    print(f"Max condensation rate (rc): {np.max(rc_src):.2e} kg/kg/s")
    print(f"Max deposition rate (ri): {np.max(ri_src):.2e} kg/kg/s")
    print(f"Max heating rate: {np.max(theta_src):.2e} K/s")
    print(f"Max cloud fraction: {np.max(cf):.3f}")
    print(f"Number of cloudy levels: {np.sum(cf > 0.01)}")
    
    # Check conservation
    total_water_source = rv_src + rc_src + ri_src
    max_conservation_error = np.max(np.abs(total_water_source))
    print(f"Water conservation error: {max_conservation_error:.2e} kg/kg/s")
    
    # Create plots
    create_plots(height_1d, profile, rv_src, rc_src, ri_src, theta_src, cf)
    
    print("\nExample completed successfully!")
    return result


def create_plots(height, profile, rv_src, rc_src, ri_src, theta_src, cf):
    """Create visualization plots."""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('GT4Py Ice Adjustment Results', fontsize=16)
        
        # Plot 1: Initial water vapor profile
        axes[0, 0].plot(profile['rv'] * 1000, height / 1000, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Water Vapor (g/kg)')
        axes[0, 0].set_ylabel('Height (km)')
        axes[0, 0].set_title('Initial Water Vapor')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Temperature profile
        axes[0, 1].plot(profile['temperature'] - 273.15, height / 1000, 'r-', linewidth=2)
        axes[0, 1].axvline(0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Temperature (°C)')
        axes[0, 1].set_ylabel('Height (km)')
        axes[0, 1].set_title('Temperature Profile')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Source terms
        axes[0, 2].plot(rv_src * 1000, height / 1000, 'b-', label='Water Vapor', linewidth=2)
        axes[0, 2].plot(rc_src * 1000, height / 1000, 'g-', label='Cloud Water', linewidth=2)
        axes[0, 2].plot(ri_src * 1000, height / 1000, 'c-', label='Cloud Ice', linewidth=2)
        axes[0, 2].axvline(0, color='k', linestyle='--', alpha=0.5)
        axes[0, 2].set_xlabel('Source Terms (g/kg/s)')
        axes[0, 2].set_ylabel('Height (km)')
        axes[0, 2].set_title('Microphysical Sources')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Heating rate
        axes[1, 0].plot(theta_src * 3600, height / 1000, 'r-', linewidth=2)
        axes[1, 0].axvline(0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Heating Rate (K/h)')
        axes[1, 0].set_ylabel('Height (km)')
        axes[1, 0].set_title('Latent Heating')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Cloud fraction
        axes[1, 1].plot(cf, height / 1000, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Cloud Fraction')
        axes[1, 1].set_ylabel('Height (km)')
        axes[1, 1].set_title('Cloud Fraction')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Conservation check
        total_source = rv_src + rc_src + ri_src
        axes[1, 2].plot(total_source * 1e6, height / 1000, 'k-', linewidth=2)
        axes[1, 2].axvline(0, color='r', linestyle='--', alpha=0.5)
        axes[1, 2].set_xlabel('Total Water Source (mg/kg/s)')
        axes[1, 2].set_ylabel('Height (km)')
        axes[1, 2].set_title('Water Conservation')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ice_adjust_example.png', dpi=150, bbox_inches='tight')
        print("Plots saved as 'ice_adjust_example.png'")
        
    except ImportError:
        print("Matplotlib not available - skipping plots")


if __name__ == "__main__":
    # Run the example
    result = run_ice_adjust_example()
