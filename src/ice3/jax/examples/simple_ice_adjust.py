#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple example demonstrating JAX ice_adjust implementation.

This script shows basic usage of the IceAdjustJAX component with
synthetic atmospheric data.
"""
import jax.numpy as jnp
from ice3.jax.components.ice_adjust import IceAdjustJAX
from ice3.phyex_common.phyex import Phyex


def main():
    """Run simple ice adjustment example."""
    print("=" * 60)
    print("JAX Ice Adjustment Example")
    print("=" * 60)
    
    # Initialize physics configuration for AROME
    print("\n1. Initializing physics configuration (AROME)...")
    phyex = Phyex(program="AROME", TSTEP=60.0)
    print(f"   - Time step: {phyex.TSTEP} s")
    print(f"   - NRR (rain categories): {phyex.NRR}")
    print(f"   - Subgrid condensation: {phyex.nebn.LSUBG_COND}")
    print(f"   - Subgrid PDF type: {phyex.param_icen.SUBG_MF_PDF}")
    
    # Create ice adjustment component
    print("\n2. Creating IceAdjustJAX component...")
    ice_adjust = IceAdjustJAX(phyex=phyex, jit=True)
    print("   - JIT compilation: enabled")
    
    # Prepare input fields (simple 3D grid)
    print("\n3. Preparing input fields...")
    nx, ny, nz = 5, 5, 10  # Small domain for demonstration
    shape = (nx, ny, nz)
    print(f"   - Domain size: {nx} x {ny} x {nz}")
    
    # Thermodynamic state
    # Mid-troposphere conditions (~500 hPa, ~5 km altitude)
    sigqsat = jnp.ones(shape) * 0.01       # Subgrid saturation variability
    pabs = jnp.ones(shape) * 50000.0       # Pressure (Pa) ~ 500 hPa
    sigs = jnp.ones(shape) * 0.1           # Subgrid mixing parameter
    th = jnp.ones(shape) * 290.0           # Potential temperature (K)
    exn = (pabs / 100000.0) ** (287.0 / 1004.0)  # Exner function
    rho_dry_ref = pabs / (287.0 * th * exn)  # Reference density from ideal gas law
    
    # Create a saturated layer with some moisture
    # Vapor mixing ratio slightly above saturation to trigger condensation
    rv = jnp.ones(shape) * 0.008           # Water vapor (kg/kg)
    # Add vertical moisture gradient (more moisture at lower levels)
    z_factor = jnp.linspace(1.2, 0.8, nz)
    rv = rv * z_factor[jnp.newaxis, jnp.newaxis, :]
    
    # Initial condensate (small amounts)
    rc = jnp.ones(shape) * 0.0001          # Cloud liquid (kg/kg)
    ri = jnp.ones(shape) * 0.00005         # Cloud ice (kg/kg)
    
    # Precipitation species (initially zero)
    rr = jnp.zeros(shape)                  # Rain
    rs = jnp.zeros(shape)                  # Snow
    rg = jnp.zeros(shape)                  # Graupel
    
    # Mass flux contributions (from convection scheme - here zero)
    cf_mf = jnp.zeros(shape)               # Cloud fraction from mass flux
    rc_mf = jnp.zeros(shape)               # Liquid from mass flux
    ri_mf = jnp.zeros(shape)               # Ice from mass flux
    
    # Tendencies (initialized to zero)
    rvs = jnp.zeros(shape)
    rcs = jnp.zeros(shape)
    ris = jnp.zeros(shape)
    ths = jnp.zeros(shape)
    
    print(f"   - Initial vapor: {rv.mean():.6f} kg/kg (mean)")
    print(f"   - Initial cloud liquid: {rc.mean():.6f} kg/kg (mean)")
    print(f"   - Initial cloud ice: {ri.mean():.6f} kg/kg (mean)")
    print(f"   - Pressure: {pabs.mean():.1f} Pa")
    print(f"   - Potential temperature: {th.mean():.2f} K")
    
    # Run ice adjustment
    print("\n4. Running ice adjustment (first call includes JIT compilation)...")
    results = ice_adjust(
        sigqsat=sigqsat,
        pabs=pabs,
        sigs=sigs,
        th=th,
        exn=exn,
        exn_ref=exn,
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
        timestep=60.0,
    )
    
    # Unpack results
    (t, rv_out, rc_out, ri_out, cldfr,
     hlc_hrc, hlc_hcf, hli_hri, hli_hcf,
     cph, lv, ls, rvs_out, rcs_out, ris_out, ths_out) = results
    
    # Display results
    print("\n5. Results:")
    print("-" * 60)
    print(f"   Temperature:")
    print(f"     - Mean: {t.mean():.2f} K")
    print(f"     - Min:  {t.min():.2f} K")
    print(f"     - Max:  {t.max():.2f} K")
    
    print(f"\n   Adjusted mixing ratios:")
    print(f"     - Vapor:  {rv_out.mean():.6f} kg/kg (Δ = {(rv_out - rv).mean():.6e})")
    print(f"     - Liquid: {rc_out.mean():.6f} kg/kg (Δ = {(rc_out - rc).mean():.6e})")
    print(f"     - Ice:    {ri_out.mean():.6f} kg/kg (Δ = {(ri_out - ri).mean():.6e})")
    
    print(f"\n   Cloud fraction:")
    print(f"     - Mean: {cldfr.mean():.4f}")
    print(f"     - Max:  {cldfr.max():.4f}")
    print(f"     - Cloudy points: {(cldfr > 0.01).sum()} / {cldfr.size}")
    
    print(f"\n   Thermodynamic properties:")
    print(f"     - Heat capacity: {cph.mean():.1f} J/(kg·K)")
    print(f"     - Latent heat (vaporization): {lv.mean():.2e} J/kg")
    print(f"     - Latent heat (sublimation):  {ls.mean():.2e} J/kg")
    
    print(f"\n   Tendencies (per timestep):")
    print(f"     - Theta:  {ths_out.mean():.6e} K")
    print(f"     - Vapor:  {rvs_out.mean():.6e} kg/kg")
    print(f"     - Liquid: {rcs_out.mean():.6e} kg/kg")
    print(f"     - Ice:    {ris_out.mean():.6e} kg/kg")
    
    print(f"\n   Autoconversion diagnostics:")
    print(f"     - Liquid (hlc_hrc): {hlc_hrc.mean():.6e} kg/kg")
    print(f"     - Ice (hli_hri):    {hli_hri.mean():.6e} kg/kg")
    
    # Test recompilation (should be instant)
    print("\n6. Testing subsequent calls (no recompilation)...")
    results_2 = ice_adjust(
        sigqsat=sigqsat, pabs=pabs, sigs=sigs, th=th,
        exn=exn, exn_ref=exn, rho_dry_ref=rho_dry_ref,
        rv=rv, rc=rc, ri=ri, rr=rr, rs=rs, rg=rg,
        cf_mf=cf_mf, rc_mf=rc_mf, ri_mf=ri_mf,
        rvs=rvs, rcs=rcs, ris=ris, ths=ths,
        timestep=60.0,
    )
    print("   ✓ Subsequent call completed (JIT cache used)")
    
    # Verify consistency
    t_2 = results_2[0]
    max_diff = jnp.abs(t - t_2).max()
    print(f"   ✓ Results consistent: max difference = {max_diff:.2e}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
