"""
Example demonstrating IceAdjustJAX with real atmospheric test data.

This script shows how to use the IceAdjustJAX component with actual
atmospheric data from the PHYEX test dataset (ice_adjust.nc).
"""
import jax.numpy as jnp
import numpy as np
import xarray as xr
from pathlib import Path

from ice3.jax.ice_adjust import IceAdjustJAX
from ice3.phyex_common.phyex import Phyex


def load_test_data():
    """Load real atmospheric test data from ice_adjust.nc."""
    data_path = Path(__file__).parent.parent / "data" / "ice_adjust.nc"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {data_path}\n"
            "Please ensure ice_adjust.nc is available in the data directory."
        )

    return xr.open_dataset(data_path)


def reshape_for_jax(var):
    """Reshape dataset variable from (ngpblks, nflevg, nproma) to (ngpblks, nproma, nflevg)."""
    return jnp.asarray(np.swapaxes(var, 1, 2))


def main():
    """Run IceAdjustJAX example with real test data."""

    print("=" * 70)
    print("ICE_ADJUST JAX EXAMPLE")
    print("Mixed-phase cloud saturation adjustment")
    print("=" * 70)

    # Load real atmospheric data
    print("\n1. Loading test data from ice_adjust.nc...")
    dataset = load_test_data()

    shape = (
        dataset.sizes["ngpblks"],
        dataset.sizes["nproma"],
        dataset.sizes["nflevg"]
    )

    print(f"   Dataset shape (ngpblks × nproma × nflevg): {shape[0]} × {shape[1]} × {shape[2]}")
    print(f"   Total grid points: {shape[0] * shape[1]}")
    print(f"   Vertical levels: {shape[2]}")

    # Initialize physics configuration
    print("\n2. Initializing PHYEX configuration and JAX component...")
    phyex = Phyex("AROME", TSTEP=50.0)
    ice_adjust = IceAdjustJAX(phyex=phyex, jit=True)
    print("   ✓ IceAdjustJAX initialized with JIT compilation")

    # Load and prepare input data
    print("\n3. Preparing input atmospheric state...")

    # Atmospheric state variables
    pabs = reshape_for_jax(dataset["PPABSM"].values)
    exn = reshape_for_jax(dataset["PEXNREF"].values)
    exn_ref = reshape_for_jax(dataset["PEXNREF"].values)
    rho_dry_ref = reshape_for_jax(dataset["PRHODREF"].values)

    # Load state from ZRS (ngpblks, krr1, nflevg, nproma)
    zrs = dataset["ZRS"].values
    zrs = np.swapaxes(zrs, 2, 3)  # → (ngpblks, krr1, nproma, nflevg)

    th = jnp.asarray(zrs[:, 0, :, :])  # potential temperature
    rv = jnp.asarray(zrs[:, 1, :, :])  # water vapor mixing ratio
    rc = jnp.asarray(zrs[:, 2, :, :])  # cloud water mixing ratio
    rr = jnp.asarray(zrs[:, 3, :, :])  # rain mixing ratio
    ri = jnp.asarray(zrs[:, 4, :, :])  # ice mixing ratio
    rs = jnp.asarray(zrs[:, 5, :, :])  # snow mixing ratio
    rg = jnp.asarray(zrs[:, 6, :, :])  # graupel mixing ratio

    # Load input tendencies from PRS (ngpblks, krr, nflevg, nproma)
    prs = dataset["PRS"].values
    prs = np.swapaxes(prs, 2, 3)  # → (ngpblks, krr, nproma, nflevg)

    rvs = jnp.asarray(prs[:, 0, :, :])
    rcs = jnp.asarray(prs[:, 1, :, :])
    ris = jnp.asarray(prs[:, 3, :, :])

    # Temperature tendency
    ths = reshape_for_jax(dataset["PTHS"].values)

    # Mass flux variables (from convection scheme)
    cf_mf = reshape_for_jax(dataset["PCF_MF"].values)
    rc_mf = reshape_for_jax(dataset["PRC_MF"].values)
    ri_mf = reshape_for_jax(dataset["PRI_MF"].values)

    # Subgrid variability parameters
    zsigqsat = dataset["ZSIGQSAT"].values
    sigqsat = jnp.asarray(zsigqsat[:, :, np.newaxis])  # Expand to 3D
    sigs = reshape_for_jax(dataset["PSIGS"].values)

    print("   ✓ Input data loaded")
    print(f"\n   Initial atmospheric state:")
    print(f"     Potential temperature: {float(th.min()):.1f} - {float(th.max()):.1f} K")
    print(f"     Pressure: {float(pabs.min())/100:.1f} - {float(pabs.max())/100:.1f} hPa")
    print(f"     Water vapor: {float(rv.min())*1000:.3f} - {float(rv.max())*1000:.3f} g/kg")
    print(f"     Cloud water: {float(rc.min())*1000:.3f} - {float(rc.max())*1000:.3f} g/kg")
    print(f"     Ice: {float(ri.min())*1000:.3f} - {float(ri.max())*1000:.3f} g/kg")

    # Run ICE_ADJUST
    print("\n4. Running ICE_ADJUST saturation adjustment...")
    timestep = 50.0  # seconds

    result = ice_adjust(
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
        timestep=timestep,
    )

    print("   ✓ ICE_ADJUST completed")

    # Extract results
    # Returns: t, rv_out, rc_out, ri_out, cldfr, hlc_hrc, hlc_hcf,
    #          hli_hri, hli_hcf, cph, lv, ls, rvs_out, rcs_out, ris_out, ths_out
    t_out = result[0]
    rv_out = result[1]
    rc_out = result[2]
    ri_out = result[3]
    cldfr = result[4]
    hlc_hrc = result[5]  # heating from cloud water condensation
    hlc_hcf = result[6]  # heating from cloud fraction
    hli_hri = result[7]  # heating from ice deposition
    hli_hcf = result[8]  # heating from ice cloud fraction
    cph = result[9]      # cloud phase indicator
    lv = result[10]      # latent heat of vaporization
    ls = result[11]      # latent heat of sublimation
    rvs_out = result[12]
    rcs_out = result[13]
    ris_out = result[14]
    ths_out = result[15]

    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\nAdjusted atmospheric state:")
    print(f"  Temperature: {float(t_out.min()):.1f} - {float(t_out.max()):.1f} K")
    print(f"  Water vapor: {float(rv_out.min())*1000:.3f} - {float(rv_out.max())*1000:.3f} g/kg")
    print(f"  Cloud water: {float(rc_out.min())*1000:.3f} - {float(rc_out.max())*1000:.3f} g/kg")
    print(f"  Ice: {float(ri_out.min())*1000:.3f} - {float(ri_out.max())*1000:.3f} g/kg")
    print(f"  Cloud fraction: {float(cldfr.min()):.3f} - {float(cldfr.max()):.3f}")

    print("\nPhysical tendencies (changes per timestep):")
    print(f"  Vapor tendency: {float(rvs_out.min())*1000:.6f} - {float(rvs_out.max())*1000:.6f} g/kg/s")
    print(f"  Cloud tendency: {float(rcs_out.min())*1000:.6f} - {float(rcs_out.max())*1000:.6f} g/kg/s")
    print(f"  Ice tendency: {float(ris_out.min())*1000:.6f} - {float(ris_out.max())*1000:.6f} g/kg/s")
    print(f"  Theta tendency: {float(ths_out.min()):.6f} - {float(ths_out.max()):.6f} K/s")

    print("\nLatent heating:")
    print(f"  Condensation heating: {float(hlc_hrc.min()):.6f} - {float(hlc_hrc.max()):.6f} K/s")
    print(f"  Deposition heating: {float(hli_hri.min()):.6f} - {float(hli_hri.max()):.6f} K/s")

    print("\nThermodynamic properties:")
    print(f"  Latent heat (vaporization): {float(lv.min()):.0f} - {float(lv.max()):.0f} J/kg")
    print(f"  Latent heat (sublimation): {float(ls.min()):.0f} - {float(ls.max()):.0f} J/kg")

    # Find cloudy points
    cloudy_points = (cldfr > 0.01).sum()
    total_points = shape[0] * shape[1] * shape[2]
    print(f"\nCloud statistics:")
    print(f"  Cloudy grid points: {cloudy_points} / {total_points} ({100*cloudy_points/total_points:.1f}%)")

    # Find a representative cloudy column
    cloudy_mask = cldfr > 0.1
    if cloudy_mask.any():
        # Get indices of first cloudy point
        cloudy_indices = jnp.where(cloudy_mask)
        i_block = int(cloudy_indices[0][0])
        i_point = int(cloudy_indices[1][0])
        i_level = int(cloudy_indices[2][0])

        print(f"\nExample cloudy point (block={i_block}, point={i_point}, level={i_level}):")
        print(f"  Cloud fraction: {float(cldfr[i_block, i_point, i_level]):.3f}")
        print(f"  Temperature: {float(t_out[i_block, i_point, i_level]):.1f} K")
        print(f"  Pressure: {float(pabs[i_block, i_point, i_level])/100:.1f} hPa")
        print(f"  Cloud water: {float(rc_out[i_block, i_point, i_level])*1000:.3f} g/kg")
        print(f"  Ice: {float(ri_out[i_block, i_point, i_level])*1000:.3f} g/kg")
        print(f"  Phase indicator: {float(cph[i_block, i_point, i_level]):.3f}")

    # Physical validation checks
    print("\n" + "=" * 70)
    print("PHYSICAL VALIDATION")
    print("=" * 70)

    # Check conservation
    total_water_in = rv + rc + ri
    total_water_out = rv_out + rc_out + ri_out
    water_conservation_error = jnp.abs(total_water_out - total_water_in).max()

    print(f"\nConservation checks:")
    print(f"  Total water conservation error: {float(water_conservation_error)*1000:.6e} g/kg")
    print(f"  {'✓' if water_conservation_error < 1e-10 else '✗'} Water is conserved")

    # Check physical bounds
    print(f"\nPhysical bounds:")
    print(f"  Cloud fraction in [0,1]: {'✓' if (cldfr >= 0).all() and (cldfr <= 1).all() else '✗'}")
    print(f"  Mixing ratios >= 0: {'✓' if (rv_out >= 0).all() and (rc_out >= 0).all() and (ri_out >= 0).all() else '✗'}")
    print(f"  Temperature > 0K: {'✓' if (t_out > 0).all() else '✗'}")

    print("\n" + "=" * 70)
    print("This example demonstrates:")
    print("  • Loading real atmospheric test data from NetCDF")
    print("  • Running mixed-phase saturation adjustment with IceAdjustJAX")
    print("  • Analyzing cloud formation and phase partitioning")
    print("  • Validating physical conservation laws")
    print("  • Computing latent heating from phase changes")
    print("=" * 70)

    return result


if __name__ == "__main__":
    result = main()
