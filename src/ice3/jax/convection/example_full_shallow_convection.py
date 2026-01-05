"""
Complete example of shallow convection (part 1 + part 2).

This script demonstrates the full shallow convection workflow:
1. Part 1: Compute thermodynamics and trigger convection
2. Part 2: Compute updraft properties, closure, and tendencies
"""
import jax.numpy as jnp
from shallow_convection_part1 import shallow_convection_part1, ConvectionParameters
from shallow_convection_part2 import shallow_convection_part2


def main():
    """Run complete shallow convection example."""

    print("=" * 70)
    print("COMPLETE SHALLOW CONVECTION EXAMPLE")
    print("=" * 70)

    # Set up dimensions
    nit = 100  # horizontal points
    nkt = 60   # vertical levels
    kch1 = 5   # number of chemical species

    # Convection parameters (using defaults)
    convection_params = ConvectionParameters()

    # Extra vertical levels
    jcvexb = 0  # Extra levels on bottom
    jcvext = 0  # Extra levels on top

    # Create sample atmospheric profiles
    # Pressure decreases exponentially with height
    z = jnp.linspace(0, 15000, nkt)  # Height from 0 to 15 km
    p_1d = 100000.0 * jnp.exp(-z / 7000.0)  # Pressure profile
    ppabst = jnp.tile(p_1d, (nit, 1))  # Replicate for all horizontal points

    # Height array
    pzz = jnp.tile(z, (nit, 1))

    # Temperature profile (decreasing with height, ~6.5 K/km lapse rate)
    t_1d = 288.0 - 0.0065 * z
    ptt = jnp.tile(t_1d, (nit, 1))

    # Water vapor mixing ratio (decreasing with height)
    rv_1d = 0.01 * jnp.exp(-z / 2000.0)
    prvt = jnp.tile(rv_1d, (nit, 1))

    # Cloud water and ice (small values)
    prct = 0.0001 * jnp.ones((nit, nkt))
    prit = 0.00001 * jnp.ones((nit, nkt))

    # Vertical velocity (small updraft)
    pwt = 0.1 * jnp.ones((nit, nkt))

    # TKE in surface layer
    ptkecls = 0.5 * jnp.ones(nit)

    # Initialize tendency arrays (will be reset in the function)
    ptten = jnp.zeros((nit, nkt))
    prvten = jnp.zeros((nit, nkt))
    prcten = jnp.zeros((nit, nkt))
    priten = jnp.zeros((nit, nkt))

    # Initialize cloud indices
    kcltop = jnp.zeros(nit, dtype=jnp.int32)
    kclbas = jnp.zeros(nit, dtype=jnp.int32)

    # Updraft mass flux
    pumf = jnp.zeros((nit, nkt))

    # Chemical species (optional)
    pch1 = jnp.zeros((nit, nkt, kch1))
    pch1ten = jnp.zeros((nit, nkt, kch1))

    # ===== PART 1: Trigger and prepare =====
    print("\nRunning Part 1: Thermodynamics and Trigger")
    print("-" * 70)

    part1_outputs = shallow_convection_part1(
        ppabst=ppabst,
        pzz=pzz,
        ptkecls=ptkecls,
        ptt=ptt,
        prvt=prvt,
        prct=prct,
        prit=prit,
        pwt=pwt,
        ptten=ptten,
        prvten=prvten,
        prcten=prcten,
        priten=priten,
        kcltop=kcltop,
        kclbas=kclbas,
        pumf=pumf,
        pch1=pch1,
        pch1ten=pch1ten,
        jcvexb=jcvexb,
        jcvext=jcvext,
        convection_params=convection_params,
        och1conv=False,
    )

    # Print part 1 results
    print(f"Potential temperature range: {part1_outputs.ptht.min():.2f} - {part1_outputs.ptht.max():.2f} K")
    print(f"Virtual potential temperature range: {part1_outputs.psthv.min():.2f} - {part1_outputs.psthv.max():.2f} K")
    print(f"Equivalent potential temperature range: {part1_outputs.psthes.min():.2f} - {part1_outputs.psthes.max():.2f} K")
    print(f"Number of triggered columns: {part1_outputs.otrig1.sum()}")
    print(f"LCL height range: {part1_outputs.pszlcl.min():.1f} - {part1_outputs.pszlcl.max():.1f} m")
    print(f"DPL indices range: {part1_outputs.ksdpl.min()} - {part1_outputs.ksdpl.max()}")

    # ===== PART 2: Updraft and closure =====
    print("\nRunning Part 2: Updraft, Closure, and Tendencies")
    print("-" * 70)

    # Compute Rd/Cp
    from constants import PHYS_CONSTANTS
    cst = PHYS_CONSTANTS
    prdocp = cst.rd / cst.cpd

    part2_outputs = shallow_convection_part2(
        ppabst=ppabst,
        pzz=pzz,
        ptt=ptt,
        prvt=prvt,
        prct=prct,
        prit=prit,
        pch1=pch1,
        prdocp=prdocp,
        ptht=part1_outputs.ptht,
        psthv=part1_outputs.psthv,
        psthes=part1_outputs.psthes,
        isdpl=part1_outputs.ksdpl,
        ispbl=part1_outputs.kspbl,
        islcl=part1_outputs.kslcl,
        psthlcl=part1_outputs.psthlcl,
        pstlcl=part1_outputs.pstlcl,
        psrvlcl=part1_outputs.psrvlcl,
        pswlcl=part1_outputs.pswlcl,
        pszlcl=part1_outputs.pszlcl,
        psthvelcl=part1_outputs.psthvelcl,
        gtrig1=part1_outputs.otrig1,
        kice=1,
        jcvexb=jcvexb,
        jcvext=jcvext,
        convection_params=convection_params,
        osettadj=False,
        ptadjs=10800.0,
        och1conv=False,
    )

    # Print part 2 results
    print(f"Cloud top level range: {part2_outputs.ictl.min()} - {part2_outputs.ictl.max()}")
    print(f"Mass flux range: {part2_outputs.pumf.min():.6f} - {part2_outputs.pumf.max():.6f} kg/(s·m²)")
    print(f"Temperature tendency range: {part2_outputs.pthc.min():.8f} - {part2_outputs.pthc.max():.8f} K/s")
    print(f"Water vapor tendency range: {part2_outputs.prvc.min():.10f} - {part2_outputs.prvc.max():.10f} 1/s")
    print(f"Cloud water tendency range: {part2_outputs.prcc.min():.10f} - {part2_outputs.prcc.max():.10f} 1/s")
    print(f"Ice tendency range: {part2_outputs.pric.min():.10f} - {part2_outputs.pric.max():.10f} 1/s")

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_triggered = part1_outputs.otrig1.sum()
    n_cloudy = (part2_outputs.ictl > 0).sum()
    print(f"Grid points: {nit}")
    print(f"Vertical levels: {nkt}")
    print(f"Triggered convection: {n_triggered} points ({100*n_triggered/nit:.1f}%)")
    print(f"Cloudy points (CTL > 0): {n_cloudy} points ({100*n_cloudy/nit:.1f}%)")

    if n_cloudy > 0:
        # Find a cloudy column for detailed output
        cloudy_idx = jnp.where(part2_outputs.ictl > 0)[0][0]
        print(f"\nExample column {cloudy_idx}:")
        print(f"  LCL level: {part1_outputs.kslcl[cloudy_idx]}")
        print(f"  LCL height: {part1_outputs.pszlcl[cloudy_idx]:.1f} m")
        print(f"  Cloud top level: {part2_outputs.ictl[cloudy_idx]}")
        print(f"  DPL level: {part1_outputs.ksdpl[cloudy_idx]}")
        print(f"  Max mass flux: {part2_outputs.pumf[cloudy_idx, :].max():.6f} kg/(s·m²)")

    print("\nShallow convection computation complete!")
    print("=" * 70)

    return part1_outputs, part2_outputs


if __name__ == "__main__":
    part1_out, part2_out = main()
