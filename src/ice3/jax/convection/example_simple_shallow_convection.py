"""
Simplest example using the main shallow_convection wrapper.

This script demonstrates the easiest way to use the shallow convection scheme
with automatic optimization.
"""
import jax.numpy as jnp
from shallow_convection import shallow_convection, ConvectionParameters


def main():
    """Run simple shallow convection example with automatic optimization."""

    print("=" * 70)
    print("SIMPLE SHALLOW CONVECTION EXAMPLE")
    print("(Automatic optimization)")
    print("=" * 70)

    # Set up dimensions
    nit = 100  # horizontal points
    nkt = 60   # vertical levels
    kch1 = 1   # number of chemical species (minimal)

    # Create sample atmospheric profiles
    z = jnp.linspace(0, 15000, nkt)  # Height from 0 to 15 km
    p_1d = 100000.0 * jnp.exp(-z / 7000.0)  # Pressure profile
    ppabst = jnp.tile(p_1d, (nit, 1))

    # Height array
    pzz = jnp.tile(z, (nit, 1))

    # Temperature profile (decreasing with height)
    t_1d = 288.0 - 0.0065 * z
    ptt = jnp.tile(t_1d, (nit, 1))

    # Water vapor mixing ratio
    rv_1d = 0.01 * jnp.exp(-z / 2000.0)
    prvt = jnp.tile(rv_1d, (nit, 1))

    # Cloud water and ice
    prct = 0.0001 * jnp.ones((nit, nkt))
    prit = 0.00001 * jnp.ones((nit, nkt))

    # Vertical velocity
    pwt = 0.1 * jnp.ones((nit, nkt))

    # TKE in surface layer
    ptkecls = 0.5 * jnp.ones(nit)

    # Initialize arrays (these will be overwritten)
    ptten = jnp.zeros((nit, nkt))
    prvten = jnp.zeros((nit, nkt))
    prcten = jnp.zeros((nit, nkt))
    priten = jnp.zeros((nit, nkt))
    kcltop = jnp.zeros(nit, dtype=jnp.int32)
    kclbas = jnp.zeros(nit, dtype=jnp.int32)
    pumf = jnp.zeros((nit, nkt))
    pch1 = jnp.zeros((nit, nkt, kch1))
    pch1ten = jnp.zeros((nit, nkt, kch1))

    # ===== MAIN CALL: Just one function! =====
    print("\nRunning shallow convection (automatic optimization)...")
    print("-" * 70)

    outputs = shallow_convection(
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
        kice=1,  # Include ice
        convection_params=ConvectionParameters(),  # Use defaults
        osettadj=False,  # Use default adjustment time
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    n_triggered = (outputs.kcltop > 0).sum()
    print(f"Grid points: {nit}")
    print(f"Vertical levels: {nkt}")
    print(f"Columns with convection: {n_triggered} ({100*n_triggered/nit:.1f}%)")

    print(f"\nCloud top levels range: {outputs.kcltop.min()} - {outputs.kcltop.max()}")
    print(f"Mass flux range: {outputs.pumf.min():.6f} - {outputs.pumf.max():.6f} kg/(s·m²)")

    print(f"\nTendency ranges:")
    print(f"  Temperature: {outputs.ptten.min():.8f} - {outputs.ptten.max():.8f} K/s")
    print(f"  Water vapor: {outputs.prvten.min():.10f} - {outputs.prvten.max():.10f} 1/s")
    print(f"  Cloud water: {outputs.prcten.min():.10f} - {outputs.prcten.max():.10f} 1/s")
    print(f"  Ice: {outputs.priten.min():.10f} - {outputs.priten.max():.10f} 1/s")

    if n_triggered > 0:
        # Find a convective column
        conv_idx = jnp.where(outputs.kcltop > 0)[0][0]
        print(f"\nExample convective column {conv_idx}:")
        print(f"  Cloud top level: {outputs.kcltop[conv_idx]}")
        print(f"  Cloud base level: {outputs.kclbas[conv_idx]}")
        print(f"  Max mass flux: {outputs.pumf[conv_idx, :].max():.6f} kg/(s·m²)")
        print(f"  Max temperature tendency: {outputs.ptten[conv_idx, :].max():.8f} K/s")

    print("\n" + "=" * 70)
    print("ADVANTAGE: Single function call with automatic optimization!")
    print("The routine automatically chose the best computation method")
    print("based on the fraction of triggered columns.")
    print("=" * 70)

    return outputs


if __name__ == "__main__":
    result = main()
