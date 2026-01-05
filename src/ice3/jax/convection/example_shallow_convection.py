"""
Example usage of JAX shallow convection part 1.

This script demonstrates how to use the shallow_convection_part1 function
with typical atmospheric data.
"""
import jax.numpy as jnp
from shallow_convection_part1 import shallow_convection_part1, ConvectionParameters


def main():
    """Run a simple example of shallow convection part 1."""

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

    # Run shallow convection part 1
    outputs = shallow_convection_part1(
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

    # Print some results
    print("Shallow Convection Part 1 - Results")
    print("=" * 50)
    print(f"Potential temperature range: {outputs.ptht.min():.2f} - {outputs.ptht.max():.2f} K")
    print(f"Virtual potential temperature range: {outputs.psthv.min():.2f} - {outputs.psthv.max():.2f} K")
    print(f"Equivalent potential temperature range: {outputs.psthes.min():.2f} - {outputs.psthes.max():.2f} K")
    print(f"Number of triggered columns: {outputs.otrig1.sum()}")
    print(f"DPL indices range: {outputs.ksdpl.min()} - {outputs.ksdpl.max()}")
    print(f"PBL indices range: {outputs.kspbl.min()} - {outputs.kspbl.max()}")
    print(f"LCL indices range: {outputs.kslcl.min()} - {outputs.kslcl.max()}")

    return outputs


if __name__ == "__main__":
    outputs = main()
