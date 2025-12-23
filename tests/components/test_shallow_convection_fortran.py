"""
Test script for the shallow_convection Cython wrapper.

This script demonstrates how to use the PHYEX shallow convection scheme
from Python through the Cython/Fortran bridge.

Before running this script, you need to compile the Cython extension:
    cd PHYEX-IAL_CY50T1/bridge
    python setup.py build_ext --inplace
"""
import sys
from pathlib import Path
import numpy as np

# Add the bridge directory to the path so we can import _phyex_wrapper
bridge_dir = Path(__file__).parent.parent.parent / "PHYEX-IAL_CY50T1" / "bridge"
sys.path.insert(0, str(bridge_dir))

# Import the Cython wrapper
try:
    from _phyex_wrapper import shallow_convection
except ImportError as e:
    print("Error importing _phyex_wrapper:")
    print(e)
    print("\nYou need to compile the Cython extension first:")
    print(f"    cd {bridge_dir}")
    print("    python setup.py build_ext --inplace")
    exit(1)


def main():
    """Test the shallow_convection wrapper with sample atmospheric data."""

    print("=" * 70)
    print("TESTING PHYEX SHALLOW_CONVECTION CYTHON WRAPPER")
    print("=" * 70)

    # Set up dimensions
    nlon = 100  # horizontal grid points
    nlev = 60   # vertical levels
    kch1 = 1    # number of chemical species (minimal)

    print(f"\nDimensions:")
    print(f"  Horizontal points: {nlon}")
    print(f"  Vertical levels: {nlev}")
    print(f"  Chemical species: {kch1}")

    # Create sample atmospheric profiles (Fortran order, single precision)
    print("\nInitializing atmospheric profiles...")

    # Height array (0 to 15 km)
    z = np.linspace(0, 15000, nlev, dtype=np.float32)
    pzz = np.tile(z, (nlon, 1)).astype(np.float32, order='F')

    # Pressure profile (exponential decrease with height)
    p_1d = 100000.0 * np.exp(-z / 7000.0)
    ppabst = np.tile(p_1d, (nlon, 1)).astype(np.float32, order='F')

    # Temperature profile (decreasing with height)
    t_1d = 288.0 - 0.0065 * z
    ptt = np.tile(t_1d, (nlon, 1)).astype(np.float32, order='F')

    # Water vapor mixing ratio (exponential decrease)
    rv_1d = 0.01 * np.exp(-z / 2000.0)
    prvt = np.tile(rv_1d, (nlon, 1)).astype(np.float32, order='F')

    # Cloud water and ice (small constant values)
    prct = np.ones((nlon, nlev), dtype=np.float32, order='F') * 0.0001
    prit = np.ones((nlon, nlev), dtype=np.float32, order='F') * 0.00001

    # Vertical velocity (small positive value)
    pwt = np.ones((nlon, nlev), dtype=np.float32, order='F') * 0.1

    # TKE in surface layer
    ptkecls = np.ones(nlon, dtype=np.float32, order='F') * 0.5

    # Initialize output arrays (these will be overwritten)
    ptten = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    prvten = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    prcten = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    priten = np.zeros((nlon, nlev), dtype=np.float32, order='F')
    kcltop = np.zeros(nlon, dtype=np.int32, order='F')
    kclbas = np.zeros(nlon, dtype=np.int32, order='F')
    pumf = np.zeros((nlon, nlev), dtype=np.float32, order='F')

    # Chemical tracer arrays (3D)
    pch1 = np.zeros((nlon, nlev, kch1), dtype=np.float32, order='F')
    pch1ten = np.zeros((nlon, nlev, kch1), dtype=np.float32, order='F')

    # Convection parameters
    kice = 1            # Include ice
    kbdia = 1           # Start computations at level 1
    ktdia = 1           # End computations at top
    osettadj = False    # Use default adjustment time
    ptadjs = 10800.0    # Adjustment time (seconds) - only used if osettadj=True
    och1conv = False    # No chemical tracer transport

    print("\n" + "-" * 70)
    print("Calling shallow_convection...")
    print("-" * 70)

    # Call the shallow convection routine
    try:
        shallow_convection(
            kice=kice,
            kbdia=kbdia,
            ktdia=ktdia,
            osettadj=osettadj,
            ptadjs=ptadjs,
            och1conv=och1conv,
            kch1=kch1,
            ptkecls=ptkecls,
            ppabst=ppabst,
            pzz=pzz,
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
            pch1ten=pch1ten
        )
        print("Shallow convection completed successfully!")
    except Exception as e:
        print(f"Error calling shallow_convection: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    n_triggered = (kcltop > 0).sum()
    print(f"\nGrid points: {nlon}")
    print(f"Vertical levels: {nlev}")
    print(f"Columns with convection: {n_triggered} ({100*n_triggered/nlon:.1f}%)")

    if n_triggered > 0:
        print(f"\nCloud top levels range: {kcltop.min()} - {kcltop.max()}")
        print(f"Cloud base levels range: {kclbas.min()} - {kclbas.max()}")
        print(f"Mass flux range: {pumf.min():.6f} - {pumf.max():.6f} kg/(s·m²)")

        print(f"\nTendency ranges:")
        print(f"  Temperature: {ptten.min():.8f} - {ptten.max():.8f} K/s")
        print(f"  Water vapor: {prvten.min():.10f} - {prvten.max():.10f} 1/s")
        print(f"  Cloud water: {prcten.min():.10f} - {prcten.max():.10f} 1/s")
        print(f"  Ice: {priten.min():.10f} - {priten.max():.10f} 1/s")

        # Find a convective column
        conv_idx = np.where(kcltop > 0)[0][0]
        print(f"\nExample convective column {conv_idx}:")
        print(f"  Cloud top level: {kcltop[conv_idx]}")
        print(f"  Cloud base level: {kclbas[conv_idx]}")
        print(f"  Max mass flux: {pumf[conv_idx, :].max():.6f} kg/(s·m²)")
        print(f"  Max temperature tendency: {ptten[conv_idx, :].max():.8f} K/s")
    else:
        print("\nNo convection was triggered in any column.")
        print("This may be expected with the simple test profile.")

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
