"""
Test script for the shallow_convection Fortran wrapper.

This test validates the Fortran SHALLOW_CONVECTION implementation through
the Cython/Fortran bridge using reproduction data.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

# Add the bridge directory to the path
build_dir = Path(__file__).parent.parent.parent / 'build'
if build_dir.exists():
    for sub in build_dir.iterdir():
        if sub.is_dir() and sub.name.startswith('cp'):
            sys.path.insert(0, str(sub))
            break

try:
    from ice3._phyex_wrapper import shallow_convection
except ImportError:
    # Fallback to local import if building in-place
    bridge_dir = Path(__file__).parent.parent.parent / "PHYEX-IAL_CY50T1" / "bridge"
    sys.path.insert(0, str(bridge_dir))
    try:
        from _phyex_wrapper import shallow_convection
    except ImportError:
        shallow_convection = None


def test_shallow_convection_wrapper_basic():
    """Test the Cython wrapper for SHALLOW_CONVECTION with simple data."""
    print("\n" + "="*70)
    print("Testing Fortran SHALLOW_CONVECTION Wrapper - Basic")
    print("="*70)

    if shallow_convection is None:
        pytest.skip("Cython wrapper not available")

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
        print("✓ Shallow convection completed successfully")
    except Exception as e:
        print(f"✗ Error calling shallow_convection: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"shallow_convection call failed: {e}")

    # Verify results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    n_triggered = (kcltop > 0).sum()
    print(f"\nGrid points: {nlon}")
    print(f"Vertical levels: {nlev}")
    print(f"Columns with convection: {n_triggered} ({100*n_triggered/nlon:.1f}%)")

    # Check output arrays have expected shapes
    assert ptten.shape == (nlon, nlev)
    assert prvten.shape == (nlon, nlev)
    assert prcten.shape == (nlon, nlev)
    assert priten.shape == (nlon, nlev)
    assert kcltop.shape == (nlon,)
    assert kclbas.shape == (nlon,)
    assert pumf.shape == (nlon, nlev)

    # Physical checks
    assert np.all(np.isfinite(ptten)), "Non-finite values in temperature tendency"
    assert np.all(np.isfinite(prvten)), "Non-finite values in vapor tendency"
    assert np.all(np.isfinite(pumf)), "Non-finite values in mass flux"

    print("\n✓ All basic checks passed")


def test_shallow_convection_with_repro_data(shallow_convection_repro_ds):
    """
    Test Fortran SHALLOW_CONVECTION with reproduction dataset.

    Parameters
    ----------
    shallow_convection_repro_ds : xr.Dataset
        Reference dataset from shallow_convection.nc (or shallow.nc) fixture
    """
    print("\n" + "="*70)
    print("TEST: Fortran SHALLOW_CONVECTION with Reproduction Data")
    print("="*70)

    if shallow_convection is None:
        pytest.skip("Cython wrapper not available")

    try:
        from numpy.testing import assert_allclose

        # Get dataset dimensions
        # The shallow.nc file has dimensions: time, dim_2, points_1500, dim_100, points_9000, dim_50
        # We need to understand the data structure
        ds = shallow_convection_repro_ds

        print(f"\nDataset dimensions: {dict(ds.sizes)}")
        print(f"Available variables: {list(ds.data_vars.keys())[:10]}...")

        # NOTE: Since the data format is not fully documented yet,
        # we'll load a time slice and extract dimensions from the data
        # This is a placeholder that should be updated when data structure is clarified

        # For now, let's assume a simple structure
        # You may need to adjust this based on actual data layout
        time_idx = 0

        # Try to infer dimensions from the data
        # Most variables appear to be (time, points) format
        # Let's assume points_1500 = nlon * nlev format or similar

        # This is a placeholder - adjust based on actual data documentation
        print("\n⚠️  WARNING: Data loading needs to be customized based on")
        print("   actual shallow_convection.nc variable mapping")
        print("   Currently using placeholder structure")

        # Placeholder: Skip detailed comparison until data structure is documented
        pytest.skip("Shallow convection data structure needs documentation - see test for placeholder")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Test failed: {e}")


if __name__ == "__main__":
    # Run basic test without pytest
    test_shallow_convection_wrapper_basic()
    print("\nNote: run with pytest to include repro data test.")
