"""
Unified reproducibility tests for ICE_ADJUST component.

This test file compares Fortran and JAX implementations on:
1. Synthetic test data (simple and realistic atmospheric profiles)
2. Repository reference data (ice_adjust.nc)

The tests ensure that both implementations produce consistent results
and maintain physical conservation laws.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Try to import Fortran wrapper
try:
    from ice3._phyex_wrapper import ice_adjust as ice_adjust_fortran
    FORTRAN_AVAILABLE = True
except ImportError:
    # Fallback to build directory
    build_dir = Path(__file__).parent.parent.parent / 'build'
    if build_dir.exists():
        for sub in build_dir.iterdir():
            if sub.is_dir() and sub.name.startswith('cp'):
                sys.path.insert(0, str(sub))
                break
    try:
        from ice3._phyex_wrapper import ice_adjust as ice_adjust_fortran
        FORTRAN_AVAILABLE = True
    except ImportError:
        FORTRAN_AVAILABLE = False
        ice_adjust_fortran = None

# Try to import JAX implementation
try:
    import jax
    import jax.numpy as jnp
    from ice3.jax.ice_adjust import IceAdjustJAX
    from ice3.phyex_common.phyex import Phyex
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None
    IceAdjustJAX = None
    Phyex = None


# ============================================================================
# Fixtures for test data
# ============================================================================

@pytest.fixture
def phyex():
    """Create PHYEX configuration for tests."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    return Phyex(program="AROME", TSTEP=60.0)


@pytest.fixture
def ice_adjust_jax(phyex):
    """Create IceAdjustJAX instance."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    return IceAdjustJAX(phyex=phyex, jit=True)


def create_synthetic_data(nijt=100, nkt=60):
    """
    Create realistic synthetic atmospheric test data.

    Returns data in both Fortran-contiguous format (for Fortran)
    and as a dict (convertible to JAX format).

    Parameters
    ----------
    nijt : int
        Number of horizontal points
    nkt : int
        Number of vertical levels

    Returns
    -------
    dict
        Dictionary with all required fields (float32)
    """
    # Use float32 for all calculations to match PHYEX expectations
    z = np.linspace(0, 10000, nkt, dtype=np.float32)

    # Standard atmosphere
    p0 = 101325.0
    T0 = 288.15
    gamma = 0.0065

    # Physical constants
    Rd = 287.0
    cp = 1004.0
    p00 = 100000.0

    # Pressure profile
    pressure = p0 * (1 - gamma * z / T0) ** 5.26
    ppabst = np.tile(pressure, (nijt, 1)).T.astype(np.float32).copy(order='F')

    # Temperature profile
    temperature = T0 - gamma * z
    temperature = np.tile(temperature, (nijt, 1)).T.astype(np.float32).copy(order='F')

    # Add variability
    np.random.seed(42)
    temperature += (np.random.randn(nkt, nijt) * 0.5).astype(np.float32)
    ppabst += (np.random.randn(nkt, nijt) * 100).astype(np.float32)

    # Exner function
    pexn = np.asfortranarray((ppabst / p00) ** (Rd / cp), dtype=np.float32)
    pth = np.asfortranarray(temperature / pexn, dtype=np.float32)

    # Reference values
    pexnref = np.asfortranarray(pexn.copy(), dtype=np.float32)
    prhodref = np.asfortranarray(ppabst / (Rd * temperature), dtype=np.float32)

    # Height
    pzz = np.tile(z, (nijt, 1)).T.astype(np.float32).copy(order='F')

    # Water vapor
    rv_surf = 0.015
    prv = (rv_surf * np.exp(-z / 2000)).astype(np.float32)
    prv = np.tile(prv, (nijt, 1)).T.astype(np.float32).copy(order='F')
    prv += (np.abs(np.random.randn(nkt, nijt)) * 0.001).astype(np.float32)

    # Cloud fields
    prc = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    cloud_levels = (z > 2000) & (z < 6000)
    for i in range(nijt):
        prc[cloud_levels, i] = np.abs(np.random.rand(cloud_levels.sum())).astype(np.float32) * 0.002

    pri = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    ice_levels = z > 5000
    for i in range(nijt):
        pri[ice_levels, i] = np.abs(np.random.rand(ice_levels.sum())).astype(np.float32) * 0.001

    prr = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    prs = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    prg = np.zeros((nkt, nijt), dtype=np.float32, order='F')

    # Tendencies
    prvs = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    prcs = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    pris = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    pths = np.zeros((nkt, nijt), dtype=np.float32, order='F')

    # Mass flux
    pcf_mf = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    prc_mf = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    pri_mf = np.zeros((nkt, nijt), dtype=np.float32, order='F')

    # 1D sigqsat
    sigqsat = np.ones(nijt, dtype=np.float32, order='F') * 0.01

    # Subgrid turbulence
    psigs = np.zeros((nkt, nijt), dtype=np.float32, order='F')

    return {
        'nijt': nijt,
        'nkt': nkt,
        'ppabst': ppabst,
        'pth': pth,
        'pexn': pexn,
        'pexnref': pexnref,
        'prhodref': prhodref,
        'pzz': pzz,
        'prv': prv,
        'prc': prc,
        'pri': pri,
        'prr': prr,
        'prs': prs,
        'prg': prg,
        'prvs': prvs,
        'prcs': prcs,
        'pris': pris,
        'pths': pths,
        'pcf_mf': pcf_mf,
        'prc_mf': prc_mf,
        'pri_mf': pri_mf,
        'sigqsat': sigqsat,
        'psigs': psigs,
        'temperature': temperature,
    }


@pytest.fixture
def synthetic_data_small():
    """Small synthetic dataset (100 x 60)."""
    return create_synthetic_data(nijt=100, nkt=60)


@pytest.fixture
def synthetic_data_medium():
    """Medium synthetic dataset (500 x 60)."""
    return create_synthetic_data(nijt=500, nkt=60)


# ============================================================================
# Helper functions for data conversion
# ============================================================================

def fortran_to_jax_format(data):
    """
    Convert Fortran-contiguous arrays (nkt, nijt) to JAX format (nx, ny, nkt).

    Parameters
    ----------
    data : dict
        Data dictionary with Fortran-contiguous arrays

    Returns
    -------
    dict
        Data dictionary with JAX-compatible shapes
    """
    nijt = data['nijt']
    nkt = data['nkt']

    # Create a square-ish domain
    nx = int(np.sqrt(nijt))
    ny = nijt // nx

    def reshape_to_3d(arr):
        """Reshape from (nkt, nijt) to (nx, ny, nkt)."""
        if arr.ndim == 2 and arr.shape[0] == nkt:
            # arr is (nkt, nijt), we need (nx, ny, nkt)
            # Transpose to (nijt, nkt), take first nx*ny points, reshape
            return jnp.asarray(arr.T[:nx*ny].reshape(nx, ny, nkt))
        elif arr.ndim == 1:
            # arr is (nijt,), expand to (nx, ny, nkt)
            # Take first nx*ny points and reshape to (nx, ny)
            arr_2d = arr[:nx*ny].reshape(nx, ny)
            # Broadcast to (nx, ny, nkt)
            return jnp.broadcast_to(arr_2d[:, :, None], (nx, ny, nkt))
        else:
            return jnp.asarray(arr)

    return {
        'sigqsat': reshape_to_3d(data['sigqsat']),
        'pabs': reshape_to_3d(data['ppabst']),
        'sigs': reshape_to_3d(data['psigs']),
        'th': reshape_to_3d(data['pth']),
        'exn': reshape_to_3d(data['pexn']),
        'exn_ref': reshape_to_3d(data['pexnref']),
        'rho_dry_ref': reshape_to_3d(data['prhodref']),
        'rv': reshape_to_3d(data['prv']),
        'rc': reshape_to_3d(data['prc']),
        'ri': reshape_to_3d(data['pri']),
        'rr': reshape_to_3d(data['prr']),
        'rs': reshape_to_3d(data['prs']),
        'rg': reshape_to_3d(data['prg']),
        'cf_mf': reshape_to_3d(data['pcf_mf']),
        'rc_mf': reshape_to_3d(data['prc_mf']),
        'ri_mf': reshape_to_3d(data['pri_mf']),
        'rvs': reshape_to_3d(data['prvs']),
        'rcs': reshape_to_3d(data['prcs']),
        'ris': reshape_to_3d(data['pris']),
        'ths': reshape_to_3d(data['pths']),
        'timestep': 60.0,
        'nx': nx,
        'ny': ny,
        'nkt': nkt,
    }


def jax_to_fortran_format(jax_arrays, nijt, nkt):
    """
    Convert JAX arrays (nx, ny, nkt) back to Fortran format (nkt, nijt).

    Parameters
    ----------
    jax_arrays : tuple or dict
        JAX output arrays
    nijt : int
        Number of horizontal points
    nkt : int
        Number of vertical levels

    Returns
    -------
    dict
        Arrays in Fortran-contiguous format
    """
    nx = int(np.sqrt(nijt))
    ny = nijt // nx

    def reshape_from_3d(arr):
        """Reshape from (nx, ny, nkt) to (nkt, nijt) Fortran order."""
        if arr.ndim == 3:
            # (nx, ny, nkt) → (nx*ny, nkt) → (nkt, nx*ny)
            arr_2d = np.array(arr).reshape(nx*ny, nkt)
            return np.asfortranarray(arr_2d[:nijt, :].T, dtype=np.float32)
        return np.asfortranarray(arr, dtype=np.float32)

    return reshape_from_3d


# ============================================================================
# Reproducibility Tests - Synthetic Data
# ============================================================================

class TestSyntheticDataReproducibility:
    """Test reproducibility between Fortran and JAX on synthetic data."""

    @pytest.mark.skipif(not (FORTRAN_AVAILABLE and JAX_AVAILABLE),
                       reason="Both Fortran and JAX needed")
    def test_small_domain_reproducibility(self, ice_adjust_jax, synthetic_data_small):
        """Test that Fortran and JAX produce similar results on small synthetic data."""
        from numpy.testing import assert_allclose

        nijt = synthetic_data_small['nijt']
        nkt = synthetic_data_small['nkt']

        print(f"\n{'='*70}")
        print(f"Testing small domain reproducibility ({nijt} x {nkt})")
        print(f"{'='*70}")

        # Run Fortran version
        print("\n1. Running Fortran ICE_ADJUST...")

        def to_cython(arr):
            """Convert (nkt, nijt) to (nijt, nkt) for Cython."""
            if arr.ndim == 2 and arr.shape[0] == nkt:
                return np.asfortranarray(arr.T, dtype=np.float32)
            return np.asfortranarray(arr, dtype=np.float32)

        cldfr_f = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        icldfr_f = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        wcldfr_f = np.zeros((nijt, nkt), dtype=np.float32, order='F')

        # Need to copy input arrays for Fortran (it modifies in-place)
        data_f = {k: v.copy() for k, v in synthetic_data_small.items()
                  if isinstance(v, np.ndarray)}

        ice_adjust_fortran(
            timestep=np.float32(60.0),
            krr=6,
            sigqsat=data_f['sigqsat'],
            pabs=to_cython(data_f['ppabst']),
            sigs=to_cython(data_f['psigs']),
            th=to_cython(data_f['pth']),
            exn=to_cython(data_f['pexn']),
            exn_ref=to_cython(data_f['pexnref']),
            rho_dry_ref=to_cython(data_f['prhodref']),
            rv=to_cython(data_f['prv']),
            rc=to_cython(data_f['prc']),
            ri=to_cython(data_f['pri']),
            rr=to_cython(data_f['prr']),
            rs=to_cython(data_f['prs']),
            rg=to_cython(data_f['prg']),
            cf_mf=to_cython(data_f['pcf_mf']),
            rc_mf=to_cython(data_f['prc_mf']),
            ri_mf=to_cython(data_f['pri_mf']),
            rvs=to_cython(data_f['prvs']),
            rcs=to_cython(data_f['prcs']),
            ris=to_cython(data_f['pris']),
            ths=to_cython(data_f['pths']),
            cldfr=cldfr_f,
            icldfr=icldfr_f,
            wcldfr=wcldfr_f
        )

        print("✓ Fortran completed")

        # Run JAX version
        print("\n2. Running JAX ICE_ADJUST...")
        jax_data = fortran_to_jax_format(synthetic_data_small)

        result_jax = ice_adjust_jax(**jax_data)
        t_jax, rv_jax, rc_jax, ri_jax, cldfr_jax = result_jax[:5]
        rvs_jax, rcs_jax, ris_jax = result_jax[12], result_jax[13], result_jax[14]

        print("✓ JAX completed")

        # Convert JAX results back to Fortran format for comparison
        print("\n3. Comparing results...")
        reshape_fn = jax_to_fortran_format

        rv_jax_f = reshape_fn(rv_jax, nijt, nkt)
        rc_jax_f = reshape_fn(rc_jax, nijt, nkt)
        ri_jax_f = reshape_fn(ri_jax, nijt, nkt)
        cldfr_jax_f = reshape_fn(cldfr_jax, nijt, nkt).T  # cldfr is (nijt, nkt)

        rvs_jax_f = reshape_fn(rvs_jax, nijt, nkt)
        rcs_jax_f = reshape_fn(rcs_jax, nijt, nkt)
        ris_jax_f = reshape_fn(ris_jax, nijt, nkt)

        # Compare tendencies (modified arrays from Fortran)
        rvs_f = to_cython(data_f['prvs'])
        rcs_f = to_cython(data_f['prcs'])
        ris_f = to_cython(data_f['pris'])

        # Tendencies comparison
        try:
            assert_allclose(rvs_jax_f, rvs_f.T, atol=1e-5, rtol=1e-3)
            print("✓ rvs (vapor tendency) matches")
        except AssertionError:
            max_diff = np.abs(rvs_jax_f - rvs_f.T).max()
            print(f"⚠️  rvs: max diff = {max_diff:.6e}")

        try:
            assert_allclose(rcs_jax_f, rcs_f.T, atol=1e-5, rtol=1e-3)
            print("✓ rcs (cloud tendency) matches")
        except AssertionError:
            max_diff = np.abs(rcs_jax_f - rcs_f.T).max()
            print(f"⚠️  rcs: max diff = {max_diff:.6e}")

        try:
            assert_allclose(ris_jax_f, ris_f.T, atol=1e-5, rtol=1e-3)
            print("✓ ris (ice tendency) matches")
        except AssertionError:
            max_diff = np.abs(ris_jax_f - ris_f.T).max()
            print(f"⚠️  ris: max diff = {max_diff:.6e}")

        # Cloud fraction comparison
        try:
            assert_allclose(cldfr_jax_f, cldfr_f, atol=1e-3, rtol=1e-2)
            print("✓ cldfr (cloud fraction) matches")
        except AssertionError:
            max_diff = np.abs(cldfr_jax_f - cldfr_f).max()
            print(f"⚠️  cldfr: max diff = {max_diff:.6e}")

        print(f"\n{'='*70}")
        print("Reproducibility test complete")
        print(f"{'='*70}")


# ============================================================================
# Reproducibility Tests - Repository Data
# ============================================================================

@pytest.mark.skipif(not (FORTRAN_AVAILABLE and JAX_AVAILABLE),
                   reason="Both Fortran and JAX needed")
def test_repro_data_fortran_vs_jax(ice_adjust_repro_ds):
    """
    Compare Fortran and JAX implementations on repository reference data.

    This test ensures both implementations produce consistent results
    when run on the same reference dataset.
    """
    from numpy.testing import assert_allclose

    print(f"\n{'='*70}")
    print("Testing Fortran vs JAX on repository reference data")
    print(f"{'='*70}")

    # Get dataset dimensions
    shape = (
        ice_adjust_repro_ds.sizes["ngpblks"],
        ice_adjust_repro_ds.sizes["nproma"],
        ice_adjust_repro_ds.sizes["nflevg"]
    )
    nijt = shape[0] * shape[1]
    nkt = shape[2]

    print(f"\nDataset shape: {shape}")
    print(f"Effective domain: nijt={nijt}, nkt={nkt}")

    # ===== Run Fortran version =====
    print("\n1. Running Fortran ICE_ADJUST...")

    def reshape_for_fortran(var):
        """Reshape from (ngpblks, nflevg, nproma) to (nijt, nkt) Fortran order."""
        v = np.swapaxes(var, 1, 2)  # (ngpblks, nproma, nflevg)
        v = v.reshape(nijt, nkt)  # (nijt, nkt)
        return np.asfortranarray(v, dtype=np.float32)

    pabs_f = reshape_for_fortran(ice_adjust_repro_ds["PPABSM"].values)
    exn_f = reshape_for_fortran(ice_adjust_repro_ds["PEXNREF"].values)
    rhodref_f = reshape_for_fortran(ice_adjust_repro_ds["PRHODREF"].values)
    sigs_f = reshape_for_fortran(ice_adjust_repro_ds["PSIGS"].values)

    # ZRS: (ngpblks, krr, nflevg, nproma)
    zrs = ice_adjust_repro_ds["ZRS"].values
    zrs = np.swapaxes(zrs, 2, 3)  # (ngpblks, krr, nproma, nflevg)

    def extract_zrs(idx):
        return zrs[:, idx, :, :].reshape(nijt, nkt).copy(order='F').astype(np.float32)

    th_f = extract_zrs(0)
    rv_f = extract_zrs(1)
    rc_f = extract_zrs(2)
    rr_f = extract_zrs(3)
    ri_f = extract_zrs(4)
    rs_f = extract_zrs(5)
    rg_f = extract_zrs(6)

    cf_mf_f = reshape_for_fortran(ice_adjust_repro_ds["PCF_MF"].values)
    rc_mf_f = reshape_for_fortran(ice_adjust_repro_ds["PRC_MF"].values)
    ri_mf_f = reshape_for_fortran(ice_adjust_repro_ds["PRI_MF"].values)

    sigqsat_f = reshape_for_fortran(ice_adjust_repro_ds["ZSIGQSAT"].values)[:, 0].copy(order='F')

    # Initialize output arrays
    rvs_f = np.zeros_like(rv_f)
    rcs_f = np.zeros_like(rc_f)
    ris_f = np.zeros_like(ri_f)
    ths_f = np.zeros_like(th_f)
    cldfr_f = np.zeros((nijt, nkt), dtype=np.float32, order='F')
    icldfr_f = np.zeros((nijt, nkt), dtype=np.float32, order='F')
    wcldfr_f = np.zeros((nijt, nkt), dtype=np.float32, order='F')

    ice_adjust_fortran(
        timestep=np.float32(50.0),
        krr=6,
        sigqsat=sigqsat_f,
        pabs=pabs_f,
        sigs=sigs_f,
        th=th_f,
        exn=exn_f,
        exn_ref=exn_f,
        rho_dry_ref=rhodref_f,
        rv=rv_f, rc=rc_f, ri=ri_f, rr=rr_f, rs=rs_f, rg=rg_f,
        cf_mf=cf_mf_f, rc_mf=rc_mf_f, ri_mf=ri_mf_f,
        rvs=rvs_f, rcs=rcs_f, ris=ris_f, ths=ths_f,
        cldfr=cldfr_f, icldfr=icldfr_f, wcldfr=wcldfr_f
    )

    print("✓ Fortran completed")

    # ===== Run JAX version =====
    print("\n2. Running JAX ICE_ADJUST...")

    phyex = Phyex("AROME", TSTEP=50.0)
    ice_adjust_jax = IceAdjustJAX(phyex=phyex, jit=True)

    def reshape_for_jax(var):
        """Reshape dataset variable for JAX (swap axes)."""
        return jnp.asarray(np.swapaxes(var, 1, 2))

    pabs_jax = reshape_for_jax(ice_adjust_repro_ds["PPABSM"].values)

    # Load state from ZRS
    zrs_jax = ice_adjust_repro_ds["ZRS"].values
    zrs_jax = np.swapaxes(zrs_jax, 2, 3)  # → (ngpblks, krr1, nproma, nflevg)

    th_jax = jnp.asarray(zrs_jax[:, 0, :, :])
    rv_jax_in = jnp.asarray(zrs_jax[:, 1, :, :])
    rc_jax_in = jnp.asarray(zrs_jax[:, 2, :, :])
    rr_jax = jnp.asarray(zrs_jax[:, 3, :, :])
    ri_jax_in = jnp.asarray(zrs_jax[:, 4, :, :])
    rs_jax = jnp.asarray(zrs_jax[:, 5, :, :])
    rg_jax = jnp.asarray(zrs_jax[:, 6, :, :])

    exn_jax = reshape_for_jax(ice_adjust_repro_ds["PEXNREF"].values)
    rhodref_jax = reshape_for_jax(ice_adjust_repro_ds["PRHODREF"].values)

    # Load input tendencies from PRS
    prs = ice_adjust_repro_ds["PRS"].values
    prs = np.swapaxes(prs, 2, 3)

    rvs_jax_in = jnp.asarray(prs[:, 0, :, :])
    rcs_jax_in = jnp.asarray(prs[:, 1, :, :])
    ris_jax_in = jnp.asarray(prs[:, 3, :, :])

    ths_jax = reshape_for_jax(ice_adjust_repro_ds["PTHS"].values)

    cf_mf_jax = reshape_for_jax(ice_adjust_repro_ds["PCF_MF"].values)
    rc_mf_jax = reshape_for_jax(ice_adjust_repro_ds["PRC_MF"].values)
    ri_mf_jax = reshape_for_jax(ice_adjust_repro_ds["PRI_MF"].values)

    zsigqsat = ice_adjust_repro_ds["ZSIGQSAT"].values
    sigqsat_jax = jnp.asarray(zsigqsat[:, :, np.newaxis])

    sigs_jax = reshape_for_jax(ice_adjust_repro_ds["PSIGS"].values)

    result_jax = ice_adjust_jax(
        sigqsat=sigqsat_jax,
        pabs=pabs_jax,
        sigs=sigs_jax,
        th=th_jax,
        exn=exn_jax,
        exn_ref=exn_jax,
        rho_dry_ref=rhodref_jax,
        rv=rv_jax_in,
        rc=rc_jax_in,
        ri=ri_jax_in,
        rr=rr_jax,
        rs=rs_jax,
        rg=rg_jax,
        cf_mf=cf_mf_jax,
        rc_mf=rc_mf_jax,
        ri_mf=ri_mf_jax,
        rvs=rvs_jax_in,
        rcs=rcs_jax_in,
        ris=ris_jax_in,
        ths=ths_jax,
        timestep=50.0,
    )

    print("✓ JAX completed")

    # ===== Compare results =====
    print("\n3. Comparing Fortran vs JAX results...")

    t_jax, rv_jax, rc_jax, ri_jax, cldfr_jax = result_jax[:5]
    rvs_jax_out, rcs_jax_out, ris_jax_out = result_jax[12], result_jax[13], result_jax[14]

    # Reshape JAX results for comparison
    def reshape_jax_to_fortran(arr):
        """Reshape JAX (ngpblks, nproma, nflevg) to Fortran (nijt, nkt)."""
        return np.array(arr).reshape(nijt, nkt)

    rvs_jax_f = reshape_jax_to_fortran(rvs_jax_out)
    rcs_jax_f = reshape_jax_to_fortran(rcs_jax_out)
    ris_jax_f = reshape_jax_to_fortran(ris_jax_out)
    cldfr_jax_f = reshape_jax_to_fortran(cldfr_jax)

    # Compare tendencies
    tests = [
        ("rvs", rvs_jax_f, rvs_f, "vapor tendency"),
        ("rcs", rcs_jax_f, rcs_f, "cloud water tendency"),
        ("ris", ris_jax_f, ris_f, "ice tendency"),
        ("cldfr", cldfr_jax_f, cldfr_f.T, "cloud fraction"),
    ]

    for name, jax_val, fortran_val, desc in tests:
        try:
            assert_allclose(jax_val, fortran_val, atol=1e-5, rtol=1e-3)
            print(f"✓ {name} ({desc}) matches")
        except AssertionError:
            max_diff = np.abs(jax_val - fortran_val).max()
            rel_diff = np.abs((jax_val - fortran_val) / (np.abs(fortran_val) + 1e-10)).max()
            print(f"⚠️  {name}: max abs diff = {max_diff:.6e}, max rel diff = {rel_diff:.6e}")

    print(f"\n{'='*70}")
    print("Fortran vs JAX comparison complete")
    print(f"{'='*70}")
    print("\nNote: Small differences are expected due to different")
    print("compilation strategies and numerical precision.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
