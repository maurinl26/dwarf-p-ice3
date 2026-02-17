# -*- coding: utf-8 -*-
"""Benchmark comparing Fortran and JAX implementations of ICE_ADJUST on CPU."""
import numpy as np
import pytest
import sys
from pathlib import Path

# Try to import Fortran wrapper
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
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "PHYEX-IAL_CY50T1/bridge"))
    try:
        from _phyex_wrapper import ice_adjust as ice_adjust_fortran
        FORTRAN_AVAILABLE = True
    except ImportError:
        FORTRAN_AVAILABLE = False
        ice_adjust_fortran = None

# Try to import JAX implementation
try:
    import os
    import jax
    jax.config.update('jax_platform_name', os.environ.get('JAX_PLATFORM_NAME', 'cpu'))
    import jax.numpy as jnp
    from ice3.jax.ice_adjust import IceAdjustJAX
    from ice3.phyex_common.phyex import Phyex
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    IceAdjustJAX = None
    Phyex = None


def create_test_atmosphere(nijt=2500, nkt=40, backend='fortran'):
    """
    Create realistic atmospheric test data for ICE_ADJUST benchmarks.

    Parameters
    ----------
    nijt : int
        Number of horizontal points (flattened)
    nkt : int
        Number of vertical levels
    backend : str
        'fortran' or 'jax' - determines output format

    Returns
    -------
    dict
        Dictionary with all required fields in the appropriate format
    """
    # Create vertical coordinate (0-10 km)
    z = np.linspace(0, 10000, nkt, dtype=np.float32)

    # Standard atmosphere parameters
    p0 = 101325.0  # Pa
    T0 = 288.15    # K
    gamma = 0.0065  # K/m

    # Physical constants
    Rd = 287.0
    cp = 1004.0
    p00 = 100000.0

    # Pressure profile
    pressure = p0 * (1 - gamma * z / T0) ** 5.26
    ppabst = np.tile(pressure, (nijt, 1)).T.astype(np.float32)

    # Temperature profile
    temperature = T0 - gamma * z
    temperature = np.tile(temperature, (nijt, 1)).T.astype(np.float32)

    # Add variability
    np.random.seed(42)
    temperature += np.random.randn(nkt, nijt).astype(np.float32) * 0.5
    ppabst += np.random.randn(nkt, nijt).astype(np.float32) * 100

    # Exner function
    pexn = (ppabst / p00) ** (Rd / cp)
    pth = temperature / pexn

    # Reference values
    pexnref = pexn.copy()
    prhodref = ppabst / (Rd * temperature)

    # Water vapor (decreasing with height)
    rv_surf = 0.015
    rv_profile = rv_surf * np.exp(-z / 2000)
    prv = np.tile(rv_profile, (nijt, 1)).T.astype(np.float32)
    prv += np.abs(np.random.randn(nkt, nijt).astype(np.float32)) * 0.001

    # Cloud fields
    prc = np.zeros((nkt, nijt), dtype=np.float32)
    cloud_levels = (z > 2000) & (z < 6000)
    for i in range(nijt):
        prc[cloud_levels, i] = np.abs(np.random.rand(cloud_levels.sum())).astype(np.float32) * 0.002

    pri = np.zeros((nkt, nijt), dtype=np.float32)
    ice_levels = z > 5000
    for i in range(nijt):
        pri[ice_levels, i] = np.abs(np.random.rand(ice_levels.sum())).astype(np.float32) * 0.001

    # Precipitation
    prr = np.zeros((nkt, nijt), dtype=np.float32)
    prs = np.zeros((nkt, nijt), dtype=np.float32)
    prg = np.zeros((nkt, nijt), dtype=np.float32)

    # Mass flux
    pcf_mf = np.zeros((nkt, nijt), dtype=np.float32)
    prc_mf = np.zeros((nkt, nijt), dtype=np.float32)
    pri_mf = np.zeros((nkt, nijt), dtype=np.float32)

    # Tendencies
    prvs = np.zeros((nkt, nijt), dtype=np.float32)
    prcs = np.zeros((nkt, nijt), dtype=np.float32)
    pris = np.zeros((nkt, nijt), dtype=np.float32)
    pths = np.zeros((nkt, nijt), dtype=np.float32)

    # Turbulence
    psigs = np.full((nkt, nijt), 0.1, dtype=np.float32)
    sigqsat = np.ones(nijt, dtype=np.float32) * 0.01

    if backend == 'fortran':
        # Fortran expects Fortran-contiguous arrays
        return {
            'ppabst': np.asfortranarray(ppabst),
            'pth': np.asfortranarray(pth),
            'pexn': np.asfortranarray(pexn),
            'pexnref': np.asfortranarray(pexnref),
            'prhodref': np.asfortranarray(prhodref),
            'prv': np.asfortranarray(prv),
            'prc': np.asfortranarray(prc),
            'pri': np.asfortranarray(pri),
            'prr': np.asfortranarray(prr),
            'prs': np.asfortranarray(prs),
            'prg': np.asfortranarray(prg),
            'pcf_mf': np.asfortranarray(pcf_mf),
            'prc_mf': np.asfortranarray(prc_mf),
            'pri_mf': np.asfortranarray(pri_mf),
            'prvs': np.asfortranarray(prvs),
            'prcs': np.asfortranarray(prcs),
            'pris': np.asfortranarray(pris),
            'pths': np.asfortranarray(pths),
            'psigs': np.asfortranarray(psigs),
            'sigqsat': np.asfortranarray(sigqsat),
        }
    elif backend == 'jax':
        # For JAX, we need to reshape from (nkt, nijt) to 3D
        # We'll create a square-ish domain
        nx = int(np.sqrt(nijt))
        ny = nijt // nx

        def reshape_to_3d(arr):
            """Reshape from (nkt, nijt) to (nx, ny, nkt)."""
            if arr.ndim == 2:
                # arr is (nkt, nijt), we need (nx, ny, nkt)
                # Transpose to (nijt, nkt), take first nx*ny points, reshape
                return jnp.asarray(arr.T[:nx*ny].reshape(nx, ny, nkt))
            else:
                # arr is (nijt,), expand to (nx, ny, nkt)
                # Take first nx*ny points and reshape to (nx, ny)
                arr_2d = arr[:nx*ny].reshape(nx, ny)
                # Broadcast to (nx, ny, nkt)
                return jnp.broadcast_to(arr_2d[:, :, None], (nx, ny, nkt))

        return {
            'sigqsat': reshape_to_3d(sigqsat),
            'pabs': reshape_to_3d(ppabst),
            'sigs': reshape_to_3d(psigs),
            'th': reshape_to_3d(pth),
            'exn': reshape_to_3d(pexn),
            'exn_ref': reshape_to_3d(pexnref),
            'rho_dry_ref': reshape_to_3d(prhodref),
            'rv': reshape_to_3d(prv),
            'rc': reshape_to_3d(prc),
            'ri': reshape_to_3d(pri),
            'rr': reshape_to_3d(prr),
            'rs': reshape_to_3d(prs),
            'rg': reshape_to_3d(prg),
            'cf_mf': reshape_to_3d(pcf_mf),
            'rc_mf': reshape_to_3d(prc_mf),
            'ri_mf': reshape_to_3d(pri_mf),
            'rvs': reshape_to_3d(prvs),
            'rcs': reshape_to_3d(prcs),
            'ris': reshape_to_3d(pris),
            'ths': reshape_to_3d(pths),
            'timestep': 50.0,
        }


@pytest.fixture
def small_data_fortran():
    """Small domain for Fortran (100 horizontal points, 20 levels)."""
    return create_test_atmosphere(nijt=100, nkt=20, backend='fortran')


@pytest.fixture
def small_data_jax():
    """Small domain for JAX (10x10x20)."""
    return create_test_atmosphere(nijt=100, nkt=20, backend='jax')


@pytest.fixture
def medium_data_fortran():
    """Medium domain for Fortran (2500 horizontal points, 40 levels)."""
    return create_test_atmosphere(nijt=2500, nkt=40, backend='fortran')


@pytest.fixture
def medium_data_jax():
    """Medium domain for JAX (50x50x40)."""
    return create_test_atmosphere(nijt=2500, nkt=40, backend='jax')


@pytest.fixture
def large_data_fortran():
    """Large domain for Fortran (10000 horizontal points, 60 levels)."""
    return create_test_atmosphere(nijt=10000, nkt=60, backend='fortran')


@pytest.fixture
def large_data_jax():
    """Large domain for JAX (100x100x60)."""
    return create_test_atmosphere(nijt=10000, nkt=60, backend='jax')


@pytest.fixture
def ice_adjust_jax_cpu():
    """Create IceAdjustJAX instance configured for CPU with JIT."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    phyex = Phyex("AROME", TSTEP=50.0)
    return IceAdjustJAX(phyex=phyex, jit=True)


def swap_to_cython(arr, nkt, nijt):
    """
    Convert from (nkt, nijt) to (nijt, nkt) for Cython wrapper.

    Parameters
    ----------
    arr : np.ndarray
        Array in (nkt, nijt) format
    nkt : int
        Number of vertical levels
    nijt : int
        Number of horizontal points

    Returns
    -------
    np.ndarray
        Array in (nijt, nkt) Fortran order
    """
    if arr.ndim == 2 and arr.shape[0] == nkt:
        return np.asfortranarray(arr.T, dtype=np.float32)
    return np.asfortranarray(arr, dtype=np.float32)


# ============================================================================
# Small Domain Benchmarks (100 points × 20 levels = 2,000 grid points)
# ============================================================================

@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrapper not available")
class TestSmallDomainFortran:
    """Fortran benchmarks on small domain."""

    def test_fortran_small(self, benchmark, small_data_fortran):
        """Benchmark Fortran implementation on small domain."""
        nijt, nkt = 100, 20
        total_points = nijt * nkt

        # Prepare output arrays
        cldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        icldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        wcldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')

        def run_ice_adjust():
            ice_adjust_fortran(
                timestep=np.float32(50.0),
                krr=6,
                sigqsat=small_data_fortran['sigqsat'],
                pabs=swap_to_cython(small_data_fortran['ppabst'], nkt, nijt),
                sigs=swap_to_cython(small_data_fortran['psigs'], nkt, nijt),
                th=swap_to_cython(small_data_fortran['pth'], nkt, nijt),
                exn=swap_to_cython(small_data_fortran['pexn'], nkt, nijt),
                exn_ref=swap_to_cython(small_data_fortran['pexnref'], nkt, nijt),
                rho_dry_ref=swap_to_cython(small_data_fortran['prhodref'], nkt, nijt),
                rv=swap_to_cython(small_data_fortran['prv'], nkt, nijt),
                rc=swap_to_cython(small_data_fortran['prc'], nkt, nijt),
                ri=swap_to_cython(small_data_fortran['pri'], nkt, nijt),
                rr=swap_to_cython(small_data_fortran['prr'], nkt, nijt),
                rs=swap_to_cython(small_data_fortran['prs'], nkt, nijt),
                rg=swap_to_cython(small_data_fortran['prg'], nkt, nijt),
                cf_mf=swap_to_cython(small_data_fortran['pcf_mf'], nkt, nijt),
                rc_mf=swap_to_cython(small_data_fortran['prc_mf'], nkt, nijt),
                ri_mf=swap_to_cython(small_data_fortran['pri_mf'], nkt, nijt),
                rvs=swap_to_cython(small_data_fortran['prvs'], nkt, nijt),
                rcs=swap_to_cython(small_data_fortran['prcs'], nkt, nijt),
                ris=swap_to_cython(small_data_fortran['pris'], nkt, nijt),
                ths=swap_to_cython(small_data_fortran['pths'], nkt, nijt),
                cldfr=cldfr,
                icldfr=icldfr,
                wcldfr=wcldfr
            )

        result = benchmark(run_ice_adjust)

        # Print statistics (only when benchmark is enabled)
        if hasattr(result, 'stats'):
            mean_time = result.stats['mean']
            throughput = total_points / mean_time
            print(f"\n[Fortran] Small domain ({nijt}×{nkt}={total_points:,} points)")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e6:.2f} M points/s")


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestSmallDomainJAX:
    """JAX benchmarks on small domain."""

    def test_jax_small(self, benchmark, ice_adjust_jax_cpu, small_data_jax):
        """Benchmark JAX implementation on small domain (CPU)."""
        nx = int(np.sqrt(100))
        ny = 100 // nx
        nz = 20
        total_points = nx * ny * nz

        # Warm-up to trigger JIT compilation (multiple iterations to ensure full optimization)
        for _ in range(3):
            _ = ice_adjust_jax_cpu(**small_data_jax)

        # Block until all JAX operations complete
        if JAX_AVAILABLE:
            import jax
            jax.block_until_ready(ice_adjust_jax_cpu(**small_data_jax))

        result = benchmark(lambda: ice_adjust_jax_cpu(**small_data_jax))

        # Print statistics (only when benchmark is enabled)
        if hasattr(result, 'stats'):
            mean_time = result.stats['mean']
            throughput = total_points / mean_time
            print(f"\n[JAX CPU] Small domain ({nx}×{ny}×{nz}={total_points:,} points)")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e6:.2f} M points/s")


# ============================================================================
# Medium Domain Benchmarks (2500 points × 40 levels = 100,000 grid points)
# ============================================================================

@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrapper not available")
class TestMediumDomainFortran:
    """Fortran benchmarks on medium domain."""

    def test_fortran_medium(self, benchmark, medium_data_fortran):
        """Benchmark Fortran implementation on medium domain."""
        nijt, nkt = 2500, 40
        total_points = nijt * nkt

        cldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        icldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        wcldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')

        def run_ice_adjust():
            ice_adjust_fortran(
                timestep=np.float32(50.0),
                krr=6,
                sigqsat=medium_data_fortran['sigqsat'],
                pabs=swap_to_cython(medium_data_fortran['ppabst'], nkt, nijt),
                sigs=swap_to_cython(medium_data_fortran['psigs'], nkt, nijt),
                th=swap_to_cython(medium_data_fortran['pth'], nkt, nijt),
                exn=swap_to_cython(medium_data_fortran['pexn'], nkt, nijt),
                exn_ref=swap_to_cython(medium_data_fortran['pexnref'], nkt, nijt),
                rho_dry_ref=swap_to_cython(medium_data_fortran['prhodref'], nkt, nijt),
                rv=swap_to_cython(medium_data_fortran['prv'], nkt, nijt),
                rc=swap_to_cython(medium_data_fortran['prc'], nkt, nijt),
                ri=swap_to_cython(medium_data_fortran['pri'], nkt, nijt),
                rr=swap_to_cython(medium_data_fortran['prr'], nkt, nijt),
                rs=swap_to_cython(medium_data_fortran['prs'], nkt, nijt),
                rg=swap_to_cython(medium_data_fortran['prg'], nkt, nijt),
                cf_mf=swap_to_cython(medium_data_fortran['pcf_mf'], nkt, nijt),
                rc_mf=swap_to_cython(medium_data_fortran['prc_mf'], nkt, nijt),
                ri_mf=swap_to_cython(medium_data_fortran['pri_mf'], nkt, nijt),
                rvs=swap_to_cython(medium_data_fortran['prvs'], nkt, nijt),
                rcs=swap_to_cython(medium_data_fortran['prcs'], nkt, nijt),
                ris=swap_to_cython(medium_data_fortran['pris'], nkt, nijt),
                ths=swap_to_cython(medium_data_fortran['pths'], nkt, nijt),
                cldfr=cldfr,
                icldfr=icldfr,
                wcldfr=wcldfr
            )

        result = benchmark(run_ice_adjust)

        # Print statistics (only when benchmark is enabled)
        if hasattr(result, 'stats'):
            mean_time = result.stats['mean']
            throughput = total_points / mean_time
            print(f"\n[Fortran] Medium domain ({nijt}×{nkt}={total_points:,} points)")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e6:.2f} M points/s")


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestMediumDomainJAX:
    """JAX benchmarks on medium domain."""

    def test_jax_medium(self, benchmark, ice_adjust_jax_cpu, medium_data_jax):
        """Benchmark JAX implementation on medium domain (CPU)."""
        nx = int(np.sqrt(2500))
        ny = 2500 // nx
        nz = 40
        total_points = nx * ny * nz

        # Warm-up to trigger JIT compilation (multiple iterations)
        for _ in range(3):
            _ = ice_adjust_jax_cpu(**medium_data_jax)

        # Block until all JAX operations complete
        if JAX_AVAILABLE:
            import jax
            jax.block_until_ready(ice_adjust_jax_cpu(**medium_data_jax))

        result = benchmark(lambda: ice_adjust_jax_cpu(**medium_data_jax))

        # Print statistics (only when benchmark is enabled)
        if hasattr(result, 'stats'):
            mean_time = result.stats['mean']
            throughput = total_points / mean_time
            print(f"\n[JAX CPU] Medium domain ({nx}×{ny}×{nz}={total_points:,} points)")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e6:.2f} M points/s")


# ============================================================================
# Large Domain Benchmarks (10000 points × 60 levels = 600,000 grid points)
# ============================================================================

@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrapper not available")
class TestLargeDomainFortran:
    """Fortran benchmarks on large domain."""

    def test_fortran_large(self, benchmark, large_data_fortran):
        """Benchmark Fortran implementation on large domain."""
        nijt, nkt = 10000, 60
        total_points = nijt * nkt

        cldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        icldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        wcldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')

        def run_ice_adjust():
            ice_adjust_fortran(
                timestep=np.float32(50.0),
                krr=6,
                sigqsat=large_data_fortran['sigqsat'],
                pabs=swap_to_cython(large_data_fortran['ppabst'], nkt, nijt),
                sigs=swap_to_cython(large_data_fortran['psigs'], nkt, nijt),
                th=swap_to_cython(large_data_fortran['pth'], nkt, nijt),
                exn=swap_to_cython(large_data_fortran['pexn'], nkt, nijt),
                exn_ref=swap_to_cython(large_data_fortran['pexnref'], nkt, nijt),
                rho_dry_ref=swap_to_cython(large_data_fortran['prhodref'], nkt, nijt),
                rv=swap_to_cython(large_data_fortran['prv'], nkt, nijt),
                rc=swap_to_cython(large_data_fortran['prc'], nkt, nijt),
                ri=swap_to_cython(large_data_fortran['pri'], nkt, nijt),
                rr=swap_to_cython(large_data_fortran['prr'], nkt, nijt),
                rs=swap_to_cython(large_data_fortran['prs'], nkt, nijt),
                rg=swap_to_cython(large_data_fortran['prg'], nkt, nijt),
                cf_mf=swap_to_cython(large_data_fortran['pcf_mf'], nkt, nijt),
                rc_mf=swap_to_cython(large_data_fortran['prc_mf'], nkt, nijt),
                ri_mf=swap_to_cython(large_data_fortran['pri_mf'], nkt, nijt),
                rvs=swap_to_cython(large_data_fortran['prvs'], nkt, nijt),
                rcs=swap_to_cython(large_data_fortran['prcs'], nkt, nijt),
                ris=swap_to_cython(large_data_fortran['pris'], nkt, nijt),
                ths=swap_to_cython(large_data_fortran['pths'], nkt, nijt),
                cldfr=cldfr,
                icldfr=icldfr,
                wcldfr=wcldfr
            )

        result = benchmark(run_ice_adjust)

        # Print statistics (only when benchmark is enabled)
        if hasattr(result, 'stats'):
            mean_time = result.stats['mean']
            throughput = total_points / mean_time
            print(f"\n[Fortran] Large domain ({nijt}×{nkt}={total_points:,} points)")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e6:.2f} M points/s")

            if hasattr(result.stats, 'stddev'):
                print(f"  Std dev: {result.stats['stddev']*1000:.3f} ms")


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestLargeDomainJAX:
    """JAX benchmarks on large domain."""

    def test_jax_large(self, benchmark, ice_adjust_jax_cpu, large_data_jax):
        """Benchmark JAX implementation on large domain (CPU)."""
        nx = int(np.sqrt(10000))
        ny = 10000 // nx
        nz = 60
        total_points = nx * ny * nz

        # Warm-up to trigger JIT compilation (multiple iterations)
        for _ in range(3):
            _ = ice_adjust_jax_cpu(**large_data_jax)

        # Block until all JAX operations complete
        if JAX_AVAILABLE:
            import jax
            jax.block_until_ready(ice_adjust_jax_cpu(**large_data_jax))

        result = benchmark(lambda: ice_adjust_jax_cpu(**large_data_jax))

        # Print statistics (only when benchmark is enabled)
        if hasattr(result, 'stats'):
            mean_time = result.stats['mean']
            throughput = total_points / mean_time
            print(f"\n[JAX CPU] Large domain ({nx}×{ny}×{nz}={total_points:,} points)")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e6:.2f} M points/s")

            if hasattr(result.stats, 'stddev'):
                print(f"  Std dev: {result.stats['stddev']*1000:.3f} ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only", "--benchmark-group-by=group"])
