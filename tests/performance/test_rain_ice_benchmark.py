# -*- coding: utf-8 -*-
"""Benchmark comparing Fortran and JAX implementations of RAIN_ICE on CPU."""
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
    from ice3._phyex_wrapper import rain_ice as rain_ice_fortran
    from ice3._phyex_wrapper import init_rain_ice
    FORTRAN_AVAILABLE = True
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "PHYEX-IAL_CY50T1/bridge"))
    try:
        from _phyex_wrapper import rain_ice as rain_ice_fortran
        from _phyex_wrapper import init_rain_ice
        FORTRAN_AVAILABLE = True
    except ImportError:
        FORTRAN_AVAILABLE = False
        rain_ice_fortran = None
        init_rain_ice = None

# Try to import JAX implementation
try:
    import jax
    # Force JAX to use CPU only
    jax.config.update('jax_platform_name', 'cpu')
    import jax.numpy as jnp
    from ice3.jax.rain_ice import RainIceJAX
    from ice3.phyex_common.phyex import Phyex
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    RainIceJAX = None
    Phyex = None


def create_test_atmosphere_rain_ice(nijt=2500, nkt=60, backend='fortran'):
    """
    Create realistic atmospheric test data for RAIN_ICE benchmarks.

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
    ptht = temperature / pexn

    # Reference values
    pexnref = pexn.copy()
    prhodref = ppabst / (Rd * temperature)
    prhodj = prhodref.copy()

    # Layer thickness (100m per level)
    pdzz = np.full((nkt, nijt), 100.0, dtype=np.float32)

    # Water vapor (decreasing with height)
    rv_surf = 0.015
    prvt = rv_surf * np.exp(-z / 2000)
    prvt = np.tile(prvt, (nijt, 1)).T.astype(np.float32)
    prvt += np.abs(np.random.randn(nkt, nijt).astype(np.float32)) * 0.002

    # Cloud water
    prct = np.zeros((nkt, nijt), dtype=np.float32)
    cloud_levels = (z > 2000) & (z < 6000)
    for i in range(nijt):
        prct[cloud_levels, i] = np.abs(np.random.rand(cloud_levels.sum())).astype(np.float32) * 0.003

    # Rain
    prrt = np.zeros((nkt, nijt), dtype=np.float32)
    rain_levels = (z > 1000) & (z < 4000)
    for i in range(nijt):
        prrt[rain_levels, i] = np.abs(np.random.rand(rain_levels.sum())).astype(np.float32) * 0.001

    # Ice
    prit = np.zeros((nkt, nijt), dtype=np.float32)
    ice_levels = z > 5000
    for i in range(nijt):
        prit[ice_levels, i] = np.abs(np.random.rand(ice_levels.sum())).astype(np.float32) * 0.002

    # Snow
    prst = np.zeros((nkt, nijt), dtype=np.float32)
    snow_levels = (z > 3000) & (z < 7000)
    for i in range(nijt):
        prst[snow_levels, i] = np.abs(np.random.rand(snow_levels.sum())).astype(np.float32) * 0.001

    # Graupel
    prgt = np.zeros((nkt, nijt), dtype=np.float32)
    graupel_levels = (z > 2000) & (z < 6000)
    for i in range(nijt):
        prgt[graupel_levels, i] = np.abs(np.random.rand(graupel_levels.sum())).astype(np.float32) * 0.0005

    # Ice concentration
    pcit = np.where(prit > 0, 1000.0, 0.0).astype(np.float32)

    # Tendencies
    pths = np.zeros((nkt, nijt), dtype=np.float32)
    prvs = np.zeros((nkt, nijt), dtype=np.float32)
    prcs = np.zeros((nkt, nijt), dtype=np.float32)
    prrs = np.zeros((nkt, nijt), dtype=np.float32)
    pris = np.zeros((nkt, nijt), dtype=np.float32)
    prss = np.zeros((nkt, nijt), dtype=np.float32)
    prgs = np.zeros((nkt, nijt), dtype=np.float32)
    pcis = np.zeros((nkt, nijt), dtype=np.float32)

    # Turbulence
    psigs = np.full((nkt, nijt), 0.1, dtype=np.float32)

    if backend == 'fortran':
        # Fortran expects Fortran-contiguous arrays
        return {
            'ppabst': np.asfortranarray(ppabst),
            'ptht': np.asfortranarray(ptht),
            'pexn': np.asfortranarray(pexn),
            'pexnref': np.asfortranarray(pexnref),
            'prhodref': np.asfortranarray(prhodref),
            'prhodj': np.asfortranarray(prhodj),
            'pdzz': np.asfortranarray(pdzz),
            'prvt': np.asfortranarray(prvt),
            'prct': np.asfortranarray(prct),
            'prrt': np.asfortranarray(prrt),
            'prit': np.asfortranarray(prit),
            'prst': np.asfortranarray(prst),
            'prgt': np.asfortranarray(prgt),
            'pcit': np.asfortranarray(pcit),
            'pths': np.asfortranarray(pths),
            'prvs': np.asfortranarray(prvs),
            'prcs': np.asfortranarray(prcs),
            'prrs': np.asfortranarray(prrs),
            'pris': np.asfortranarray(pris),
            'prss': np.asfortranarray(prss),
            'prgs': np.asfortranarray(prgs),
            'pcis': np.asfortranarray(pcis),
            'psigs': np.asfortranarray(psigs),
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

        # Return both state dict and timestep separately for JAX
        # Compute actual temperature from potential temperature and Exner function
        state = {
            'pres': reshape_to_3d(ppabst),
            'th_t': reshape_to_3d(ptht),
            't': reshape_to_3d(temperature),  # Actual temperature
            'exn': reshape_to_3d(pexn),
            'rhodref': reshape_to_3d(prhodref),
            'dzz': reshape_to_3d(pdzz),
            'rv_t': reshape_to_3d(prvt),
            'rc_t': reshape_to_3d(prct),
            'rr_t': reshape_to_3d(prrt),
            'ri_t': reshape_to_3d(prit),
            'rs_t': reshape_to_3d(prst),
            'rg_t': reshape_to_3d(prgt),
            'ci_t': reshape_to_3d(pcit),
            'ths': reshape_to_3d(pths),
            'rvs': reshape_to_3d(prvs),
            'rcs': reshape_to_3d(prcs),
            'rrs': reshape_to_3d(prrs),
            'ris': reshape_to_3d(pris),
            'rss': reshape_to_3d(prss),
            'rgs': reshape_to_3d(prgs),
            'cis': reshape_to_3d(pcis),
            'sigs': reshape_to_3d(psigs),
        }
        return {'state': state, 'timestep': 50.0}


@pytest.fixture(scope="module")
def init_rain_ice_fortran():
    """Initialize RAIN_ICE module once for all Fortran tests."""
    if FORTRAN_AVAILABLE and init_rain_ice is not None:
        init_rain_ice(timestep=50.0, dzmin=60.0, krr=6, hcloud="AROME")
    return True


@pytest.fixture
def small_data_fortran():
    """Small domain for Fortran (100 horizontal points, 20 levels)."""
    return create_test_atmosphere_rain_ice(nijt=100, nkt=20, backend='fortran')


@pytest.fixture
def small_data_jax():
    """Small domain for JAX (10x10x20)."""
    return create_test_atmosphere_rain_ice(nijt=100, nkt=20, backend='jax')


@pytest.fixture
def medium_data_fortran():
    """Medium domain for Fortran (2500 horizontal points, 40 levels)."""
    return create_test_atmosphere_rain_ice(nijt=2500, nkt=40, backend='fortran')


@pytest.fixture
def medium_data_jax():
    """Medium domain for JAX (50x50x40)."""
    return create_test_atmosphere_rain_ice(nijt=2500, nkt=40, backend='jax')


@pytest.fixture
def large_data_fortran():
    """Large domain for Fortran (10000 horizontal points, 60 levels)."""
    return create_test_atmosphere_rain_ice(nijt=10000, nkt=60, backend='fortran')


@pytest.fixture
def large_data_jax():
    """Large domain for JAX (100x100x60)."""
    return create_test_atmosphere_rain_ice(nijt=10000, nkt=60, backend='jax')


@pytest.fixture
def rain_ice_jax_cpu():
    """Create RainIceJAX instance configured for CPU."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    phyex = Phyex("AROME", TSTEP=50.0)
    return RainIceJAX(constants=phyex.to_externals())


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

    def test_fortran_small(self, benchmark, init_rain_ice_fortran, small_data_fortran):
        """Benchmark Fortran implementation on small domain."""
        nijt, nkt = 100, 20
        total_points = nijt * nkt

        def run_rain_ice():
            rain_ice_fortran(
                timestep=np.float32(50.0),
                krr=6,
                pexn=swap_to_cython(small_data_fortran['pexn'], nkt, nijt),
                rhodref=swap_to_cython(small_data_fortran['prhodref'], nkt, nijt),
                pres=swap_to_cython(small_data_fortran['ppabst'], nkt, nijt),
                dzz=swap_to_cython(small_data_fortran['pdzz'], nkt, nijt),
                th_t=swap_to_cython(small_data_fortran['ptht'], nkt, nijt),
                rv_t=swap_to_cython(small_data_fortran['prvt'], nkt, nijt),
                rc_t=swap_to_cython(small_data_fortran['prct'], nkt, nijt),
                rr_t=swap_to_cython(small_data_fortran['prrt'], nkt, nijt),
                ri_t=swap_to_cython(small_data_fortran['prit'], nkt, nijt),
                rs_t=swap_to_cython(small_data_fortran['prst'], nkt, nijt),
                rg_t=swap_to_cython(small_data_fortran['prgt'], nkt, nijt),
                ci_t=swap_to_cython(small_data_fortran['pcit'], nkt, nijt),
                ths=swap_to_cython(small_data_fortran['pths'], nkt, nijt),
                rvs=swap_to_cython(small_data_fortran['prvs'], nkt, nijt),
                rcs=swap_to_cython(small_data_fortran['prcs'], nkt, nijt),
                rrs=swap_to_cython(small_data_fortran['prrs'], nkt, nijt),
                ris=swap_to_cython(small_data_fortran['pris'], nkt, nijt),
                rss=swap_to_cython(small_data_fortran['prss'], nkt, nijt),
                rgs=swap_to_cython(small_data_fortran['prgs'], nkt, nijt),
                cis=swap_to_cython(small_data_fortran['pcis'], nkt, nijt),
                sigs=swap_to_cython(small_data_fortran['psigs'], nkt, nijt)
            )

        result = benchmark(run_rain_ice)

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

    def test_jax_small(self, benchmark, rain_ice_jax_cpu, small_data_jax):
        """Benchmark JAX implementation on small domain (CPU)."""
        nx = int(np.sqrt(100))
        ny = 100 // nx
        nz = 20
        total_points = nx * ny * nz

        state = small_data_jax['state']
        dt = small_data_jax['timestep']

        # Warm-up to trigger JIT compilation (multiple iterations)
        for _ in range(3):
            _ = rain_ice_jax_cpu(state, dt)

        # Block until all JAX operations complete
        if JAX_AVAILABLE:
            import jax
            jax.block_until_ready(rain_ice_jax_cpu(state, dt))

        result = benchmark(lambda: rain_ice_jax_cpu(state, dt))

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

    def test_fortran_medium(self, benchmark, init_rain_ice_fortran, medium_data_fortran):
        """Benchmark Fortran implementation on medium domain."""
        nijt, nkt = 2500, 40
        total_points = nijt * nkt

        def run_rain_ice():
            rain_ice_fortran(
                timestep=np.float32(50.0),
                krr=6,
                pexn=swap_to_cython(medium_data_fortran['pexn'], nkt, nijt),
                rhodref=swap_to_cython(medium_data_fortran['prhodref'], nkt, nijt),
                pres=swap_to_cython(medium_data_fortran['ppabst'], nkt, nijt),
                dzz=swap_to_cython(medium_data_fortran['pdzz'], nkt, nijt),
                th_t=swap_to_cython(medium_data_fortran['ptht'], nkt, nijt),
                rv_t=swap_to_cython(medium_data_fortran['prvt'], nkt, nijt),
                rc_t=swap_to_cython(medium_data_fortran['prct'], nkt, nijt),
                rr_t=swap_to_cython(medium_data_fortran['prrt'], nkt, nijt),
                ri_t=swap_to_cython(medium_data_fortran['prit'], nkt, nijt),
                rs_t=swap_to_cython(medium_data_fortran['prst'], nkt, nijt),
                rg_t=swap_to_cython(medium_data_fortran['prgt'], nkt, nijt),
                ci_t=swap_to_cython(medium_data_fortran['pcit'], nkt, nijt),
                ths=swap_to_cython(medium_data_fortran['pths'], nkt, nijt),
                rvs=swap_to_cython(medium_data_fortran['prvs'], nkt, nijt),
                rcs=swap_to_cython(medium_data_fortran['prcs'], nkt, nijt),
                rrs=swap_to_cython(medium_data_fortran['prrs'], nkt, nijt),
                ris=swap_to_cython(medium_data_fortran['pris'], nkt, nijt),
                rss=swap_to_cython(medium_data_fortran['prss'], nkt, nijt),
                rgs=swap_to_cython(medium_data_fortran['prgs'], nkt, nijt),
                cis=swap_to_cython(medium_data_fortran['pcis'], nkt, nijt),
                sigs=swap_to_cython(medium_data_fortran['psigs'], nkt, nijt)
            )

        result = benchmark(run_rain_ice)

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

    def test_jax_medium(self, benchmark, rain_ice_jax_cpu, medium_data_jax):
        """Benchmark JAX implementation on medium domain (CPU)."""
        nx = int(np.sqrt(2500))
        ny = 2500 // nx
        nz = 40
        total_points = nx * ny * nz

        state = medium_data_jax['state']
        dt = medium_data_jax['timestep']

        # Warm-up to trigger JIT compilation (multiple iterations)
        for _ in range(3):
            _ = rain_ice_jax_cpu(state, dt)

        # Block until all JAX operations complete
        if JAX_AVAILABLE:
            import jax
            jax.block_until_ready(rain_ice_jax_cpu(state, dt))

        result = benchmark(lambda: rain_ice_jax_cpu(state, dt))

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

    def test_fortran_large(self, benchmark, init_rain_ice_fortran, large_data_fortran):
        """Benchmark Fortran implementation on large domain."""
        nijt, nkt = 10000, 60
        total_points = nijt * nkt

        def run_rain_ice():
            rain_ice_fortran(
                timestep=np.float32(50.0),
                krr=6,
                pexn=swap_to_cython(large_data_fortran['pexn'], nkt, nijt),
                rhodref=swap_to_cython(large_data_fortran['prhodref'], nkt, nijt),
                pres=swap_to_cython(large_data_fortran['ppabst'], nkt, nijt),
                dzz=swap_to_cython(large_data_fortran['pdzz'], nkt, nijt),
                th_t=swap_to_cython(large_data_fortran['ptht'], nkt, nijt),
                rv_t=swap_to_cython(large_data_fortran['prvt'], nkt, nijt),
                rc_t=swap_to_cython(large_data_fortran['prct'], nkt, nijt),
                rr_t=swap_to_cython(large_data_fortran['prrt'], nkt, nijt),
                ri_t=swap_to_cython(large_data_fortran['prit'], nkt, nijt),
                rs_t=swap_to_cython(large_data_fortran['prst'], nkt, nijt),
                rg_t=swap_to_cython(large_data_fortran['prgt'], nkt, nijt),
                ci_t=swap_to_cython(large_data_fortran['pcit'], nkt, nijt),
                ths=swap_to_cython(large_data_fortran['pths'], nkt, nijt),
                rvs=swap_to_cython(large_data_fortran['prvs'], nkt, nijt),
                rcs=swap_to_cython(large_data_fortran['prcs'], nkt, nijt),
                rrs=swap_to_cython(large_data_fortran['prrs'], nkt, nijt),
                ris=swap_to_cython(large_data_fortran['pris'], nkt, nijt),
                rss=swap_to_cython(large_data_fortran['prss'], nkt, nijt),
                rgs=swap_to_cython(large_data_fortran['prgs'], nkt, nijt),
                cis=swap_to_cython(large_data_fortran['pcis'], nkt, nijt),
                sigs=swap_to_cython(large_data_fortran['psigs'], nkt, nijt)
            )

        result = benchmark(run_rain_ice)

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

    def test_jax_large(self, benchmark, rain_ice_jax_cpu, large_data_jax):
        """Benchmark JAX implementation on large domain (CPU)."""
        nx = int(np.sqrt(10000))
        ny = 10000 // nx
        nz = 60
        total_points = nx * ny * nz

        state = large_data_jax['state']
        dt = large_data_jax['timestep']

        # Warm-up to trigger JIT compilation (multiple iterations)
        for _ in range(3):
            _ = rain_ice_jax_cpu(state, dt)

        # Block until all JAX operations complete
        if JAX_AVAILABLE:
            import jax
            jax.block_until_ready(rain_ice_jax_cpu(state, dt))

        result = benchmark(lambda: rain_ice_jax_cpu(state, dt))

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
