# -*- coding: utf-8 -*-
"""Benchmark comparing Fortran and JAX implementations of SHALLOW_CONVECTION on CPU."""
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
    from ice3._phyex_wrapper import shallow_convection as shallow_convection_fortran
    FORTRAN_AVAILABLE = True
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "PHYEX-IAL_CY50T1/bridge"))
    try:
        from _phyex_wrapper import shallow_convection as shallow_convection_fortran
        FORTRAN_AVAILABLE = True
    except ImportError:
        FORTRAN_AVAILABLE = False
        shallow_convection_fortran = None

# Try to import JAX implementation
try:
    import os
    import jax
    jax.config.update('jax_platform_name', os.environ.get('JAX_PLATFORM_NAME', 'cpu'))
    import jax.numpy as jnp
    from ice3.jax.convection.shallow_convection import shallow_convection as shallow_convection_jax
    from ice3.phyex_common.phyex import Phyex
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    shallow_convection_jax = None
    Phyex = None


def create_test_atmosphere_shallow_conv(nlon=100, nlev=60, backend='fortran'):
    """
    Create realistic atmospheric test data for SHALLOW_CONVECTION benchmarks.

    Parameters
    ----------
    nlon : int
        Number of horizontal points
    nlev : int
        Number of vertical levels
    backend : str
        'fortran' or 'jax' - determines output format

    Returns
    -------
    dict
        Dictionary with all required fields in the appropriate format
    """
    # Create vertical coordinate (0-15 km)
    z = np.linspace(0, 15000, nlev, dtype=np.float32)

    # Standard atmosphere parameters
    p0 = 101325.0  # Pa
    T0 = 288.15    # K
    gamma = 0.0065  # K/m

    # Pressure profile
    pressure = p0 * (1 - gamma * z / T0) ** 5.26
    ppabst = np.tile(pressure, (nlon, 1)).astype(np.float32)

    # Temperature profile
    temperature = T0 - gamma * z
    ptt = np.tile(temperature, (nlon, 1)).astype(np.float32)

    # Add variability
    np.random.seed(42)
    ptt += np.random.randn(nlon, nlev).astype(np.float32) * 0.5

    # Water vapor (decreasing with height)
    rv_surf = 0.015  # 15 g/kg
    rv = rv_surf * np.exp(-z / 2000)  # Scale height 2km
    prvt = np.tile(rv, (nlon, 1)).astype(np.float32)
    prvt += np.abs(np.random.randn(nlon, nlev).astype(np.float32)) * 0.002

    # Cloud water at mid-levels (2-6 km)
    prct = np.zeros((nlon, nlev), dtype=np.float32)
    cloud_mask = (z > 2000) & (z < 6000)
    for i in range(nlon):
        prct[i, cloud_mask] = np.abs(np.random.rand(cloud_mask.sum())).astype(np.float32) * 0.001

    # Ice at upper levels (> 5 km)
    prit = np.zeros((nlon, nlev), dtype=np.float32)
    ice_mask = z > 5000
    for i in range(nlon):
        prit[i, ice_mask] = np.abs(np.random.rand(ice_mask.sum())).astype(np.float32) * 0.0005

    # Vertical velocity (weak updraft)
    pwt = np.full((nlon, nlev), 0.1, dtype=np.float32)

    # TKE in cloud layer
    ptkecls = np.full(nlon, 0.5, dtype=np.float32)

    # Height
    pzz = np.tile(z, (nlon, 1)).astype(np.float32)

    # Initialize output arrays
    ptten = np.zeros((nlon, nlev), dtype=np.float32)
    prvten = np.zeros((nlon, nlev), dtype=np.float32)
    prcten = np.zeros((nlon, nlev), dtype=np.float32)
    priten = np.zeros((nlon, nlev), dtype=np.float32)
    kcltop = np.zeros(nlon, dtype=np.int32)
    kclbas = np.zeros(nlon, dtype=np.int32)
    pumf = np.zeros((nlon, nlev), dtype=np.float32)

    # Chemical tracer arrays (minimal)
    kch1 = 1
    pch1 = np.zeros((nlon, nlev, kch1), dtype=np.float32)
    pch1ten = np.zeros((nlon, nlev, kch1), dtype=np.float32)

    if backend == 'fortran':
        # Fortran expects Fortran-contiguous arrays
        return {
            'ppabst': np.asfortranarray(ppabst),
            'pzz': np.asfortranarray(pzz),
            'ptkecls': np.asfortranarray(ptkecls),
            'ptt': np.asfortranarray(ptt),
            'prvt': np.asfortranarray(prvt),
            'prct': np.asfortranarray(prct),
            'prit': np.asfortranarray(prit),
            'pwt': np.asfortranarray(pwt),
            'ptten': np.asfortranarray(ptten),
            'prvten': np.asfortranarray(prvten),
            'prcten': np.asfortranarray(prcten),
            'priten': np.asfortranarray(priten),
            'kcltop': np.asfortranarray(kcltop),
            'kclbas': np.asfortranarray(kclbas),
            'pumf': np.asfortranarray(pumf),
            'pch1': np.asfortranarray(pch1),
            'pch1ten': np.asfortranarray(pch1ten),
            'kice': 1,
            'kbdia': 1,
            'ktdia': 1,
            'osettadj': False,
            'ptadjs': 10800.0,
            'och1conv': False,
            'kch1': kch1,
        }
    elif backend == 'jax':
        # JAX uses C-contiguous arrays
        return {
            'ppabst': jnp.asarray(ppabst),
            'pzz': jnp.asarray(pzz),
            'ptkecls': jnp.asarray(ptkecls),
            'ptt': jnp.asarray(ptt),
            'prvt': jnp.asarray(prvt),
            'prct': jnp.asarray(prct),
            'prit': jnp.asarray(prit),
            'pwt': jnp.asarray(pwt),
            'ptten': jnp.asarray(ptten),
            'prvten': jnp.asarray(prvten),
            'prcten': jnp.asarray(prcten),
            'priten': jnp.asarray(priten),
            'kcltop': jnp.asarray(kcltop),
            'kclbas': jnp.asarray(kclbas),
            'pumf': jnp.asarray(pumf),
            'pch1': jnp.asarray(pch1),
            'pch1ten': jnp.asarray(pch1ten),
            'kice': 1,
            'kbdia': 1,
            'ktdia': 1,
            'osettadj': False,
            'ptadjs': 10800.0,
            'och1conv': False,
        }


@pytest.fixture
def small_data_fortran():
    """Small domain for Fortran (50 horizontal points, 30 levels)."""
    return create_test_atmosphere_shallow_conv(nlon=50, nlev=30, backend='fortran')


@pytest.fixture
def small_data_jax():
    """Small domain for JAX (50 horizontal points, 30 levels)."""
    return create_test_atmosphere_shallow_conv(nlon=50, nlev=30, backend='jax')


@pytest.fixture
def medium_data_fortran():
    """Medium domain for Fortran (100 horizontal points, 60 levels)."""
    return create_test_atmosphere_shallow_conv(nlon=100, nlev=60, backend='fortran')


@pytest.fixture
def medium_data_jax():
    """Medium domain for JAX (100 horizontal points, 60 levels)."""
    return create_test_atmosphere_shallow_conv(nlon=100, nlev=60, backend='jax')


@pytest.fixture
def large_data_fortran():
    """Large domain for Fortran (500 horizontal points, 90 levels)."""
    return create_test_atmosphere_shallow_conv(nlon=500, nlev=90, backend='fortran')


@pytest.fixture
def large_data_jax():
    """Large domain for JAX (500 horizontal points, 90 levels)."""
    return create_test_atmosphere_shallow_conv(nlon=500, nlev=90, backend='jax')


@pytest.fixture
def convection_params():
    """Create convection parameters for JAX."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    from ice3.jax.convection.shallow_convection import ConvectionParameters
    # Use default AROME parameters
    return ConvectionParameters()


# ============================================================================
# Small Domain Benchmarks (50 points × 30 levels = 1,500 grid points)
# ============================================================================

@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrapper not available")
class TestSmallDomainFortran:
    """Fortran benchmarks on small domain."""

    def test_fortran_small(self, benchmark, small_data_fortran):
        """Benchmark Fortran implementation on small domain."""
        nlon, nlev = 50, 30
        total_points = nlon * nlev

        def run_shallow_conv():
            shallow_convection_fortran(
                kice=small_data_fortran['kice'],
                kbdia=small_data_fortran['kbdia'],
                ktdia=small_data_fortran['ktdia'],
                osettadj=small_data_fortran['osettadj'],
                ptadjs=small_data_fortran['ptadjs'],
                och1conv=small_data_fortran['och1conv'],
                kch1=small_data_fortran['kch1'],
                ptkecls=small_data_fortran['ptkecls'],
                ppabst=small_data_fortran['ppabst'],
                pzz=small_data_fortran['pzz'],
                ptt=small_data_fortran['ptt'],
                prvt=small_data_fortran['prvt'],
                prct=small_data_fortran['prct'],
                prit=small_data_fortran['prit'],
                pwt=small_data_fortran['pwt'],
                ptten=small_data_fortran['ptten'],
                prvten=small_data_fortran['prvten'],
                prcten=small_data_fortran['prcten'],
                priten=small_data_fortran['priten'],
                kcltop=small_data_fortran['kcltop'],
                kclbas=small_data_fortran['kclbas'],
                pumf=small_data_fortran['pumf'],
                pch1=small_data_fortran['pch1'],
                pch1ten=small_data_fortran['pch1ten']
            )

        result = benchmark(run_shallow_conv)

        # Print statistics (only when benchmark is enabled)
        if hasattr(result, 'stats'):
            mean_time = result.stats['mean']
            throughput = total_points / mean_time
            print(f"\n[Fortran] Small domain ({nlon}×{nlev}={total_points:,} points)")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e6:.2f} M points/s")


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestSmallDomainJAX:
    """JAX benchmarks on small domain."""

    def test_jax_small(self, benchmark, small_data_jax, convection_params):
        """Benchmark JAX implementation on small domain (CPU)."""
        nlon, nlev = 50, 30
        total_points = nlon * nlev

        # Warm-up to trigger JIT compilation (multiple iterations)
        for _ in range(3):
            _ = shallow_convection_jax(**small_data_jax, convection_params=convection_params)

        # Block until all JAX operations complete
        if JAX_AVAILABLE:
            import jax
            jax.block_until_ready(shallow_convection_jax(**small_data_jax, convection_params=convection_params))

        result = benchmark(lambda: shallow_convection_jax(**small_data_jax, convection_params=convection_params))

        # Print statistics (only when benchmark is enabled)
        if hasattr(result, 'stats'):
            mean_time = result.stats['mean']
            throughput = total_points / mean_time
            print(f"\n[JAX CPU] Small domain ({nlon}×{nlev}={total_points:,} points)")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e6:.2f} M points/s")


# ============================================================================
# Medium Domain Benchmarks (100 points × 60 levels = 6,000 grid points)
# ============================================================================

@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrapper not available")
class TestMediumDomainFortran:
    """Fortran benchmarks on medium domain."""

    def test_fortran_medium(self, benchmark, medium_data_fortran):
        """Benchmark Fortran implementation on medium domain."""
        nlon, nlev = 100, 60
        total_points = nlon * nlev

        def run_shallow_conv():
            shallow_convection_fortran(
                kice=medium_data_fortran['kice'],
                kbdia=medium_data_fortran['kbdia'],
                ktdia=medium_data_fortran['ktdia'],
                osettadj=medium_data_fortran['osettadj'],
                ptadjs=medium_data_fortran['ptadjs'],
                och1conv=medium_data_fortran['och1conv'],
                kch1=medium_data_fortran['kch1'],
                ptkecls=medium_data_fortran['ptkecls'],
                ppabst=medium_data_fortran['ppabst'],
                pzz=medium_data_fortran['pzz'],
                ptt=medium_data_fortran['ptt'],
                prvt=medium_data_fortran['prvt'],
                prct=medium_data_fortran['prct'],
                prit=medium_data_fortran['prit'],
                pwt=medium_data_fortran['pwt'],
                ptten=medium_data_fortran['ptten'],
                prvten=medium_data_fortran['prvten'],
                prcten=medium_data_fortran['prcten'],
                priten=medium_data_fortran['priten'],
                kcltop=medium_data_fortran['kcltop'],
                kclbas=medium_data_fortran['kclbas'],
                pumf=medium_data_fortran['pumf'],
                pch1=medium_data_fortran['pch1'],
                pch1ten=medium_data_fortran['pch1ten']
            )

        result = benchmark(run_shallow_conv)

        # Print statistics (only when benchmark is enabled)
        if hasattr(result, 'stats'):
            mean_time = result.stats['mean']
            throughput = total_points / mean_time
            print(f"\n[Fortran] Medium domain ({nlon}×{nlev}={total_points:,} points)")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e6:.2f} M points/s")


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestMediumDomainJAX:
    """JAX benchmarks on medium domain."""

    def test_jax_medium(self, benchmark, medium_data_jax, convection_params):
        """Benchmark JAX implementation on medium domain (CPU)."""
        nlon, nlev = 100, 60
        total_points = nlon * nlev

        # Warm-up to trigger JIT compilation (multiple iterations)
        for _ in range(3):
            _ = shallow_convection_jax(**medium_data_jax, convection_params=convection_params)

        # Block until all JAX operations complete
        if JAX_AVAILABLE:
            import jax
            jax.block_until_ready(shallow_convection_jax(**medium_data_jax, convection_params=convection_params))

        result = benchmark(lambda: shallow_convection_jax(**medium_data_jax, convection_params=convection_params))

        # Print statistics (only when benchmark is enabled)
        if hasattr(result, 'stats'):
            mean_time = result.stats['mean']
            throughput = total_points / mean_time
            print(f"\n[JAX CPU] Medium domain ({nlon}×{nlev}={total_points:,} points)")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e6:.2f} M points/s")


# ============================================================================
# Large Domain Benchmarks (500 points × 90 levels = 45,000 grid points)
# ============================================================================

@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrapper not available")
class TestLargeDomainFortran:
    """Fortran benchmarks on large domain."""

    def test_fortran_large(self, benchmark, large_data_fortran):
        """Benchmark Fortran implementation on large domain."""
        nlon, nlev = 500, 90
        total_points = nlon * nlev

        def run_shallow_conv():
            shallow_convection_fortran(
                kice=large_data_fortran['kice'],
                kbdia=large_data_fortran['kbdia'],
                ktdia=large_data_fortran['ktdia'],
                osettadj=large_data_fortran['osettadj'],
                ptadjs=large_data_fortran['ptadjs'],
                och1conv=large_data_fortran['och1conv'],
                kch1=large_data_fortran['kch1'],
                ptkecls=large_data_fortran['ptkecls'],
                ppabst=large_data_fortran['ppabst'],
                pzz=large_data_fortran['pzz'],
                ptt=large_data_fortran['ptt'],
                prvt=large_data_fortran['prvt'],
                prct=large_data_fortran['prct'],
                prit=large_data_fortran['prit'],
                pwt=large_data_fortran['pwt'],
                ptten=large_data_fortran['ptten'],
                prvten=large_data_fortran['prvten'],
                prcten=large_data_fortran['prcten'],
                priten=large_data_fortran['priten'],
                kcltop=large_data_fortran['kcltop'],
                kclbas=large_data_fortran['kclbas'],
                pumf=large_data_fortran['pumf'],
                pch1=large_data_fortran['pch1'],
                pch1ten=large_data_fortran['pch1ten']
            )

        result = benchmark(run_shallow_conv)

        # Print statistics (only when benchmark is enabled)
        if hasattr(result, 'stats'):
            mean_time = result.stats['mean']
            throughput = total_points / mean_time
            print(f"\n[Fortran] Large domain ({nlon}×{nlev}={total_points:,} points)")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e6:.2f} M points/s")

            if hasattr(result.stats, 'stddev'):
                print(f"  Std dev: {result.stats['stddev']*1000:.3f} ms")


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestLargeDomainJAX:
    """JAX benchmarks on large domain."""

    def test_jax_large(self, benchmark, large_data_jax, convection_params):
        """Benchmark JAX implementation on large domain (CPU)."""
        nlon, nlev = 500, 90
        total_points = nlon * nlev

        # Warm-up to trigger JIT compilation (multiple iterations)
        for _ in range(3):
            _ = shallow_convection_jax(**large_data_jax, convection_params=convection_params)

        # Block until all JAX operations complete
        if JAX_AVAILABLE:
            import jax
            jax.block_until_ready(shallow_convection_jax(**large_data_jax, convection_params=convection_params))

        result = benchmark(lambda: shallow_convection_jax(**large_data_jax, convection_params=convection_params))

        # Print statistics (only when benchmark is enabled)
        if hasattr(result, 'stats'):
            mean_time = result.stats['mean']
            throughput = total_points / mean_time
            print(f"\n[JAX CPU] Large domain ({nlon}×{nlev}={total_points:,} points)")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e6:.2f} M points/s")

            if hasattr(result.stats, 'stddev'):
                print(f"  Std dev: {result.stats['stddev']*1000:.3f} ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only", "--benchmark-group-by=group"])
