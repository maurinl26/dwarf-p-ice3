# -*- coding: utf-8 -*-
"""Performance tests for Fortran CPU implementation of RAIN_ICE component."""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add build directory to path
build_dir = Path(__file__).parent.parent.parent / 'build'
if build_dir.exists():
    for sub in build_dir.iterdir():
        if sub.is_dir() and sub.name.startswith('cp'):
            sys.path.insert(0, str(sub))
            break

try:
    from ice3._phyex_wrapper import rain_ice, init_rain_ice
    FORTRAN_AVAILABLE = True
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "PHYEX-IAL_CY50T1/bridge"))
    try:
        from _phyex_wrapper import rain_ice, init_rain_ice
        FORTRAN_AVAILABLE = True
    except ImportError:
        FORTRAN_AVAILABLE = False
        rain_ice = None
        init_rain_ice = None


def create_fortran_rain_ice_test_data(nijt=2500, nkt=60):
    """
    Create realistic atmospheric test data for Fortran RAIN_ICE.

    Parameters
    ----------
    nijt : int
        Number of horizontal points (flattened)
    nkt : int
        Number of vertical levels

    Returns
    -------
    dict
        Dictionary with all required fields (Fortran-contiguous, float32)
    """
    z = np.linspace(0, 10000, nkt, dtype=np.float32)

    # Standard atmosphere
    p0 = 101325.0
    T0 = 288.15
    gamma = 0.0065

    # Physical constants
    Rd = 287.0
    cp = 1004.0
    p00 = 100000.0

    data = {}

    # Pressure profile
    pressure = p0 * (1 - gamma * z / T0) ** 5.26
    data['ppabst'] = np.tile(pressure, (nijt, 1)).T.astype(np.float32).copy(order='F')

    # Temperature profile
    temperature = T0 - gamma * z
    data['temperature'] = np.tile(temperature, (nijt, 1)).T.astype(np.float32).copy(order='F')

    # Add variability
    np.random.seed(42)
    data['temperature'] += (np.random.randn(nkt, nijt) * 0.5).astype(np.float32)
    data['ppabst'] += (np.random.randn(nkt, nijt) * 100).astype(np.float32)

    # Exner function
    data['pexn'] = np.asfortranarray((data['ppabst'] / p00) ** (Rd / cp), dtype=np.float32)
    data['ptht'] = np.asfortranarray(data['temperature'] / data['pexn'], dtype=np.float32)

    # Reference values
    data['pexnref'] = np.asfortranarray(data['pexn'].copy(), dtype=np.float32)
    data['prhodref'] = np.asfortranarray(data['ppabst'] / (Rd * data['temperature']), dtype=np.float32)
    data['prhodj'] = np.asfortranarray(data['prhodref'].copy(), dtype=np.float32)

    # Layer thickness (100m per level)
    data['pdzz'] = np.full((nkt, nijt), 100.0, dtype=np.float32, order='F')

    # Water vapor
    rv_surf = 0.015
    data['prvt'] = (rv_surf * np.exp(-z / 2000)).astype(np.float32)
    data['prvt'] = np.tile(data['prvt'], (nijt, 1)).T.astype(np.float32).copy(order='F')
    data['prvt'] += (np.abs(np.random.randn(nkt, nijt)) * 0.002).astype(np.float32)

    # Cloud water
    data['prct'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    cloud_levels = (z > 2000) & (z < 6000)
    for i in range(nijt):
        data['prct'][cloud_levels, i] = np.abs(np.random.rand(cloud_levels.sum())).astype(np.float32) * 0.003

    # Rain
    data['prrt'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    rain_levels = (z > 1000) & (z < 4000)
    for i in range(nijt):
        data['prrt'][rain_levels, i] = np.abs(np.random.rand(rain_levels.sum())).astype(np.float32) * 0.001

    # Ice
    data['prit'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    ice_levels = z > 5000
    for i in range(nijt):
        data['prit'][ice_levels, i] = np.abs(np.random.rand(ice_levels.sum())).astype(np.float32) * 0.002

    # Snow
    data['prst'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    snow_levels = (z > 3000) & (z < 7000)
    for i in range(nijt):
        data['prst'][snow_levels, i] = np.abs(np.random.rand(snow_levels.sum())).astype(np.float32) * 0.001

    # Graupel
    data['prgt'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    graupel_levels = (z > 2000) & (z < 6000)
    for i in range(nijt):
        data['prgt'][graupel_levels, i] = np.abs(np.random.rand(graupel_levels.sum())).astype(np.float32) * 0.0005

    # Ice concentration
    data['pcit'] = np.where(data['prit'] > 0, 1000.0, 0.0).astype(np.float32)

    # Tendencies
    data['pths'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['prvs'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['prcs'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['prrs'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['pris'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['prss'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['prgs'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['pcis'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')

    # Subgrid turbulence
    data['psigs'] = np.full((nkt, nijt), 0.1, dtype=np.float32, order='F')

    # Cloud fraction (for input)
    data['pcldfr'] = np.where(data['prct'] > 0, 0.5, 0.0).astype(np.float32)

    # HLCLOUDS arrays (high-res diagnostics)
    data['phlc_hrc'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['phlc_hcf'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['phli_hri'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['phli_hcf'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')

    return data


def swap_to_cython(arr, nkt, nijt):
    """Convert from (nkt, nijt) to (nijt, nkt) for Cython wrapper."""
    if arr.ndim == 2 and arr.shape[0] == nkt:
        return np.asfortranarray(arr.T, dtype=np.float32)
    return np.asfortranarray(arr, dtype=np.float32)


@pytest.fixture(scope="module")
def init_rain_ice_module():
    """Initialize RAIN_ICE once for all tests."""
    if FORTRAN_AVAILABLE and init_rain_ice is not None:
        init_rain_ice(timestep=50.0, dzmin=60.0, krr=6, scheme="ICE3")
    return True


@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrapper not available")
class TestRainIceFortranPerformanceSmall:
    """Performance tests on small domain."""

    def test_small_domain(self, benchmark, init_rain_ice_module):
        """Benchmark small domain."""
        print("\n" + "="*75)
        print("Fortran RAIN_ICE Performance: Small Domain (100x20)")
        print("="*75)

        nijt = 100
        nkt = 20
        total_points = nijt * nkt

        print(f"Horizontal points: {nijt:,}")
        print(f"Vertical levels: {nkt}")
        print(f"Total grid points: {total_points:,}")

        data = create_fortran_rain_ice_test_data(nijt, nkt)

        def run_rain_ice():
            rain_ice(
                timestep=np.float32(50.0),
                krr=6,
                pexn=swap_to_cython(data['pexn'], nkt, nijt),
                rhodref=swap_to_cython(data['prhodref'], nkt, nijt),
                pres=swap_to_cython(data['ppabst'], nkt, nijt),
                dzz=swap_to_cython(data['pdzz'], nkt, nijt),
                th_t=swap_to_cython(data['ptht'], nkt, nijt),
                rv_t=swap_to_cython(data['prvt'], nkt, nijt),
                rc_t=swap_to_cython(data['prct'], nkt, nijt),
                rr_t=swap_to_cython(data['prrt'], nkt, nijt),
                ri_t=swap_to_cython(data['prit'], nkt, nijt),
                rs_t=swap_to_cython(data['prst'], nkt, nijt),
                rg_t=swap_to_cython(data['prgt'], nkt, nijt),
                ci_t=swap_to_cython(data['pcit'], nkt, nijt),
                ths=swap_to_cython(data['pths'], nkt, nijt),
                rvs=swap_to_cython(data['prvs'], nkt, nijt),
                rcs=swap_to_cython(data['prcs'], nkt, nijt),
                rrs=swap_to_cython(data['prrs'], nkt, nijt),
                ris=swap_to_cython(data['pris'], nkt, nijt),
                rss=swap_to_cython(data['prss'], nkt, nijt),
                rgs=swap_to_cython(data['prgs'], nkt, nijt),
                cis=swap_to_cython(data['pcis'], nkt, nijt),
                sigs=swap_to_cython(data['psigs'], nkt, nijt)
            )

        result = benchmark(run_rain_ice)

        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")
        print(f"  Performance: {throughput*50.0/1e6:.2f} M point-steps/s")


@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrapper not available")
class TestRainIceFortranPerformanceMedium:
    """Performance tests on medium domain."""

    def test_medium_domain(self, benchmark, init_rain_ice_module):
        """Benchmark medium domain."""
        print("\n" + "="*75)
        print("Fortran RAIN_ICE Performance: Medium Domain (2500x40)")
        print("="*75)

        nijt = 2500
        nkt = 40
        total_points = nijt * nkt

        print(f"Horizontal points: {nijt:,}")
        print(f"Vertical levels: {nkt}")
        print(f"Total grid points: {total_points:,}")

        data = create_fortran_rain_ice_test_data(nijt, nkt)

        def run_rain_ice():
            rain_ice(
                timestep=np.float32(50.0),
                krr=6,
                pexn=swap_to_cython(data['pexn'], nkt, nijt),
                rhodref=swap_to_cython(data['prhodref'], nkt, nijt),
                pres=swap_to_cython(data['ppabst'], nkt, nijt),
                dzz=swap_to_cython(data['pdzz'], nkt, nijt),
                th_t=swap_to_cython(data['ptht'], nkt, nijt),
                rv_t=swap_to_cython(data['prvt'], nkt, nijt),
                rc_t=swap_to_cython(data['prct'], nkt, nijt),
                rr_t=swap_to_cython(data['prrt'], nkt, nijt),
                ri_t=swap_to_cython(data['prit'], nkt, nijt),
                rs_t=swap_to_cython(data['prst'], nkt, nijt),
                rg_t=swap_to_cython(data['prgt'], nkt, nijt),
                ci_t=swap_to_cython(data['pcit'], nkt, nijt),
                ths=swap_to_cython(data['pths'], nkt, nijt),
                rvs=swap_to_cython(data['prvs'], nkt, nijt),
                rcs=swap_to_cython(data['prcs'], nkt, nijt),
                rrs=swap_to_cython(data['prrs'], nkt, nijt),
                ris=swap_to_cython(data['pris'], nkt, nijt),
                rss=swap_to_cython(data['prss'], nkt, nijt),
                rgs=swap_to_cython(data['prgs'], nkt, nijt),
                cis=swap_to_cython(data['pcis'], nkt, nijt),
                sigs=swap_to_cython(data['psigs'], nkt, nijt)
            )

        result = benchmark(run_rain_ice)

        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")
        print(f"  Performance: {throughput*50.0/1e6:.2f} M point-steps/s")


@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrapper not available")
class TestRainIceFortranPerformanceLarge:
    """Performance tests on large domain."""

    def test_large_domain(self, benchmark, init_rain_ice_module):
        """Benchmark large domain."""
        print("\n" + "="*75)
        print("Fortran RAIN_ICE Performance: Large Domain (10000x60)")
        print("="*75)

        nijt = 10000
        nkt = 60
        total_points = nijt * nkt

        print(f"Horizontal points: {nijt:,}")
        print(f"Vertical levels: {nkt}")
        print(f"Total grid points: {total_points:,}")

        data = create_fortran_rain_ice_test_data(nijt, nkt)

        def run_rain_ice():
            rain_ice(
                timestep=np.float32(50.0),
                krr=6,
                pexn=swap_to_cython(data['pexn'], nkt, nijt),
                rhodref=swap_to_cython(data['prhodref'], nkt, nijt),
                pres=swap_to_cython(data['ppabst'], nkt, nijt),
                dzz=swap_to_cython(data['pdzz'], nkt, nijt),
                th_t=swap_to_cython(data['ptht'], nkt, nijt),
                rv_t=swap_to_cython(data['prvt'], nkt, nijt),
                rc_t=swap_to_cython(data['prct'], nkt, nijt),
                rr_t=swap_to_cython(data['prrt'], nkt, nijt),
                ri_t=swap_to_cython(data['prit'], nkt, nijt),
                rs_t=swap_to_cython(data['prst'], nkt, nijt),
                rg_t=swap_to_cython(data['prgt'], nkt, nijt),
                ci_t=swap_to_cython(data['pcit'], nkt, nijt),
                ths=swap_to_cython(data['pths'], nkt, nijt),
                rvs=swap_to_cython(data['prvs'], nkt, nijt),
                rcs=swap_to_cython(data['prcs'], nkt, nijt),
                rrs=swap_to_cython(data['prrs'], nkt, nijt),
                ris=swap_to_cython(data['pris'], nkt, nijt),
                rss=swap_to_cython(data['prss'], nkt, nijt),
                rgs=swap_to_cython(data['prgs'], nkt, nijt),
                cis=swap_to_cython(data['pcis'], nkt, nijt),
                sigs=swap_to_cython(data['psigs'], nkt, nijt)
            )

        result = benchmark(run_rain_ice)

        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")
        print(f"  Performance: {throughput*50.0/1e6:.2f} M point-steps/s")

        if hasattr(result.stats, 'stddev'):
            print(f"  Std dev: {result.stats['stddev']*1000:.3f} ms")
        if hasattr(result.stats, 'min'):
            print(f"  Min time: {result.stats['min']*1000:.3f} ms")
        if hasattr(result.stats, 'max'):
            print(f"  Max time: {result.stats['max']*1000:.3f} ms")


@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrapper not available")
class TestRainIceFortranScaling:
    """Test scaling characteristics."""

    def test_scaling_with_domain_size(self, init_rain_ice_module):
        """Test how performance scales with domain size."""
        print("\n" + "="*75)
        print("Fortran RAIN_ICE: Scaling Analysis")
        print("="*75)

        sizes = [
            (100, 10),
            (400, 20),
            (1000, 30),
        ]

        results = []

        for nijt, nkt in sizes:
            data = create_fortran_rain_ice_test_data(nijt, nkt)
            total_points = nijt * nkt

            # Time single execution
            import time
            start = time.time()

            rain_ice(
                timestep=np.float32(50.0),
                krr=6,
                pexn=swap_to_cython(data['pexn'], nkt, nijt),
                rhodref=swap_to_cython(data['prhodref'], nkt, nijt),
                pres=swap_to_cython(data['ppabst'], nkt, nijt),
                dzz=swap_to_cython(data['pdzz'], nkt, nijt),
                th_t=swap_to_cython(data['ptht'], nkt, nijt),
                rv_t=swap_to_cython(data['prvt'], nkt, nijt),
                rc_t=swap_to_cython(data['prct'], nkt, nijt),
                rr_t=swap_to_cython(data['prrt'], nkt, nijt),
                ri_t=swap_to_cython(data['prit'], nkt, nijt),
                rs_t=swap_to_cython(data['prst'], nkt, nijt),
                rg_t=swap_to_cython(data['prgt'], nkt, nijt),
                ci_t=swap_to_cython(data['pcit'], nkt, nijt),
                ths=swap_to_cython(data['pths'], nkt, nijt),
                rvs=swap_to_cython(data['prvs'], nkt, nijt),
                rcs=swap_to_cython(data['prcs'], nkt, nijt),
                rrs=swap_to_cython(data['prrs'], nkt, nijt),
                ris=swap_to_cython(data['pris'], nkt, nijt),
                rss=swap_to_cython(data['prss'], nkt, nijt),
                rgs=swap_to_cython(data['prgs'], nkt, nijt),
                cis=swap_to_cython(data['pcis'], nkt, nijt),
                sigs=swap_to_cython(data['psigs'], nkt, nijt)
            )

            elapsed = time.time() - start
            throughput = total_points / elapsed
            results.append((total_points, elapsed, throughput))

            print(f"\nDomain {nijt}x{nkt}: {total_points:,} points")
            print(f"  Time: {elapsed*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e6:.2f} M points/s")

        # Check scaling
        print("\n" + "-"*75)
        print("Scaling Summary:")
        print("-"*75)
        for i, (points, time_i, tput) in enumerate(results):
            if i > 0:
                size_ratio = points / results[0][0]
                time_ratio = time_i / results[0][1]
                print(f"  {points:>10,} points: {time_ratio:.2f}x time for {size_ratio:.2f}x points")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
