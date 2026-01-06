# -*- coding: utf-8 -*-
"""Performance tests for JAX implementation of RAIN_ICE component."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from ice3.jax.rain_ice import RainIceJAX
from ice3.phyex_common.phyex import Phyex


def create_rain_ice_jax_test_data(shape=(100, 100, 60)):
    """
    Create realistic atmospheric test data for JAX RAIN_ICE performance tests.

    Parameters
    ----------
    shape : tuple
        Domain shape (nx, ny, nz)

    Returns
    -------
    dict
        Dictionary with all required JAX arrays
    """
    nx, ny, nz = shape

    # Create vertical coordinate (0-10 km)
    z = jnp.linspace(0, 10000, nz)

    # Standard atmosphere
    p0 = 101325.0  # Pa
    T0 = 288.15    # K
    gamma = 0.0065  # K/m

    # Pressure profile
    pressure = p0 * (1 - gamma * z / T0) ** 5.26
    pres = jnp.tile(pressure, (nx, ny, 1))

    # Temperature profile
    temperature = T0 - gamma * z
    temperature = jnp.tile(temperature, (nx, ny, 1))

    # Add variability
    key = jax.random.PRNGKey(42)
    temp_noise = jax.random.normal(key, shape) * 0.5
    temperature = temperature + temp_noise

    # Exner function
    p00 = 100000.0
    Rd = 287.0
    cp = 1004.0
    exn = (pres / p00) ** (Rd / cp)
    th_t = temperature / exn

    # Reference density
    rhodref = pres / (Rd * temperature)

    # Layer thickness (simple: 100m per level)
    dzz = jnp.full(shape, 100.0)

    # Water vapor (decreasing with height)
    rv_surf = 0.015  # 15 g/kg
    rv_t = rv_surf * jnp.exp(-z / 2000)
    rv_t = jnp.tile(rv_t, (nx, ny, 1))

    # Add variability
    key, subkey = jax.random.split(key)
    rv_noise = jax.random.uniform(subkey, shape) * 0.002
    rv_t = rv_t + rv_noise

    # Cloud water at mid-levels (2-6 km)
    rc_t = jnp.zeros(shape)
    cloud_mask = (z > 2000) & (z < 6000)
    key, subkey = jax.random.split(key)
    cloud_water = jax.random.uniform(subkey, shape) * 0.003
    rc_t = jnp.where(jnp.tile(cloud_mask, (nx, ny, 1)), cloud_water, 0.0)

    # Rain
    rr_t = jnp.zeros(shape)
    rain_mask = (z > 1000) & (z < 4000)
    key, subkey = jax.random.split(key)
    rain_water = jax.random.uniform(subkey, shape) * 0.001
    rr_t = jnp.where(jnp.tile(rain_mask, (nx, ny, 1)), rain_water, 0.0)

    # Ice at upper levels (> 5 km)
    ri_t = jnp.zeros(shape)
    ice_mask = z > 5000
    key, subkey = jax.random.split(key)
    ice_water = jax.random.uniform(subkey, shape) * 0.002
    ri_t = jnp.where(jnp.tile(ice_mask, (nx, ny, 1)), ice_water, 0.0)

    # Snow
    rs_t = jnp.zeros(shape)
    snow_mask = (z > 3000) & (z < 7000)
    key, subkey = jax.random.split(key)
    snow_water = jax.random.uniform(subkey, shape) * 0.001
    rs_t = jnp.where(jnp.tile(snow_mask, (nx, ny, 1)), snow_water, 0.0)

    # Graupel
    rg_t = jnp.zeros(shape)
    graupel_mask = (z > 2000) & (z < 6000)
    key, subkey = jax.random.split(key)
    graupel_water = jax.random.uniform(subkey, shape) * 0.0005
    rg_t = jnp.where(jnp.tile(graupel_mask, (nx, ny, 1)), graupel_water, 0.0)

    # Ice crystal concentration
    ci_t = jnp.where(ri_t > 0, 1000.0, 0.0)  # 1000 crystals/L where ice exists

    data = {
        'exn': exn,
        'rhodref': rhodref,
        'pres': pres,
        'dzz': dzz,
        'th_t': th_t,
        'rv_t': rv_t,
        'rc_t': rc_t,
        'rr_t': rr_t,
        'ri_t': ri_t,
        'rs_t': rs_t,
        'rg_t': rg_t,
        'ci_t': ci_t,
        'ths': jnp.zeros(shape),
        'rvs': jnp.zeros(shape),
        'rcs': jnp.zeros(shape),
        'rrs': jnp.zeros(shape),
        'ris': jnp.zeros(shape),
        'rss': jnp.zeros(shape),
        'rgs': jnp.zeros(shape),
        'cis': jnp.zeros(shape),
        'sigs': jnp.ones(shape) * 0.1,
        'timestep': 50.0,
    }

    return data


@pytest.fixture
def small_domain_rain_ice_data():
    """Small domain for quick tests."""
    return create_rain_ice_jax_test_data(shape=(10, 10, 20))


@pytest.fixture
def medium_domain_rain_ice_data():
    """Medium domain for standard tests."""
    return create_rain_ice_jax_test_data(shape=(50, 50, 40))


@pytest.fixture
def large_domain_rain_ice_data():
    """Large domain for performance tests."""
    return create_rain_ice_jax_test_data(shape=(100, 100, 60))


@pytest.fixture
def rain_ice_jax():
    """Create RainIceJAX instance."""
    phyex = Phyex("AROME", TSTEP=50.0)
    return RainIceJAX(constants=phyex.to_externals())


class TestRainIceJAXPerformanceSmall:
    """Performance tests on small domain (10x10x20)."""

    def test_small_domain_rain_ice_jax(self, benchmark, rain_ice_jax, small_domain_rain_ice_data):
        """Benchmark small domain with JAX."""
        print("\n" + "="*75)
        print("JAX RAIN_ICE Performance: Small Domain (10x10x20)")
        print("="*75)

        shape = (10, 10, 20)
        total_points = np.prod(shape)
        print(f"Domain: {shape}")
        print(f"Grid points: {total_points:,}")

        # Warm-up (JIT compilation)
        _ = rain_ice_jax(**small_domain_rain_ice_data)

        # Benchmark
        result = benchmark(lambda: rain_ice_jax(**small_domain_rain_ice_data))

        # Statistics
        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")
        print(f"  Performance: {throughput*small_domain_rain_ice_data['timestep']/1e6:.2f} M point-steps/s")


class TestRainIceJAXPerformanceMedium:
    """Performance tests on medium domain (50x50x40)."""

    def test_medium_domain_rain_ice_jax(self, benchmark, rain_ice_jax, medium_domain_rain_ice_data):
        """Benchmark medium domain with JAX."""
        print("\n" + "="*75)
        print("JAX RAIN_ICE Performance: Medium Domain (50x50x40)")
        print("="*75)

        shape = (50, 50, 40)
        total_points = np.prod(shape)
        print(f"Domain: {shape}")
        print(f"Grid points: {total_points:,}")

        # Warm-up
        _ = rain_ice_jax(**medium_domain_rain_ice_data)

        # Benchmark
        result = benchmark(lambda: rain_ice_jax(**medium_domain_rain_ice_data))

        # Statistics
        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")
        print(f"  Performance: {throughput*medium_domain_rain_ice_data['timestep']/1e6:.2f} M point-steps/s")


class TestRainIceJAXPerformanceLarge:
    """Performance tests on large domain (100x100x60)."""

    def test_large_domain_rain_ice_jax(self, benchmark, rain_ice_jax, large_domain_rain_ice_data):
        """Benchmark large domain with JAX."""
        print("\n" + "="*75)
        print("JAX RAIN_ICE Performance: Large Domain (100x100x60)")
        print("="*75)

        shape = (100, 100, 60)
        total_points = np.prod(shape)
        print(f"Domain: {shape}")
        print(f"Grid points: {total_points:,}")

        # Warm-up
        _ = rain_ice_jax(**large_domain_rain_ice_data)

        # Benchmark
        result = benchmark(lambda: rain_ice_jax(**large_domain_rain_ice_data))

        # Statistics
        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")
        print(f"  Performance: {throughput*large_domain_rain_ice_data['timestep']/1e6:.2f} M point-steps/s")

        if hasattr(result.stats, 'stddev'):
            print(f"  Std dev: {result.stats['stddev']*1000:.3f} ms")
        if hasattr(result.stats, 'min'):
            print(f"  Min time: {result.stats['min']*1000:.3f} ms")
        if hasattr(result.stats, 'max'):
            print(f"  Max time: {result.stats['max']*1000:.3f} ms")


class TestRainIceJAXScaling:
    """Test JAX scaling characteristics."""

    def test_jax_scaling_analysis(self, rain_ice_jax):
        """Analyze how JAX performance scales with domain size."""
        print("\n" + "="*75)
        print("JAX RAIN_ICE: Scaling Analysis")
        print("="*75)

        sizes = [
            (10, 10, 10),
            (20, 20, 20),
            (30, 30, 30),
        ]

        results = []

        for shape in sizes:
            data = create_rain_ice_jax_test_data(shape=shape)
            total_points = np.prod(shape)

            # Warm-up
            _ = rain_ice_jax(**data)

            # Time single execution
            import time
            start = time.time()
            _ = rain_ice_jax(**data)
            elapsed = time.time() - start

            throughput = total_points / elapsed
            results.append((total_points, elapsed, throughput))

            print(f"\nDomain {shape}: {total_points:,} points")
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


def test_rain_ice_jax_with_repro_data(benchmark, rain_ice_repro_ds):
    """
    Performance test with reproduction dataset from rain_ice.nc.

    Parameters
    ----------
    benchmark : pytest_benchmark fixture
        Benchmark fixture for timing
    rain_ice_repro_ds : xr.Dataset
        Reference dataset from rain_ice.nc fixture
    """
    print("\n" + "="*75)
    print("JAX RAIN_ICE Performance: Reproduction Dataset")
    print("="*75)

    # Get dataset dimensions
    shape = (
        rain_ice_repro_ds.sizes["ngpblks"],
        rain_ice_repro_ds.sizes["nproma"],
        rain_ice_repro_ds.sizes["nflevg"]
    )

    print(f"Dataset shape: {shape}")
    print(f"Domain (ngpblks × nproma × nflevg): {shape[0]} × {shape[1]} × {shape[2]}")
    total_points = np.prod(shape)
    print(f"Total grid points: {total_points:,}")

    # Initialize JAX component
    phyex = Phyex("AROME", TSTEP=50.0)
    rain_ice = RainIceJAX(constants=phyex.to_externals())

    # Helper to reshape
    def reshape_for_jax(var):
        """Reshape dataset variable for JAX (swap axes)."""
        return jnp.asarray(np.swapaxes(var, 1, 2))

    # Load data
    exn = reshape_for_jax(rain_ice_repro_ds["PEXNREF"].values)
    rhodref = reshape_for_jax(rain_ice_repro_ds["PRHODREF"].values)
    pres = reshape_for_jax(rain_ice_repro_ds["PPABSM"].values)
    dzz = reshape_for_jax(rain_ice_repro_ds["PDZZ"].values)

    # State from PRT
    prt = rain_ice_repro_ds["PRT"].values
    prt = np.swapaxes(prt, 2, 3)

    th_t = reshape_for_jax(rain_ice_repro_ds["PTHT"].values)
    rv_t = jnp.asarray(prt[:, 0, :, :])
    rc_t = jnp.asarray(prt[:, 1, :, :])
    rr_t = jnp.asarray(prt[:, 2, :, :])
    ri_t = jnp.asarray(prt[:, 3, :, :])
    rs_t = jnp.asarray(prt[:, 4, :, :])
    rg_t = jnp.asarray(prt[:, 5, :, :])

    ci_t = reshape_for_jax(rain_ice_repro_ds["PCIT"].values)

    # Tendencies
    prs = rain_ice_repro_ds["PRS"].values
    prs = np.swapaxes(prs, 2, 3)

    ths = reshape_for_jax(rain_ice_repro_ds["PTHS"].values)
    rvs = jnp.asarray(prs[:, 0, :, :])
    rcs = jnp.asarray(prs[:, 1, :, :])
    rrs = jnp.asarray(prs[:, 2, :, :])
    ris = jnp.asarray(prs[:, 3, :, :])
    rss = jnp.asarray(prs[:, 4, :, :])
    rgs = jnp.asarray(prs[:, 5, :, :])

    sigs = reshape_for_jax(rain_ice_repro_ds["PSIGS"].values)

    timestep = 50.0

    # Warm-up
    print("\nWarming up (JIT compilation)...")
    _ = rain_ice(
        exn=exn, rhodref=rhodref, pres=pres, dzz=dzz,
        th_t=th_t, rv_t=rv_t, rc_t=rc_t, rr_t=rr_t, ri_t=ri_t, rs_t=rs_t, rg_t=rg_t,
        ci_t=ci_t, ths=ths, rvs=rvs, rcs=rcs, rrs=rrs, ris=ris, rss=rss, rgs=rgs,
        cis=jnp.zeros_like(ci_t), sigs=sigs, timestep=timestep
    )

    # Benchmark
    print("Running benchmark...")

    def run_rain_ice():
        return rain_ice(
            exn=exn, rhodref=rhodref, pres=pres, dzz=dzz,
            th_t=th_t, rv_t=rv_t, rc_t=rc_t, rr_t=rr_t, ri_t=ri_t, rs_t=rs_t, rg_t=rg_t,
            ci_t=ci_t, ths=ths, rvs=rvs, rcs=rcs, rrs=rrs, ris=ris, rss=rss, rgs=rgs,
            cis=jnp.zeros_like(ci_t), sigs=sigs, timestep=timestep
        )

    result = benchmark(run_rain_ice)

    # Statistics
    print("\n" + "-"*75)
    print("PERFORMANCE STATISTICS")
    print("-"*75)

    mean_time = result.stats['mean']
    throughput = total_points / mean_time

    print(f"Mean time: {mean_time*1000:.3f} ms")
    print(f"Throughput: {throughput/1e6:.2f} M points/s")
    print(f"Performance: {throughput*timestep/1e6:.2f} M point-steps/s")

    if hasattr(result.stats, 'stddev'):
        print(f"Std dev: {result.stats['stddev']*1000:.3f} ms")
    if hasattr(result.stats, 'min'):
        print(f"Min time: {result.stats['min']*1000:.3f} ms")
    if hasattr(result.stats, 'max'):
        print(f"Max time: {result.stats['max']*1000:.3f} ms")

    print("-"*75)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
