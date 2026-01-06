# -*- coding: utf-8 -*-
"""Performance tests for JAX implementation of ICE_ADJUST component."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from ice3.jax.ice_adjust import IceAdjustJAX
from ice3.phyex_common.phyex import Phyex


def create_jax_test_data(shape=(100, 100, 60)):
    """
    Create realistic atmospheric test data for JAX ICE_ADJUST performance tests.

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
    pabs = jnp.tile(pressure, (nx, ny, 1))

    # Temperature profile
    temperature = T0 - gamma * z
    temperature = jnp.tile(temperature, (nx, ny, 1))

    # Add some variability
    key = jax.random.PRNGKey(42)
    temp_noise = jax.random.normal(key, shape) * 0.5
    temperature = temperature + temp_noise

    # Exner function
    p00 = 100000.0
    Rd = 287.0
    cp = 1004.0
    exn = (pabs / p00) ** (Rd / cp)
    th = temperature / exn

    # Reference values
    rho_dry_ref = pabs / (Rd * temperature)
    exn_ref = exn

    # Water vapor (decreasing with height)
    rv_surf = 0.015  # 15 g/kg
    rv = rv_surf * jnp.exp(-z / 2000)  # Scale height 2km
    rv = jnp.tile(rv, (nx, ny, 1))

    # Add variability
    key, subkey = jax.random.split(key)
    rv_noise = jax.random.uniform(subkey, shape) * 0.002
    rv = rv + rv_noise

    # Add some cloud water at mid-levels (2-6 km)
    rc = jnp.zeros(shape)
    cloud_mask = (z > 2000) & (z < 6000)
    key, subkey = jax.random.split(key)
    cloud_water = jax.random.uniform(subkey, shape) * 0.002
    rc = jnp.where(
        jnp.tile(cloud_mask, (nx, ny, 1)),
        cloud_water,
        0.0
    )

    # Add some ice at upper levels (> 5 km)
    ri = jnp.zeros(shape)
    ice_mask = z > 5000
    key, subkey = jax.random.split(key)
    ice_water = jax.random.uniform(subkey, shape) * 0.001
    ri = jnp.where(
        jnp.tile(ice_mask, (nx, ny, 1)),
        ice_water,
        0.0
    )

    data = {
        'sigqsat': jnp.ones((nx, ny, nz)) * 0.01,
        'pabs': pabs,
        'sigs': jnp.ones((nx, ny, nz)) * 0.1,
        'th': th,
        'exn': exn,
        'exn_ref': exn_ref,
        'rho_dry_ref': rho_dry_ref,
        'rv': rv,
        'rc': rc,
        'ri': ri,
        'rr': jnp.zeros(shape),
        'rs': jnp.zeros(shape),
        'rg': jnp.zeros(shape),
        'cf_mf': jnp.zeros(shape),
        'rc_mf': jnp.zeros(shape),
        'ri_mf': jnp.zeros(shape),
        'rvs': jnp.zeros(shape),
        'rcs': jnp.zeros(shape),
        'ris': jnp.zeros(shape),
        'ths': jnp.zeros(shape),
        'timestep': 50.0,
    }

    return data


@pytest.fixture
def small_domain_data():
    """Small domain for quick tests."""
    return create_jax_test_data(shape=(10, 10, 20))


@pytest.fixture
def medium_domain_data():
    """Medium domain for standard tests."""
    return create_jax_test_data(shape=(50, 50, 40))


@pytest.fixture
def large_domain_data():
    """Large domain for performance tests."""
    return create_jax_test_data(shape=(100, 100, 60))


@pytest.fixture
def ice_adjust_jax():
    """Create IceAdjustJAX instance with JIT enabled."""
    phyex = Phyex("AROME", TSTEP=50.0)
    return IceAdjustJAX(phyex=phyex, jit=True)


@pytest.fixture
def ice_adjust_jax_no_jit():
    """Create IceAdjustJAX instance without JIT."""
    phyex = Phyex("AROME", TSTEP=50.0)
    return IceAdjustJAX(phyex=phyex, jit=False)


class TestIceAdjustJAXPerformanceSmall:
    """Performance tests on small domain (10x10x20)."""

    def test_small_domain_with_jit(self, benchmark, ice_adjust_jax, small_domain_data):
        """Benchmark small domain with JIT compilation."""
        print("\n" + "="*75)
        print("JAX ICE_ADJUST Performance: Small Domain (10x10x20) with JIT")
        print("="*75)

        shape = (10, 10, 20)
        total_points = np.prod(shape)
        print(f"Domain: {shape}")
        print(f"Grid points: {total_points:,}")

        # Warm-up to trigger JIT compilation
        _ = ice_adjust_jax(**small_domain_data)

        # Benchmark
        result = benchmark(lambda: ice_adjust_jax(**small_domain_data))

        # Statistics
        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")
        print(f"  Performance: {throughput*small_domain_data['timestep']/1e6:.2f} M point-steps/s")

    def test_small_domain_no_jit(self, benchmark, ice_adjust_jax_no_jit, small_domain_data):
        """Benchmark small domain without JIT (pure Python)."""
        print("\n" + "="*75)
        print("JAX ICE_ADJUST Performance: Small Domain (10x10x20) without JIT")
        print("="*75)

        shape = (10, 10, 20)
        total_points = np.prod(shape)
        print(f"Domain: {shape}")
        print(f"Grid points: {total_points:,}")

        # Benchmark
        result = benchmark(lambda: ice_adjust_jax_no_jit(**small_domain_data))

        # Statistics
        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")


class TestIceAdjustJAXPerformanceMedium:
    """Performance tests on medium domain (50x50x40)."""

    def test_medium_domain_with_jit(self, benchmark, ice_adjust_jax, medium_domain_data):
        """Benchmark medium domain with JIT compilation."""
        print("\n" + "="*75)
        print("JAX ICE_ADJUST Performance: Medium Domain (50x50x40) with JIT")
        print("="*75)

        shape = (50, 50, 40)
        total_points = np.prod(shape)
        print(f"Domain: {shape}")
        print(f"Grid points: {total_points:,}")

        # Warm-up
        _ = ice_adjust_jax(**medium_domain_data)

        # Benchmark
        result = benchmark(lambda: ice_adjust_jax(**medium_domain_data))

        # Statistics
        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")
        print(f"  Performance: {throughput*medium_domain_data['timestep']/1e6:.2f} M point-steps/s")


class TestIceAdjustJAXPerformanceLarge:
    """Performance tests on large domain (100x100x60)."""

    def test_large_domain_with_jit(self, benchmark, ice_adjust_jax, large_domain_data):
        """Benchmark large domain with JIT compilation."""
        print("\n" + "="*75)
        print("JAX ICE_ADJUST Performance: Large Domain (100x100x60) with JIT")
        print("="*75)

        shape = (100, 100, 60)
        total_points = np.prod(shape)
        print(f"Domain: {shape}")
        print(f"Grid points: {total_points:,}")

        # Warm-up
        _ = ice_adjust_jax(**large_domain_data)

        # Benchmark
        result = benchmark(lambda: ice_adjust_jax(**large_domain_data))

        # Statistics
        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")
        print(f"  Performance: {throughput*large_domain_data['timestep']/1e6:.2f} M point-steps/s")

        if hasattr(result.stats, 'stddev'):
            print(f"  Std dev: {result.stats['stddev']*1000:.3f} ms")
        if hasattr(result.stats, 'min'):
            print(f"  Min time: {result.stats['min']*1000:.3f} ms")
        if hasattr(result.stats, 'max'):
            print(f"  Max time: {result.stats['max']*1000:.3f} ms")


class TestIceAdjustJAXCompilationTime:
    """Test JIT compilation overhead."""

    def test_compilation_overhead(self, ice_adjust_jax, small_domain_data):
        """Measure JIT compilation overhead on first call."""
        import time

        print("\n" + "="*75)
        print("JAX ICE_ADJUST: JIT Compilation Overhead")
        print("="*75)

        # First call (includes compilation)
        start = time.time()
        result1 = ice_adjust_jax(**small_domain_data)
        first_call_time = time.time() - start

        # Second call (no compilation)
        start = time.time()
        result2 = ice_adjust_jax(**small_domain_data)
        second_call_time = time.time() - start

        compilation_overhead = first_call_time - second_call_time

        print(f"\nResults:")
        print(f"  First call (with compilation): {first_call_time*1000:.3f} ms")
        print(f"  Second call (cached): {second_call_time*1000:.3f} ms")
        print(f"  Compilation overhead: {compilation_overhead*1000:.3f} ms")
        print(f"  Speedup after compilation: {first_call_time/second_call_time:.2f}x")

        # Verify results are identical
        assert jnp.allclose(result1[0], result2[0])


class TestIceAdjustJAXScaling:
    """Test scaling characteristics."""

    def test_scaling_with_domain_size(self, benchmark, ice_adjust_jax):
        """Test how performance scales with domain size."""
        print("\n" + "="*75)
        print("JAX ICE_ADJUST: Scaling Analysis")
        print("="*75)

        sizes = [
            (10, 10, 10),
            (20, 20, 20),
            (30, 30, 30),
        ]

        results = []
        for shape in sizes:
            data = create_jax_test_data(shape=shape)
            total_points = np.prod(shape)

            # Warm-up
            _ = ice_adjust_jax(**data)

            # Time single execution
            import time
            start = time.time()
            _ = ice_adjust_jax(**data)
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
        for i, (points, time, tput) in enumerate(results):
            if i > 0:
                size_ratio = points / results[0][0]
                time_ratio = time / results[0][1]
                print(f"  {points:>10,} points: {time_ratio:.2f}x time for {size_ratio:.2f}x points")


def test_ice_adjust_jax_with_repro_data(benchmark, ice_adjust_repro_ds):
    """
    Performance test with reproduction dataset from ice_adjust.nc.

    This test measures performance on real atmospheric data from PHYEX.

    Parameters
    ----------
    benchmark : pytest_benchmark fixture
        Benchmark fixture for timing
    ice_adjust_repro_ds : xr.Dataset
        Reference dataset from ice_adjust.nc fixture
    """
    print("\n" + "="*75)
    print("JAX ICE_ADJUST Performance: Reproduction Dataset")
    print("="*75)

    # Get dataset dimensions
    shape = (
        ice_adjust_repro_ds.sizes["ngpblks"],
        ice_adjust_repro_ds.sizes["nproma"],
        ice_adjust_repro_ds.sizes["nflevg"]
    )

    print(f"Dataset shape: {shape}")
    print(f"Domain (ngpblks × nproma × nflevg): {shape[0]} × {shape[1]} × {shape[2]}")
    total_points = np.prod(shape)
    print(f"Total grid points: {total_points:,}")

    # Initialize JAX component
    phyex = Phyex("AROME", TSTEP=50.0)
    ice_adjust = IceAdjustJAX(phyex=phyex, jit=True)

    # Helper to reshape: (ngpblks, nflevg, nproma) → (ngpblks, nproma, nflevg)
    def reshape_for_jax(var):
        """Reshape dataset variable for JAX (swap axes)."""
        return jnp.asarray(np.swapaxes(var, 1, 2))

    # Load atmospheric state
    pabs = reshape_for_jax(ice_adjust_repro_ds["PPABSM"].values)

    # Load state from ZRS (ngpblks, krr1, nflevg, nproma)
    zrs = ice_adjust_repro_ds["ZRS"].values
    zrs = np.swapaxes(zrs, 2, 3)  # → (ngpblks, krr1, nproma, nflevg)

    th = jnp.asarray(zrs[:, 0, :, :])
    rv = jnp.asarray(zrs[:, 1, :, :])
    rc = jnp.asarray(zrs[:, 2, :, :])
    rr = jnp.asarray(zrs[:, 3, :, :])
    ri = jnp.asarray(zrs[:, 4, :, :])
    rs = jnp.asarray(zrs[:, 5, :, :])
    rg = jnp.asarray(zrs[:, 6, :, :])

    exn = reshape_for_jax(ice_adjust_repro_ds["PEXNREF"].values)
    exn_ref = reshape_for_jax(ice_adjust_repro_ds["PEXNREF"].values)
    rho_dry_ref = reshape_for_jax(ice_adjust_repro_ds["PRHODREF"].values)

    # Load input tendencies from PRS (ngpblks, krr, nflevg, nproma)
    prs = ice_adjust_repro_ds["PRS"].values
    prs = np.swapaxes(prs, 2, 3)

    rvs = jnp.asarray(prs[:, 0, :, :])
    rcs = jnp.asarray(prs[:, 1, :, :])
    ris = jnp.asarray(prs[:, 3, :, :])

    # Load ths from PTHS
    pths = ice_adjust_repro_ds["PTHS"].values
    ths = reshape_for_jax(pths)

    # Mass flux variables
    cf_mf = reshape_for_jax(ice_adjust_repro_ds["PCF_MF"].values)
    rc_mf = reshape_for_jax(ice_adjust_repro_ds["PRC_MF"].values)
    ri_mf = reshape_for_jax(ice_adjust_repro_ds["PRI_MF"].values)

    # Sigma variables
    zsigqsat = ice_adjust_repro_ds["ZSIGQSAT"].values
    sigqsat = jnp.asarray(zsigqsat[:, :, np.newaxis])

    sigs = reshape_for_jax(ice_adjust_repro_ds["PSIGS"].values)

    timestep = 50.0

    # Warm-up
    print("\nWarming up (JIT compilation)...")
    _ = ice_adjust(
        sigqsat=sigqsat, pabs=pabs, sigs=sigs, th=th, exn=exn,
        exn_ref=exn_ref, rho_dry_ref=rho_dry_ref, rv=rv, rc=rc, ri=ri,
        rr=rr, rs=rs, rg=rg, cf_mf=cf_mf, rc_mf=rc_mf, ri_mf=ri_mf,
        rvs=rvs, rcs=rcs, ris=ris, ths=ths, timestep=timestep,
    )

    # Benchmark
    print("Running benchmark...")

    def run_ice_adjust():
        return ice_adjust(
            sigqsat=sigqsat, pabs=pabs, sigs=sigs, th=th, exn=exn,
            exn_ref=exn_ref, rho_dry_ref=rho_dry_ref, rv=rv, rc=rc, ri=ri,
            rr=rr, rs=rs, rg=rg, cf_mf=cf_mf, rc_mf=rc_mf, ri_mf=ri_mf,
            rvs=rvs, rcs=rcs, ris=ris, ths=ths, timestep=timestep,
        )

    result = benchmark(run_ice_adjust)

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
