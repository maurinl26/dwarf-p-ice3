# -*- coding: utf-8 -*-
"""Benchmark for JAX implementation of TURBULENCE scheme on CPU."""
import numpy as np
import pytest
import sys
from pathlib import Path

# Try to import JAX implementation
try:
    import jax
    # Force JAX to use CPU only
    jax.config.update('jax_platform_name', 'cpu')
    import jax.numpy as jnp
    from ice3.jax.turbulence.turb import turb_scheme
    from ice3.jax.turbulence.constants import TurbulenceConstants
    from ice3.phyex_common.phyex import Phyex
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    turb_scheme = None
    TurbulenceConstants = None
    Phyex = None


def create_boundary_layer_profile(nz=50, nlon=1, ztop=3000.0, backend='jax'):
    """
    Create a simple boundary layer profile for turbulence testing.

    Note: Turbulence scheme operates on 1D vertical columns, so nlon is not used
    in the actual computation, but we keep it for benchmarking multiple columns.

    Parameters
    ----------
    nz : int
        Number of vertical levels
    nlon : int
        Number of columns to benchmark (not used in turbulence scheme itself)
    ztop : float
        Top of domain (m)
    backend : str
        'jax' - determines output format

    Returns
    -------
    dict
        Dictionary with all required fields (1D profiles)
    """
    # Vertical grid (1D for turbulence scheme)
    z = np.linspace(0, ztop, nz, dtype=np.float32)
    zz = z  # Keep 1D

    # Grid spacing (1D)
    dz = ztop / (nz - 1)
    dzz = np.full(nz, dz, dtype=np.float32)

    # Standard atmosphere at surface
    p_surf = 101325.0  # Pa
    T_surf = 288.15    # K
    theta_surf = 288.15  # K (potential temperature)

    # Lapse rate
    gamma = 0.0065  # K/m

    # Temperature profile (1D)
    temperature = (T_surf - gamma * z).astype(np.float32)

    # Potential temperature (slightly increasing with height for stability - 1D)
    theta = (theta_surf + 0.003 * z).astype(np.float32)

    # Pressure profile (1D)
    pressure = (p_surf * (1 - gamma * z / T_surf) ** 5.26).astype(np.float32)

    # Exner function
    Rd = 287.0
    cp = 1004.0
    p00 = 100000.0
    exner = (pressure / p00) ** (Rd / cp)

    # Wind profiles (log profile in boundary layer - 1D)
    z0 = 0.1  # roughness length (m)
    u_star = 0.4  # friction velocity (m/s)
    kappa = 0.4  # von Karman constant

    u_wind = np.zeros(nz, dtype=np.float32)
    v_wind = np.zeros(nz, dtype=np.float32)

    for i in range(nz):
        if z[i] > z0:
            u_wind[i] = (u_star / kappa) * np.log(z[i] / z0)

    # Water vapor (decreasing with height - 1D)
    rv_surf = 0.012  # 12 g/kg at surface
    rv_mix = (rv_surf * np.exp(-z / 2000.0)).astype(np.float32)

    # TKE profile (maximum in boundary layer - 1D)
    tke_max = 1.0  # m^2/s^2
    tke = (tke_max * np.exp(-(z / 500.0)**2)).astype(np.float32)

    # Cloud water (some clouds in boundary layer - 1D)
    rc = np.zeros(nz, dtype=np.float32)
    cloud_levels = (z > 500) & (z < 1500)
    rc[cloud_levels] = 0.0005  # 0.5 g/kg

    if backend == 'jax':
        # Compute derived quantities for turbulence scheme
        # Total water mixing ratio
        rt = rv_mix + rc
        # Liquid potential temperature (approximation)
        thl = theta.copy()
        # Virtual potential temperature (reference)
        thvref = theta * (1 + 0.61 * rv_mix)
        # Ice mixing ratio
        ri = np.zeros_like(rc)
        # Vertical velocity
        w = np.zeros_like(u_wind)

        return {
            'zz': jnp.asarray(zz),
            'dzz': jnp.asarray(dzz),
            'theta': jnp.asarray(theta),
            'thl': jnp.asarray(thl),
            'rt': jnp.asarray(rt),
            'rv': jnp.asarray(rv_mix),
            'rc': jnp.asarray(rc),
            'ri': jnp.asarray(ri),
            'u': jnp.asarray(u_wind),
            'v': jnp.asarray(v_wind),
            'w': jnp.asarray(w),
            'tke': jnp.asarray(tke),
            'thvref': jnp.asarray(thvref),
            'pabst': jnp.asarray(pressure),
            'exn': jnp.asarray(exner),
            'surf_flux_u': 0.0,
            'surf_flux_v': 0.0,
            'surf_flux_th': 0.01,  # Small surface heat flux
            'surf_flux_rv': 0.0001,  # Small surface moisture flux
            'dt': 60.0,  # 60 second time step
        }


@pytest.fixture
def turb_constants():
    """Create turbulence constants for AROME."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    return TurbulenceConstants.arome()


@pytest.fixture
def phys_constants():
    """Create physics constants."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    phyex = Phyex("AROME", TSTEP=60.0)
    externals = phyex.to_externals()

    # The turb_scheme expects specific lowercase keys
    # Map PHYEX constant names to turbulence scheme expectations
    key_mapping = {
        'GRAVITY0': 'g',
        'RD': 'rd',
        'RV': 'rv',
        'KARMAN': 'karman',
    }

    phys_const = externals.copy()
    for phyex_key, turb_key in key_mapping.items():
        if phyex_key in externals:
            phys_const[turb_key] = externals[phyex_key]

    # Add alphaoc if not present (thermal expansion coefficient for ocean)
    if 'alphaoc' not in phys_const:
        phys_const['alphaoc'] = 1.9e-4

    return phys_const


@pytest.fixture
def small_data_jax():
    """Small domain for JAX (30 vertical levels)."""
    return create_boundary_layer_profile(nz=30, nlon=1, ztop=2000.0, backend='jax')


@pytest.fixture
def medium_data_jax():
    """Medium domain for JAX (50 vertical levels)."""
    return create_boundary_layer_profile(nz=50, nlon=1, ztop=3000.0, backend='jax')


@pytest.fixture
def large_data_jax():
    """Large domain for JAX (90 vertical levels)."""
    return create_boundary_layer_profile(nz=90, nlon=1, ztop=5000.0, backend='jax')


# ============================================================================
# Small Domain Benchmarks (1 column × 30 levels = 30 grid points)
# ============================================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestSmallDomainJAX:
    """JAX benchmarks on small domain."""

    def test_jax_small(self, benchmark, small_data_jax, turb_constants, phys_constants):
        """Benchmark JAX implementation on small domain (CPU)."""
        nz = 30
        total_points = nz

        # Warm-up to trigger JIT compilation (multiple iterations)
        for _ in range(3):
            _ = turb_scheme(
                **small_data_jax,
                turb_constants=turb_constants,
                phys_constants=phys_constants
            )

        # Block until all JAX operations complete
        if JAX_AVAILABLE:
            import jax
            jax.block_until_ready(turb_scheme(
                **small_data_jax,
                turb_constants=turb_constants,
                phys_constants=phys_constants
            ))

        result = benchmark(
            lambda: turb_scheme(
                **small_data_jax,
                turb_constants=turb_constants,
                phys_constants=phys_constants
            )
        )

        # Print statistics (only when benchmark is enabled)
        if hasattr(result, 'stats'):
            mean_time = result.stats['mean']
            throughput = total_points / mean_time
            print(f"\n[JAX CPU] Small domain (1 column × {nz} levels = {total_points:,} points)")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e3:.2f} K points/s")


# ============================================================================
# Medium Domain Benchmarks (1 column × 50 levels = 50 grid points)
# ============================================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestMediumDomainJAX:
    """JAX benchmarks on medium domain."""

    def test_jax_medium(self, benchmark, medium_data_jax, turb_constants, phys_constants):
        """Benchmark JAX implementation on medium domain (CPU)."""
        nz = 50
        total_points = nz

        # Warm-up to trigger JIT compilation (multiple iterations)
        for _ in range(3):
            _ = turb_scheme(
                **medium_data_jax,
                turb_constants=turb_constants,
                phys_constants=phys_constants
            )

        # Block until all JAX operations complete
        if JAX_AVAILABLE:
            import jax
            jax.block_until_ready(turb_scheme(
                **medium_data_jax,
                turb_constants=turb_constants,
                phys_constants=phys_constants
            ))

        result = benchmark(
            lambda: turb_scheme(
                **medium_data_jax,
                turb_constants=turb_constants,
                phys_constants=phys_constants
            )
        )

        # Print statistics (only when benchmark is enabled)
        if hasattr(result, 'stats'):
            mean_time = result.stats['mean']
            throughput = total_points / mean_time
            print(f"\n[JAX CPU] Medium domain (1 column × {nz} levels = {total_points:,} points)")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e3:.2f} K points/s")


# ============================================================================
# Large Domain Benchmarks (1 column × 90 levels = 90 grid points)
# ============================================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestLargeDomainJAX:
    """JAX benchmarks on large domain."""

    def test_jax_large(self, benchmark, large_data_jax, turb_constants, phys_constants):
        """Benchmark JAX implementation on large domain (CPU)."""
        nz = 90
        total_points = nz

        # Warm-up to trigger JIT compilation (multiple iterations)
        for _ in range(3):
            _ = turb_scheme(
                **large_data_jax,
                turb_constants=turb_constants,
                phys_constants=phys_constants
            )

        # Block until all JAX operations complete
        if JAX_AVAILABLE:
            import jax
            jax.block_until_ready(turb_scheme(
                **large_data_jax,
                turb_constants=turb_constants,
                phys_constants=phys_constants
            ))

        result = benchmark(
            lambda: turb_scheme(
                **large_data_jax,
                turb_constants=turb_constants,
                phys_constants=phys_constants
            )
        )

        # Print statistics (only when benchmark is enabled)
        if hasattr(result, 'stats'):
            mean_time = result.stats['mean']
            throughput = total_points / mean_time
            print(f"\n[JAX CPU] Large domain (1 column × {nz} levels = {total_points:,} points)")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e3:.2f} K points/s")

            if hasattr(result.stats, 'stddev'):
                print(f"  Std dev: {result.stats['stddev']*1000:.3f} ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only", "--benchmark-group-by=group"])
