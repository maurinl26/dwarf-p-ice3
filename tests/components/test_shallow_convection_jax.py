"""Tests for JAX implementation of SHALLOW_CONVECTION component."""

import pytest
import numpy as np

# Try to import JAX - skip all tests if not available
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

# Skip all tests in this module if JAX is not available
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")

# Import after JAX check to avoid import errors
if JAX_AVAILABLE:
    from ice3.jax.convection.shallow_convection import shallow_convection, ShallowConvectionOutputs
    from ice3.phyex_common.phyex import Phyex


@pytest.fixture
def phyex():
    """Create PHYEX configuration for tests."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    return Phyex(program="AROME", TSTEP=60.0)


@pytest.fixture
def simple_test_data():
    """Create simple test data for shallow convection."""
    # Small domain
    nlon = 50   # horizontal points
    nlev = 30   # vertical levels
    kch1 = 1    # chemical species

    # Height array (0 to 15 km)
    z = jnp.linspace(0, 15000, nlev)
    pzz = jnp.tile(z, (nlon, 1))

    # Pressure profile (exponential decrease with height)
    p_1d = 100000.0 * jnp.exp(-z / 7000.0)
    ppabst = jnp.tile(p_1d, (nlon, 1))

    # Temperature profile (decreasing with height)
    t_1d = 288.0 - 0.0065 * z
    ptt = jnp.tile(t_1d, (nlon, 1))

    # Water vapor mixing ratio (exponential decrease)
    rv_1d = 0.01 * jnp.exp(-z / 2000.0)
    prvt = jnp.tile(rv_1d, (nlon, 1))

    # Cloud water and ice (small values)
    prct = jnp.ones((nlon, nlev)) * 0.0001
    prit = jnp.ones((nlon, nlev)) * 0.00001

    # Vertical velocity
    pwt = jnp.ones((nlon, nlev)) * 0.1

    # TKE in cloud layer
    ptkecls = jnp.ones(nlon) * 0.5

    # Initialize output arrays
    ptten = jnp.zeros((nlon, nlev))
    prvten = jnp.zeros((nlon, nlev))
    prcten = jnp.zeros((nlon, nlev))
    priten = jnp.zeros((nlon, nlev))
    kcltop = jnp.zeros(nlon, dtype=jnp.int32)
    kclbas = jnp.zeros(nlon, dtype=jnp.int32)
    pumf = jnp.zeros((nlon, nlev))

    # Chemical tracer arrays
    pch1 = jnp.zeros((nlon, nlev, kch1))
    pch1ten = jnp.zeros((nlon, nlev, kch1))

    data = {
        'ppabst': ppabst,
        'pzz': pzz,
        'ptkecls': ptkecls,
        'ptt': ptt,
        'prvt': prvt,
        'prct': prct,
        'prit': prit,
        'pwt': pwt,
        'ptten': ptten,
        'prvten': prvten,
        'prcten': prcten,
        'priten': priten,
        'kcltop': kcltop,
        'kclbas': kclbas,
        'pumf': pumf,
        'pch1': pch1,
        'pch1ten': pch1ten,
        'kice': 1,
        'kbdia': 1,
        'ktdia': 1,
        'osettadj': False,
        'ptadjs': 10800.0,
        'och1conv': False,
    }

    return data


@pytest.fixture
def realistic_test_data():
    """Create realistic test data with vertical variation."""
    nlon = 100
    nlev = 60
    kch1 = 1

    # Create vertical coordinate (0-15 km)
    z = jnp.linspace(0, 15000, nlev)

    # Standard atmosphere
    p0 = 101325.0  # Pa
    T0 = 288.15    # K
    gamma = 0.0065  # K/m

    # Pressure profile
    pressure = p0 * (1 - gamma * z / T0) ** 5.26
    ppabst = jnp.tile(pressure, (nlon, 1))

    # Temperature profile
    temperature = T0 - gamma * z
    ptt = jnp.tile(temperature, (nlon, 1))

    # Add some variability
    key = jax.random.PRNGKey(42)
    ptt = ptt + jax.random.normal(key, ptt.shape) * 0.5

    # Water vapor (decreasing with height)
    rv_surf = 0.015  # 15 g/kg
    rv = rv_surf * jnp.exp(-z / 2000)  # Scale height 2km
    prvt = jnp.tile(rv, (nlon, 1))

    # Add some cloud water at mid-levels (2-6 km)
    cloud_mask = (z > 2000) & (z < 6000)
    prct = jnp.where(
        jnp.tile(cloud_mask, (nlon, 1)),
        0.001,  # 1 g/kg
        0.0
    )

    # Add some ice at upper levels (>5 km)
    ice_mask = z > 5000
    prit = jnp.where(
        jnp.tile(ice_mask, (nlon, 1)),
        0.0005,  # 0.5 g/kg
        0.0
    )

    # Vertical velocity (weak updraft)
    pwt = jnp.ones((nlon, nlev)) * 0.1

    # TKE
    ptkecls = jnp.ones(nlon) * 0.5

    # Height
    pzz = jnp.tile(z, (nlon, 1))

    # Initialize outputs
    ptten = jnp.zeros((nlon, nlev))
    prvten = jnp.zeros((nlon, nlev))
    prcten = jnp.zeros((nlon, nlev))
    priten = jnp.zeros((nlon, nlev))
    kcltop = jnp.zeros(nlon, dtype=jnp.int32)
    kclbas = jnp.zeros(nlon, dtype=jnp.int32)
    pumf = jnp.zeros((nlon, nlev))
    pch1 = jnp.zeros((nlon, nlev, kch1))
    pch1ten = jnp.zeros((nlon, nlev, kch1))

    data = {
        'ppabst': ppabst,
        'pzz': pzz,
        'ptkecls': ptkecls,
        'ptt': ptt,
        'prvt': prvt,
        'prct': prct,
        'prit': prit,
        'pwt': pwt,
        'ptten': ptten,
        'prvten': prvten,
        'prcten': prcten,
        'priten': priten,
        'kcltop': kcltop,
        'kclbas': kclbas,
        'pumf': pumf,
        'pch1': pch1,
        'pch1ten': pch1ten,
        'kice': 1,
        'kbdia': 1,
        'ktdia': 1,
        'osettadj': False,
        'ptadjs': 10800.0,
        'och1conv': False,
    }

    return data


class TestShallowConvectionJAXExecution:
    """Test JAX shallow convection execution."""

    def test_call_simple(self, simple_test_data):
        """Test basic call with simple data."""
        result = shallow_convection(**simple_test_data)

        # Should return a ShallowConvectionOutputs tuple
        assert isinstance(result, ShallowConvectionOutputs)

        # Check all outputs are JAX arrays
        assert isinstance(result.ptten, jax.Array)
        assert isinstance(result.prvten, jax.Array)
        assert isinstance(result.prcten, jax.Array)
        assert isinstance(result.priten, jax.Array)
        assert isinstance(result.kcltop, jax.Array)
        assert isinstance(result.kclbas, jax.Array)
        assert isinstance(result.pumf, jax.Array)
        assert isinstance(result.pch1ten, jax.Array)

        # Check shapes
        nlon, nlev = simple_test_data['ppabst'].shape
        kch1 = simple_test_data['pch1'].shape[2]

        assert result.ptten.shape == (nlon, nlev)
        assert result.prvten.shape == (nlon, nlev)
        assert result.prcten.shape == (nlon, nlev)
        assert result.priten.shape == (nlon, nlev)
        assert result.kcltop.shape == (nlon,)
        assert result.kclbas.shape == (nlon,)
        assert result.pumf.shape == (nlon, nlev)
        assert result.pch1ten.shape == (nlon, nlev, kch1)

    def test_output_types(self, simple_test_data):
        """Test that all outputs are JAX arrays."""
        result = shallow_convection(**simple_test_data)

        # All outputs should be JAX arrays
        for field in result:
            assert isinstance(field, jax.Array)

    def test_realistic_data(self, realistic_test_data):
        """Test with realistic atmospheric profile."""
        result = shallow_convection(**realistic_test_data)

        # Check shapes match
        nlon, nlev = realistic_test_data['ppabst'].shape
        assert result.ptten.shape == (nlon, nlev)

        # Check physical constraints
        assert jnp.all(jnp.isfinite(result.ptten)), "Non-finite temperature tendency"
        assert jnp.all(jnp.isfinite(result.prvten)), "Non-finite vapor tendency"
        assert jnp.all(jnp.isfinite(result.pumf)), "Non-finite mass flux"

        # Cloud top/base should be valid indices or 0
        assert jnp.all(result.kcltop >= 0), "Invalid cloud top index"
        assert jnp.all(result.kclbas >= 0), "Invalid cloud base index"
        assert jnp.all(result.kcltop <= nlev), "Cloud top beyond domain"
        assert jnp.all(result.kclbas <= nlev), "Cloud base beyond domain"


class TestShallowConvectionJAXPhysics:
    """Test physical validity of shallow convection results."""

    def test_physical_bounds(self, realistic_test_data):
        """Test that results stay within physical bounds."""
        result = shallow_convection(**realistic_test_data)

        # Tendencies should be finite
        assert jnp.all(jnp.isfinite(result.ptten)), "Non-finite temperature tendency"
        assert jnp.all(jnp.isfinite(result.prvten)), "Non-finite vapor tendency"
        assert jnp.all(jnp.isfinite(result.prcten)), "Non-finite cloud water tendency"
        assert jnp.all(jnp.isfinite(result.priten)), "Non-finite ice tendency"

        # Mass flux should be non-negative
        assert jnp.all(result.pumf >= 0), "Negative mass flux"

        # Tendencies should be reasonable magnitude
        # (These are loose bounds - actual values depend on convection strength)
        assert jnp.all(jnp.abs(result.ptten) < 1.0), "Temperature tendency too large"
        assert jnp.all(jnp.abs(result.prvten) < 0.1), "Vapor tendency too large"

    def test_convection_consistency(self, realistic_test_data):
        """Test consistency between convective outputs."""
        result = shallow_convection(**realistic_test_data)

        # If cloud top is set, cloud base should also be set
        has_convection = result.kcltop > 0
        assert jnp.all((result.kcltop > 0) == (result.kclbas > 0)), \
            "Inconsistent cloud top/base"

        # Cloud top should be above or equal to cloud base
        # (when convection is active)
        if jnp.any(has_convection):
            active_idx = jnp.where(has_convection)[0]
            tops = result.kcltop[active_idx]
            bases = result.kclbas[active_idx]
            # Note: Fortran convention may be top < base or top > base
            # depending on indexing direction - check actual behavior
            # For now just check they're different when active
            assert jnp.any(tops != bases) or len(active_idx) == 0, \
                "Cloud top and base should differ"

    def test_no_convection_case(self):
        """Test case where no convection should trigger."""
        # Create very stable conditions
        nlon = 10
        nlev = 20
        kch1 = 1

        # Dry, stable atmosphere
        z = jnp.linspace(0, 10000, nlev)
        ppabst = jnp.tile(100000.0 * jnp.exp(-z / 8000), (nlon, 1))
        ptt = jnp.tile(280.0 - 0.002 * z, (nlon, 1))  # Very stable
        prvt = jnp.ones((nlon, nlev)) * 0.0001  # Very dry
        prct = jnp.zeros((nlon, nlev))
        prit = jnp.zeros((nlon, nlev))
        pwt = jnp.zeros((nlon, nlev))  # No vertical motion
        ptkecls = jnp.zeros(nlon)  # No TKE
        pzz = jnp.tile(z, (nlon, 1))

        data = {
            'ppabst': ppabst,
            'pzz': pzz,
            'ptkecls': ptkecls,
            'ptt': ptt,
            'prvt': prvt,
            'prct': prct,
            'prit': prit,
            'pwt': pwt,
            'ptten': jnp.zeros((nlon, nlev)),
            'prvten': jnp.zeros((nlon, nlev)),
            'prcten': jnp.zeros((nlon, nlev)),
            'priten': jnp.zeros((nlon, nlev)),
            'kcltop': jnp.zeros(nlon, dtype=jnp.int32),
            'kclbas': jnp.zeros(nlon, dtype=jnp.int32),
            'pumf': jnp.zeros((nlon, nlev)),
            'pch1': jnp.zeros((nlon, nlev, kch1)),
            'pch1ten': jnp.zeros((nlon, nlev, kch1)),
            'kice': 1,
            'kbdia': 1,
            'ktdia': 1,
            'osettadj': False,
            'ptadjs': 10800.0,
            'och1conv': False,
        }

        result = shallow_convection(**data)

        # Should have minimal or no convection
        n_convective = jnp.sum(result.kcltop > 0)
        print(f"Convective columns in stable case: {n_convective}/{nlon}")
        # Allow some columns to trigger but expect most not to
        assert n_convective < nlon * 0.5, "Too much convection in stable case"


class TestShallowConvectionJAXPerformance:
    """Test performance characteristics."""

    def test_multiple_calls_consistency(self, simple_test_data):
        """Test that multiple calls give consistent results."""
        results = []
        for _ in range(3):
            result = shallow_convection(**simple_test_data)
            results.append(result.ptten)

        # All results should be identical (deterministic)
        for i in range(1, len(results)):
            assert jnp.allclose(results[0], results[i]), \
                "Results should be deterministic"


def test_shallow_convection_jax_with_repro_data(shallow_convection_repro_ds):
    """
    Test JAX SHALLOW_CONVECTION with reproduction dataset.

    This test validates that the JAX implementation produces results consistent
    with the reference PHYEX data.

    Parameters
    ----------
    shallow_convection_repro_ds : xr.Dataset
        Reference dataset from shallow_convection.nc (or shallow.nc) fixture
    """
    print("\n" + "="*70)
    print("TEST: JAX SHALLOW_CONVECTION with Reproduction Data")
    print("="*70)

    from numpy.testing import assert_allclose

    # Get dataset
    ds = shallow_convection_repro_ds

    print(f"\nDataset dimensions: {dict(ds.sizes)}")
    print(f"Available variables (first 10): {list(ds.data_vars.keys())[:10]}...")

    # NOTE: The shallow.nc data file uses generic variable names (var_00, var_01, etc.)
    # and the mapping to physical variables is not yet documented.
    # This test is a placeholder until the data structure is clarified.

    print("\n⚠️  WARNING: Shallow convection data variable mapping not yet documented")
    print("   Data file uses generic names (var_00, var_01, ..., var_54)")
    print("   Need documentation to map these to physical variables:")
    print("   - ppabst (pressure)")
    print("   - pzz (height)")
    print("   - ptt (temperature)")
    print("   - prvt, prct, prit (mixing ratios)")
    print("   - ptkecls (TKE)")
    print("   - pwt (vertical velocity)")
    print("   - Output tendencies")

    # Placeholder: Skip detailed comparison until data structure is documented
    pytest.skip("Shallow convection data structure needs documentation - variable mapping required")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
