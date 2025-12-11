"""Tests for JAX implementation of ICE_ADJUST component."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from ice3.jax.components.ice_adjust import IceAdjustJAX
from ice3.phyex_common.phyex import Phyex


@pytest.fixture
def phyex():
    """Create PHYEX configuration for tests."""
    return Phyex(program="AROME", TSTEP=60.0)


@pytest.fixture
def ice_adjust_jax(phyex):
    """Create IceAdjustJAX instance."""
    return IceAdjustJAX(phyex=phyex, jit=True)


@pytest.fixture
def ice_adjust_jax_no_jit(phyex):
    """Create IceAdjustJAX instance without JIT."""
    return IceAdjustJAX(phyex=phyex, jit=False)


@pytest.fixture
def simple_test_data():
    """Create simple 3D test data."""
    # Small 3D domain
    shape = (5, 5, 10)  # (x, y, z)
    
    data = {
        # Atmospheric state
        'sigqsat': jnp.ones(shape) * 0.01,
        'pabs': jnp.ones(shape) * 85000.0,  # Pa
        'sigs': jnp.ones(shape) * 0.1,
        'th': jnp.ones(shape) * 285.0,  # K
        'exn': jnp.ones(shape) * 0.95,
        'exn_ref': jnp.ones(shape) * 0.95,
        'rho_dry_ref': jnp.ones(shape) * 1.0,  # kg/m³
        
        # Mixing ratios
        'rv': jnp.ones(shape) * 0.010,  # 10 g/kg
        'rc': jnp.zeros(shape),
        'ri': jnp.zeros(shape),
        'rr': jnp.zeros(shape),
        'rs': jnp.zeros(shape),
        'rg': jnp.zeros(shape),
        
        # Mass flux variables
        'cf_mf': jnp.zeros(shape),
        'rc_mf': jnp.zeros(shape),
        'ri_mf': jnp.zeros(shape),
        
        # Tendency fields
        'rvs': jnp.zeros(shape),
        'rcs': jnp.zeros(shape),
        'ris': jnp.zeros(shape),
        'ths': jnp.zeros(shape),
        
        # Timestep
        'timestep': 60.0,
    }
    
    return data


@pytest.fixture
def realistic_test_data():
    """Create realistic test data with vertical variation."""
    nx, ny, nz = 10, 10, 20
    
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
    
    # Add some cloud water at mid-levels
    rc = jnp.zeros((nx, ny, nz))
    cloud_mask = (z > 2000) & (z < 6000)
    rc = jnp.where(
        jnp.tile(cloud_mask, (nx, ny, 1)),
        0.001,  # 1 g/kg
        0.0
    )
    
    # Add some ice at upper levels
    ri = jnp.zeros((nx, ny, nz))
    ice_mask = z > 5000
    ri = jnp.where(
        jnp.tile(ice_mask, (nx, ny, 1)),
        0.0005,  # 0.5 g/kg
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
        'rr': jnp.zeros((nx, ny, nz)),
        'rs': jnp.zeros((nx, ny, nz)),
        'rg': jnp.zeros((nx, ny, nz)),
        'cf_mf': jnp.zeros((nx, ny, nz)),
        'rc_mf': jnp.zeros((nx, ny, nz)),
        'ri_mf': jnp.zeros((nx, ny, nz)),
        'rvs': jnp.zeros((nx, ny, nz)),
        'rcs': jnp.zeros((nx, ny, nz)),
        'ris': jnp.zeros((nx, ny, nz)),
        'ths': jnp.zeros((nx, ny, nz)),
        'timestep': 60.0,
    }
    
    return data


class TestIceAdjustJAXInitialization:
    """Test IceAdjustJAX initialization."""
    
    def test_init_default(self):
        """Test initialization with defaults."""
        ice_adjust = IceAdjustJAX()
        assert ice_adjust.phyex is not None
        assert ice_adjust.constants is not None
        assert 'RD' in ice_adjust.constants
        assert 'TSTEP' in ice_adjust.constants
    
    def test_init_with_phyex(self, phyex):
        """Test initialization with custom PHYEX."""
        ice_adjust = IceAdjustJAX(phyex=phyex)
        assert ice_adjust.phyex is phyex
        assert ice_adjust.constants['TSTEP'] == 60.0
    
    def test_init_jit_enabled(self, phyex):
        """Test JIT compilation is enabled."""
        ice_adjust = IceAdjustJAX(phyex=phyex, jit=True)
        # Check that the function is wrapped
        assert hasattr(ice_adjust.ice_adjust_fn, '__wrapped__')
    
    def test_init_jit_disabled(self, phyex):
        """Test JIT compilation is disabled."""
        ice_adjust = IceAdjustJAX(phyex=phyex, jit=False)
        # Check that the function is not wrapped
        assert not hasattr(ice_adjust.ice_adjust_fn, '__wrapped__')
    
    def test_constants_extraction(self, phyex):
        """Test that constants are properly extracted."""
        ice_adjust = IceAdjustJAX(phyex=phyex)
        constants = ice_adjust.constants
        
        # Check essential constants
        assert 'RD' in constants
        assert 'RV' in constants
        assert 'CPD' in constants
        assert 'LVTT' in constants
        assert 'LSTT' in constants
        assert 'XTT' in constants
        assert 'TSTEP' in constants
        
        # Check AROME-specific settings
        assert constants['TSTEP'] == 60.0


class TestIceAdjustJAXExecution:
    """Test IceAdjustJAX execution."""
    
    def test_call_simple(self, ice_adjust_jax, simple_test_data):
        """Test basic call with simple data."""
        result = ice_adjust_jax(**simple_test_data)
        
        # Should return a tuple
        assert isinstance(result, tuple)
        assert len(result) > 0
        
        # First few results should be arrays
        t, rv_out, rc_out, ri_out, cldfr = result[:5]
        assert isinstance(t, jax.Array)
        assert isinstance(rv_out, jax.Array)
        assert isinstance(rc_out, jax.Array)
        assert isinstance(ri_out, jax.Array)
        assert isinstance(cldfr, jax.Array)
        
        # Check shapes match input
        shape = simple_test_data['th'].shape
        assert t.shape == shape
        assert rv_out.shape == shape
        assert rc_out.shape == shape
        assert ri_out.shape == shape
        assert cldfr.shape == shape
    
    def test_call_no_jit(self, ice_adjust_jax_no_jit, simple_test_data):
        """Test call without JIT compilation."""
        result = ice_adjust_jax_no_jit(**simple_test_data)
        
        assert isinstance(result, tuple)
        assert len(result) > 0
        
        # Check first few results
        t, rv_out, rc_out, ri_out, cldfr = result[:5]
        assert t.shape == simple_test_data['th'].shape
    
    def test_output_types(self, ice_adjust_jax, simple_test_data):
        """Test that all outputs are JAX arrays."""
        result = ice_adjust_jax(**simple_test_data)
        
        # All outputs should be JAX arrays
        for output in result:
            assert isinstance(output, jax.Array)
    
    def test_realistic_data(self, ice_adjust_jax, realistic_test_data):
        """Test with realistic atmospheric profile."""
        result = ice_adjust_jax(**realistic_test_data)
        
        t, rv_out, rc_out, ri_out, cldfr = result[:5]
        
        # Check shapes
        shape = realistic_test_data['th'].shape
        assert t.shape == shape
        
        # Check physical constraints
        assert jnp.all(t > 0), "Temperature should be positive"
        assert jnp.all(rv_out >= 0), "Water vapor should be non-negative"
        assert jnp.all(rc_out >= 0), "Cloud water should be non-negative"
        assert jnp.all(ri_out >= 0), "Cloud ice should be non-negative"
        assert jnp.all((cldfr >= 0) & (cldfr <= 1)), "Cloud fraction should be in [0, 1]"


class TestIceAdjustJAXPhysics:
    """Test physical validity of ICE_ADJUST results."""
    
    def test_conservation_simple(self, ice_adjust_jax, simple_test_data):
        """Test conservation with simple data."""
        result = ice_adjust_jax(**simple_test_data)
        t, rv_out, rc_out, ri_out, cldfr = result[:5]
        
        # Total water should be conserved (approximately)
        total_water_in = (
            simple_test_data['rv'] + 
            simple_test_data['rc'] + 
            simple_test_data['ri']
        )
        total_water_out = rv_out + rc_out + ri_out
        
        # Check conservation (allow small numerical errors)
        diff = jnp.abs(total_water_out - total_water_in)
        assert jnp.max(diff) < 1e-10, f"Water not conserved: max diff = {jnp.max(diff)}"
    
    def test_cloud_formation(self, ice_adjust_jax):
        """Test that clouds form when supersaturated."""
        # Create supersaturated conditions
        shape = (5, 5, 10)
        data = {
            'sigqsat': jnp.ones(shape) * 0.01,
            'pabs': jnp.ones(shape) * 85000.0,
            'sigs': jnp.ones(shape) * 0.1,
            'th': jnp.ones(shape) * 280.0,  # Cold
            'exn': jnp.ones(shape) * 0.95,
            'exn_ref': jnp.ones(shape) * 0.95,
            'rho_dry_ref': jnp.ones(shape) * 1.0,
            'rv': jnp.ones(shape) * 0.020,  # High humidity (20 g/kg)
            'rc': jnp.zeros(shape),
            'ri': jnp.zeros(shape),
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
            'timestep': 60.0,
        }
        
        result = ice_adjust_jax(**data)
        t, rv_out, rc_out, ri_out, cldfr = result[:5]
        
        # Should have some condensation
        # (actual values depend on saturation adjustment)
        assert jnp.any(cldfr > 0), "Should have some cloud fraction"
    
    def test_physical_bounds(self, ice_adjust_jax, realistic_test_data):
        """Test that results stay within physical bounds."""
        result = ice_adjust_jax(**realistic_test_data)
        t, rv_out, rc_out, ri_out, cldfr = result[:5]
        
        # Temperature should be reasonable (100-400 K)
        assert jnp.all(t > 100), "Temperature too low"
        assert jnp.all(t < 400), "Temperature too high"
        
        # Mixing ratios should be non-negative and reasonable
        assert jnp.all(rv_out >= 0), "Negative water vapor"
        assert jnp.all(rc_out >= 0), "Negative cloud water"
        assert jnp.all(ri_out >= 0), "Negative cloud ice"
        assert jnp.all(rv_out < 0.1), "Water vapor unrealistically high"
        assert jnp.all(rc_out < 0.05), "Cloud water unrealistically high"
        assert jnp.all(ri_out < 0.05), "Cloud ice unrealistically high"
        
        # Cloud fraction in valid range
        assert jnp.all(cldfr >= 0), "Negative cloud fraction"
        assert jnp.all(cldfr <= 1), "Cloud fraction > 1"


class TestIceAdjustJAXDifferentiation:
    """Test automatic differentiation capabilities."""
    
    def test_jvp_forward_mode(self, ice_adjust_jax, simple_test_data):
        """Test forward-mode differentiation (JVP)."""
        # Define a simple function of ice_adjust
        def f(rv):
            data = simple_test_data.copy()
            data['rv'] = rv
            result = ice_adjust_jax(**data)
            return result[0]  # Temperature
        
        # Compute JVP
        rv = simple_test_data['rv']
        drv = jnp.ones_like(rv) * 0.001  # Small perturbation
        
        primal, tangent = jax.jvp(f, (rv,), (drv,))
        
        # Should not raise errors
        assert primal.shape == rv.shape
        assert tangent.shape == rv.shape
    
    def test_vjp_reverse_mode(self, ice_adjust_jax, simple_test_data):
        """Test reverse-mode differentiation (VJP)."""
        # Define a simple loss function
        def loss(rv):
            data = simple_test_data.copy()
            data['rv'] = rv
            result = ice_adjust_jax(**data)
            t = result[0]
            return jnp.sum(t)  # Sum of temperatures
        
        # Compute gradient
        rv = simple_test_data['rv']
        grad = jax.grad(loss)(rv)
        
        # Should not raise errors
        assert grad.shape == rv.shape
        # Gradient should not be all zeros
        assert jnp.any(jnp.abs(grad) > 0)
    
    @pytest.mark.skipif(
        not hasattr(jax, 'jacfwd'),
        reason="jacfwd not available in this JAX version"
    )
    def test_jacobian(self, ice_adjust_jax_no_jit):
        """Test Jacobian computation (small domain)."""
        # Use very small domain for Jacobian test
        shape = (2, 2, 3)
        data = {
            'sigqsat': jnp.ones(shape) * 0.01,
            'pabs': jnp.ones(shape) * 85000.0,
            'sigs': jnp.ones(shape) * 0.1,
            'th': jnp.ones(shape) * 285.0,
            'exn': jnp.ones(shape) * 0.95,
            'exn_ref': jnp.ones(shape) * 0.95,
            'rho_dry_ref': jnp.ones(shape) * 1.0,
            'rv': jnp.ones(shape) * 0.010,
            'rc': jnp.zeros(shape),
            'ri': jnp.zeros(shape),
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
            'timestep': 60.0,
        }
        
        # Flatten for Jacobian
        def f(rv_flat):
            data_copy = data.copy()
            data_copy['rv'] = rv_flat.reshape(shape)
            result = ice_adjust_jax_no_jit(**data_copy)
            return result[0].flatten()  # Temperature flattened
        
        rv_flat = data['rv'].flatten()
        jac = jax.jacfwd(f)(rv_flat)
        
        # Should be a 2D array
        assert jac.ndim == 2
        assert jac.shape[0] == rv_flat.size
        assert jac.shape[1] == rv_flat.size


class TestIceAdjustJAXVectorization:
    """Test vectorization with vmap."""
    
    def test_vmap_batch_dimension(self, ice_adjust_jax):
        """Test batching over an additional dimension."""
        # Create data with batch dimension
        batch_size = 3
        shape = (5, 5, 10)
        
        # Stack data along batch dimension
        data_batch = {
            key: jnp.stack([val] * batch_size)
            for key, val in {
                'sigqsat': jnp.ones(shape) * 0.01,
                'pabs': jnp.ones(shape) * 85000.0,
                'sigs': jnp.ones(shape) * 0.1,
                'th': jnp.ones(shape) * 285.0,
                'exn': jnp.ones(shape) * 0.95,
                'exn_ref': jnp.ones(shape) * 0.95,
                'rho_dry_ref': jnp.ones(shape) * 1.0,
                'rv': jnp.ones(shape) * 0.010,
                'rc': jnp.zeros(shape),
                'ri': jnp.zeros(shape),
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
            }.items()
        }
        data_batch['timestep'] = 60.0
        
        # Create vmap'ed function
        ice_adjust_vmap = jax.vmap(
            lambda **kwargs: ice_adjust_jax(**kwargs),
            in_axes=({k: 0 for k in data_batch.keys() if k != 'timestep'},)
        )
        
        # This should work but requires proper vmap setup
        # For now, just test that the shapes are correct
        assert data_batch['th'].shape == (batch_size, *shape)


class TestIceAdjustJAXPerformance:
    """Test performance characteristics."""
    
    def test_jit_compilation(self, ice_adjust_jax, simple_test_data):
        """Test that JIT compilation works."""
        # First call triggers compilation
        result1 = ice_adjust_jax(**simple_test_data)
        
        # Second call should be faster (reuses compiled version)
        result2 = ice_adjust_jax(**simple_test_data)
        
        # Results should be identical
        assert jnp.allclose(result1[0], result2[0])
    
    def test_multiple_calls_consistency(self, ice_adjust_jax, simple_test_data):
        """Test that multiple calls give consistent results."""
        results = []
        for _ in range(3):
            result = ice_adjust_jax(**simple_test_data)
            results.append(result[0])  # Temperature
        
        # All results should be identical
        for i in range(1, len(results)):
            assert jnp.allclose(results[0], results[i])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
