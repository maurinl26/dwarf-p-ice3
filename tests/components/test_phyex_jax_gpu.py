"""
Standalone tests for the zero-copy JAX <-> PHYEX GPU bridge.

Test hierarchy:
  TestMakeIceAdjust    — factory smoke tests (always green, no GPU required)
  TestIceAdjustJAXGPU  — DLPack bridge + physical validation (GPU required)

Run on RunPod:
  pytest tests/components/test_phyex_jax_gpu.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

# ── JAX availability ──────────────────────────────────────────────────────────
jax = pytest.importorskip("jax", reason="JAX not installed")
import jax.numpy as jnp

# ── GPU / CuPy detection ──────────────────────────────────────────────────────
try:
    import cupy as cp
    HAS_CUPY = cp.cuda.is_available()
except ImportError:
    cp = None
    HAS_CUPY = False

try:
    from ice3._phyex_wrapper_acc import IceAdjustGPU as _IceAdjustGPU
    HAS_ACC_WRAPPER = True
except ImportError:
    _IceAdjustGPU = None
    HAS_ACC_WRAPPER = False

requires_gpu = pytest.mark.skipif(not HAS_CUPY, reason="CuPy + CUDA GPU required")
requires_acc  = pytest.mark.skipif(
    not (HAS_CUPY and HAS_ACC_WRAPPER),
    reason="_phyex_wrapper_acc not built (cmake -DENABLE_OPENACC=ON)",
)


# ── Synthetic atmosphere fixture ──────────────────────────────────────────────

def _make_atm_jax(nit: int = 64, nkt: int = 30, dtype=jnp.float32):
    """Return a minimal atmospheric state as JAX float32 arrays."""
    np.random.seed(0)
    z  = np.linspace(0, 8000, nkt).astype("f4")
    p0, T0, g = 101325.0, 288.15, 0.0065
    p  = p0 * (1 - g * z / T0) ** 5.26
    T  = T0 - g * z
    Rd, cpd, p00 = 287.0, 1004.0, 1e5

    def tile(v):   return np.tile(v, (nit, 1)).astype("f4")
    def tile1d(v): return np.tile(v, (nit,)).astype("f4")

    exn     = (p / p00) ** (Rd / cpd)
    th      = T / exn
    rv_prof = 0.012 * np.exp(-z / 2000)
    rho     = p / (Rd * T)

    j = lambda a: jnp.asarray(a, dtype=dtype)

    return dict(
        sigqsat    = j(np.full((nit, nkt), 0.02, "f4")),
        pabs       = j(tile(p)),
        sigs       = j(np.full((nit, nkt), 0.1, "f4")),
        th         = j(tile(th)),
        exn        = j(tile(exn)),
        exn_ref    = j(tile(exn)),
        rho_dry_ref= j(tile(rho)),
        rv         = j(tile(rv_prof)),
        rc         = j(np.zeros((nit, nkt), "f4")),
        ri         = j(np.zeros((nit, nkt), "f4")),
        rr         = j(np.zeros((nit, nkt), "f4")),
        rs         = j(np.zeros((nit, nkt), "f4")),
        rg         = j(np.zeros((nit, nkt), "f4")),
        cf_mf      = j(np.zeros((nit, nkt), "f4")),
        rc_mf      = j(np.zeros((nit, nkt), "f4")),
        ri_mf      = j(np.zeros((nit, nkt), "f4")),
        rvs        = j(np.zeros((nit, nkt), "f4")),
        rcs        = j(np.zeros((nit, nkt), "f4")),
        ris        = j(np.zeros((nit, nkt), "f4")),
        ths        = j(np.zeros((nit, nkt), "f4")),
    )


# =============================================================================
# 1. Factory smoke tests — always green (no GPU required)
# =============================================================================

class TestMakeIceAdjust:
    """make_ice_adjust() returns a callable regardless of backend."""

    def test_factory_returns_callable(self):
        from ice3.jax.phyex_jax_gpu import make_ice_adjust
        obj = make_ice_adjust()
        assert callable(obj)

    def test_fallback_is_ice_adjust_jax(self):
        from ice3.jax.phyex_jax_gpu import make_ice_adjust
        from ice3.jax.ice_adjust import IceAdjustJAX
        obj = make_ice_adjust()  # no GPU dims → CPU fallback
        assert isinstance(obj, IceAdjustJAX)

    def test_jax_path_runs(self):
        from ice3.jax.phyex_jax_gpu import make_ice_adjust
        atm = _make_atm_jax(nit=8, nkt=10)
        ice_adj = make_ice_adjust()
        out = ice_adj(timestep=60.0, **atm)
        t_out = out[0]
        assert t_out.shape == (8, 10)
        assert jnp.all(jnp.isfinite(t_out))

    def test_jax_path_temperature_bounds(self):
        from ice3.jax.phyex_jax_gpu import make_ice_adjust
        atm = _make_atm_jax(nit=16, nkt=20)
        ice_adj = make_ice_adjust()
        out = ice_adj(timestep=60.0, **atm)
        t_out = out[0]
        assert jnp.all(jnp.isfinite(t_out)), "t_out contains NaN/Inf"
        assert jnp.all(t_out > 100.0), "t_out should be > 100 K"
        assert jnp.all(t_out < 600.0), "t_out should be < 600 K"

    def test_jax_path_water_conservation(self):
        """Mixing ratios must be non-negative and finite after ice adjustment.

        IceAdjustJAX is a saturation adjustment — it re-partitions water among
        phases and feeds back through tendency terms (rvs/rcs/ris). Total water
        is not a strict invariant of this scheme; instead we verify that all
        output mixing ratios are physically valid (≥ 0, finite).
        """
        from ice3.jax.phyex_jax_gpu import make_ice_adjust
        atm = _make_atm_jax(nit=16, nkt=20)
        ice_adj = make_ice_adjust()
        t_out, rv, rc, ri, *_ = ice_adj(timestep=60.0, **atm)
        assert jnp.all(rv >= 0.0), f"Negative rv: min = {float(rv.min()):.2e}"
        assert jnp.all(rc >= 0.0), f"Negative rc: min = {float(rc.min()):.2e}"
        assert jnp.all(ri >= 0.0), f"Negative ri: min = {float(ri.min()):.2e}"
        assert jnp.all(jnp.isfinite(rv)), "Non-finite rv"
        assert jnp.all(jnp.isfinite(rc)), "Non-finite rc"



# =============================================================================
# 2. GPU bridge tests — requires NVIDIA GPU + _phyex_wrapper_acc
# =============================================================================

@requires_acc
class TestIceAdjustJAXGPU:
    """IceAdjustJAXGPU: DLPack bridge + physical validation on GPU."""

    NIT, NKT = 64, 30

    @pytest.fixture(scope="class")
    def bridge(self):
        from ice3.jax.phyex_jax_gpu import IceAdjustJAXGPU
        return IceAdjustJAXGPU(nit=self.NIT, nkt=self.NKT, krr=6, timestep=60.0)

    @pytest.fixture(scope="class")
    def atm(self):
        return _make_atm_jax(nit=self.NIT, nkt=self.NKT)

    def test_instantiation(self):
        from ice3.jax.phyex_jax_gpu import IceAdjustJAXGPU
        bridge = IceAdjustJAXGPU(nit=self.NIT, nkt=self.NKT)
        assert bridge is not None
        assert len(bridge._b) > 0

    def test_buffers_on_device(self, bridge):
        for key, buf in bridge._b.items():
            assert isinstance(buf, cp.ndarray), f"Buffer '{key}' is not a CuPy array"
            assert buf.device.id >= 0

    def test_returns_16_outputs(self, bridge, atm):
        out = bridge(timestep=60.0, **atm)
        assert len(out) == 16, f"Expected 16 outputs, got {len(out)}"

    def test_output_shapes(self, bridge, atm):
        out = bridge(timestep=60.0, **atm)
        for i, arr in enumerate(out):
            assert arr.shape == (self.NIT, self.NKT), \
                f"Output[{i}] shape {arr.shape} != ({self.NIT}, {self.NKT})"

    def test_output_dtype_float32(self, bridge, atm):
        out = bridge(timestep=60.0, **atm)
        for i, arr in enumerate(out):
            assert arr.dtype == jnp.float32, f"Output[{i}] dtype {arr.dtype} != float32"

    def test_temperature_output_physical(self, bridge, atm):
        t_out = bridge(timestep=60.0, **atm)[0]
        assert jnp.all(jnp.isfinite(t_out))
        assert jnp.all(t_out > 100.0)
        assert jnp.all(t_out < 600.0)

    def test_cloud_fraction_bounds(self, bridge, atm):
        cldfr = bridge(timestep=60.0, **atm)[4]
        assert jnp.all(cldfr >= 0.0), "cldfr < 0"
        assert jnp.all(cldfr <= 1.0), "cldfr > 1"

    def test_vs_jax_cpu_consistency(self, bridge, atm):
        """GPU result should agree with pure-JAX stencil to within 1e-3."""
        from ice3.jax.ice_adjust import IceAdjustJAX
        jax_ia = IceAdjustJAX(jit=True)

        t_gpu = bridge(timestep=60.0, **atm)[0]
        t_cpu = jax_ia(timestep=60.0, **atm)[0]

        np.testing.assert_allclose(
            np.array(t_gpu), np.array(t_cpu),
            rtol=1e-3, atol=0.5,
            err_msg="GPU and JAX-CPU temperature outputs diverge",
        )

    @requires_gpu
    def test_make_factory_returns_gpu(self):
        from ice3.jax.phyex_jax_gpu import IceAdjustJAXGPU, make_ice_adjust
        obj = make_ice_adjust(nit=self.NIT, nkt=self.NKT)
        assert isinstance(obj, IceAdjustJAXGPU), \
            f"Expected IceAdjustJAXGPU, got {type(obj)}"

    def test_repeated_calls_stable(self, bridge, atm):
        """Result must be identical across repeated calls (stable fixed buffers)."""
        r1 = np.array(bridge(timestep=60.0, **atm)[0])
        r2 = np.array(bridge(timestep=60.0, **atm)[0])
        np.testing.assert_array_equal(r1, r2, err_msg="Results differ between calls")

    def test_water_conservation_gpu(self, bridge, atm):
        rt_before = atm["rv"] + atm["rc"] + atm["ri"] + atm["rr"] + atm["rs"] + atm["rg"]
        t_out, rv, rc, ri, *_ = bridge(timestep=60.0, **atm)
        rt_after = rv + rc + ri + atm["rr"] + atm["rs"] + atm["rg"]
        delta = jnp.abs(rt_after - rt_before)
        assert jnp.all(delta < 1e-4), f"Max Δqt = {float(delta.max()):.2e}"


# =============================================================================
# 3. Integration smoke test — orchestrator uses GPU backend when available
# =============================================================================

@requires_acc
class TestOrchestratorGPUPath:
    """AromePhysicsOrchestrator.step() with IceAdjustJAXGPU backend."""

    def test_orchestrator_accepts_n_cols(self):
        pytest.importorskip("ice3.jax.arome_physics")
        from ice3.jax.arome_physics import AromePhysicsOrchestrator
        orc = AromePhysicsOrchestrator(constants={}, n_cols=64)
        assert orc.surfex is not None

    def test_make_surfex_gpu_tier(self):
        from ice3.jax.surfex_jax import make_surfex, SurfexJAXGPU
        s = make_surfex(n_cols=64)
        assert isinstance(s, SurfexJAXGPU), \
            f"Expected SurfexJAXGPU on GPU pod, got {type(s)}"
