"""
Scientific validation of SurfexJAXGPU vs SurfexJAX (CPU).

TestSurfexCPUGPUConsistency  — numerical CPU vs GPU comparison
TestSurfexPhysicsInvariants  — physical sign/bound checks (GPU only)
TestSurfexMultiTile          — finite outputs for each tile type

Both GPU classes skip cleanly without NVIDIA GPU + CuPy + _surfex_wrapper_acc.
TestSurfexCPUGPUConsistency additionally skips without libsurfex_offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False

try:
    import cupy as cp
    HAS_CUPY = True
    HAS_GPU = cp.cuda.is_available()
except ImportError:
    cp = None
    HAS_CUPY = False
    HAS_GPU = False

_build_dir = Path(__file__).parent.parent.parent / "build-gpu"
if not _build_dir.exists():
    _build_dir = Path(__file__).parent.parent.parent / "build"
if _build_dir.exists():
    for _sub in _build_dir.iterdir():
        if _sub.is_dir() and _sub.name.startswith("cp"):
            sys.path.insert(0, str(_sub))
            break

try:
    from _surfex_wrapper_acc import SurfexGPUWrapper, TILE_NATURE, TILE_SEA, TILE_LAKE
    HAS_SURFEX_GPU = True
except ImportError:
    SurfexGPUWrapper = None
    TILE_NATURE, TILE_SEA, TILE_LAKE = 1, 2, 3
    HAS_SURFEX_GPU = False

if HAS_JAX:
    from ice3.jax.surfex_jax import (
        SurfexState, SurfexFluxes, SurfexJAXGPU, _NullSurfex, make_surfex,
        TILE_NATURE as _TN, TILE_SEA as _TS, TILE_LAKE as _TL,
    )

# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------
requires_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
requires_jax_gpu = pytest.mark.skipif(
    not (HAS_JAX and HAS_CUPY and HAS_GPU and HAS_SURFEX_GPU),
    reason="Requires JAX + CuPy + NVIDIA GPU + _surfex_wrapper_acc",
)

try:
    from ice3.jax.surfex_jax import SurfexJAX
    _cpu_surfex = SurfexJAX()
    HAS_SURFEX_CPU = _cpu_surfex._lib._available
except Exception:
    HAS_SURFEX_CPU = False

requires_surfex_cpu = pytest.mark.skipif(
    not HAS_SURFEX_CPU,
    reason="Requires libsurfex_offline.{so,dylib} — run build_libsurfex.sh",
)

# ---------------------------------------------------------------------------
# Helpers (local copies — no cross-test imports)
# ---------------------------------------------------------------------------

def _make_forcing(n_cols: int, rng=None, dtype=np.float32):
    if rng is None:
        rng = np.random.default_rng(42)
    return dict(
        t_a=rng.uniform(260.0, 310.0, n_cols).astype(dtype),
        q_a=rng.uniform(0.001, 0.020, n_cols).astype(dtype),
        u_a=rng.uniform(-10.0, 10.0, n_cols).astype(dtype),
        v_a=rng.uniform(-10.0, 10.0, n_cols).astype(dtype),
        p_a=rng.uniform(90000.0, 101325.0, n_cols).astype(dtype),
        rhodref=rng.uniform(1.1, 1.3, n_cols).astype(dtype),
        sw_down=rng.uniform(0.0, 800.0, n_cols).astype(dtype),
        lw_down=rng.uniform(200.0, 450.0, n_cols).astype(dtype),
        rain_rate=np.zeros(n_cols, dtype=dtype),
        snow_rate=np.zeros(n_cols, dtype=dtype),
    )


def _make_tiles(n_cols: int) -> np.ndarray:
    tiles = np.ones(n_cols, dtype=np.int32) * TILE_NATURE
    tiles[n_cols // 3 : 2 * n_cols // 3] = TILE_SEA
    tiles[2 * n_cols // 3 :] = TILE_LAKE
    return tiles


def _make_jax_state(n_cols: int, t_skin=None) -> "SurfexState":
    f = _make_forcing(n_cols)
    ts = jnp.zeros(n_cols, dtype=jnp.float32) if t_skin is None else t_skin
    return SurfexState(
        t_a=jnp.array(f["t_a"]),
        q_a=jnp.array(f["q_a"]),
        u_a=jnp.array(f["u_a"]),
        v_a=jnp.array(f["v_a"]),
        p_a=jnp.array(f["p_a"]),
        rhodref=jnp.array(f["rhodref"]),
        sw_down=jnp.array(f["sw_down"]),
        lw_down=jnp.array(f["lw_down"]),
        rain_rate=jnp.zeros(n_cols, dtype=jnp.float32),
        snow_rate=jnp.zeros(n_cols, dtype=jnp.float32),
        psurf_flux_th=jnp.full(n_cols, 0.05, dtype=jnp.float32),
        psurf_flux_rv=jnp.full(n_cols, 5e-4, dtype=jnp.float32),
        psurf_flux_u=jnp.full(n_cols, -0.1, dtype=jnp.float32),
        psurf_flux_v=jnp.full(n_cols, -0.05, dtype=jnp.float32),
        t_skin=ts,
    )


# ---------------------------------------------------------------------------
# CPU vs GPU consistency
# ---------------------------------------------------------------------------

class TestSurfexCPUGPUConsistency:
    """
    Compare SurfexJAX (CPU Fortran, float64 internally) with SurfexJAXGPU
    (OpenACC, float32 native) on the same forcing.

    Skipped if either backend is unavailable.
    """

    @requires_jax_gpu
    @requires_surfex_cpu
    def test_flux_th_cpu_vs_gpu(self):
        n = 64
        state = _make_jax_state(n)
        cpu = SurfexJAX()
        gpu = SurfexJAXGPU(n_cols=n)
        f_cpu = cpu(state, dt=60.0)
        f_gpu = gpu(state, dt=60.0)
        np.testing.assert_allclose(
            np.array(f_gpu.surf_flux_th), np.array(f_cpu.surf_flux_th),
            atol=1e-3, rtol=1e-2,
            err_msg="flux_th CPU vs GPU mismatch",
        )

    @requires_jax_gpu
    @requires_surfex_cpu
    def test_flux_rv_cpu_vs_gpu(self):
        n = 64
        state = _make_jax_state(n)
        cpu = SurfexJAX()
        gpu = SurfexJAXGPU(n_cols=n)
        f_cpu = cpu(state, dt=60.0)
        f_gpu = gpu(state, dt=60.0)
        np.testing.assert_allclose(
            np.array(f_gpu.surf_flux_rv), np.array(f_cpu.surf_flux_rv),
            atol=1e-5, rtol=1e-2,
            err_msg="flux_rv CPU vs GPU mismatch",
        )

    @requires_jax_gpu
    @requires_surfex_cpu
    def test_flux_u_cpu_vs_gpu(self):
        n = 64
        state = _make_jax_state(n)
        cpu = SurfexJAX()
        gpu = SurfexJAXGPU(n_cols=n)
        f_cpu = cpu(state, dt=60.0)
        f_gpu = gpu(state, dt=60.0)
        np.testing.assert_allclose(
            np.array(f_gpu.surf_flux_u), np.array(f_cpu.surf_flux_u),
            atol=1e-3, rtol=1e-2,
            err_msg="flux_u CPU vs GPU mismatch",
        )

    @requires_jax_gpu
    @requires_surfex_cpu
    def test_flux_v_cpu_vs_gpu(self):
        n = 64
        state = _make_jax_state(n)
        cpu = SurfexJAX()
        gpu = SurfexJAXGPU(n_cols=n)
        f_cpu = cpu(state, dt=60.0)
        f_gpu = gpu(state, dt=60.0)
        np.testing.assert_allclose(
            np.array(f_gpu.surf_flux_v), np.array(f_cpu.surf_flux_v),
            atol=1e-3, rtol=1e-2,
            err_msg="flux_v CPU vs GPU mismatch",
        )

    @requires_jax_gpu
    @requires_surfex_cpu
    def test_albedo_cpu_vs_gpu(self):
        n = 64
        state = _make_jax_state(n)
        cpu = SurfexJAX()
        gpu = SurfexJAXGPU(n_cols=n)
        f_cpu = cpu(state, dt=60.0)
        f_gpu = gpu(state, dt=60.0)
        np.testing.assert_allclose(
            np.array(f_gpu.albedo), np.array(f_cpu.albedo),
            atol=5e-4, rtol=1e-3,
            err_msg="albedo CPU vs GPU mismatch",
        )

    @requires_jax_gpu
    @requires_surfex_cpu
    def test_emissivity_cpu_vs_gpu(self):
        n = 64
        state = _make_jax_state(n)
        cpu = SurfexJAX()
        gpu = SurfexJAXGPU(n_cols=n)
        f_cpu = cpu(state, dt=60.0)
        f_gpu = gpu(state, dt=60.0)
        np.testing.assert_allclose(
            np.array(f_gpu.emissivity), np.array(f_cpu.emissivity),
            atol=5e-4, rtol=1e-3,
            err_msg="emissivity CPU vs GPU mismatch",
        )


# ---------------------------------------------------------------------------
# Physical invariants
# ---------------------------------------------------------------------------

class TestSurfexPhysicsInvariants:
    """Sign and bound checks on GPU SURFEX output."""

    @requires_jax_gpu
    def test_flux_th_warm_surface(self):
        """t_a < surface → sensible heat flux must be positive (upward)."""
        n = 32
        f = _make_forcing(n)
        # Force cool atmosphere, warm surface (t_skin=305 K)
        f["t_a"][:] = 280.0
        state = SurfexState(
            t_a=jnp.array(f["t_a"]),
            q_a=jnp.array(f["q_a"]),
            u_a=jnp.array(f["u_a"]),
            v_a=jnp.array(f["v_a"]),
            p_a=jnp.array(f["p_a"]),
            rhodref=jnp.array(f["rhodref"]),
            sw_down=jnp.array(f["sw_down"]),
            lw_down=jnp.array(f["lw_down"]),
            rain_rate=jnp.zeros(n, dtype=jnp.float32),
            snow_rate=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_th=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_rv=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_u=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_v=jnp.zeros(n, dtype=jnp.float32),
            t_skin=jnp.full(n, 305.0, dtype=jnp.float32),
        )
        surf = SurfexJAXGPU(n_cols=n)
        fluxes = surf(state, dt=60.0)
        assert np.all(np.array(fluxes.surf_flux_th) > 0), \
            "Warm surface must give positive upward sensible heat flux"

    @requires_jax_gpu
    def test_flux_th_cool_surface(self):
        """t_a > surface → sensible heat flux must be negative (downward)."""
        n = 32
        f = _make_forcing(n)
        f["t_a"][:] = 310.0
        state = SurfexState(
            t_a=jnp.array(f["t_a"]),
            q_a=jnp.array(f["q_a"]),
            u_a=jnp.array(f["u_a"]),
            v_a=jnp.array(f["v_a"]),
            p_a=jnp.array(f["p_a"]),
            rhodref=jnp.array(f["rhodref"]),
            sw_down=jnp.array(f["sw_down"]),
            lw_down=jnp.array(f["lw_down"]),
            rain_rate=jnp.zeros(n, dtype=jnp.float32),
            snow_rate=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_th=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_rv=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_u=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_v=jnp.zeros(n, dtype=jnp.float32),
            t_skin=jnp.full(n, 280.0, dtype=jnp.float32),
        )
        surf = SurfexJAXGPU(n_cols=n)
        fluxes = surf(state, dt=60.0)
        assert np.all(np.array(fluxes.surf_flux_th) < 0), \
            "Cool surface (t_a > t_skin) must give negative sensible heat flux"

    @requires_jax_gpu
    def test_momentum_flux_opposes_wind(self):
        """Positive u_a must produce negative flux_u (surface drag)."""
        n = 32
        f = _make_forcing(n)
        f["u_a"][:] = 5.0
        f["v_a"][:] = 0.0
        state = _make_jax_state(n)
        state = state._replace(
            u_a=jnp.full(n, 5.0, dtype=jnp.float32),
            v_a=jnp.zeros(n, dtype=jnp.float32),
        )
        surf = SurfexJAXGPU(n_cols=n)
        fluxes = surf(state, dt=60.0)
        assert np.all(np.array(fluxes.surf_flux_u) < 0), \
            "Positive u_a must produce negative momentum flux (drag)"

    @requires_jax_gpu
    def test_albedo_bounds(self):
        n = 64
        state = _make_jax_state(n)
        surf = SurfexJAXGPU(n_cols=n)
        fluxes = surf(state, dt=60.0)
        alb = np.array(fluxes.albedo)
        assert np.all(alb >= 0.0), "Albedo below 0"
        assert np.all(alb <= 1.0), "Albedo above 1"

    @requires_jax_gpu
    def test_emissivity_bounds(self):
        n = 64
        state = _make_jax_state(n)
        surf = SurfexJAXGPU(n_cols=n)
        fluxes = surf(state, dt=60.0)
        emiss = np.array(fluxes.emissivity)
        assert np.all(emiss > 0.0), "Emissivity <= 0"
        assert np.all(emiss <= 1.0), "Emissivity > 1"

    @requires_jax_gpu
    def test_t_skin_netatmo_effect(self):
        """t_skin=305 K must produce different flux_th than sentinel (t_skin=0)."""
        n = 32
        state_sentinel = _make_jax_state(n, t_skin=jnp.zeros(n, dtype=jnp.float32))
        state_warm = _make_jax_state(n, t_skin=jnp.full(n, 305.0, dtype=jnp.float32))
        surf = SurfexJAXGPU(n_cols=n)
        f_sentinel = surf(state_sentinel, dt=60.0)
        f_warm = surf(state_warm, dt=60.0)
        assert not np.allclose(
            np.array(f_sentinel.surf_flux_th),
            np.array(f_warm.surf_flux_th),
            rtol=1e-3,
        ), "Netatmo t_skin=305 K must change flux_th relative to sentinel"


# ---------------------------------------------------------------------------
# Multi-tile shape and finiteness checks
# ---------------------------------------------------------------------------

class TestSurfexMultiTile:
    """Check that all tile types produce finite outputs with the expected shape."""

    def _run_tile(self, n: int, tile_id: int) -> "SurfexFluxes":
        state = _make_jax_state(n)
        surf = SurfexJAXGPU(n_cols=n, tile=tile_id)
        return surf(state, dt=60.0)

    @requires_jax_gpu
    def test_nature_tile_outputs_finite(self):
        fluxes = self._run_tile(128, TILE_NATURE)
        for arr in fluxes:
            assert np.all(np.isfinite(np.array(arr))), \
                f"TILE_NATURE: {arr} contains non-finite values"

    @requires_jax_gpu
    def test_sea_tile_outputs_finite(self):
        fluxes = self._run_tile(128, TILE_SEA)
        for arr in fluxes:
            assert np.all(np.isfinite(np.array(arr))), \
                f"TILE_SEA: {arr} contains non-finite values"

    @requires_jax_gpu
    def test_lake_tile_outputs_finite(self):
        fluxes = self._run_tile(128, TILE_LAKE)
        for arr in fluxes:
            assert np.all(np.isfinite(np.array(arr))), \
                f"TILE_LAKE: {arr} contains non-finite values"

    @requires_jax_gpu
    def test_mixed_tile_shapes(self):
        """256-column mixed forcing → all flux fields have shape (256,)."""
        n = 256
        state = _make_jax_state(n)
        surf = SurfexJAXGPU(n_cols=n)
        fluxes = surf(state, dt=60.0)
        for arr in fluxes:
            assert np.array(arr).shape == (n,), \
                f"Expected shape ({n},), got {np.array(arr).shape}"
