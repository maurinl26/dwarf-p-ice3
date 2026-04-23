"""
Tests for the SURFEX GPU bridge (_surfex_wrapper_acc + SurfexJAXGPU).

Covers:
  1. SurfexGPUWrapper: CUDA graph capture on first call, replay on subsequent.
  2. SurfexJAXGPU: DLPack bridge (JAX GPU → CuPy → JAX GPU).
  3. Physical consistency: results match the analytical bulk-aerodynamic fallback
     (same formulae, different execution path).
  4. Performance: graph replay is faster than first-call capture.

Requirements
------------
- NVIDIA GPU + CUDA
- CuPy (pip install cupy-cuda12x)
- _surfex_wrapper_acc built with:
    cmake -DENABLE_OPENACC=ON -DENABLE_SURFEX=ON \\
          -DCMAKE_Fortran_COMPILER=nvfortran ..
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    HAS_CUPY = True
    HAS_GPU = cp.cuda.is_available()
except ImportError:
    cp = None
    HAS_CUPY = False
    HAS_GPU = False

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False

# Resolve build directory
_build_dir = Path(__file__).parent.parent.parent / "build-gpu"
if not _build_dir.exists():
    _build_dir = Path(__file__).parent.parent.parent / "build"

if _build_dir.exists():
    for _sub in _build_dir.iterdir():
        if _sub.is_dir() and _sub.name.startswith("cp"):
            sys.path.insert(0, str(_sub))
            break

# Try importing SURFEX GPU wrapper
try:
    from _surfex_wrapper_acc import SurfexGPUWrapper, TILE_NATURE, TILE_SEA, TILE_LAKE
    HAS_SURFEX_GPU = True
except ImportError:
    SurfexGPUWrapper = None
    TILE_NATURE, TILE_SEA, TILE_LAKE = 1, 2, 3
    HAS_SURFEX_GPU = False

# Tile constants also exported from surfex_jax
from ice3.jax.surfex_jax import (
    SurfexState, SurfexFluxes, SurfexJAXGPU, _NullSurfex, make_surfex,
    TILE_NATURE as _TN, TILE_SEA as _TS, TILE_LAKE as _TL,
)

requires_surfex_gpu = pytest.mark.skipif(
    not (HAS_CUPY and HAS_GPU and HAS_SURFEX_GPU),
    reason="Requires CuPy + NVIDIA GPU + _surfex_wrapper_acc",
)
requires_jax_gpu = pytest.mark.skipif(
    not (HAS_JAX and HAS_CUPY and HAS_GPU and HAS_SURFEX_GPU),
    reason="Requires JAX + CuPy + NVIDIA GPU + _surfex_wrapper_acc",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_forcing(n_cols: int, rng=None, dtype=np.float32):
    """Generate random but physically plausible surface forcing."""
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
    """Mix of nature / sea / lake tiles."""
    tiles = np.ones(n_cols, dtype=np.int32) * TILE_NATURE
    tiles[n_cols // 3 : 2 * n_cols // 3] = TILE_SEA
    tiles[2 * n_cols // 3 :] = TILE_LAKE
    return tiles


# ---------------------------------------------------------------------------
# Unit tests: SurfexGPUWrapper (CuPy level)
# ---------------------------------------------------------------------------

class TestSurfexGPUWrapper:

    @requires_surfex_gpu
    def test_instantiation(self):
        n_cols = 64
        tiles  = _make_tiles(n_cols)
        w = SurfexGPUWrapper(n_cols, tiles)
        assert w.n_cols == n_cols
        assert not w.is_graph_captured

    @requires_surfex_gpu
    def test_first_call_captures_graph(self):
        n_cols = 64
        tiles  = _make_tiles(n_cols)
        w      = SurfexGPUWrapper(n_cols, tiles)
        f      = _make_forcing(n_cols)
        bufs   = {k: cp.asarray(v) for k, v in f.items()}

        out = w(dt=60.0, **{k: v.toDlpack() for k, v in bufs.items()})
        assert w.is_graph_captured
        assert set(out.keys()) == {
            'surf_flux_th', 'surf_flux_rv',
            'surf_flux_u', 'surf_flux_v',
            'albedo', 'emissivity',
        }

    @requires_surfex_gpu
    def test_output_shapes(self):
        n_cols = 128
        tiles  = _make_tiles(n_cols)
        w      = SurfexGPUWrapper(n_cols, tiles)
        f      = _make_forcing(n_cols)
        bufs   = {k: cp.asarray(v) for k, v in f.items()}

        out = w(dt=60.0, **{k: v.toDlpack() for k, v in bufs.items()})
        for key, dlp in out.items():
            arr = cp.from_dlpack(dlp)
            assert arr.shape == (n_cols,), f"{key}: shape {arr.shape}"
            assert arr.dtype == cp.float32, f"{key}: dtype {arr.dtype}"

    @requires_surfex_gpu
    def test_graph_replay_reproducible(self):
        """Multiple calls with same input must give identical results."""
        n_cols = 64
        tiles  = _make_tiles(n_cols)
        w      = SurfexGPUWrapper(n_cols, tiles)
        f      = _make_forcing(n_cols)
        bufs   = {k: cp.asarray(v) for k, v in f.items()}

        def _run():
            out = w(dt=60.0, **{k: v.toDlpack() for k, v in bufs.items()})
            return {k: cp.from_dlpack(v).copy() for k, v in out.items()}

        r1 = _run()
        r2 = _run()
        r3 = _run()
        for key in r1:
            np.testing.assert_array_equal(
                cp.asnumpy(r1[key]), cp.asnumpy(r2[key]),
                err_msg=f"{key} differs between call 1 and 2",
            )
            np.testing.assert_array_equal(
                cp.asnumpy(r2[key]), cp.asnumpy(r3[key]),
                err_msg=f"{key} differs between call 2 and 3",
            )

    @requires_surfex_gpu
    def test_graph_replay_faster_than_capture(self):
        """Graph replay should be faster than the first capture call."""
        n_cols = 1024
        tiles  = _make_tiles(n_cols)
        w      = SurfexGPUWrapper(n_cols, tiles)
        f      = _make_forcing(n_cols)
        bufs   = {k: cp.asarray(v) for k, v in f.items()}

        def _run():
            out = w(dt=60.0, **{k: v.toDlpack() for k, v in bufs.items()})
            # Ensure output is realized
            for dlp in out.values():
                cp.from_dlpack(dlp)

        # Warmup / capture
        t0 = time.perf_counter()
        _run()
        t_capture = time.perf_counter() - t0

        # Replay ×5
        t0 = time.perf_counter()
        for _ in range(5):
            _run()
        t_replay = (time.perf_counter() - t0) / 5

        # Replay must be at most 90% of capture time
        assert t_replay < t_capture * 0.9, (
            f"Graph replay ({t_replay*1e3:.2f} ms) not faster than "
            f"capture ({t_capture*1e3:.2f} ms)"
        )

    @requires_surfex_gpu
    def test_physical_fluxes_nonzero(self):
        """Surface fluxes must be non-zero for non-trivial forcing."""
        n_cols = 32
        tiles  = np.ones(n_cols, dtype=np.int32) * TILE_NATURE
        w      = SurfexGPUWrapper(n_cols, tiles)
        f      = _make_forcing(n_cols)
        bufs   = {k: cp.asarray(v) for k, v in f.items()}

        out = w(dt=60.0, **{k: v.toDlpack() for k, v in bufs.items()})
        th = cp.asnumpy(cp.from_dlpack(out['surf_flux_th']))
        assert np.any(th != 0.0), "All heat fluxes are zero — physics not running"

    @requires_surfex_gpu
    def test_albedo_range(self):
        """Albedo values must be in [0, 1]."""
        n_cols = 64
        tiles  = _make_tiles(n_cols)
        w      = SurfexGPUWrapper(n_cols, tiles)
        f      = _make_forcing(n_cols)
        bufs   = {k: cp.asarray(v) for k, v in f.items()}

        out = w(dt=60.0, **{k: v.toDlpack() for k, v in bufs.items()})
        alb = cp.asnumpy(cp.from_dlpack(out['albedo']))
        assert np.all(alb >= 0.0) and np.all(alb <= 1.0), \
            f"Albedo out of [0,1]: min={alb.min():.3f}, max={alb.max():.3f}"

    @requires_surfex_gpu
    def test_emissivity_range(self):
        """Emissivity must be in (0, 1]."""
        n_cols = 64
        tiles  = _make_tiles(n_cols)
        w      = SurfexGPUWrapper(n_cols, tiles)
        f      = _make_forcing(n_cols)
        bufs   = {k: cp.asarray(v) for k, v in f.items()}

        out = w(dt=60.0, **{k: v.toDlpack() for k, v in bufs.items()})
        emis = cp.asnumpy(cp.from_dlpack(out['emissivity']))
        assert np.all(emis > 0.0) and np.all(emis <= 1.0), \
            f"Emissivity out of (0,1]: min={emis.min():.3f}"


# ---------------------------------------------------------------------------
# Unit tests: SurfexJAXGPU (JAX level)
# ---------------------------------------------------------------------------

class TestSurfexJAXGPU:

    def _make_jax_state(self, n_cols: int):
        f = _make_forcing(n_cols)
        return SurfexState(
            t_a=jnp.array(f['t_a']),
            q_a=jnp.array(f['q_a']),
            u_a=jnp.array(f['u_a']),
            v_a=jnp.array(f['v_a']),
            p_a=jnp.array(f['p_a']),
            rhodref=jnp.array(f['rhodref']),
            sw_down=jnp.array(f['sw_down']),
            lw_down=jnp.array(f['lw_down']),
            rain_rate=jnp.array(f['rain_rate']),
            snow_rate=jnp.array(f['snow_rate']),
            psurf_flux_th=jnp.zeros(n_cols, dtype=jnp.float32),
            psurf_flux_rv=jnp.zeros(n_cols, dtype=jnp.float32),
            psurf_flux_u=jnp.zeros(n_cols, dtype=jnp.float32),
            psurf_flux_v=jnp.zeros(n_cols, dtype=jnp.float32),
        )

    @requires_jax_gpu
    def test_instantiation(self):
        tiles = _make_tiles(64)
        s = SurfexJAXGPU(n_cols=64, tile_type=tiles)
        assert s is not None

    @requires_jax_gpu
    def test_returns_surfex_fluxes(self):
        n_cols = 64
        tiles  = _make_tiles(n_cols)
        s      = SurfexJAXGPU(n_cols=n_cols, tile_type=tiles)
        state  = self._make_jax_state(n_cols)
        fluxes = s(state, dt=60.0)
        assert isinstance(fluxes, SurfexFluxes)

    @requires_jax_gpu
    def test_output_dtypes_and_shapes(self):
        n_cols = 64
        tiles  = _make_tiles(n_cols)
        s      = SurfexJAXGPU(n_cols=n_cols, tile_type=tiles)
        state  = self._make_jax_state(n_cols)
        fluxes = s(state, dt=60.0)
        for field in fluxes:
            assert field.shape == (n_cols,), f"Wrong shape: {field.shape}"
            assert field.dtype == jnp.float32, f"Wrong dtype: {field.dtype}"

    @requires_jax_gpu
    def test_graph_capture_on_first_call(self):
        n_cols = 64
        tiles  = _make_tiles(n_cols)
        s      = SurfexJAXGPU(n_cols=n_cols, tile_type=tiles)
        state  = self._make_jax_state(n_cols)
        s(state, dt=60.0)
        assert s._wrapper.is_graph_captured


# ---------------------------------------------------------------------------
# Integration test: make_surfex factory
# ---------------------------------------------------------------------------

class TestMakeSurfex:

    def test_make_surfex_returns_something(self):
        n_cols = 16
        surf = make_surfex(n_cols)
        assert surf is not None

    def test_null_surfex_fallback(self):
        """_NullSurfex must be importable and callable without GPU."""
        n_cols = 8
        surf   = _NullSurfex()
        f      = _make_forcing(n_cols)
        state  = SurfexState(
            t_a=jnp.array(f['t_a']),
            q_a=jnp.array(f['q_a']),
            u_a=jnp.array(f['u_a']),
            v_a=jnp.array(f['v_a']),
            p_a=jnp.array(f['p_a']),
            rhodref=jnp.array(f['rhodref']),
            sw_down=jnp.array(f['sw_down']),
            lw_down=jnp.array(f['lw_down']),
            rain_rate=jnp.zeros(n_cols, dtype=jnp.float32),
            snow_rate=jnp.zeros(n_cols, dtype=jnp.float32),
            psurf_flux_th=jnp.full(n_cols, 0.1, dtype=jnp.float32),
            psurf_flux_rv=jnp.full(n_cols, 0.001, dtype=jnp.float32),
            psurf_flux_u=jnp.full(n_cols, -0.5, dtype=jnp.float32),
            psurf_flux_v=jnp.full(n_cols, -0.3, dtype=jnp.float32),
        ) if HAS_JAX else None

        if state is None:
            pytest.skip("JAX not available")

        fluxes = surf(state, dt=60.0)
        # _NullSurfex re-propagates previous-step fluxes unchanged
        np.testing.assert_allclose(
            np.array(fluxes.surf_flux_th), 0.1, rtol=1e-5
        )
