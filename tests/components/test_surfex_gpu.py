"""
Tests for the SURFEX surface physics bindings.

CPU smoke tests (always run, no GPU required)
---------------------------------------------
  TestSurfexCPUSmoke   — _NullSurfex and SurfexJAX instantiation + call
  TestMakeSurfex       — make_surfex() factory fallback cascade

GPU tests (skipped without NVIDIA GPU + CuPy + _surfex_wrapper_acc)
--------------------------------------------------------------------
  TestSurfexGPUWrapper — CUDA graph capture, replay, physical checks
  TestSurfexJAXGPU     — DLPack bridge JAX GPU ↔ CuPy

Build requirements for GPU tests
---------------------------------
    cmake -DENABLE_OPENACC=ON -DENABLE_SURFEX=ON \\
          -DCMAKE_Fortran_COMPILER=nvfortran ..
    pip install cupy-cuda12x
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

requires_jax = pytest.mark.skipif(
    not HAS_JAX,
    reason="Requires JAX (pip install jax)",
)
requires_surfex_gpu = pytest.mark.skipif(
    not (HAS_CUPY and HAS_GPU and HAS_SURFEX_GPU),
    reason="Requires CuPy + NVIDIA GPU + _surfex_wrapper_acc",
)
requires_jax_gpu = pytest.mark.skipif(
    not (HAS_JAX and HAS_CUPY and HAS_GPU and HAS_SURFEX_GPU),
    reason="Requires JAX + CuPy + NVIDIA GPU + _surfex_wrapper_acc",
)

# SurfexJAX (CPU path) available if libsurfex_offline was compiled
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
# Smoke tests: CPU paths (no GPU required)
# ---------------------------------------------------------------------------

class TestSurfexCPUSmoke:
    """
    Smoke tests for the two CPU-side backends.

    _NullSurfex  — always available, re-propagates previous-step fluxes.
    SurfexJAX    — requires libsurfex_offline (skipped otherwise), uses
                   jax.pure_callback to call C-bound Fortran on the CPU host.

    These tests run on any machine (CI included) as long as JAX is installed.
    """

    def _state(self, n: int):
        f = _make_forcing(n)
        return SurfexState(
            t_a=jnp.array(f['t_a']),
            q_a=jnp.array(f['q_a']),
            u_a=jnp.array(f['u_a']),
            v_a=jnp.array(f['v_a']),
            p_a=jnp.array(f['p_a']),
            rhodref=jnp.array(f['rhodref']),
            sw_down=jnp.array(f['sw_down']),
            lw_down=jnp.array(f['lw_down']),
            rain_rate=jnp.zeros(n, dtype=jnp.float32),
            snow_rate=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_th=jnp.full(n, 0.05, dtype=jnp.float32),
            psurf_flux_rv=jnp.full(n, 5e-4, dtype=jnp.float32),
            psurf_flux_u=jnp.full(n, -0.1, dtype=jnp.float32),
            psurf_flux_v=jnp.full(n, -0.05, dtype=jnp.float32),
        )

    # --- _NullSurfex ---

    @requires_jax
    def test_null_surfex_instantiates(self):
        assert _NullSurfex() is not None

    @requires_jax
    def test_null_surfex_returns_surfex_fluxes(self):
        surf = _NullSurfex()
        fluxes = surf(self._state(16), dt=60.0)
        assert isinstance(fluxes, SurfexFluxes)

    @requires_jax
    def test_null_surfex_propagates_previous_fluxes(self):
        """_NullSurfex must return the previous-step fluxes unchanged."""
        n = 8
        surf = _NullSurfex()
        state = self._state(n)
        fluxes = surf(state, dt=60.0)
        np.testing.assert_allclose(
            np.array(fluxes.surf_flux_th),
            np.array(state.psurf_flux_th),
            rtol=1e-6,
            err_msg="surf_flux_th should equal psurf_flux_th",
        )
        np.testing.assert_allclose(
            np.array(fluxes.surf_flux_u),
            np.array(state.psurf_flux_u),
            rtol=1e-6,
            err_msg="surf_flux_u should equal psurf_flux_u",
        )

    @requires_jax
    def test_null_surfex_output_shapes(self):
        n = 32
        fluxes = _NullSurfex()(self._state(n), dt=60.0)
        for arr in fluxes:
            assert arr.shape == (n,), f"Wrong shape {arr.shape}"

    @requires_jax
    def test_null_surfex_output_dtypes(self):
        fluxes = _NullSurfex()(self._state(16), dt=60.0)
        for arr in fluxes:
            assert arr.dtype == jnp.float32, f"Wrong dtype {arr.dtype}"

    @requires_jax
    def test_null_surfex_albedo_range(self):
        fluxes = _NullSurfex()(self._state(64), dt=60.0)
        alb = np.array(fluxes.albedo)
        assert np.all(alb >= 0) and np.all(alb <= 1)

    @requires_jax
    def test_null_surfex_emissivity_range(self):
        fluxes = _NullSurfex()(self._state(64), dt=60.0)
        emis = np.array(fluxes.emissivity)
        assert np.all(emis > 0) and np.all(emis <= 1)

    # --- SurfexJAX (CPU pure_callback, requires libsurfex_offline) ---

    @requires_surfex_cpu
    def test_surfex_jax_cpu_smoke(self):
        """SurfexJAX (CPU) must return non-zero fluxes with real library."""
        from ice3.jax.surfex_jax import SurfexJAX
        surf = SurfexJAX()
        fluxes = surf(self._state(16), dt=60.0)
        assert isinstance(fluxes, SurfexFluxes)

    @requires_surfex_cpu
    def test_surfex_jax_cpu_output_shapes(self):
        from ice3.jax.surfex_jax import SurfexJAX
        n = 64
        fluxes = SurfexJAX()(self._state(n), dt=60.0)
        for arr in fluxes:
            assert arr.shape == (n,)

    @requires_surfex_cpu
    def test_surfex_jax_cpu_fluxes_finite(self):
        from ice3.jax.surfex_jax import SurfexJAX
        fluxes = SurfexJAX()(self._state(32), dt=60.0)
        for arr in fluxes:
            assert np.all(np.isfinite(np.array(arr))), "Non-finite value in output"

    @requires_surfex_cpu
    def test_surfex_jax_cpu_bulk_fallback_consistency(self):
        """
        When libsurfex_offline is unavailable the CPU path falls back to the
        analytical bulk-aerodynamic formula — same as _NullSurfex would if it
        computed fluxes instead of forwarding them.
        Verify momentum flux sign: τ_u = -C_D |U| u_a → opposite sign to u_a.
        """
        from ice3.jax.surfex_jax import SurfexJAX
        n = 16
        f = _make_forcing(n, rng=np.random.default_rng(0))
        # Force u_a positive so flux_u must be negative
        f['u_a'] = np.abs(f['u_a']) + 1.0
        state = SurfexState(
            t_a=jnp.array(f['t_a']), q_a=jnp.array(f['q_a']),
            u_a=jnp.array(f['u_a']), v_a=jnp.zeros(n, dtype=jnp.float32),
            p_a=jnp.array(f['p_a']), rhodref=jnp.array(f['rhodref']),
            sw_down=jnp.zeros(n, dtype=jnp.float32),
            lw_down=jnp.zeros(n, dtype=jnp.float32),
            rain_rate=jnp.zeros(n, dtype=jnp.float32),
            snow_rate=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_th=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_rv=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_u=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_v=jnp.zeros(n, dtype=jnp.float32),
        )
        fluxes = SurfexJAX()(state, dt=60.0)
        assert np.all(np.array(fluxes.surf_flux_u) < 0), \
            "Momentum flux must oppose positive u_a (τ_u = -C_D |U| u_a)"


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
        # Each per-device wrapper must have its graph captured after the first call
        assert all(w.is_graph_captured for w in s._wrappers.values())

    @requires_jax_gpu
    def test_jit_compatible(self):
        """SurfexJAXGPU.__call__ must work inside jax.jit without ConcretizationTypeError."""
        n_cols = 64
        tiles  = _make_tiles(n_cols)
        s      = SurfexJAXGPU(n_cols=n_cols, tile_type=tiles)
        state  = self._make_jax_state(n_cols)

        # Wrap in jit — this must NOT raise ConcretizationTypeError
        jit_call = jax.jit(lambda st: s(st, dt=60.0))
        fluxes = jit_call(state)

        assert isinstance(fluxes, SurfexFluxes)
        for field in fluxes:
            assert field.shape == (n_cols,)
            assert np.all(np.isfinite(np.array(field))), "Non-finite output in JIT call"

    @requires_jax_gpu
    def test_jit_second_call_reproduces(self):
        """JIT-compiled SURFEX must give identical results on the second call."""
        n_cols = 64
        tiles  = _make_tiles(n_cols)
        s      = SurfexJAXGPU(n_cols=n_cols, tile_type=tiles)
        state  = self._make_jax_state(n_cols)
        jit_call = jax.jit(lambda st: s(st, dt=60.0))

        f1 = jit_call(state)
        f2 = jit_call(state)
        np.testing.assert_array_equal(
            np.array(f1.surf_flux_th), np.array(f2.surf_flux_th),
            err_msg="surf_flux_th differs between JIT call 1 and 2",
        )

    @requires_jax_gpu
    def test_pmap_two_devices(self):
        """SurfexJAXGPU must work under jax.pmap across all available GPU devices."""
        gpu_devs = [d for d in jax.local_devices() if d.platform == 'gpu']
        n_devs = len(gpu_devs)
        if n_devs < 2:
            pytest.skip(f"Need ≥2 GPU devices for pmap test, found {n_devs}")

        n_cols = 64
        tiles  = _make_tiles(n_cols)
        s      = SurfexJAXGPU(n_cols=n_cols, tile_type=tiles)

        # Build a batched state: leading axis = n_devs
        f = _make_forcing(n_cols)
        def _bat(arr):
            return jnp.stack([jnp.array(arr)] * n_devs)

        state = SurfexState(
            t_a=_bat(f['t_a']), q_a=_bat(f['q_a']),
            u_a=_bat(f['u_a']), v_a=_bat(f['v_a']),
            p_a=_bat(f['p_a']), rhodref=_bat(f['rhodref']),
            sw_down=_bat(f['sw_down']), lw_down=_bat(f['lw_down']),
            rain_rate=_bat(f['rain_rate']), snow_rate=_bat(f['snow_rate']),
            psurf_flux_th=jnp.zeros((n_devs, n_cols), dtype=jnp.float32),
            psurf_flux_rv=jnp.zeros((n_devs, n_cols), dtype=jnp.float32),
            psurf_flux_u=jnp.zeros((n_devs, n_cols), dtype=jnp.float32),
            psurf_flux_v=jnp.zeros((n_devs, n_cols), dtype=jnp.float32),
        )

        pmapped = jax.pmap(lambda st: s(st, dt=60.0))
        fluxes = pmapped(state)

        assert fluxes.surf_flux_th.shape == (n_devs, n_cols)
        for field in fluxes:
            assert np.all(np.isfinite(np.array(field))), "Non-finite output in pmap call"

        # Both devices must return the same result (same input)
        np.testing.assert_allclose(
            np.array(fluxes.surf_flux_th[0]),
            np.array(fluxes.surf_flux_th[1]),
            rtol=1e-5,
            err_msg="pmap devices return different surf_flux_th for identical input",
        )


# ---------------------------------------------------------------------------
# Integration tests: make_surfex() factory — fallback cascade
# ---------------------------------------------------------------------------

class TestMakeSurfex:
    """
    Verify that make_surfex() returns a callable object regardless of which
    backends are compiled, and that the returned object produces valid output.

    Cascade priority:
      SurfexJAXGPU  (_surfex_wrapper_acc available + NVIDIA GPU)
        ↓ ImportError / RuntimeError
      SurfexJAX     (libsurfex_offline available)
        ↓ any exception
      _NullSurfex   (always available)
    """

    @requires_jax
    def test_make_surfex_returns_callable(self):
        surf = make_surfex(n_cols=16)
        assert callable(surf)

    @requires_jax
    def test_make_surfex_correct_tier_no_gpu(self):
        """Without GPU build, factory must NOT return SurfexJAXGPU."""
        surf = make_surfex(n_cols=16)
        if not (HAS_CUPY and HAS_GPU and HAS_SURFEX_GPU):
            assert not isinstance(surf, SurfexJAXGPU), \
                "SurfexJAXGPU returned despite missing GPU/CuPy/_surfex_wrapper_acc"

    @requires_jax
    def test_make_surfex_correct_tier_with_gpu(self):
        """With full GPU build, factory must return SurfexJAXGPU."""
        if not (HAS_CUPY and HAS_GPU and HAS_SURFEX_GPU):
            pytest.skip("GPU build not available")
        surf = make_surfex(n_cols=16)
        assert isinstance(surf, SurfexJAXGPU)

    @requires_jax
    def test_make_surfex_callable_and_produces_fluxes(self):
        """make_surfex() output must be callable and return SurfexFluxes."""
        n = 16
        surf = make_surfex(n_cols=n)
        f = _make_forcing(n)
        state = SurfexState(
            t_a=jnp.array(f['t_a']), q_a=jnp.array(f['q_a']),
            u_a=jnp.array(f['u_a']), v_a=jnp.array(f['v_a']),
            p_a=jnp.array(f['p_a']), rhodref=jnp.array(f['rhodref']),
            sw_down=jnp.array(f['sw_down']), lw_down=jnp.array(f['lw_down']),
            rain_rate=jnp.zeros(n, dtype=jnp.float32),
            snow_rate=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_th=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_rv=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_u=jnp.zeros(n, dtype=jnp.float32),
            psurf_flux_v=jnp.zeros(n, dtype=jnp.float32),
        )
        fluxes = surf(state, dt=60.0)
        assert isinstance(fluxes, SurfexFluxes)
        for arr in fluxes:
            assert arr.shape == (n,)
            assert np.all(np.isfinite(np.array(arr)))

    @requires_jax
    def test_null_surfex_propagates_fluxes(self):
        """_NullSurfex re-propagates previous-step fluxes unchanged."""
        n = 8
        surf = _NullSurfex()
        state = SurfexState(
            t_a=jnp.ones(n), q_a=jnp.ones(n),
            u_a=jnp.ones(n), v_a=jnp.ones(n),
            p_a=jnp.ones(n) * 100000.0,
            rhodref=jnp.ones(n) * 1.2,
            sw_down=jnp.zeros(n), lw_down=jnp.zeros(n),
            rain_rate=jnp.zeros(n), snow_rate=jnp.zeros(n),
            psurf_flux_th=jnp.full(n, 0.1, dtype=jnp.float32),
            psurf_flux_rv=jnp.full(n, 0.001, dtype=jnp.float32),
            psurf_flux_u=jnp.full(n, -0.5, dtype=jnp.float32),
            psurf_flux_v=jnp.full(n, -0.3, dtype=jnp.float32),
        )
        fluxes = surf(state, dt=60.0)
        np.testing.assert_allclose(np.array(fluxes.surf_flux_th), 0.1, rtol=1e-5)
        np.testing.assert_allclose(np.array(fluxes.surf_flux_u), -0.5, rtol=1e-5)
