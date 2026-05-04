"""
GPU performance benchmarks for IceAdjust (PHYEX), RainIce (PHYEX), and SURFEX.

Measures latency, throughput (cols/s), and GPU graph replay speedup vs
first-call (graph capture) latency. Also compares GPU against JAX-CPU wall
time to quantify the OpenACC portage gain.

Requirements:
  - CuPy + NVIDIA GPU
  - _phyex_wrapper_acc  (cmake -DENABLE_OPENACC=ON)
  - _surfex_wrapper_acc (cmake -DENABLE_OPENACC=ON -DENABLE_SURFEX=ON)
  - pytest-benchmark    (uv add pytest-benchmark)

Skip behaviour:
  - All tests skip cleanly on CPU-only runners (CI ubuntu-latest).
  - jax.block_until_ready() is called before stopping the timer — mandatory
    for accurate GPU timing because JAX dispatches are asynchronous.
"""
from __future__ import annotations

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
    HAS_CUPY = cp.cuda.is_available()
except ImportError:
    cp = None
    HAS_CUPY = False

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False

# Add Cython extension build directory to path
_build_dir = Path(__file__).parent.parent.parent / "build-gpu"
if not _build_dir.exists():
    _build_dir = Path(__file__).parent.parent.parent / "build"
if _build_dir.exists():
    for _sub in _build_dir.iterdir():
        if _sub.is_dir() and _sub.name.startswith("cp"):
            sys.path.insert(0, str(_sub))
            break

try:
    from ice3._phyex_wrapper_acc import IceAdjustGPU as _IceAdjustGPU
    from ice3._phyex_wrapper_acc import RainIceGPU as _RainIceGPU
    HAS_ACC_WRAPPER = True
except ImportError:
    _IceAdjustGPU = None
    _RainIceGPU = None
    HAS_ACC_WRAPPER = False

try:
    from _surfex_wrapper_acc import SurfexGPUWrapper, TILE_NATURE
    HAS_SURFEX_GPU = True
except ImportError:
    SurfexGPUWrapper = None
    TILE_NATURE = 1
    HAS_SURFEX_GPU = False

if HAS_JAX:
    from ice3.jax.surfex_jax import SurfexState, SurfexJAXGPU
    from ice3.jax.phyex_jax_gpu import make_rain_ice, RainIceJAXGPU

# Skip everything if no GPU
pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="No NVIDIA GPU / CuPy not installed")

requires_acc = pytest.mark.skipif(
    not (HAS_CUPY and HAS_ACC_WRAPPER),
    reason="_phyex_wrapper_acc not built",
)
requires_jax_gpu = pytest.mark.skipif(
    not (HAS_JAX and HAS_CUPY and HAS_SURFEX_GPU),
    reason="Requires JAX + CuPy + _surfex_wrapper_acc",
)
requires_rain_ice_gpu = pytest.mark.skipif(
    not (HAS_JAX and HAS_CUPY and HAS_ACC_WRAPPER),
    reason="Requires JAX + CuPy + _phyex_wrapper_acc (RainIceGPU)",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_atm(nit: int, nkt: int, dtype=jnp.float32):
    np.random.seed(0)
    z = np.linspace(0, 8000, nkt).astype("f4")
    p0, T0, g = 101325.0, 288.15, 0.0065
    p = p0 * (1 - g * z / T0) ** 5.26
    T = T0 - g * z
    Rd, cpd, p00 = 287.0, 1004.0, 1e5
    exn = (p / p00) ** (Rd / cpd)
    th = T / exn
    rv = 0.012 * np.exp(-z / 2000)
    rho = p / (Rd * T)

    def tile(v):
        return np.tile(v, (nit, 1)).astype("f4")

    j = lambda a: jnp.asarray(a, dtype=dtype)
    return dict(
        sigqsat=j(np.full((nit, nkt), 0.02, "f4")),
        pabs=j(tile(p)), sigs=j(np.full((nit, nkt), 0.1, "f4")),
        th=j(tile(th)), exn=j(tile(exn)), exn_ref=j(tile(exn)),
        rho_dry_ref=j(tile(rho)), rv=j(tile(rv)),
        rc=j(np.zeros((nit, nkt), "f4")), ri=j(np.zeros((nit, nkt), "f4")),
        rr=j(np.zeros((nit, nkt), "f4")), rs=j(np.zeros((nit, nkt), "f4")),
        rg=j(np.zeros((nit, nkt), "f4")), cf_mf=j(np.zeros((nit, nkt), "f4")),
        rc_mf=j(np.zeros((nit, nkt), "f4")), ri_mf=j(np.zeros((nit, nkt), "f4")),
        rvs=j(np.zeros((nit, nkt), "f4")), rcs=j(np.zeros((nit, nkt), "f4")),
        ris=j(np.zeros((nit, nkt), "f4")), ths=j(np.zeros((nit, nkt), "f4")),
    )


def _make_surfex_state(n: int) -> "SurfexState":
    rng = np.random.default_rng(42)
    return SurfexState(
        t_a=jnp.array(rng.uniform(260.0, 310.0, n).astype("f4")),
        q_a=jnp.array(rng.uniform(0.001, 0.020, n).astype("f4")),
        u_a=jnp.array(rng.uniform(-10.0, 10.0, n).astype("f4")),
        v_a=jnp.array(rng.uniform(-10.0, 10.0, n).astype("f4")),
        p_a=jnp.array(rng.uniform(90000.0, 101325.0, n).astype("f4")),
        rhodref=jnp.array(rng.uniform(1.1, 1.3, n).astype("f4")),
        sw_down=jnp.array(rng.uniform(0.0, 800.0, n).astype("f4")),
        lw_down=jnp.array(rng.uniform(200.0, 450.0, n).astype("f4")),
        rain_rate=jnp.zeros(n, dtype=jnp.float32),
        snow_rate=jnp.zeros(n, dtype=jnp.float32),
        psurf_flux_th=jnp.full(n, 0.05, dtype=jnp.float32),
        psurf_flux_rv=jnp.full(n, 5e-4, dtype=jnp.float32),
        psurf_flux_u=jnp.full(n, -0.1, dtype=jnp.float32),
        psurf_flux_v=jnp.full(n, -0.05, dtype=jnp.float32),
        t_skin=jnp.zeros(n, dtype=jnp.float32),
        # Prognostic soil/snow state (ISBA-3L + D95)
        wg1=jnp.full(n, 0.25, dtype=jnp.float32),
        wg2=jnp.full(n, 0.28, dtype=jnp.float32),
        wg3=jnp.full(n, 0.30, dtype=jnp.float32),
        wgi1=jnp.zeros(n, dtype=jnp.float32),
        wgi2=jnp.zeros(n, dtype=jnp.float32),
        tg1=jnp.full(n, 285.0, dtype=jnp.float32),
        tg2=jnp.full(n, 280.0, dtype=jnp.float32),
        wsnow1=jnp.zeros(n, dtype=jnp.float32),
        rho1=jnp.full(n, 300.0, dtype=jnp.float32),
        alb=jnp.full(n, 0.15, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# IceAdjust GPU benchmarks
# ---------------------------------------------------------------------------

class TestIceAdjustGPUBenchmark:

    def _bench_ice_adjust(self, benchmark, nit: int, nkt: int):
        from ice3.jax.phyex_jax_gpu import make_ice_adjust
        bridge = make_ice_adjust(n_cols=nit, n_levs=nkt)
        atm = _make_atm(nit, nkt)

        def _run():
            out = bridge(timestep=60.0, **atm)
            jax.block_until_ready(out)

        # Warm up JIT + CUDA graph capture outside the timer
        for _ in range(3):
            _run()

        benchmark.pedantic(_run, warmup_rounds=3, rounds=10)
        cols_per_s = nit / benchmark.stats["mean"]
        benchmark.extra_info["throughput_cols_per_s"] = cols_per_s

    @requires_acc
    def test_256cols_60lev(self, benchmark):
        self._bench_ice_adjust(benchmark, 256, 60)

    @requires_acc
    def test_1024cols_60lev(self, benchmark):
        self._bench_ice_adjust(benchmark, 1024, 60)

    @requires_acc
    def test_4096cols_60lev(self, benchmark):
        self._bench_ice_adjust(benchmark, 4096, 60)

    @requires_acc
    def test_16384cols_60lev(self, benchmark):
        # 16384 × 60 × 4B × 20 fields ≈ 79 MB — fits in A100 80 GB
        self._bench_ice_adjust(benchmark, 16384, 60)

    @requires_acc
    def test_speedup_vs_cpu_1024cols(self, benchmark):
        """GPU/CPU wall-time ratio; warns if speedup < 2×."""
        from ice3.jax.ice_adjust import IceAdjustJAX
        from ice3.jax.phyex_jax_gpu import make_ice_adjust

        nit, nkt = 1024, 60
        atm = _make_atm(nit, nkt)

        # CPU timing
        cpu_adj = IceAdjustJAX(jit=True)
        for _ in range(2):
            out_cpu = cpu_adj(timestep=60.0, **atm)
            jax.block_until_ready(out_cpu)
        t0 = time.perf_counter()
        for _ in range(5):
            out_cpu = cpu_adj(timestep=60.0, **atm)
            jax.block_until_ready(out_cpu)
        cpu_ms = (time.perf_counter() - t0) / 5 * 1e3

        # GPU timing
        gpu_adj = make_ice_adjust(n_cols=nit, n_levs=nkt)
        for _ in range(3):
            out_gpu = gpu_adj(timestep=60.0, **atm)
            jax.block_until_ready(out_gpu)

        def _run_gpu():
            out = gpu_adj(timestep=60.0, **atm)
            jax.block_until_ready(out)

        benchmark.pedantic(_run_gpu, warmup_rounds=3, rounds=10)
        gpu_ms = benchmark.stats["mean"] * 1e3
        speedup = cpu_ms / gpu_ms

        benchmark.extra_info["cpu_ms"] = cpu_ms
        benchmark.extra_info["gpu_ms"] = gpu_ms
        benchmark.extra_info["speedup"] = speedup

        if speedup < 2.0:
            import warnings
            warnings.warn(
                f"IceAdjust GPU speedup={speedup:.1f}× < 2× expected "
                f"(CPU={cpu_ms:.1f} ms, GPU={gpu_ms:.1f} ms)",
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# SURFEX GPU benchmarks
# ---------------------------------------------------------------------------

class TestSurfexGPUBenchmark:

    def _bench_surfex(self, benchmark, n_cols: int):
        state = _make_surfex_state(n_cols)
        surf = SurfexJAXGPU(n_cols=n_cols)

        def _run():
            fluxes, _next = surf(state, dt=60.0)
            jax.block_until_ready(tuple(fluxes))

        for _ in range(3):
            _run()

        benchmark.pedantic(_run, warmup_rounds=3, rounds=10)
        cols_per_s = n_cols / benchmark.stats["mean"]
        benchmark.extra_info["throughput_cols_per_s"] = cols_per_s

    @requires_jax_gpu
    def test_256cols(self, benchmark):
        self._bench_surfex(benchmark, 256)

    @requires_jax_gpu
    def test_1024cols(self, benchmark):
        self._bench_surfex(benchmark, 1024)

    @requires_jax_gpu
    def test_4096cols(self, benchmark):
        self._bench_surfex(benchmark, 4096)

    @requires_jax_gpu
    def test_16384cols(self, benchmark):
        self._bench_surfex(benchmark, 16384)

    @requires_jax_gpu
    def test_65536cols(self, benchmark):
        # SURFEX is 1D (no vertical dim) → budget ≈ 3 MB at 65 k cols
        self._bench_surfex(benchmark, 65536)

    @requires_jax_gpu
    def test_speedup_vs_cpu_1024cols(self, benchmark):
        """GPU/CPU wall-time ratio for SURFEX; warns if speedup < 2×."""
        from ice3.jax.surfex_jax import SurfexJAX

        n = 1024
        state = _make_surfex_state(n)

        # CPU timing (requires libsurfex_offline — skip gracefully if absent)
        try:
            cpu = SurfexJAX()
            if not cpu._lib._available:
                pytest.skip("libsurfex_offline not available for CPU baseline")
        except Exception:
            pytest.skip("SurfexJAX CPU unavailable")

        for _ in range(2):
            jax.block_until_ready(tuple(cpu(state, dt=60.0)[0]))
        t0 = time.perf_counter()
        for _ in range(5):
            jax.block_until_ready(tuple(cpu(state, dt=60.0)[0]))
        cpu_ms = (time.perf_counter() - t0) / 5 * 1e3

        # GPU timing
        gpu = SurfexJAXGPU(n_cols=n)
        for _ in range(3):
            jax.block_until_ready(tuple(gpu(state, dt=60.0)[0]))

        def _run_gpu():
            fluxes, _next = gpu(state, dt=60.0)
            jax.block_until_ready(tuple(fluxes))

        benchmark.pedantic(_run_gpu, warmup_rounds=3, rounds=10)
        gpu_ms = benchmark.stats["mean"] * 1e3
        speedup = cpu_ms / gpu_ms

        benchmark.extra_info["cpu_ms"] = cpu_ms
        benchmark.extra_info["gpu_ms"] = gpu_ms
        benchmark.extra_info["speedup"] = speedup

        if speedup < 2.0:
            import warnings
            warnings.warn(
                f"SURFEX GPU speedup={speedup:.1f}× < 2× expected "
                f"(CPU={cpu_ms:.1f} ms, GPU={gpu_ms:.1f} ms)",
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Warmup vs steady-state (graph capture vs replay)
# ---------------------------------------------------------------------------

class TestWarmupVsSteadyState:
    """
    Verify that CUDA graph replay is faster than first-call graph capture.
    The first call captures the graph and should be noticeably slower than
    subsequent replay calls.
    """

    @requires_acc
    def test_phyex_capture_vs_replay_1024cols(self):
        from ice3.jax.phyex_jax_gpu import make_ice_adjust
        nit, nkt = 1024, 60
        atm = _make_atm(nit, nkt)
        bridge = make_ice_adjust(n_cols=nit, n_levs=nkt)

        # First call — graph capture
        t0 = time.perf_counter()
        out = bridge(timestep=60.0, **atm)
        jax.block_until_ready(out)
        t_capture = time.perf_counter() - t0

        # Subsequent calls — graph replay
        replay_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            out = bridge(timestep=60.0, **atm)
            jax.block_until_ready(out)
            replay_times.append(time.perf_counter() - t0)
        t_replay = np.mean(replay_times)

        assert t_replay < 0.9 * t_capture, (
            f"Expected replay ({t_replay*1e3:.1f} ms) < 90% of capture "
            f"({t_capture*1e3:.1f} ms)"
        )

    @requires_jax_gpu
    def test_surfex_capture_vs_replay_1024cols(self):
        n = 1024
        state = _make_surfex_state(n)
        surf = SurfexJAXGPU(n_cols=n)

        t0 = time.perf_counter()
        jax.block_until_ready(tuple(surf(state, dt=60.0)[0]))
        t_capture = time.perf_counter() - t0

        replay_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            jax.block_until_ready(tuple(surf(state, dt=60.0)[0]))
            replay_times.append(time.perf_counter() - t0)
        t_replay = np.mean(replay_times)

        assert t_replay < 0.9 * t_capture, (
            f"Expected SURFEX replay ({t_replay*1e3:.1f} ms) < 90% of capture "
            f"({t_capture*1e3:.1f} ms)"
        )

    @requires_rain_ice_gpu
    def test_rain_ice_capture_vs_replay_1024cols(self):
        nit, nkt = 1024, 60
        bridge = make_rain_ice(nit=nit, nkt=nkt)
        state = _make_rain_ice_state(nit, nkt)

        t0 = time.perf_counter()
        out, _ = bridge(state, dt=60.0)
        jax.block_until_ready(out["th_t"])
        t_capture = time.perf_counter() - t0

        replay_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            out, _ = bridge(state, dt=60.0)
            jax.block_until_ready(out["th_t"])
            replay_times.append(time.perf_counter() - t0)
        t_replay = np.mean(replay_times)

        assert t_replay < 0.9 * t_capture, (
            f"Expected RainIce replay ({t_replay*1e3:.1f} ms) < 90% of capture "
            f"({t_capture*1e3:.1f} ms)"
        )


# ---------------------------------------------------------------------------
# RainIce GPU benchmarks
# ---------------------------------------------------------------------------

def _make_rain_ice_state(nit: int, nkt: int) -> dict:
    """Realistic microphysics state for RainIce GPU benchmark."""
    rng = np.random.default_rng(7)
    z = np.linspace(0, 10000, nkt, dtype="f4")
    p0, T0, g = 101325.0, 288.15, 0.0065
    Rd, cp, p00 = 287.0, 1004.0, 1e5
    p = p0 * (1 - g * z / T0) ** 5.26
    T = T0 - g * z
    exn = (p / p00) ** (Rd / cp)
    th = T / exn
    rv = 0.012 * np.exp(-z / 2000)

    def tile(v):
        return jnp.array(np.tile(v, (nit, 1)).astype("f4"))

    cldfr = jnp.zeros((nit, nkt), dtype=jnp.float32)
    # Small cloud water in mid-troposphere
    rc = jnp.where(
        jnp.broadcast_to(jnp.array((z > 2000) & (z < 6000)), (nit, nkt)),
        jnp.full((nit, nkt), 2e-4, dtype=jnp.float32),
        jnp.zeros((nit, nkt), dtype=jnp.float32),
    )

    return dict(
        exn=tile(exn), dzz=tile(np.diff(z, prepend=z[0])),
        rhodref=tile(p / (Rd * T)), pres=tile(p),
        th_t=tile(th), rv_t=tile(rv),
        rc_t=rc, rr_t=jnp.zeros((nit, nkt), dtype=jnp.float32),
        ri_t=jnp.zeros((nit, nkt), dtype=jnp.float32),
        rs_t=jnp.zeros((nit, nkt), dtype=jnp.float32),
        rg_t=jnp.zeros((nit, nkt), dtype=jnp.float32),
        ci_t=cldfr, cldfr=cldfr, icldfr=cldfr,
    )


class TestRainIceGPUBenchmark:
    """GPU latency benchmarks for RAIN_ICE OpenACC kernel."""

    def _bench(self, benchmark, nit: int, nkt: int):
        bridge = make_rain_ice(nit=nit, nkt=nkt)
        state = _make_rain_ice_state(nit, nkt)

        def _run():
            out, _ = bridge(state, dt=60.0)
            jax.block_until_ready(out["th_t"])

        for _ in range(3):
            _run()

        benchmark.pedantic(_run, warmup_rounds=3, rounds=10)
        benchmark.extra_info["throughput_cols_per_s"] = (
            nit * nkt / benchmark.stats["mean"]
        )

    @requires_rain_ice_gpu
    def test_256cols_60lev(self, benchmark):
        self._bench(benchmark, 256, 60)

    @requires_rain_ice_gpu
    def test_1024cols_60lev(self, benchmark):
        self._bench(benchmark, 1024, 60)

    @requires_rain_ice_gpu
    def test_4096cols_60lev(self, benchmark):
        self._bench(benchmark, 4096, 60)

    @requires_rain_ice_gpu
    def test_16384cols_60lev(self, benchmark):
        self._bench(benchmark, 16384, 60)

    @requires_rain_ice_gpu
    def test_speedup_vs_jax_cpu_1024cols(self, benchmark):
        """GPU/CPU wall-time ratio for RainIce; warns if speedup < 2×."""
        from ice3.jax.rain_ice import RainIceJAX
        nit, nkt = 1024, 60
        state = _make_rain_ice_state(nit, nkt)

        # CPU timing — pure JAX (no constants needed for timing; use defaults)
        try:
            from ice3.phyex_common.phyex import Phyex
            _ph = Phyex("AROME", TSTEP=60.0)
            cpu = RainIceJAX(constants=_ph.to_dict())
        except Exception:
            pytest.skip("RainIceJAX constants unavailable for CPU baseline")

        jit_cpu = jax.jit(lambda s: cpu(s, dt=60.0))
        for _ in range(2):
            out_cpu, _ = jit_cpu(state)
            jax.block_until_ready(out_cpu["th_t"])
        t0 = time.perf_counter()
        for _ in range(5):
            out_cpu, _ = jit_cpu(state)
            jax.block_until_ready(out_cpu["th_t"])
        cpu_ms = (time.perf_counter() - t0) / 5 * 1e3

        # GPU timing
        gpu = make_rain_ice(nit=nit, nkt=nkt)
        for _ in range(3):
            out_gpu, _ = gpu(state, dt=60.0)
            jax.block_until_ready(out_gpu["th_t"])

        def _run_gpu():
            out, _ = gpu(state, dt=60.0)
            jax.block_until_ready(out["th_t"])

        benchmark.pedantic(_run_gpu, warmup_rounds=3, rounds=10)
        gpu_ms = benchmark.stats["mean"] * 1e3
        speedup = cpu_ms / gpu_ms

        benchmark.extra_info["cpu_ms"] = cpu_ms
        benchmark.extra_info["gpu_ms"] = gpu_ms
        benchmark.extra_info["speedup"] = speedup

        if speedup < 2.0:
            import warnings
            warnings.warn(
                f"RainIce GPU speedup={speedup:.1f}× < 2× expected "
                f"(CPU={cpu_ms:.1f} ms, GPU={gpu_ms:.1f} ms)",
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# make_arome_physics smoke test
# ---------------------------------------------------------------------------

class TestAromePhysicsSmoke:
    """
    Smoke test for make_arome_physics() factory on GPU.
    Verifies that one orchestrator step runs without error and produces
    finite hydrometeor outputs.
    """

    @requires_acc
    def test_single_step_finite(self):
        """make_arome_physics() step produces finite state on GPU."""
        import jax.numpy as jnp
        from ice3.jax.arome_physics import make_arome_physics, AromeState

        nit, nkt = 256, 40

        # Minimal constants dict (RainIceJAX fallback will use these)
        try:
            from ice3.phyex_common.phyex import Phyex
            _ph = Phyex("AROME", TSTEP=60.0)
            constants = _ph.to_dict()
        except Exception:
            pytest.skip("Phyex constants unavailable")

        orch = make_arome_physics(constants=constants, nit=nit, nkt=nkt)

        z = np.linspace(0, 8000, nkt).astype("f4")
        p0, T0, g = 101325.0, 288.15, 0.0065
        Rd, cp_, p00 = 287.0, 1004.0, 1e5
        p = p0 * (1 - g * z / T0) ** 5.26
        T = T0 - g * z
        exn = (p / p00) ** (Rd / cp_)
        th = T / exn
        rv = 0.012 * np.exp(-z / 2000)
        rho = p / (Rd * T)

        def _tile(v):
            return jnp.array(np.tile(v, (nit, 1)).astype("f4"))

        _f = jnp.float32
        state = AromeState(
            pabst=_tile(p), pzz=_tile(z),
            dzz=_tile(np.diff(z, prepend=z[0])),
            pt=_tile(T), pth=_tile(th), pthl=_tile(th),
            prv=_tile(rv),
            prc=jnp.zeros((nit, nkt), dtype=_f),
            pri=jnp.zeros((nit, nkt), dtype=_f),
            prr=jnp.zeros((nit, nkt), dtype=_f),
            prs=jnp.zeros((nit, nkt), dtype=_f),
            prg=jnp.zeros((nit, nkt), dtype=_f),
            prt=_tile(rv),
            pu=jnp.zeros((nit, nkt), dtype=_f),
            pv=jnp.zeros((nit, nkt), dtype=_f),
            pw=jnp.zeros((nit, nkt), dtype=_f),
            ptke=jnp.full((nit, nkt), 0.1, dtype=_f),
            ptkecls=jnp.full((nit,), 0.1, dtype=_f),
            psurf_flux_u=jnp.full((nit,), -0.1, dtype=_f),
            psurf_flux_v=jnp.full((nit,), -0.05, dtype=_f),
            psurf_flux_th=jnp.full((nit,), 0.05, dtype=_f),
            psurf_flux_rv=jnp.full((nit,), 5e-4, dtype=_f),
            pthvref=_tile(th),
            pexn=_tile(exn), pexn_ref=_tile(exn),
            prho_dry_ref=_tile(rho),
            # SURFEX prognostic initial state
            wg1=jnp.full((nit,), 0.25, dtype=_f),
            wg2=jnp.full((nit,), 0.28, dtype=_f),
            wg3=jnp.full((nit,), 0.30, dtype=_f),
            wgi1=jnp.zeros((nit,), dtype=_f),
            wgi2=jnp.zeros((nit,), dtype=_f),
            tg1=jnp.full((nit,), 285.0, dtype=_f),
            tg2=jnp.full((nit,), 280.0, dtype=_f),
            wsnow1=jnp.zeros((nit,), dtype=_f),
            rho1=jnp.full((nit,), 300.0, dtype=_f),
            alb=jnp.full((nit,), 0.15, dtype=_f),
        )

        out_state, diag = orch.step(state, dt=60.0)
        jax.block_until_ready(out_state.prv)

        assert jnp.all(jnp.isfinite(out_state.pt)),  "Temperature not finite"
        assert jnp.all(jnp.isfinite(out_state.prv)), "Water vapor not finite"
        assert jnp.all(jnp.isfinite(out_state.prc)), "Cloud water not finite"
        assert jnp.all(jnp.isfinite(out_state.pri)), "Cloud ice not finite"
        assert jnp.all(jnp.isfinite(out_state.wg1)), "Soil moisture not finite"
