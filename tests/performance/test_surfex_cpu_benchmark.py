"""
CPU performance benchmarks for SURFEX using SurfexJAX (libsurfex_offline).

These benchmarks establish an honest CPU reference latency using the real
Fortran SURFEX physics (via jax.pure_callback → libsurfex_offline).  They
are submitted to Bencher with testbed ``ubuntu-latest`` and are compared
directly against the GPU series produced by test_gpu_benchmark.py.

Benchmark naming convention:
  surfex/cpu/<N>cols   — latency for N surface columns

Skip behaviour:
  All tests skip if libsurfex_offline is not compiled (ImportError or lib
  unavailable).  This is by design: the honest CPU reference requires the
  compiled library, which is present on RunPod / dev machines but not on the
  vanilla GitHub Actions runner.

Requirements:
  - libsurfex_offline.so compiled and on LD_LIBRARY_PATH
  - cffi installed
  - pytest-benchmark
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional JAX imports
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False

# ---------------------------------------------------------------------------
# SurfexJAX (CPU) — requires libsurfex_offline
# ---------------------------------------------------------------------------
HAS_SURFEX_CPU = False
SurfexJAX = None
SurfexState = None

if HAS_JAX:
    try:
        from ice3.jax.surfex_jax import SurfexJAX, SurfexState
        _probe = SurfexJAX()
        HAS_SURFEX_CPU = _probe._lib._available
    except Exception:
        pass

requires_surfex_cpu = pytest.mark.skipif(
    not HAS_SURFEX_CPU,
    reason="libsurfex_offline not available (compile with ENABLE_SURFEX=ON)",
)


# ---------------------------------------------------------------------------
# Shared fixture: physically meaningful SurfexState
# ---------------------------------------------------------------------------

def _make_state(n: int) -> "SurfexState":
    """Build a realistic SurfexState with prognostic soil/snow fields."""
    rng = np.random.default_rng(0)
    return SurfexState(
        t_a=jnp.array(rng.uniform(265.0, 308.0, n).astype("f4")),
        q_a=jnp.array(rng.uniform(0.002, 0.018, n).astype("f4")),
        u_a=jnp.array(rng.uniform(-8.0, 8.0, n).astype("f4")),
        v_a=jnp.array(rng.uniform(-8.0, 8.0, n).astype("f4")),
        p_a=jnp.array(rng.uniform(90000.0, 101325.0, n).astype("f4")),
        rhodref=jnp.array(rng.uniform(1.1, 1.3, n).astype("f4")),
        sw_down=jnp.array(rng.uniform(0.0, 750.0, n).astype("f4")),
        lw_down=jnp.array(rng.uniform(220.0, 430.0, n).astype("f4")),
        rain_rate=jnp.zeros(n, dtype=jnp.float32),
        snow_rate=jnp.zeros(n, dtype=jnp.float32),
        psurf_flux_th=jnp.full(n, 0.05, dtype=jnp.float32),
        psurf_flux_rv=jnp.full(n, 5e-4, dtype=jnp.float32),
        psurf_flux_u=jnp.full(n, -0.1, dtype=jnp.float32),
        psurf_flux_v=jnp.full(n, -0.05, dtype=jnp.float32),
        t_skin=jnp.zeros(n, dtype=jnp.float32),
        # Prognostic soil state — loamy soil at field capacity
        wg1=jnp.full(n, 0.25, dtype=jnp.float32),
        wg2=jnp.full(n, 0.28, dtype=jnp.float32),
        wg3=jnp.full(n, 0.30, dtype=jnp.float32),
        wgi1=jnp.zeros(n, dtype=jnp.float32),
        wgi2=jnp.zeros(n, dtype=jnp.float32),
        tg1=jnp.full(n, 285.0, dtype=jnp.float32),
        tg2=jnp.full(n, 280.0, dtype=jnp.float32),
        # Prognostic snow state — bare ground
        wsnow1=jnp.zeros(n, dtype=jnp.float32),
        rho1=jnp.full(n, 300.0, dtype=jnp.float32),
        alb=jnp.full(n, 0.15, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# CPU latency benchmarks — one test per column count
# ---------------------------------------------------------------------------

class TestSurfexCPUBenchmark:
    """
    Honest CPU reference benchmarks for SURFEX using libsurfex_offline.

    All tests are tagged ``surfex/cpu/<N>cols`` so Bencher can plot them on
    the same chart as the GPU series produced by TestSurfexGPUBenchmark.
    """

    def _bench(self, benchmark, n_cols: int):
        surf = SurfexJAX()
        state = _make_state(n_cols)

        # JIT-warm the pure_callback path (2 calls)
        jit_surf = jax.jit(lambda st: surf(st, dt=60.0))
        for _ in range(2):
            fluxes, _ = jit_surf(state)
            jax.block_until_ready(tuple(fluxes))

        def _run():
            fluxes, _ = jit_surf(state)
            jax.block_until_ready(tuple(fluxes))

        benchmark.pedantic(_run, warmup_rounds=2, rounds=10)
        benchmark.extra_info["throughput_cols_per_s"] = (
            n_cols / benchmark.stats["mean"]
        )

    @requires_surfex_cpu
    def test_256cols(self, benchmark):
        """surfex/cpu/256cols — latency for 256 surface columns."""
        self._bench(benchmark, 256)

    @requires_surfex_cpu
    def test_1024cols(self, benchmark):
        """surfex/cpu/1024cols — latency for 1024 surface columns."""
        self._bench(benchmark, 1024)

    @requires_surfex_cpu
    def test_4096cols(self, benchmark):
        """surfex/cpu/4096cols — latency for 4096 surface columns."""
        self._bench(benchmark, 4096)

    @requires_surfex_cpu
    def test_16384cols(self, benchmark):
        """surfex/cpu/16384cols — latency for 16384 surface columns."""
        self._bench(benchmark, 16384)


# ---------------------------------------------------------------------------
# Multi-step throughput test (10 consecutive steps, state evolves)
# ---------------------------------------------------------------------------

class TestSurfexCPUMultiStep:
    """
    Measures wall-time for 10 consecutive prognostic time steps where the
    state output of each step feeds the next.  This is the most realistic
    operational mode and exposes any overhead in SurfexState construction.
    """

    @requires_surfex_cpu
    def test_10steps_1024cols(self, benchmark):
        """surfex/cpu/10steps_1024cols — 10-step prognostic integration."""
        n = 1024
        surf = SurfexJAX()
        jit_surf = jax.jit(lambda st: surf(st, dt=60.0))
        state = _make_state(n)

        # Warm up
        for _ in range(2):
            fluxes, state = jit_surf(state)
            jax.block_until_ready(tuple(fluxes))

        def _run_10_steps():
            s = state
            for _ in range(10):
                fluxes, s = jit_surf(s)
            jax.block_until_ready(tuple(fluxes))

        benchmark.pedantic(_run_10_steps, warmup_rounds=2, rounds=5)
        step_ms = benchmark.stats["mean"] / 10 * 1e3
        benchmark.extra_info["mean_step_ms"] = step_ms


# ---------------------------------------------------------------------------
# CPU/GPU speedup reference (no benchmark fixture, manual timing)
# ---------------------------------------------------------------------------

class TestSurfexCPUReference:
    """
    Emits CPU timing metadata used by the GPU speedup test.
    Runs independently of pytest-benchmark so it produces values even when
    the benchmark plugin is disabled.
    """

    @requires_surfex_cpu
    def test_reference_timing_1024cols(self):
        """
        Manually times 10 JIT-compiled calls and prints mean latency.
        This is not a benchmark (no ``benchmark`` fixture) but ensures the
        CPU path is exercised and its latency is visible in CI logs.
        """
        n = 1024
        surf = SurfexJAX()
        jit_surf = jax.jit(lambda st: surf(st, dt=60.0))
        state = _make_state(n)

        # Warm up
        for _ in range(3):
            fluxes, state = jit_surf(state)
            jax.block_until_ready(tuple(fluxes))

        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            fluxes, state = jit_surf(state)
            jax.block_until_ready(tuple(fluxes))
            times.append(time.perf_counter() - t0)

        mean_ms = np.mean(times) * 1e3
        std_ms = np.std(times) * 1e3
        cols_per_s = n / np.mean(times)
        print(
            f"\n[SURFEX CPU] 1024 cols  "
            f"mean={mean_ms:.2f} ms  std={std_ms:.2f} ms  "
            f"throughput={cols_per_s/1e3:.1f} k cols/s"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only", "--benchmark-group-by=group"])
