# -*- coding: utf-8 -*-
"""Performance tests for GPU-accelerated Fortran RAIN_ICE (OpenACC)."""
import numpy as np
import pytest
import sys
from pathlib import Path
import time

# Try to import CuPy
try:
    import cupy as cp
    HAS_CUPY = True
    HAS_GPU = cp.cuda.is_available()
except ImportError:
    cp = None
    HAS_CUPY = False
    HAS_GPU = False

# Add build directory to path
build_dir = Path(__file__).parent.parent.parent / 'build-gpu'
if not build_dir.exists():
    build_dir = Path(__file__).parent.parent.parent / 'build'

if build_dir.exists():
    for sub in build_dir.iterdir():
        if sub.is_dir() and sub.name.startswith('cp'):
            sys.path.insert(0, str(sub))
            break

# Try to import GPU wrapper
try:
    from ice3._phyex_wrapper_acc import RainIceGPU
    HAS_GPU_WRAPPER = True
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "PHYEX-IAL_CY50T1/bridge"))
    try:
        from _phyex_wrapper_acc import RainIceGPU
        HAS_GPU_WRAPPER = True
    except ImportError:
        RainIceGPU = None
        HAS_GPU_WRAPPER = False

# Try to import CPU reference
try:
    from ice3._phyex_wrapper import rain_ice as rain_ice_cpu, init_rain_ice
    HAS_CPU_WRAPPER = True
except ImportError:
    try:
        from _phyex_wrapper import rain_ice as rain_ice_cpu, init_rain_ice
        HAS_CPU_WRAPPER = True
    except ImportError:
        rain_ice_cpu = None
        init_rain_ice = None
        HAS_CPU_WRAPPER = False


def create_test_atmosphere_rain_ice_gpu(nijt=100, nkt=60):
    """
    Create realistic atmospheric test data for RAIN_ICE on GPU.

    Parameters
    ----------
    nijt : int
        Number of horizontal points
    nkt : int
        Number of vertical levels

    Returns
    -------
    dict
        Dictionary with all required fields as CuPy arrays (float32)
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy required for GPU tests")

    # Create on CPU first
    z = np.linspace(0, 10000, nkt, dtype=np.float32)

    # Standard atmosphere
    p0 = 101325.0
    T0 = 288.15
    gamma = 0.0065

    # Physical constants
    Rd = 287.0
    cp = 1004.0
    p00 = 100000.0

    data_cpu = {}

    # Pressure profile
    pressure = p0 * (1 - gamma * z / T0) ** 5.26
    data_cpu['ppabst'] = np.tile(pressure, (nijt, 1)).astype(np.float32)

    # Temperature profile
    temperature = T0 - gamma * z
    data_cpu['temperature'] = np.tile(temperature, (nijt, 1)).astype(np.float32)

    # Add variability
    np.random.seed(42)
    data_cpu['temperature'] += np.random.randn(nijt, nkt).astype(np.float32) * 0.5
    data_cpu['ppabst'] += np.random.randn(nijt, nkt).astype(np.float32) * 100

    # Exner function
    data_cpu['pexn'] = (data_cpu['ppabst'] / p00) ** (Rd / cp)
    data_cpu['ptht'] = data_cpu['temperature'] / data_cpu['pexn']

    # Reference values
    data_cpu['pexnref'] = data_cpu['pexn'].copy()
    data_cpu['prhodref'] = data_cpu['ppabst'] / (Rd * data_cpu['temperature'])
    data_cpu['prhodj'] = data_cpu['prhodref'].copy()

    # Layer thickness
    data_cpu['pdzz'] = np.full((nijt, nkt), 100.0, dtype=np.float32)

    # Water vapor
    rv_surf = 0.015
    rv_profile = rv_surf * np.exp(-z / 2000)
    data_cpu['prvt'] = np.tile(rv_profile, (nijt, 1)).astype(np.float32)
    data_cpu['prvt'] += np.abs(np.random.randn(nijt, nkt).astype(np.float32)) * 0.002

    # Cloud water
    data_cpu['prct'] = np.zeros((nijt, nkt), dtype=np.float32)
    cloud_levels = (z > 2000) & (z < 6000)
    data_cpu['prct'][:, cloud_levels] = np.abs(np.random.rand(nijt, cloud_levels.sum()).astype(np.float32)) * 0.003

    # Rain
    data_cpu['prrt'] = np.zeros((nijt, nkt), dtype=np.float32)
    rain_levels = (z > 1000) & (z < 4000)
    data_cpu['prrt'][:, rain_levels] = np.abs(np.random.rand(nijt, rain_levels.sum()).astype(np.float32)) * 0.001

    # Ice
    data_cpu['prit'] = np.zeros((nijt, nkt), dtype=np.float32)
    ice_levels = z > 5000
    data_cpu['prit'][:, ice_levels] = np.abs(np.random.rand(nijt, ice_levels.sum()).astype(np.float32)) * 0.002

    # Snow
    data_cpu['prst'] = np.zeros((nijt, nkt), dtype=np.float32)
    snow_levels = (z > 3000) & (z < 7000)
    data_cpu['prst'][:, snow_levels] = np.abs(np.random.rand(nijt, snow_levels.sum()).astype(np.float32)) * 0.001

    # Graupel
    data_cpu['prgt'] = np.zeros((nijt, nkt), dtype=np.float32)
    graupel_levels = (z > 2000) & (z < 6000)
    data_cpu['prgt'][:, graupel_levels] = np.abs(np.random.rand(nijt, graupel_levels.sum()).astype(np.float32)) * 0.0005

    # Ice concentration
    data_cpu['pcit'] = np.where(data_cpu['prit'] > 0, 1000.0, 0.0).astype(np.float32)

    # Tendencies
    data_cpu['pths'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['prvs'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['prcs'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['prrs'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['pris'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['prss'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['prgs'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['pcis'] = np.zeros((nijt, nkt), dtype=np.float32)

    # Turbulence
    data_cpu['psigs'] = np.full((nijt, nkt), 0.1, dtype=np.float32)

    # Transfer to GPU
    data_gpu = {}
    for key, value in data_cpu.items():
        data_gpu[key] = cp.asarray(np.ascontiguousarray(value), dtype=cp.float32)

    return data_gpu


@pytest.fixture(scope="module")
def init_rain_ice_gpu():
    """Initialize RAIN_ICE for GPU tests."""
    if HAS_GPU_WRAPPER and RainIceGPU is not None:
        # GPU wrapper handles initialization internally
        pass
    return True


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
@pytest.mark.skipif(not HAS_GPU, reason="No CUDA GPU available")
@pytest.mark.skipif(not HAS_GPU_WRAPPER, reason="GPU wrapper not built")
class TestRainIceGPUPerformanceSmall:
    """Performance tests on small GPU domain."""

    def test_small_domain_gpu(self, benchmark, init_rain_ice_gpu):
        """Benchmark small domain on GPU."""
        print("\n" + "="*75)
        print("GPU RAIN_ICE Performance: Small Domain (100x60)")
        print("="*75)

        nijt, nkt = 100, 60
        total_points = nijt * nkt

        print(f"Horizontal points: {nijt:,}")
        print(f"Vertical levels: {nkt}")
        print(f"Total grid points: {total_points:,}")

        data = create_test_atmosphere_rain_ice_gpu(nijt, nkt)
        rain_ice_gpu = RainIceGPU(krr=6, timestep=50.0, dzmin=60.0, scheme="ICE3")

        # Warm-up
        rain_ice_gpu(
            data['pexn'], data['prhodref'], data['ppabst'], data['pdzz'],
            data['ptht'], data['prvt'], data['prct'], data['prrt'],
            data['prit'], data['prst'], data['prgt'], data['pcit'],
            data['pths'], data['prvs'], data['prcs'], data['prrs'],
            data['pris'], data['prss'], data['prgs'], data['pcis'],
            data['psigs']
        )
        cp.cuda.Stream.null.synchronize()

        def run_rain_ice_gpu():
            rain_ice_gpu(
                data['pexn'], data['prhodref'], data['ppabst'], data['pdzz'],
                data['ptht'], data['prvt'], data['prct'], data['prrt'],
                data['prit'], data['prst'], data['prgt'], data['pcit'],
                data['pths'], data['prvs'], data['prcs'], data['prrs'],
                data['pris'], data['prss'], data['prgs'], data['pcis'],
                data['psigs']
            )
            cp.cuda.Stream.null.synchronize()

        result = benchmark(run_rain_ice_gpu)

        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")
        print(f"  Performance: {throughput*50.0/1e6:.2f} M point-steps/s")


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
@pytest.mark.skipif(not HAS_GPU, reason="No CUDA GPU available")
@pytest.mark.skipif(not HAS_GPU_WRAPPER, reason="GPU wrapper not built")
class TestRainIceGPUPerformanceMedium:
    """Performance tests on medium GPU domain."""

    def test_medium_domain_gpu(self, benchmark, init_rain_ice_gpu):
        """Benchmark medium domain on GPU."""
        print("\n" + "="*75)
        print("GPU RAIN_ICE Performance: Medium Domain (2500x60)")
        print("="*75)

        nijt, nkt = 2500, 60
        total_points = nijt * nkt

        print(f"Horizontal points: {nijt:,}")
        print(f"Vertical levels: {nkt}")
        print(f"Total grid points: {total_points:,}")

        data = create_test_atmosphere_rain_ice_gpu(nijt, nkt)
        rain_ice_gpu = RainIceGPU(krr=6, timestep=50.0, dzmin=60.0, scheme="ICE3")

        # Warm-up
        rain_ice_gpu(
            data['pexn'], data['prhodref'], data['ppabst'], data['pdzz'],
            data['ptht'], data['prvt'], data['prct'], data['prrt'],
            data['prit'], data['prst'], data['prgt'], data['pcit'],
            data['pths'], data['prvs'], data['prcs'], data['prrs'],
            data['pris'], data['prss'], data['prgs'], data['pcis'],
            data['psigs']
        )
        cp.cuda.Stream.null.synchronize()

        def run_rain_ice_gpu():
            rain_ice_gpu(
                data['pexn'], data['prhodref'], data['ppabst'], data['pdzz'],
                data['ptht'], data['prvt'], data['prct'], data['prrt'],
                data['prit'], data['prst'], data['prgt'], data['pcit'],
                data['pths'], data['prvs'], data['prcs'], data['prrs'],
                data['pris'], data['prss'], data['prgs'], data['pcis'],
                data['psigs']
            )
            cp.cuda.Stream.null.synchronize()

        result = benchmark(run_rain_ice_gpu)

        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")
        print(f"  Performance: {throughput*50.0/1e6:.2f} M point-steps/s")


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
@pytest.mark.skipif(not HAS_GPU, reason="No CUDA GPU available")
@pytest.mark.skipif(not HAS_GPU_WRAPPER, reason="GPU wrapper not built")
class TestRainIceGPUPerformanceLarge:
    """Performance tests on large GPU domain."""

    def test_large_domain_gpu(self, benchmark, init_rain_ice_gpu):
        """Benchmark large domain on GPU."""
        print("\n" + "="*75)
        print("GPU RAIN_ICE Performance: Large Domain (10000x60)")
        print("="*75)

        nijt, nkt = 10000, 60
        total_points = nijt * nkt

        print(f"Horizontal points: {nijt:,}")
        print(f"Vertical levels: {nkt}")
        print(f"Total grid points: {total_points:,}")

        data = create_test_atmosphere_rain_ice_gpu(nijt, nkt)
        rain_ice_gpu = RainIceGPU(krr=6, timestep=50.0, dzmin=60.0, scheme="ICE3")

        # Warm-up
        rain_ice_gpu(
            data['pexn'], data['prhodref'], data['ppabst'], data['pdzz'],
            data['ptht'], data['prvt'], data['prct'], data['prrt'],
            data['prit'], data['prst'], data['prgt'], data['pcit'],
            data['pths'], data['prvs'], data['prcs'], data['prrs'],
            data['pris'], data['prss'], data['prgs'], data['pcis'],
            data['psigs']
        )
        cp.cuda.Stream.null.synchronize()

        def run_rain_ice_gpu():
            rain_ice_gpu(
                data['pexn'], data['prhodref'], data['ppabst'], data['pdzz'],
                data['ptht'], data['prvt'], data['prct'], data['prrt'],
                data['prit'], data['prst'], data['prgt'], data['pcit'],
                data['pths'], data['prvs'], data['prcs'], data['prrs'],
                data['pris'], data['prss'], data['prgs'], data['pcis'],
                data['psigs']
            )
            cp.cuda.Stream.null.synchronize()

        result = benchmark(run_rain_ice_gpu)

        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")
        print(f"  Performance: {throughput*50.0/1e6:.2f} M point-steps/s")

        if hasattr(result.stats, 'stddev'):
            print(f"  Std dev: {result.stats['stddev']*1000:.3f} ms")
        if hasattr(result.stats, 'min'):
            print(f"  Min time: {result.stats['min']*1000:.3f} ms")
        if hasattr(result.stats, 'max'):
            print(f"  Max time: {result.stats['max']*1000:.3f} ms")


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
@pytest.mark.skipif(not HAS_GPU, reason="No CUDA GPU available")
@pytest.mark.skipif(not HAS_GPU_WRAPPER, reason="GPU wrapper not built")
@pytest.mark.skipif(not HAS_CPU_WRAPPER, reason="CPU wrapper needed for comparison")
class TestRainIceGPUvsCPU:
    """Compare GPU and CPU performance."""

    def test_gpu_vs_cpu_speedup(self):
        """Measure GPU speedup over CPU."""
        print("\n" + "="*75)
        print("GPU RAIN_ICE: GPU vs CPU Speedup Analysis")
        print("="*75)

        if init_rain_ice is not None:
            init_rain_ice(timestep=50.0, dzmin=60.0, krr=6, scheme="ICE3")

        sizes = [
            (100, 60),
            (500, 60),
            (1000, 60),
            (2500, 60),
            (5000, 60),
            (10000, 60),
        ]

        print(f"\nGPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        print(f"CUDA Version: {cp.cuda.runtime.runtimeGetVersion()}")

        results = []

        for nijt, nkt in sizes:
            total_points = nijt * nkt

            # Create data on GPU
            data_gpu = create_test_atmosphere_rain_ice_gpu(nijt, nkt)

            # Copy to CPU (Fortran order, transposed)
            data_cpu = {key: np.asfortranarray(cp.asnumpy(val).T) for key, val in data_gpu.items()}

            # GPU benchmark
            rain_ice_gpu = RainIceGPU(krr=6, timestep=50.0, dzmin=60.0, scheme="ICE3")

            # Warm-up
            rain_ice_gpu(
                data_gpu['pexn'], data_gpu['prhodref'], data_gpu['ppabst'], data_gpu['pdzz'],
                data_gpu['ptht'], data_gpu['prvt'], data_gpu['prct'], data_gpu['prrt'],
                data_gpu['prit'], data_gpu['prst'], data_gpu['prgt'], data_gpu['pcit'],
                data_gpu['pths'], data_gpu['prvs'], data_gpu['prcs'], data_gpu['prrs'],
                data_gpu['pris'], data_gpu['prss'], data_gpu['prgs'], data_gpu['pcis'],
                data_gpu['psigs']
            )
            cp.cuda.Stream.null.synchronize()

            # Timed runs (GPU)
            n_runs = 20
            t0 = time.time()
            for _ in range(n_runs):
                rain_ice_gpu(
                    data_gpu['pexn'], data_gpu['prhodref'], data_gpu['ppabst'], data_gpu['pdzz'],
                    data_gpu['ptht'], data_gpu['prvt'], data_gpu['prct'], data_gpu['prrt'],
                    data_gpu['prit'], data_gpu['prst'], data_gpu['prgt'], data_gpu['pcit'],
                    data_gpu['pths'], data_gpu['prvs'], data_gpu['prcs'], data_gpu['prrs'],
                    data_gpu['pris'], data_gpu['prss'], data_gpu['prgs'], data_gpu['pcis'],
                    data_gpu['psigs']
                )
            cp.cuda.Stream.null.synchronize()
            gpu_time = (time.time() - t0) / n_runs

            # CPU benchmark
            t0 = time.time()
            for _ in range(n_runs):
                rain_ice_cpu(
                    timestep=np.float32(50.0),
                    krr=6,
                    pexn=data_cpu['pexn'],
                    rhodref=data_cpu['prhodref'],
                    pres=data_cpu['ppabst'],
                    dzz=data_cpu['pdzz'],
                    th_t=data_cpu['ptht'],
                    rv_t=data_cpu['prvt'],
                    rc_t=data_cpu['prct'],
                    rr_t=data_cpu['prrt'],
                    ri_t=data_cpu['prit'],
                    rs_t=data_cpu['prst'],
                    rg_t=data_cpu['prgt'],
                    ci_t=data_cpu['pcit'],
                    ths=data_cpu['pths'],
                    rvs=data_cpu['prvs'],
                    rcs=data_cpu['prcs'],
                    rrs=data_cpu['prrs'],
                    ris=data_cpu['pris'],
                    rss=data_cpu['prss'],
                    rgs=data_cpu['prgs'],
                    cis=data_cpu['pcis'],
                    sigs=data_cpu['psigs']
                )
            cpu_time = (time.time() - t0) / n_runs

            speedup = cpu_time / gpu_time
            cpu_throughput = total_points / cpu_time
            gpu_throughput = total_points / gpu_time

            results.append((total_points, cpu_time, gpu_time, speedup))

            print(f"\n{nijt:6d} × {nkt:3d} ({total_points:,} points):")
            print(f"  CPU: {cpu_time*1000:7.2f} ms ({cpu_throughput/1e6:6.2f} M points/s)")
            print(f"  GPU: {gpu_time*1000:7.2f} ms ({gpu_throughput/1e6:6.2f} M points/s)")
            print(f"  Speedup: {speedup:6.2f}×")

        # Summary
        print("\n" + "-"*75)
        print("Speedup Summary:")
        print("-"*75)
        for points, cpu_t, gpu_t, speedup in results:
            print(f"  {points:>10,} points: {speedup:6.2f}× speedup")

        # Best speedup
        best_idx = np.argmax([r[3] for r in results])
        best_points, _, _, best_speedup = results[best_idx]
        print(f"\nBest speedup: {best_speedup:.2f}× at {best_points:,} points")


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
@pytest.mark.skipif(not HAS_GPU, reason="No CUDA GPU available")
@pytest.mark.skipif(not HAS_GPU_WRAPPER, reason="GPU wrapper not built")
class TestRainIceGPUScaling:
    """Test GPU scaling characteristics."""

    def test_gpu_scaling_analysis(self, init_rain_ice_gpu):
        """Analyze how GPU performance scales with domain size."""
        print("\n" + "="*75)
        print("GPU RAIN_ICE: Scaling Analysis")
        print("="*75)

        sizes = [
            (100, 20),
            (500, 40),
            (1000, 60),
            (2500, 60),
            (5000, 60),
        ]

        results = []

        for nijt, nkt in sizes:
            total_points = nijt * nkt
            data = create_test_atmosphere_rain_ice_gpu(nijt, nkt)
            rain_ice_gpu = RainIceGPU(krr=6, timestep=50.0, dzmin=60.0, scheme="ICE3")

            # Warm-up
            rain_ice_gpu(
                data['pexn'], data['prhodref'], data['ppabst'], data['pdzz'],
                data['ptht'], data['prvt'], data['prct'], data['prrt'],
                data['prit'], data['prst'], data['prgt'], data['pcit'],
                data['pths'], data['prvs'], data['prcs'], data['prrs'],
                data['pris'], data['prss'], data['prgs'], data['pcis'],
                data['psigs']
            )
            cp.cuda.Stream.null.synchronize()

            # Time execution
            n_runs = 10
            t0 = time.time()
            for _ in range(n_runs):
                rain_ice_gpu(
                    data['pexn'], data['prhodref'], data['ppabst'], data['pdzz'],
                    data['ptht'], data['prvt'], data['prct'], data['prrt'],
                    data['prit'], data['prst'], data['prgt'], data['pcit'],
                    data['pths'], data['prvs'], data['prcs'], data['prrs'],
                    data['pris'], data['prss'], data['prgs'], data['pcis'],
                    data['psigs']
                )
            cp.cuda.Stream.null.synchronize()
            elapsed = (time.time() - t0) / n_runs

            throughput = total_points / elapsed
            results.append((total_points, elapsed, throughput))

            print(f"\nDomain {nijt}×{nkt}: {total_points:,} points")
            print(f"  Time: {elapsed*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e6:.2f} M points/s")

        # Check scaling efficiency
        print("\n" + "-"*75)
        print("Scaling Efficiency:")
        print("-"*75)
        for i, (points, time_i, tput) in enumerate(results):
            if i > 0:
                size_ratio = points / results[0][0]
                time_ratio = time_i / results[0][1]
                efficiency = size_ratio / time_ratio * 100
                print(f"  {points:>10,} points: {efficiency:6.1f}% efficiency ({time_ratio:.2f}x time for {size_ratio:.2f}x points)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
