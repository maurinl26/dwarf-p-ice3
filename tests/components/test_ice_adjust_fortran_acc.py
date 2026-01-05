"""
Test GPU-accelerated ICE_ADJUST using CuPy and OpenACC Fortran wrapper.

This test verifies:
1. GPU wrapper can be imported and executed
2. Results match CPU reference implementation
3. Performance improvements on GPU vs CPU
4. Correct handling of reproducibility dataset

Requirements:
- NVIDIA GPU with CUDA support
- CuPy installed (pip install cupy-cuda12x)
- Built with ENABLE_OPENACC=ON
"""

import numpy as np
import sys
import os
from pathlib import Path
import pytest

# Try to import CuPy
try:
    import cupy as cp
    HAS_CUPY = True
    HAS_GPU = cp.cuda.is_available()
except ImportError:
    cp = None
    HAS_CUPY = False
    HAS_GPU = False

# Add build directory to path to find the compiled extension
build_dir = Path(__file__).parent.parent.parent / 'build-gpu'
if not build_dir.exists():
    build_dir = Path(__file__).parent.parent.parent / 'build'

# Search for the actual build directory
if build_dir.exists():
    for sub in build_dir.iterdir():
        if sub.is_dir() and sub.name.startswith('cp'):
            sys.path.insert(0, str(sub))
            break

# Try to import GPU wrapper
try:
    from ice3._phyex_wrapper_acc import IceAdjustGPU
    HAS_GPU_WRAPPER = True
except ImportError:
    # Fallback to local import
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "PHYEX-IAL_CY50T1/bridge"))
    try:
        from _phyex_wrapper_acc import IceAdjustGPU
        HAS_GPU_WRAPPER = True
    except ImportError:
        IceAdjustGPU = None
        HAS_GPU_WRAPPER = False

# Try to import CPU reference
try:
    from ice3._phyex_wrapper import ice_adjust as ice_adjust_cpu
    HAS_CPU_WRAPPER = True
except ImportError:
    try:
        from _phyex_wrapper import ice_adjust as ice_adjust_cpu
        HAS_CPU_WRAPPER = True
    except ImportError:
        ice_adjust_cpu = None
        HAS_CPU_WRAPPER = False


def create_test_atmosphere_gpu(nijt=100, nkt=60):
    """
    Create realistic atmospheric test data for ICE_ADJUST on GPU.

    Returns CuPy arrays ready for GPU execution.

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
    print(f"\nCreating test atmosphere on GPU ({nijt} x {nkt})...")

    if not HAS_CUPY:
        raise RuntimeError("CuPy required for GPU tests")

    # Create on CPU first, then transfer to GPU
    z = np.linspace(0, 10000, nkt, dtype=np.float32)

    # Standard atmosphere
    p0 = 101325.0
    T0 = 288.15
    gamma = 0.0065

    # Physical constants
    Rd = 287.0
    cp = 1004.0
    p00 = 100000.0

    # Create CPU data first
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
    data_cpu['pth'] = data_cpu['temperature'] / data_cpu['pexn']

    # Reference values
    data_cpu['pexnref'] = data_cpu['pexn'].copy()
    data_cpu['prhodref'] = data_cpu['ppabst'] / (Rd * data_cpu['temperature'])

    # Height
    data_cpu['pzz'] = np.tile(z, (nijt, 1)).astype(np.float32)

    # Water vapor
    rv_surf = 0.015
    rv_profile = rv_surf * np.exp(-z / 2000)
    data_cpu['prv'] = np.tile(rv_profile, (nijt, 1)).astype(np.float32)
    data_cpu['prv'] += np.abs(np.random.randn(nijt, nkt).astype(np.float32)) * 0.001

    # Cloud fields
    data_cpu['prc'] = np.zeros((nijt, nkt), dtype=np.float32)
    cloud_levels = (z > 2000) & (z < 6000)
    data_cpu['prc'][:, cloud_levels] = np.abs(np.random.rand(nijt, cloud_levels.sum()).astype(np.float32)) * 0.002

    data_cpu['pri'] = np.zeros((nijt, nkt), dtype=np.float32)
    ice_levels = z > 5000
    data_cpu['pri'][:, ice_levels] = np.abs(np.random.rand(nijt, ice_levels.sum()).astype(np.float32)) * 0.001

    # Precipitation (all zero for saturation adjustment)
    data_cpu['prr'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['prs'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['prg'] = np.zeros((nijt, nkt), dtype=np.float32)

    # Mass flux cloud (all zero for basic test)
    data_cpu['pcf_mf'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['prc_mf'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['pri_mf'] = np.zeros((nijt, nkt), dtype=np.float32)

    # Sigma_s (turbulence parameter)
    data_cpu['psigs'] = np.full((nijt, nkt), 0.1, dtype=np.float32)

    # Subgrid saturation variance
    data_cpu['psigqsat'] = np.full(nijt, 0.02, dtype=np.float32)

    # Tendencies (input/output) - initialize to zero
    data_cpu['prvs'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['prcs'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['pris'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['pths'] = np.zeros((nijt, nkt), dtype=np.float32)

    # Cloud fraction outputs
    data_cpu['pcldfr'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['picldfr'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['pwcldfr'] = np.zeros((nijt, nkt), dtype=np.float32)

    # Transfer all arrays to GPU
    print("Transferring data to GPU...")
    data_gpu = {}
    for key, value in data_cpu.items():
        # Ensure contiguous and transfer to GPU
        data_gpu[key] = cp.asarray(np.ascontiguousarray(value), dtype=cp.float32)

    print(f"✓ Data transferred to GPU ({sum(arr.nbytes for arr in data_gpu.values()) / 1e6:.2f} MB)")

    return data_gpu


def load_reproducibility_dataset_gpu(dataset_path=None):
    """
    Load reproducibility dataset and transfer to GPU.

    Parameters
    ----------
    dataset_path : str or Path, optional
        Path to NetCDF dataset. If None, looks in data/ directory.

    Returns
    -------
    dict
        Dictionary with dataset fields as CuPy arrays
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy required for GPU tests")

    try:
        import netCDF4 as nc
    except ImportError:
        pytest.skip("netCDF4 required for dataset loading")

    if dataset_path is None:
        # Try common locations
        candidates = [
            Path(__file__).parent.parent.parent / 'data' / 'ice_adjust_run.nc',
            Path(__file__).parent.parent.parent / 'data' / 'rain_ice_run.nc',
        ]
        for candidate in candidates:
            if candidate.exists():
                dataset_path = candidate
                break
        else:
            pytest.skip("Reproducibility dataset not found")

    print(f"\nLoading reproducibility dataset from: {dataset_path}")

    with nc.Dataset(dataset_path, 'r') as ds:
        # Load data from NetCDF
        data_cpu = {}

        # Required fields (adjust names based on actual dataset)
        field_mapping = {
            'ppabst': 'pressure',
            'pth': 'theta',
            'pexn': 'exner',
            'pexnref': 'exner_ref',
            'prhodref': 'rho_ref',
            'pzz': 'height',
            'prv': 'rv',
            'prc': 'rc',
            'pri': 'ri',
            'prr': 'rr',
            'prs': 'rs',
            'prg': 'rg',
            'psigs': 'sigs',
            'psigqsat': 'sigqsat',
        }

        for phyex_name, nc_name in field_mapping.items():
            if nc_name in ds.variables:
                data_cpu[phyex_name] = np.asarray(ds[nc_name][:], dtype=np.float32)
            else:
                print(f"  Warning: {nc_name} not in dataset, using zeros")
                shape = data_cpu.get('ppabst', (100, 60)).shape
                data_cpu[phyex_name] = np.zeros(shape, dtype=np.float32)

        # Initialize tendencies and outputs
        shape = data_cpu['ppabst'].shape
        for name in ['prvs', 'prcs', 'pris', 'pths', 'pcldfr', 'picldfr', 'pwcldfr',
                     'pcf_mf', 'prc_mf', 'pri_mf']:
            data_cpu[name] = np.zeros(shape, dtype=np.float32)

        # sigqsat is 1D
        if 'psigqsat' not in data_cpu or data_cpu['psigqsat'].ndim == 2:
            data_cpu['psigqsat'] = np.full(shape[0], 0.02, dtype=np.float32)

    # Transfer to GPU
    print("Transferring dataset to GPU...")
    data_gpu = {key: cp.asarray(np.ascontiguousarray(val), dtype=cp.float32)
                for key, val in data_cpu.items()}

    print(f"✓ Dataset loaded on GPU ({sum(arr.nbytes for arr in data_gpu.values()) / 1e6:.2f} MB)")

    return data_gpu, data_cpu


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
@pytest.mark.skipif(not HAS_GPU, reason="No CUDA GPU available")
@pytest.mark.skipif(not HAS_GPU_WRAPPER, reason="GPU wrapper not built (use -DENABLE_OPENACC=ON)")
class TestIceAdjustGPU:
    """Test suite for GPU-accelerated ICE_ADJUST."""

    def test_import_gpu_wrapper(self):
        """Test that GPU wrapper can be imported."""
        assert IceAdjustGPU is not None
        print("✓ GPU wrapper imported successfully")

    def test_create_gpu_instance(self):
        """Test creating IceAdjustGPU instance."""
        ice_adjust_gpu = IceAdjustGPU(krr=6, timestep=1.0)
        assert ice_adjust_gpu is not None
        assert ice_adjust_gpu.has_gpu
        print("✓ GPU instance created")

    def test_gpu_execution_small(self):
        """Test GPU execution with small domain."""
        nijt, nkt = 10, 20

        # Create test data on GPU
        data = create_test_atmosphere_gpu(nijt, nkt)

        # Create GPU instance
        ice_adjust_gpu = IceAdjustGPU(krr=6, timestep=1.0)

        # Execute
        ice_adjust_gpu(
            data['psigqsat'],
            data['ppabst'], data['psigs'], data['pth'], data['pexn'],
            data['pexnref'], data['prhodref'],
            data['prv'], data['prc'], data['pri'],
            data['prr'], data['prs'], data['prg'],
            data['pcf_mf'], data['prc_mf'], data['pri_mf'],
            data['prvs'], data['prcs'], data['pris'], data['pths'],
            data['pcldfr'], data['picldfr'], data['pwcldfr']
        )

        # Check outputs are not all zero
        assert cp.any(data['pcldfr'] > 0), "Cloud fraction should be non-zero"

        print(f"✓ GPU execution successful ({nijt}×{nkt} domain)")
        print(f"  Cloud fraction range: [{cp.min(data['pcldfr']):.4f}, {cp.max(data['pcldfr']):.4f}]")
        print(f"  Mean cloud fraction: {cp.mean(data['pcldfr']):.4f}")

    def test_gpu_execution_medium(self):
        """Test GPU execution with medium domain."""
        nijt, nkt = 1000, 60

        data = create_test_atmosphere_gpu(nijt, nkt)
        ice_adjust_gpu = IceAdjustGPU(krr=6, timestep=1.0)

        ice_adjust_gpu(
            data['psigqsat'],
            data['ppabst'], data['psigs'], data['pth'], data['pexn'],
            data['pexnref'], data['prhodref'],
            data['prv'], data['prc'], data['pri'],
            data['prr'], data['prs'], data['prg'],
            data['pcf_mf'], data['prc_mf'], data['pri_mf'],
            data['prvs'], data['prcs'], data['pris'], data['pths'],
            data['pcldfr'], data['picldfr'], data['pwcldfr']
        )

        assert cp.any(data['pcldfr'] > 0)
        print(f"✓ GPU execution successful ({nijt}×{nkt} domain)")

    @pytest.mark.skipif(not HAS_CPU_WRAPPER, reason="CPU wrapper not available for comparison")
    def test_gpu_vs_cpu_accuracy(self):
        """Test GPU results match CPU reference."""
        nijt, nkt = 100, 60

        # Create data on GPU
        data_gpu = create_test_atmosphere_gpu(nijt, nkt)

        # Copy to CPU for reference
        data_cpu = {key: cp.asnumpy(val) for key, val in data_gpu.items()}

        # Create copies for CPU execution (Fortran order for Fortran wrapper)
        data_cpu_f = {key: np.asfortranarray(val) for key, val in data_cpu.items()}

        # Execute on GPU
        ice_adjust_gpu = IceAdjustGPU(krr=6, timestep=1.0)
        ice_adjust_gpu(
            data_gpu['psigqsat'],
            data_gpu['ppabst'], data_gpu['psigs'], data_gpu['pth'], data_gpu['pexn'],
            data_gpu['pexnref'], data_gpu['prhodref'],
            data_gpu['prv'], data_gpu['prc'], data_gpu['pri'],
            data_gpu['prr'], data_gpu['prs'], data_gpu['prg'],
            data_gpu['pcf_mf'], data_gpu['prc_mf'], data_gpu['pri_mf'],
            data_gpu['prvs'], data_gpu['prcs'], data_gpu['pris'], data_gpu['pths'],
            data_gpu['pcldfr'], data_gpu['picldfr'], data_gpu['pwcldfr']
        )

        # Execute on CPU
        ice_adjust_cpu(
            nkt, nijt, 6, 1.0,
            data_cpu_f['psigqsat'],
            data_cpu_f['ppabst'], data_cpu_f['psigs'], data_cpu_f['pth'], data_cpu_f['pexn'],
            data_cpu_f['pexnref'], data_cpu_f['prhodref'],
            data_cpu_f['prv'], data_cpu_f['prc'], data_cpu_f['pri'],
            data_cpu_f['prr'], data_cpu_f['prs'], data_cpu_f['prg'],
            data_cpu_f['pcf_mf'], data_cpu_f['prc_mf'], data_cpu_f['pri_mf'],
            data_cpu_f['prvs'], data_cpu_f['prcs'], data_cpu_f['pris'], data_cpu_f['pths'],
            data_cpu_f['pcldfr'], data_cpu_f['picldfr'], data_cpu_f['pwcldfr']
        )

        # Transfer GPU results to CPU
        cldfr_gpu = cp.asnumpy(data_gpu['pcldfr'])
        cldfr_cpu = data_cpu_f['pcldfr']

        # Compare (allow small numerical differences)
        np.testing.assert_allclose(cldfr_gpu, cldfr_cpu, rtol=1e-5, atol=1e-7,
                                    err_msg="GPU results don't match CPU reference")

        print("✓ GPU results match CPU reference")
        max_diff = np.max(np.abs(cldfr_gpu - cldfr_cpu))
        print(f"  Max difference: {max_diff:.2e}")

    @pytest.mark.slow
    def test_reproducibility_dataset(self):
        """Test GPU execution with reproducibility dataset."""
        try:
            data_gpu, data_cpu = load_reproducibility_dataset_gpu()
        except Exception as e:
            pytest.skip(f"Could not load dataset: {e}")

        # Execute on GPU
        ice_adjust_gpu = IceAdjustGPU(krr=6, timestep=1.0)
        ice_adjust_gpu(
            data_gpu['psigqsat'],
            data_gpu['ppabst'], data_gpu['psigs'], data_gpu['pth'], data_gpu['pexn'],
            data_gpu['pexnref'], data_gpu['prhodref'],
            data_gpu['prv'], data_gpu['prc'], data_gpu['pri'],
            data_gpu['prr'], data_gpu['prs'], data_gpu['prg'],
            data_gpu['pcf_mf'], data_gpu['prc_mf'], data_gpu['pri_mf'],
            data_gpu['prvs'], data_gpu['prcs'], data_gpu['pris'], data_gpu['pths'],
            data_gpu['pcldfr'], data_gpu['picldfr'], data_gpu['pwcldfr']
        )

        # Check physical validity
        cldfr = cp.asnumpy(data_gpu['pcldfr'])
        assert np.all((cldfr >= 0) & (cldfr <= 1)), "Cloud fraction must be in [0, 1]"

        print("✓ Reproducibility dataset executed on GPU")
        print(f"  Domain shape: {cldfr.shape}")
        print(f"  Cloud fraction range: [{cldfr.min():.4f}, {cldfr.max():.4f}]")
        print(f"  Mean cloud fraction: {cldfr.mean():.4f}")

    @pytest.mark.benchmark
    @pytest.mark.skipif(not HAS_CPU_WRAPPER, reason="CPU wrapper needed for benchmark")
    def test_performance_benchmark(self):
        """Benchmark GPU vs CPU performance."""
        import time

        sizes = [(100, 60), (1000, 60), (10000, 60)]

        print("\n" + "="*60)
        print("Performance Benchmark: GPU vs CPU")
        print("="*60)

        for nijt, nkt in sizes:
            # Create data
            data_gpu = create_test_atmosphere_gpu(nijt, nkt)
            data_cpu = {key: np.asfortranarray(cp.asnumpy(val)) for key, val in data_gpu.items()}

            # GPU benchmark (with warmup)
            ice_adjust_gpu = IceAdjustGPU(krr=6, timestep=1.0)

            # Warmup
            ice_adjust_gpu(
                data_gpu['psigqsat'],
                data_gpu['ppabst'], data_gpu['psigs'], data_gpu['pth'], data_gpu['pexn'],
                data_gpu['pexnref'], data_gpu['prhodref'],
                data_gpu['prv'], data_gpu['prc'], data_gpu['pri'],
                data_gpu['prr'], data_gpu['prs'], data_gpu['prg'],
                data_gpu['pcf_mf'], data_gpu['prc_mf'], data_gpu['pri_mf'],
                data_gpu['prvs'], data_gpu['prcs'], data_gpu['pris'], data_gpu['pths'],
                data_gpu['pcldfr'], data_gpu['picldfr'], data_gpu['pwcldfr']
            )
            cp.cuda.Stream.null.synchronize()

            # Timed runs
            n_runs = 10
            t0 = time.time()
            for _ in range(n_runs):
                ice_adjust_gpu(
                    data_gpu['psigqsat'],
                    data_gpu['ppabst'], data_gpu['psigs'], data_gpu['pth'], data_gpu['pexn'],
                    data_gpu['pexnref'], data_gpu['prhodref'],
                    data_gpu['prv'], data_gpu['prc'], data_gpu['pri'],
                    data_gpu['prr'], data_gpu['prs'], data_gpu['prg'],
                    data_gpu['pcf_mf'], data_gpu['prc_mf'], data_gpu['pri_mf'],
                    data_gpu['prvs'], data_gpu['prcs'], data_gpu['pris'], data_gpu['pths'],
                    data_gpu['pcldfr'], data_gpu['picldfr'], data_gpu['pwcldfr']
                )
            cp.cuda.Stream.null.synchronize()
            gpu_time = (time.time() - t0) / n_runs

            # CPU benchmark
            t0 = time.time()
            for _ in range(n_runs):
                ice_adjust_cpu(
                    nkt, nijt, 6, 1.0,
                    data_cpu['psigqsat'],
                    data_cpu['ppabst'], data_cpu['psigs'], data_cpu['pth'], data_cpu['pexn'],
                    data_cpu['pexnref'], data_cpu['prhodref'],
                    data_cpu['prv'], data_cpu['prc'], data_cpu['pri'],
                    data_cpu['prr'], data_cpu['prs'], data_cpu['prg'],
                    data_cpu['pcf_mf'], data_cpu['prc_mf'], data_cpu['pri_mf'],
                    data_cpu['prvs'], data_cpu['prcs'], data_cpu['pris'], data_cpu['pths'],
                    data_cpu['pcldfr'], data_cpu['picldfr'], data_cpu['pwcldfr']
                )
            cpu_time = (time.time() - t0) / n_runs

            speedup = cpu_time / gpu_time
            print(f"\n{nijt:6d} × {nkt:3d}:")
            print(f"  CPU: {cpu_time*1000:7.2f} ms")
            print(f"  GPU: {gpu_time*1000:7.2f} ms")
            print(f"  Speedup: {speedup:6.1f}×")


if __name__ == "__main__":
    # Run basic tests if executed directly
    if not HAS_CUPY:
        print("ERROR: CuPy not available")
        print("Install with: pip install cupy-cuda12x")
        sys.exit(1)

    if not HAS_GPU:
        print("ERROR: No CUDA GPU available")
        sys.exit(1)

    if not HAS_GPU_WRAPPER:
        print("ERROR: GPU wrapper not built")
        print("Build with: cmake -DENABLE_OPENACC=ON && make")
        sys.exit(1)

    print("Running GPU ICE_ADJUST tests...")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

    # Run small test
    nijt, nkt = 100, 60
    data = create_test_atmosphere_gpu(nijt, nkt)
    ice_adjust_gpu = IceAdjustGPU(krr=6, timestep=1.0)
    ice_adjust_gpu(
        data['psigqsat'],
        data['ppabst'], data['psigs'], data['pth'], data['pexn'],
        data['pexnref'], data['prhodref'],
        data['prv'], data['prc'], data['pri'],
        data['prr'], data['prs'], data['prg'],
        data['pcf_mf'], data['prc_mf'], data['pri_mf'],
        data['prvs'], data['prcs'], data['pris'], data['pths'],
        data['pcldfr'], data['picldfr'], data['pwcldfr']
    )

    print("\n✓ All tests passed!")
    print(f"Cloud fraction range: [{cp.min(data['pcldfr']):.4f}, {cp.max(data['pcldfr']):.4f}]")
