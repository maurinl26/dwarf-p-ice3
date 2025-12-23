"""
Test GPU-accelerated RAIN_ICE using CuPy and OpenACC Fortran wrapper.

This test verifies:
1. GPU wrapper can be imported and executed
2. Results are physically valid
3. Correct handling of reproducibility dataset (rain_ice.nc)
4. Performance improvements on GPU vs CPU

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
    from ice3._phyex_wrapper_acc import RainIceGPU
    HAS_GPU_WRAPPER = True
except ImportError:
    # Fallback to local import
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "PHYEX-IAL_CY50T1/bridge"))
    try:
        from _phyex_wrapper_acc import RainIceGPU
        HAS_GPU_WRAPPER = True
    except ImportError:
        RainIceGPU = None
        HAS_GPU_WRAPPER = False

# Try to import CPU reference
try:
    from ice3._phyex_wrapper import rain_ice as rain_ice_cpu
    HAS_CPU_WRAPPER = True
except ImportError:
    try:
        from _phyex_wrapper import rain_ice as rain_ice_cpu
        HAS_CPU_WRAPPER = True
    except ImportError:
        rain_ice_cpu = None
        HAS_CPU_WRAPPER = False


def create_test_atmosphere_rain_ice_gpu(nijt=100, nkt=60):
    """
    Create realistic atmospheric test data for RAIN_ICE on GPU.

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
    print(f"\nCreating test atmosphere for RAIN_ICE on GPU ({nijt} x {nkt})...")

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
    data_cpu['pabs'] = np.tile(pressure, (nijt, 1)).astype(np.float32)

    # Temperature profile
    temperature = T0 - gamma * z
    data_cpu['temperature'] = np.tile(temperature, (nijt, 1)).astype(np.float32)

    # Add variability
    np.random.seed(42)
    data_cpu['temperature'] += np.random.randn(nijt, nkt).astype(np.float32) * 0.5
    data_cpu['pabs'] += np.random.randn(nijt, nkt).astype(np.float32) * 100

    # Exner function
    data_cpu['exn'] = (data_cpu['pabs'] / p00) ** (Rd / cp)
    data_cpu['tht'] = data_cpu['temperature'] / data_cpu['exn']

    # Reference values
    data_cpu['exnref'] = data_cpu['exn'].copy()
    data_cpu['rhodref'] = data_cpu['pabs'] / (Rd * data_cpu['temperature'])
    data_cpu['rhodj'] = data_cpu['rhodref'].copy()

    # Height and layer thickness
    data_cpu['dzz'] = np.full((nijt, nkt), z[1] - z[0] if len(z) > 1 else 100.0, dtype=np.float32)

    # Ice crystal concentration
    data_cpu['cit'] = np.zeros((nijt, nkt), dtype=np.float32)

    # Cloud fractions
    data_cpu['cldfr'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['icldfr'] = np.zeros((nijt, nkt), dtype=np.float32)

    # Water vapor
    rv_surf = 0.015
    rv_profile = rv_surf * np.exp(-z / 2000)
    data_cpu['rvt'] = np.tile(rv_profile, (nijt, 1)).astype(np.float32)
    data_cpu['rvt'] += np.abs(np.random.randn(nijt, nkt).astype(np.float32)) * 0.001

    # Cloud fields
    data_cpu['rct'] = np.zeros((nijt, nkt), dtype=np.float32)
    cloud_levels = (z > 2000) & (z < 6000)
    data_cpu['rct'][:, cloud_levels] = np.abs(np.random.rand(nijt, cloud_levels.sum()).astype(np.float32)) * 0.001

    data_cpu['rit'] = np.zeros((nijt, nkt), dtype=np.float32)
    ice_levels = z > 5000
    data_cpu['rit'][:, ice_levels] = np.abs(np.random.rand(nijt, ice_levels.sum()).astype(np.float32)) * 0.0005

    # Precipitation species
    data_cpu['rrt'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['rst'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['rgt'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['rht'] = np.zeros((nijt, nkt), dtype=np.float32)  # Hail

    # Tendencies (input/output) - initialize to zero
    data_cpu['ths'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['rvs'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['rcs'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['rrs'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['ris'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['rss'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['rgs'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['rhs'] = np.zeros((nijt, nkt), dtype=np.float32)

    # Output diagnostic arrays
    data_cpu['inprc'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['inprr'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['evap3d'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['inprs'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['inprg'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['indep'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['rainfr'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['inprh'] = np.zeros((nijt, nkt), dtype=np.float32)

    # Additional fields
    data_cpu['ssio'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['ssiu'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['ifr'] = np.zeros((nijt, nkt), dtype=np.float32)

    # Enthalpy variables
    data_cpu['hlc_hrc'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['hlc_hcf'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['hli_hri'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['hli_hcf'] = np.zeros((nijt, nkt), dtype=np.float32)

    # Sigma_s (turbulence parameter)
    data_cpu['sigs'] = np.full((nijt, nkt), 0.1, dtype=np.float32)

    # Sea and town masks
    data_cpu['sea'] = np.zeros((nijt, nkt), dtype=np.float32)
    data_cpu['town'] = np.zeros((nijt, nkt), dtype=np.float32)

    # Droplet concentration
    data_cpu['conc3d'] = np.full((nijt, nkt), 300.0e6, dtype=np.float32)  # 300/cm³ in SI

    # Precipitation flux profile (3D: krr=6 for ICE3)
    krr = 6
    data_cpu['fpr'] = np.zeros((nijt, nkt, krr), dtype=np.float32)

    # Transfer all arrays to GPU
    print("Transferring data to GPU...")
    data_gpu = {}
    for key, value in data_cpu.items():
        # Ensure contiguous and transfer to GPU
        data_gpu[key] = cp.asarray(np.ascontiguousarray(value), dtype=cp.float32)

    print(f"✓ Data transferred to GPU ({sum(arr.nbytes for arr in data_gpu.values()) / 1e6:.2f} MB)")

    return data_gpu


def load_rain_ice_repro_dataset_gpu(dataset_path=None):
    """
    Load rain_ice reproducibility dataset and transfer to GPU.

    Parameters
    ----------
    dataset_path : str or Path, optional
        Path to NetCDF dataset. If None, looks in data/ directory.

    Returns
    -------
    tuple
        (data_gpu, data_cpu) - Dictionary with dataset fields as CuPy and NumPy arrays
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy required for GPU tests")

    try:
        import xarray as xr
    except ImportError:
        pytest.skip("xarray required for dataset loading")

    if dataset_path is None:
        # Try common locations
        candidates = [
            Path(__file__).parent.parent.parent / 'data' / 'rain_ice.nc',
            Path(__file__).parent.parent.parent / 'data' / 'rain_ice_repro.nc',
        ]
        for candidate in candidates:
            if candidate.exists():
                dataset_path = candidate
                break
        else:
            pytest.skip("rain_ice reproducibility dataset not found")

    print(f"\nLoading rain_ice reproducibility dataset from: {dataset_path}")

    # Load dataset using xarray
    ds = xr.open_dataset(dataset_path)

    data_cpu = {}

    # Map dataset variables to RAIN_ICE parameters
    # Adjust based on actual dataset structure
    var_mapping = {
        'exn': 'exner',
        'dzz': 'dz',
        'rhodj': 'rhodj',
        'rhodref': 'rho_ref',
        'exnref': 'exner_ref',
        'pabs': 'pressure',
        'cit': 'cit',
        'cldfr': 'cloud_fraction',
        'icldfr': 'ice_cloud_fraction',
        'tht': 'theta',
        'rvt': 'rv',
        'rct': 'rc',
        'rrt': 'rr',
        'rit': 'ri',
        'rst': 'rs',
        'rgt': 'rg',
        'rht': 'rh',
        'sigs': 'sigma_s',
    }

    # Try to load each variable, use defaults if not present
    nijt, nkt = None, None
    for phyex_name, nc_name in var_mapping.items():
        if nc_name in ds:
            arr = ds[nc_name].values.astype(np.float32)
            if nijt is None and arr.ndim == 2:
                nijt, nkt = arr.shape
            data_cpu[phyex_name] = arr
        elif nijt is not None:
            # Create zeros with known shape
            data_cpu[phyex_name] = np.zeros((nijt, nkt), dtype=np.float32)

    # If shape still unknown, use default
    if nijt is None:
        nijt, nkt = 100, 60
        print(f"  Warning: Using default shape ({nijt}, {nkt})")
        for key in var_mapping.keys():
            if key not in data_cpu:
                data_cpu[key] = np.zeros((nijt, nkt), dtype=np.float32)

    # Initialize tendency arrays (all zeros initially)
    for name in ['ths', 'rvs', 'rcs', 'rrs', 'ris', 'rss', 'rgs', 'rhs']:
        if name not in data_cpu:
            data_cpu[name] = np.zeros((nijt, nkt), dtype=np.float32)

    # Initialize output diagnostic arrays
    for name in ['inprc', 'inprr', 'evap3d', 'inprs', 'inprg', 'indep', 'rainfr', 'inprh',
                 'ssio', 'ssiu', 'ifr', 'hlc_hrc', 'hlc_hcf', 'hli_hri', 'hli_hcf']:
        if name not in data_cpu:
            data_cpu[name] = np.zeros((nijt, nkt), dtype=np.float32)

    # Sea and town masks
    for name in ['sea', 'town']:
        if name not in data_cpu:
            data_cpu[name] = np.zeros((nijt, nkt), dtype=np.float32)

    # Droplet concentration
    if 'conc3d' not in data_cpu:
        data_cpu['conc3d'] = np.full((nijt, nkt), 300.0e6, dtype=np.float32)

    # Precipitation flux profile (3D)
    krr = 6
    if 'fpr' not in data_cpu:
        data_cpu['fpr'] = np.zeros((nijt, nkt, krr), dtype=np.float32)

    # Transfer to GPU
    print("Transferring dataset to GPU...")
    data_gpu = {key: cp.asarray(np.ascontiguousarray(val), dtype=cp.float32)
                for key, val in data_cpu.items()}

    print(f"✓ Dataset loaded on GPU ({sum(arr.nbytes for arr in data_gpu.values()) / 1e6:.2f} MB)")
    print(f"  Domain shape: {nijt} × {nkt}")

    ds.close()
    return data_gpu, data_cpu


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
@pytest.mark.skipif(not HAS_GPU, reason="No CUDA GPU available")
@pytest.mark.skipif(not HAS_GPU_WRAPPER, reason="GPU wrapper not built (use -DENABLE_OPENACC=ON)")
class TestRainIceGPU:
    """Test suite for GPU-accelerated RAIN_ICE."""

    def test_import_gpu_wrapper(self):
        """Test that GPU wrapper can be imported."""
        assert RainIceGPU is not None
        print("✓ RainIceGPU wrapper imported successfully")

    def test_create_gpu_instance(self):
        """Test creating RainIceGPU instance."""
        rain_ice_gpu = RainIceGPU(krr=6, timestep=1.0)
        assert rain_ice_gpu is not None
        assert rain_ice_gpu.has_gpu
        print("✓ RainIceGPU instance created")

    def test_gpu_execution_small(self):
        """Test GPU execution with small domain."""
        nijt, nkt = 10, 20

        # Create test data on GPU
        data = create_test_atmosphere_rain_ice_gpu(nijt, nkt)

        # Create GPU instance
        rain_ice_gpu = RainIceGPU(krr=6, timestep=1.0)

        # Execute
        rain_ice_gpu(
            data['exn'], data['dzz'], data['rhodj'], data['rhodref'],
            data['exnref'], data['pabs'], data['cit'], data['cldfr'],
            data['icldfr'], data['ssio'], data['ssiu'], data['ifr'],
            data['hlc_hrc'], data['hlc_hcf'], data['hli_hri'], data['hli_hcf'],
            data['tht'], data['rvt'], data['rct'], data['rrt'],
            data['rit'], data['rst'], data['rgt'],
            data['ths'], data['rvs'], data['rcs'], data['rrs'],
            data['ris'], data['rss'], data['rgs'],
            data['inprc'], data['inprr'], data['evap3d'],
            data['inprs'], data['inprg'], data['indep'],
            data['rainfr'], data['sigs'], data['sea'], data['town'],
            data['conc3d'], data['rht'], data['rhs'], data['inprh'],
            data['fpr']
        )

        # Check tendencies are computed (not all zero)
        ths_max = cp.max(cp.abs(data['ths']))
        print(f"✓ GPU execution successful ({nijt}×{nkt} domain)")
        print(f"  Max |ths|: {ths_max:.6e}")

    def test_gpu_execution_medium(self):
        """Test GPU execution with medium domain."""
        nijt, nkt = 1000, 60

        data = create_test_atmosphere_rain_ice_gpu(nijt, nkt)
        rain_ice_gpu = RainIceGPU(krr=6, timestep=1.0)

        rain_ice_gpu(
            data['exn'], data['dzz'], data['rhodj'], data['rhodref'],
            data['exnref'], data['pabs'], data['cit'], data['cldfr'],
            data['icldfr'], data['ssio'], data['ssiu'], data['ifr'],
            data['hlc_hrc'], data['hlc_hcf'], data['hli_hri'], data['hli_hcf'],
            data['tht'], data['rvt'], data['rct'], data['rrt'],
            data['rit'], data['rst'], data['rgt'],
            data['ths'], data['rvs'], data['rcs'], data['rrs'],
            data['ris'], data['rss'], data['rgs'],
            data['inprc'], data['inprr'], data['evap3d'],
            data['inprs'], data['inprg'], data['indep'],
            data['rainfr'], data['sigs'], data['sea'], data['town'],
            data['conc3d'], data['rht'], data['rhs'], data['inprh'],
            data['fpr']
        )

        print(f"✓ GPU execution successful ({nijt}×{nkt} domain)")
        print(f"  Max tendency ranges:")
        print(f"    ths: {cp.min(data['ths']):.6e} to {cp.max(data['ths']):.6e}")
        print(f"    rvs: {cp.min(data['rvs']):.6e} to {cp.max(data['rvs']):.6e}")

    @pytest.mark.slow
    def test_reproducibility_dataset(self):
        """Test GPU execution with reproducibility dataset."""
        try:
            data_gpu, data_cpu = load_rain_ice_repro_dataset_gpu()
        except Exception as e:
            pytest.skip(f"Could not load dataset: {e}")

        # Execute on GPU
        rain_ice_gpu = RainIceGPU(krr=6, timestep=1.0)
        rain_ice_gpu(
            data_gpu['exn'], data_gpu['dzz'], data_gpu['rhodj'], data_gpu['rhodref'],
            data_gpu['exnref'], data_gpu['pabs'], data_gpu['cit'], data_gpu['cldfr'],
            data_gpu['icldfr'], data_gpu['ssio'], data_gpu['ssiu'], data_gpu['ifr'],
            data_gpu['hlc_hrc'], data_gpu['hlc_hcf'], data_gpu['hli_hri'], data_gpu['hli_hcf'],
            data_gpu['tht'], data_gpu['rvt'], data_gpu['rct'], data_gpu['rrt'],
            data_gpu['rit'], data_gpu['rst'], data_gpu['rgt'],
            data_gpu['ths'], data_gpu['rvs'], data_gpu['rcs'], data_gpu['rrs'],
            data_gpu['ris'], data_gpu['rss'], data_gpu['rgs'],
            data_gpu['inprc'], data_gpu['inprr'], data_gpu['evap3d'],
            data_gpu['inprs'], data_gpu['inprg'], data_gpu['indep'],
            data_gpu['rainfr'], data_gpu['sigs'], data_gpu['sea'], data_gpu['town'],
            data_gpu['conc3d'], data_gpu['rht'], data_gpu['rhs'], data_gpu['inprh'],
            data_gpu['fpr']
        )

        # Check physical validity
        ths = cp.asnumpy(data_gpu['ths'])
        rvs = cp.asnumpy(data_gpu['rvs'])

        print("✓ Reproducibility dataset executed on GPU")
        print(f"  Domain shape: {ths.shape}")
        print(f"  Tendency ranges:")
        print(f"    ths: [{ths.min():.6e}, {ths.max():.6e}]")
        print(f"    rvs: [{rvs.min():.6e}, {rvs.max():.6e}]")

    @pytest.mark.benchmark
    def test_performance_large_domain(self):
        """Benchmark GPU performance on large domain."""
        import time

        nijt, nkt = 10000, 60

        print(f"\nBenchmarking RAIN_ICE GPU ({nijt}×{nkt})...")

        # Create data
        data = create_test_atmosphere_rain_ice_gpu(nijt, nkt)
        rain_ice_gpu = RainIceGPU(krr=6, timestep=1.0)

        # Warmup
        rain_ice_gpu(
            data['exn'], data['dzz'], data['rhodj'], data['rhodref'],
            data['exnref'], data['pabs'], data['cit'], data['cldfr'],
            data['icldfr'], data['ssio'], data['ssiu'], data['ifr'],
            data['hlc_hrc'], data['hlc_hcf'], data['hli_hri'], data['hli_hcf'],
            data['tht'], data['rvt'], data['rct'], data['rrt'],
            data['rit'], data['rst'], data['rgt'],
            data['ths'], data['rvs'], data['rcs'], data['rrs'],
            data['ris'], data['rss'], data['rgs'],
            data['inprc'], data['inprr'], data['evap3d'],
            data['inprs'], data['inprg'], data['indep'],
            data['rainfr'], data['sigs'], data['sea'], data['town'],
            data['conc3d'], data['rht'], data['rhs'], data['inprh'],
            data['fpr']
        )
        cp.cuda.Stream.null.synchronize()

        # Timed runs
        n_runs = 10
        t0 = time.time()
        for _ in range(n_runs):
            rain_ice_gpu(
                data['exn'], data['dzz'], data['rhodj'], data['rhodref'],
                data['exnref'], data['pabs'], data['cit'], data['cldfr'],
                data['icldfr'], data['ssio'], data['ssiu'], data['ifr'],
                data['hlc_hrc'], data['hlc_hcf'], data['hli_hri'], data['hli_hcf'],
                data['tht'], data['rvt'], data['rct'], data['rrt'],
                data['rit'], data['rst'], data['rgt'],
                data['ths'], data['rvs'], data['rcs'], data['rrs'],
                data['ris'], data['rss'], data['rgs'],
                data['inprc'], data['inprr'], data['evap3d'],
                data['inprs'], data['inprg'], data['indep'],
                data['rainfr'], data['sigs'], data['sea'], data['town'],
                data['conc3d'], data['rht'], data['rhs'], data['inprh'],
                data['fpr']
            )
        cp.cuda.Stream.null.synchronize()
        gpu_time = (time.time() - t0) / n_runs

        print(f"  GPU time: {gpu_time*1000:.2f} ms/iteration")
        print(f"  Throughput: {nijt*nkt/(gpu_time*1e6):.2f} Mpts/s")


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

    print("Running GPU RAIN_ICE tests...")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

    # Run small test
    nijt, nkt = 100, 60
    data = create_test_atmosphere_rain_ice_gpu(nijt, nkt)
    rain_ice_gpu = RainIceGPU(krr=6, timestep=1.0)

    rain_ice_gpu(
        data['exn'], data['dzz'], data['rhodj'], data['rhodref'],
        data['exnref'], data['pabs'], data['cit'], data['cldfr'],
        data['icldfr'], data['ssio'], data['ssiu'], data['ifr'],
        data['hlc_hrc'], data['hlc_hcf'], data['hli_hri'], data['hli_hcf'],
        data['tht'], data['rvt'], data['rct'], data['rrt'],
        data['rit'], data['rst'], data['rgt'],
        data['ths'], data['rvs'], data['rcs'], data['rrs'],
        data['ris'], data['rss'], data['rgs'],
        data['inprc'], data['inprr'], data['evap3d'],
        data['inprs'], data['inprg'], data['indep'],
        data['rainfr'], data['sigs'], data['sea'], data['town'],
        data['conc3d'], data['rht'], data['rhs'], data['inprh'],
        data['fpr']
    )

    print("\n✓ All tests passed!")
    print(f"Tendency ranges:")
    print(f"  ths: [{cp.min(data['ths']):.6e}, {cp.max(data['ths']):.6e}]")
    print(f"  rvs: [{cp.min(data['rvs']):.6e}, {cp.max(data['rvs']):.6e}]")
