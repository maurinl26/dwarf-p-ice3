# Performance Tests

This directory contains performance benchmarks for the dwarf-p-ice3 physics components.

## Test Files

### ICE_ADJUST Component

- **`test_ice_adjust_jax.py`** - Performance tests for JAX implementation
  - Small domain (10×10×20)
  - Medium domain (50×50×40)
  - Large domain (100×100×60)
  - JIT compilation overhead analysis
  - Scaling characteristics
  - Reproduction dataset benchmarks

- **`test_ice_adjust_fortran.py`** - Performance tests for Fortran CPU implementation
  - Small domain (100 horizontal × 20 vertical)
  - Medium domain (2500 × 40)
  - Large domain (10000 × 60)
  - Scaling characteristics
  - Reproduction dataset benchmarks

- **`test_ice_adjust_fortran_acc.py`** - Performance tests for Fortran GPU implementation (OpenACC)
  - Small domain (100 × 60)
  - Medium domain (2500 × 60)
  - Large domain (10000 × 60)
  - GPU vs CPU speedup analysis
  - GPU scaling characteristics
  - **Requires**: NVIDIA GPU, CuPy, built with `-DENABLE_OPENACC=ON`

- **`test_ice_adjust.py`** - GT4Py implementation performance tests
- **`test_ice_adjust_modular.py`** - Modular GT4Py implementation performance tests

### RAIN_ICE Component

- **`test_rain_ice_jax.py`** - Performance tests for JAX implementation
  - Small domain (10×10×20)
  - Medium domain (50×50×40)
  - Large domain (100×100×60)
  - Scaling characteristics
  - Reproduction dataset benchmarks

- **`test_rain_ice_fortran.py`** - Performance tests for Fortran CPU implementation
  - Small domain (100 × 20)
  - Medium domain (2500 × 40)
  - Large domain (10000 × 60)
  - Scaling characteristics

- **`test_rain_ice_fortran_acc.py`** - Performance tests for Fortran GPU implementation (OpenACC)
  - Small domain (100 × 60)
  - Medium domain (2500 × 60)
  - Large domain (10000 × 60)
  - GPU vs CPU speedup analysis
  - GPU scaling characteristics
  - **Requires**: NVIDIA GPU, CuPy, built with `-DENABLE_OPENACC=ON`

## Running Performance Tests

### Run All Performance Tests

```bash
pytest tests/performance/ --benchmark-only -v
```

### Run Specific Component Tests

**JAX implementation:**
```bash
pytest tests/performance/test_ice_adjust_jax.py --benchmark-only -v
```

**Fortran CPU implementation:**
```bash
pytest tests/performance/test_ice_adjust_fortran.py --benchmark-only -v
```

**Fortran GPU implementation (OpenACC):**
```bash
pytest tests/performance/test_ice_adjust_fortran_acc.py --benchmark-only -v
```

**GT4Py implementation:**
```bash
pytest tests/performance/test_ice_adjust.py --benchmark-only -v
```

### Run Specific Domain Size

**Small domain (fast, for CI):**
```bash
pytest tests/performance/test_ice_adjust_jax.py::TestIceAdjustJAXPerformanceSmall --benchmark-only -v
```

**Large domain (detailed benchmarks):**
```bash
pytest tests/performance/test_ice_adjust_jax.py::TestIceAdjustJAXPerformanceLarge --benchmark-only -v
```

### Run with Reproduction Dataset

```bash
pytest tests/performance/test_ice_adjust_jax.py::test_ice_adjust_jax_with_repro_data --benchmark-only -v
pytest tests/performance/test_ice_adjust_fortran.py::test_ice_adjust_fortran_with_repro_data --benchmark-only -v
```

## Benchmark Options

### Save Results

```bash
pytest tests/performance/ --benchmark-only --benchmark-save=my_benchmark
```

### Compare Results

```bash
pytest tests/performance/ --benchmark-only --benchmark-compare=my_benchmark
```

### Generate Histogram

```bash
pytest tests/performance/ --benchmark-only --benchmark-histogram
```

### Adjust Iterations

```bash
# Fewer iterations (faster)
pytest tests/performance/ --benchmark-only --benchmark-min-rounds=3

# More iterations (more accurate)
pytest tests/performance/ --benchmark-only --benchmark-min-rounds=10
```

## Performance Metrics

The benchmarks report the following metrics:

- **Mean time**: Average execution time per call
- **Throughput**: Grid points processed per second (M points/s)
- **Performance**: Point-steps per second (M point-steps/s)
- **Standard deviation**: Variability in execution time
- **Min/Max time**: Range of execution times

## Test Organization

### JAX Tests

- **Initialization tests**: JIT compilation, constants extraction
- **Small/Medium/Large domain**: Different domain sizes
- **Compilation overhead**: JIT compilation cost analysis
- **Scaling tests**: How performance scales with domain size

### Fortran Tests

- **Small/Medium/Large domain**: Different domain sizes
- **Scaling tests**: Performance scaling characteristics
- **Reproduction data**: Real atmospheric data from PHYEX

## Requirements

### JAX Tests
```bash
pip install jax jaxlib pytest-benchmark
```

### Fortran CPU Tests
```bash
# Fortran extension must be built
pip install -e .
```

### Fortran GPU Tests (OpenACC)
```bash
# Build with GPU support
pip install -e . --config-settings=cmake.args="-DCMAKE_Fortran_COMPILER=nvfortran;-DENABLE_OPENACC=ON"

# Install CuPy
pip install cupy-cuda12x  # or appropriate CUDA version
```

### GT4Py Tests
```bash
pip install gt4py pytest-benchmark
```

## Interpreting Results

### Good Performance Indicators

- **CPU implementations**: Throughput > 10 M points/s
- **GPU implementations**: Throughput > 100 M points/s
- **GPU speedup**: 10-50× over CPU (domain size dependent)
- **Scaling**: Near-linear with domain size
- **Consistency**: Standard deviation < 5% of mean

### Performance Comparison

Typical relative performance:
- Fortran (CPU): Baseline reference
- JAX (CPU, JIT): 0.5-2× Fortran CPU
- GT4Py (CPU): 0.5-1.5× Fortran CPU
- Fortran (GPU, OpenACC): 10-50× Fortran CPU
- JAX (GPU): 10-50× Fortran CPU (domain dependent)
- GT4Py (GPU): 10-50× Fortran CPU (domain dependent)

GPU speedup increases with domain size due to better parallelization.

## Notes

- First run may be slower due to JIT compilation (JAX) or code generation (GT4Py)
- GPU tests require appropriate hardware and drivers
- Reproduction dataset tests require `data/ice_adjust.nc` file
- Results depend on hardware, system load, and compiler optimizations

## Troubleshooting

### Fortran Tests Skipped

If you see "Fortran wrapper not available":
```bash
# Build the Fortran extension
pip install -e .
```

### GPU Tests Skipped

If GPU tests are skipped:
```bash
# Check CuPy installation
pip install cupy-cuda12x  # adjust for your CUDA version

# Rebuild with GPU support
pip install -e . --config-settings=cmake.args="-DCMAKE_Fortran_COMPILER=nvfortran;-DENABLE_OPENACC=ON"

# Verify GPU is available
python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0))"
```

### Out of Memory

For large domains, you may need to:
- Reduce domain size
- Run on a machine with more RAM
- For GPU: Use GPU with more memory or reduce domain size

### Slow Performance

- Ensure JIT compilation has completed (warm-up)
- Check system load and close other applications
- Verify optimization flags during compilation
- Use Release build type for Fortran
