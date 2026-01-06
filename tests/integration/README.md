# Integration Tests

This directory contains integration tests demonstrating interoperability between different computational frameworks used in the dwarf-p-ice3 project.

## Test Files

### `test_gt4py_to_jax.py`

Tests the integration between GT4Py storage arrays and JAX for the ice_adjust component.

**What it demonstrates:**
- GT4Py storage arrays (or numpy arrays as fallback) can be created with `zeros()`
- Arrays can be efficiently converted to JAX arrays using DLPack protocol (zero-copy when possible)
- JAX components (`IceAdjustJAX`) work seamlessly with converted arrays
- Physical constraints (non-negative values, valid temperature ranges) are satisfied
- Conservation laws (total water) are maintained

**Key features:**
- **DLPack protocol**: Enables zero-copy conversion between frameworks for efficient memory usage
- **Automatic fallback**: Works with numpy arrays if GT4Py is not installed
- **JIT compilation**: JAX JIT-compiles the ice_adjust stencil for performance

**Usage:**
```bash
# Run all tests
uv run pytest tests/integration/test_gt4py_to_jax.py -v

# Run specific test
uv run pytest tests/integration/test_gt4py_to_jax.py::TestGT4PyToJAXCompatibility::test_storage_array_creation -v
```

**Requirements:**
- JAX >= 0.5.3
- Flax >= 0.10.2
- NumPy < 2.0
- GT4Py (optional, will use numpy fallback if not available)

### `test_gt4py_to_cupy.py`

Tests the integration between GT4Py storage arrays, CuPy GPU arrays, and Fortran OpenACC accelerated ice_adjust.

**What it demonstrates:**
- GT4Py storage arrays can be created with `zeros()`
- Arrays can be converted to CuPy GPU arrays for GPU computation
- CuPy GPU arrays work with Fortran OpenACC accelerated `ice_adjust_acc`
- Results can be transferred back from GPU to CPU
- Physical constraints and conservation laws are maintained on GPU

**Key features:**
- **Zero-copy GPU operations**: Direct transfer from GT4Py → CuPy → Fortran OpenACC
- **GPU acceleration**: Fortran OpenACC provides significant speedup on NVIDIA GPUs
- **Flexible input**: Works with both GT4Py storage and numpy arrays

**Usage:**
```bash
# Run all tests (requires GPU and Fortran OpenACC build)
uv run pytest tests/integration/test_gt4py_to_cupy.py -v

# Run specific GPU test
uv run pytest tests/integration/test_gt4py_to_cupy.py::TestGT4PyToCuPyIceAdjust -v
```

**Requirements:**
- CuPy (cuda12x or appropriate version)
- NVIDIA GPU with CUDA support
- Fortran OpenACC build (`cmake -DENABLE_OPENACC=ON`)
- GT4Py (optional, will use numpy fallback)

## Architecture Overview

```
┌─────────────┐
│   GT4Py     │  ← Domain-specific stencil computations
│   Storage   │
└──────┬──────┘
       │ DLPack (zero-copy)
       ├────────────────┬─────────────────┐
       ↓                ↓                 ↓
┌──────────────┐  ┌──────────┐   ┌──────────────┐
│     JAX      │  │  CuPy    │   │    NumPy     │
│  (CPU/GPU)   │  │  (GPU)   │   │    (CPU)     │
└──────┬───────┘  └─────┬────┘   └──────┬───────┘
       │                │                │
       │                │                │
       ↓                ↓                ↓
┌──────────────┐  ┌──────────┐   ┌──────────────┐
│ ice_adjust   │  │ Fortran  │   │  Reference   │
│   JAX JIT    │  │ OpenACC  │   │     Impl     │
└──────────────┘  └──────────┘   └──────────────┘
```

## DLPack Protocol

DLPack is an open standard for tensor/array exchange between frameworks. It enables:
- **Zero-copy transfers**: No data duplication when converting between frameworks
- **Cross-platform support**: Works with CPU and GPU arrays
- **Framework agnostic**: Supported by JAX, CuPy, PyTorch, TensorFlow, and more

Example:
```python
# GT4Py → JAX (zero-copy via DLPack)
gt4py_array = gt4py.storage.zeros(shape=(100, 100), dtype=np.float32)
jax_array = jax.dlpack.from_dlpack(gt4py_array)

# GT4Py → CuPy → GPU (efficient transfer)
cupy_array = cp.asarray(gt4py_array)  # Uses DLPack internally
```

## Running Tests Without GT4Py

Both test files work without GT4Py installed by using numpy arrays as a fallback:

```python
# Automatic fallback in tests
try:
    from gt4py.storage import zeros as gt4py_zeros
    HAS_GT4PY = True
except ImportError:
    HAS_GT4PY = False
    # Use numpy.zeros instead

# Create arrays (GT4Py if available, otherwise numpy)
array = create_zeros_array(shape=(5, 5, 10), dtype=np.float32)
```

This allows testing the integration principle even without GT4Py.

## Physical Validation

All integration tests verify:

1. **Physical Constraints:**
   - Non-negative mixing ratios (rv, rc, ri ≥ 0)
   - Reasonable temperature range (100 K < T < 400 K)
   - Cloud fraction bounds (0 ≤ cldfr ≤ 1)

2. **Conservation Laws:**
   - Total water conservation: rv + rc + ri = constant
   - Maximum difference < 1e-6 to 1e-8 kg/kg

3. **Numerical Stability:**
   - No NaN or Inf values
   - Convergence of saturation adjustment

## Performance Considerations

### JAX (CPU/GPU)
- JIT compilation provides ~10-100x speedup
- First call includes compilation overhead
- Subsequent calls reuse compiled code
- Automatic differentiation available

### Fortran OpenACC (GPU)
- ~50-200x speedup on NVIDIA GPUs vs CPU
- Optimized for atmospheric physics kernels
- Lower compilation overhead than JAX
- Production-ready for operational models

### Memory Efficiency
- DLPack enables zero-copy transfers (same memory)
- GPU transfers minimize CPU-GPU data movement
- GT4Py storage uses efficient memory layouts

## Contributing

When adding new integration tests:

1. Follow the existing test structure
2. Include proper skip markers for optional dependencies
3. Verify physical constraints and conservation
4. Document key features in test docstrings
5. Add usage examples to this README

## References

- [GT4Py Documentation](https://github.com/GridTools/gt4py)
- [JAX Documentation](https://jax.readthedocs.io/)
- [CuPy Documentation](https://docs.cupy.dev/)
- [DLPack Specification](https://github.com/dmlc/dlpack)
- [OpenACC Documentation](https://www.openacc.org/)
