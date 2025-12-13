# Running the PHYEX ICE_ADJUST Cython Example

## ✅ Success! The extension works

The Cython extension `_phyex_wrapper` has been successfully built and tested.

## How to Run

### Option 1: Use regular Python (Recommended)
```bash
python src/ice3/fortran/ice_adjust_cython.py
```

**Output**:
```
✓ ICE_ADJUST completed
Cloud fraction: 0.0000 (mean), max: 0.0000
Total water conservation check: 0.0000000000e+00
```

### Option 2: Use uv (requires rebuild)
If you want to use `uv run`, you need to ensure NumPy is available during the build:

```bash
# Install with uv first
uv pip install -e . --no-build-isolation

# Then run
uv run python src/ice3/fortran/ice_adjust_cython.py
```

## Why `uv run` fails

`uv run` creates an isolated build environment that doesn't have NumPy installed. CMake's `find_package(Python ... NumPy)` fails because NumPy headers aren't available.

**Error**: `Could NOT find Python (missing: Python_NumPy_INCLUDE_DIRS ...)`

## Solution

The extension is already built in `build/cp312-cp312-macosx_26_0_arm64/_phyex_wrapper.so`. Just use regular `python` to run scripts that import it.

## Files

- **Extension**: `build/cp312-cp312-macosx_26_0_arm64/_phyex_wrapper.so`
- **Example**: `src/ice3/fortran/ice_adjust_cython.py`
- **Bridge**: `PHYEX-IAL_CY50T1/bridge/phyex_bridge.F90`
- **Wrapper**: `PHYEX-IAL_CY50T1/bridge/_phyex_wrapper.pyx`
