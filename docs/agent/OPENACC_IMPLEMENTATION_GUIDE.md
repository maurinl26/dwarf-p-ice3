# OpenACC Implementation Guide for ICE3 Microphysics

## Overview

This document describes the OpenACC GPU acceleration implementation added to the PHYEX ICE_ADJUST microphysics scheme for GPU execution via Cython wrappers.

**Generated:** December 20, 2025
**Based on:** MesoNH PHYEX-IAL_CY50T1

---

## Files Created

### 1. OpenACC-Enabled Fortran Code

**Location:** `PHYEX-IAL_CY50T1/micro/ice_adjust_acc.F90`

This is the GPU-accelerated version of `ice_adjust.F90` with OpenACC directives added for GPU parallelization.

---

## OpenACC Directives Added

### Data Management

```fortran
!$acc data &
!$acc&  present(PTH, PEXN, ZT, ZLV, ZLS, CST, D) &
!$acc&  present(PRV, PRC, PRI, PRVS, PRCS, PRIS, PTHS) &
!$acc&  present(PRHODJ, PEXNREF, PRHODREF, PPABST, PZZ) &
!$acc&  present(PCF_MF, PRC_MF, PRI_MF, PWEIGHT_MF_CLOUD) &
!$acc&  present(PRR, PRS, PRG, PEXN) &
!$acc&  present(PCLDFR, PICLDFR, PWCLDFR, PSSIO, PSSIU, PIFR) &
!$acc&  present(ZRV, ZRC, ZRI, ZCPH, ZSRCS) &
!$acc&  copyin(PSIGQSAT, PSIGS)
```

The `!$acc data` directive creates a data region where arrays are made available on the GPU:
- `present()`: Arrays already on GPU (passed from calling code)
- `copyin()`: Arrays copied from CPU to GPU at region entry

### Parallel Loops

#### Pattern 1: Simple 2D Collapse
```fortran
!$acc parallel loop gang vector collapse(2) private(JIJ, JK)
DO JK=IKTB,IKTE
  DO JIJ=IIJB,IIJE
    ! Computation
  ENDDO
ENDDO
!$acc end parallel loop
```

- `gang vector`: Distributes work across GPU streaming multiprocessors (gang) and threads (vector)
- `collapse(2)`: Flattens nested loops for better GPU utilization
- `private(JIJ, JK)`: Loop indices are private to each thread

#### Pattern 2: Loops with Local Variables
```fortran
!$acc parallel loop gang vector collapse(2) &
!$acc&  private(JIJ, JK, ZW1, ZW2, ZCRIAUT, ZHCF, ZHR)
DO JK=IKTB,IKTE
  DO JIJ=IIJB,IIJE
    ! Computations using ZW1, ZW2, etc.
  ENDDO
ENDDO
!$acc end parallel loop
```

- Additional `private()` clause for temporary scalars used within loop
- Each thread gets its own copy of these variables

---

## Key Parallelization Zones

### 1. Latent Heat Computation (Lines ~265-270)
```fortran
!$acc parallel loop gang vector collapse(2) private(JIJ, JK)
DO JK=IKTB,IKTE
  DO JIJ=IIJB,IIJE
    IF (JITER==1) ZT(JIJ,JK) = PTH(JIJ,JK) * PEXN(JIJ,JK)
    ZLV(JIJ,JK) = CST%XLVTT + ( CST%XCPV - CST%XCL ) * ( ZT(JIJ,JK) -CST%XTT )
    ZLS(JIJ,JK) = CST%XLSTT + ( CST%XCPV - CST%XCI ) * ( ZT(JIJ,JK) -CST%XTT )
  ENDDO
ENDDO
!$acc end parallel loop
```
**Purpose:** Initialize temperature and compute latent heats for all grid points in parallel.

### 2. Mass Flux Cloud Ponderation (Lines ~293-308)
```fortran
!$acc parallel loop gang vector collapse(2) private(JIJ, JK)
DO JK=IKTB,IKTE
  DO JIJ=IIJB,IIJE
    ZRC(JIJ,JK)=ZRC(JIJ,JK)*(1.-PWEIGHT_MF_CLOUD(JIJ,JK))
    ZRI(JIJ,JK)=ZRI(JIJ,JK)*(1.-PWEIGHT_MF_CLOUD(JIJ,JK))
    PCLDFR(JIJ,JK)=PCLDFR(JIJ,JK)*(1.-PWEIGHT_MF_CLOUD(JIJ,JK))
    ZSRCS(JIJ,JK)=ZSRCS(JIJ,JK)*(1.-PWEIGHT_MF_CLOUD(JIJ,JK))
    ! ...
  ENDDO
ENDDO
!$acc end parallel loop
```
**Purpose:** Apply weighting between condensation and mass flux cloud schemes.

### 3. Source Term Computation (Lines ~313-345)
```fortran
!$acc parallel loop gang vector collapse(2) private(JIJ, JK, ZW1, ZW2)
DO JK=IKTB,IKTE
  DO JIJ=IIJB,IIJE
    ZW1 = (ZRC(JIJ,JK) - PRC(JIJ,JK)) / PTSTEP
    ZW2 = (ZRI(JIJ,JK) - PRI(JIJ,JK)) / PTSTEP
    ! Update PRVS, PRCS, PRIS, PTHS
  ENDDO
ENDDO
!$acc end parallel loop
```
**Purpose:** Compute and apply microphysical tendencies.

### 4. Cloud Fraction (Non-Subgrid) (Lines ~353-362)
```fortran
!$acc parallel loop gang vector collapse(2) private(JIJ, JK)
DO JK=IKTB,IKTE
  DO JIJ=IIJB,IIJE
    IF (PRCS(JIJ,JK) + PRIS(JIJ,JK) > 1.E-12 / PTSTEP) THEN
      PCLDFR(JIJ,JK)  = 1.
    ELSE
      PCLDFR(JIJ,JK)  = 0.
    ENDIF
  ENDDO
ENDDO
!$acc end parallel loop
```
**Purpose:** Simple all-or-nothing cloud fraction diagnostic.

### 5. Subgrid Condensation (Lines ~373-464)
```fortran
!$acc parallel loop gang vector collapse(2) &
!$acc&  private(JIJ, JK, ZW1, ZW2, ZCRIAUT, ZHCF, ZHR)
DO JK=IKTB,IKTE
  DO JIJ=IIJB,IIJE
    ! Complex subgrid condensation with autoconversion
    ! Handles NONE, TRIANGLE, and BIGA PDF options
  ENDDO
ENDDO
!$acc end parallel loop
```
**Purpose:** Most computationally intensive section - subgrid-scale condensation with autoconversion diagnostics.

### 6. Specific Heat Calculation (ITERATION subroutine, Lines ~501-524)
```fortran
!$acc parallel loop gang vector collapse(2) private(JIJ, JK)
DO JK=IKTB,IKTE
  DO JIJ=IIJB,IIJE
    SELECT CASE(KRR)
      CASE(7)
        ZCPH(JIJ,JK) = CST%XCPD + CST%XCPV * PRV_IN(JIJ,JK) + ...
      ! Other cases
    END SELECT
  ENDDO
ENDDO
!$acc end parallel loop
```
**Purpose:** Compute moist air specific heat for different moisture variable configurations.

---

## Dependencies Requiring OpenACC

Based on the dependency analysis, the following files will also need OpenACC directives for a complete GPU implementation:

### Critical Path (Required)
1. **condensation.F90** - Called by ice_adjust, core condensation logic
2. **mode_tiwmx.F90** - Saturation function lookups (frequently called)
3. **mode_qsatmx_tab.F90** - Saturation mixing ratio table
4. **mode_icecloud.F90** - Ice cloud fraction calculations

### Data Structures (GPU Memory Management)
5. **modd_rain_ice_paramn.F90** - Microphysical parameters (copy to GPU constant memory)
6. **modd_tiwmx.F90** - Saturation tables (pre-load on GPU)
7. **modd_nebn.F90** - Cloud parameters
8. **modd_param_icen.F90** - Ice microphysics parameters
9. **modd_cst.F90** - Physical constants (GPU constant memory)

---

## Compilation Requirements

### Required Compiler Flags

For **NVIDIA HPC SDK** (formerly PGI):
```bash
nvfortran -acc -Minfo=accel -gpu=cc80,cuda12.0 ice_adjust_acc.F90
```

Key flags:
- `-acc`: Enable OpenACC
- `-Minfo=accel`: Show what gets accelerated
- `-gpu=cc80`: Target GPU compute capability (adjust for your GPU)
- `-gpu=cuda12.0`: CUDA version

For **GNU GCC** (requires GCC 10+ with OpenACC support):
```bash
gfortran -fopenacc -foffload=nvptx-none ice_adjust_acc.F90
```

### Recommended Additional Flags

```bash
# For NVIDIA HPC:
-fast              # Aggressive optimizations
-Mcuda             # Enable CUDA Fortran features
-gpu=managed       # Use CUDA Unified Memory
-gpu=lineinfo      # Line info for profiling

# For debugging:
-g -Minfo=accel -traceback
```

---

## Next Steps: GPU-Enabled Cython Wrapper

### Architecture Overview

```
Python/Cython
    ↓
Cython Wrapper (ice3_gpu.pyx)
    ↓
OpenACC Fortran (ice_adjust_acc.F90)
    ↓
GPU Execution
```

### Implementation Strategy

#### Option 1: Direct OpenACC from Cython (Recommended)

**File:** `src/ice3/cython/ice_adjust_gpu.pyx`

```python
# cython: language_level=3
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

cdef extern from "openacc.h":
    void* acc_malloc(size_t)
    void acc_free(void*)
    void acc_memcpy_to_device(void*, void*, size_t)
    void acc_memcpy_from_device(void*, void*, size_t)

cdef extern void ice_adjust(
    # Fortran interface - matches ice_adjust_acc.F90 signature
    double* pth, double* prv, double* prc, double* pri,
    double* prvs, double* prcs, double* pris, double* pths,
    double* pcldfr,
    int* nijt, int* nkt,
    # ... other parameters
) nogil

def ice_adjust_gpu(
    cnp.ndarray[cnp.float64_t, ndim=3] th,
    cnp.ndarray[cnp.float64_t, ndim=3] rv,
    cnp.ndarray[cnp.float64_t, ndim=3] rc,
    cnp.ndarray[cnp.float64_t, ndim=3] ri,
    # ... other inputs
):
    """GPU-accelerated ICE_ADJUST via OpenACC."""

    cdef int ngpblks = th.shape[0]
    cdef int nproma = th.shape[1]
    cdef int nflevg = th.shape[2]
    cdef int nijt = ngpblks * nproma
    cdef int nkt = nflevg

    # Arrays are automatically managed by OpenACC
    # when Fortran has !$acc data directives

    with nogil:
        ice_adjust(
            <double*>th.data,
            <double*>rv.data,
            <double*>rc.data,
            <double*>ri.data,
            # ... pass all arrays
            &nijt, &nkt
        )

    return # results are in-place modifications
```

**Advantages:**
- OpenACC manages GPU memory automatically
- Minimal Python/Cython overhead
- Direct Fortran-Cython interface

#### Option 2: CuPy for GPU Arrays

**File:** `src/ice3/cupy/ice_adjust_cupy.py`

```python
import cupy as cp
import numpy as np
from ice3.cython import ice_adjust_fortran  # Compiled wrapper

def ice_adjust_gpu(th_gpu, rv_gpu, rc_gpu, ri_gpu, ...):
    """
    GPU-accelerated ICE_ADJUST using CuPy arrays.

    Parameters
    ----------
    th_gpu : cupy.ndarray
        Potential temperature on GPU
    rv_gpu : cupy.ndarray
        Water vapor mixing ratio on GPU
    ...
    """
    # Get raw GPU pointers
    th_ptr = th_gpu.data.ptr
    rv_ptr = rv_gpu.data.ptr

    # Call Fortran OpenACC code
    # (requires custom CFFI/ctypes interface)
    ice_adjust_fortran.call_gpu(
        th_ptr, rv_ptr, rc_ptr, ri_ptr, ...
    )

    return th_gpu, rv_gpu, rc_gpu, ri_gpu  # Modified on GPU
```

**Advantages:**
- Seamless NumPy-like API for GPU
- Easy integration with other GPU Python libraries
- Can keep data on GPU between calls

---

## Build System Integration

### setup.py Modifications

```python
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Detect OpenACC compiler
import shutil
if shutil.which('nvfortran'):
    fortran_compiler = 'nvfortran'
    acc_flags = ['-acc', '-gpu=managed', '-Minfo=accel']
elif shutil.which('gfortran'):
    fortran_compiler = 'gfortran'
    acc_flags = ['-fopenacc', '-foffload=nvptx-none']
else:
    raise RuntimeError("No OpenACC-capable compiler found")

# Build Fortran library with OpenACC
fortran_sources = [
    'PHYEX-IAL_CY50T1/micro/ice_adjust_acc.F90',
    'PHYEX-IAL_CY50T1/micro/condensation.F90',  # Will need OpenACC too
    # ... other dependencies
]

# Cython extension
extensions = [
    Extension(
        "ice3.cython.ice_adjust_gpu",
        sources=["src/ice3/cython/ice_adjust_gpu.pyx"] + fortran_sources,
        include_dirs=[np.get_include()],
        extra_compile_args=acc_flags,
        extra_link_args=acc_flags,
        language="c",
    )
]

setup(
    name="ice3-gpu",
    ext_modules=cythonize(extensions),
    # ... other setup parameters
)
```

---

## Testing GPU Implementation

### 1. Verify OpenACC Acceleration

```bash
# Compile with info
nvfortran -acc -Minfo=accel ice_adjust_acc.F90

# Expected output should show:
# ice_adjust_acc.f90:
#     265, Generating present(PTH(:,:),PEXN(:,:),...)
#         Generating NVIDIA GPU code
#     267, Loop is parallelizable
#          Generating Gang, Vector(128)
```

### 2. Profile GPU Execution

```bash
# NVIDIA Nsight Systems
nsys profile --stats=true python test_ice_adjust_gpu.py

# NVIDIA Nsight Compute (kernel-level)
ncu --set full python test_ice_adjust_gpu.py
```

### 3. Validation Test

```python
import numpy as np
from ice3.jax.ice_adjust import IceAdjustJAX  # CPU reference
from ice3.cython.ice_adjust_gpu import ice_adjust_gpu

# Create test data
shape = (100, 32, 60)  # ngpblks, nproma, nflevg
th = np.random.uniform(280, 300, shape).astype(np.float64)
rv = np.random.uniform(0.001, 0.015, shape).astype(np.float64)
# ... other inputs

# CPU reference
th_cpu = th.copy()
result_cpu = ice_adjust_jax(th=th_cpu, rv=rv, ...)

# GPU OpenACC
th_gpu = th.copy()
result_gpu = ice_adjust_gpu(th=th_gpu, rv=rv, ...)

# Compare
np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-6)
print("✓ GPU results match CPU reference")
```

---

## Performance Expectations

### Typical Speedups (NVIDIA A100 vs Intel Xeon)

- **Small domains** (< 1000 grid points): 2-5x
- **Medium domains** (10k - 100k points): 10-30x
- **Large domains** (> 1M points): 50-100x

Speedup depends on:
1. Domain size (larger = better GPU utilization)
2. Memory bandwidth (data transfer overhead)
3. Arithmetic intensity (compute vs memory access)

### Optimization Tips

1. **Minimize CPU-GPU transfers**: Keep data on GPU across multiple time steps
2. **Batch operations**: Process multiple columns/time steps together
3. **Use GPU profiler**: Identify bottlenecks with `nsys` or `ncu`
4. **Tune gang/vector sizes**: Experiment with `gang(N)` and `vector(M)` clauses

---

## Known Limitations & Future Work

### Current Limitations

1. **CONDENSATION subroutine**: Not yet ported to OpenACC (critical dependency)
2. **Helper modules**: mode_tiwmx, mode_icecloud need OpenACC directives
3. **I/O and budget calls**: Left on CPU (not critical path)
4. **Derived types**: OpenACC support varies by compiler version

### Recommended Next Steps

1. ✅ **ice_adjust_acc.F90** - COMPLETED
2. ⬜ **Add OpenACC to condensation.F90** - HIGH PRIORITY
3. ⬜ **Add OpenACC to mode_tiwmx.F90** - MEDIUM PRIORITY
4. ⬜ **Add OpenACC to mode_icecloud.F90** - MEDIUM PRIORITY
5. ⬜ **Create Cython GPU wrapper** - HIGH PRIORITY
6. ⬜ **Benchmark and optimize** - FINAL STEP

---

## Additional Resources

### OpenACC Documentation
- [OpenACC Specification](https://www.openacc.org/specification)
- [NVIDIA OpenACC Best Practices](https://docs.nvidia.com/hpc-sdk/compilers/openacc-gs/)
- [OpenACC Fortran Examples](https://github.com/OpenACC/openacc-examples)

### MesoNH GPU Work
- [MesoNH OpenACC branches](https://src.koda.cnrs.fr/benoit.vie/mesonh-code/-/tree/MESONH-v55-OpenACC) (if accessible)
- VIE Benoît's OpenACC work on atmospheric models

### Tools
- [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) (includes nvfortran compiler)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems) (profiling)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute) (kernel analysis)

---

## Contact & Support

For questions about this implementation:
- Check existing JAX implementation in `src/ice3/jax/` for reference logic
- Review Cython bridge examples in `PHYEX-IAL_CY50T1/bridge/`
- Consult MesoNH documentation for physics details

**Generated with Claude Code** - December 20, 2025
