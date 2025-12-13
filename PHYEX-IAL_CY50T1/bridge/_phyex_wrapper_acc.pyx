# distutils: language = c
# cython: language_level = 3
"""
Cython wrapper for PHYEX ICE_ADJUST GPU-accelerated routine.

This module provides a Python/JAX/CuPy interface to the Fortran ICE_ADJUST_ACC
subroutine through a C bridge defined in phyex_bridge_acc.F90.

GPU Acceleration:
- Requires NVIDIA GPU with OpenACC support
- Uses CuPy for zero-copy GPU array management
- Compatible with JAX via DLPack protocol
- Falls back to NumPy/CPU if GPU unavailable

Example usage:
    import cupy as cp
    from ice3.fortran_gpu import IceAdjustGPU

    # Create instance
    ice_adjust_gpu = IceAdjustGPU()

    # Allocate arrays on GPU
    th = cp.random.uniform(250, 300, (1000, 60), dtype=cp.float32)
    rv = cp.random.uniform(0.001, 0.015, (1000, 60), dtype=cp.float32)
    # ... other arrays

    # Run GPU kernel
    ice_adjust_gpu(th, rv, rc, ri, ...)

    # Results are updated in-place on GPU
"""

import numpy as np
cimport numpy as np

# Try to import CuPy for GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False
    print("Warning: CuPy not available - GPU acceleration disabled")

# Declare memory view data types for single precision (JPRB/float32)
ctypedef np.float32_t DTYPE_t

# External C Declaration matching phyex_bridge_acc.F90
cdef extern:
    void c_ice_adjust_acc(
        int nlon,
        int nlev,
        int krr,
        float timestep,
        # GPU device pointers (passed from CuPy)
        float *ptr_sigqsat,
        float *ptr_pabs,
        float *ptr_sigs,
        float *ptr_th,
        float *ptr_exn,
        float *ptr_exn_ref,
        float *ptr_rho_dry_ref,
        float *ptr_rv,
        float *ptr_rc,
        float *ptr_ri,
        float *ptr_rr,
        float *ptr_rs,
        float *ptr_rg,
        float *ptr_cf_mf,
        float *ptr_rc_mf,
        float *ptr_ri_mf,
        # Tendencies (input/output on GPU)
        float *ptr_rvs,
        float *ptr_rcs,
        float *ptr_ris,
        float *ptr_ths,
        # Output on GPU
        float *ptr_cldfr,
        float *ptr_icldfr,
        float *ptr_wcldfr
    ) nogil


class IceAdjustGPU:
    """
    GPU-accelerated ICE_ADJUST wrapper using OpenACC and CuPy.

    This class provides a high-level interface to the Fortran OpenACC
    implementation of ICE_ADJUST, handling GPU memory management and
    data transfer automatically.

    Attributes:
        has_gpu (bool): Whether GPU acceleration is available
        default_krr (int): Default number of hydrometeor species (6 for ICE3)
        default_timestep (float): Default timestep in seconds

    Methods:
        __call__: Execute ice_adjust on GPU arrays
        from_numpy: Convert NumPy arrays to GPU and execute
        to_numpy: Execute and return results as NumPy arrays
    """

    def __init__(self, krr=6, timestep=1.0):
        """
        Initialize GPU ICE_ADJUST wrapper.

        Parameters:
            krr (int): Number of hydrometeor species (default: 6 for ICE3)
            timestep (float): Model timestep in seconds (default: 1.0)
        """
        self.krr = krr
        self.timestep = timestep
        self.has_gpu = HAS_CUPY

        if not self.has_gpu:
            raise RuntimeError(
                "CuPy is required for GPU acceleration. "
                "Install with: pip install cupy-cuda12x"
            )

    def __call__(self,
                 sigqsat,  # 1D: (nlon,)
                 pabs, sigs, th, exn, exn_ref, rho_dry_ref,  # 2D inputs
                 rv, rc, ri, rr, rs, rg,  # 2D hydrometeors
                 cf_mf, rc_mf, ri_mf,  # 2D mass flux
                 rvs, rcs, ris, ths,  # 2D tendencies (input/output)
                 cldfr, icldfr, wcldfr):  # 2D output
        """
        Execute ICE_ADJUST on GPU arrays (CuPy).

        All arrays must be CuPy arrays on GPU with dtype=float32.
        Arrays are modified IN-PLACE on the GPU.

        Parameters:
            sigqsat: (nlon,) - Subgrid saturation variance
            pabs: (nlon, nlev) - Absolute pressure [Pa]
            sigs: (nlon, nlev) - Sigma_s turbulence parameter
            th: (nlon, nlev) - Potential temperature [K]
            exn: (nlon, nlev) - Exner function
            exn_ref: (nlon, nlev) - Reference Exner function
            rho_dry_ref: (nlon, nlev) - Reference dry air density [kg/m³]
            rv: (nlon, nlev) - Water vapor mixing ratio [kg/kg]
            rc: (nlon, nlev) - Cloud water mixing ratio [kg/kg]
            ri: (nlon, nlev) - Cloud ice mixing ratio [kg/kg]
            rr: (nlon, nlev) - Rain mixing ratio [kg/kg]
            rs: (nlon, nlev) - Snow mixing ratio [kg/kg]
            rg: (nlon, nlev) - Graupel mixing ratio [kg/kg]
            cf_mf: (nlon, nlev) - Cloud fraction from mass flux
            rc_mf: (nlon, nlev) - Cloud water from mass flux [kg/kg]
            ri_mf: (nlon, nlev) - Cloud ice from mass flux [kg/kg]
            rvs: (nlon, nlev) - Water vapor source term [kg/kg/s] (IN/OUT)
            rcs: (nlon, nlev) - Cloud water source term [kg/kg/s] (IN/OUT)
            ris: (nlon, nlev) - Cloud ice source term [kg/kg/s] (IN/OUT)
            ths: (nlon, nlev) - Potential temp source term [K/s] (IN/OUT)
            cldfr: (nlon, nlev) - Total cloud fraction (OUTPUT)
            icldfr: (nlon, nlev) - Ice cloud fraction (OUTPUT)
            wcldfr: (nlon, nlev) - Liquid cloud fraction (OUTPUT)

        Returns:
            None (arrays modified in-place)

        Raises:
            ValueError: If array shapes are inconsistent or not on GPU
            TypeError: If arrays are not CuPy arrays
        """
        # Validate inputs
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available")

        # Check all arrays are CuPy arrays
        arrays = [pabs, sigs, th, exn, exn_ref, rho_dry_ref,
                  rv, rc, ri, rr, rs, rg,
                  cf_mf, rc_mf, ri_mf,
                  rvs, rcs, ris, ths,
                  cldfr, icldfr, wcldfr]

        for arr in arrays:
            if not isinstance(arr, cp.ndarray):
                raise TypeError(f"Expected CuPy array, got {type(arr)}")
            if arr.dtype != np.float32:
                raise ValueError(f"Expected float32 dtype, got {arr.dtype}")

        # Get dimensions
        nlon, nlev = pabs.shape

        # Validate 1D array
        if sigqsat.shape != (nlon,):
            raise ValueError(f"sigqsat shape {sigqsat.shape} != ({nlon},)")

        # Validate all 2D arrays have same shape
        for arr in arrays:
            if arr.shape != (nlon, nlev):
                raise ValueError(f"Array shape {arr.shape} != ({nlon}, {nlev})")

        # Ensure arrays are contiguous in memory (required for GPU kernels)
        sigqsat = cp.ascontiguousarray(sigqsat)
        arrays_contig = [cp.ascontiguousarray(arr) for arr in arrays]

        # Extract GPU device pointers
        cdef long ptr_sigqsat_val = sigqsat.data.ptr
        cdef long ptr_pabs_val = arrays_contig[0].data.ptr
        cdef long ptr_sigs_val = arrays_contig[1].data.ptr
        cdef long ptr_th_val = arrays_contig[2].data.ptr
        cdef long ptr_exn_val = arrays_contig[3].data.ptr
        cdef long ptr_exn_ref_val = arrays_contig[4].data.ptr
        cdef long ptr_rho_dry_ref_val = arrays_contig[5].data.ptr
        cdef long ptr_rv_val = arrays_contig[6].data.ptr
        cdef long ptr_rc_val = arrays_contig[7].data.ptr
        cdef long ptr_ri_val = arrays_contig[8].data.ptr
        cdef long ptr_rr_val = arrays_contig[9].data.ptr
        cdef long ptr_rs_val = arrays_contig[10].data.ptr
        cdef long ptr_rg_val = arrays_contig[11].data.ptr
        cdef long ptr_cf_mf_val = arrays_contig[12].data.ptr
        cdef long ptr_rc_mf_val = arrays_contig[13].data.ptr
        cdef long ptr_ri_mf_val = arrays_contig[14].data.ptr
        cdef long ptr_rvs_val = arrays_contig[15].data.ptr
        cdef long ptr_rcs_val = arrays_contig[16].data.ptr
        cdef long ptr_ris_val = arrays_contig[17].data.ptr
        cdef long ptr_ths_val = arrays_contig[18].data.ptr
        cdef long ptr_cldfr_val = arrays_contig[19].data.ptr
        cdef long ptr_icldfr_val = arrays_contig[20].data.ptr
        cdef long ptr_wcldfr_val = arrays_contig[21].data.ptr

        # Call Fortran GPU kernel
        with nogil:
            c_ice_adjust_acc(
                <int>nlon,
                <int>nlev,
                <int>self.krr,
                <float>self.timestep,
                <float*>ptr_sigqsat_val,
                <float*>ptr_pabs_val,
                <float*>ptr_sigs_val,
                <float*>ptr_th_val,
                <float*>ptr_exn_val,
                <float*>ptr_exn_ref_val,
                <float*>ptr_rho_dry_ref_val,
                <float*>ptr_rv_val,
                <float*>ptr_rc_val,
                <float*>ptr_ri_val,
                <float*>ptr_rr_val,
                <float*>ptr_rs_val,
                <float*>ptr_rg_val,
                <float*>ptr_cf_mf_val,
                <float*>ptr_rc_mf_val,
                <float*>ptr_ri_mf_val,
                <float*>ptr_rvs_val,
                <float*>ptr_rcs_val,
                <float*>ptr_ris_val,
                <float*>ptr_ths_val,
                <float*>ptr_cldfr_val,
                <float*>ptr_icldfr_val,
                <float*>ptr_wcldfr_val
            )

        # Results are updated in-place on GPU
        # Copy back to original arrays if they weren't contiguous
        for i, arr_orig in enumerate(arrays):
            if arr_orig.data.ptr != arrays_contig[i].data.ptr:
                arr_orig[:] = arrays_contig[i]

    def from_numpy(self,
                   sigqsat_cpu,
                   pabs_cpu, sigs_cpu, th_cpu, exn_cpu, exn_ref_cpu, rho_dry_ref_cpu,
                   rv_cpu, rc_cpu, ri_cpu, rr_cpu, rs_cpu, rg_cpu,
                   cf_mf_cpu, rc_mf_cpu, ri_mf_cpu,
                   rvs_cpu, rcs_cpu, ris_cpu, ths_cpu,
                   cldfr_cpu, icldfr_cpu, wcldfr_cpu):
        """
        Execute ICE_ADJUST on NumPy arrays (CPU -> GPU -> CPU).

        This method:
        1. Transfers NumPy arrays to GPU (CuPy)
        2. Executes GPU kernel
        3. Transfers results back to CPU (NumPy)

        Use this for small arrays or testing. For production, use CuPy directly.

        Parameters:
            Same as __call__ but accepts NumPy arrays

        Returns:
            tuple: (rvs, rcs, ris, ths, cldfr, icldfr, wcldfr) as NumPy arrays
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available")

        # Transfer to GPU
        sigqsat_gpu = cp.asarray(sigqsat_cpu, dtype=np.float32)
        pabs_gpu = cp.asarray(pabs_cpu, dtype=np.float32)
        sigs_gpu = cp.asarray(sigs_cpu, dtype=np.float32)
        th_gpu = cp.asarray(th_cpu, dtype=np.float32)
        exn_gpu = cp.asarray(exn_cpu, dtype=np.float32)
        exn_ref_gpu = cp.asarray(exn_ref_cpu, dtype=np.float32)
        rho_dry_ref_gpu = cp.asarray(rho_dry_ref_cpu, dtype=np.float32)
        rv_gpu = cp.asarray(rv_cpu, dtype=np.float32)
        rc_gpu = cp.asarray(rc_cpu, dtype=np.float32)
        ri_gpu = cp.asarray(ri_cpu, dtype=np.float32)
        rr_gpu = cp.asarray(rr_cpu, dtype=np.float32)
        rs_gpu = cp.asarray(rs_cpu, dtype=np.float32)
        rg_gpu = cp.asarray(rg_cpu, dtype=np.float32)
        cf_mf_gpu = cp.asarray(cf_mf_cpu, dtype=np.float32)
        rc_mf_gpu = cp.asarray(rc_mf_cpu, dtype=np.float32)
        ri_mf_gpu = cp.asarray(ri_mf_cpu, dtype=np.float32)
        rvs_gpu = cp.asarray(rvs_cpu, dtype=np.float32)
        rcs_gpu = cp.asarray(rcs_cpu, dtype=np.float32)
        ris_gpu = cp.asarray(ris_cpu, dtype=np.float32)
        ths_gpu = cp.asarray(ths_cpu, dtype=np.float32)
        cldfr_gpu = cp.asarray(cldfr_cpu, dtype=np.float32)
        icldfr_gpu = cp.asarray(icldfr_cpu, dtype=np.float32)
        wcldfr_gpu = cp.asarray(wcldfr_cpu, dtype=np.float32)

        # Execute on GPU
        self(sigqsat_gpu,
             pabs_gpu, sigs_gpu, th_gpu, exn_gpu, exn_ref_gpu, rho_dry_ref_gpu,
             rv_gpu, rc_gpu, ri_gpu, rr_gpu, rs_gpu, rg_gpu,
             cf_mf_gpu, rc_mf_gpu, ri_mf_gpu,
             rvs_gpu, rcs_gpu, ris_gpu, ths_gpu,
             cldfr_gpu, icldfr_gpu, wcldfr_gpu)

        # Transfer results back to CPU
        return (cp.asnumpy(rvs_gpu),
                cp.asnumpy(rcs_gpu),
                cp.asnumpy(ris_gpu),
                cp.asnumpy(ths_gpu),
                cp.asnumpy(cldfr_gpu),
                cp.asnumpy(icldfr_gpu),
                cp.asnumpy(wcldfr_gpu))


# JAX integration helper
def ice_adjust_jax_gpu(th, rv, rc, ri, rr, rs, rg, **kwargs):
    """
    JAX-compatible wrapper for GPU ICE_ADJUST.

    This function enables using ICE_ADJUST within JAX pipelines by:
    1. Converting JAX arrays to CuPy (zero-copy via DLPack)
    2. Executing GPU kernel
    3. Converting back to JAX (zero-copy)

    Example:
        import jax.numpy as jnp
        from ice3.fortran_gpu import ice_adjust_jax_gpu

        # JAX arrays on GPU
        th = jnp.ones((1000, 60), dtype=jnp.float32) * 280.0
        rv = jnp.ones((1000, 60), dtype=jnp.float32) * 0.01
        # ... other inputs

        # Execute (zero-copy)
        cldfr, icldfr, wcldfr = ice_adjust_jax_gpu(th, rv, rc, ...)

    Parameters:
        Same as IceAdjustGPU.__call__ but accepts JAX arrays

    Returns:
        tuple: (cldfr, icldfr, wcldfr) as JAX arrays

    Note:
        Requires jax[cuda] and cupy to be installed.
        Not differentiable (Fortran kernel is not autodiff-aware).
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy required for JAX GPU integration")

    try:
        import jax
        import jax.dlpack
    except ImportError:
        raise RuntimeError("JAX required for JAX integration")

    # Convert JAX -> CuPy (zero-copy via DLPack)
    def jax_to_cupy(arr):
        dlpack = jax.dlpack.to_dlpack(arr)
        return cp.from_dlpack(dlpack)

    def cupy_to_jax(arr):
        dlpack = arr.toDlpack()
        return jax.dlpack.from_dlpack(dlpack)

    # Create GPU wrapper instance
    ice_adjust = IceAdjustGPU(krr=kwargs.get('krr', 6),
                               timestep=kwargs.get('timestep', 1.0))

    # Convert inputs to CuPy (zero-copy)
    sigqsat_cp = jax_to_cupy(kwargs['sigqsat'])
    pabs_cp = jax_to_cupy(kwargs['pabs'])
    sigs_cp = jax_to_cupy(kwargs['sigs'])
    th_cp = jax_to_cupy(th)
    exn_cp = jax_to_cupy(kwargs['exn'])
    exn_ref_cp = jax_to_cupy(kwargs['exn_ref'])
    rho_dry_ref_cp = jax_to_cupy(kwargs['rho_dry_ref'])
    rv_cp = jax_to_cupy(rv)
    rc_cp = jax_to_cupy(rc)
    ri_cp = jax_to_cupy(ri)
    rr_cp = jax_to_cupy(rr)
    rs_cp = jax_to_cupy(rs)
    rg_cp = jax_to_cupy(rg)
    cf_mf_cp = jax_to_cupy(kwargs['cf_mf'])
    rc_mf_cp = jax_to_cupy(kwargs['rc_mf'])
    ri_mf_cp = jax_to_cupy(kwargs['ri_mf'])

    # Allocate outputs on GPU
    nlon, nlev = th.shape
    rvs_cp = cp.zeros((nlon, nlev), dtype=cp.float32)
    rcs_cp = cp.zeros((nlon, nlev), dtype=cp.float32)
    ris_cp = cp.zeros((nlon, nlev), dtype=cp.float32)
    ths_cp = cp.zeros((nlon, nlev), dtype=cp.float32)
    cldfr_cp = cp.zeros((nlon, nlev), dtype=cp.float32)
    icldfr_cp = cp.zeros((nlon, nlev), dtype=cp.float32)
    wcldfr_cp = cp.zeros((nlon, nlev), dtype=cp.float32)

    # Execute GPU kernel
    ice_adjust(sigqsat_cp,
               pabs_cp, sigs_cp, th_cp, exn_cp, exn_ref_cp, rho_dry_ref_cp,
               rv_cp, rc_cp, ri_cp, rr_cp, rs_cp, rg_cp,
               cf_mf_cp, rc_mf_cp, ri_mf_cp,
               rvs_cp, rcs_cp, ris_cp, ths_cp,
               cldfr_cp, icldfr_cp, wcldfr_cp)

    # Convert outputs back to JAX (zero-copy)
    return (cupy_to_jax(cldfr_cp),
            cupy_to_jax(icldfr_cp),
            cupy_to_jax(wcldfr_cp))
