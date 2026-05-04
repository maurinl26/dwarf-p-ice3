# distutils: language = c
# cython: language_level = 3
"""
Cython bridge for SURFEX GPU-accelerated surface physics with Netatmo OI.

Wraps c_surfex_step_acc (OpenACC Fortran compiled with nvfortran) and
captures a CUDA graph on the first call for subsequent zero-overhead replay.

Netatmo assimilation (t_skin)
------------------------------
An additional float32 device pointer ``ptr_t_skin`` is passed to the Fortran
kernel.  When ``t_skin[col] > 0``, the Fortran uses it as the effective skin
temperature for that column instead of the tile-type climatological default.

The Netatmo OI step (``netatmo_oi.NetatmoOI``) runs in pure JAX BEFORE this
wrapper is called.  ``SurfexJAXGPU`` stages the resulting ``t_skin_a`` array
into ``_buf_t_skin`` via ``cp.copyto`` (D2D, no host bounce), exactly like the
other atmospheric-forcing inputs.

CUDA graph capture strategy
---------------------------
CUDA graphs encode raw device pointer values at capture time, so input/output
CuPy buffers are allocated once at __init__ and never reallocated.  Callers
copy JAX GPU arrays into these buffers via cupy.copyto (D2D, no host bounce)
then launch the pre-captured graph.

The key to capturing OpenACC kernels in a CUDA graph:

  1. Create a cupy.cuda.Stream.
  2. Call acc_set_cuda_stream(1, stream.ptr) — binds OpenACC async-queue 1
     to the stream that will undergo cudaStreamBeginCapture.
  3. Open cupy.cuda.graph.Graph() context (starts cudaStreamBeginCapture
     on the active stream).
  4. Call c_surfex_step_acc — kernels are submitted to queue 1 == the
     capture stream, so they get recorded, NOT executed.
  5. Close the Graph() context (ends capture, instantiates executable graph).
  6. Subsequent calls: graph.launch(stream) replays all kernels atomically.

The graph must be re-captured only when dt changes (dt is baked into some
coefficient computations inside the Fortran; currently the shim ignores dt
so re-capture is never needed, but the flag is kept for future use).

Tile-type constants (must match surfex_c_api_acc.F90)
------------------------------------------------------
TILE_NATURE = 1  (ISBA land)
TILE_SEA    = 2  (SEAFLUX / open ocean)
TILE_LAKE   = 3  (FLAKE / freshwater lake)
"""

import numpy as np
cimport numpy as np

try:
    import cupy as cp
    import cupy.cuda.graph as _cpgraph
    HAS_CUPY = True
except ImportError:
    cp = None
    _cpgraph = None
    HAS_CUPY = False

from libc.stdint cimport uintptr_t

# ---------------------------------------------------------------------------
# Tile-type constants (mirrors surfex_c_api_acc.F90 INTEGER parameters)
# ---------------------------------------------------------------------------
TILE_NATURE = 1
TILE_SEA    = 2
TILE_LAKE   = 3

# ---------------------------------------------------------------------------
# External C declarations
# ---------------------------------------------------------------------------

cdef extern void c_surfex_step_acc(
    int    n_cols,
    double dt,
    # tile flag — GPU device pointer (int32)
    void  *ptr_tile,
    # Netatmo-analysed skin temperature — GPU device pointer (float32)
    # 0.0 signals "use tile-type default" for that column
    void  *ptr_t_skin,
    # atmospheric forcing — GPU device pointers (float32)
    void  *ptr_t_a,
    void  *ptr_q_a,
    void  *ptr_u_a,
    void  *ptr_v_a,
    void  *ptr_p_a,
    void  *ptr_rhodref,
    void  *ptr_sw_down,
    void  *ptr_lw_down,
    void  *ptr_rain_rate,
    void  *ptr_snow_rate,
    # surface fluxes — GPU device pointers (float32, OUTPUT)
    void  *ptr_surf_flux_th,
    void  *ptr_surf_flux_rv,
    void  *ptr_surf_flux_u,
    void  *ptr_surf_flux_v,
    void  *ptr_albedo,
    void  *ptr_emissivity
) nogil

# acc_set_cuda_stream: NVIDIA OpenACC runtime API
# Signature (NVIDIA HPC SDK): int acc_set_cuda_stream(int async, void *stream)
# Binds OpenACC async queue <async> to the CUDA stream <stream>.
cdef extern int acc_set_cuda_stream(int async_queue, void *stream_ptr) nogil


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

cdef class SurfexGPUWrapper:
    """
    Cython wrapper for SURFEX OpenACC surface physics with CUDA graph capture.

    Buffers are allocated once at __init__; pointers remain stable across
    calls, satisfying the CUDA graph requirement.

    Parameters
    ----------
    n_cols : int
        Number of atmospheric columns (fixed for the lifetime of the object).
    tile_type : numpy array of int32, shape (n_cols,)
        Tile classification per column (TILE_NATURE/SEA/LAKE).

    Netatmo t_skin
    --------------
    Pass ``t_skin`` (float32, shape (n_cols,)) on each call.
    Zero values instruct the Fortran to use the tile-type climatological
    default.  Non-zero values (from ``NetatmoOI``) override the default.

    Usage
    -----
    wrapper = SurfexGPUWrapper(n_cols=1024, tile_type=tiles_np)

    # Pass JAX GPU arrays via DLPack (zero-copy D2D)
    out = wrapper(
        t_skin=jax.dlpack.to_dlpack(t_skin_a),   # from NetatmoOI
        t_a=jax.dlpack.to_dlpack(state.t_a),
        q_a=jax.dlpack.to_dlpack(state.q_a),
        ...
        dt=60.0
    )
    surf_flux_th = jax.dlpack.from_dlpack(out['surf_flux_th'])
    """

    cdef int    _n_cols
    cdef double _captured_dt
    cdef bint   _captured

    # Persistent CuPy buffers (stable device pointers for CUDA graph)
    cdef object _buf_tile         # int32  (n_cols,) — constant after init
    cdef object _buf_t_skin       # float32 (n_cols,) — Netatmo skin T (0 = use default)
    cdef object _buf_t_a          # float32 inputs
    cdef object _buf_q_a
    cdef object _buf_u_a
    cdef object _buf_v_a
    cdef object _buf_p_a
    cdef object _buf_rhodref
    cdef object _buf_sw_down
    cdef object _buf_lw_down
    cdef object _buf_rain_rate
    cdef object _buf_snow_rate
    cdef object _buf_surf_flux_th  # float32 outputs
    cdef object _buf_surf_flux_rv
    cdef object _buf_surf_flux_u
    cdef object _buf_surf_flux_v
    cdef object _buf_albedo
    cdef object _buf_emissivity

    cdef object _stream            # cupy.cuda.Stream
    cdef object _graph             # cupy.cuda.graph.Graph (captured)

    def __init__(self, int n_cols, tile_type):
        if not HAS_CUPY:
            raise RuntimeError(
                "CuPy is required for SurfexGPUWrapper. "
                "Install with: pip install cupy-cuda12x"
            )

        self._n_cols      = n_cols
        self._captured    = False
        self._captured_dt = -1.0
        self._graph       = None
        self._stream      = cp.cuda.Stream()

        # Tile-type buffer: constant for the lifetime of this object.
        self._buf_tile = cp.asarray(
            np.asarray(tile_type, dtype=np.int32), dtype=cp.int32
        )

        # Netatmo skin temperature: initialised to zeros (= use tile default).
        self._buf_t_skin = cp.zeros(n_cols, dtype=cp.float32)

        # Input buffers (float32, n_cols)
        for attr in ('_buf_t_a', '_buf_q_a', '_buf_u_a', '_buf_v_a',
                     '_buf_p_a', '_buf_rhodref',
                     '_buf_sw_down', '_buf_lw_down',
                     '_buf_rain_rate', '_buf_snow_rate'):
            setattr(self, attr, cp.empty(n_cols, dtype=cp.float32))

        # Output buffers (float32, n_cols)
        for attr in ('_buf_surf_flux_th', '_buf_surf_flux_rv',
                     '_buf_surf_flux_u',  '_buf_surf_flux_v',
                     '_buf_albedo', '_buf_emissivity'):
            setattr(self, attr, cp.empty(n_cols, dtype=cp.float32))

    # ------------------------------------------------------------------
    # CUDA graph capture (called once, or when dt changes)
    # ------------------------------------------------------------------
    cdef _do_capture(self, double dt):
        """
        Capture all OpenACC kernels into a CUDA graph on self._stream.

        Steps:
          1. Bind OpenACC async-queue 1 to the capture stream.
          2. Enter cupy.cuda.graph.Graph() context (starts capture).
          3. Call c_surfex_step_acc — kernels are recorded, not run.
          4. Exit context (ends capture, graph is instantiated).
        """
        cdef uintptr_t sptr = self._stream.ptr
        cdef void* raw_stream = <void*>sptr

        with self._stream:
            with nogil:
                acc_set_cuda_stream(1, raw_stream)

            with _cpgraph.Graph() as g:
                self._call_fortran(dt)

        self._graph       = g
        self._captured    = True
        self._captured_dt = dt

    # ------------------------------------------------------------------
    # Raw Fortran call (used both inside graph capture and without GPU)
    # ------------------------------------------------------------------
    cdef _call_fortran(self, double dt):
        cdef uintptr_t ptr_tile     = self._buf_tile.data.ptr
        cdef uintptr_t ptr_t_skin   = self._buf_t_skin.data.ptr
        cdef uintptr_t ptr_t_a      = self._buf_t_a.data.ptr
        cdef uintptr_t ptr_q_a      = self._buf_q_a.data.ptr
        cdef uintptr_t ptr_u_a      = self._buf_u_a.data.ptr
        cdef uintptr_t ptr_v_a      = self._buf_v_a.data.ptr
        cdef uintptr_t ptr_p_a      = self._buf_p_a.data.ptr
        cdef uintptr_t ptr_rhodref  = self._buf_rhodref.data.ptr
        cdef uintptr_t ptr_sw_down  = self._buf_sw_down.data.ptr
        cdef uintptr_t ptr_lw_down  = self._buf_lw_down.data.ptr
        cdef uintptr_t ptr_rain     = self._buf_rain_rate.data.ptr
        cdef uintptr_t ptr_snow     = self._buf_snow_rate.data.ptr
        cdef uintptr_t ptr_th       = self._buf_surf_flux_th.data.ptr
        cdef uintptr_t ptr_rv       = self._buf_surf_flux_rv.data.ptr
        cdef uintptr_t ptr_fu       = self._buf_surf_flux_u.data.ptr
        cdef uintptr_t ptr_fv       = self._buf_surf_flux_v.data.ptr
        cdef uintptr_t ptr_alb      = self._buf_albedo.data.ptr
        cdef uintptr_t ptr_emis     = self._buf_emissivity.data.ptr

        with nogil:
            c_surfex_step_acc(
                self._n_cols, dt,
                <void*>ptr_tile,
                <void*>ptr_t_skin,
                <void*>ptr_t_a,    <void*>ptr_q_a,
                <void*>ptr_u_a,    <void*>ptr_v_a,
                <void*>ptr_p_a,    <void*>ptr_rhodref,
                <void*>ptr_sw_down, <void*>ptr_lw_down,
                <void*>ptr_rain,   <void*>ptr_snow,
                <void*>ptr_th,     <void*>ptr_rv,
                <void*>ptr_fu,     <void*>ptr_fv,
                <void*>ptr_alb,    <void*>ptr_emis
            )

    # ------------------------------------------------------------------
    # Public call interface (DLPack in/out, graph replay)
    # ------------------------------------------------------------------
    def __call__(self,
                 t_a, q_a, u_a, v_a, p_a, rhodref,
                 sw_down, lw_down, rain_rate, snow_rate,
                 double dt,
                 t_skin=None):
        """
        Execute SURFEX surface physics on GPU.

        Parameters
        ----------
        t_a … snow_rate : DLPack capsules (JAX GPU arrays, float32, shape (n_cols,))
        dt : float  Physics timestep (seconds).
        t_skin : DLPack capsule, optional
            Netatmo-analysed skin temperature (float32, shape (n_cols,)).
            Zero values use the Fortran tile-type climatological default.
            If None, the buffer is left at its previous value (or zeros on the
            first call).

        Returns
        -------
        dict of DLPack capsules:
            surf_flux_th, surf_flux_rv, surf_flux_u, surf_flux_v,
            albedo, emissivity
        """
        # -- Copy JAX GPU arrays into our stable CuPy buffers (D2D, zero-copy) --
        cp.copyto(self._buf_t_a,      cp.from_dlpack(t_a))
        cp.copyto(self._buf_q_a,      cp.from_dlpack(q_a))
        cp.copyto(self._buf_u_a,      cp.from_dlpack(u_a))
        cp.copyto(self._buf_v_a,      cp.from_dlpack(v_a))
        cp.copyto(self._buf_p_a,      cp.from_dlpack(p_a))
        cp.copyto(self._buf_rhodref,  cp.from_dlpack(rhodref))
        cp.copyto(self._buf_sw_down,  cp.from_dlpack(sw_down))
        cp.copyto(self._buf_lw_down,  cp.from_dlpack(lw_down))
        cp.copyto(self._buf_rain_rate, cp.from_dlpack(rain_rate))
        cp.copyto(self._buf_snow_rate, cp.from_dlpack(snow_rate))

        # -- Netatmo skin temperature (optional; zeros = use Fortran default) --
        if t_skin is not None:
            cp.copyto(self._buf_t_skin, cp.from_dlpack(t_skin))
        # else: leave _buf_t_skin at its current value (zeros on first call)

        # -- (Re)capture graph if needed --
        if not self._captured or dt != self._captured_dt:
            self._do_capture(dt)

        # -- Replay CUDA graph (zero Python overhead for the kernels) --
        self._graph.launch(stream=self._stream)
        self._stream.synchronize()

        return {
            'surf_flux_th': self._buf_surf_flux_th.toDlpack(),
            'surf_flux_rv': self._buf_surf_flux_rv.toDlpack(),
            'surf_flux_u':  self._buf_surf_flux_u.toDlpack(),
            'surf_flux_v':  self._buf_surf_flux_v.toDlpack(),
            'albedo':       self._buf_albedo.toDlpack(),
            'emissivity':   self._buf_emissivity.toDlpack(),
        }

    @property
    def is_graph_captured(self) -> bool:
        return self._captured

    @property
    def n_cols(self) -> int:
        return self._n_cols
