"""
Zero-copy GPU bridge: JAX arrays <-> PHYEX Fortran OpenACC kernels.

Architecture (ZERO_COPY_GPU_BRIDGE.md):
  JAX XLA buffer
    -> jax.dlpack.to_dlpack  (zero-copy view, no PCIe)
    -> cp.from_dlpack
    -> cp.copyto(fixed_buf)  (D2D on-device copy, ~0.1 ms on A100)
    -> PHYEX OpenACC kernel  (in-place on fixed_buf)
    -> jax.dlpack.from_dlpack(fixed_buf)  (zero-copy back to JAX)

jax.experimental.io_callback(ordered=True) ensures the callback runs
after upstream JAX ops complete and before downstream ops start.

No PCIe host<->device traffic at runtime.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# IceAdjustJAXGPU
# ---------------------------------------------------------------------------

class IceAdjustJAXGPU:
    """
    Zero-copy JAX bridge for ICE_ADJUST Fortran OpenACC kernel.

    Fixed CuPy staging buffers are allocated once at __init__ so CUDA graphs
    can be captured against their stable device pointers. Each __call__:
      1. Stages JAX inputs into fixed buffers via DLPack + cp.copyto (D2D).
      2. Runs the Fortran GPU kernel (in-place).
      3. Returns modified buffers as JAX arrays via DLPack (zero-copy).

    Output interface matches IceAdjustJAX so AromePhysicsOrchestrator can
    swap backends without changes:
      t_out, rv, rc, ri, cldfr,
      hlc_hrc, hlc_hcf, hli_hri, hli_hcf, cph, lv, ls,  <- zeros (unused)
      rvs, rcs, ris, ths
    """

    def __init__(self, nit: int, nkt: int, krr: int = 6, timestep: float = 60.0):
        import cupy as cp
        from ice3._phyex_wrapper_acc import IceAdjustGPU

        self._gpu = IceAdjustGPU(krr=krr, timestep=timestep)
        self._nit = nit
        self._nkt = nkt

        # Fixed-address device buffers — allocated once, stable pointers
        def _2d(): return cp.zeros((nit, nkt), dtype=cp.float32)
        def _1d(): return cp.zeros(nit, dtype=cp.float32)

        self._b = {
            "sigqsat":   _1d(), "pabs": _2d(), "sigs": _2d(),
            "th":        _2d(), "exn":  _2d(), "exn_ref": _2d(), "rho_dry_ref": _2d(),
            "rv":        _2d(), "rc":   _2d(), "ri": _2d(),
            "rr":        _2d(), "rs":   _2d(), "rg": _2d(),
            "cf_mf":     _2d(), "rc_mf": _2d(), "ri_mf": _2d(),
            "rvs":       _2d(), "rcs":  _2d(), "ris": _2d(), "ths": _2d(),
            "cldfr":     _2d(), "icldfr": _2d(), "wcldfr": _2d(),
            "_zero":     _2d(),   # shared zero for unused diagnostics
        }

    # ----- private callback (runs inside JAX execution) ----------------------

    def _run(self, sigqsat, pabs, sigs, th, exn, exn_ref, rho_dry_ref,
             rv, rc, ri, rr, rs, rg, cf_mf, rc_mf, ri_mf,
             rvs, rcs, ris, ths):
        import cupy as cp

        # Stage JAX GPU arrays into fixed CuPy buffers (D2D, no PCIe)
        field_map = [
            ("sigqsat", sigqsat), ("pabs", pabs), ("sigs", sigs),
            ("th", th), ("exn", exn), ("exn_ref", exn_ref), ("rho_dry_ref", rho_dry_ref),
            ("rv", rv), ("rc", rc), ("ri", ri),
            ("rr", rr), ("rs", rs), ("rg", rg),
            ("cf_mf", cf_mf), ("rc_mf", rc_mf), ("ri_mf", ri_mf),
            ("rvs", rvs), ("rcs", rcs), ("ris", ris), ("ths", ths),
        ]
        for key, arr in field_map:
            cp.copyto(self._b[key], cp.from_dlpack(jax.dlpack.to_dlpack(arr)))

        # Reset cloud-fraction output buffers before kernel
        self._b["cldfr"].fill(0.0)
        self._b["icldfr"].fill(0.0)
        self._b["wcldfr"].fill(0.0)

        b = self._b
        self._gpu(
            b["sigqsat"], b["pabs"],  b["sigs"],  b["th"],   b["exn"],
            b["exn_ref"], b["rho_dry_ref"],
            b["rv"],      b["rc"],    b["ri"],    b["rr"],   b["rs"],   b["rg"],
            b["cf_mf"],   b["rc_mf"], b["ri_mf"],
            b["rvs"],     b["rcs"],   b["ris"],   b["ths"],
            b["cldfr"],   b["icldfr"], b["wcldfr"],
        )

        # t_out = th_modified * exn (Fortran adjusts pth in-place)
        t_buf = b["th"] * b["exn"]
        z = b["_zero"]

        def _j(arr): return jax.dlpack.from_dlpack(arr.toDlpack())

        return (
            _j(t_buf),          # t_out   (temperature K)
            _j(b["rv"]),        # rv_out
            _j(b["rc"]),        # rc_out
            _j(b["ri"]),        # ri_out
            _j(b["cldfr"]),     # cldfr
            _j(z), _j(z), _j(z), _j(z), _j(z), _j(z), _j(z),  # hlc*, hli*, cph, lv, ls
            _j(b["rvs"]),       # rvs_out
            _j(b["rcs"]),       # rcs_out
            _j(b["ris"]),       # ris_out
            _j(b["ths"]),       # ths_out
        )

    # ----- public interface (matches IceAdjustJAX) ---------------------------

    def __call__(
        self,
        sigqsat, pabs, sigs, th, exn, exn_ref, rho_dry_ref,
        rv, rc, ri, rr, rs, rg, cf_mf, rc_mf, ri_mf,
        rvs, rcs, ris, ths,
        timestep=None,
    ):
        _fdt = pabs.dtype
        nit, nkt = pabs.shape
        _s = lambda: jax.ShapeDtypeStruct((nit, nkt), _fdt)

        out_shapes = (
            _s(), _s(), _s(), _s(), _s(),   # t, rv, rc, ri, cldfr
            _s(), _s(), _s(), _s(), _s(), _s(), _s(),  # unused diagnostics (zeros)
            _s(), _s(), _s(), _s(),          # rvs, rcs, ris, ths
        )

        return jax.experimental.io_callback(
            self._run,
            out_shapes,
            sigqsat, pabs, sigs, th, exn, exn_ref, rho_dry_ref,
            rv, rc, ri, rr, rs, rg, cf_mf, rc_mf, ri_mf,
            rvs, rcs, ris, ths,
            ordered=True,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_ice_adjust(
    nit: Optional[int] = None,
    nkt: Optional[int] = None,
    krr: int = 6,
    timestep: float = 60.0,
):
    """
    Return the best available IceAdjust backend.

    Priority:
      1. IceAdjustJAXGPU  — Fortran OpenACC + zero-copy DLPack bridge
                            (requires _phyex_wrapper_acc + NVIDIA GPU)
      2. IceAdjustJAX     — pure JAX stencil (works on CPU and GPU via XLA)

    Parameters
    ----------
    nit, nkt  : domain dimensions — required for GPU buffer pre-allocation.
                If None, GPU path is skipped.
    """
    from ice3.jax.ice_adjust import IceAdjustJAX

    if nit is not None and nkt is not None:
        try:
            import cupy as cp
            if not cp.cuda.is_available():
                raise RuntimeError("no CUDA GPU")
            return IceAdjustJAXGPU(nit=nit, nkt=nkt, krr=krr, timestep=timestep)
        except (ImportError, RuntimeError):
            pass

    return IceAdjustJAX(jit=True)


# ---------------------------------------------------------------------------
# RainIceJAXGPU
# ---------------------------------------------------------------------------

class RainIceJAXGPU:
    """
    Zero-copy JAX bridge for RAIN_ICE Fortran OpenACC kernel.

    Fixed CuPy staging buffers are allocated once at ``__init__`` so CUDA
    graphs can be captured against their stable device pointers.  Each call:
      1. Stages JAX inputs into fixed buffers via DLPack + cp.copyto (D2D).
      2. Runs the Fortran GPU kernel in-place.
      3. Returns modified hydrometeors as JAX arrays via DLPack (zero-copy).

    Output interface matches ``RainIceJAX.__call__``:
      (state_out: Dict[str, Array], diag: Dict[str, Array])

    where ``state_out`` contains the 7 updated hydrometeor keys:
      ``th_t, rv_t, rc_t, rr_t, ri_t, rst, rg_t``
    and ``diag`` contains:
      ``inprc, inprr, inprs, inprg``

    Parameters
    ----------
    nit       : number of horizontal columns
    nkt       : number of vertical levels
    krr       : number of hydrometeor species (default 6 = ICE3)
    timestep  : model timestep in seconds
    device_idx: which CuPy device to allocate on (for per-device pmap usage)
    """

    def __init__(
        self,
        nit: int,
        nkt: int,
        krr: int = 6,
        timestep: float = 60.0,
        device_idx: int = 0,
    ):
        import cupy as cp
        from ice3._phyex_wrapper_acc import RainIceGPU

        with cp.cuda.Device(device_idx):
            self._gpu = RainIceGPU(krr=krr, timestep=timestep)
            self._nit = nit
            self._nkt = nkt
            self._krr = krr
            self._device_idx = device_idx

            def _2d():
                return cp.zeros((nit, nkt), dtype=cp.float32)

            # ---- State / tendency buffers (staged from JAX each call) ----
            self._b_exn      = _2d()
            self._b_dzz      = _2d()
            self._b_rhodj    = _2d()
            self._b_rhodref  = _2d()
            self._b_exnref   = _2d()
            self._b_pabs     = _2d()
            self._b_cit      = _2d()
            self._b_cldfr    = _2d()
            self._b_icldfr   = _2d()
            self._b_tht      = _2d()
            self._b_rvt      = _2d()
            self._b_rct      = _2d()
            self._b_rrt      = _2d()
            self._b_rit      = _2d()
            self._b_rst      = _2d()
            self._b_rgt      = _2d()
            self._b_ths      = _2d()
            self._b_rvs      = _2d()
            self._b_rcs      = _2d()
            self._b_rrs      = _2d()
            self._b_ris      = _2d()
            self._b_rss      = _2d()
            self._b_rgs      = _2d()

            # ---- Permanent zero buffers (mask/diagnostic arrays unused here) ----
            self._b_zero     = _2d()   # shared zero for: ssio, ssiu, ifr,
                                       # hlc_hrc, hlc_hcf, hli_hri, hli_hcf,
                                       # sea, town, conc3d, rht, rhs
            self._b_inprc    = _2d()
            self._b_inprr    = _2d()
            self._b_evap3d   = _2d()
            self._b_inprs    = _2d()
            self._b_inprg    = _2d()
            self._b_indep    = _2d()
            self._b_rainfr   = _2d()
            self._b_inprh    = _2d()
            self._b_fpr      = cp.zeros((nit, nkt, krr), dtype=cp.float32)

    # -------------------------------------------------------------------------
    # Private callback (runs inside jax.experimental.io_callback)
    # -------------------------------------------------------------------------

    def _run(
        self,
        exn, dzz, rhodj, rhodref, exnref, pabs,
        cit, cldfr, icldfr,
        tht, rvt, rct, rrt, rit, rst, rgt,
        ths, rvs, rcs, rrs, ris, rss, rgs,
    ):
        import cupy as cp

        def _stage(buf, jax_arr):
            """D2D copy: JAX GPU buffer → fixed CuPy buffer."""
            cp.copyto(buf, cp.from_dlpack(jax.dlpack.to_dlpack(jax_arr)))

        def _jax(arr):
            """Zero-copy DLPack view: CuPy → JAX."""
            return jax.dlpack.from_dlpack(arr.toDlpack())

        with cp.cuda.Device(self._device_idx):
            # Stage inputs
            _stage(self._b_exn,     exn)
            _stage(self._b_dzz,     dzz)
            _stage(self._b_rhodj,   rhodj)
            _stage(self._b_rhodref, rhodref)
            _stage(self._b_exnref,  exnref)
            _stage(self._b_pabs,    pabs)
            _stage(self._b_cit,     cit)
            _stage(self._b_cldfr,   cldfr)
            _stage(self._b_icldfr,  icldfr)
            _stage(self._b_tht,     tht)
            _stage(self._b_rvt,     rvt)
            _stage(self._b_rct,     rct)
            _stage(self._b_rrt,     rrt)
            _stage(self._b_rit,     rit)
            _stage(self._b_rst,     rst)
            _stage(self._b_rgt,     rgt)
            _stage(self._b_ths,     ths)
            _stage(self._b_rvs,     rvs)
            _stage(self._b_rcs,     rcs)
            _stage(self._b_rrs,     rrs)
            _stage(self._b_ris,     ris)
            _stage(self._b_rss,     rss)
            _stage(self._b_rgs,     rgs)

            # Reset diagnostic output buffers
            self._b_inprc.fill(0.0)
            self._b_inprr.fill(0.0)
            self._b_evap3d.fill(0.0)
            self._b_inprs.fill(0.0)
            self._b_inprg.fill(0.0)
            self._b_indep.fill(0.0)
            self._b_rainfr.fill(0.0)
            self._b_inprh.fill(0.0)
            self._b_fpr.fill(0.0)
            z = self._b_zero

            self._gpu(
                self._b_exn, self._b_dzz, self._b_rhodj, self._b_rhodref,
                self._b_exnref, self._b_pabs,
                self._b_cit, self._b_cldfr, self._b_icldfr,
                z, z, z,                        # ssio, ssiu, ifr
                z, z, z, z,                     # hlc_hrc, hlc_hcf, hli_hri, hli_hcf
                self._b_tht, self._b_rvt, self._b_rct, self._b_rrt,
                self._b_rit, self._b_rst, self._b_rgt,
                self._b_ths, self._b_rvs, self._b_rcs, self._b_rrs,
                self._b_ris, self._b_rss, self._b_rgs,
                self._b_inprc, self._b_inprr, self._b_evap3d,
                self._b_inprs, self._b_inprg, self._b_indep,
                self._b_rainfr, z,              # sigs (zero — not used)
                z, z, z,                        # sea, town, conc3d
                z, z,                           # rht, rhs
                self._b_inprh, self._b_fpr,
            )

            return (
                # Updated hydrometeors
                _jax(self._b_tht * self._b_exn),  # t_out = th * exn
                _jax(self._b_tht),   # th_t
                _jax(self._b_rvt),   # rv_t
                _jax(self._b_rct),   # rc_t
                _jax(self._b_rrt),   # rr_t
                _jax(self._b_rit),   # ri_t
                _jax(self._b_rst),   # rs_t
                _jax(self._b_rgt),   # rg_t
                # Precipitation diagnostics
                _jax(self._b_inprc),
                _jax(self._b_inprr),
                _jax(self._b_inprs),
                _jax(self._b_inprg),
            )

    # -------------------------------------------------------------------------
    # Public interface — matches RainIceJAX.__call__ signature
    # -------------------------------------------------------------------------

    def __call__(
        self,
        state: dict,
        dt: float,
        **kwargs,
    ):
        """
        Execute one RAIN_ICE GPU step via io_callback.

        Parameters
        ----------
        state : dict
            Must contain keys: exn, dzz, rhodref, pres, th_t, rv_t, rc_t,
            rr_t, ri_t, rs_t, rg_t, ci_t, rcs, rrs, ris, rss, rgs.
        dt : float
            Time step (seconds) — passed as static to io_callback.

        Returns
        -------
        (state_out, diag) matching RainIceJAX return convention.
        """
        _fdt = state["pres"].dtype
        nit, nkt = state["pres"].shape
        _s = lambda: jax.ShapeDtypeStruct((nit, nkt), _fdt)

        out_shapes = (
            _s(),   # t_out (K)
            _s(),   # th_t
            _s(),   # rv_t
            _s(),   # rc_t
            _s(),   # rr_t
            _s(),   # ri_t
            _s(),   # rs_t
            _s(),   # rg_t
            _s(),   # inprc
            _s(),   # inprr
            _s(),   # inprs
            _s(),   # inprg
        )

        rhodj = state.get("rhodj", state["rhodref"])   # fallback if absent
        cit   = state.get("ci_t",  jnp.zeros((nit, nkt), dtype=_fdt))
        cldfr = state.get("cldfr", jnp.zeros((nit, nkt), dtype=_fdt))
        icldfr= state.get("icldfr",jnp.zeros((nit, nkt), dtype=_fdt))
        ths   = state.get("ths",   jnp.zeros((nit, nkt), dtype=_fdt))
        rvs   = state.get("rvs",   jnp.zeros((nit, nkt), dtype=_fdt))
        rcs   = state.get("rcs",   jnp.zeros((nit, nkt), dtype=_fdt))
        rrs   = state.get("rrs",   jnp.zeros((nit, nkt), dtype=_fdt))
        ris   = state.get("ris",   jnp.zeros((nit, nkt), dtype=_fdt))
        rss   = state.get("rss",   jnp.zeros((nit, nkt), dtype=_fdt))
        rgs   = state.get("rgs",   jnp.zeros((nit, nkt), dtype=_fdt))

        (t_out, th_t, rv_t, rc_t, rr_t, ri_t, rs_t, rg_t,
         inprc, inprr, inprs, inprg) = jax.experimental.io_callback(
            self._run,
            out_shapes,
            state["exn"], state["dzz"], rhodj, state["rhodref"],
            state["exn"], state["pres"],
            cit, cldfr, icldfr,
            state["th_t"], state["rv_t"], state["rc_t"], state["rr_t"],
            state["ri_t"], state["rs_t"], state["rg_t"],
            ths, rvs, rcs, rrs, ris, rss, rgs,
            ordered=True,
        )

        state_out = {
            **state,
            "th_t": th_t,
            "rv_t": rv_t,
            "rc_t": rc_t,
            "rr_t": rr_t,
            "ri_t": ri_t,
            "rs_t": rs_t,
            "rg_t": rg_t,
        }
        diag = {
            "inprc": inprc,
            "inprr": inprr,
            "inprs": inprs,
            "inprg": inprg,
        }
        return state_out, diag


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_rain_ice(
    nit: Optional[int] = None,
    nkt: Optional[int] = None,
    krr: int = 6,
    timestep: float = 60.0,
    device_idx: int = 0,
):
    """
    Return the best available RainIce backend.

    Priority:
      1. RainIceJAXGPU — Fortran OpenACC + zero-copy DLPack bridge
                         (requires _phyex_wrapper_acc + NVIDIA GPU)
      2. RainIceJAX    — pure JAX stencil (runs on CPU and GPU via XLA)

    Parameters
    ----------
    nit, nkt    : domain dimensions — required for GPU buffer pre-allocation.
                  If None, GPU path is skipped.
    device_idx  : CuPy device index (for per-device pmap usage).
    """
    if nit is not None and nkt is not None:
        try:
            import cupy as cp
            if not cp.cuda.is_available():
                raise RuntimeError("no CUDA GPU")
            return RainIceJAXGPU(
                nit=nit, nkt=nkt, krr=krr,
                timestep=timestep, device_idx=device_idx,
            )
        except (ImportError, RuntimeError):
            pass

    from ice3.jax.rain_ice import RainIceJAX
    # RainIceJAX requires a constants dict; return a lambda so callers can
    # still do make_rain_ice()(state, dt) after passing constants separately.
    return None  # Caller must fall back to RainIceJAX(constants=...)

