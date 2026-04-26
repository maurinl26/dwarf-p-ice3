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
