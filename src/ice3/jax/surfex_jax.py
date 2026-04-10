# -*- coding: utf-8 -*-
"""
SURFEX Offline Tight-Coupling - cffi + jax.pure_callback.

Overview
--------
This module bridges the **Open SURFEX** land-surface model (CPU, compiled
Fortran) with the JAX-based AROME atmospheric physics (GPU) through a zero-copy
in-memory FFI coupling using Python's ``cffi`` and JAX's ``jax.pure_callback``.

Scientific Schemes (Open SURFEX V9.1)
--------------------------------------
Open SURFEX is a mosaic surface model. Each atmospheric column is split into
sub-tiles whose fractional coverage is described by ECOCLIMAP. The active
schemes in the offline configuration are:

**1. ISBA — Interaction Soil–Biosphere–Atmosphere** (Nature tiles)
  The reference soil-vegetation-atmosphere transfer scheme.
  Solves the surface energy balance:

    Rn - H - LE - G = 0

  where Rn (net radiation), H (sensible heat), LE (latent heat) and G (ground
  heat flux) in W m⁻². Soil water is tracked through the multi-layer ISBA-DIF
  diffusion scheme. Vegetation resistance uses A-gs photosynthesis in ISBA-A-gs
  or a simpler Jarvis-type formulation.

  Reference: Noilhan & Mahfouf (1996), Balsamo et al. (2009).

**2. TEB — Town Energy Balance** (Urban tiles)
  A single-canyon geometry model solving coupled energy budgets for roofs,
  walls and road facets. TEB outputs an effective albedo, emissivity, and
  surface heat / momentum fluxes that are homogenised with ISBA outputs.

  Reference: Masson (2000), Lemonsu et al. (2004).

**3. Flake** (Lake tiles)
  A bulk freshwater lake model with a prognostic mixed-layer depth. Provides
  lake surface temperature, evaporation, and sensible heat flux.

  Reference: Mironov et al. (2010).

**4. Sea / Sea-Ice** (Ocean tiles)
  Sea-surface temperature is prescribed (from forcing or an SST analysis).
  A simple sea-ice scheme computes flux corrections for ice-covered columns.
  Surface roughness follows the Charnock (1955) relation over open ocean.

Surface-to-Atmosphere Fluxes (bulk aerodynamic surrogate in the C shim)
------------------------------------------------------------------------
The Fortran shim ``surfex_c_api.F90`` contains a simplified **Bulk Aerodynamic**
approximation that is active until the full SURFEX tile infrastructure is
linked. The momentum, sensible-heat and latent-heat transfer coefficients
follow MOST (Monin–Obukhov Similarity Theory) under the neutral limit:

  C_D = C_H = C_E = (κ / ln(z / z₀))²

  H  / (ρ Cp)  = CH · |U| · (θs − θa)     [K m s⁻¹]
  LE / (ρ Lv)  = CE · |U| · (qs − qa)     [kg kg⁻¹ m s⁻¹]
  τ_u = −CD · |U| · ua                     [m² s⁻²]

where κ = 0.4 (von Kármán), z = 10 m (reference height), z₀ = 0.05 m
(neutral roughness length), θs is a default surface potential temperature
(295 K), qs is the saturation specific humidity at θs.

Architecture
------------
  Layer 1 — Fortran/C API  (``surfex_c_api.F90``)
      ISO_C_BINDING shim exposing ``c_surfex_step`` as a plain C symbol.
      Compiled into ``libsurfex_offline.{so|dylib}`` via ``build_libsurfex.sh``.

  Layer 2 — cffi binding  (``_SurfexLib`` class)
      Loads the shared library as a singleton at process start.
      Uses the CFFI ABI mode: no header generation step needed.
      Routes numpy arrays to/from the library in-memory (zero disk I/O).

  Layer 3 — JAX callback  (``SurfexJAX.__call__``)
      ``jax.pure_callback`` suspends the GPU computation, moves the
      O(n_cols) surface-level arrays GPU → CPU, calls the cffi binding
      synchronously, then returns results as JAX GPU arrays.

Performance Notes
-----------------
* Only the lowest model level is exchanged (O(n_cols) floats), not the full
  3-D state — transfer overhead is negligible at typical LAM resolutions.
* The XLA computation graph is JIT-compiled end-to-end; SURFEX appears as
  an opaque leaf, enabling XLA to optimize surrounding operations.
* JAX async dispatch allows the GPU to continue executing other ops while
  the CPU runs SURFEX (no explicit synchronization needed).

Build Instructions
------------------
1. Compile the shim::

     cd /path/to/open-SURFEX-V9-1-0
     bash build_libsurfex.sh
     # → lib/libsurfex_offline.dylib  (macOS)
     # → lib/libsurfex_offline.so     (Linux)

2. Set the library path::

     export SURFEX_LIB=/path/to/lib/libsurfex_offline.dylib

3. Import — the library loads at first instantiation of ``SurfexJAX``.

Scientific References
---------------------
- Masson, V. et al. (2013). The SURFEXv7.2 land and ocean surface platform.
  *Geosci. Model Dev.*, 6, 929–960. https://doi.org/10.5194/gmd-6-929-2013
- Noilhan, J. & Mahfouf, J.-F. (1996). ISBA land surface scheme.
  *Global Planet. Change*, 13, 145–159.
- Masson, V. (2000). A physically-based scheme for the urban energy balance.
  *Bound.-Layer Meteor.*, 94, 357–397.
- Mironov, D. et al. (2010). Implementation of the lake parameterisation
  scheme FLake. *Boreal Env. Res.*, 15, 178–198.
- Brutsaert, W. (1982). *Evaporation into the Atmosphere*. Reidel, 299 pp.
"""

from __future__ import annotations

import os
import ctypes
from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# C header declaration for cffi
# ---------------------------------------------------------------------------
_CDEF = """
void c_surfex_step(
    int    n_cols,
    double dt,
    double *t_a,
    double *q_a,
    double *u_a,
    double *v_a,
    double *p_a,
    double *rhodref,
    double *sw_down,
    double *lw_down,
    double *rain_rate,
    double *snow_rate,
    double *surf_flux_th,
    double *surf_flux_rv,
    double *surf_flux_u,
    double *surf_flux_v,
    double *albedo,
    double *emissivity
);
"""


def _locate_library() -> Path:
    """
    Resolve the path to `libsurfex_offline.{so,dylib}`.

    Priority:
      1. $SURFEX_LIB environment variable (explicit override)
      2. <project_root>/open-SURFEX-V9-1-0/lib/ (default install location)
    """
    if env := os.environ.get("SURFEX_LIB"):
        p = Path(env)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"SURFEX_LIB={env} does not exist. "
            "Run build_libsurfex.sh first."
        )

    # Auto-detect: walk up from this file looking for open-SURFEX-V9-1-0
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent.parent / "open-SURFEX-V9-1-0" / "lib"
        for suffix in ("dylib", "so"):
            lib = candidate / f"libsurfex_offline.{suffix}"
            if lib.exists():
                return lib

    raise FileNotFoundError(
        "Cannot locate libsurfex_offline.{so,dylib}.\n"
        "Build with:  bash /path/to/open-SURFEX-V9-1-0/build_libsurfex.sh\n"
        "Then set:    export SURFEX_LIB=/path/to/lib/libsurfex_offline.dylib"
    )


class _SurfexLib:
    """
    Lazy cffi binding to the compiled SURFEX shared library.

    The library is loaded once per process; subsequent instantiations
    of `SurfexJAX` share the same `_SurfexLib` handle.
    """
    _instance: "_SurfexLib | None" = None

    def __new__(cls) -> "_SurfexLib":
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._load()
            cls._instance = obj
        return cls._instance

    def _load(self) -> None:
        try:
            from cffi import FFI
            self._ffi = FFI()
            self._ffi.cdef(_CDEF)
            lib_path = _locate_library()
            self._lib = self._ffi.dlopen(str(lib_path))
            self._available = True
            print(f"[SurfexLib] Loaded {lib_path}")
        except (ImportError, FileNotFoundError, OSError) as exc:
            import warnings
            warnings.warn(
                f"[SurfexLib] SURFEX shared library not available: {exc}\n"
                "Falling back to analytical bulk-aerodynamic surrogate.",
                RuntimeWarning,
                stacklevel=3,
            )
            self._available = False

    def call(
        self,
        n_cols: int,
        dt: float,
        t_a: np.ndarray,
        q_a: np.ndarray,
        u_a: np.ndarray,
        v_a: np.ndarray,
        p_a: np.ndarray,
        rhodref: np.ndarray,
        sw_down: np.ndarray,
        lw_down: np.ndarray,
        rain_rate: np.ndarray,
        snow_rate: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """
        Calls `c_surfex_step` in the shared library and returns output arrays.
        All inputs and outputs are 64-bit float numpy arrays of shape (n_cols,).
        """
        if not self._available:
            return _bulk_aerodynamic_fallback(
                n_cols, t_a, q_a, u_a, v_a, p_a, rhodref
            )

        ffi = self._ffi
        lib = self._lib

        def _ptr(arr: np.ndarray):
            return ffi.cast("double *", arr.ctypes.data)

        # Ensure C-contiguous float64
        def _c64(arr):
            return np.ascontiguousarray(arr, dtype=np.float64)

        t_a, q_a, u_a, v_a = _c64(t_a), _c64(q_a), _c64(u_a), _c64(v_a)
        p_a, rhodref       = _c64(p_a),  _c64(rhodref)
        sw_down, lw_down   = _c64(sw_down), _c64(lw_down)
        rain_rate, snow_rate = _c64(rain_rate), _c64(snow_rate)

        surf_flux_th = np.empty(n_cols, dtype=np.float64)
        surf_flux_rv = np.empty(n_cols, dtype=np.float64)
        surf_flux_u  = np.empty(n_cols, dtype=np.float64)
        surf_flux_v  = np.empty(n_cols, dtype=np.float64)
        albedo       = np.empty(n_cols, dtype=np.float64)
        emissivity   = np.empty(n_cols, dtype=np.float64)

        lib.c_surfex_step(
            n_cols, dt,
            _ptr(t_a), _ptr(q_a), _ptr(u_a), _ptr(v_a),
            _ptr(p_a), _ptr(rhodref),
            _ptr(sw_down), _ptr(lw_down),
            _ptr(rain_rate), _ptr(snow_rate),
            _ptr(surf_flux_th), _ptr(surf_flux_rv),
            _ptr(surf_flux_u),  _ptr(surf_flux_v),
            _ptr(albedo), _ptr(emissivity),
        )

        return (
            surf_flux_th.astype(np.float32),
            surf_flux_rv.astype(np.float32),
            surf_flux_u.astype(np.float32),
            surf_flux_v.astype(np.float32),
            albedo.astype(np.float32),
            emissivity.astype(np.float32),
        )


def _bulk_aerodynamic_fallback(
    n_cols, t_a, q_a, u_a, v_a, p_a, rhodref
) -> Tuple[np.ndarray, ...]:
    """
    Neutral-stability Bulk Aerodynamic fluxes as a library-free fallback.

    Based on Monin–Obukhov similarity theory under neutral stratification:
      C_D = (kappa / ln(z/z0))^2
      H   = rho * Cp * C_H * |U| * (Ts - Ta)
      LE  = rho * Lv * C_E * |U| * (qs - qa)
    """
    kappa, z, z0 = 0.4, 10.0, 0.05
    Cp, Ts_default = 1004.0, 295.0
    cd = (kappa / np.log(z / z0)) ** 2
    wspd = np.maximum(np.sqrt(u_a**2 + v_a**2), 0.01)
    theta_a = t_a * (1e5 / p_a) ** (287.05 / Cp)

    flux_th = cd * wspd * (Ts_default - theta_a)
    flux_rv = cd * wspd * np.maximum(0.0, 0.018 - q_a)
    flux_u  = -cd * wspd * u_a
    flux_v  = -cd * wspd * v_a
    alb     = np.full(n_cols, 0.2,  dtype=np.float32)
    emis    = np.full(n_cols, 0.98, dtype=np.float32)

    return (
        flux_th.astype(np.float32), flux_rv.astype(np.float32),
        flux_u.astype(np.float32),  flux_v.astype(np.float32),
        alb, emis,
    )


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

class SurfexState(NamedTuple):
    """
    Atmospheric forcing for the surface scheme.
    All fields have shape ``(n_columns,)`` — lowest atmospheric level only.

    The ``psurf_flux_*`` fields carry the surface fluxes from the previous
    time step (or the driver-prescribed bulk values).  They allow null stubs
    (e.g. ``_NullSurfex``) to forward the existing fluxes unchanged, and let
    the real ``SurfexJAX`` callback know the previous values for continuity.
    """
    t_a: Array           # Air temperature (K)
    q_a: Array           # Specific humidity (kg/kg)
    u_a: Array           # Zonal wind (m/s)
    v_a: Array           # Meridional wind (m/s)
    p_a: Array           # Pressure (Pa)
    rhodref: Array       # Air density (kg/m³)
    sw_down: Array       # Downward shortwave flux from ecRad (W/m²)
    lw_down: Array       # Downward longwave flux from ecRad (W/m²)
    rain_rate: Array     # Liquid precipitation rate (kg/m²/s)
    snow_rate: Array     # Solid precipitation rate (kg/m²/s)
    psurf_flux_th: Array # Previous-step kinematic sensible heat flux (K m/s)
    psurf_flux_rv: Array # Previous-step kinematic moisture flux (kg/kg m/s)
    psurf_flux_u: Array  # Previous-step surface momentum flux U (m²/s²)
    psurf_flux_v: Array  # Previous-step surface momentum flux V (m²/s²)


class SurfexFluxes(NamedTuple):
    """
    Surface fluxes returned by SURFEX to the atmospheric model.
    All fields have shape ``(n_columns,)``.
    """
    surf_flux_th: Array  # Kinematic sensible heat flux (K m/s)
    surf_flux_rv: Array  # Kinematic moisture flux (kg/kg m/s)
    surf_flux_u: Array   # Surface momentum flux — U component (m²/s²)
    surf_flux_v: Array   # Surface momentum flux — V component (m²/s²)
    albedo: Array        # Effective SW albedo
    emissivity: Array    # Effective LW emissivity


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SurfexJAX:
    """
    Tight CPU/GPU coupling for the SURFEX offline surface scheme.

    SURFEX runs on the **CPU** via the compiled `libsurfex_offline` shared
    library.  The JAX physics orchestrator runs on the **GPU**.  Data transfer
    is orchestrated by ``jax.pure_callback``, which:

      1. Copies only the O(n_cols) surface-level arrays from GPU → CPU.
      2. Calls the cffi-bound `c_surfex_step` synchronously on the CPU host.
      3. Copies the returned flux arrays from CPU → GPU.

    The GPU XLA graph remains JIT-compiled; only the surface step is executed
    on the host, making overlapping of radiation/dynamics with surface feasible
    through  JAX async dispatch.

    Usage
    -----
    >>> surfex = SurfexJAX()
    >>> state  = SurfexState(t_a=..., q_a=..., ...)
    >>> fluxes = surfex(state, dt=60.0)
    """

    def __init__(self) -> None:
        self._lib = _SurfexLib()   # singleton — loaded only once

    def _host_fn(
        self,
        t_a, q_a, u_a, v_a, p_a, rhodref,
        sw_down, lw_down, rain_rate, snow_rate,
        dt: float,
    ):
        """
        Host-side function invoked by ``jax.pure_callback``.
        All inputs arrive as **numpy** arrays on the CPU.
        """
        n_cols = t_a.shape[0]
        return self._lib.call(
            n_cols, float(dt),
            t_a, q_a, u_a, v_a, p_a, rhodref,
            sw_down, lw_down, rain_rate, snow_rate,
        )

    def __call__(self, state: SurfexState, dt: float) -> SurfexFluxes:
        """
        Execute a SURFEX time step inside JIT-compiled JAX code.

        The call is fully transparent to ``@jax.jit``; the host callback
        appears as an opaque leaf in the XLA computation graph.

        Parameters
        ----------
        state : SurfexState
            Atmospheric forcing at the lowest model level (GPU arrays).
        dt : float
            Physics time step, seconds.

        Returns
        -------
        SurfexFluxes
            Surface fluxes on the same device as ``state``.
        """
        n_cols = state.t_a.shape[0]

        # Output shape specs — all (n_cols,) float32
        result_shapes = tuple(
            jax.ShapeDtypeStruct((n_cols,), jnp.float32)
            for _ in range(6)
        )

        # Bind dt into the host function via closure (scalar, not a JAX array)
        def _bound_host(*arrays):
            return self._host_fn(*arrays, dt=dt)

        (
            surf_flux_th, surf_flux_rv,
            surf_flux_u,  surf_flux_v,
            albedo, emissivity,
        ) = jax.pure_callback(
            _bound_host,
            result_shapes,
            state.t_a, state.q_a, state.u_a, state.v_a,
            state.p_a, state.rhodref,
            state.sw_down, state.lw_down,
            state.rain_rate, state.snow_rate,
            vmap_method="sequential",
        )

        return SurfexFluxes(
            surf_flux_th=surf_flux_th,
            surf_flux_rv=surf_flux_rv,
            surf_flux_u=surf_flux_u,
            surf_flux_v=surf_flux_v,
            albedo=albedo,
            emissivity=emissivity,
        )
