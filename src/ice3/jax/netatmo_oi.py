# -*- coding: utf-8 -*-
"""
Netatmo Near-Surface Data Assimilation — Optimal Interpolation (OI).

Overview
--------
Netatmo personal weather stations provide dense observations of near-surface
(screen-level, z ≈ 1.5–2 m) temperature T2m and relative humidity RH2m at
urban and peri-urban scales. This module implements a column-by-column Optimal
Interpolation (OI) analysis that converts these observations into a corrected
land-surface skin temperature ``t_skin_a`` (K) that is then consumed by the
SURFEX physics driver (see :mod:`ice3.jax.surfex_jax`).

Scientific Basis
----------------

Screen-level Forward Operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Under Monin-Obukhov Similarity Theory (MOST), the screen-level temperature
diagnosed from the surface layer is

.. math::

    T_{2m} = T_{\\rm skin}
             + \\frac{\\theta_a - T_{\\rm skin}}{\\ln(z_a/z_0)} \\,
               \\ln\\!\\left(\\frac{2\\,\\text{m}}{z_0}\\right)

which simplifies to

.. math::

    T_{2m} = T_{\\rm skin} \\,(1 - \\alpha) + \\alpha\\,\\theta_a, \\qquad
    \\alpha = \\frac{\\ln(2\\,\\text{m}/z_0)}{\\ln(z_a/z_0)}

For the neutral limit at :math:`z_a = 10\\,\\text{m}` and
:math:`z_0 = 0.1\\,\\text{m}` (vegetation):

.. math::

    \\alpha = \\frac{\\ln 20}{\\ln 100} \\approx 0.65, \\qquad
    H = \\frac{\\partial T_{2m}}{\\partial T_{\\rm skin}} = 1 - \\alpha \\approx 0.35

For sea and lake tiles (:math:`z_0 \\approx 10^{-5}\\,\\text{m}`),
:math:`\\alpha \\to 1` and :math:`H \\to 0`, meaning Netatmo T2m over water
carries no information about the skin temperature. The default
:math:`\\alpha = 0.65` (land) is a conservative choice.

Optimal Interpolation Formulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The analysis increment for skin temperature is

.. math::

    \\Delta T_{\\rm skin}
    = K \\,(T_{2m}^{\\rm obs} - H \\,T_{\\rm skin}^{\\rm b} - (1-H)\\,\\theta_a)

where the OI gain is

.. math::

    K = \\frac{H \\sigma_b^2}{H^2 \\sigma_b^2 + \\sigma_o^2}

and

* :math:`\\sigma_b` — background error standard deviation for skin temperature
  (default 2 K; set higher over the sea where the model SST is poorly known)
* :math:`\\sigma_o` — Netatmo observation error standard deviation (default 1.5 K,
  reflecting calibration uncertainty and representativity error)
* :math:`H` — linearised forward operator ≈ 0.35 for land tiles

The analysed skin temperature is

.. math::

    T_{\\rm skin}^a = T_{\\rm skin}^b + \\Delta T_{\\rm skin}

clamped to a physically plausible range [220 K, 340 K].

GPU Execution
^^^^^^^^^^^^^
:class:`NetatmoOI` is a **pure JAX** implementation — all operations are XLA
primitives. It therefore runs natively on GPU, requires no host-device data
transfer, and is fully compatible with :func:`jax.jit` and :func:`jax.pmap`.
The assimilation step is fused into the XLA computation graph alongside the
SURFEX surface physics step.

Integration in the physics pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    Netatmo obs  (t2m, q2m, valid)          ← from network ingest
         |
    NetatmoOI.__call__()                    ← pure JAX, GPU kernel
         | t_skin_a  (n_cols,) float32      ← stays on GPU, zero copy
         |
    SurfexState.t_skin = t_skin_a
         |
    SurfexJAXGPU / SurfexJAX               ← OpenACC/cffi Fortran
         | Corrected surface fluxes
         |
    AromePhysicsOrchestrator.step()

References
----------
Mahfouf, J.-F. & Bouttier, F. (2002).
  *Surface analyses at ECMWF.* ECMWF Technical Memorandum No. 377.
Decker, M. et al. (2022).
  *Assimilation of Netatmo citizen-weather-station data in a high-resolution
  NWP system.* Mon. Wea. Rev., 150(8), 2073–2091.
Brousseau, P. et al. (2016).
  *Diurnal cycling of the AROME-France convective-scale surface analysis.*
  Q.J.R. Meteorol. Soc., 142, 2827–2838.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Observation container
# ---------------------------------------------------------------------------

class NetatmoObs(NamedTuple):
    """
    Netatmo near-surface observations, shape ``(n_cols,)`` per field.

    Each column corresponds to one atmospheric column.  When multiple
    Netatmo stations fall inside the same column, the driver must pre-process
    them (e.g. thinning / averaging) before constructing this object.

    Attributes
    ----------
    t2m : Array
        2-metre temperature observations (K).  Unused where ``valid == 0``.
    q2m : Array
        2-metre specific humidity observations (kg/kg).  Unused where
        ``valid == 0`` and currently not assimilated in skin-T OI
        (reserved for future moisture analysis).
    valid : Array
        Integer validity flag per column (dtype int32).
        ``1`` — observation available and quality-controlled;
        ``0`` — no observation or observation rejected by QC.
    sigma_o : Array or float, optional
        Per-column observation error standard deviation (K).
        Scalar or shape ``(n_cols,)``.  When scalar, broadcasts to all columns.
        Defaults to the :class:`NetatmoOI` class-level ``sigma_o`` if ``None``.
    """
    t2m:    Array
    q2m:    Array
    valid:  Array
    sigma_o: Array = jnp.float32(1.5)   # per-column, or scalar broadcast


# ---------------------------------------------------------------------------
# Analysis class
# ---------------------------------------------------------------------------

class NetatmoOI:
    """
    Column-parallel Optimal Interpolation of SURFEX skin temperature from
    Netatmo 2-metre temperature observations.

    The analysis is a **pure JAX** operation — all arithmetic is XLA-traceable
    and the object is transparent to :func:`jax.jit` and :func:`jax.pmap`.

    Parameters
    ----------
    sigma_b : float
        Background error standard deviation for skin temperature (K).
        Typical value 2 K for land; should be increased to ≥ 5 K over sea
        where SST is prescribed and poorly corrected.
    H : float
        Linearised forward-operator coefficient
        :math:`H = \\partial T_{2m} / \\partial T_{\\rm skin}`.
        Default 0.35, derived from MOST for vegetation at :math:`z_a = 10` m
        and :math:`z_0 = 0.1` m.
    t_skin_min, t_skin_max : float
        Physical bounds applied after analysis (K).  Prevent runaway values in
        areas with no background or spurious observations.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from ice3.jax.netatmo_oi import NetatmoObs, NetatmoOI
    >>> n = 4
    >>> obs = NetatmoObs(
    ...     t2m=jnp.array([290., 288., 0., 295.]),
    ...     q2m=jnp.zeros(n),
    ...     valid=jnp.array([1, 1, 0, 1], dtype=jnp.int32),
    ... )
    >>> oi = NetatmoOI(sigma_b=2.0)
    >>> t_skin_bg = jnp.full(n, 285., dtype=jnp.float32)
    >>> t_a       = jnp.full(n, 284., dtype=jnp.float32)
    >>> t_skin_a  = oi(t_skin_bg, t_a, obs)
    """

    def __init__(
        self,
        sigma_b: float = 2.0,
        H: float = 0.35,
        t_skin_min: float = 220.0,
        t_skin_max: float = 340.0,
    ) -> None:
        self.sigma_b    = float(sigma_b)
        self.H          = float(H)
        self.t_skin_min = float(t_skin_min)
        self.t_skin_max = float(t_skin_max)

    # ------------------------------------------------------------------
    # Forward operator
    # ------------------------------------------------------------------

    def diagnose_t2m(self, t_skin: Array, t_a: Array) -> Array:
        """
        Linearised screen-level temperature diagnostic.

        .. math::

            T_{2m}^{\\rm bg} = H \\cdot T_{\\rm skin} + (1 - H) \\cdot \\theta_a

        Parameters
        ----------
        t_skin : shape (n_cols,)
            Background skin temperature (K).
        t_a : shape (n_cols,)
            Lowest-model-level air temperature used as a proxy for potential
            temperature at the reference height (K).

        Returns
        -------
        Array, shape (n_cols,)  — diagnosed T2m (K).
        """
        return self.H * t_skin + (1.0 - self.H) * t_a

    # ------------------------------------------------------------------
    # OI gain
    # ------------------------------------------------------------------

    def _gain(self, sigma_o: Array) -> Array:
        """
        Optimal Interpolation gain K for each column.

        .. math::

            K = \\frac{H \\sigma_b^2}{H^2 \\sigma_b^2 + \\sigma_o^2}

        Parameters
        ----------
        sigma_o : scalar or shape (n_cols,)
            Per-column observation error standard deviation (K).
        """
        H, sb2 = self.H, self.sigma_b ** 2
        return H * sb2 / (H * H * sb2 + sigma_o ** 2)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def __call__(
        self,
        t_skin_bg: Array,
        t_a: Array,
        obs: NetatmoObs,
    ) -> Array:
        """
        Apply OI analysis to produce the corrected skin temperature.

        Parameters
        ----------
        t_skin_bg : Array, shape (n_cols,), float32
            Background skin temperature from the tile-type climatology or the
            previous time step (K).
        t_a : Array, shape (n_cols,), float32
            Lowest atmospheric level temperature — used in the forward operator
            to diagnose T2m from the model background (K).
        obs : NetatmoObs
            Netatmo 2-m observations for the current time step.

        Returns
        -------
        t_skin_a : Array, shape (n_cols,), float32
            Analysed skin temperature (K).  Columns without valid observations
            (``obs.valid == 0``) are returned unchanged from ``t_skin_bg``.

        Notes
        -----
        The analysis is a pure function of its inputs — no internal state is
        mutated.  It is safe to call inside :func:`jax.jit` and
        :func:`jax.pmap`.
        """
        _fdt = t_skin_bg.dtype

        # Diagnosed background T2m
        t2m_bg = self.diagnose_t2m(t_skin_bg, t_a).astype(_fdt)

        # Innovation: observation − background (screen-level)
        innovation = (obs.t2m - t2m_bg).astype(_fdt)

        # OI gain (may be per-column if sigma_o is an array)
        sigma_o = jnp.asarray(obs.sigma_o, dtype=_fdt)
        K = jnp.asarray(self._gain(sigma_o), dtype=_fdt)

        # Analysis increment (zero where obs invalid)
        mask = (obs.valid != 0).astype(_fdt)
        delta = mask * K * innovation

        # Analysed skin temperature, clamped to physical range
        t_skin_a = jnp.clip(
            t_skin_bg + delta,
            a_min=jnp.asarray(self.t_skin_min, dtype=_fdt),
            a_max=jnp.asarray(self.t_skin_max, dtype=_fdt),
        )
        return t_skin_a


# ---------------------------------------------------------------------------
# Helper: tile-based background skin temperature
# ---------------------------------------------------------------------------

#: Default skin temperature per tile type (K) — used when no prognostic
#: surface temperature is available.  Matches the Fortran constants in
#: ``surfex_c_api_acc.F90`` and ``surfex_c_api.F90``.
TILE_T_SKIN_DEFAULT = {
    1: 285.0,  # ISBA / land
    2: 290.0,  # SEAFLUX / open ocean
    3: 287.0,  # FLAKE / freshwater lake
}


def make_t_skin_background(
    tile_type: "np.ndarray",   # int32, shape (n_cols,)
    dtype=jnp.float32,
) -> Array:
    """
    Build a per-column background skin temperature from tile-type classification.

    Uses the same defaults as the Fortran shim so that, in the absence of any
    observations, the OI analysis leaves the skin temperature unchanged.

    Parameters
    ----------
    tile_type : np.ndarray of int32, shape (n_cols,)
        Tile classification (1 = NATURE, 2 = SEA, 3 = LAKE).
    dtype : JAX dtype
        Output dtype (default float32).

    Returns
    -------
    Array, shape (n_cols,)
    """
    import numpy as np

    t_skin = np.empty(len(tile_type), dtype=np.float32)
    for tile_id, ts in TILE_T_SKIN_DEFAULT.items():
        t_skin[tile_type == tile_id] = ts
    t_skin[~((tile_type == 1) | (tile_type == 2) | (tile_type == 3))] = 285.0
    return jnp.asarray(t_skin, dtype=dtype)
