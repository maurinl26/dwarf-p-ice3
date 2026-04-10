# -*- coding: utf-8 -*-
"""
Shared constants, profile builders and field containers for PMAP tests.

Importable from any test file via:
    from cold_night_profile import ColdNightFields, CST, FakeConfig, ...

``tests/utils/`` is listed in ``pythonpath`` in pyproject.toml, so this
module is always findable regardless of ``--import-mode``.
"""
from __future__ import annotations

import numpy as np


# ── Physical constants ───────────────────────────────────────────────────────

class PhysicalConstants:
    Rd       = 287.05       # J kg⁻¹ K⁻¹  dry-air gas constant
    Rv       = 461.5        # J kg⁻¹ K⁻¹  water-vapour gas constant
    cpd      = 1004.0       # J kg⁻¹ K⁻¹  specific heat at constant pressure
    p0       = 100_000.0    # Pa           reference pressure
    t0       = 288.0        # K            reference temperature
    gravity0 = 9.80665      # m s⁻²
    t_melt   = 273.16       # K
    latheat_vap  = 2.501e6  # J kg⁻¹
    latheat_sub  = 2.8345e6 # J kg⁻¹
    latheat_melt = 3.337e5  # J kg⁻¹
    Rd_cpd   = Rd / cpd     # ≈ 0.2859

CST = PhysicalConstants()


# ── Grid sizes ───────────────────────────────────────────────────────────────

DX = DY = DZ = 100.0   # m  — spatial resolution

# Small grid: bridge / adapter unit tests
NX_SM, NY_SM, NZ_SM = 8, 8, 20      # 0.8 km × 0.8 km × 2 km

# Standard grid: physics driver tests
NX, NY, NZ = 16, 16, 50             # 1.6 km × 1.6 km × 5 km


# ── Atmospheric profile builders ─────────────────────────────────────────────

def _z_centers(nz: int, dz: float = DZ) -> np.ndarray:
    """Height of vertical level centres: z_k = (k + 0.5) * dz."""
    return (np.arange(nz) + 0.5) * dz


def _z_full_levels(nz: int, dz: float = DZ) -> np.ndarray:
    """Height of full levels (cell tops): z_k = (k + 1) * dz."""
    return (np.arange(nz) + 1.0) * dz


def _exner(z: np.ndarray, p_surface: float = 95_000.0) -> np.ndarray:
    """Hydrostatic Exner function.

    p(z) = p_surface * exp(−g z / (Rd T̄))   T̄ = 274 K (cold night mean)
    π(z) = (p(z) / p₀)^(Rd/cpd)
    """
    T_mean = 274.0
    p = p_surface * np.exp(-CST.gravity0 * z / (CST.Rd * T_mean))
    return (p / CST.p0) ** CST.Rd_cpd


def _theta_cold_night(z: np.ndarray) -> np.ndarray:
    """Potential temperature profile for a frost night.

    Layer 1 (0–500 m): strong inversion, dθ/dz = +16 K/km
      θ(0) = 274.5 K → θ(500) = 282.5 K
    Layer 2 (500 m–5 km): stable free troposphere, dθ/dz = +4 K/km
    """
    return np.where(
        z <= 500.0,
        274.5 + 16.0 * z / 1000.0,
        282.5 +  4.0 * (z - 500.0) / 1000.0,
    )


def _qvsat_tetens(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Saturation specific humidity via Tetens formula (liquid water)."""
    es = 611.2 * np.exp(17.67 * (T - CST.t_melt) / (T - 29.65))
    es = np.clip(es, 0.0, p * 0.99)
    return 0.622 * es / (p - es)


def _cold_night_profiles(nz: int = NZ, dz: float = DZ):
    """Return (theta, exner, qv, density, pressure) 1-D arrays for nz levels."""
    z   = _z_centers(nz, dz)
    exn = _exner(z)
    theta = _theta_cold_night(z)
    T   = theta * exn
    p   = CST.p0 * exn ** (CST.cpd / CST.Rd)
    qvsat = _qvsat_tetens(T, p)
    qv  = 0.90 * qvsat
    rho = p / (CST.Rd * T * (1.0 + (CST.Rv / CST.Rd - 1.0) * qv))
    return theta, exn, qv, rho, p


# ── Field factory helpers ─────────────────────────────────────────────────────

def _field_3d(arr_3d: np.ndarray) -> dict:
    return {"covering": arr_3d.copy()}


def _field_2d(arr_2d: np.ndarray) -> dict:
    return {"covering": arr_2d.copy()}


def _uniform_3d(nx: int, ny: int, nz: int, value: float,
                dtype=np.float32) -> dict:
    return _field_3d(np.full((nx, ny, nz), value, dtype=dtype))


def _uniform_2d(nx: int, ny: int, value: float, dtype=np.float32) -> dict:
    return _field_2d(np.full((nx, ny), value, dtype=dtype))


def _broadcast_profile(profile_1d: np.ndarray, nx: int, ny: int) -> dict:
    """Broadcast a (nz,) profile to (nx, ny, nz) field."""
    arr = np.broadcast_to(profile_1d[None, None, :], (nx, ny, len(profile_1d)))
    return _field_3d(arr.astype(np.float32))


# ── Field containers ──────────────────────────────────────────────────────────

class ColdNightFields:
    """Minimal field container for cold-night tests at 100 m resolution."""

    def __init__(self, nx: int = NX, ny: int = NY, nz: int = NZ,
                 dz: float = DZ):
        self.nx, self.ny, self.nz = nx, ny, nz
        theta_1d, exn_1d, qv_1d, rho_1d, p_1d = _cold_night_profiles(nz, dz)

        self.theta_total       = _broadcast_profile(theta_1d, nx, ny)
        self.exner_total       = _broadcast_profile(exn_1d,   nx, ny)
        self.exner_ambient     = _broadcast_profile(exn_1d,   nx, ny)
        self.theta_ambient     = _broadcast_profile(theta_1d, nx, ny)
        self.rvapour_ambient   = _broadcast_profile(qv_1d,    nx, ny)
        self.density           = _broadcast_profile(rho_1d,   nx, ny)

        self.frc_theta_perturbation = _uniform_3d(nx, ny, nz, 0.0)

        self.rvapour  = _broadcast_profile(qv_1d, nx, ny)
        self.rliquid  = _uniform_3d(nx, ny, nz, 0.0)
        self.rrain    = _uniform_3d(nx, ny, nz, 0.0)
        self.rice     = _uniform_3d(nx, ny, nz, 0.0)
        self.rsnow    = _uniform_3d(nx, ny, nz, 0.0)
        self.rgraupel = _uniform_3d(nx, ny, nz, 0.0)
        self.cloud_fraction = _uniform_3d(nx, ny, nz, 0.0)

        self.uvel = [
            _uniform_3d(nx, ny, nz,  2.0),
            _uniform_3d(nx, ny, nz,  0.0),
            _uniform_3d(nx, ny, nz,  0.0),
        ]

        self.tke = _uniform_3d(nx, ny, nz, 0.01)

        self.surface_sensible_heat_flux = _uniform_2d(nx, ny, -0.017)
        self.surface_latent_heat_flux   = _uniform_2d(nx, ny, -1.7e-6)
        self.surface_precipitation      = _uniform_2d(nx, ny,  0.0)


class SmallColdNightFields(ColdNightFields):
    """Same as ColdNightFields but on the small 8×8×20 grid (adapter tests)."""

    def __init__(self):
        super().__init__(nx=NX_SM, ny=NY_SM, nz=NZ_SM)


# ── Fake config objects ───────────────────────────────────────────────────────

class FakeConstants:
    Rd          = CST.Rd
    Rv          = CST.Rv
    cpd         = CST.cpd
    p0          = CST.p0
    t0          = CST.t0
    latheat_vap = CST.latheat_vap
    latheat_sub = CST.latheat_sub
    latheat_melt= CST.latheat_melt
    t_melt      = CST.t_melt
    gravity0    = CST.gravity0


class FakeAromePhysicsConfig:
    enabled             = True
    use_ecrad           = False
    use_surfex          = False
    krr                 = 6
    jit                 = True
    turb_enabled        = True
    shallow_conv_enabled= True
    ice_adjust_enabled  = True
    rain_ice_enabled    = True
    dtype               = "float32"
    use_dlpack          = "auto"
    cd_neutral          = 1.0e-3    # neutral bulk drag coefficient [−]


class FakeAromePhysicsConfigF64(FakeAromePhysicsConfig):
    dtype = "float64"


class FakePhyexConfig:
    enabled = False


class FakeConfig:
    """PMAP-like config mock for physics driver tests."""

    constants       = FakeConstants()
    arome_physics   = FakeAromePhysicsConfig()
    phyex           = FakePhyexConfig()
    dt              = 30.0   # s

    def __init__(self, nx: int = NX, ny: int = NY, nz: int = NZ,
                 dz: float = DZ, dtype: str = "float32"):
        self.arome_physics = FakeAromePhysicsConfig()
        self.arome_physics.dtype = dtype

        z_full = _z_full_levels(nz, dz)
        zcr_3d = np.broadcast_to(
            z_full[None, None, :], (nx, ny, nz)
        ).astype(np.float32)

        class _Coords:
            zcr = {"covering": zcr_3d}

        self.coordinates = _Coords()
