"""
Scientific validation of IceAdjustJAXGPU vs IceAdjustJAX (CPU).

Tests four atmospheric profiles:
  standard   — midlatitude clear-sky
  cloudy     — boundary-layer cloud layer 1-4 km
  tropical   — warm moist column, saturated BL
  cold_polar — cold dry column, ice in upper levels

All tests skip cleanly without CUDA + OpenACC wrapper.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="JAX not installed")
import jax.numpy as jnp

try:
    import cupy as cp
    HAS_CUPY = cp.cuda.is_available()
except ImportError:
    cp = None
    HAS_CUPY = False

try:
    from ice3._phyex_wrapper_acc import IceAdjustGPU as _IceAdjustGPU
    HAS_ACC_WRAPPER = True
except ImportError:
    _IceAdjustGPU = None
    HAS_ACC_WRAPPER = False

requires_acc = pytest.mark.skipif(
    not (HAS_CUPY and HAS_ACC_WRAPPER),
    reason="_phyex_wrapper_acc not built (cmake -DENABLE_OPENACC=ON)",
)

# ---------------------------------------------------------------------------
# Per-profile tolerances
# ---------------------------------------------------------------------------

PROFILE_TOL = {
    "standard": {
        "t_atol": 0.05,
        "rv_atol": 1e-4,
        "rc_atol": 1e-5,
        "cldfr_atol": 1e-3,
        "qt_conservation_atol": 1e-4,
    },
    "cloudy": {
        "t_atol": 0.5,
        "rv_atol": 1e-4,
        "rc_atol": 1e-5,
        "cldfr_atol": 5e-2,
        "qt_conservation_atol": 1e-4,
    },
    "tropical": {
        "t_atol": 0.5,
        "rv_atol": 1e-4,
        "rc_atol": 1e-5,
        "cldfr_atol": 5e-2,
        "qt_conservation_atol": 1e-4,
    },
    "cold_polar": {
        "t_atol": 0.05,
        "rv_atol": 1e-4,
        "rc_atol": 1e-5,
        "cldfr_atol": 1e-3,
        "qt_conservation_atol": 1e-4,
    },
}


# ---------------------------------------------------------------------------
# Atmospheric profile generators
# ---------------------------------------------------------------------------

def _base_atm(nit: int, nkt: int, T0: float, p0: float, rv_scale: float, rv_h: float):
    """Shared boilerplate for all profiles."""
    np.random.seed(42)
    z = np.linspace(0, 10000, nkt).astype("f4")
    g = 9.81
    Rd, cpd, p00 = 287.0, 1004.0, 1e5

    # Hydrostatic pressure and temperature
    p = (p0 * (1.0 - 0.0065 * z / T0) ** 5.26).astype("f4")
    T = (T0 - 0.0065 * z).astype("f4")

    exn = (p / p00) ** (Rd / cpd)
    th = T / exn
    rv_prof = (rv_scale * np.exp(-z / rv_h)).astype("f4")
    rho = p / (Rd * T)

    def tile(v):
        return np.tile(v, (nit, 1)).astype("f4")

    j = lambda a: jnp.asarray(a, dtype=jnp.float32)

    return dict(
        sigqsat=j(np.full((nit, nkt), 0.02, "f4")),
        pabs=j(tile(p)),
        sigs=j(np.full((nit, nkt), 0.1, "f4")),
        th=j(tile(th)),
        exn=j(tile(exn)),
        exn_ref=j(tile(exn)),
        rho_dry_ref=j(tile(rho)),
        rv=j(tile(rv_prof)),
        rc=j(np.zeros((nit, nkt), "f4")),
        ri=j(np.zeros((nit, nkt), "f4")),
        rr=j(np.zeros((nit, nkt), "f4")),
        rs=j(np.zeros((nit, nkt), "f4")),
        rg=j(np.zeros((nit, nkt), "f4")),
        cf_mf=j(np.zeros((nit, nkt), "f4")),
        rc_mf=j(np.zeros((nit, nkt), "f4")),
        ri_mf=j(np.zeros((nit, nkt), "f4")),
        rvs=j(np.zeros((nit, nkt), "f4")),
        rcs=j(np.zeros((nit, nkt), "f4")),
        ris=j(np.zeros((nit, nkt), "f4")),
        ths=j(np.zeros((nit, nkt), "f4")),
    )


def _make_atm_profiles(nit: int = 64, nkt: int = 30) -> dict[str, dict]:
    """Return four physically distinct atmospheric profiles."""
    profiles = {}

    # Standard — midlatitude clear-sky (same as _make_atm_jax)
    profiles["standard"] = _base_atm(nit, nkt, T0=288.15, p0=101325.0,
                                      rv_scale=0.012, rv_h=2000.0)

    # Cloudy — boundary-layer cloud layer between 1 and 4 km
    p = _base_atm(nit, nkt, T0=288.15, p0=101325.0, rv_scale=0.012, rv_h=2000.0)
    z = np.linspace(0, 10000, nkt).astype("f4")
    rc_prof = (3e-4 * np.exp(-z / 2000.0)).astype("f4")
    cloud_mask = ((z >= 1000.0) & (z <= 4000.0)).astype("f4")
    rc_field = np.tile(rc_prof * cloud_mask, (nit, 1)).astype("f4")
    p["rc"] = jnp.asarray(rc_field, dtype=jnp.float32)
    p["sigs"] = jnp.full((nit, nkt), 0.2, dtype=jnp.float32)
    profiles["cloudy"] = p

    # Tropical — warm moist column, saturated boundary layer
    profiles["tropical"] = _base_atm(nit, nkt, T0=300.0, p0=101325.0,
                                      rv_scale=0.020, rv_h=3000.0)

    # Cold polar — cold dry column, ice in upper levels
    polar = _base_atm(nit, nkt, T0=250.0, p0=97000.0,
                      rv_scale=0.002, rv_h=1000.0)
    z = np.linspace(0, 10000, nkt).astype("f4")
    ri_prof = (5e-5 * np.exp(-z / 3000.0)).astype("f4")
    upper_mask = (z >= 5000.0).astype("f4")
    ri_field = np.tile(ri_prof * upper_mask, (nit, 1)).astype("f4")
    polar["ri"] = jnp.asarray(ri_field, dtype=jnp.float32)
    profiles["cold_polar"] = polar

    return profiles


# Parametrize at collection time so profile names appear in test IDs
_PROFILES = list(_make_atm_profiles(nit=4, nkt=10).keys())


# ---------------------------------------------------------------------------
# Validation test class
# ---------------------------------------------------------------------------

class TestPhyexProfileValidation:
    """
    Compare IceAdjustJAXGPU (OpenACC Fortran) vs IceAdjustJAX (pure JAX CPU)
    across four atmospheric profiles.

    All tests require the OpenACC wrapper; they skip cleanly otherwise.
    """

    @classmethod
    def _run_both(cls, profile_name: str, nit: int = 64, nkt: int = 30):
        """Return (cpu_out, gpu_out) for the given profile."""
        from ice3.jax.ice_adjust import IceAdjustJAX
        from ice3.jax.phyex_jax_gpu import make_ice_adjust

        profiles = _make_atm_profiles(nit=nit, nkt=nkt)
        atm = profiles[profile_name]

        cpu_adj = IceAdjustJAX(jit=True)
        gpu_adj = make_ice_adjust(n_cols=nit, n_levs=nkt)

        cpu_out = cpu_adj(timestep=60.0, **atm)
        gpu_out = gpu_adj(timestep=60.0, **atm)
        return cpu_out, gpu_out

    # --- Physical invariants on GPU output ---

    @requires_acc
    @pytest.mark.parametrize("profile", _PROFILES)
    def test_temperature_finite(self, profile):
        _, gpu_out = self._run_both(profile)
        t_out = np.array(gpu_out[0])
        assert np.all(np.isfinite(t_out)), f"[{profile}] t_out contains NaN/Inf"

    @requires_acc
    @pytest.mark.parametrize("profile", _PROFILES)
    def test_temperature_bounds(self, profile):
        _, gpu_out = self._run_both(profile)
        t_out = np.array(gpu_out[0])
        assert np.all(t_out > 100.0), f"[{profile}] t_out below 100 K"
        assert np.all(t_out < 600.0), f"[{profile}] t_out above 600 K"

    @requires_acc
    @pytest.mark.parametrize("profile", _PROFILES)
    def test_mixing_ratio_nonneg(self, profile):
        _, gpu_out = self._run_both(profile)
        rv_out = np.array(gpu_out[1])
        rc_out = np.array(gpu_out[2])
        ri_out = np.array(gpu_out[3])
        assert np.all(rv_out >= -1e-7), f"[{profile}] rv_out < 0"
        assert np.all(rc_out >= -1e-7), f"[{profile}] rc_out < 0"
        assert np.all(ri_out >= -1e-7), f"[{profile}] ri_out < 0"

    @requires_acc
    @pytest.mark.parametrize("profile", _PROFILES)
    def test_cloud_fraction_bounds(self, profile):
        _, gpu_out = self._run_both(profile)
        cldfr = np.array(gpu_out[4])
        assert np.all(cldfr >= 0.0), f"[{profile}] cldfr < 0"
        assert np.all(cldfr <= 1.0), f"[{profile}] cldfr > 1"

    @requires_acc
    @pytest.mark.parametrize("profile", _PROFILES)
    def test_water_conservation(self, profile):
        profiles = _make_atm_profiles(nit=64, nkt=30)
        atm = profiles[profile]
        from ice3.jax.phyex_jax_gpu import make_ice_adjust
        gpu_adj = make_ice_adjust(n_cols=64, n_levs=30)
        gpu_out = gpu_adj(timestep=60.0, **atm)

        qt_in = np.array(atm["rv"]) + np.array(atm["rc"]) + np.array(atm["ri"])
        qt_out = np.array(gpu_out[1]) + np.array(gpu_out[2]) + np.array(gpu_out[3])
        tol = PROFILE_TOL[profile]["qt_conservation_atol"]
        delta = np.abs(qt_out - qt_in)
        assert np.all(delta < tol), \
            f"[{profile}] water not conserved, max |Δqt|={delta.max():.2e} > {tol}"

    # --- CPU vs GPU numerical consistency ---

    @requires_acc
    @pytest.mark.parametrize("profile", _PROFILES)
    def test_cpu_gpu_temperature(self, profile):
        cpu_out, gpu_out = self._run_both(profile)
        t_cpu = np.array(cpu_out[0])
        t_gpu = np.array(gpu_out[0])
        tol = PROFILE_TOL[profile]
        np.testing.assert_allclose(
            t_gpu, t_cpu,
            atol=tol["t_atol"], rtol=1e-3,
            err_msg=f"[{profile}] GPU/CPU temperature mismatch",
        )

    @requires_acc
    @pytest.mark.parametrize("profile", _PROFILES)
    def test_cpu_gpu_rv(self, profile):
        cpu_out, gpu_out = self._run_both(profile)
        np.testing.assert_allclose(
            np.array(gpu_out[1]), np.array(cpu_out[1]),
            atol=PROFILE_TOL[profile]["rv_atol"], rtol=1e-2,
            err_msg=f"[{profile}] GPU/CPU rv mismatch",
        )

    @requires_acc
    @pytest.mark.parametrize("profile", _PROFILES)
    def test_cpu_gpu_rc(self, profile):
        cpu_out, gpu_out = self._run_both(profile)
        np.testing.assert_allclose(
            np.array(gpu_out[2]), np.array(cpu_out[2]),
            atol=PROFILE_TOL[profile]["rc_atol"], rtol=1e-2,
            err_msg=f"[{profile}] GPU/CPU rc mismatch",
        )

    @requires_acc
    @pytest.mark.parametrize("profile", _PROFILES)
    def test_cpu_gpu_cldfr(self, profile):
        cpu_out, gpu_out = self._run_both(profile)
        np.testing.assert_allclose(
            np.array(gpu_out[4]), np.array(cpu_out[4]),
            atol=PROFILE_TOL[profile]["cldfr_atol"],
            err_msg=f"[{profile}] GPU/CPU cloud fraction mismatch",
        )

    @requires_acc
    @pytest.mark.parametrize("profile", _PROFILES)
    def test_cpu_gpu_tendencies(self, profile):
        """Source tendencies (rvs, rcs, ris, ths) must agree to 1e-5."""
        cpu_out, gpu_out = self._run_both(profile)
        # Indices: [0]=t [1]=rv [2]=rc [3]=ri [4]=cldfr
        # [5]=hlc_hrc [6]=hlc_hcf [7]=hli_hri [8]=hli_hcf [9]=cph [10]=lv [11]=ls
        # [12]=rvs [13]=rcs [14]=ris [15]=ths
        for idx, name in [(12, "rvs"), (13, "rcs"), (14, "ris"), (15, "ths")]:
            np.testing.assert_allclose(
                np.array(gpu_out[idx]), np.array(cpu_out[idx]),
                atol=1e-5,
                err_msg=f"[{profile}] GPU/CPU {name} mismatch",
            )
