# -*- coding: utf-8 -*-
"""
Standalone functional tests for AromePhysicsOrchestrator.

Mirror of the PMAP-LES-shared test_arome_physics_driver.py but with no PMAP
dependency.  All PMAP infrastructure (GT4Py fields, ProcessDriver, field
adapters) is replaced by direct NumPy → JAX array construction here.

The field mapping exactly follows AromePhysicsDriver.__call__ from PMAP:

    AromeState field      Source formula
    ─────────────────────────────────────────────────────────────────────
    pabst  [Pa]           p = p₀ · exn^(cpd/Rd)
    pzz    [m]            full-level heights (z_k = k * dz, k=1..nz)
    dzz    [m]            Δz_k = zz_k − zz_{k-1}  (uniform → DZ)
    pt     [K]            T = θ · Π
    pth    [K]            θ  (direct)
    pthl   [K]            θ − (Lv/cpd)/Π · rc − (Ls/cpd)/Π · ri
    prv…prg [kg/kg]       direct
    prt    [kg/kg]        rv + rc + rr + ri + rs + rg
    pu,pv,pw [m/s]        direct
    ptke   [m²/s²]        direct
    ptkecls [m²/s²]       tke[:, 0]
    pthvref [K]           θ_amb · (1 + (Rv/Rd − 1) · rv_amb)
    pexn   [−]            exner_total  (direct)
    pexn_ref [−]          exner_ambient  (direct)
    prho_dry_ref [kg/m³]  density  (direct)
    psurf_flux_u [m²/s²]  −Cd · |V₁| · u₁
    psurf_flux_v [m²/s²]  −Cd · |V₁| · v₁
    psurf_flux_th [K·m/s] H / (ρ₁ · cpd)
    psurf_flux_rv [kg/(kg·m/s)]  E / (ρ₁ · Lv)
"""
from __future__ import annotations

import numpy as np
import pytest

# ── Module-level skip if ice3 JAX is not installed ───────────────────────────
pytest.importorskip(
    "ice3.jax.arome_physics",
    reason="dwarf-p-ice3 JAX not installed; skipping standalone physics tests.",
)

pytestmark = pytest.mark.physics

# ── Physical constants (must match ColdNightFields / ice3 defaults) ───────────
_Rd  = 287.05
_Rv  = 461.5
_cpd = 1004.0
_p0  = 100_000.0
_Lv  = 2.501e6
_Ls  = 2.8345e6
_Cd  = 1.0e-3       # neutral bulk drag coefficient


# ─────────────────────────────────────────────────────────────────────────────
# Field-mapping helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to2d(field_dict: dict, nx: int, ny: int) -> np.ndarray:
    """(nx, ny, nz) covering → (nit, nkt) with nit = nx*ny."""
    arr = field_dict["covering"]
    return arr.reshape(nx * ny, arr.shape[2])


def _to1d(field_dict: dict, nx: int, ny: int) -> np.ndarray:
    """(nx, ny) covering → (nit,) with nit = nx*ny."""
    return field_dict["covering"].reshape(nx * ny)


def _build_arome_state(fields, nx: int, ny: int, nz: int, dz: float, dtype):
    """Construct AromeState from ColdNightFields — no PMAP dependency."""
    import jax.numpy as jnp
    from ice3.jax.arome_physics import AromeState

    nit = nx * ny

    def j(arr):
        return jnp.asarray(arr, dtype=dtype)

    # ── 3-D thermodynamic fields ──────────────────────────────────────────
    exn   = j(_to2d(fields.exner_total,   nx, ny))          # (nit, nkt)
    pabst = j(_p0 * (exn ** (_cpd / _Rd)))                  # Pa
    pth   = j(_to2d(fields.theta_total,   nx, ny))          # K
    pt    = pth * exn                                        # K
    prc   = j(_to2d(fields.rliquid,       nx, ny))
    pri   = j(_to2d(fields.rice,          nx, ny))
    # θl = θ − (Lv/cpd)/Π · rc − (Ls/cpd)/Π · ri
    pthl  = pth - (_Lv / _cpd) / exn * prc - (_Ls / _cpd) / exn * pri
    prv   = j(_to2d(fields.rvapour,       nx, ny))
    prr   = j(_to2d(fields.rrain,         nx, ny))
    prs   = j(_to2d(fields.rsnow,         nx, ny))
    prg   = j(_to2d(fields.rgraupel,      nx, ny))
    prt   = prv + prc + prr + pri + prs + prg

    # ── Wind and TKE ──────────────────────────────────────────────────────
    pu    = j(_to2d(fields.uvel[0],       nx, ny))
    pv    = j(_to2d(fields.uvel[1],       nx, ny))
    pw    = j(_to2d(fields.uvel[2],       nx, ny))
    ptke  = j(_to2d(fields.tke,           nx, ny))

    # ── Reference state ───────────────────────────────────────────────────
    rho   = j(_to2d(fields.density,       nx, ny))
    exn_ref = j(_to2d(fields.exner_ambient, nx, ny))
    th_amb  = j(_to2d(fields.theta_ambient, nx, ny))
    rv_amb  = j(_to2d(fields.rvapour_ambient, nx, ny))
    pthvref = th_amb * (1.0 + (_Rv / _Rd - 1.0) * rv_amb)

    # ── Static grid (full-level heights and layer thicknesses) ────────────
    z_full = (np.arange(1, nz + 1) * dz).astype(np.float32)  # (nz,)
    pzz  = jnp.broadcast_to(
        jnp.asarray(z_full, dtype=dtype)[None, :], (nit, nz)
    )
    # dzz: Δz at each level; dzz[k=0] = z[0], dzz[k>0] = z[k] - z[k-1]
    dzz_1d = np.concatenate([[z_full[0]], np.diff(z_full)]).astype(np.float32)
    pdzz = jnp.broadcast_to(
        jnp.asarray(dzz_1d, dtype=dtype)[None, :], (nit, nz)
    )

    # ── Surface fields (nit,) ─────────────────────────────────────────────
    ptkecls = ptke[:, 0]

    rho1  = rho[:, 0]
    u1    = pu[:, 0]
    v1    = pv[:, 0]
    wspd1 = jnp.sqrt(u1 ** 2 + v1 ** 2)
    psurf_flux_u  = -_Cd * wspd1 * u1
    psurf_flux_v  = -_Cd * wspd1 * v1

    shf = j(_to1d(fields.surface_sensible_heat_flux, nx, ny))   # W/m²
    lhf = j(_to1d(fields.surface_latent_heat_flux,   nx, ny))   # W/m²
    psurf_flux_th = shf / (rho1 * _cpd)    # K·m/s
    psurf_flux_rv = lhf / (rho1 * _Lv)    # kg/(kg·m/s)

    return AromeState(
        pabst=pabst, pzz=pzz, dzz=pdzz,
        pt=pt, pth=pth, pthl=pthl,
        prv=prv, prc=prc, pri=pri, prr=prr, prs=prs, prg=prg, prt=prt,
        pu=pu, pv=pv, pw=pw,
        ptke=ptke, ptkecls=ptkecls,
        psurf_flux_u=psurf_flux_u,
        psurf_flux_v=psurf_flux_v,
        psurf_flux_th=psurf_flux_th,
        psurf_flux_rv=psurf_flux_rv,
        pthvref=pthvref,
        pexn=exn, pexn_ref=exn_ref, prho_dry_ref=rho,
    )


class _NullEcRad:
    """No-op ecRad stub: returns zero radiative fluxes."""
    def __call__(self, state, dt):
        import jax.numpy as jnp
        nit, nkt = state.pabst.shape
        _fdt = state.pabst.dtype

        class _Fluxes:
            sw_dn = jnp.zeros((nit, nkt), dtype=_fdt)
            lw_dn = jnp.zeros((nit, nkt), dtype=_fdt)

        return _Fluxes(), {}


class _NullSurfex:
    """No-op SURFEX stub: returns zero surface fluxes.

    AromePhysicsOrchestrator checks isinstance(self.surfex, SurfexJAX) at
    JIT trace time (self is static_argnums=0).  This stub is NOT a SurfexJAX
    instance, so the orchestrator will NOT overwrite the bulk-drag fluxes that
    were embedded in AromeState before the step call.
    """
    def __call__(self, state, dt):
        import jax.numpy as jnp
        nit = state.psurf_flux_th.shape[0]
        _fdt = state.psurf_flux_th.dtype

        class _Fluxes:
            surf_flux_th = jnp.zeros((nit,), dtype=_fdt)
            surf_flux_rv = jnp.zeros((nit,), dtype=_fdt)
            surf_flux_u  = jnp.zeros((nit,), dtype=_fdt)
            surf_flux_v  = jnp.zeros((nit,), dtype=_fdt)

        return _Fluxes()


def _make_orchestrator(tstep: float = 30.0):
    """Instantiate AromePhysicsOrchestrator using Phyex for full constants.

    Phyex.to_externals() provides the ~325 uppercase keys (CPD, RD, RV, TT,
    LVTT, LSTT, TIMAUTI, TEXAUTI, TMAXMIX, TMINMIX, …) that RainIceJAX and
    IceAdjustJAX expect.  Passing phyex also initialises IceAdjustJAX with the
    correct PHYEX microphysics configuration.
    """
    from ice3.jax.arome_physics import AromePhysicsOrchestrator
    from ice3.phyex_common.phyex import Phyex

    phyex = Phyex(program="AROME", TSTEP=tstep)
    constants = phyex.to_externals()
    orc = AromePhysicsOrchestrator(constants=constants, phyex=phyex)
    orc.ecrad  = _NullEcRad()
    orc.surfex = _NullSurfex()
    return orc


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _total_water(state) -> float:
    """Sum of all hydrometeor mixing ratios over the whole domain."""
    import jax.numpy as jnp
    return float(jnp.sum(state.prv + state.prc + state.prr +
                         state.pri + state.prs + state.prg))


def _bl_mean_theta(state, k_top: int = 5) -> float:
    """Mean θ over the lowest k_top levels."""
    return float(state.pth[:, :k_top].mean())


def _surface_temp(state) -> np.ndarray:
    """T at the surface level (k=0), shape (nit,)."""
    return np.asarray(state.pt[:, 0])


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

DT  = 30.0   # s — physics timestep
DZ  = 100.0  # m — vertical resolution


@pytest.fixture(scope="module")
def orchestrator():
    return _make_orchestrator()


@pytest.fixture()
def state_f32():
    """Fresh AromeState (float32) from the cold-night profile."""
    import jax.numpy as jnp
    from cold_night_profile import ColdNightFields, NX, NY, NZ, DZ as _DZ
    return _build_arome_state(
        ColdNightFields(), NX, NY, NZ, _DZ, jnp.float32
    )


@pytest.fixture()
def state_f64():
    """Fresh AromeState (float64) from the cold-night profile."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from cold_night_profile import ColdNightFields, NX, NY, NZ, DZ as _DZ
    fields = ColdNightFields()
    # Cast all numpy arrays to float64
    for attr in ("theta_total", "exner_total", "exner_ambient", "theta_ambient",
                 "rvapour_ambient", "density", "rvapour", "rliquid", "rrain",
                 "rice", "rsnow", "rgraupel", "tke",
                 "surface_sensible_heat_flux", "surface_latent_heat_flux"):
        getattr(fields, attr)["covering"] = \
            getattr(fields, attr)["covering"].astype(np.float64)
    for i in range(3):
        fields.uvel[i]["covering"] = fields.uvel[i]["covering"].astype(np.float64)
    return _build_arome_state(fields, NX, NY, NZ, _DZ, jnp.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestInstantiation:

    def test_orchestrator_instantiates(self, orchestrator):
        assert orchestrator is not None

    def test_state_f32_has_correct_shape(self, state_f32):
        from cold_night_profile import NX, NY, NZ
        assert state_f32.pabst.shape == (NX * NY, NZ)

    def test_step_runs_f32(self, orchestrator, state_f32):
        new_state, _ = orchestrator.step(state_f32, DT)
        assert new_state.pth.shape == state_f32.pth.shape


# ─────────────────────────────────────────────────────────────────────────────
# Conservation
# ─────────────────────────────────────────────────────────────────────────────

class TestConservation:

    def test_total_water_conservation_f32(self, orchestrator, state_f32):
        qt_before = _total_water(state_f32)
        new_state, _ = orchestrator.step(state_f32, DT)
        qt_after = _total_water(new_state)
        rel_err = abs(qt_after - qt_before) / (abs(qt_before) + 1e-30)
        assert rel_err < 1e-4, (
            f"Float32 water not conserved: rel error = {rel_err:.2e}"
        )

    def test_total_water_conservation_f64(self, orchestrator, state_f64):
        qt_before = _total_water(state_f64)
        new_state, _ = orchestrator.step(state_f64, DT)
        qt_after = _total_water(new_state)
        rel_err = abs(qt_after - qt_before) / (abs(qt_before) + 1e-30)
        assert rel_err < 1e-8, (
            f"Float64 water not conserved: rel error = {rel_err:.2e}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Physical bounds
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldBounds:

    def test_temperature_physically_plausible(self, orchestrator, state_f32):
        new_state, _ = orchestrator.step(state_f32, DT)
        T = np.asarray(new_state.pt)
        assert T.min() > 200.0, f"Temperature too cold: {T.min():.1f} K"
        assert T.max() < 340.0, f"Temperature too hot: {T.max():.1f} K"

    def test_tke_non_negative(self, orchestrator, state_f32):
        new_state, _ = orchestrator.step(state_f32, DT)
        tke = np.asarray(new_state.ptke)
        assert tke.min() >= 0.0, f"Negative TKE: min = {tke.min():.4f} m²/s²"

    def test_hydrometeors_non_negative(self, orchestrator, state_f32):
        new_state, _ = orchestrator.step(state_f32, DT)
        for name, arr in (
            ("prv", new_state.prv), ("prc", new_state.prc),
            ("prr", new_state.prr), ("pri", new_state.pri),
            ("prs", new_state.prs), ("prg", new_state.prg),
        ):
            arr_np = np.asarray(arr)
            assert arr_np.min() >= 0.0, (
                f"{name} went negative: min = {arr_np.min():.2e} kg/kg"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Cold-night meteorology
# ─────────────────────────────────────────────────────────────────────────────

class TestColdNightMeteorology:

    def test_stable_bl_no_shallow_convection(self, orchestrator, state_f32):
        """Stable nocturnal BL: BL θ must not jump by more than 0.5 K."""
        theta_bl_before = _bl_mean_theta(state_f32)
        new_state, _ = orchestrator.step(state_f32, DT)
        delta = _bl_mean_theta(new_state) - theta_bl_before
        assert delta < 0.5, (
            f"BL θ jumped {delta:.2f} K — shallow convection on stable night?"
        )

    def test_nocturnal_radiative_cooling(self, orchestrator, state_f32):
        """Negative surface flux: BL must not warm more than 0.1 K in 30 s."""
        theta_bl_before = _bl_mean_theta(state_f32)
        new_state, _ = orchestrator.step(state_f32, DT)
        delta = _bl_mean_theta(new_state) - theta_bl_before
        assert delta < 0.1, f"BL warmed {delta:.3f} K during nocturnal step."

    def test_frost_surface_temperature(self, orchestrator, state_f32):
        """Surface T must stay below 5 °C and change < 2 K in 30 s."""
        T_before = _surface_temp(state_f32)
        new_state, _ = orchestrator.step(state_f32, DT)
        T_after = _surface_temp(new_state)
        delta = T_after - T_before
        assert np.all(delta < 2.0), f"Surface T jumped {delta.max():.2f} K"
        assert np.all(T_after < 278.15), f"Surface T above 5°C: {T_after.max():.1f} K"

    def test_no_rain_at_frost(self, orchestrator, state_f32):
        """Clear-sky frost night: no rain should be produced."""
        new_state, _ = orchestrator.step(state_f32, DT)
        rr = np.asarray(new_state.prr)
        assert rr.max() < 1e-8, f"Rain on frost night: max = {rr.max():.2e} kg/kg"

    def test_inversion_preserved(self, orchestrator, state_f32):
        """Nocturnal inversion (θ increases with z) must survive one step."""
        new_state, _ = orchestrator.step(state_f32, DT)
        th = np.asarray(new_state.pth)          # (nit, nkt)
        frac = np.mean(th[:, 4] > th[:, 0])
        assert frac > 0.90, (
            f"Inversion destroyed: only {100*frac:.0f}% columns inverted"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Ice microphysics
# ─────────────────────────────────────────────────────────────────────────────

class TestIcePhysics:

    def test_ice_nucleation_supersaturated(self, orchestrator):
        """At T=271 K, 110% qvsat_ice: ice_adjust must nucleate ice."""
        import jax.numpy as jnp
        from cold_night_profile import (
            ColdNightFields, NX, NY, NZ, DZ as _DZ,
            CST, _z_centers, _exner, _broadcast_profile,
        )

        fields = ColdNightFields()

        z   = _z_centers(NZ, _DZ)
        exn = _exner(z).astype(np.float32)
        T_ice = np.full(NZ, 271.0, dtype=np.float32)
        theta_ice = (T_ice / exn).astype(np.float32)
        p = (CST.p0 * exn ** (CST.cpd / CST.Rd)).astype(np.float32)

        es_ice   = 611.2 * np.exp(22.46 * (T_ice - CST.t_melt) / (T_ice - 0.53))
        qvsat_ice = (0.622 * es_ice / (p - es_ice)).astype(np.float32)
        qv_sup   = (qvsat_ice * 1.10).astype(np.float32)

        fields.theta_total    = _broadcast_profile(theta_ice, NX, NY)
        fields.exner_total    = _broadcast_profile(exn,       NX, NY)
        fields.theta_ambient  = _broadcast_profile(theta_ice, NX, NY)
        fields.exner_ambient  = _broadcast_profile(exn,       NX, NY)
        fields.rvapour        = _broadcast_profile(qv_sup,    NX, NY)
        fields.rvapour_ambient = _broadcast_profile(qv_sup,   NX, NY)

        state = _build_arome_state(fields, NX, NY, NZ, _DZ, jnp.float32)
        new_state, _ = orchestrator.step(state, DT)

        ice = float(jnp.max(new_state.pri)) + float(jnp.max(new_state.prs))
        assert ice > 0.0, (
            "ice_adjust / rain_ice did not nucleate ice at T=271K, qv=110% qvsat_ice"
        )
