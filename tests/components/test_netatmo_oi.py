"""
Tests for ice3.jax.netatmo_oi — Netatmo Optimal Interpolation.

All tests run on CPU (pure JAX, no GPU required).
"""

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = jnp = None
    HAS_JAX = False

requires_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")

if HAS_JAX:
    from ice3.jax.netatmo_oi import NetatmoObs, NetatmoOI, make_t_skin_background, TILE_T_SKIN_DEFAULT


# ---------------------------------------------------------------------------
# NetatmoOI basic correctness
# ---------------------------------------------------------------------------

class TestNetatmoOI:

    @requires_jax
    def test_no_obs_returns_background(self):
        """All valid=0 → analysed t_skin must equal background everywhere."""
        n = 8
        oi = NetatmoOI()
        t_bg = jnp.full(n, 285.0, dtype=jnp.float32)
        t_a  = jnp.full(n, 284.0, dtype=jnp.float32)
        obs  = NetatmoObs(
            t2m=jnp.full(n, 288.0, dtype=jnp.float32),
            q2m=jnp.zeros(n, dtype=jnp.float32),
            valid=jnp.zeros(n, dtype=jnp.int32),
        )
        t_a_out = oi(t_bg, t_a, obs)
        np.testing.assert_allclose(
            np.array(t_a_out), np.array(t_bg), rtol=1e-6,
            err_msg="No-obs case must return background unchanged",
        )

    @requires_jax
    def test_all_obs_shifts_toward_observation(self):
        """
        When T2m_obs > T2m_bg, the analysis must increase t_skin above background.
        """
        n = 4
        oi = NetatmoOI(sigma_b=2.0, H=0.35)
        t_bg = jnp.full(n, 285.0, dtype=jnp.float32)
        t_a  = jnp.full(n, 284.0, dtype=jnp.float32)
        t2m_obs = jnp.full(n, 292.0, dtype=jnp.float32)   # warmer than bg
        obs = NetatmoObs(
            t2m=t2m_obs, q2m=jnp.zeros(n), valid=jnp.ones(n, dtype=jnp.int32)
        )
        t_a_out = oi(t_bg, t_a, obs)
        assert np.all(np.array(t_a_out) > np.array(t_bg)), \
            "Warm obs must increase t_skin above background"

    @requires_jax
    def test_cold_obs_decreases_t_skin(self):
        """When T2m_obs < T2m_bg, the analysis must decrease t_skin."""
        n = 4
        oi = NetatmoOI(sigma_b=2.0, H=0.35)
        t_bg = jnp.full(n, 290.0, dtype=jnp.float32)
        t_a  = jnp.full(n, 289.0, dtype=jnp.float32)
        t2m_obs = jnp.full(n, 281.0, dtype=jnp.float32)   # colder
        obs = NetatmoObs(
            t2m=t2m_obs, q2m=jnp.zeros(n), valid=jnp.ones(n, dtype=jnp.int32)
        )
        t_a_out = oi(t_bg, t_a, obs)
        assert np.all(np.array(t_a_out) < np.array(t_bg)), \
            "Cold obs must decrease t_skin below background"

    @requires_jax
    def test_clamp_lower_bound(self):
        """Analysis must never produce t_skin below t_skin_min (220 K)."""
        n = 4
        oi = NetatmoOI(sigma_b=20.0, H=0.35, t_skin_min=220.0)
        t_bg = jnp.full(n, 230.0, dtype=jnp.float32)
        t_a  = jnp.full(n, 229.0, dtype=jnp.float32)
        obs  = NetatmoObs(
            t2m=jnp.full(n, 100.0),  # unrealistically cold obs
            q2m=jnp.zeros(n),
            valid=jnp.ones(n, dtype=jnp.int32),
        )
        t_a_out = oi(t_bg, t_a, obs)
        assert np.all(np.array(t_a_out) >= 220.0), "Analysis below t_skin_min"

    @requires_jax
    def test_clamp_upper_bound(self):
        """Analysis must never produce t_skin above t_skin_max (340 K)."""
        n = 4
        oi = NetatmoOI(sigma_b=20.0, H=0.35, t_skin_max=340.0)
        t_bg = jnp.full(n, 330.0, dtype=jnp.float32)
        t_a  = jnp.full(n, 329.0, dtype=jnp.float32)
        obs  = NetatmoObs(
            t2m=jnp.full(n, 500.0),  # unrealistically hot obs
            q2m=jnp.zeros(n),
            valid=jnp.ones(n, dtype=jnp.int32),
        )
        t_a_out = oi(t_bg, t_a, obs)
        assert np.all(np.array(t_a_out) <= 340.0), "Analysis above t_skin_max"

    @requires_jax
    def test_partial_valid(self):
        """Columns with valid=0 must return background; valid=1 must be updated."""
        n = 4
        oi = NetatmoOI(sigma_b=2.0, H=0.35)
        t_bg = jnp.array([285., 285., 285., 285.], dtype=jnp.float32)
        t_a  = jnp.array([284., 284., 284., 284.], dtype=jnp.float32)
        obs  = NetatmoObs(
            t2m=jnp.array([292., 292., 0., 0.], dtype=jnp.float32),
            q2m=jnp.zeros(n),
            valid=jnp.array([1, 1, 0, 0], dtype=jnp.int32),
        )
        t_a_out = np.array(oi(t_bg, t_a, obs))
        # First two columns should be updated (warmer obs → higher t_skin)
        assert np.all(t_a_out[:2] > 285.0), "Valid obs columns must be updated"
        # Last two columns must remain at background
        np.testing.assert_allclose(t_a_out[2:], 285.0, rtol=1e-6,
                                   err_msg="Invalid obs columns must equal background")

    @requires_jax
    def test_per_column_sigma_o(self):
        """Per-column sigma_o must modulate the gain (higher sigma_o → smaller increment)."""
        n = 2
        oi = NetatmoOI(sigma_b=2.0, H=0.35)
        t_bg = jnp.full(n, 285.0, dtype=jnp.float32)
        t_a  = jnp.full(n, 284.0, dtype=jnp.float32)
        t2m  = jnp.full(n, 292.0, dtype=jnp.float32)
        # col 0: small obs error → large correction; col 1: large → small
        obs_low  = NetatmoObs(t2m=t2m, q2m=jnp.zeros(n),
                              valid=jnp.ones(n, dtype=jnp.int32),
                              sigma_o=jnp.array([0.5, 5.0], dtype=jnp.float32))
        t_out = np.array(oi(t_bg, t_a, obs_low))
        assert t_out[0] > t_out[1], \
            "Lower sigma_o must yield larger analysis increment"

    @requires_jax
    def test_output_dtype_matches_input(self):
        """Output dtype must match the input background dtype."""
        n = 8
        oi = NetatmoOI()
        t_bg = jnp.full(n, 285.0, dtype=jnp.float32)
        t_a  = jnp.full(n, 284.0, dtype=jnp.float32)
        obs  = NetatmoObs(t2m=jnp.full(n, 288.), q2m=jnp.zeros(n),
                          valid=jnp.ones(n, dtype=jnp.int32))
        t_out = oi(t_bg, t_a, obs)
        assert t_out.dtype == jnp.float32

    @requires_jax
    def test_jit_compatible(self):
        """NetatmoOI.__call__ must work inside jax.jit."""
        n = 16
        oi = NetatmoOI()
        t_bg = jnp.full(n, 285.0, dtype=jnp.float32)
        t_a  = jnp.full(n, 284.0, dtype=jnp.float32)
        obs  = NetatmoObs(t2m=jnp.full(n, 288., dtype=jnp.float32),
                          q2m=jnp.zeros(n, dtype=jnp.float32),
                          valid=jnp.ones(n, dtype=jnp.int32))

        jit_oi = jax.jit(lambda bg, a, o: oi(bg, a, o))
        t_out = jit_oi(t_bg, t_a, obs)
        assert t_out.shape == (n,)
        assert np.all(np.isfinite(np.array(t_out)))

    @requires_jax
    def test_gain_formula(self):
        """Verify OI gain K = H*sb^2 / (H^2*sb^2 + so^2) against direct formula."""
        oi = NetatmoOI(sigma_b=2.0, H=0.35)
        sigma_o = jnp.array([1.5], dtype=jnp.float32)
        K = float(np.array(oi._gain(sigma_o))[0])
        H, sb2 = 0.35, 4.0
        K_ref = H * sb2 / (H * H * sb2 + 1.5**2)
        np.testing.assert_allclose(K, K_ref, rtol=1e-5)

    @requires_jax
    def test_forward_operator(self):
        """diagnose_t2m must match the linear formula H*Ts + (1-H)*Ta."""
        oi = NetatmoOI(H=0.35)
        t_skin = jnp.array([285.0], dtype=jnp.float32)
        t_a    = jnp.array([284.0], dtype=jnp.float32)
        t2m    = float(np.array(oi.diagnose_t2m(t_skin, t_a))[0])
        t2m_ref = 0.35 * 285.0 + 0.65 * 284.0
        np.testing.assert_allclose(t2m, t2m_ref, rtol=1e-6)


# ---------------------------------------------------------------------------
# make_t_skin_background helper
# ---------------------------------------------------------------------------

class TestMakeTSkinBackground:

    @requires_jax
    def test_land_tile_default(self):
        tiles = np.array([1, 1, 1], dtype=np.int32)
        bg = np.array(make_t_skin_background(tiles))
        np.testing.assert_allclose(bg, TILE_T_SKIN_DEFAULT[1], rtol=1e-6)

    @requires_jax
    def test_sea_tile_default(self):
        tiles = np.array([2, 2], dtype=np.int32)
        bg = np.array(make_t_skin_background(tiles))
        np.testing.assert_allclose(bg, TILE_T_SKIN_DEFAULT[2], rtol=1e-6)

    @requires_jax
    def test_lake_tile_default(self):
        tiles = np.array([3], dtype=np.int32)
        bg = np.array(make_t_skin_background(tiles))
        np.testing.assert_allclose(bg, TILE_T_SKIN_DEFAULT[3], rtol=1e-6)

    @requires_jax
    def test_mixed_tiles(self):
        tiles = np.array([1, 2, 3], dtype=np.int32)
        bg = np.array(make_t_skin_background(tiles))
        assert bg[0] == TILE_T_SKIN_DEFAULT[1]
        assert bg[1] == TILE_T_SKIN_DEFAULT[2]
        assert bg[2] == TILE_T_SKIN_DEFAULT[3]

    @requires_jax
    def test_output_dtype(self):
        tiles = np.array([1, 2, 3], dtype=np.int32)
        bg = make_t_skin_background(tiles)
        assert bg.dtype == jnp.float32

    @requires_jax
    def test_unknown_tile_gets_land_default(self):
        """Unknown tile IDs fall back to 285 K."""
        tiles = np.array([99], dtype=np.int32)
        bg = np.array(make_t_skin_background(tiles))
        np.testing.assert_allclose(bg, 285.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end: OI increment flows through SurfexState → changes fluxes
# ---------------------------------------------------------------------------

class TestNetatmoOIIntegration:

    @requires_jax
    def test_nonzero_t_skin_changes_sensible_heat_flux(self):
        """
        When Netatmo OI produces a non-zero t_skin, the _bulk_aerodynamic_fallback
        must return different surf_flux_th than the all-zero sentinel case.
        """
        from ice3.jax.surfex_jax import _bulk_aerodynamic_fallback
        n = 8
        rng = np.random.default_rng(7)
        t_a = rng.uniform(280., 300., n)
        q_a = rng.uniform(0.005, 0.015, n)
        u_a = rng.uniform(2., 8., n)
        v_a = rng.uniform(-3., 3., n)
        p_a = np.full(n, 100000.)
        rhodref = np.full(n, 1.2)

        # Sentinel zeros → all columns use default 295 K
        t_skin_zero = np.zeros(n)
        flux_ref = _bulk_aerodynamic_fallback(n, t_skin_zero, t_a, q_a, u_a, v_a, p_a, rhodref)

        # Netatmo analysis: half the columns get a warm t_skin
        t_skin_warm = np.zeros(n)
        t_skin_warm[:n // 2] = 305.0
        flux_warm = _bulk_aerodynamic_fallback(n, t_skin_warm, t_a, q_a, u_a, v_a, p_a, rhodref)

        # Columns with t_skin > 295 must have higher sensible heat flux
        assert np.all(flux_warm[0][:n // 2] > flux_ref[0][:n // 2]), \
            "Warm Netatmo t_skin must increase sensible heat flux"
        # Columns with t_skin = 0 (sentinel) must be identical
        np.testing.assert_allclose(
            flux_warm[0][n // 2:], flux_ref[0][n // 2:], rtol=1e-6,
            err_msg="Sentinel columns must be unchanged",
        )
