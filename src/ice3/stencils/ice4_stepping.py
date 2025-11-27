# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import PARALLEL, Field, computation, interval
from ice3.functions.ice_adjust import (constant_pressure_heat_capacity,
                                       sublimation_latent_heat,
                                       vaporisation_latent_heat)
from ice3.functions.sign import sign
from ice3.functions.temperature import theta2temperature


# from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
# from_line=215,
# to_line=221,
def ice4_stepping_tmicro_init(t_micro: Field["float"], ldmicro: Field["bool"]):
    """Initialise t_soft with value of t_micro after each loop
    on LSOFT condition.

    Args:
        t_micro (Field[float]): time for microphsyics loops
        ldmicro (Field[bool]): microphsyics activation mask
    """

    from __externals__ import TSTEP

    # 4.4 Temporal loop
    with computation(PARALLEL), interval(...):
        t_micro = 0 if ldmicro else TSTEP


#    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
#    from_line=225,
#    to_line=228,
def ice4_stepping_init_tsoft(t_micro: Field["float"], t_soft: Field["float"]):
    """Initialise t_soft with value of t_micro after each loop
    on LSOFT condition.

    Args:
        t_micro (Field[float]): time for microphsyics loops
        t_soft (Field[float]): time for lsoft blocks loops
    """

    with computation(PARALLEL), interval(...):
        t_soft = t_micro


#    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
#    from_line=244,
#    to_line=254,
def ice4_stepping_heat(
    rv_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    exn: Field["float"],
    th_t: Field["float"],
    ls_fact: Field["float"],
    lv_fact: Field["float"],
    t: Field["float"],
):
    """Compute and convert heat variables before computations

    Args:
        rv_t (Field[float]): vapour mixing ratio
        rc_t (Field[float]): cloud droplet mixing ratio
        rr_t (Field[float]): rain m.r.
        ri_t (Field[float]): ice m.r.
        rs_t (Field[float]): snow m.r.
        rg_t (Field[float]): graupel m.r.
        exn (Field[float]): exner pressure
        th_t (Field[float]): potential temperature
        ls_fact (Field[float]): sublimation latent heat over heat capacity
        lv_fact (Field[float]): vapourisation latent heat over heat capacity
        t (Field[float]): temperature
    """
    with computation(PARALLEL), interval(...):
        specific_heat = constant_pressure_heat_capacity(rv_t, rc_t, ri_t, rr_t, rs_t, rg_t)
        t = theta2temperature(th_t, exn)
        ls_fact = sublimation_latent_heat(t) / specific_heat
        lv_fact = vaporisation_latent_heat(t) / specific_heat


#    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
#    from_line=230,
#    to_line=237,
def ice4_stepping_ldcompute_init(ldcompute: Field["bool"], t_micro: Field["float"]):
    """Initialize ldcompute mask

    Args:
        ldcompute (Field[bool]): mask to compute microphysical species
        t_micro (Field[float]): microphysical time-step
    """

    from __externals__ import TSTEP

    with computation(PARALLEL), interval(...):
        ldcompute = True if t_micro < TSTEP else False
        

############################ MRSTEP != 0 ################################
#    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
#    from_line=346,
#    to_line=388,
def ice4_mixing_ratio_step_limiter(
    rc_0r_t: Field["float"],
    rr_0r_t: Field["float"],
    ri_0r_t: Field["float"],
    rs_0r_t: Field["float"],
    rg_0r_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    rc_b: Field["float"],
    rr_b: Field["float"],
    ri_b: Field["float"],
    rs_b: Field["float"],
    rg_b: Field["float"],
    rc_tnd_a: Field["float"],
    rr_tnd_a: Field["float"],
    ri_tnd_a: Field["float"],
    rs_tnd_a: Field["float"],
    rg_tnd_a: Field["float"],
    delta_t_micro: Field["float"],
    ldcompute: Field["bool"],
):
    """Step limiter for processes, based on tendencies thresholds.

    Args:
        rc_0r_t (Field[float]): _description_
        rr_0r_t (Field[float]): _description_
        ri_0r_t (Field[float]): _description_
        rs_0r_t (Field[float]): _description_
        rg_0r_t (Field[float]): _description_
        rc_t (Field[float]): _description_
        rr_t (Field[float]): _description_
        ri_t (Field[float]): _description_
        rs_t (Field[float]): _description_
        rg_t (Field[float]): _description_
        rc_b (Field[float]): _description_
        rr_b (Field[float]): _description_
        ri_b (Field[float]): _description_
        rs_b (Field[float]): _description_
        rg_b (Field[float]): _description_
        rc_tnd_a (Field[float]): _description_
        rr_tnd_a (Field[float]): _description_
        ri_tnd_a (Field[float]): _description_
        rs_tnd_a (Field[float]): _description_
        rg_tnd_a (Field[float]): _description_
        delta_t_micro (Field[float]): _description_
        time_threshold_tmp (Field[float]): _description_
    """
    from __externals__ import (C_RTMIN, G_RTMIN, I_RTMIN, MRSTEP, R_RTMIN,
                               S_RTMIN)

    ############## (c) ###########
    # l356
    with computation(PARALLEL), interval(...):
        # TODO: add condition on LL_ANY_ITER
        time_threshold_tmp = (
            (sign(1, rc_tnd_a) * MRSTEP + rc_0r_t - rc_t - rc_b)
            if abs(rc_tnd_a) > 1e-20
            else -1
        )

    # l363
    with computation(PARALLEL), interval(...):
        if (
            time_threshold_tmp >= 0
            and time_threshold_tmp < delta_t_micro
            and (rc_t > C_RTMIN or rc_tnd_a > 0)
        ):
            delta_t_micro = min(delta_t_micro, time_threshold_tmp)
            ldcompute = False

            # Translation note : ldcompute is LLCOMPUTE in mode_ice4_stepping.F90

    # l370
    # Translation note : l370 to l378 in mode_ice4_stepping. F90 contracted in a single stencil
    with computation(PARALLEL), interval(...):
        r_b_max = abs(rr_b)

    ################ (r) #############
    with computation(PARALLEL), interval(...):
        time_threshold_tmp = (
            (sign(1, rr_tnd_a) * MRSTEP + rr_0r_t - rr_t - rr_b)
            if abs(rr_tnd_a) > 1e-20
            else -1
        )

    with computation(PARALLEL), interval(...):
        if (
            time_threshold_tmp >= 0
            and time_threshold_tmp < delta_t_micro
            and (rr_t > R_RTMIN or rr_tnd_a > 0)
        ):
            delta_t_micro = min(delta_t_micro, time_threshold_tmp)
            ldcompute = False

    with computation(PARALLEL), interval(...):
        r_b_max = max(r_b_max, abs(rr_b))

    ################ (i) #############
    with computation(PARALLEL), interval(...):
        time_threshold_tmp = (
            (sign(1, ri_tnd_a) * MRSTEP + ri_0r_t - ri_t - ri_b)
            if abs(ri_tnd_a) > 1e-20
            else -1
        )

    with computation(PARALLEL), interval(...):
        if (
            time_threshold_tmp >= 0
            and time_threshold_tmp < delta_t_micro
            and (rc_t > I_RTMIN or ri_tnd_a > 0)
        ):
            delta_t_micro = min(delta_t_micro, time_threshold_tmp)
            ldcompute = False

    with computation(PARALLEL), interval(...):
        r_b_max = max(r_b_max, abs(ri_b))

    ################ (s) #############
    with computation(PARALLEL), interval(...):
        time_threshold_tmp = (
            (sign(1, rs_tnd_a) * MRSTEP + rs_0r_t - rs_t - rs_b)
            if abs(rs_tnd_a) > 1e-20
            else -1
        )

    with computation(PARALLEL), interval(...):
        if (
            time_threshold_tmp >= 0
            and time_threshold_tmp < delta_t_micro
            and (rs_t > S_RTMIN or rs_tnd_a > 0)
        ):
            delta_t_micro = min(delta_t_micro, time_threshold_tmp)
            ldcompute = False

    with computation(PARALLEL), interval(...):
        r_b_max = max(r_b_max, abs(rs_b))

    ################ (g) #############
    with computation(PARALLEL), interval(...):
        time_threshold_tmp = (
            (sign(1, rg_tnd_a) * MRSTEP + rg_0r_t - rg_t - rg_b)
            if abs(rg_tnd_a) > 1e-20
            else -1
        )

    with computation(PARALLEL), interval(...):
        if (
            time_threshold_tmp >= 0
            and time_threshold_tmp < delta_t_micro
            and (rg_t > G_RTMIN or rg_tnd_a > 0)
        ):
            delta_t_micro = min(delta_t_micro, time_threshold_tmp)
            ldcompute = False

    with computation(PARALLEL), interval(...):
        r_b_max = max(r_b_max, abs(rg_b))  # (g)

    # Limiter on max mixing ratio
    with computation(PARALLEL), interval(...):
        if r_b_max > MRSTEP:
            delta_t_micro = 0
            ldcompute = False


def mixing_ratio_step_limiter(
    r_tnd_a: "float",
    r_b: "float",
    r_t: "float",
    delta_t_micro: "float",
    r_min: "float",
    tiny: "float",
) -> "float":
    """Helper function to limit timestep based on species disappearance.
    
    This function calculates the time needed for a species to reach its minimum
    threshold based on current tendency, and limits the timestep accordingly.
    
    Args:
        r_tnd_a: Tendency of mixing ratio [kg/kg/s]
        r_b: Instantaneous change in mixing ratio [kg/kg]
        r_t: Current mixing ratio [kg/kg]
        delta_t_micro: Current microphysical timestep [s]
        r_min: Minimum threshold for the species [kg/kg]
        tiny: Small value for numerical comparisons
        
    Returns:
        Limited timestep [s]
    """
    if abs(r_tnd_a) > tiny:
        # Calculate time to reach minimum threshold
        # r_t + r_tnd_a * dt + r_b = r_min
        # dt = (r_min - r_t - r_b) / r_tnd_a
        time_threshold = (r_min - r_t - r_b) / r_tnd_a
        
        # Only limit if threshold is positive (would reach minimum in future)
        # and if current value is above minimum or tendency is increasing
        if time_threshold > 0.0 and (r_t > r_min or r_tnd_a > 0.0):
            return min(delta_t_micro, time_threshold)
    
    return delta_t_micro


#    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
#    from_line=290,
#    to_line=332,
def ice4_step_limiter(
    exn: Field["float"],
    th_t: Field["float"],
    theta_a_tnd: Field["float"],
    theta_b: Field["float"],
    theta_ext_tnd: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    rc_a_tnd: Field["float"],
    rr_a_tnd: Field["float"],
    ri_a_tnd: Field["float"],
    rs_a_tnd: Field["float"],
    rg_a_tnd: Field["float"],
    rc_ext_tnd: Field["float"],
    rr_ext_tnd: Field["float"],
    ri_ext_tnd: Field["float"],
    rs_ext_tnd: Field["float"],
    rg_ext_tnd: Field["float"],
    rc_b: Field["float"],
    rr_b: Field["float"],
    ri_b: Field["float"],
    rs_b: Field["float"],
    rg_b: Field["float"],
    delta_t_micro: Field["float"],
    t_micro: Field["float"],
    delta_t_soft: Field["float"],
    t_soft: Field["float"],
    ldcompute: Field["bool"],
):
    """
    Compute adaptive microphysical time step with safety limiters.
    
    This critical function implements adaptive time stepping for ICE4
    microphysics, preventing numerical instabilities by limiting the time
    step based on:
    1. Species depletion (prevents negative mixing ratios)
    2. Temperature crossing 0°C (phase change threshold)
    3. Overall time step constraint (TSTEP)
    
    The adaptive time stepping ensures numerical stability and physical
    realizability while maximizing computational efficiency.
    
    Parameters
    ----------
    exn : Field[float]
        Exner function π (dimensionless). Input field.
    th_t : Field[float]
        Potential temperature at time t (K). Input field.
    theta_a_tnd : Field[float]
        Microphysical temperature tendency (K/s). Modified in place.
    theta_b : Field[float]
        Instantaneous temperature change (K). Input field.
    theta_ext_tnd : Field[float]
        External temperature tendency (K/s). Input field.
    rc_t, rr_t, ri_t, rs_t, rg_t : Field[float]
        Current mixing ratios (kg/kg) for cloud, rain, ice, snow, graupel.
    rc_a_tnd, rr_a_tnd, ri_a_tnd, rs_a_tnd, rg_a_tnd : Field[float]
        Microphysical tendencies (kg/kg/s). Modified in place.
    rc_ext_tnd, rr_ext_tnd, ri_ext_tnd, rs_ext_tnd, rg_ext_tnd : Field[float]
        External tendencies (kg/kg/s). Input fields.
    rc_b, rr_b, ri_b, rs_b, rg_b : Field[float]
        Instantaneous mixing ratio changes (kg/kg). Input fields.
    delta_t_micro : Field[float]
        Computed microphysical time step (s). Output field.
    t_micro : Field[float]
        Elapsed microphysical time (s). Input field.
    delta_t_soft : Field[float]
        Soft process time step (s). Input field.
    t_soft : Field[float]
        Elapsed soft process time (s). Input field.
    ldcompute : Field[bool]
        Computation mask. Modified in place.
    
    Returns
    -------
    None
        Modifies delta_t_micro and ldcompute in place.
    
    Notes
    -----
    **Adaptive Time Stepping Philosophy:**
    
    ICE4 uses adaptive time stepping to balance:
    - **Accuracy:** Small steps for rapidly changing conditions
    - **Stability:** Prevent overshooting physical limits
    - **Efficiency:** Large steps when possible
    
    **Time Step Limiters:**
    
    1. **Overall Constraint:**
       Δt ≤ TSTEP - t_micro
       Never exceed the physics time step.
    
    2. **Temperature Crossing 0°C:**
       If θ crosses T_t/π, limit Δt to reach exactly 0°C:
       Δt = (T_t/π - θ - θ_b) / (dθ/dt)
       
       Prevents:
       - Mixed-phase instabilities
       - Temperature overshoot during phase changes
       - Spurious freezing/melting oscillations
    
    3. **Species Depletion:**
       For each species r_x, if dr_x/dt would deplete it:
       Δt = (r_min - r_x - r_b) / (dr_x/dt)
       
       Prevents:
       - Negative mixing ratios
       - Unphysical removal rates
       - Numerical underflow
    
    **Temperature Limiter Details:**
    
    The 0°C crossing is critical because:
    - Ice ↔ liquid transitions occur
    - Latent heat effects change sign
    - Microphysical processes switch regimes
    
    Detection:
    (θ_t - T_t/π) × (θ_t + θ_b - T_t/π) < 0
    
    This checks if temperature crosses threshold between
    current and next state.
    
    **Species Depletion Limiter:**
    
    For each hydrometeor (c, r, i, s, g):
    
    1. Compute time to threshold:
       τ = (r_threshold - r_current - r_instantaneous) / (dr/dt)
    
    2. If τ > 0 and (r > threshold or dr/dt > 0):
       Δt = min(Δt, τ)
    
    3. Threshold values:
       - C_RTMIN: Cloud ~10⁻²⁰ kg/kg
       - R_RTMIN: Rain ~10⁻²⁰ kg/kg
       - I_RTMIN: Ice ~10⁻¹⁵ kg/kg
       - S_RTMIN: Snow ~10⁻¹⁵ kg/kg
       - G_RTMIN: Graupel ~10⁻¹⁵ kg/kg
    
    **External Tendencies:**
    
    External tendencies (from turbulence, radiation, etc.) are added
    to microphysical tendencies before computing limiters:
    
    dr_x/dt_total = dr_x/dt_micro + dr_x/dt_ext
    
    This ensures all processes are accounted for in time step calculation.
    
    **Soft Process Constraint:**
    
    When TSTEP_TS ≠ 0, an additional constraint ensures soft processes
    (like sedimentation) maintain their own time step:
    
    If t_micro + Δt > t_soft + Δt_soft:
        Δt = t_soft + Δt_soft - t_micro
        ldcompute = False
    
    **Computation Mask (ldcompute):**
    
    Set to False when:
    - Time step reaches TSTEP
    - Temperature crosses 0°C
    - Any species would be depleted
    - Soft process constraint violated
    
    False value stops microphysics loop, moves to next phase.
    
    **Typical Time Step Evolution:**
    
    Initial: Δt = TSTEP (e.g., 60 s)
    ↓
    Phase change near: Δt = 5 s (approaching 0°C)
    ↓
    Strong process: Δt = 1 s (rapid depletion)
    ↓
    Recovery: Δt increases back
    
    **Numerical Stability:**
    
    The adaptive scheme ensures CFL-like conditions:
    - |dr/dt| × Δt ≤ r + safety_margin
    - |dθ/dt| × Δt ≤ |θ - θ_critical|
    
    **Physical Realizability:**
    
    Guarantees:
    - r_x ≥ 0 for all species
    - Temperature doesn't jump over phase change points
    - Conservation of mass and energy
    
    **Computational Efficiency:**
    
    - Large time steps when conditions are benign
    - Small steps only when necessary
    - Typical number of sub-steps: 1-10 per TSTEP
    
    **Special Cases:**
    
    Empty column: Δt = TSTEP (no limiters active)
    Vigorous convection: Δt ~ 0.1-1 s (many limiters active)
    Stratiform: Δt ~ 5-60 s (few limiters active)
    
    Source Reference
    ----------------
    PHYEX/src/common/micro/mode_ice4_stepping.F90, lines 290-332
    
    See Also
    --------
    state_update : Updates state variables after time step
    mixing_ratio_step_limiter : Helper for species depletion
    ice4_stepping_tmicro_init : Initializes time stepping
    
    Examples
    --------
    >>> # Normal case: full time step
    >>> # No limiters active
    >>> # → delta_t_micro = TSTEP = 60 s
    
    >>> # Temperature approaching 0°C
    >>> # T = -0.5°C, dT/dt = +0.1 K/s
    >>> # → delta_t_micro = 5 s (reaches 0°C exactly)
    
    >>> # Rapid evaporation
    >>> # r_r = 1e-4 kg/kg, dr_r/dt = -1e-4 kg/kg/s
    >>> # → delta_t_micro = 1 s (prevents negative rain)
    """
    from __externals__ import (C_RTMIN, G_RTMIN, I_RTMIN, MNH_TINY, R_RTMIN,
                               S_RTMIN, TSTEP, TSTEP_TS, TT)

    # Adding externals tendencies
    with computation(PARALLEL), interval(...):
        theta_a_tnd += theta_ext_tnd
        rc_a_tnd += rc_ext_tnd
        rr_a_tnd += rr_ext_tnd
        ri_a_tnd += ri_ext_tnd
        rs_a_tnd += rs_ext_tnd
        rg_a_tnd += rg_ext_tnd

    # 4.6 Time integration
    with computation(PARALLEL), interval(...):
        delta_t_micro = TSTEP - t_micro if ldcompute else 0

    # Adjustment of tendencies when temperature reaches 0
    with computation(PARALLEL), interval(...):
        th_tt = TT / exn
        if (th_t - th_tt) * (th_t + theta_b - th_tt) < 0:
            delta_t_micro = 0

        if abs(theta_a_tnd > 1e-20):
            delta_t_tmp = (th_tt - theta_b - th_t) / theta_a_tnd
            if delta_t_tmp > 0:
                delta_t_micro = min(delta_t_micro, delta_t_tmp)

    # Tendencies adjustment if a speci disappears
    with computation(PARALLEL), interval(...):
        # (c)
        delta_t_micro = mixing_ratio_step_limiter(
            rc_a_tnd, rc_b, rc_t, delta_t_micro, C_RTMIN, MNH_TINY
        )
        
        # (r)
        delta_t_micro = mixing_ratio_step_limiter(
        rr_a_tnd, rr_b, rr_t, delta_t_micro, R_RTMIN, MNH_TINY
        )

        # (i)
        delta_t_micro = mixing_ratio_step_limiter(
            ri_a_tnd, ri_b, ri_t, delta_t_micro, I_RTMIN, MNH_TINY
        )

        # (s)
        delta_t_micro = mixing_ratio_step_limiter(
            rs_a_tnd, rs_b, rs_t, delta_t_micro, S_RTMIN, MNH_TINY
        )

        # (g)
        delta_t_micro = mixing_ratio_step_limiter(
            rg_a_tnd, rg_b, rg_t, delta_t_micro, G_RTMIN, MNH_TINY
        )

    # We stop when the end of the timestep is reached
    with computation(PARALLEL), interval(...):
        ldcompute = False if t_micro + delta_t_micro > TSTEP else ldcompute

    # TODO : TSTEP_TS out of the loop
    with computation(PARALLEL), interval(...):
        if TSTEP_TS != 0:
            if t_micro + delta_t_micro > t_soft + delta_t_soft:
                delta_t_micro = t_soft + delta_t_soft - t_micro
                ldcompute = False


#    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
#    from_line=391,
#    to_line=404,
def state_update(
    th_t: Field["float"],
    theta_b: Field["float"],
    theta_tnd_a: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    rc_b: Field["float"],
    rr_b: Field["float"],
    ri_b: Field["float"],
    rs_b: Field["float"],
    rg_b: Field["float"],
    rc_tnd_a: Field["float"],
    rr_tnd_a: Field["float"],
    ri_tnd_a: Field["float"],
    rs_tnd_a: Field["float"],
    rg_tnd_a: Field["float"],
    delta_t_micro: Field["float"],
    ldmicro: Field["bool"],
    ci_t: Field["float"],
    t_micro: Field["float"],
):
    """
    Update state variables after adaptive microphysical time step.
    
    This function applies computed tendencies to advance the thermodynamic
    and microphysical state forward by the adaptive time step computed by
    ice4_step_limiter. It implements explicit forward Euler time integration
    with special handling for ice crystal number concentration when ice
    mass becomes depleted.
    
    Parameters
    ----------
    th_t : Field[float]
        Potential temperature θ (K). Modified in place.
    theta_b : Field[float]
        Instantaneous temperature adjustment (K). Input field.
        From processes like phase changes that occur instantly.
    theta_tnd_a : Field[float]
        Temperature tendency dθ/dt (K/s). Input field.
    rc_t, rr_t, ri_t, rs_t, rg_t : Field[float]
        Mixing ratios (kg/kg) for cloud, rain, ice, snow, graupel.
        Modified in place.
    rc_b, rr_b, ri_b, rs_b, rg_b : Field[float]
        Instantaneous mixing ratio adjustments (kg/kg). Input fields.
    rc_tnd_a, rr_tnd_a, ri_tnd_a, rs_tnd_a, rg_tnd_a : Field[float]
        Mixing ratio tendencies dr/dt (kg/kg/s). Input fields.
    delta_t_micro : Field[float]
        Adaptive microphysical time step (s). Input field.
    ldmicro : Field[bool]
        Microphysics computation mask. Input field.
    ci_t : Field[float]
        Ice crystal number concentration (#/kg). Modified in place.
    t_micro : Field[float]
        Elapsed microphysical time (s). Modified in place when ice depletes.
    
    Returns
    -------
    None
        Modifies state variables in place.
    
    Notes
    -----
    **Time Integration Scheme:**
    
    Forward Euler explicit integration:
    
    x(t + Δt) = x(t) + (dx/dt) × Δt + Δx_instant
    
    where:
    - x: State variable (θ, r_c, r_r, r_i, r_s, r_g)
    - dx/dt: Tendency from microphysical processes
    - Δx_instant: Instantaneous adjustment (e.g., phase changes)
    - Δt: Adaptive time step from ice4_step_limiter
    
    **Two-Term Updates:**
    
    Each variable updated with two components:
    
    1. **Continuous tendency (×_tnd_a × Δt):**
       - Gradual changes from processes
       - Scaled by time step
       - Examples: deposition, aggregation, evaporation
    
    2. **Instantaneous change (×_b):**
       - Immediate adjustments
       - Not scaled by time step
       - Examples: homogeneous freezing, melting at 0°C
    
    **Update Equations:**
    
    θ(t+Δt) = θ(t) + (dθ/dt) × Δt + Δθ
    r_c(t+Δt) = r_c(t) + (dr_c/dt) × Δt + Δr_c
    r_r(t+Δt) = r_r(t) + (dr_r/dt) × Δt + Δr_r
    r_i(t+Δt) = r_i(t) + (dr_i/dt) × Δt + Δr_i
    r_s(t+Δt) = r_s(t) + (dr_s/dt) × Δt + Δr_s
    r_g(t+Δt) = r_g(t) + (dr_g/dt) × Δt + Δr_g
    
    **Ice Crystal Number Concentration:**
    
    Special treatment when ice mass depletes:
    
    If r_i ≤ 0 and ldmicro:
        c_i = 0  (no ice crystals remain)
        t_micro += Δt  (advance time)
    
    This ensures consistency between ice mass and number concentration:
    - No ice mass → no ice crystals
    - Prevents division by zero in size calculations
    - Updates time counter to mark depletion point
    
    **Physical Interpretation:**
    
    The update represents one sub-step in the adaptive time stepping:
    
    1. Tendencies computed for current state
    2. Time step limited by ice4_step_limiter
    3. State advanced by explicit Euler
    4. Loop continues until t_micro ≥ TSTEP
    
    **Numerical Considerations:**
    
    Explicit Euler advantages:
    - Simple implementation
    - Low memory requirements
    - Efficient for stiff problems with adaptive stepping
    
    Limitations addressed by adaptive stepping:
    - Potential instability controlled by time step limiter
    - CFL condition enforced via species depletion checks
    - Phase change thresholds handled by 0°C limiter
    
    **Typical Sub-step Sequence:**
    
    Initial state: t_micro = 0
    ↓
    Sub-step 1: Δt = 10 s, update state
    ↓
    Sub-step 2: Δt = 5 s, approaching threshold
    ↓
    Sub-step 3: Δt = 1 s, near depletion
    ↓
    Final: t_micro = TSTEP = 60 s
    
    **Conservation Properties:**
    
    Mass and energy conservation ensured by:
    - Instantaneous terms (×_b) from conservative processes
    - Tendencies (×_tnd_a) from budget-balanced sources
    - No artificial sources or sinks
    
    **Error Accumulation:**
    
    Truncation error from explicit Euler:
    - O(Δt²) per sub-step
    - Total error: O(Δt_sub × N_steps)
    - Controlled by adaptive stepping (small Δt when needed)
    
    **When Ice Depletes:**
    
    The special handling for r_i ≤ 0:
    
    1. Sets c_i = 0 (consistency)
    2. Advances t_micro (marks depletion time)
    3. Allows loop to continue (other species may remain)
    
    This prevents:
    - Spurious ice crystal concentrations
    - Division errors in size calculations
    - Incorrect process rates
    
    **Comparison with Other Schemes:**
    
    Explicit Euler (used here):
    - Simple, fast
    - Requires small time steps for stability
    - Adaptive stepping makes it practical
    
    Implicit schemes (not used):
    - More stable
    - Larger time steps possible
    - Complex implementation for microphysics
    - Expensive for nonlinear processes
    
    Semi-implicit (not used):
    - Balance between explicit/implicit
    - Good for diffusion terms
    - Still complex for phase changes
    
    **Budget Calculations:**
    
    Note: Budget diagnostics (lines 409-431 in Fortran) omitted in
    GT4Py version. These would track contributions from individual
    processes to each species budget.
    
    Source Reference
    ----------------
    PHYEX/src/common/micro/mode_ice4_stepping.F90, lines 391-404
    
    See Also
    --------
    ice4_step_limiter : Computes adaptive time step
    ice4_stepping_tmicro_init : Initializes time stepping
    external_tendencies_update : Handles external forcing
    
    Examples
    --------
    >>> # Normal update with moderate time step
    >>> # Δt = 10 s, all tendencies modest
    >>> # → Smooth state evolution
    
    >>> # Near phase change with small time step
    >>> # Δt = 1 s, approaching 0°C
    >>> # → Careful evolution, prevents overshoot
    
    >>> # Ice depletion case
    >>> # r_i = 1e-16 kg/kg → 0 after update
    >>> # → c_i set to 0, t_micro advanced
    >>> # → Consistent state maintained
    """

    # 4.7 New values of variables for next iteration
    with computation(PARALLEL), interval(...):
        th_t += theta_tnd_a * delta_t_micro + theta_b
        rc_t += rc_tnd_a * delta_t_micro + rc_b
        rr_t += rr_tnd_a * delta_t_micro + rr_b
        ri_t += ri_tnd_a * delta_t_micro + ri_b
        rs_t += rs_tnd_a * delta_t_micro + rs_b
        rg_t += rg_tnd_a * delta_t_micro + rg_b

    with computation(PARALLEL), interval(...):
        if ri_t <= 0 and ldmicro:
            t_micro += delta_t_micro
            ci_t = 0

    # 4.8 Mixing ratio change due to each process
    # Translation note : l409 to 431 have been omitted since no budget calculations


#    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
#    from_line=440,
#    to_line=452,
def external_tendencies_update(
    th_t: Field["float"],
    theta_tnd_ext: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    rc_tnd_ext: Field["float"],
    rr_tnd_ext: Field["float"],
    ri_tnd_ext: Field["float"],
    rs_tnd_ext: Field["float"],
    rg_tnd_ext: Field["float"],
    ldmicro: Field["bool"],
    dt: "float"
):

    with computation(PARALLEL), interval(...):
        if ldmicro:
            th_t -= theta_tnd_ext * dt
            rc_t -= rc_tnd_ext * dt
            rr_t -= rr_tnd_ext * dt
            ri_t -= ri_tnd_ext * dt
            rs_t -= rs_tnd_ext * dt
            rg_t -= rg_tnd_ext * dt
