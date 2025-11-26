# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field

# "PHYEX/src/common/micro/mode_ice4_rrhong.F90"
def ice4_rrhong(
    ldcompute: Field["bool"],
    t: Field["float"],
    exn: Field["float"],
    lvfact: Field["float"],
    lsfact: Field["float"],
    tht: Field["float"],  # theta at time t
    rrhong_mr: Field["float"],
    rrt: Field["float"],  # rain water mixing ratio at t
):
    """
    Compute spontaneous (homogeneous) freezing of supercooled rain drops.
    
    This stencil computes the RRHONG process: instantaneous freezing of
    liquid rain drops at very cold temperatures. When temperature drops
    below approximately -35°C to -40°C, supercooled rain drops become
    unstable and freeze spontaneously into graupel/frozen drops, following
    Fletcher's classical nucleation theory.
    
    Parameters
    ----------
    ldcompute : Field[bool]
        Computation mask (True = compute, False = skip). Input field.
    t : Field[float]
        Temperature (K). Input field.
    exn : Field[float]
        Exner function π = (p/p₀)^(R_d/c_pd) (dimensionless). Input field.
    lvfact : Field[float]
        Vaporization latent heat factor L_v/(c_ph×π) (K). Input field.
        Used for temperature feedback calculation.
    lsfact : Field[float]
        Sublimation latent heat factor L_s/(c_ph×π) (K). Input field.
        Used for temperature feedback calculation.
    tht : Field[float]
        Potential temperature θ (K). Input field.
    rrhong_mr : Field[float]
        Rain homogeneous freezing rate: r_r → r_g (kg/kg/s). Output field.
    rrt : Field[float]
        Rain mixing ratio (kg/kg). Input field.
    
    Returns
    -------
    None
        Modifies rrhong_mr in place.
    
    Notes
    -----
    **Physical Process:**
    
    Homogeneous freezing (RRHONG) represents the spontaneous freezing
    of supercooled rain drops without the need for ice nuclei. This is
    a rapid, irreversible phase transition that occurs when:
    
    1. Temperature drops below critical threshold (~-35°C to -40°C)
    2. Molecular thermal motion insufficient to prevent ice crystal formation
    3. All liquid water in drops converts to ice instantaneously
    
    **Freezing Condition:**
    
    T < T_t - 35°C  (T < 238.15 K)
    
    This threshold represents the homogeneous freezing point where pure
    water droplets freeze without ice nuclei, following classical
    nucleation theory (Fletcher, 1962).
    
    **Conversion Rate:**
    
    When conditions are met:
    
    dr_r/dt = -r_r  (instantaneous, all rain freezes)
    
    The frozen rain becomes graupel (r_g), representing dense ice particles
    with embedded liquid water pockets that quickly freeze.
    
    **Temperature Feedback (LFEEDBACKT=True):**
    
    Freezing releases latent heat, warming the environment:
    
    ΔT = (L_s - L_v) × Δr_r / c_ph
    
    where:
    - L_s: Sublimation latent heat (~2.83×10⁶ J/kg)
    - L_v: Vaporization latent heat (~2.50×10⁶ J/kg)
    - L_m = L_s - L_v: Melting latent heat (~3.35×10⁵ J/kg)
    
    To prevent unphysical temperature overshoot (T > -35°C after freezing),
    the freezing rate is limited:
    
    dr_r/dt ≤ max(0, ((T_t - 35°C)/π - θ) / (L_s_fact - L_v_fact))
    
    This ensures T_final ≈ -35°C, preventing the temperature from rising
    above the freezing threshold due to latent heat release.
    
    **Without Feedback (LFEEDBACKT=False):**
    
    All available rain freezes instantaneously:
    
    dr_r/dt = -r_r
    
    Temperature feedback ignored (diagnostic mode or for testing).
    
    **Physical Interpretation:**
    
    RRHONG represents:
    - The cold-cloud limit where liquid phase becomes impossible
    - Conversion of rain to graupel/hail embryos
    - Source term for graupel in deep convective systems
    - Temperature "floor" at approximately -35°C to -40°C
    
    **Graupel Formation:**
    
    Frozen rain drops become graupel because:
    - Rapid freezing traps air bubbles → lower density than pure ice
    - Surface remains rough → high collection efficiency
    - Falls faster than snow → continues to grow by riming
    
    **Comparison with Other Freezing Processes:**
    
    Homogeneous nucleation (RRHONG):
    - Very cold (T < -35°C)
    - No ice nuclei needed
    - Instantaneous
    - All drops freeze
    - Rain → Graupel
    
    Heterogeneous nucleation (contact, immersion):
    - Warmer (-10°C to -35°C)
    - Requires ice nuclei (INP)
    - Rate depends on INP concentration
    - Partial freezing
    - Cloud → Ice crystals
    
    Riming (accretion):
    - Any T < 0°C
    - Liquid collected by falling ice
    - Continuous growth
    - Requires both phases
    - Various conversions
    
    **Timescale:**
    
    τ ~ 0.01-1 s (essentially instantaneous on model timescale)
    
    Once threshold is crossed, freezing completes within milliseconds.
    Model treats as instantaneous adjustment.
    
    **Threshold Uncertainty:**
    
    Exact threshold varies:
    - Pure water: ~-38°C (laboratory)
    - Natural rain: -35°C to -40°C (depends on drop size, impurities)
    - Smaller drops: slightly lower threshold (surface effects)
    - Larger drops: slightly higher threshold (volume effects)
    
    ICE4 uses -35°C as practical threshold representing typical
    atmospheric conditions.
    
    **Temperature Limitation Logic:**
    
    The feedback limiter prevents:
    
    1. Unrealistic warming: Freezing 1 g/kg of rain releases enough
       heat to warm air by ~8 K, potentially bringing T > -35°C
    
    2. Oscillations: Without limiter, T could oscillate across threshold,
       causing numerical instability
    
    3. Energy conservation: Ensures latent heat release consistent with
       temperature change
    
    The limiter computes maximum freezable rain:
    
    r_max = (T_threshold - T_current) / (L_m/c_ph)
    
    Only enough rain freezes to bring T up to threshold, remaining
    rain stays liquid (temporarily, until next time step).
    
    **Numerical Considerations:**
    
    - Process checked at each time step
    - Binary activation (on/off based on temperature)
    - Rate limiter prevents overshooting
    - Diagnostic when LFEEDBACKT=False
    
    **Physical Realism:**
    
    This parameterization captures:
    - Observed temperature floor in deep convection
    - Graupel production in anvil regions
    - Liquid water limit in cold clouds
    - Thermodynamic consistency (energy conservation)
    
    Source Reference
    ----------------
    PHYEX/src/common/micro/mode_ice4_rrhong.F90
    
    See Also
    --------
    ice4_slow : Other slow processes including nucleation
    ice4_rimltc : Melting of frozen hydrometeors
    ice4_fast_rg : Graupel processes
    
    References
    ----------
    Fletcher, N.H., 1962: The Physics of Rainclouds. Cambridge
    University Press, 386 pp.
    
    Pruppacher, H.R., and J.D. Klett, 1997: Microphysics of Clouds
    and Precipitation, 2nd ed. Kluwer Academic Publishers, 954 pp.
    
    Examples
    --------
    >>> # Very cold cloud with supercooled rain
    >>> # T = -40°C, r_r = 2e-3 kg/kg
    >>> # → All rain freezes: rrhong_mr = 2e-3 kg/kg/s
    >>> # → Becomes graupel immediately
    
    >>> # Approaching threshold
    >>> # T = -34°C, r_r = 1e-3 kg/kg
    >>> # → No freezing yet: rrhong_mr = 0
    >>> # → Rain remains supercooled (metastable)
    
    >>> # With feedback limiter
    >>> # T = -36°C, r_r = 5e-3 kg/kg, large latent heat
    >>> # → Partial freezing: rrhong_mr limited to prevent T > -35°C
    >>> # → Remaining rain freezes in subsequent time steps
    """

    from __externals__ import LFEEDBACKT, R_RTMIN, TT

    # 3.3 compute the spontaneous frezzing source: RRHONG
    with computation(PARALLEL), interval(...):
        if (
            t < TT - 35.0 
            and rrt > R_RTMIN 
            and ldcompute
        ):
            # limitation for -35 degrees crossing
            rrhong_mr = rrt
            if LFEEDBACKT:
                rrhong_mr = min(rrhong_mr, max(0., ((TT - 35.) / exn - tht) / (lsfact - lvfact)))

        else:
            rrhong_mr = 0.0
