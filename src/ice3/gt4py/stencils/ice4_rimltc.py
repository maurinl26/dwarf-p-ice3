# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import PARALLEL, Field, computation, interval


# "PHYEX/src/common/micro/mode_ice4_rimltc.F90"
def ice4_rimltc(
    ldcompute: Field["bool"],
    t: Field["float"],
    exn: Field["float"],
    lvfact: Field["float"],
    lsfact: Field["float"],
    tht: Field["float"],  # theta at time t
    rit: Field["float"],  # rain water mixing ratio at t
    rimltc_mr: Field["float"],
):
    """
    Compute melting of cloud ice crystals above freezing temperature.
    
    This stencil computes the RIMLTC process: instantaneous melting of
    ice crystals when temperature rises above 0°C. This phase transition
    converts ice crystals to cloud droplets, releasing latent heat of
    melting and providing a thermodynamic constraint on the melting layer.
    
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
    rit : Field[float]
        Ice crystal mixing ratio (kg/kg). Input field.
    rimltc_mr : Field[float]
        Ice melting rate: r_i → r_c (kg/kg/s). Output field.
    
    Returns
    -------
    None
        Modifies rimltc_mr in place.
    
    Notes
    -----
    **Physical Process:**
    
    Ice crystal melting (RIMLTC) represents the phase transition from
    ice to liquid water when temperature exceeds the melting point.
    This is a fundamental process that:
    
    1. Defines the 0°C bright band in radar observations
    2. Consumes latent heat, cooling the environment
    3. Converts ice crystals to cloud droplets
    4. Creates the melting layer in stratiform precipitation
    
    **Melting Condition:**
    
    T > T_t = 273.15 K  (T > 0°C)
    
    Ice crystals are thermodynamically unstable above freezing and
    melt rapidly when exposed to positive temperatures.
    
    **Conversion Rate:**
    
    When conditions are met:
    
    dr_i/dt = -r_i  (instantaneous, all ice melts)
    
    The melted ice becomes cloud droplets (r_c), which may then
    undergo warm rain processes (autoconversion, accretion).
    
    **Temperature Feedback (LFEEDBACKT=True):**
    
    Melting consumes latent heat, cooling the environment:
    
    ΔT = -(L_s - L_v) × Δr_i / c_ph
    
    where:
    - L_s: Sublimation latent heat (~2.83×10⁶ J/kg)
    - L_v: Vaporization latent heat (~2.50×10⁶ J/kg)
    - L_m = L_s - L_v: Melting latent heat (~3.35×10⁵ J/kg)
    
    Note: Melting is an endothermic process (ΔT < 0), opposite of freezing.
    
    To prevent unphysical temperature undershoot (T < 0°C after melting),
    the melting rate is limited:
    
    dr_i/dt ≤ max(0, (θ - T_t/π) / (L_s_fact - L_v_fact))
    
    This ensures T_final ≈ 0°C, preventing the temperature from dropping
    below the melting threshold due to latent heat consumption.
    
    **Without Feedback (LFEEDBACKT=False):**
    
    All available ice melts instantaneously:
    
    dr_i/dt = -r_i
    
    Temperature feedback ignored (diagnostic mode or for testing).
    
    **Physical Interpretation:**
    
    RIMLTC represents:
    - The warm-cloud limit where ice phase becomes impossible
    - Bright band in radar (enhanced reflectivity from melting)
    - Conversion of ice to liquid in warm layer
    - Temperature "ceiling" at approximately 0°C
    
    **Melting Layer Characteristics:**
    
    In stratiform precipitation, melting creates distinctive features:
    
    1. **Bright Band:**
       - Enhanced radar reflectivity at melting level
       - Due to water coating on ice (larger dielectric constant)
       - Particles also aggregate during melting (larger size)
    
    2. **Temperature Profile:**
       - Isothermal layer near 0°C
       - Latent heat cooling balances adiabatic warming
       - "Onion layer" structure in soundings
    
    3. **Particle Transition:**
       - Ice → Wet ice/slush → Liquid drops
       - Density decreases during melting
       - Fall speed increases (ice → rain)
    
    **Comparison with Other Phase Change Processes:**
    
    Melting (RIMLTC):
    - Warm side of 0°C (T > 0°C)
    - Endothermic (cools environment)
    - Instantaneous (parameterized)
    - Ice → Cloud droplets
    - All ice melts
    
    Freezing (RRHONG, RCHONI):
    - Cold side of 0°C (T < 0°C, typically < -35°C)
    - Exothermic (warms environment)
    - Instantaneous (parameterized)
    - Liquid → Ice/Graupel
    - All liquid freezes
    
    Deposition/Sublimation:
    - Any T < 0°C
    - Vapor ↔ Ice directly
    - Continuous (diffusion-limited)
    - Depends on supersaturation
    - Gradual process
    
    **Timescale:**
    
    τ ~ 1-100 s (rapid but not instantaneous)
    
    In reality, melting takes finite time (~1-10 minutes for precipitation
    particles), but model treats as instantaneous adjustment relative to
    model time step.
    
    **Energy Budget:**
    
    Melting consumes approximately:
    - 335 J/g (latent heat of melting)
    - For 1 g/kg ice: ΔT ≈ -0.3 K
    
    This cooling is significant and must be accounted for to:
    - Maintain energy conservation
    - Prevent spurious temperature changes
    - Represent melting layer dynamics correctly
    
    **Temperature Limitation Logic:**
    
    The feedback limiter prevents:
    
    1. **Unrealistic cooling:** Melting large amounts of ice could
       cool air below 0°C, creating unphysical situation
    
    2. **Oscillations:** Without limiter, T could oscillate across
       melting point, causing numerical instability
    
    3. **Energy conservation:** Ensures latent heat consumption
       consistent with temperature change
    
    The limiter computes maximum meltable ice:
    
    r_max = (T_current - T_threshold) / (L_m/c_ph)
    
    Only enough ice melts to bring T down to 0°C, remaining ice
    stays frozen (unphysical, but prevents numerical issues).
    
    **Practical Considerations:**
    
    In nature, melting layer is typically:
    - 200-500 m thick
    - Located just below 0°C isotherm
    - Characterized by rapid changes in hydrometeor properties
    
    Model must resolve this layer adequately:
    - Vertical resolution important
    - Time step affects melting rate
    - Feedback crucial for realistic profiles
    
    **Applications:**
    
    RIMLTC is important for:
    - Radar forward operators (simulating bright band)
    - Surface precipitation type (rain vs snow)
    - Aviation icing forecasts (wet snow/slush)
    - Flood prediction (rain intensity below melting layer)
    
    **Cloud Microphysics Context:**
    
    Melting fits into overall ice-liquid transitions:
    
    Freezing processes (liquid → ice):
    - Homogeneous nucleation (< -35°C)
    - Heterogeneous nucleation (-10°C to -35°C)
    - Riming (0°C to -40°C)
    
    Melting processes (ice → liquid):
    - RIMLTC: Ice crystals melt (> 0°C)
    - RSMLTR: Snow melts (> 0°C, not shown here)
    - RGMLTR: Graupel melts (> 0°C, not shown here)
    
    **Numerical Stability:**
    
    - Binary activation (on/off based on temperature)
    - Rate limiter prevents temperature overshoot
    - All ice melts unless limited by feedback
    - Simple, robust parameterization
    
    Source Reference
    ----------------
    PHYEX/src/common/micro/mode_ice4_rimltc.F90
    
    See Also
    --------
    ice4_rrhong : Homogeneous freezing of rain
    ice4_slow : Other slow processes including deposition
    ice4_fast_rs : Snow processes (including snow melting)
    
    References
    ----------
    Pruppacher, H.R., and J.D. Klett, 1997: Microphysics of Clouds
    and Precipitation, 2nd ed. Kluwer Academic Publishers, 954 pp.
    
    Stewart, R.E., 1985: Precipitation types in winter storms.
    Pure Appl. Geophys., 123, 597-609.
    
    Examples
    --------
    >>> # Warm air advection, ice present above freezing
    >>> # T = +2°C, r_i = 1e-4 kg/kg
    >>> # → All ice melts: rimltc_mr = 1e-4 kg/kg/s
    >>> # → Becomes cloud droplets, may autoconvert to rain
    
    >>> # Just below freezing
    >>> # T = -1°C, r_i = 5e-4 kg/kg
    >>> # → No melting: rimltc_mr = 0
    >>> # → Ice remains frozen
    
    >>> # With feedback limiter, large ice content
    >>> # T = +1°C, r_i = 3e-3 kg/kg
    >>> # → Partial melting: rimltc_mr limited to prevent T < 0°C
    >>> # → Latent heat cooling brings T close to 0°C
    >>> # → Melting layer maintained near freezing level
    """

    from __externals__ import LFEEDBACKT, TT

    with computation(PARALLEL), interval(...):
        # 7.1 cloud ice melting
        if rit > 0 and t > TT and ldcompute:
            rimltc_mr = rit

            # limitation due to zero crossing of temperature
            if LFEEDBACKT:
                rimltc_mr = min(
                    rimltc_mr, max(0, (tht - TT / exn) / (lsfact - lvfact))
                )

        else:
            rimltc_mr = 0
