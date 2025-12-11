# -*- coding: utf-8 -*-
"""
Fast ice crystal processes - Bergeron-Findeisen effect for ICE4 scheme.

This module implements the Bergeron-Findeisen effect, where cloud droplets
evaporate to provide water vapor that deposits onto ice crystals. This
preferential growth of ice at the expense of liquid water occurs in
mixed-phase clouds due to the lower saturation vapor pressure over ice
compared to liquid water at the same temperature.

Source: PHYEX/src/common/micro/mode_ice4_fast_ri.F90
"""
from __future__ import annotations

from gt4py.cartesian.gtscript import (PARALLEL, Field, computation,
                                      interval)


def ice4_fast_ri(
    ldcompute: Field["bool"],
    rhodref: Field["float"],
    ai: Field["float"],
    cj: Field["float"],
    cit: Field["float"],
    ssi: Field["float"],
    rct: Field["float"],
    rit: Field["float"],
    rc_beri_tnd: Field["float"],
    ldsoft: "bool",
):
    """
    Compute the Bergeron-Findeisen effect (RCBERI tendency).
    
    This function calculates the rate at which cloud droplets evaporate
    to deposit as ice on existing ice crystals in supersaturated conditions
    with respect to ice. This process is fundamental in mixed-phase clouds
    and leads to rapid glaciation.
    
    Parameters
    ----------
    ldcompute : Field[bool]
        Mask indicating which grid points require microphysical computation.
    rhodref : Field[float]
        Reference air density (kg/m³).
    ai : Field[float]
        Thermodynamical function for vapor diffusion (m²·s·kg⁻¹).
        Related to vapor diffusivity and thermal conductivity.
    cj : Field[float]
        Ventilation coefficient function (dimensionless).
        Accounts for enhanced deposition due to particle motion.
    cit : Field[float]
        Ice crystal number concentration (m⁻³).
    ssi : Field[float]
        Supersaturation with respect to ice (dimensionless).
        ssi = (e/e_sat_i) - 1, where ssi > 0 indicates supersaturation.
    rct : Field[float]
        Cloud liquid water mixing ratio at time t (kg/kg).
    rit : Field[float]
        Pristine ice crystal mixing ratio at time t (kg/kg).
    rc_beri_tnd : Field[float]
        Output: Tendency for cloud water due to Bergeron-Findeisen effect (kg/kg/s).
        Negative values indicate evaporation of cloud droplets.
    ldsoft : bool
        If True, use previously computed tendencies without recalculation.
        If False, compute new tendencies.
        
    Notes
    -----
    Physical Process:
    The Bergeron-Findeisen effect occurs when:
    - Air is supersaturated with respect to ice (ssi > 0)
    - Air is subsaturated with respect to liquid water
    - Both liquid droplets and ice crystals coexist
    
    In this condition:
    - Cloud droplets evaporate (providing water vapor)
    - Water vapor deposits onto ice crystals
    - Net result: conversion of liquid to ice
    
    The deposition rate depends on:
    - Ice crystal size (via lambda_i, computed from rit and cit)
    - Supersaturation magnitude (ssi)
    - Ventilation effects (enhanced by particle motion, via cj)
    - Crystal number concentration (cit)
    
    Algorithm:
    1. Check if conditions are met (ssi > 0, sufficient rct, rit, cit)
    2. Compute ice crystal size parameter lambda_i from mass and number
    3. Calculate deposition rate using diffusion theory
    4. Include ventilation correction for moving particles
    
    External Parameters:
    - C_RTMIN, I_RTMIN: Minimum thresholds for cloud and ice (kg/kg)
    - LBI, LBEXI: Parameters for ice crystal size distribution
    - O0DEPI, O2DEPI: Deposition coefficients (no ventilation, with ventilation)
    - DI: Exponent for ice crystal size distribution
    
    The lambda_i calculation is capped at 1e8 m⁻¹ to avoid numerical issues
    with very small ice crystals.
    
    References
    ----------
    Bergeron, T., 1935: On the physics of clouds and precipitation.
    Proces Verbaux de l'Association de Météorologie, UGGI, 156–178.
    
    Findeisen, W., 1938: Die kolloidmeteorologischen Vorgänge bei der
    Niederschlagsbildung (Colloidal meteorological processes in the
    formation of precipitation). Meteor. Z., 55, 121–133.
    """
    from __externals__ import C_RTMIN, DI, I_RTMIN, LBEXI, LBI, O0DEPI, O2DEPI

    # 7.2 Bergeron-Findeisen effect: RCBERI
    with computation(PARALLEL), interval(...):
        if (
            ssi > 0
            and rct > C_RTMIN
            and rit > I_RTMIN
            and cit > 1e-20
            and ldcompute
        ):
            if not ldsoft:
                # Compute ice crystal slope parameter lambda_i
                rc_beri_tnd = min(
                    1e8, LBI * (rhodref * rit / cit) ** LBEXI
                )  # lambda_i
                
                # Compute deposition rate with ventilation correction
                rc_beri_tnd = (
                    (ssi / (rhodref * ai))
                    * cit
                    * (O0DEPI / rc_beri_tnd + O2DEPI * cj ** 2 / rc_beri_tnd ** (DI + 2.0))
                )

        else:
            rc_beri_tnd = 0
