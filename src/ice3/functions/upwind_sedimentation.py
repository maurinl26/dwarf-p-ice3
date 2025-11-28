# -*- coding: utf-8 -*-
"""
Upwind sedimentation scheme functions for microphysics.

This module provides GT4Py implementations of functions used in the
upwind sedimentation scheme for hydrometeor species. The scheme handles
vertical transport of precipitation particles with adaptive time stepping
to maintain numerical stability.
"""
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    IJ,
    Field,
    function,
)


@function
def upper_air_flux(
    wsed: Field["float"],
    max_tstep: Field[IJ, "float"],
    TSTEP: "float",
):
    """
    Calculate the upward sedimentation flux at the upper boundary.
    
    This function computes the sedimentation flux entering from above
    based on the sedimentation velocity and the adaptive time step ratio.
    
    Parameters
    ----------
    wsed : Field[float]
        Sedimentation velocity field (m/s), positive downward.
    max_tstep : Field[IJ, float]
        Maximum stable time step for the current iteration (s).
    TSTEP : float
        Full model time step (s).
        
    Returns
    -------
    Field[float]
        Upward flux at the upper boundary, normalized by the time step ratio.
        
    Notes
    -----
    The flux is scaled by the ratio max_tstep/TSTEP to account for
    sub-stepping when the CFL condition requires smaller time steps.
    """
    return wsed * (max_tstep / TSTEP)


@function
def mixing_ratio_update(
    max_tstep: Field[IJ, "float"],
    oorhodz: Field["float"],
    wsed: Field["float"],
    rs: Field["float"],
    r_t: Field["float"],
    TSTEP: "float",
) -> Field["float"]:
    """
    Update mixing ratio due to sedimentation flux divergence.
    
    This function applies the upwind scheme to update the mixing ratio
    based on the vertical flux divergence and local tendencies.
    
    Parameters
    ----------
    max_tstep : Field[IJ, float]
        Maximum stable time step for the current iteration (s).
    oorhodz : Field[float]
        Inverse of (air density * layer thickness), 1/(ρ·Δz) (m/kg).
    wsed : Field[float]
        Sedimentation flux at cell interfaces (kg/(m²·s)).
    rs : Field[float]
        Source/sink tendency for mixing ratio (kg/kg/s).
    r_t : Field[float]
        Current mixing ratio at time t (kg/kg).
    TSTEP : float
        Full model time step (s).
        
    Returns
    -------
    Field[float]
        Updated tendency rs including sedimentation effects (kg/kg/s).
        
    Notes
    -----
    The mixing ratio change is computed from the flux divergence:
    Δr = Δt · (1/ρΔz) · (flux_in - flux_out)
    
    Both r_t and rs are updated in place:
    - r_t accumulates the instantaneous change
    - rs accumulates the tendency normalized by TSTEP
    """
    mrchange = max_tstep[0, 0] * oorhodz * (wsed[0, 0, 1] - wsed[0, 0, 0])
    r_t += mrchange + rs * max_tstep
    rs += mrchange / TSTEP

    return rs


@function
def maximum_time_step(
    rtmin: "float",
    rhodref: Field["float"],
    max_tstep: Field[IJ, "float"],
    r: Field["float"],
    dz: Field["float"],
    wsed: Field["float"],
    remaining_time: Field[IJ, "float"],
) -> Field["float"]:
    """
    Compute maximum stable time step for sedimentation scheme.
    
    This function determines the maximum time step that satisfies the
    CFL (Courant-Friedrichs-Lewy) stability condition for the upwind
    sedimentation scheme.
    
    Parameters
    ----------
    rtmin : float
        Minimum mixing ratio threshold below which sedimentation is ignored (kg/kg).
    rhodref : Field[float]
        Reference air density (kg/m³).
    max_tstep : Field[IJ, float]
        Current maximum time step (s), updated if smaller step is needed.
    r : Field[float]
        Mixing ratio of the sedimenting species (kg/kg).
    dz : Field[float]
        Layer thickness (m).
    wsed : Field[float]
        Sedimentation flux (kg/(m²·s)).
    remaining_time : Field[IJ, float]
        Time remaining in the full time step (s).
        
    Returns
    -------
    Field[float]
        Updated maximum stable time step (s).
        
    Notes
    -----
    The CFL condition for stability is:
    Δt ≤ SPLIT_MAXCFL * (ρ · r · Δz) / flux
    
    where SPLIT_MAXCFL is an external safety factor (typically < 1).
    
    The time step is only computed when:
    - r > rtmin (sufficient mass present)
    - wsed > 1e-20 (non-negligible flux)
    - remaining_time > 0 (time remains in step)
    """
    from __externals__ import SPLIT_MAXCFL

    tstep = max_tstep
    if r > rtmin and wsed > 1e-20 and remaining_time > 0:
        tstep[0, 0] = min(
            max_tstep,
            SPLIT_MAXCFL * rhodref[0, 0, 0] * r[0, 0, 0] * dz[0, 0, 0] / wsed[0, 0, 0],
        )

    return tstep


@function
def instant_precipitation(
    wsed: Field["float"], max_tstep: Field["float"], TSTEP: "float"
) -> Field["float"]:
    """
    Calculate instantaneous precipitation rate at the surface.
    
    This function converts the sedimentation flux at the ground into
    a precipitation rate in standard units (mm/s or mm/h).
    
    Parameters
    ----------
    wsed : Field[float]
        Sedimentation flux at the ground level (kg/(m²·s)).
    max_tstep : Field[float]
        Time step used for the sedimentation calculation (s).
    TSTEP : float
        Full model time step (s).
        
    Returns
    -------
    Field[float]
        Instantaneous precipitation rate (mm/s), assuming the flux
        represents liquid water equivalent.
        
    Notes
    -----
    The conversion uses RHOLW (density of liquid water, typically
    1000 kg/m³) as an external constant.
    
    The formula is:
    precip = (flux / ρ_water) * (max_tstep / TSTEP)
    
    This gives precipitation in m/s, which equals mm/s when
    considering the density conversion.
    """
    from __externals__ import RHOLW

    return wsed[0, 0, 0] / RHOLW * (max_tstep / TSTEP)
