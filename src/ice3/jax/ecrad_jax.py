# -*- coding: utf-8 -*-
"""
ecRad JAX Parameterization - Minimal Version.

This module provides a simple, structured JAX interface for the ecRad 
radiation scheme. Direct execution of the full ECMWF ecRad in a pure JAX 
graph typically requires an XLA CustomCall or FFI binding.

This minimal, documented JAX version defines the correct input/output arrays,
allowing it to be seamlessly compiled (`@jax.jit`) into the `arome_physics.py`
orchestrator.

To implement the true physics in JAX natively in the future, the internal `_compute_fluxes` 
method would be filled with two-stream solver approximations or gas-optics lookup tables.
"""

from typing import NamedTuple, Tuple, Dict
import jax
import jax.numpy as jnp
from jax import Array

class EcRadState(NamedTuple):
    """
    Input state for the active atmospheric columns.
    Typical dimensions: (n_columns, n_levels) unless specified.
    """
    pres: Array           # Pressure at layer centers (Pa)
    pres_hl: Array        # Pressure at layer half-levels (Pa) (n_columns, n_levels+1)
    temp: Array           # Temperature (K)
    q: Array              # Specific humidity (kg/kg)
    q_liquid: Array       # Cloud liquid water mixing ratio (kg/kg)
    q_ice: Array          # Cloud ice mixing ratio (kg/kg)
    cloud_frac: Array     # Cloud fraction (0-1)
    albedo_sw: Array      # Surface shortwave albedo (n_columns,)
    emissivity_lw: Array  # Surface longwave emissivity (n_columns,)
    cos_zenith: Array     # Cosine of solar zenith angle (n_columns,)

class EcRadFluxes(NamedTuple):
    """
    Output radiative fluxes and tendencies.
    """
    sw_dn: Array          # Shortwave downward flux at half-levels (W m-2) (n_columns, n_levels+1)
    sw_up: Array          # Shortwave upward flux at half-levels (W m-2) (n_columns, n_levels+1)
    lw_dn: Array          # Longwave downward flux at half-levels (W m-2) (n_columns, n_levels+1)
    lw_up: Array          # Longwave upward flux at half-levels (W m-2) (n_columns, n_levels+1)
    temp_tendency: Array  # Temperature tendency (K/s) (n_columns, n_levels)


class EcRadJAX:
    """
    JAX-compatible interface for ecRad radiation parameterization.

    Theoretical Framework
    --------------------
    Computes radiative fluxes (shortwave and longwave) and the resulting
    temperature tendencies across atmospheric columns.

    1. **Radiative Transfer Equation (RTE)**:
       The change in spectral radiance $I_{\\nu}$ along path $s$ follows Schwarzschild's equation:
       $$ \\frac{dI_{\\nu}}{ds} = -k_{\\nu} (I_{\\nu} - B_{\\nu}(T)) $$
       where $k_{\\nu}$ is the absorption coefficient and $B_{\\nu}(T)$ is the Planck function.
       The spectral integration is approximated using the Correlated-k method (RRTMG).

    2. **Flux Divergence and Temperature Tendency**:
       Layer heating rates are driven by the vertical divergence of the net flux:
       $$ F_{net} = (F^{\\uparrow}_{SW} - F^{\\downarrow}_{SW}) + (F^{\\uparrow}_{LW} - F^{\\downarrow}_{LW}) $$
       $$ \\frac{\\partial T}{\\partial t} = \\frac{g}{C_p} \\frac{\\partial F_{net}}{\\partial p} $$
       where $g$ is gravity, $C_p$ is specific heat at constant pressure, and $p$ is pressure.

    3. **Cloud Solvers (McICA)**:
       Sub-grid cloud overlaps are resolved via the Monte Carlo Independent Column 
       Approximation (McICA). This statistically samples cloud state profiles 
       relying on an assumed overlap decorrelation length (e.g., exponential-random).

    Scientific References
    --------------------
    - Hogan, R. J., & Bozzo, A. (2018). A flexible and efficient radiation 
      scheme for the ECMWF model. *J. Adv. Model. Earth Syst.*, 10, 1990-2008.
      https://doi.org/10.1029/2018MS001364
    - Mlawer, E. J., et al. (1997). Radiative transfer for inhomogeneous 
      atmospheres: RRTM, a validated correlated-k model for the longwave. 
      *J. Geophys. Res.*, 102, 16663-16682.
    - Pincus, R., Barker, H. W., & Morcrette, J.-J. (2003). A fast, flexible, 
      approximate technique for computing radiative transfer in inhomogeneous 
      cloud fields. *J. Geophys. Res.*, 108(D13), 4376.
    """
    def __init__(self, use_jit: bool = True):
        """
        Initialize the ecRad JAX simplified parameterization.

        Parameters
        ----------
        use_jit : bool
            Enable or disable JIT compilation of the step.
        """
        self.g = 9.80665    # Gravity (m/s2)
        self.cp = 1004.0    # Specific heat of dry air (J/kg/K)
        
        if use_jit:
            self.compute = jax.jit(self._compute)
        else:
            self.compute = self._compute

    def _compute(self, state: EcRadState, dt: float) -> Tuple[EcRadFluxes, Dict]:
        """
        Internal pure-JAX computation of radiative fluxes and tendencies.
        
        Currently acts as a minimal structural stand-in to ensure shapes,
        types, and variable flow constraints are met within an overarching
        JAX orchestration loop like AROME physics.
        """
        n_cols, n_levs = state.temp.shape
        
        # -------------------------------------------------------------
        # 1. Simplified Flux Computations (Placeholders)
        # -------------------------------------------------------------
        # In a real JAX port of ecRad, this section would run McICA,
        # the RRTMG gas optics formulation, and the two-stream solvers.
        
        # Output flux arrays
        sw_dn = jnp.zeros((n_cols, n_levs + 1))
        sw_up = jnp.zeros((n_cols, n_levs + 1))
        lw_dn = jnp.zeros((n_cols, n_levs + 1))
        lw_up = jnp.zeros((n_cols, n_levs + 1))
        
        # Solar forcing at Top of Atmosphere (approx 1361 W/m2)
        toa_sw_dn = state.cos_zenith * 1361.0 * (state.cos_zenith > 0)
        sw_dn = sw_dn.at[:, 0].set(toa_sw_dn)
        
        # Minimal Transmission Approximation for SW
        # Attenuate SW down roughly based on mass and cloud fraction
        transmission = 1.0 - jnp.clip(state.cloud_frac * 0.5, 0.0, 1.0)
        
        def sw_attenuation_scan(carry, x):
            # carry is incoming sw_dn to the layer, x is transmission
            sw_out = carry * x
            return sw_out, sw_out
            
        # Very crude proxy logic just to produce arrays that map gradients
        _, layer_sw_dn = jax.lax.scan(sw_attenuation_scan, sw_dn[:, 0], transmission.T)
        sw_dn = sw_dn.at[:, 1:].set(layer_sw_dn.T)

        # -------------------------------------------------------------
        # 2. Computations of Tendencies
        # -------------------------------------------------------------
        # Flux divergence drives the temperature tendency
        # dT/dt = -g / (Cp) * dF/dp
        
        # Net flux at each half level (F_up - F_dn)
        net_flux = (sw_up + lw_up) - (sw_dn + lw_dn)
        
        # dF = Flux(level+1) - Flux(level) -> downward is increasing pressure
        d_flux = net_flux[:, 1:] - net_flux[:, :-1]
        d_pres = state.pres_hl[:, 1:] - state.pres_hl[:, :-1]
        
        # dT/dt = (g / cp) * (dF_net / dp)
        # Factor applies correctly if d_flux and d_pres are consistent
        temp_tendency = (self.g / self.cp) * (d_flux / jnp.maximum(d_pres, 1.0))
        
        fluxes = EcRadFluxes(
            sw_dn=sw_dn, sw_up=sw_up, 
            lw_dn=lw_dn, lw_up=lw_up, 
            temp_tendency=temp_tendency
        )
        
        diagnostics = {
            "net_flux": net_flux,
            "toa_sw": toa_sw_dn
        }
        
        return fluxes, diagnostics

    def __call__(self, state: EcRadState, dt: float) -> Tuple[EcRadFluxes, Dict]:
        return self.compute(state, dt)
