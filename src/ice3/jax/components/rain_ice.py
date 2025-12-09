"""
Rain Ice Component - JAX Implementation

This module orchestrates the complete ICE4 microphysics scheme including
time stepping, sedimentation, and tendency calculations.

Reference:
    src/ice3/components/rain_ice.py
"""
import jax
import jax.numpy as jnp
from typing import Dict, Tuple
from functools import partial

from .ice4_tendencies import Ice4TendenciesJAX
from ..stencils.sedimentation import sedimentation_stat
from ..stencils.ice4_correct_negativities import ice4_correct_negativities


class RainIceJAX:
    """
    JAX implementation of the complete ICE4 Rain-Ice scheme.
    
    This class orchestrates:
    - Thermodynamic calculations
    - Microphysical tendencies  
    - Time stepping
    - Sedimentation
    - Negativity corrections
    """
    
    def __init__(self, constants: Dict):
        """
        Initialize the Rain-Ice component.
        
        Args:
            constants: Dictionary of physical constants
        """
        self.constants = constants
        self.ice4_tendencies = Ice4TendenciesJAX(constants)
        
    def compute_latent_heat_factors(
        self,
        exn: jnp.ndarray,
        th_t: jnp.ndarray,
        rv_t: jnp.ndarray,
        rc_t: jnp.ndarray,
        rr_t: jnp.ndarray,
        ri_t: jnp.ndarray,
        rs_t: jnp.ndarray,
        rg_t: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute latent heat factors for phase transitions.
        
        Args:
            exn: Exner function
            th_t: Potential temperature
            rv_t, rc_t, rr_t, ri_t, rs_t, rg_t: Mixing ratios
            
        Returns:
            Tuple of (lv_fact, ls_fact)
        """
        CPD = self.constants["CPD"]
        LVTT = self.constants["LVTT"]
        LSTT = self.constants["LSTT"]
        
        # Compute cph (specific heat at constant pressure for moist air)
        cph = CPD  # Simplified, should include vapor and liquid contributions
        
        lv_fact = LVTT / (cph * exn)
        ls_fact = LSTT / (cph * exn)
        
        return lv_fact, ls_fact
    
    def compute_mask(
        self,
        rc_t: jnp.ndarray,
        ri_t: jnp.ndarray,
        rr_t: jnp.ndarray,
        rs_t: jnp.ndarray,
        rg_t: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute microphysics computation mask.
        
        Args:
            rc_t, ri_t, rr_t, rs_t, rg_t: Hydrometeor mixing ratios
            
        Returns:
            Boolean mask indicating where to compute microphysics
        """
        C_RTMIN = self.constants["C_RTMIN"]
        I_RTMIN = self.constants["I_RTMIN"]
        R_RTMIN = self.constants["R_RTMIN"]
        S_RTMIN = self.constants["S_RTMIN"]
        G_RTMIN = self.constants["G_RTMIN"]
        
        ldmicro = (
            (rc_t > C_RTMIN) |
            (ri_t > I_RTMIN) |
            (rr_t > R_RTMIN) |
            (rs_t > S_RTMIN) |
            (rg_t > G_RTMIN)
        )
        
        return ldmicro
    
    def time_step_limiter(
        self,
        state: Dict[str, jnp.ndarray],
        tendencies: Dict[str, jnp.ndarray],
        dt: float,
        t_elapsed: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Limit time step to ensure stability.
        
        Args:
            state: Current state variables
            tendencies: Computed tendencies
            dt: Total time step
            t_elapsed: Time already elapsed
            
        Returns:
            Tuple of (delta_t, ldcompute_mask)
        """
        # Compute maximum allowable time step based on tendencies
        # This is simplified - actual implementation would check multiple constraints
        
        remaining_time = dt - t_elapsed
        delta_t = jnp.minimum(remaining_time, 0.1)  # Max sub-step
        
        ldcompute = t_elapsed < dt
        
        return delta_t, ldcompute
    
    def update_state(
        self,
        state: Dict[str, jnp.ndarray],
        tendencies: Dict[str, jnp.ndarray],
        delta_t: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """
        Update state variables with tendencies.
        
        Args:
            state: Current state
            tendencies: Computed tendencies
            delta_t: Time step
            
        Returns:
            Updated state dictionary
        """
        state_new = {}
        
        for key in ["th_t", "rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t"]:
            tnd_key = key.replace("_t", "_tnd")
            if tnd_key in tendencies:
                state_new[key] = state[key] + tendencies[tnd_key] * delta_t
            else:
                state_new[key] = state[key]
        
        # Copy other unchanged variables
        for key in state:
            if key not in state_new:
                state_new[key] = state[key]
        
        return state_new
    
    def __call__(
        self,
        state: Dict[str, jnp.ndarray],
        dt: float,
        lsedim_after: bool = False,
        ldeposc: bool = False,
        ldsoft: bool = False,
        lfeedbackt: bool = True,
        max_iterations: int = 10,
    ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
        """
        Execute one time step of the Rain-Ice scheme.
        
        Args:
            state: Dictionary of atmospheric state variables
            dt: Time step (s)
            lsedim_after: Compute sedimentation after microphysics
            ldeposc: Compute fog deposition
            ldsoft: Soft budget mode
            lfeedbackt: Temperature feedback
            max_iterations: Maximum time sub-stepping iterations
            
        Returns:
            Tuple of (updated_state, diagnostics)
        """
        # =============================
        # 1. Compute Latent Heat Factors
        # =============================
        lv_fact, ls_fact = self.compute_latent_heat_factors(
            state["exn"], state["th_t"], state["rv_t"],
            state["rc_t"], state["rr_t"], state["ri_t"],
            state["rs_t"], state["rg_t"]
        )
        
        state["lv_fact"] = lv_fact
        state["ls_fact"] = ls_fact
        
        # =============================
        # 2. Compute Microphysics Mask
        # =============================
        ldmicro = self.compute_mask(
            state["rc_t"], state["ri_t"], state["rr_t"],
            state["rs_t"], state["rg_t"]
        )
        
        state["ldcompute"] = ldmicro
        
        # =============================
        # 3. Sedimentation (if before)
        # =============================
        diagnostics = {}
        
        if not lsedim_after:
            (rcs_new, rrs_new, ris_new, rss_new, rgs_new,
             fpr_c, fpr_r, fpr_i, fpr_s, fpr_g,
             inprc, inprr, inpri, inprs, inprg) = sedimentation_stat(
                state["rhodref"], state["dzz"], state["pres"],
                state["th_t"], state["rcs"], state["rrs"],
                state["ris"], state["rss"], state["rgs"],
                state.get("sea", jnp.zeros(2)),
                state.get("town", jnp.zeros(2)),
                self.constants
            )
            
            state["rcs"] = rcs_new
            state["rrs"] = rrs_new
            state["ris"] = ris_new
            state["rss"] = rss_new
            state["rgs"] = rgs_new
            
            diagnostics.update({
                "inprc": inprc,
                "inprr": inprr,
                "inpri": inpri,
                "inprs": inprs,
                "inprg": inprg,
            })
        
        # =============================
        # 4. Save Initial Values
        # =============================
        initial_state = {
            key: value.copy() for key, value in state.items()
            if key in ["th_t", "rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t"]
        }
        
        # =============================
        # 5. Time Stepping Loop
        # =============================
        t_elapsed = jnp.zeros_like(state["th_t"])
        iteration = 0
        
        # Simplified time stepping (no actual loop in JAX)
        # In practice, use jax.lax.while_loop for dynamic iteration
        
        while iteration < max_iterations:
            # Compute whether to continue
            ldcompute = t_elapsed < dt
            
            if not ldcompute.any():
                break
            
            # Compute time step
            delta_t, _ = self.time_step_limiter(
                state, {}, dt, t_elapsed
            )
            
            # Compute tendencies
            tendencies, micro_diag = self.ice4_tendencies(
                state,
                ldsoft=ldsoft,
                lfeedbackt=lfeedbackt,
            )
            
            # Update state
            state = self.update_state(state, tendencies, delta_t)
            
            # Update elapsed time
            t_elapsed = t_elapsed + delta_t
            
            iteration += 1
            ldsoft = True  # Enable soft mode after first iteration
        
        # =============================
        # 6. Compute Total Tendencies
        # =============================
        for key in ["th_t", "rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t"]:
            tnd_key = key.replace("_t", "s")  # e.g., th_t -> ths
            if tnd_key in state:
                state[tnd_key] = (state[key] - initial_state[key]) / dt
        
        # =============================
        # 7. Negativity Correction
        # =============================
        (th_t_corr, rv_t_corr, rc_t_corr, rr_t_corr,
         ri_t_corr, rs_t_corr, rg_t_corr) = ice4_correct_negativities(
            state["th_t"], state["rv_t"], state["rc_t"],
            state["rr_t"], state["ri_t"], state["rs_t"],
            state["rg_t"], lv_fact, ls_fact, self.constants
        )
        
        state.update({
            "th_t": th_t_corr,
            "rv_t": rv_t_corr,
            "rc_t": rc_t_corr,
            "rr_t": rr_t_corr,
            "ri_t": ri_t_corr,
            "rs_t": rs_t_corr,
            "rg_t": rg_t_corr,
        })
        
        # =============================
        # 8. Sedimentation (if after)
        # =============================
        if lsedim_after:
            (rcs_new, rrs_new, ris_new, rss_new, rgs_new,
             fpr_c, fpr_r, fpr_i, fpr_s, fpr_g,
             inprc, inprr, inpri, inprs, inprg) = sedimentation_stat(
                state["rhodref"], state["dzz"], state["pres"],
                state["th_t"], state["rcs"], state["rrs"],
                state["ris"], state["rss"], state["rgs"],
                state.get("sea", jnp.zeros(2)),
                state.get("town", jnp.zeros(2)),
                self.constants
            )
            
            state["rcs"] = rcs_new
            state["rrs"] = rrs_new
            state["ris"] = ris_new
            state["rss"] = rss_new
            state["rgs"] = rgs_new
            
            diagnostics.update({
                "inprc": inprc,
                "inprr": inprr,
                "inpri": inpri,
                "inprs": inprs,
                "inprg": inprg,
            })
        
        # =============================
        # 9. Fog Deposition (optional)
        # =============================
        if ldeposc:
            # Simplified fog deposition
            # Would need actual implementation
            pass
        
        return state, diagnostics
