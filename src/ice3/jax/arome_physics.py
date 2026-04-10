"""
AROME Physical Parameterizations JAX Orchestrator.

This module unifies the four physics modules:
1. Cloud Adjustment (Ice Adjust)
2. Shallow Convection
3. Turbulence
4. Microphysics (Rain Ice)

The orchestration mimics the sequence in APL_AROME.
"""

import functools
from typing import Dict, Tuple, NamedTuple, Optional
import jax
import jax.numpy as jnp
from jax import Array

# Apple Silicon / Metal note: Metal does not support float64 or int64.
# All scan carries and index arrays must stay int32 / float32.
# Nvidia CUDA: x64 is off by default; jax.config.update("jax_enable_x64", True)
# would re-enable it, but should be avoided for cross-platform compatibility.

# Import the submodules
from ice3.jax.ice_adjust import IceAdjustJAX
from ice3.jax.convection.shallow_convection import shallow_convection, ShallowConvectionOutputs
from ice3.jax.turbulence.turb import turb_scheme
from ice3.jax.rain_ice import RainIceJAX
from ice3.jax.ecrad_jax import EcRadJAX, EcRadState
from ice3.jax.surfex_jax import SurfexJAX, SurfexState

# Define a DataClass/NamedTuple representing the prognostic state
class AromeState(NamedTuple):
    # Dimensions: (nit, nkt) arrays unless noted
    pabst: Array       # Absolute pressure (Pa)
    pzz: Array         # Height of layers (m)
    dzz: Array         # Vertical layer thickness (m)
    pt: Array          # Temperature (K)
    pth: Array         # Potential temperature (K)
    pthl: Array       # Liquid potential temperature (K)
    prv: Array         # Water vapor (kg/kg)
    prc: Array         # Cloud water (kg/kg)
    pri: Array         # Cloud ice (kg/kg)
    prr: Array         # Rain water (kg/kg)
    prs: Array         # Snow (kg/kg)
    prg: Array         # Graupel (kg/kg)
    prt: Array         # Total water (kg/kg)
    pu: Array          # U wind (m/s)
    pv: Array          # V wind (m/s)
    pw: Array          # W (vertical) wind (m/s)
    ptke: Array        # Turbulent kinetic energy (m2/s2)
    # Surface variables (nit,)
    ptkecls: Array     # TKE at surface (m2/s2)
    psurf_flux_u: Array
    psurf_flux_v: Array
    psurf_flux_th: Array
    psurf_flux_rv: Array
    # Reference vars
    pthvref: Array
    pexn: Array
    pexn_ref: Array
    prho_dry_ref: Array

class AromePhysicsOrchestrator:
    def __init__(self, constants: Dict, phyex=None):
        self.constants = constants
        
        # Instantiate class-based modules
        self.ice_adjust = IceAdjustJAX(phyex=phyex, jit=True)
        self.rain_ice = RainIceJAX(constants=constants)
        self.ecrad = EcRadJAX(use_jit=True)
        self.surfex = SurfexJAX()
        
        # Vmapped turbulence module
        # turb_scheme takes 1D arrays (nz,) for fields
        # We vmap over the batch dimension (axis 0)
        # in_axes must match the number of args passed at the call site (22):
        #   zz, dzz, theta, thl, rt, rv, rc, ri  → 8  (batched, axis 0)
        #   u, v, w, tke                          → 4  (batched, axis 0)
        #   thvref, pabst, exn                    → 3  (batched, axis 0)
        #   surf_flux_u/v/th/rv                   → 4  (batched, axis 0: shape (nit,)→scalar)
        #   dt                                    → 1  (not batched: scalar)
        #   turb_constants, phys_constants        → 2  (not batched: None)
        # ximpl, xlini, tke_min use their defaults and are NOT passed → no in_axes entries.
        self.vmap_turb_scheme = jax.jit(
            jax.vmap(
                turb_scheme,
                in_axes=(
                    0, 0, 0, 0, 0, 0, 0, 0,  # zz, dzz, theta, thl, rt, rv, rc, ri
                    0, 0, 0, 0,              # u, v, w, tke
                    0, 0, 0,                 # thvref, pabst, exn
                    0, 0, 0, 0,              # surf_flux_u, surf_flux_v, surf_flux_th, surf_flux_rv
                    None,                    # dt (scalar, broadcast to all columns)
                    None, None,              # turb_constants, phys_constants (None → defaults)
                ),
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0, 2))
    def step(self, state: AromeState, dt: float) -> Tuple[AromeState, Dict]:
        """
        Takes a time step for the AROME physical parameterizations.
        Shapes are (nit, nkt) for vertical fields, and (nit,) for surface fields.

        JIT notes
        ---------
        * ``self`` is static (argnum 0): uses Python id-based hash so each
          orchestrator instance gets its own compiled cache entry.
        * ``dt`` is static (argnum 2): recompile only when the time step
          length changes (rare).
        * ``state`` (argnum 1) is a NamedTuple — JAX handles it as a pytree.
        * shallow_convection uses ``jax.lax.switch`` so no Python conditionals
          on traced values remain inside this function.

        PMAP usage
        ----------
        Wrap with ``jax.pmap`` after construction.  The leading axis of every
        field in ``state`` must equal the number of local devices.
        """
        nit, nkt = state.pabst.shape
        diagnostics = {}
        
        # Create zero tendencies and dummy arrays for required missing args
        sigqsat = jnp.zeros_like(state.pabst)
        sigs = jnp.zeros_like(state.pabst)
        ths = jnp.zeros_like(state.pabst)
        rvs = jnp.zeros_like(state.pabst)
        rcs = jnp.zeros_like(state.pabst)
        ris = jnp.zeros_like(state.pabst)
        
        cf_mf = jnp.zeros_like(state.pabst)
        rc_mf = jnp.zeros_like(state.pabst)
        ri_mf = jnp.zeros_like(state.pabst)

        # -------------------------------------------------------------
        # 0. Radiation and Surface (ecRad & SURFEX)
        # -------------------------------------------------------------
        # Create half-level pressure approximation for Radiation
        pres_hl = jnp.pad(state.pabst, ((0,0), (1,0)), mode='edge')
        ecrad_state = EcRadState(
            pres=state.pabst,
            pres_hl=pres_hl,
            temp=state.pt,
            q=state.prv,
            q_liquid=state.prc,
            q_ice=state.pri,
            cloud_frac=jnp.zeros_like(state.pabst),
            albedo_sw=jnp.full((nit,), 0.2),
            emissivity_lw=jnp.full((nit,), 0.98),
            cos_zenith=jnp.ones((nit,))
        )
        ecrad_fluxes, ecrad_diag = self.ecrad(ecrad_state, dt)
        
        # Execute the tight pure_callback to the CPU SURFEX driver
        surfex_state = SurfexState(
            t_a=state.pt[:, -1],           # Lowest model layer
            q_a=state.prv[:, -1],
            u_a=state.pu[:, -1],
            v_a=state.pv[:, -1],
            p_a=state.pabst[:, -1],
            rhodref=state.prho_dry_ref[:, -1],
            sw_down=ecrad_fluxes.sw_dn[:, -1],
            lw_down=ecrad_fluxes.lw_dn[:, -1],
            rain_rate=jnp.zeros((nit,)),    # Will be available recursively next step
            snow_rate=jnp.zeros((nit,))
        )
        surf_fluxes = self.surfex(surfex_state, dt)
        
        # Override the input surface fluxes for turbulence later on
        state = state._replace(
            psurf_flux_th=surf_fluxes.surf_flux_th,
            psurf_flux_rv=surf_fluxes.surf_flux_rv,
            psurf_flux_u=surf_fluxes.surf_flux_u,
            psurf_flux_v=surf_fluxes.surf_flux_v,
        )
        diagnostics['ecrad'] = ecrad_diag

        # -------------------------------------------------------------
        # 1. Cloud Adjustment (Ice Adjust)
        # -------------------------------------------------------------
        t_out, rv_out, rc_out, ri_out, cldfr, hlc_hrc, hlc_hcf, \
        hli_hri, hli_hcf, cph, lv, ls, \
        rvs_out, rcs_out, ris_out, ths_out = self.ice_adjust(
            sigqsat=sigqsat, pabs=state.pabst, sigs=sigs, th=state.pth,
            exn=state.pexn, exn_ref=state.pexn_ref, rho_dry_ref=state.prho_dry_ref,
            rv=state.prv, rc=state.prc, ri=state.pri, rr=state.prr, rs=state.prs, rg=state.prg,
            cf_mf=cf_mf, rc_mf=rc_mf, ri_mf=ri_mf,
            rvs=rvs, rcs=rcs, ris=ris, ths=ths,
            timestep=dt
        )
        # Update state after adjustment
        # Convert output temperature T to potential temperature TH = T / exn
        pth_new = t_out / state.pexn
        state = state._replace(pt=t_out, pth=pth_new, prv=rv_out, prc=rc_out, pri=ri_out)
        diagnostics['cldfr'] = cldfr

        # -------------------------------------------------------------
        # 2. Shallow Convection
        # -------------------------------------------------------------
        ptten = jnp.zeros_like(state.pt)
        prvten = jnp.zeros_like(state.prv)
        prcten = jnp.zeros_like(state.prc)
        priten = jnp.zeros_like(state.pri)
        kcltop = jnp.zeros((nit,), dtype=jnp.int32)
        kclbas = jnp.zeros((nit,), dtype=jnp.int32)
        pumf = jnp.zeros_like(state.pt)
        pch1 = jnp.zeros((nit, nkt, 1))
        pch1ten = jnp.zeros((nit, nkt, 1))

        shallow_out: ShallowConvectionOutputs = shallow_convection(
            ppabst=state.pabst,
            pzz=state.pzz,
            ptkecls=state.ptkecls,
            ptt=state.pt,
            prvt=state.prv,
            prct=state.prc,
            prit=state.pri,
            pwt=state.pw,
            ptten=ptten,
            prvten=prvten,
            prcten=prcten,
            priten=priten,
            kcltop=kcltop,
            kclbas=kclbas,
            pumf=pumf,
            pch1=pch1,
            pch1ten=pch1ten,
            kice=1
        )
        # Apply explicit tendencies for dt
        # TH changes based on T changes: delta_PT = ptten * dt
        # Note: shallow_convection_part1/2 uses temperature (ptt), so ptten is temperature tendency
        pt_new = state.pt + shallow_out.ptten * dt
        pth_new = pt_new / state.pexn
        
        state = state._replace(
            pt=pt_new,
            pth=pth_new,
            prv=state.prv + shallow_out.prvten * dt,
            prc=state.prc + shallow_out.prcten * dt,
            pri=state.pri + shallow_out.priten * dt
        )
        diagnostics['pumf'] = shallow_out.pumf

        # -------------------------------------------------------------
        # 3. Turbulence
        # -------------------------------------------------------------
        du_dt, dv_dt, dthl_dt, drt_dt, dtke_dt, turb_diag = self.vmap_turb_scheme(
            state.pzz, state.dzz, state.pth, state.pthl, state.prt,
            state.prv, state.prc, state.pri,
            state.pu, state.pv, state.pw, state.ptke,
            state.pthvref, state.pabst, state.pexn,
            state.psurf_flux_u, state.psurf_flux_v, state.psurf_flux_th, state.psurf_flux_rv,
            dt, None, None
        )
        # Apply implicit tendencies
        state = state._replace(
            pu=state.pu + du_dt * dt,
            pv=state.pv + dv_dt * dt,
            pthl=state.pthl + dthl_dt * dt,
            prt=state.prt + drt_dt * dt,
            ptke=state.ptke + dtke_dt * dt
        )
        diagnostics['turb'] = turb_diag

        # -------------------------------------------------------------
        # 4. Microphysics (Rain Ice)
        # -------------------------------------------------------------
        # RainIce expects a state dictionary
        rain_ice_state = {
            "th_t": state.pth,
            "rv_t": state.prv,
            "rc_t": state.prc,
            "rr_t": state.prr,
            "ri_t": state.pri,
            "rs_t": state.prs,
            "rg_t": state.prg,
            "ci_t": cldfr,
            "exn": state.pexn,
            "rhodref": state.prho_dry_ref,
            "dzz": state.dzz,
            "pres": state.pabst,
            "rcs": rcs_out,
            "rrs": jnp.zeros_like(state.pabst),
            "ris": ris_out,
            "rss": jnp.zeros_like(state.pabst),
            "rgs": jnp.zeros_like(state.pabst),
        }
        rain_ice_out, rain_ice_diag = self.rain_ice(state=rain_ice_state, dt=dt)
        
        state = state._replace(
            pth=rain_ice_out["th_t"],
            pt=rain_ice_out["th_t"] * state.pexn,
            prv=rain_ice_out["rv_t"],
            prc=rain_ice_out["rc_t"],
            prr=rain_ice_out["rr_t"],
            pri=rain_ice_out["ri_t"],
            prs=rain_ice_out["rs_t"],
            prg=rain_ice_out["rg_t"]
        )
        diagnostics['rain_ice'] = rain_ice_diag

        return state, diagnostics
