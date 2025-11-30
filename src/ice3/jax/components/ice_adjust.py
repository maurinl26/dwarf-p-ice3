# -*- coding: utf-8 -*-
"""
JAX implementation of Ice Adjustment component.

This module provides a high-level component for ice adjustment calculations,
wrapping the JAX stencil implementation with a convenient interface.

Translated from GT4Py implementation in src/ice3/components/ice_adjust.py
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp
from jax import Array

from ...phyex_common.phyex import Phyex
from ..stencils.ice_adjust import ice_adjust as ice_adjust_stencil

log = logging.getLogger(__name__)


class IceAdjustJAX:
    """
    JAX implementation of ice adjustment component.
    
    This class provides saturation adjustment of temperature and mixing ratios
    for mixed-phase clouds using JAX for JIT compilation and automatic differentiation.
    
    Parameters
    ----------
    phyex : Phyex, optional
        Physics configuration object. Default: Phyex("AROME")
    jit : bool, optional
        Whether to JIT-compile the stencil. Default: True
    
    Attributes
    ----------
    constants : Dict[str, Any]
        Physical constants and parameters extracted from phyex
    ice_adjust_fn : callable
        The ice_adjust function (JIT-compiled if jit=True)
    
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from ice3.jax.components.ice_adjust import IceAdjustJAX
    >>> from ice3.phyex_common.phyex import Phyex
    >>> 
    >>> # Initialize component
    >>> phyex = Phyex(program="AROME", TSTEP=60.0)
    >>> ice_adjust = IceAdjustJAX(phyex=phyex, jit=True)
    >>> 
    >>> # Prepare input fields (example with random data)
    >>> shape = (10, 10, 20)  # (x, y, z)
    >>> sigqsat = jnp.ones(shape) * 0.01
    >>> pabs = jnp.ones(shape) * 85000.0  # Pa
    >>> sigs = jnp.ones(shape) * 0.1
    >>> th = jnp.ones(shape) * 285.0  # K
    >>> exn = jnp.ones(shape) * 0.95
    >>> # ... (prepare other fields)
    >>> 
    >>> # Run ice adjustment
    >>> results = ice_adjust(
    ...     sigqsat=sigqsat, pabs=pabs, sigs=sigs, th=th,
    ...     exn=exn, exn_ref=exn, rho_dry_ref=rho_dry_ref,
    ...     rv=rv, rc=rc, ri=ri, rr=rr, rs=rs, rg=rg,
    ...     cf_mf=cf_mf, rc_mf=rc_mf, ri_mf=ri_mf,
    ...     rvs=rvs, rcs=rcs, ris=ris, ths=ths,
    ...     timestep=60.0
    ... )
    >>> 
    >>> # Access results
    >>> t, rv_out, rc_out, ri_out, cldfr = results[:5]
    """
    
    def __init__(
        self,
        phyex: Phyex = None,
        jit: bool = True,
    ) -> None:
        """
        Initialize the IceAdjustJAX component.
        
        Parameters
        ----------
        phyex : Phyex, optional
            Physics configuration. If None, uses Phyex("AROME")
        jit : bool, optional
            Whether to JIT-compile the stencil. Default: True
        """
        if phyex is None:
            phyex = Phyex("AROME")
        
        self.phyex = phyex
        self.constants = phyex.to_externals()
        
        # Add OCND2 flag (AROME default)
        self.constants["OCND2"] = False
        
        # Store the stencil function
        if jit:
            self.ice_adjust_fn = jax.jit(ice_adjust_stencil)
            log.info("IceAdjustJAX initialized with JIT compilation enabled")
        else:
            self.ice_adjust_fn = ice_adjust_stencil
            log.info("IceAdjustJAX initialized without JIT compilation")
        
        log.info(
            f"Phyex Configuration: "
            f"SUBG_COND={phyex.nebn.LSUBG_COND}, "
            f"SUBG_MF_PDF={phyex.param_icen.SUBG_MF_PDF}, "
            f"SIGMAS={phyex.nebn.LSIGMAS}, "
            f"LMFCONV={phyex.LMFCONV}"
        )
    
    def __call__(
        self,
        sigqsat: Array,
        pabs: Array,
        sigs: Array,
        th: Array,
        exn: Array,
        exn_ref: Array,
        rho_dry_ref: Array,
        rv: Array,
        rc: Array,
        ri: Array,
        rr: Array,
        rs: Array,
        rg: Array,
        cf_mf: Array,
        rc_mf: Array,
        ri_mf: Array,
        rvs: Array,
        rcs: Array,
        ris: Array,
        ths: Array,
        timestep: float,
    ) -> Tuple[Array, ...]:
        """
        Execute ice adjustment calculation.
        
        Parameters
        ----------
        sigqsat : Array
            Standard deviation of saturation mixing ratio.
        pabs : Array
            Absolute pressure (Pa).
        sigs : Array
            Sigma_s for subgrid-scale turbulent mixing.
        th : Array
            Potential temperature (K).
        exn : Array
            Exner function (dimensionless).
        exn_ref : Array
            Reference Exner function.
        rho_dry_ref : Array
            Reference dry air density (kg/m³).
        rv, rc, ri, rr, rs, rg : Array
            Mixing ratios for vapor, cloud, ice, rain, snow, graupel (kg/kg).
        cf_mf : Array
            Cloud fraction from mass flux scheme.
        rc_mf, ri_mf : Array
            Liquid/ice mixing ratios from mass flux (kg/kg).
        rvs, rcs, ris, ths : Array
            Tendency fields (per timestep).
        timestep : float
            Time step (s).
            
        Returns
        -------
        Tuple of Arrays containing:
            - t: Updated temperature (K)
            - rv_out: Adjusted vapor mixing ratio (kg/kg)
            - rc_out: Adjusted cloud liquid mixing ratio (kg/kg)
            - ri_out: Adjusted cloud ice mixing ratio (kg/kg)
            - cldfr: Cloud fraction (0-1)
            - hlc_hrc, hlc_hcf: Liquid autoconversion diagnostics
            - hli_hri, hli_hcf: Ice autoconversion diagnostics
            - cph: Specific heat capacity (J/(kg·K))
            - lv, ls: Latent heats (J/kg)
            - rvs, rcs, ris, ths: Updated tendencies
        """
        return self.ice_adjust_fn(
            sigqsat=sigqsat,
            pabs=pabs,
            sigs=sigs,
            th=th,
            exn=exn,
            exn_ref=exn_ref,
            rho_dry_ref=rho_dry_ref,
            rv=rv,
            ri=ri,
            rc=rc,
            rr=rr,
            rs=rs,
            rg=rg,
            cf_mf=cf_mf,
            rc_mf=rc_mf,
            ri_mf=ri_mf,
            rvs=rvs,
            rcs=rcs,
            ris=ris,
            ths=ths,
            dt=timestep,
            constants=self.constants,
        )
