# -*- coding: utf-8 -*-
"""
JAX implementation of shallow convection part 1.

This module provides a JAX-compatible implementation of the first part of the
shallow convection scheme, which prepares grid-scale variables and triggers
convection.

Translated from Fortran implementation in
PHYEX-IAL_CY50T1/conv/shallow_convection_part1.F90

The routine determines convective tendencies by:
1. Setting up grid scale theta, theta_v, theta_es
2. Testing for convective columns and determining properties at the LCL
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from typing import Dict, Any, Tuple, NamedTuple, Optional

# Import the existing trigger function from stencils
import sys
from pathlib import Path

# Add parent directory to path to import from stencils
stencils_path = Path(__file__).parent.parent / "stencils"
sys.path.insert(0, str(stencils_path))

from convect_trigger_shal import convect_trigger_shal, ConvectionParameters
from constants import PHYS_CONSTANTS


class ShallowConvectionOutputs(NamedTuple):
    """Output from shallow convection part 1.

    Attributes
    ----------
    ptten : Array
        Convective temperature tendency (K/s), shape (nit, nkt)
    prvten : Array
        Convective water vapor tendency (1/s), shape (nit, nkt)
    prcten : Array
        Convective cloud water tendency (1/s), shape (nit, nkt)
    priten : Array
        Convective ice tendency (1/s), shape (nit, nkt)
    kcltop : Array
        Cloud top level indices, shape (nit,)
    kclbas : Array
        Cloud base level indices, shape (nit,)
    pumf : Array
        Updraft mass flux (kg/s m2), shape (nit, nkt)
    pch1ten : Array
        Chemical species convective tendency (1/s), shape (nit, nkt, kch1)
    ptht : Array
        Grid scale potential temperature, shape (nit, nkt)
    psthv : Array
        Grid scale virtual potential temperature, shape (nit, nkt)
    psthes : Array
        Grid scale equivalent potential temperature, shape (nit, nkt)
    ksdpl : Array
        Index for parcel departure level, shape (nit,)
    kspbl : Array
        Index for source layer top, shape (nit,)
    kslcl : Array
        Index for lifting condensation level, shape (nit,)
    psthlcl : Array
        Updraft theta at LCL, shape (nit,)
    pstlcl : Array
        Updraft temperature at LCL, shape (nit,)
    psrvlcl : Array
        Updraft water vapor at LCL, shape (nit,)
    pswlcl : Array
        Updraft vertical velocity at LCL, shape (nit,)
    pszlcl : Array
        LCL height, shape (nit,)
    psthvelcl : Array
        Environmental virtual potential temperature at LCL, shape (nit,)
    otrig1 : Array
        Logical mask for convection, shape (nit,)
    """
    ptten: Array
    prvten: Array
    prcten: Array
    priten: Array
    kcltop: Array
    kclbas: Array
    pumf: Array
    pch1ten: Array
    ptht: Array
    psthv: Array
    psthes: Array
    ksdpl: Array
    kspbl: Array
    kslcl: Array
    psthlcl: Array
    pstlcl: Array
    psrvlcl: Array
    pswlcl: Array
    pszlcl: Array
    psthvelcl: Array
    otrig1: Array


def shallow_convection_part1(
    ppabst: Array,
    pzz: Array,
    ptkecls: Array,
    ptt: Array,
    prvt: Array,
    prct: Array,
    prit: Array,
    pwt: Array,
    ptten: Array,
    prvten: Array,
    prcten: Array,
    priten: Array,
    kcltop: Array,
    kclbas: Array,
    pumf: Array,
    pch1: Array,
    pch1ten: Array,
    jcvexb: int = 0,
    jcvext: int = 0,
    convection_params: Optional[ConvectionParameters] = None,
    och1conv: bool = False,
) -> ShallowConvectionOutputs:
    """
    Monitor routine to compute shallow convective tendencies.

    This function prepares grid-scale variables and triggers shallow convection.
    It computes potential temperature, virtual potential temperature, and
    equivalent potential temperature, then tests for convective columns.

    Parameters
    ----------
    ppabst : Array
        Grid scale pressure at time t (Pa), shape (nit, nkt)
    pzz : Array
        Height of model layer (m), shape (nit, nkt)
    ptkecls : Array
        Turbulent kinetic energy in the surface layer (m2/s2), shape (nit,)
    ptt : Array
        Grid scale temperature at time t (K), shape (nit, nkt)
    prvt : Array
        Grid scale water vapor mixing ratio (kg/kg), shape (nit, nkt)
    prct : Array
        Grid scale cloud water mixing ratio (kg/kg), shape (nit, nkt)
    prit : Array
        Grid scale ice mixing ratio (kg/kg), shape (nit, nkt)
    pwt : Array
        Grid scale vertical velocity (m/s), shape (nit, nkt)
    ptten : Array
        Convective temperature tendency (K/s), shape (nit, nkt) - will be reset
    prvten : Array
        Convective water vapor tendency (1/s), shape (nit, nkt) - will be reset
    prcten : Array
        Convective cloud water tendency (1/s), shape (nit, nkt) - will be reset
    priten : Array
        Convective ice tendency (1/s), shape (nit, nkt) - will be reset
    kcltop : Array
        Cloud top level indices, shape (nit,) - will be reset
    kclbas : Array
        Cloud base level indices, shape (nit,) - will be reset
    pumf : Array
        Updraft mass flux (kg/s m2), shape (nit, nkt) - will be reset
    pch1 : Array
        Grid scale chemical species concentrations, shape (nit, nkt, kch1)
    pch1ten : Array
        Species convective tendency (1/s), shape (nit, nkt, kch1) - will be reset
    jcvexb : int, optional
        Extra levels on bottom vertical boundary (default: 0)
    jcvext : int, optional
        Extra levels on top vertical boundary (default: 0)
    convection_params : ConvectionParameters, optional
        Convection parameters instance (uses defaults if None)
    och1conv : bool, optional
        Flag to include tracer transport (default: False)

    Returns
    -------
    ShallowConvectionOutputs
        Named tuple containing all output fields

    Notes
    -----
    This is part 1 of the shallow convection scheme. The main steps are:

    1. Initialize local variables and constants
    2. Compute grid-scale thermodynamic variables:
       - Potential temperature (theta)
       - Virtual potential temperature (theta_v)
       - Equivalent potential temperature (theta_es)
    3. Test for convective columns using CONVECT_TRIGGER_SHAL

    The depth of model layer k is defined by DP(k) = P(k-1) - P(k).
    All computations are done on MNH thermodynamic levels.

    References
    ----------
    - Bechtold, 1997: Meso-NH scientific documentation (31 pp)
    - Fritsch and Chappell, 1980, J. Atmos. Sci., Vol. 37, 1722-1761
    - Kain and Fritsch, 1990, J. Atmos. Sci., Vol. 47, 2784-2801
    - Kain and Fritsch, 1993, Meteor. Monographs, Vol. 24, 165-170
    """
    # Use default convection parameters if not provided
    if convection_params is None:
        convection_params = ConvectionParameters()

    # Extract dimensions
    nit = ppabst.shape[0]
    nkt = ppabst.shape[1]

    # Use physical constants from PHYS_CONSTANTS
    cst = PHYS_CONSTANTS
    xrd = cst.rd
    xrv = cst.rv
    xcpd = cst.cpd
    xp00 = cst.p00
    xalpw = cst.alpw
    xbetaw = cst.betaw
    xgamw = cst.gamw

    # Compute vertical bounds
    ikb = 1 + jcvexb
    ike = nkt - jcvext

    # Reset convective tendencies to zero
    ptten = jnp.zeros_like(ptten)
    prvten = jnp.zeros_like(prvten)
    prcten = jnp.zeros_like(prcten)
    priten = jnp.zeros_like(priten)
    pumf = jnp.zeros_like(pumf)
    kcltop = jnp.zeros_like(kcltop)
    kclbas = jnp.zeros_like(kclbas)

    if och1conv:
        pch1ten = jnp.zeros_like(pch1ten)

    # Initialize local variables
    zeps = xrd / xrv  # R_d / R_v
    zepsa = xrv / xrd  # R_v / R_d
    zrdocp = xrd / xcpd  # R_d / C_p

    # Initialize thermodynamic arrays
    ptht = jnp.full((nit, nkt), 300.0)
    psthv = jnp.full((nit, nkt), 300.0)
    psthes = jnp.full((nit, nkt), 400.0)

    # Compute grid scale theta, theta_v, theta_es
    # Only for levels where pressure > 4000 Pa
    pressure_mask = ppabst > 4000.0

    # Potential temperature: theta = T * (P00 / P)^(Rd/Cp)
    ptht = jnp.where(
        pressure_mask,
        ptt * (xp00 / ppabst) ** zrdocp,
        ptht
    )

    # Virtual potential temperature
    # theta_v = theta * (1 + eps_a * rv) / (1 + rv + rc + ri)
    psthv = jnp.where(
        pressure_mask,
        ptht * (1.0 + zepsa * prvt) / (1.0 + prvt + prct + prit),
        psthv
    )

    # Equivalent potential temperature using conservative Bolton (1980) formula
    # First compute saturation mixing ratio
    zes = jnp.exp(xalpw - xbetaw / ptt - xgamw * jnp.log(ptt))
    zes = jnp.minimum(1.0, zeps * zes / (ppabst - zes))

    # theta_es = T * (theta/T)^(1 - 0.28*es) * exp((3374.6525/T - 2.5403) * es * (1 + 0.81*es))
    psthes = jnp.where(
        pressure_mask,
        ptt * (ptht / ptt) ** (1.0 - 0.28 * zes) *
        jnp.exp((3374.6525 / ptt - 2.5403) * zes * (1.0 + 0.81 * zes)),
        psthes
    )

    # Call CONVECT_TRIGGER_SHAL to test for convective columns
    # and determine properties at the LCL
    trigger_outputs = convect_trigger_shal(
        ppres=ppabst,
        pth=ptht,
        pthv=psthv,
        pthes=psthes,
        prv=prvt,
        pw=pwt,
        pz=pzz,
        ptkecls=ptkecls,
        jcvexb=jcvexb,
        jcvext=jcvext,
        params=convection_params,
    )

    # Extract outputs from trigger function
    otrig1 = trigger_outputs.otrig
    psthlcl = trigger_outputs.pthlcl
    pstlcl = trigger_outputs.ptlcl
    psrvlcl = trigger_outputs.prvlcl
    pswlcl = trigger_outputs.pwlcl
    pszlcl = trigger_outputs.pzlcl
    psthvelcl = trigger_outputs.pthvelcl
    kslcl = trigger_outputs.klcl
    ksdpl = trigger_outputs.kdpl
    kspbl = trigger_outputs.kpbl

    return ShallowConvectionOutputs(
        ptten=ptten,
        prvten=prvten,
        prcten=prcten,
        priten=priten,
        kcltop=kcltop,
        kclbas=kclbas,
        pumf=pumf,
        pch1ten=pch1ten,
        ptht=ptht,
        psthv=psthv,
        psthes=psthes,
        ksdpl=ksdpl,
        kspbl=kspbl,
        kslcl=kslcl,
        psthlcl=psthlcl,
        pstlcl=pstlcl,
        psrvlcl=psrvlcl,
        pswlcl=pswlcl,
        pszlcl=pszlcl,
        psthvelcl=psthvelcl,
        otrig1=otrig1,
    )
