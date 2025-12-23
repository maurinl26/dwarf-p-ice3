# -*- coding: utf-8 -*-
"""
JAX implementation of shallow convection part 2.

This module provides a JAX-compatible implementation of the second part of the
shallow convection scheme, which computes updraft properties, applies closure,
and determines convective tendencies.

Translated from Fortran implementation in
PHYEX-IAL_CY50T1/conv/shallow_convection_part2.F90

The routine:
1. Computes pressure differences and environmental enthalpy/total water
2. Calls updraft scheme to compute updraft properties
3. Applies closure scheme to determine adjusted values
4. Computes grid-scale convective tendencies
5. Applies conservation corrections
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from typing import NamedTuple, Optional
import sys
from pathlib import Path

# Add parent directory to path to import from stencils
stencils_path = Path(__file__).parent.parent / "stencils"
sys.path.insert(0, str(stencils_path))

from convect_trigger_shal import ConvectionParameters
from constants import PHYS_CONSTANTS
from convect_updraft_shal import convect_updraft_shal
from convect_closure_shal import convect_closure_shal


class ShallowConvectionPart2Outputs(NamedTuple):
    """Output from shallow convection part 2.

    Attributes
    ----------
    pumf : Array
        Updraft mass flux per unit area (kg/s m²), shape (nit, nkt)
    pthc : Array
        Convective temperature tendency (K/s), shape (nit, nkt)
    prvc : Array
        Convective water vapor tendency (1/s), shape (nit, nkt)
    prcc : Array
        Convective cloud water tendency (1/s), shape (nit, nkt)
    pric : Array
        Convective ice tendency (1/s), shape (nit, nkt)
    ictl : Array
        Cloud top level indices, shape (nit,)
    iminctl : Array
        Minimum of cloud top and LCL indices, shape (nit,)
    ppch1ten : Array
        Chemical species convective tendency (1/s), shape (nit, nkt, kch1)
    """
    pumf: Array
    pthc: Array
    prvc: Array
    prcc: Array
    pric: Array
    ictl: Array
    iminctl: Array
    ppch1ten: Array


def shallow_convection_part2(
    ppabst: Array,
    pzz: Array,
    ptt: Array,
    prvt: Array,
    prct: Array,
    prit: Array,
    pch1: Array,
    prdocp: float,
    ptht: Array,
    psthv: Array,
    psthes: Array,
    isdpl: Array,
    ispbl: Array,
    islcl: Array,
    psthlcl: Array,
    pstlcl: Array,
    psrvlcl: Array,
    pswlcl: Array,
    pszlcl: Array,
    psthvelcl: Array,
    gtrig1: Array,
    kice: int = 1,
    jcvexb: int = 0,
    jcvext: int = 0,
    convection_params: Optional[ConvectionParameters] = None,
    osettadj: bool = False,
    ptadjs: float = 10800.0,
    och1conv: bool = False,
) -> ShallowConvectionPart2Outputs:
    """
    Compute shallow convective updraft, closure, and tendencies.

    This function performs the main shallow convection calculations:
    1. Prepares environmental variables (enthalpy, total water)
    2. Computes updraft properties via CONVECT_UPDRAFT_SHAL
    3. Applies closure scheme via CONVECT_CLOSURE_SHAL
    4. Computes grid-scale convective tendencies
    5. Applies conservation corrections

    Parameters
    ----------
    ppabst : Array
        Grid scale pressure (Pa), shape (nit, nkt)
    pzz : Array
        Height of model layers (m), shape (nit, nkt)
    ptt : Array
        Grid scale temperature (K), shape (nit, nkt)
    prvt : Array
        Grid scale water vapor mixing ratio (kg/kg), shape (nit, nkt)
    prct : Array
        Grid scale cloud water mixing ratio (kg/kg), shape (nit, nkt)
    prit : Array
        Grid scale ice mixing ratio (kg/kg), shape (nit, nkt)
    pch1 : Array
        Grid scale chemical species, shape (nit, nkt, kch1)
    prdocp : float
        Rd/Cp ratio
    ptht : Array
        Grid scale potential temperature (K), shape (nit, nkt)
    psthv : Array
        Grid scale virtual potential temperature (K), shape (nit, nkt)
    psthes : Array
        Grid scale equivalent potential temperature (K), shape (nit, nkt)
    isdpl : Array
        Index for parcel departure level, shape (nit,)
    ispbl : Array
        Index for source layer top, shape (nit,)
    islcl : Array
        Index for lifting condensation level, shape (nit,)
    psthlcl : Array
        Updraft theta at LCL (K), shape (nit,)
    pstlcl : Array
        Updraft temperature at LCL (K), shape (nit,)
    psrvlcl : Array
        Updraft water vapor at LCL (kg/kg), shape (nit,)
    pswlcl : Array
        Updraft vertical velocity at LCL (m/s), shape (nit,)
    pszlcl : Array
        LCL height (m), shape (nit,)
    psthvelcl : Array
        Environmental theta_v at LCL (K), shape (nit,)
    gtrig1 : Array
        Trigger mask (bool), shape (nit,)
    kice : int, optional
        Ice flag (1=include ice, 0=no ice), default 1
    jcvexb : int, optional
        Extra vertical levels at bottom, default 0
    jcvext : int, optional
        Extra vertical levels at top, default 0
    convection_params : ConvectionParameters, optional
        Convection parameters (uses defaults if None)
    osettadj : bool, optional
        Whether to use user-defined adjustment time, default False
    ptadjs : float, optional
        User-defined adjustment time (s), default 10800.0
    och1conv : bool, optional
        Include chemical tracer transport, default False

    Returns
    -------
    ShallowConvectionPart2Outputs
        Named tuple containing all output fields

    Notes
    -----
    This is part 2 of the shallow convection scheme. Main steps:

    1. **Prepare environmental variables**:
       - Compute pressure differences between layers
       - Compute environmental enthalpy: h = C_p*T + (1+r_w)*g*z - L_v*r_c - L_s*r_i
       - Compute total water: r_w = r_v + r_c + r_i

    2. **Updraft calculation** (CONVECT_UPDRAFT_SHAL):
       - Compute mass flux, entrainment, detrainment
       - Compute updraft thermodynamics and condensate
       - Accumulate CAPE and determine cloud top (CTL)

    3. **Closure** (CONVECT_CLOSURE_SHAL):
       - Adjust mass flux to remove CAPE over time PTIMEC
       - Compute adjusted environmental values
       - Ensure mass conservation via subsidence

    4. **Compute tendencies**:
       - Convert adjusted values to tendencies: (adjusted - initial) / time
       - Apply smoothing at cloud top for PBL inversions
       - Apply conservation corrections to ensure zero column integrals

    References
    ----------
    - Bechtold, 1997: Meso-NH scientific documentation
    - Fritsch and Chappell, 1980, J. Atmos. Sci., Vol. 37, 1722-1761
    - Kain and Fritsch, 1990, J. Atmos. Sci., Vol. 47, 2784-2801
    """
    # Use default convection parameters if not provided
    if convection_params is None:
        convection_params = ConvectionParameters()

    # Extract dimensions
    nit = ppabst.shape[0]
    nkt = ppabst.shape[1]
    kch1 = pch1.shape[2] if pch1.ndim == 3 else 0

    # Physical constants
    cst = PHYS_CONSTANTS

    # Vertical bounds
    ikb = 1 + jcvexb
    ike = nkt - jcvext

    # Initialize outputs for levels above computation domain
    pumf = jnp.zeros((nit, nkt))
    pthc = jnp.zeros((nit, nkt))
    prvc = jnp.zeros((nit, nkt))
    prcc = jnp.zeros((nit, nkt))
    pric = jnp.zeros((nit, nkt))
    ppch1ten = jnp.zeros((nit, nkt, kch1))

    # ===== 3.2 Compute pressure difference =====
    zdpres = jnp.zeros((nit, nkt))
    zdpres = zdpres.at[:, ikb+1:ike+1].set(
        ppabst[:, ikb:ike] - ppabst[:, ikb+1:ike+1]
    )

    # ===== 3.3 Compute environmental enthalpy and total water =====
    # Total water: r_w = r_v + r_c + r_i
    zrw = jnp.maximum(0.0, prvt) + jnp.maximum(0.0, prct) + jnp.maximum(0.0, prit)

    # Specific heat (moisture dependent)
    zcph = cst.cpd + cst.cpv * zrw

    # Latent heats (temperature dependent)
    zlv = cst.xlvtt + (cst.cpv - cst.cl) * (ptt - cst.tt)  # L_v
    zls = cst.xlstt + (cst.cpv - cst.ci) * (ptt - cst.tt)  # L_s

    # Environmental enthalpy: h = C_p*T + (1+r_w)*g*z - L_v*r_c - L_s*r_i
    zthl = (zcph * ptt +
            (1.0 + zrw) * cst.g * pzz -
            zlv * jnp.maximum(0.0, prct) -
            zls * jnp.maximum(0.0, prit))

    # ===== 4. Compute updraft properties =====
    # Set mass flux at LCL (unit mass flux with w = 1 m/s, multiplied by reference area)
    pmflcl = convection_params.xa25 * 1.0e-3

    # Create trigger mask for updraft (same as gtrig1)
    gtrig2 = gtrig1.copy()

    # Call updraft scheme
    updraft_outputs = convect_updraft_shal(
        ppres=ppabst,
        pdpres=zdpres,
        pz=pzz,
        pthl=zthl,
        pthv=psthv,
        pthes=psthes,
        prw=zrw,
        pthlcl=psthlcl,
        ptlcl=pstlcl,
        prvlcl=psrvlcl,
        pwlcl=pswlcl,
        pzlcl=pszlcl,
        pthvelcl=psthvelcl,
        pmflcl=pmflcl,
        otrig=gtrig2,
        klcl=islcl,
        kdpl=isdpl,
        kpbl=ispbl,
        gtrig1=gtrig1,
        kice=kice,
        jcvexb=jcvexb,
        jcvext=jcvext,
        xentr=convection_params.xentr,
        xcrad=convection_params.xcrad,
        xcdepth=convection_params.xcdepth,
        xcdepth_d=convection_params.xcdepth_d,
        xnhgam=convection_params.xnhgam,
        xtfrz1=convection_params.xtfrz1,
        xtfrz2=convection_params.xtfrz2,
    )

    # Extract updraft outputs
    pumf_up = updraft_outputs.pumf
    zuer = updraft_outputs.puer
    zudr = updraft_outputs.pudr
    zuthl = updraft_outputs.puthl
    zuthv = updraft_outputs.puthv
    zurw = updraft_outputs.purw
    zurc = updraft_outputs.purc
    zuri = updraft_outputs.puri
    zcape = updraft_outputs.pcape
    ictl = updraft_outputs.kctl
    ietl = updraft_outputs.ketl

    # Initialize downdraft arrays (no downdraft in shallow convection)
    zdmf = jnp.zeros((nit, nkt))
    zder = jnp.zeros((nit, nkt))
    zddr = jnp.zeros((nit, nkt))
    ilfs = jnp.full(nit, ikb, dtype=jnp.int32)

    # ===== Compute layer mass =====
    zlmass = convection_params.xa25 * zdpres / cst.g
    zlmass = zlmass.at[:, ikb].set(zlmass[:, ikb+1])

    # ===== 5. Set convective adjustment time =====
    ztimec = jnp.full(nit, convection_params.xctime_shal)
    if osettadj:
        ztimec = jnp.full(nit, ptadjs)

    # ===== 7. Closure - determine adjusted environmental values =====
    closure_outputs = convect_closure_shal(
        ppres=ppabst,
        pdpres=zdpres,
        pz=pzz,
        plmass=zlmass,
        pthl=zthl,
        pth=ptht,
        prw=zrw,
        prc=prct,
        pri=prit,
        otrig1=gtrig2,
        klcl=islcl,
        kdpl=isdpl,
        kpbl=ispbl,
        kctl=ictl,
        pumf=pumf_up,
        puer=zuer,
        pudr=zudr,
        puthl=zuthl,
        purw=zurw,
        purc=zurc,
        puri=zuri,
        pcape=zcape,
        ptimec=ztimec,
        jcvexb=jcvexb,
        jcvext=jcvext,
        xstabt=convection_params.xstabt,
        xstabc=convection_params.xstabc,
        xa25=convection_params.xa25,
    )

    # Extract closure outputs
    pthc_adj = closure_outputs.pthc
    prwc_adj = closure_outputs.prwc
    prcc_adj = closure_outputs.prcc
    pric_adj = closure_outputs.pric
    zwsub = closure_outputs.pwsub
    pumf_adj = closure_outputs.pumf
    zuer_adj = closure_outputs.puer
    zudr_adj = closure_outputs.pudr
    iftsteps = closure_outputs.kftsteps

    # ===== 8.1 Compute grid scale tendencies =====
    # Convert adjusted values to tendencies: (adjusted - initial) / time
    # Temperature tendency (convert theta to T)
    pthc_tend = jnp.zeros((nit, nkt))
    for jk in range(ikb, ike+1):
        pthc_tend = pthc_tend.at[:, jk].set(
            (pthc_adj[:, jk] - ptht[:, jk]) / ztimec *
            (ppabst[:, jk] / cst.p00) ** prdocp
        )

    # Water vapor tendency (total water - condensate)
    prvc_tend = jnp.zeros((nit, nkt))
    for jk in range(ikb, ike+1):
        prvc_tend = prvc_tend.at[:, jk].set(
            (prwc_adj[:, jk] - zrw[:, jk] +
             jnp.maximum(0.0, prct[:, jk]) + jnp.maximum(0.0, prit[:, jk])) / ztimec
        )

    # Cloud water tendency
    prcc_tend = jnp.zeros((nit, nkt))
    for jk in range(ikb, ike+1):
        prcc_tend = prcc_tend.at[:, jk].set(
            (prcc_adj[:, jk] - jnp.maximum(0.0, prct[:, jk])) / ztimec
        )

    # Ice tendency
    pric_tend = jnp.zeros((nit, nkt))
    for jk in range(ikb, ike+1):
        pric_tend = pric_tend.at[:, jk].set(
            (pric_adj[:, jk] - jnp.maximum(0.0, prit[:, jk])) / ztimec
        )

    # ===== 8.2 Apply smoothing at cloud top =====
    if convection_params.llsmooth:
        for ji in range(nit):
            jk = ictl[ji]
            jkm = jnp.maximum(2, ictl[ji] - 1)
            jkp = jnp.maximum(2, ictl[ji] - 2)

            # Redistribute cloud top tendencies to lower levels
            prvc_tend = prvc_tend.at[ji, jkm].add(0.5 * prvc_tend[ji, jk])
            prcc_tend = prcc_tend.at[ji, jkm].add(0.5 * prcc_tend[ji, jk])
            pric_tend = pric_tend.at[ji, jkm].add(0.5 * pric_tend[ji, jk])
            pthc_tend = pthc_tend.at[ji, jkm].add(0.5 * pthc_tend[ji, jk])

            prvc_tend = prvc_tend.at[ji, jkp].add(0.3 * prvc_tend[ji, jk])
            prcc_tend = prcc_tend.at[ji, jkp].add(0.3 * prcc_tend[ji, jk])
            pric_tend = pric_tend.at[ji, jkp].add(0.3 * pric_tend[ji, jk])
            pthc_tend = pthc_tend.at[ji, jkp].add(0.3 * pthc_tend[ji, jk])

            prvc_tend = prvc_tend.at[ji, jk].multiply(0.2)
            prcc_tend = prcc_tend.at[ji, jk].multiply(0.2)
            pric_tend = pric_tend.at[ji, jk].multiply(0.2)
            pthc_tend = pthc_tend.at[ji, jk].multiply(0.2)

    # ===== 8.3 Apply conservation correction =====
    # Compute vertical integrals (must be zero for conservation)
    jkm = ike
    zwork2 = jnp.zeros(nit)   # Moisture integral
    zwork2b = jnp.zeros(nit)  # Energy integral

    for jk in range(ikb+1, jkm+1):
        jkp = jk + 1
        mask = jk <= ictl

        # Moisture flux
        zw1_moist = prvc_tend[:, jk] + prcc_tend[:, jk] + pric_tend[:, jk]
        zwork2 += jnp.where(
            mask,
            zw1_moist * 0.5 * (ppabst[:, jk-1] - ppabst[:, jkp]) / cst.g,
            0.0
        )

        # Energy flux
        zlv_jk = cst.xlvtt + (cst.cpv - cst.cl) * (ptt[:, jk] - cst.tt)
        zls_jk = cst.xlstt + (cst.cpv - cst.ci) * (ptt[:, jk] - cst.tt)
        zw1_energy = ((cst.cpd + cst.cpv * zrw[:, jk]) * pthc_tend[:, jk] -
                      zlv_jk * prcc_tend[:, jk] - zls_jk * pric_tend[:, jk])
        zwork2b += jnp.where(
            mask,
            zw1_energy * 0.5 * (ppabst[:, jk-1] - ppabst[:, jkp]) / cst.g,
            0.0
        )

    # Compute correction factors
    mask_corr = ictl > ikb + 1
    jkp = ictl
    zw1_denom = cst.g / (ppabst[:, ikb] - ppabst[:, jkp] -
                         0.5 * (zdpres[:, ikb+1] - zdpres[:, jkp+1]))
    zwork2 = jnp.where(mask_corr, zwork2 * zw1_denom, 0.0)
    zwork2b = jnp.where(mask_corr, zwork2b * zw1_denom, 0.0)

    # Apply uniform correction
    for jk in range(ikb+1, jkm+1):
        mask_apply = (ictl > ikb + 1) & (jk <= ictl)
        prvc_tend = prvc_tend.at[:, jk].add(jnp.where(mask_apply, -zwork2, 0.0))
        pthc_tend = pthc_tend.at[:, jk].add(jnp.where(mask_apply, -zwork2b / cst.cpd, 0.0))

    # ===== 8.7 Chemical tracer transport (not implemented yet) =====
    # TODO: Implement CONVECT_CHEM_TRANSPORT when needed

    # ===== Normalize mass flux and finalize outputs =====
    # Convert mass flux from total to per unit area
    pumf_final = jnp.where(
        (ictl[:, None] > ikb + 1) & (jnp.arange(nkt)[None, :] >= ikb) & (jnp.arange(nkt)[None, :] <= ike),
        pumf_adj / convection_params.xa25,
        0.0
    )

    # Compute IMINCTL (minimum of LCL and CTL)
    iminctl = jnp.minimum(islcl, ictl)

    # Zero out tendencies where convection is not triggered
    pthc_final = jnp.where(gtrig1[:, None], pthc_tend, 0.0)
    prvc_final = jnp.where(gtrig1[:, None], prvc_tend, 0.0)
    prcc_final = jnp.where(gtrig1[:, None], prcc_tend, 0.0)
    pric_final = jnp.where(gtrig1[:, None], pric_tend, 0.0)
    ictl_final = jnp.where(gtrig1, ictl, 0)
    iminctl_final = jnp.where(gtrig1, iminctl, 0)

    # Zero out top level
    pumf_final = pumf_final.at[:, nkt-1].set(0.0)
    pthc_final = pthc_final.at[:, nkt-1].set(0.0)
    prvc_final = prvc_final.at[:, nkt-1].set(0.0)
    prcc_final = prcc_final.at[:, nkt-1].set(0.0)
    pric_final = pric_final.at[:, nkt-1].set(0.0)
    ppch1ten = ppch1ten.at[:, nkt-1, :].set(0.0)

    return ShallowConvectionPart2Outputs(
        pumf=pumf_final,
        pthc=pthc_final,
        prvc=prvc_final,
        prcc=prcc_final,
        pric=pric_final,
        ictl=ictl_final,
        iminctl=iminctl_final,
        ppch1ten=ppch1ten,
    )
