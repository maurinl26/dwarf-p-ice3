# -*- coding: utf-8 -*-
"""
JAX implementation of shallow convection part 2 select.

This module provides an optimized version of shallow_convection_part2 that only
processes columns where convection has been triggered, improving computational
efficiency.

Translated from Fortran implementation in
PHYEX-IAL_CY50T1/conv/shallow_convection_part2_select.F90

The routine:
1. Identifies columns where GTRIG1 is True
2. Packs/gathers these columns into a compact array
3. Calls shallow_convection_part2 on the packed arrays
4. Unpacks/scatters the results back to the original grid
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from typing import Optional
import sys
from pathlib import Path

# Import from stencils package using absolute imports
from ice3.jax.stencils.convect_trigger_shal import ConvectionParameters
from .shallow_convection_part2 import (
    shallow_convection_part2,
    ShallowConvectionPart2Outputs,
)


def shallow_convection_part2_select(
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
    Optimized version of shallow_convection_part2 for triggered columns only.

    This function improves computational efficiency by only processing horizontal
    columns where convection has been triggered (gtrig1 = True). It:
    1. Identifies triggered columns
    2. Packs input arrays to include only triggered columns
    3. Calls shallow_convection_part2 on the packed arrays
    4. Unpacks results back to the full grid

    This can provide significant speedup when only a subset of columns have
    active convection.

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
        Named tuple containing all output fields on the full grid

    Notes
    -----
    **Performance optimization**:
    - When many columns have gtrig1=False, this routine significantly reduces
      computation by only processing triggered columns
    - The overhead of packing/unpacking is small compared to the savings
    - In JAX, this can be JIT-compiled for optimal performance

    **Algorithm**:
    1. Find indices where gtrig1 is True: isort = where(gtrig1)
    2. Pack all input arrays: x_packed = x[isort, ...]
    3. Call shallow_convection_part2 on packed arrays
    4. Unpack results: result[isort, ...] = result_packed

    This is particularly useful in large-scale models where shallow convection
    is only active in certain regions (e.g., over warm oceans, during daytime).
    """
    # Extract dimensions
    nit = ppabst.shape[0]
    nkt = ppabst.shape[1]
    kch1 = pch1.shape[2] if pch1.ndim == 3 else 0

    # Find triggered columns
    isort = jnp.where(gtrig1)[0]
    kit = isort.shape[0]

    # If no columns are triggered, return zeros
    if kit == 0:
        return ShallowConvectionPart2Outputs(
            pumf=jnp.zeros((nit, nkt)),
            pthc=jnp.zeros((nit, nkt)),
            prvc=jnp.zeros((nit, nkt)),
            prcc=jnp.zeros((nit, nkt)),
            pric=jnp.zeros((nit, nkt)),
            ictl=jnp.zeros(nit, dtype=jnp.int32),
            iminctl=jnp.zeros(nit, dtype=jnp.int32),
            ppch1ten=jnp.zeros((nit, nkt, kch1)),
        )

    # Pack input arrays (gather triggered columns)
    ppabst_packed = ppabst[isort, :]
    pzz_packed = pzz[isort, :]
    ptt_packed = ptt[isort, :]
    prvt_packed = prvt[isort, :]
    prct_packed = prct[isort, :]
    prit_packed = prit[isort, :]
    pch1_packed = pch1[isort, :, :] if kch1 > 0 else jnp.zeros((kit, nkt, 1))
    ptht_packed = ptht[isort, :]
    psthv_packed = psthv[isort, :]
    psthes_packed = psthes[isort, :]
    isdpl_packed = isdpl[isort]
    ispbl_packed = ispbl[isort]
    islcl_packed = islcl[isort]
    psthlcl_packed = psthlcl[isort]
    pstlcl_packed = pstlcl[isort]
    psrvlcl_packed = psrvlcl[isort]
    pswlcl_packed = pswlcl[isort]
    pszlcl_packed = pszlcl[isort]
    psthvelcl_packed = psthvelcl[isort]
    gtrig1_packed = gtrig1[isort]

    # Call shallow_convection_part2 on packed arrays
    packed_outputs = shallow_convection_part2(
        ppabst=ppabst_packed,
        pzz=pzz_packed,
        ptt=ptt_packed,
        prvt=prvt_packed,
        prct=prct_packed,
        prit=prit_packed,
        pch1=pch1_packed,
        prdocp=prdocp,
        ptht=ptht_packed,
        psthv=psthv_packed,
        psthes=psthes_packed,
        isdpl=isdpl_packed,
        ispbl=ispbl_packed,
        islcl=islcl_packed,
        psthlcl=psthlcl_packed,
        pstlcl=pstlcl_packed,
        psrvlcl=psrvlcl_packed,
        pswlcl=pswlcl_packed,
        pszlcl=pszlcl_packed,
        psthvelcl=psthvelcl_packed,
        gtrig1=gtrig1_packed,
        kice=kice,
        jcvexb=jcvexb,
        jcvext=jcvext,
        convection_params=convection_params,
        osettadj=osettadj,
        ptadjs=ptadjs,
        och1conv=och1conv,
    )

    # Unpack results (scatter back to full grid)
    # Initialize full arrays with zeros
    pumf_full = jnp.zeros((nit, nkt))
    pthc_full = jnp.zeros((nit, nkt))
    prvc_full = jnp.zeros((nit, nkt))
    prcc_full = jnp.zeros((nit, nkt))
    pric_full = jnp.zeros((nit, nkt))
    ictl_full = jnp.zeros(nit, dtype=jnp.int32)
    iminctl_full = jnp.zeros(nit, dtype=jnp.int32)
    ppch1ten_full = jnp.zeros((nit, nkt, kch1))

    # Scatter packed results to triggered columns
    pumf_full = pumf_full.at[isort, :].set(packed_outputs.pumf)
    pthc_full = pthc_full.at[isort, :].set(packed_outputs.pthc)
    prvc_full = prvc_full.at[isort, :].set(packed_outputs.prvc)
    prcc_full = prcc_full.at[isort, :].set(packed_outputs.prcc)
    pric_full = pric_full.at[isort, :].set(packed_outputs.pric)
    ictl_full = ictl_full.at[isort].set(packed_outputs.ictl)
    iminctl_full = iminctl_full.at[isort].set(packed_outputs.iminctl)
    if kch1 > 0:
        ppch1ten_full = ppch1ten_full.at[isort, :, :].set(packed_outputs.ppch1ten)

    return ShallowConvectionPart2Outputs(
        pumf=pumf_full,
        pthc=pthc_full,
        prvc=prvc_full,
        prcc=prcc_full,
        pric=pric_full,
        ictl=ictl_full,
        iminctl=iminctl_full,
        ppch1ten=ppch1ten_full,
    )
