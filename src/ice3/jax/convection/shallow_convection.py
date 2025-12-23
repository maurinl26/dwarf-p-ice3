# -*- coding: utf-8 -*-
"""
JAX implementation of the complete shallow convection scheme.

This module provides the main entry point for shallow convection calculations,
automatically handling the workflow and selecting the optimal computation method.

Translated from Fortran implementation in
PHYEX-IAL_CY50T1/conv/shallow_convection.F90

The routine:
1. Calls shallow_convection_part1 to prepare and trigger
2. Counts triggered columns
3. Automatically selects the best computation method:
   - If no columns triggered: returns zeros
   - If <90% triggered: uses part2_select (packed computation)
   - If >=90% triggered: uses part2 (full computation)
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
from shallow_convection_part1 import shallow_convection_part1
from shallow_convection_part2 import shallow_convection_part2
from shallow_convection_part2_select import shallow_convection_part2_select


class ShallowConvectionOutputs(NamedTuple):
    """Complete output from shallow convection scheme.

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
        Updraft mass flux (kg/s m²), shape (nit, nkt)
    pch1ten : Array
        Chemical species convective tendency (1/s), shape (nit, nkt, kch1)
    """
    ptten: Array
    prvten: Array
    prcten: Array
    priten: Array
    kcltop: Array
    kclbas: Array
    pumf: Array
    pch1ten: Array


def shallow_convection(
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
    kice: int = 1,
    kbdia: int = 1,
    ktdia: int = 1,
    jcvexb: Optional[int] = None,
    jcvext: Optional[int] = None,
    convection_params: Optional[ConvectionParameters] = None,
    osettadj: bool = False,
    ptadjs: float = 10800.0,
    och1conv: bool = False,
    use_select_threshold: float = 0.9,
) -> ShallowConvectionOutputs:
    """
    Complete shallow convection scheme with automatic optimization.

    This is the main entry point for shallow convection calculations. It:
    1. Prepares thermodynamic variables and triggers convection (part1)
    2. Counts the number of triggered columns
    3. Automatically selects the best computation method:
       - No triggered columns: returns zeros immediately
       - <90% triggered (default): uses optimized select version
       - >=90% triggered: uses full computation

    Parameters
    ----------
    ppabst : Array
        Grid scale pressure (Pa), shape (nit, nkt)
    pzz : Array
        Height of model layers (m), shape (nit, nkt)
    ptkecls : Array
        Turbulent kinetic energy in surface layer (m²/s²), shape (nit,)
    ptt : Array
        Grid scale temperature (K), shape (nit, nkt)
    prvt : Array
        Grid scale water vapor mixing ratio (kg/kg), shape (nit, nkt)
    prct : Array
        Grid scale cloud water mixing ratio (kg/kg), shape (nit, nkt)
    prit : Array
        Grid scale ice mixing ratio (kg/kg), shape (nit, nkt)
    pwt : Array
        Grid scale vertical velocity (m/s), shape (nit, nkt)
    ptten : Array
        Convective temperature tendency (K/s), shape (nit, nkt) - will be overwritten
    prvten : Array
        Convective water vapor tendency (1/s), shape (nit, nkt) - will be overwritten
    prcten : Array
        Convective cloud water tendency (1/s), shape (nit, nkt) - will be overwritten
    priten : Array
        Convective ice tendency (1/s), shape (nit, nkt) - will be overwritten
    kcltop : Array
        Cloud top level indices, shape (nit,) - will be overwritten
    kclbas : Array
        Cloud base level indices, shape (nit,) - will be overwritten
    pumf : Array
        Updraft mass flux (kg/s m²), shape (nit, nkt) - will be overwritten
    pch1 : Array
        Grid scale chemical species, shape (nit, nkt, kch1)
    pch1ten : Array
        Species convective tendency (1/s), shape (nit, nkt, kch1) - will be overwritten
    kice : int, optional
        Ice flag (1=include ice, 0=no ice), default 1
    kbdia : int, optional
        Vertical computations start at kbdia (at least 1), default 1
    ktdia : int, optional
        Vertical computations limited to nkt+1-ktdia, default 1
    jcvexb : int, optional
        Extra vertical levels at bottom (computed from kbdia if None)
    jcvext : int, optional
        Extra vertical levels at top (computed from ktdia if None)
    convection_params : ConvectionParameters, optional
        Convection parameters (uses defaults if None)
    osettadj : bool, optional
        Whether to use user-defined adjustment time, default False
    ptadjs : float, optional
        User-defined adjustment time (s), default 10800.0
    och1conv : bool, optional
        Include chemical tracer transport, default False
    use_select_threshold : float, optional
        Fraction of triggered columns below which to use select version, default 0.9

    Returns
    -------
    ShallowConvectionOutputs
        Named tuple containing all output fields

    Notes
    -----
    **Automatic optimization**: This routine automatically chooses the best
    computation method based on the fraction of triggered columns:

    - **No convection** (0% triggered): Returns immediately with zero tendencies
    - **Sparse convection** (<90% triggered by default): Uses `part2_select` which
      packs only triggered columns for efficient computation
    - **Widespread convection** (>=90% triggered): Uses `part2` which processes
      all columns without packing overhead

    The threshold can be adjusted via `use_select_threshold`. Set to 1.0 to always
    use select version, or 0.0 to always use regular version.

    **Typical workflow**:
    ```python
    # Simple usage - automatic optimization
    outputs = shallow_convection(
        ppabst=pressure,
        pzz=height,
        ptt=temperature,
        # ... other fields ...
    )

    # Access tendencies
    temp_tendency = outputs.ptten
    vapor_tendency = outputs.prvten
    ```

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

    # Compute vertical bounds from kbdia/ktdia if not provided
    if jcvexb is None:
        jcvexb = max(0, kbdia - 1)
    if jcvext is None:
        jcvext = max(0, ktdia - 1)

    # Compute Rd/Cp
    cst = PHYS_CONSTANTS
    prdocp = cst.rd / cst.cpd

    # ===== PART 1: Trigger and prepare =====
    part1_outputs = shallow_convection_part1(
        ppabst=ppabst,
        pzz=pzz,
        ptkecls=ptkecls,
        ptt=ptt,
        prvt=prvt,
        prct=prct,
        prit=prit,
        pwt=pwt,
        ptten=ptten,
        prvten=prvten,
        prcten=prcten,
        priten=priten,
        kcltop=kcltop,
        kclbas=kclbas,
        pumf=pumf,
        pch1=pch1,
        pch1ten=pch1ten,
        jcvexb=jcvexb,
        jcvext=jcvext,
        convection_params=convection_params,
        och1conv=och1conv,
    )

    # Count triggered columns
    n_triggered = part1_outputs.otrig1.sum()
    fraction_triggered = n_triggered / nit

    # ===== PART 2: Updraft, closure, and tendencies =====
    # Decide which version to use based on fraction of triggered columns
    if n_triggered == 0:
        # No convection triggered - return zeros from part1
        return ShallowConvectionOutputs(
            ptten=part1_outputs.ptten,
            prvten=part1_outputs.prvten,
            prcten=part1_outputs.prcten,
            priten=part1_outputs.priten,
            kcltop=part1_outputs.kcltop,
            kclbas=part1_outputs.kclbas,
            pumf=part1_outputs.pumf,
            pch1ten=part1_outputs.pch1ten,
        )

    elif fraction_triggered < use_select_threshold:
        # Use optimized select version for sparse convection
        part2_outputs = shallow_convection_part2_select(
            ppabst=ppabst,
            pzz=pzz,
            ptt=ptt,
            prvt=prvt,
            prct=prct,
            prit=prit,
            pch1=pch1,
            prdocp=prdocp,
            ptht=part1_outputs.ptht,
            psthv=part1_outputs.psthv,
            psthes=part1_outputs.psthes,
            isdpl=part1_outputs.ksdpl,
            ispbl=part1_outputs.kspbl,
            islcl=part1_outputs.kslcl,
            psthlcl=part1_outputs.psthlcl,
            pstlcl=part1_outputs.pstlcl,
            psrvlcl=part1_outputs.psrvlcl,
            pswlcl=part1_outputs.pswlcl,
            pszlcl=part1_outputs.pszlcl,
            psthvelcl=part1_outputs.psthvelcl,
            gtrig1=part1_outputs.otrig1,
            kice=kice,
            jcvexb=jcvexb,
            jcvext=jcvext,
            convection_params=convection_params,
            osettadj=osettadj,
            ptadjs=ptadjs,
            och1conv=och1conv,
        )
    else:
        # Use regular version for widespread convection
        part2_outputs = shallow_convection_part2(
            ppabst=ppabst,
            pzz=pzz,
            ptt=ptt,
            prvt=prvt,
            prct=prct,
            prit=prit,
            pch1=pch1,
            prdocp=prdocp,
            ptht=part1_outputs.ptht,
            psthv=part1_outputs.psthv,
            psthes=part1_outputs.psthes,
            isdpl=part1_outputs.ksdpl,
            ispbl=part1_outputs.kspbl,
            islcl=part1_outputs.kslcl,
            psthlcl=part1_outputs.psthlcl,
            pstlcl=part1_outputs.pstlcl,
            psrvlcl=part1_outputs.psrvlcl,
            pswlcl=part1_outputs.pswlcl,
            pszlcl=part1_outputs.pszlcl,
            psthvelcl=part1_outputs.psthvelcl,
            gtrig1=part1_outputs.otrig1,
            kice=kice,
            jcvexb=jcvexb,
            jcvext=jcvext,
            convection_params=convection_params,
            osettadj=osettadj,
            ptadjs=ptadjs,
            och1conv=och1conv,
        )

    # Return combined outputs
    return ShallowConvectionOutputs(
        ptten=part2_outputs.pthc,
        prvten=part2_outputs.prvc,
        prcten=part2_outputs.prcc,
        priten=part2_outputs.pric,
        kcltop=part2_outputs.ictl,
        kclbas=part2_outputs.iminctl,  # Use iminctl for kclbas
        pumf=part2_outputs.pumf,
        pch1ten=part2_outputs.ppch1ten,
    )
