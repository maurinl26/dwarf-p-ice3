# -*- coding: utf-8 -*-
from __future__ import annotations

import dace
from dace.dtypes import ScheduleType, StorageType
import numpy as np
from ice3_gt4py.utils.typingx import dtype_float, dtype_int
from ice3_gt4py.utils.dims import I, J, K
from ice3_gt4py.functions.ice_adjust import (
    constant_pressure_heat_capacity,
    sublimation_latent_heat,
    vaporisation_latent_heat,
)


def aro_filter(
    exnref: dace.float64[I, J, K] @ StorageType.GPU_Global,
    cph: dace.float64[I, J, K] @ StorageType.GPU_Global,
    tht: dace.float64[I, J, K] @ StorageType.GPU_Global,
    ths: dace.float64[I, J, K] @ StorageType.GPU_Global,
    rcs: dace.float64[I, J, K] @ StorageType.GPU_Global,
    rrs: dace.float64[I, J, K] @ StorageType.GPU_Global,
    ris: dace.float64[I, J, K] @ StorageType.GPU_Global,
    rvs: dace.float64[I, J, K] @ StorageType.GPU_Global,
    rgs: dace.float64[I, J, K] @ StorageType.GPU_Global,
    rss: dace.float64[I, J, K] @ StorageType.GPU_Global,
    dt: dtype_float,
):
    """Negativity filter for sources

    Args:
        exnref (Field[float]): reference exner pressure
        tht (Field[float]): potential temperature at time t
        ths (Field[float]): potential temperature source
        rcs (Field[float]): cloud droplets source
        rrs (Field[float]): rain source
        ris (Field[float]): ice source
        rvs (Field[float]): water vapour source
        rgs (Field[float]): graupel source
        rss (Field[float]): snow source
        dt (float): time step un seconds
    """

    t = np.ndarray([I, J, K], dtype=dtype_float)
    ls = np.ndarray([I, J, K], dtype=dtype_float)
    lv = np.ndarray([I, J, K], dtype=dtype_float)
    cor = np.ndarray([I, J, K], dtype=dtype_float)

    # 3.1. Remove negative values
    for i, j, k in dace.map[0:I, 0:J, 0:K] @ ScheduleType.GPU_Device:
        rrs[i,j,k]  = max(0, rrs[i,j,k])
        rss[i,j,k]  = max(0, rss[i,j,k])
        rgs[i,j,k]  = max(0, rgs[i,j,k])

    # 3.2. Adjustment for solid and liquid cloud
    for i, j, k in dace.map[0:I, 0:J, 0:K] @ ScheduleType.GPU_Device:
        t[i,j,k] = tht[i,j,k]  * exnref[i,j,k]
        ls[i,j,k] = sublimation_latent_heat(t[i,j,k])
        lv[i,j,k] = vaporisation_latent_heat(t[i,j,k])
        cph[i,j,k] = constant_pressure_heat_capacity(rvs[i,j,k], rcs[i,j,k], ris[i,j,k], rrs[i,j,k], rss[i,j,k], rgs[i,j,k])

    for i, j, k in dace.map[0:I, 0:J, 0:K] @ ScheduleType.GPU_Device:
        if ris[i,j,k] > 0:
            rvs[i,j,k]  = rvs[i,j,k]  + ris[i,j,k]
            ths[i,j,k]  = (
                ths[i,j,k]
                - ris[i,j,k]  * ls[i,j,k]  / cph[i,j,k]  / exnref[i,j,k]
            )
            ris[i,j,k]  = 0

    for i, j, k in dace.map[0:I, 0:J, 0:K] @ ScheduleType.GPU_Device:
        if rcs[i,j,k]  < 0:
            rvs[i,j,k]  = rvs[i,j,k]  + rcs[i,j,k]
            ths[i,j,k]  = (
                ths[i,j,k]
                - rcs[i,j,k]  * lv[i,j,k]  / cph[i,j,k]  / exnref[i,j,k]
            )
            rcs[i,j,k] = 0

    # cloud droplets
    for i, j, k in dace.map[0:I, 0:J, 0:K] @ ScheduleType.GPU_Device:
        cor[i,j,k] = (
            min(-rvs[i,j,k] , rcs[i,j,k] )
            if rvs[i,j,k]  < 0 and rcs[i,j,k]  > 0
            else 0
        )
        rvs[i,j,k]  = rvs[i,j,k]  + cor[i,j,k]
        ths[i,j,k]  = (
            ths[i,j,k]  - cor[i,j,k]  * lv[i,j,k]  / cph[i,j,k]  / exnref[i,j,k]
        )
        rcs[i,j,k]  = rcs[i,j,k]  - cor[i,j,k]

    # ice
    for i, j, k in dace.map[0:I, 0:J, 0:K] @ ScheduleType.GPU_Device:
        cor[i,j,k] = (
            min(-rvs[i,j,k] , ris[i,j,k] )
            if rvs[i,j,k]  < 0 and ris[i,j,k]  > 0
            else 0
        )
        rvs[i,j,k]  = rvs[i,j,k]  + cor[i,j,k]
        ths[i,j,k]  = (
            ths[i,j,k]  - cor[i,j,k]  * lv[i,j,k]  / cph[i,j,k]  / exnref[i,j,k]
        )
        ris[i,j,k]  = ris[i,j,k]  - cor[i,j,k]

    # 9. Transform sources to tendencies (*= 2 dt)
    for i, j, k in dace.map[0:I, 0:J, 0:K] @ ScheduleType.GPU_Device:
        rvs[i,j,k]  = rvs[i,j,k]  * 2 * dt
        rcs[i,j,k]  = rcs[i,j,k]  * 2 * dt
        rrs[i,j,k]  = rrs[i,j,k] * 2 * dt
        ris[i,j,k]  = ris[i,j,k]  * 2 * dt
        rss[i,j,k]  = rss[i,j,k]  * 2 * dt
        rgs[i,j,k]  = rgs[i,j,k]  * 2 * dt

    # (Call ice_adjust - saturation adjustment - handled by AroAdjust ImplicitTendencyComponent + ice_adjust stencil)
