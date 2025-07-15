# -*- coding: utf-8 -*-
from __future__ import annotations

import dace
import numpy as np
from dace.dtypes import StorageType, ScheduleType

from ice3_gt4py.utils.typingx import dtype_int, dtype_float
from ice3_gt4py.utils.dims import I, J, K
from ice3_gt4py.functions.tiwmx import e_sat_i, e_sat_w

def condensation(
    sigqsat: dtype_float[I, J, K] @ StorageType.GPU_Global,
    pabs: dtype_float[I, J, K] @ StorageType.GPU_Global,
    sigs: dtype_float[I, J, K] @ StorageType.GPU_Global,
    t: dtype_float[I, J, K] @ StorageType.GPU_Global,
    rv: dtype_float[I, J, K] @ StorageType.GPU_Global,
    ri: dtype_float[I, J, K] @ StorageType.GPU_Global,
    rc: dtype_float[I, J, K] @ StorageType.GPU_Global,
    rv_out: dtype_float[I, J, K] @ StorageType.GPU_Global,
    rc_out: dtype_float[I, J, K] @ StorageType.GPU_Global,
    ri_out: dtype_float[I, J, K] @ StorageType.GPU_Global,
    cldfr: dtype_float[I, J, K] @ StorageType.GPU_Global,
    cph: dtype_float[I, J, K] @ StorageType.GPU_Global,
    lv: dtype_float[I, J, K] @ StorageType.GPU_Global,
    ls: dtype_float[I, J, K] @ StorageType.GPU_Global,
    q1: dtype_float[I, J, K] @ StorageType.GPU_Global,
    ext: dace.compiletime
):
    """Microphysical adjustments for specific contents due to condensation."""

    rt = np.ndarray([I, J, K], dtype=dtype_float)
    pv = np.ndarray([I, J, K], dtype=dtype_float)
    piv = np.ndarray([I, J, K], dtype=dtype_float)

    # initialize values
    for i, j, k in dace.map[0:I, 0:J, 0:K] @ ScheduleType.GPU_Device:
        cldfr[i, j, k] = 0.0
        rv_out[i, j, k] = 0.0
        rc_out[i, j, k] = 0.0
        ri_out[i, j, k] = 0.0

    # 3. subgrid condensation scheme
    for i, j, k in dace.map[0:I, 0:J, 0:K] @ ScheduleType.GPU_Device:
        prifact = 1
        frac_tmp = 0

        # store total water mixing ratio (244 -> 248)
        rt[i, j, k] = rv[i, j, k] + rc[i, j, k] + ri[i, j, k] * prifact

        # l334 to l337
        if not ext.OCND2:
            pv[i, j, k] = min(
            e_sat_w(t[i, j, k]),
            0.99 * pabs[i, j, k],
            )
            piv[i, j, k] = min(
            e_sat_i(t[i, j, k]),
            0.99 * pabs[i, j, k],
            )

        if not ext.OCND2:
            frac_tmp =(
                rc[i, j, k] / (rc[i, j, k] + ri[i, j, k])
                if rc[i, j, k] + ri[i, j, k] > 1e-20 else 0
            )

            # Compute frac ice inlined
            # Default Mode (S)
            if ext.FRAC_ICE_ADJUST == 3:
                frac_tmp = max(0, min(1, frac_tmp[i, j, k]))

            # AROME mode
            if ext.FRAC_ICE_ADJUST == 0:
                frac_tmp = max(0,
                               min(1,
                                   ((ext.TMAXMIX - t[i, j, k]) / (ext.TMAXMIX - ext.TMINMIX))
                                   ))

        
        # Supersaturation coefficients
        qsl = ext.RD / ext.RV * pv[i, j, k] / (pabs - pv[i, j, k])
        qsi = ext.RD / ext.RV * piv[i, j, k] / (pabs - piv[i, j, k])

        # interpolate between liquid and solid as a function of temperature
        qsl = (1 - frac_tmp) * qsl + frac_tmp * qsi
        lvs = (1 - frac_tmp) * lv[i, j, k] + frac_tmp * ls[i, j, k]

        # coefficients a et b
        ah = lvs * qsl / (ext.RV * t[i, j, k]**2) * (1 + ext.RV * qsl / ext.RD)
        a = 1 / (1 + lvs / cph[i, j, k] * ah)
        b = ah * a
        sbar = a * (rt[i, j, k] - qsl + ah * lvs * (rc[i, j, k] + ri[i, j, k] * prifact) / cph[i, j, k])

        if ext.LSIGMAS and not ext.LSTATNW:
            sigma = (
                sqrt((2 * sigs[i, j, k]) ** 2 + (sigqsat[i, j, k] * qsl * a) ** 2)
                if sigqsat[i, j, k] != 0
                else 2 * sigs[i, j, k]
            )

        # Translation note : l407 - l411
        sigma = max(1e-10, sigma)
        q1[i, j, k] = sbar / sigma

        # 9.2.3 Fractional cloudiness and cloud condensate
        # HCONDENS = 0 is CB02 option
        if ext.CONDENS == 0:
        # Translation note : l470 to l479
            if q1[i, j, k] > 0.0:
                cond_tmp = (
                min(exp(-1.0) + 0.66 * q1[i, j, k] + 0.086 * q1[i, j, k]**2, 2.0) if q1[i, j, k] <= 2.0 else q1[i, j, k]
            )  # we use the MIN function for continuity
            else:
                cond_tmp = exp(1.2 * q1[i, j, k] - 1.0)
            cond_tmp *= sigma

            # cloud fraction
            cldfr[i, j, k] = (
                max(0.0, min(1.0, 0.5 + 0.36 * atan(1.55 * q1[i, j, k]))) if cond_tmp >= 1e-12 else 0
            )

            # Translation note : l487 to l489
            cond_tmp = 0 if cldfr[i, j, k] == 0 else cond_tmp

            if not ext.OCND2:
                rc_out[i, j, k] = (1 - frac_tmp) * cond_tmp  # liquid condensate
                ri_out[i, j, k] = frac_tmp * cond_tmp  # solid condensate
                t[i, j, k] += ((rc_out[i, j, k] - rc[i, j, k]) * lv[i, j, k] + (ri_out[i, j, k] - ri[i, j, k]) * ls[i, j, k]) / cph[i, j, k]
                rv_out[i, j, k] = rt[i, j, k] - rc_out[i, j, k] - ri_out[i, j, k] * prifact

        # Translation note : end jiter

