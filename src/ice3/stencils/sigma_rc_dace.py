from math import floor

import dace
from dace.dtypes import ScheduleType, StorageType

from ice3_gt4py.utils.dims import I, J, K
from ice3_gt4py.utils.typingx import dtype_float, dtype_int


def sigrc_computation(
        q1: dace.float64[I, J, K] @ StorageType.GPU_Global,
        inq1: dace.int64[I, J, K] @ StorageType.GPU_Global,
        src_1d: dace.float64[34] @ StorageType.GPU_Global,
        sigrc: dace.float64[I, J, K] @ StorageType.GPU_Global,
        ext: dace.compiletime
):

    for i, j, k in dace.map[0:I, 0:J, 0:K] @ ScheduleType.GPU_Device:

            inq1 = floor(
            min(10, max(-22, min(-100, 2 * floor(q1))))
        )  # inner min/max prevents sigfpe when 2*zq1 does not fit dtype_into an "int"
            inc = 2 * q1[i, j, k] - inq1[i, j, k]
            sigrc[i, j, k] = min(1, (1 - inc) * src_1d[inq1] + inc * src_1d[inq1 + 1])

            if ext.LAMBDA3 == 0:
                sigrc[i, j, k] *= min(3, max(1, 1 - q1[i, j, k]))

