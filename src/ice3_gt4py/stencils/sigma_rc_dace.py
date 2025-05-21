from math import floor

import dace
from ifs_physics_common.framework.grid import I, J, K

I = dace.symbol(I.name)
J = dace.symbol(J.name)
K = dace.symbol(K.name)

# for the global table
F = dace.symbol("F")

@dace.program
def sigrc_computation(
        q1: dace.float32[I, J, K],
        inq1: dace.int32[I, J, K],
        src_1d: dace.int32[F],
        sigrc: dace.float32[I, J, K],
        LAMBDA3: dace.int32
):

    @dace.map
    def tasklet(i: _[0:I], j: _[0:J], k: _[0:K]):

            inq1 = floor(
            min(10, max(-22, min(-100, 2 * floor(q1[i, j, k]))))
        )  # inner min/max prevents sigfpe when 2*zq1 does not fit dtype_into an "int"
            inc = 2 * q1[i, j, k] - inq1[i, j, k]
            sigrc = min(1, (1 - inc) * src_1d.A[inq1] + inc * src_1d.A[inq1 + 1])

            # Transaltion notes : 566 -> 578 HLAMBDA3 = CB
            if LAMBDA3 == 0:
                sigrc[i, j, k] *= min(3, max(1, 1 - q1[i, j, k]))

