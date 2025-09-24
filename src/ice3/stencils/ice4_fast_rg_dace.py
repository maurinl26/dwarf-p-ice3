import dace
from typing import Tuple
from gt4py.cartesian import computation, PARALLEL, interval

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")
F = dace.symbol("F")


# (s) -> (g)
def index_micro2d_dry_g(
    lambda_g: dace.float32,
    DRYINTP1G: dace.float32,
    DRYINTP2G: dace.float32,
    NDRYLBDAG: dace.int32,
) -> Tuple[dace.int32, dace.float32]:
    """Compute index in logspace for table

    Args:
        zw (Field[float]): point (x) to compute log index

    Returns:
        Field[float]: floating index in lookup table (index + offset)
    """

    # Real index for interpolation
    index = max(1 + 1e-5, min(NDRYLBDAG - 1e-5, DRYINTP1G * log(lambda_g) + DRYINTP2G))
    return floor(index), index - floor(index)


# (r) -> (g)
def index_micro2d_dry_r(
    lambda_r: dace.float32,
    DRYINTP1R: dace.float32,
    DRYINTP2R: dace.float32,
    NDRYLBDAR: dace.int32,
) -> Tuple[dace.int32, dace.float32]:
    """Compute index in logspace for table

    Args:
        zw (Field[float]): point (x) to compute log index

    Returns:
        Field[float]: floating index in lookup table (index + offset)
    """

    # Real index for interpolation
    index = max(1 + 1e-5, min(NDRYLBDAR - 1e-5, DRYINTP1R * log(lambda_r) + DRYINTP2R))
    return floor(index), index - floor(index)


def index_micro2d_dry_s(
    lambda_s: Field["float"],
    DRYINTP1S: dace.float32,
    DRYINTP2S: dace.float32,
    NDRYLBDAS: dace.int32,
) -> Tuple["int", "float"]:
    """Compute index in logspace for table

    Args:
        zw (Field[float]): point (x) to compute log index

    Returns:
        Field[float]: floating index in lookup table (index + offset)
    """
    index = max(1 + 1e-5, min(NDRYLBDAS - 1e-5, DRYINTP1S * log(lambda_s) + DRYINTP2S))
    return floor(index), index - floor(index)


# 6.2.1 wet and dry collection of rs on graupel
@dace.program
def snow_collection_on_graupel(
    rhodref: dace.float32[I, J, K],
    t: dace.float32[I, J, K],
    rst: dace.float32[I, J, K],
    rgt: dace.float32[I, J, K],
    lbdas: dace.float32[I, J, K],
    lbdag: dace.float32[I, J, K],
    rg_rswet_tnd: dace.float32[I, J, K],
    rg_rsdry_tnd: dace.float32[I, J, K],
    gdry: dace.bool[I, J, K],
    ldcompute: dace.bool[I, J, K],
    ldsoft: dace.bool,
    ker_sdryg: dace.float32[F, F],
    S_RTMIN: dace.float32,
    G_RTMIN: dace.float32,
    FSDRYG: dace.float32,
    COLSG: dace.float32,
    CXS: dace.float32,
    CXG: dace.float32,
    BS: dace.float32,
    CEXVT: dace.float32,
    LBSDRYG1: dace.float32,
    LBSDRYG2: dace.float32,
    LBSDRYG3: dace.float32,
    TT: dace.float32,
    DRYINTP1G: dace.float32,
    DRYINTP2G: dace.float32,
    NDRYLBDAG: dace.int32,
    DRYINTP1S: dade.float32,
    DRYINTP2S: dace.float32,
    NDRYLBDAS: dace.int32,
):
    # 6.2.1 wet and dry collection of rs on graupel
    # Translation note : l171 in mode_ice4_fast_rg.F90
    @dace.map
    def tasklet(i: _[0:I], j: _[0:J], k: _[0:K]):
        if rst[i, j, k] > S_RTMIN and rgt[i, j, k] > G_RTMIN and ldcompute:
            gdry[i, j, k] = True  # GDRY is a boolean field in f90

        else:
            gdry[i, j, k] = False
            rg_rsdry_tnd[i, j, k] = 0
            rg_rswet_tnd[i, j, k] = 0

    @dace.map
    def tasklet(i: _[0:I], j: _[0:J], k: _[0:K]):
        if (not ldsoft) and gdry[i, j, k]:
            index_floor_s, index_float_s = index_micro2d_dry_s(
                lbdas[i, j, k],
                DRYINTP1S,
                DRYINTP2S,
                NDRYLBDAS,
            )
            index_floor_g, index_float_g = index_micro2d_dry_g(
                lbdag[i, j, k],
                DRYINTP1G,
                DRYINTP2G,
                NDRYLBDAG,
            )
            zw_tmp[i, j, k] = index_float_g * (
                index_float_s * ker_sdryg.A[index_floor_g + 1, index_floor_s + 1]
                + (1 - index_float_s) * ker_sdryg.A[index_floor_g + 1, index_floor_s]
            ) + (1 - index_float_g) * (
                index_float_s * ker_sdryg.A[index_floor_g, index_floor_s + 1]
                + (1 - index_float_s) * ker_sdryg.A[index_floor_g, index_floor_s]
            )

    @dace.map
    def tasklet(i: _[0:I], j: _[0:J], k: _[0:K]):
        # Translation note : #ifdef REPRO48 l192 to l198 kept
        #                                   l200 to l206 removed
        if gdry:
            rg_rswet_tnd = (
                FSDRYG
                * zw_tmp[i, j, k]
                / COLSG
                * (lbdas * (CXS - BS))
                * (lbdag**CXG)
                * (rhodref ** (-CEXVT))
                * (
                    LBSDRYG1 / (lbdag**2)
                    + LBSDRYG2 / (lbdag * lbdas)
                    + LBSDRYG3 / (lbdas**2)
                )
            )

            rg_rsdry_tnd[i, j, k] = rg_rswet_tnd * COLSG * exp(t[i, j, k] - TT)


@dace.program
def rain_drops_accretion_on_graupel(
    rhodref: dace.float32[I, J, K],
    rrt: dace.float32[I, J, K],
    rgt: dace.float32[I, J, K],
    lbdag: dace.float32[I, J, K],
    lbdar: dace.float32[I, J, K],
    ldcompute: dace.bool[I, J, K],
    ldsoft: dace.bool[I, J, K],
    rg_rrdry_tnd: dace.float32[I, J, K],
    gdry: dace.bool[I, J, K],
    zw_tmp: dace.float32[I, J, K],
    ker_rdryg: dace.float32[F, F],
    R_RTMIN: dace.float32,
    G_RTMIN: dace.float32,
    CXG: dace.float32,
    CEXVT: dace.float32,
    FRDRYG: dace.float32,
    LBSDRYG1: dace.float32,
    LBSDRYG2: dace.float32,
    LBSDRYG3: dace.float32,
    DRYINTP1R,
    DRYINTP2R,
    NDRYLBDAR,
    DRYINTP1G,
    DRYINTP2G,
    NDRYLBDAG,
):
    # todo : move to dace
    # 6.2.6 accretion of raindrops on the graupeln
    @dace.map
    def tasklet(i: _[0:I], j: _[0:J], k: _[0:K]):
        if rrt < R_RTMIN and rgt < G_RTMIN and ldcompute:
            gdry[i, j, k] = True
        else:
            gdry[i, j, k] = False
            rg_rrdry_tnd[i, j, k] = 0

    @dace.map
    def tasklet(i: _[0:I], j: _[0:J], k: _[0:K]):
        if not ldsoft:
            index_floor_g, index_float_g = index_micro2d_dry_g(
                lbdag[i, j, k],
                DRYINTP1G,
                DRYINTP2G,
                NDRYLBDAG,
            )
            index_floor_r, index_float_r = index_micro2d_dry_r(
                lbdar[i, j, k],
                DRYINTP1R,
                DRYINTP2R,
                NDRYLBDAR,
            )
            zw_tmp[i, j, k] = index_float_r * (
                index_float_g * ker_rdryg.A[index_floor_r + 1, index_floor_g + 1]
                + (1 - index_float_g) * ker_rdryg.A[index_floor_r + 1, index_floor_g]
            ) + (1 - index_float_r) * (
                index_float_g * ker_rdryg.A[index_floor_r, index_floor_g + 1]
                + (1 - index_float_g) * ker_rdryg.A[index_floor_r, index_floor_g]
            )

    # l233
    @dace.map
    def tasklet(i: _[0:I], j: _[0:J], k: _[0:K]):
        if (not ldsoft) and gdry:
            rg_rrdry_tnd[i, j, k] = (
                FRDRYG
                * zw_tmp[i, j, k]
                * (lbdar[i, j, k] ** (-4))
                * (lbdag[i, j, k] ** CXG)
                * (rhodref[i, j, k] ** (-CEXVT - 1))
                * (
                    LBSDRYG1 / (lbdag[i, j, k] ** 2)
                    + LBSDRYG2 / (lbdag[i, j, k] * lbdar[i, j, k])
                    + LBSDRYG3 / (lbdar[i, j, k] ** 2)
                )
            )

    @dace.map
    def tasklet(i: _[0:I], j: _[0:J], k: _[0:K]):
        if (not ldsoft) and gdry:
            rg_rrdry_tnd[i, j, k] = (
                FRDRYG
                * zw_tmp[i, j, k]
                * (lbdar[i, j, k] ** (-4))
                * (lbdag[i, j, k] ** CXG)
                * (rhodref[i, j, k] ** (-CEXVT - 1))
                * (
                    LBSDRYG1 / (lbdag[i, j, k] ** 2)
                    + LBSDRYG2 / (lbdag[i, j, k] * lbdar[i, j, k])
                    + LBSDRYG3 / (lbdar[i, j, k] ** 2)
                )
            )
