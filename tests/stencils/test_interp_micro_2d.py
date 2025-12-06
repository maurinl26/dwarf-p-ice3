import numpy as np
import numpy.typing as npt
import pytest

from ice3.utils.env import sp_dtypes


def compute_accretion_interp(
    lambda_s: npt.ArrayLike,
    lambda_r: npt.ArrayLike,
    ACCINTP1S: np.float32,
    ACCINTP2S: np.float32,
    NACCLBDAS: np.int32,
    ACCINTP1R: np.float32,
    ACCINTP2R: np.float32,
    NACCLBDAR: np.int32,
    ker_raccss: npt.ArrayLike,
    ker_raccs: npt.ArrayLike,
    ker_saccrg: npt.ArrayLike,
):
    """Bilinear interpolation from ice4_fast_rs"""

    # index micro2d acc s
    index_s = np.clip(
        ACCINTP1S * np.log(lambda_s) + ACCINTP2S, NACCLBDAS - 0.00001, 1.00001
    )

    idx_s = np.floor(index_s).astype(np.int32)
    idx_s2 = idx_s + 1
    weight_s = index_s - idx_s

    # index micro2d acc r
    index_r = np.clip(
        ACCINTP1R * np.log(lambda_r) + ACCINTP2R, NACCLBDAR - 0.00001, 1.00001
    )
    idx_r = np.floor(index_r).astype(np.int32)
    idx_r2 = idx_r + 1
    weight_r = index_r - idx_r

    # Bilinear interpolation for RACCSS kernel
    zzw1 = (
        weight_r * ker_raccss.take(idx_s2, axis=0).take(idx_r2)
        + (1 - weight_r) * ker_raccss.take(idx_s2, axis=0).take(idx_r)
    ) * weight_s + (
        weight_r * ker_raccss.take(idx_s, axis=0).take(idx_r2)
        + (1 - weight_r) * ker_raccss.take(idx_s, axis=0).take(idx_r)
    ) * (1 - weight_s)

    # Bilinear interpolation for RACCS kernel
    zzw2 = (
        weight_r * ker_raccs.take(idx_s2, axis=0).take(idx_r2)
        + (1 - weight_r) * ker_raccs.take(idx_s2, axis=0).take(idx_r)
    ) * weight_s + (
        weight_r * ker_raccs.take(idx_s, axis=0).take(idx_r2)
        + (1 - weight_r) * ker_raccs.take(idx_s, axis=0).take(idx_r)
    ) * (1 - weight_s)

    # Bilinear interpolation for SACCRG kernel
    zzw3 = (
        weight_r * ker_saccrg.take(idx_s2, axis=0).take(idx_r2)
        + (1 - weight_r) * ker_saccrg.take(idx_s2, axis=0).take(idx_r)
    ) * weight_s + (
        weight_r * ker_saccrg.take(idx_s, axis=0).take(idx_r2)
        + (1 - weight_r) * ker_saccrg.take(idx_s, axis=0).take(idx_r)
    ) * (1 - weight_s)

    return zzw1, zzw2, zzw3


@pytest.mark.parametrize("dtypes", [sp_dtypes])
def test_compute_accretion_interp(dtypes, externals, domain):
    IJ, K = domain[0] * domain[1], domain[2]
    IJK = domain[0] * domain[1] * domain[2]

    lambda_s = np.logspace(0.00001, 1.0, IJK).reshape((IJ, K))
    lambda_r = np.logspace(0.00001, 1.0, IJK).reshape((IJ, K))

    from ice3.phyex_common.xker_raccs import KER_RACCS, KER_RACCSS, KER_SACCRG

    (zzw1, zzw2, zzw4) = compute_accretion_interp(
        lambda_s=lambda_s,
        lambda_r=lambda_r,
        ACCINTP1S=externals["ACCINTP1S"],
        ACCINTP2S=externals["ACCINTP2S"],
        NACCLBDAS=externals["NACCLBDAS"],
        ACCINTP1R=externals["ACCINTP1R"],
        ACCINTP2R=externals["ACCINTP2R"],
        NACCLBDAR=externals["NACCLBDAR"],
        ker_raccss=KER_RACCSS,
        ker_raccs=KER_RACCS,
        ker_saccrg=KER_SACCRG,
    )

    print(f"zzw1 {zzw1.mean()}")
    print(f"zzw2 {zzw2.mean()}")
    print(f"zzw4 {zzw4.mean()}")


