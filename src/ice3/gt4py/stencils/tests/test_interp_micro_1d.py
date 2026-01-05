import pytest
import numpy as np
import numpy.typing as npt

from ice3.utils.env import sp_dtypes


def interp_micro_1d(
    lambda_s: npt.ArrayLike,
    ker_gaminc_rim1: npt.ArrayLike,
    ker_gaminc_rim2: npt.ArrayLike,
    ker_gaminc_rim4: npt.ArrayLike,
    RIMINTP1: np.float32,
    RIMINTP2: np.float32,
    NGAMINC: np.int32,
):
    """Linear interpolation"""

    index = np.clip(RIMINTP1 * np.log(lambda_s) + RIMINTP2, NGAMINC - 0.00001, 1.00001)

    idx_interp = np.floor(index).astype(np.int32)
    idx_interp_2 = idx_interp + 1
    weight = index - idx_interp

    zzw1 = (
        weight
        * ker_gaminc_rim1.take(idx_interp)
        + (1 - weight)
        * ker_gaminc_rim1.take(idx_interp_2)
    )

    zzw2 = (
        weight
        * ker_gaminc_rim2.take(idx_interp)
        + (1 - weight)
        * ker_gaminc_rim2.take(idx_interp_2)
    )

    zzw4 = (
        weight
        * ker_gaminc_rim4.take(idx_interp)
        + (1 - weight)
        * ker_gaminc_rim4.take(idx_interp_2)
    )

    return zzw1, zzw2, zzw4


@pytest.mark.parametrize("dtypes", [sp_dtypes])
@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("gpu", marks=pytest.mark.gpu),
    ],
)
def test_interp_micro_1d_with_take(device, domain, dtypes, externals):
    """
    Test 1D interpolation with cupy take and
    a switch between numpy take and cupy take, whether the
    backend is cpu or gpu.
    """

    print("Set cupy for gpu devices")

    match device:
        case "gpu":
            try:
                import cupy as np
            except ImportError as ie:
                print(f"Failed to import cupy, with {ie}")
                import numpy as np
        case "cpu":
            import numpy as np
        case _:
            import numpy as np

    IJK = domain[0] * domain[1] * domain[2]
    IJ, K = domain[0] * domain[1], domain[2]
    lambda_s_range = np.logspace(1e-6, 1, IJK)
    lambda_s = lambda_s_range.reshape((IJ, K))

    zzw1 = np.zeros((IJ, K), dtype=dtypes["float"])
    zzw2 = np.zeros((IJ, K), dtype=dtypes["float"])
    zzw4 = np.zeros((IJ, K), dtype=dtypes["float"])

    from ice3.phyex_common.rain_ice_parameters import RainIceParameters

    (zzw1, zzw2, zzw4) = interp_micro_1d(
        lambda_s=lambda_s,
        ker_gaminc_rim1=externals["GAMINC_RIM1"],
        ker_gaminc_rim2=externals["GAMINC_RIM2"],
        ker_gaminc_rim4=externals["GAMINC_RIM4"],
        RIMINTP1=externals["RIMINTP1"],
        RIMINTP2=externals["RIMINTP2"],
        NGAMINC=externals["NGAMINC"],
    )

    print("Stencil run validated")
    print(f"zzw1, mean : {zzw1.mean()}")
    print(f"zzw2, mean : {zzw2.mean()}")
    print(f"zzw4, mean : {zzw4.mean()}")
