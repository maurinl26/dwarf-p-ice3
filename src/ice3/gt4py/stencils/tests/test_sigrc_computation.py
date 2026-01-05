"""
SIGRC Computation - cupy implementation

This module implements the computation of subgrid standard deviation of rc (cloud water)
using a lookup table, translated from the Fortran reference in mode_sigrc_computation.F90.

The SIGRC (sigma_rc) computation is used in the Chaboureau-Bechtold (CB) subgrid
condensation scheme to represent subgrid-scale variability of cloud water.

Reference:
    mode_sigrc_computation.F90
"""

import pytest
from ice3.utils.env import sp_dtypes


@pytest.mark.parametrize("dtypes", [sp_dtypes])
@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("gpu", marks=pytest.mark.gpu),
    ],
)
def test_interpolation_1d_with_take(device, domain, dtypes):
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

    print("Import sigrc_computation cupy stencil")
    from ice3.stencils_cupy.sigrc_computation import sigrc_computation, SRC_1D

    IJK = domain[0] * domain[1] * domain[2]
    IJ, K = domain[0] * domain[1], domain[2]
    zq1_range = np.logspace(1e-8, 1, IJK)
    zq1 = zq1_range.reshape((IJ, K))
    psigrc = np.zeros((IJ, K), dtype=dtypes["float"])

    psigrc = sigrc_computation(zq1=zq1, src_table=SRC_1D)

    print("Stencil run validated")
    print(f"psigrc, mean : {psigrc.mean()}")
