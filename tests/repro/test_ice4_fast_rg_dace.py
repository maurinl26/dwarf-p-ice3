import logging
from ctypes import c_double, c_float

import numpy as np
import pytest
from tests.conftest import compile_fortran_stencil, get_backends
from gt4py.storage import from_array
from ifs_physics_common.framework.stencil import compile_stencil
from numpy.testing import assert_allclose

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_snow_collection_on_graupel(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    # Setting backend and precision
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

    ice4_nucleation_gt4py = compile_stencil("ice4_nucleation", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_nucleation.F90", "mode_ice4_nucleation", "ice4_nucleation"
    )

    ldcompute = np.array(
        np.random.rand(*grid.shape),
        dtype=bool,
        order="F",
    )

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_rain_drops_accretion_on_graupel(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...
