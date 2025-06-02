import logging
from ctypes import c_double, c_float

import numpy as np
import pytest

from tests.allocate_random_fields import get_fields
from tests.conftest import compile_fortran_stencil, get_backends
from gt4py.storage import from_array
from ifs_physics_common.framework.stencil import compile_stencil
from numpy.testing import assert_allclose

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_rain_contact_freezing(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    # Setting backend and precision
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

    from src.ice3_gt4py.stencils.ice4_fast_rg_split import rain_contact_freezing

    rain_contact_freezing = compile_stencil("rain_contact_freezing", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_fast_rg.F90", "mode_ice4_fast_rg", "rain_contact_freezing"
    )

    FloatFieldsIJK_Names = [
        "t",
        "rhodref",
        "pres",
        "rvt",
        "rrt",
        "rit",
        "rgt",
        "rct",
        "cit",
        "lbdar",
        "lbdag",
        "ricfrr",
        "rrcfrig",
        "ricfrrg",
        "rg_ridry_tnd",
        "rg_riwet_tnd",
    ]

    f2py_mapping = {
        "pt": "t",
        "prhodref": "rhodref",
        "ppres": "pres",
        "prvt": "rvt",
        "prrt": "rrt",
        "prit": "rit",
        "prgt": "prgt",
        "prct": "rct",
        "pcit": "cit",
        "plbdar": "lbdar",
        "plbdag": "lbdag",
        "pricfrr": "ricfrr",
        "prrcfrig": "rrcfrig",
        "pricfrrg": "ricfrrg",
        "rg_ridry_tnd":"",
        "rg_riwet_tnd":"",
    }

    GT4Py_FloatFieldsIJK, Fortran_FloatFieldsIJK = get_fields(list(f2py_mapping.values()), f2py_mapping, gt4py_config, grid)

    rain_contact_freezing(
        **GT4Py_FloatFieldsIJK,
        domain=grid.shape,
        origin=origin
    )



@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_graupel_growth(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...

