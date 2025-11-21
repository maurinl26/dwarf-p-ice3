# -*- coding: utf-8 -*-
import pytest
from gt4py.cartesian.gtscript import stencil

from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import dp_dtypes, sp_dtypes


@pytest.mark.parametrize("dtypes", [dp_dtypes, sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("debug", marks=pytest.mark.debug),
        pytest.param("numpy", marks=pytest.mark.numpy),
        pytest.param("gt:cpu_ifirst", marks=pytest.mark.cpu),
        pytest.param("gt:gpu", marks=pytest.mark.gpu),
    ],
)
def test_ice4_rrhong(dtypes, backend, externals):
    from ice3.stencils.ice4_rrhong import ice4_rrhong

    ice4_rrhong_gt4py = stencil(
        backend,
        definitions=ice4_rrhong,
        name="ice4_rrhong",
        dtypes=dtypes,
        externals=externals,
    )

    ice4_rrhong_fortran = compile_fortran_stencil(
        "mode_ice4_rrhong.F90", "mode_ice4_rrhong", "ice4_rrhong"
    )

    ...
