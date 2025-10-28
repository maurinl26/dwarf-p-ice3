# -*- coding: utf-8 -*-
import pytest
from gt4py.cartesian.gtscript import stencil

from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import (CPU_BACKEND, DEBUG_BACKEND, GPU_BACKEND, dp_dtypes,
                            sp_dtypes)


@pytest.mark.parametrize("dtypes", [dp_dtypes, sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
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
