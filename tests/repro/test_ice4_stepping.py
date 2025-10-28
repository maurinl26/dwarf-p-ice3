import pytest

from tests.conftest import get_backends, compile_fortran_stencil
from ifs_physics_common.framework.stencil import compile_stencil
from tests.allocate_random_fields import draw_fields, allocate_fields, allocate_gt4py_fields, allocate_fortran_fields, \
    get_fields

from numpy.testing import assert_allclose

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_stepping_heat(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    # Setting backend and precision
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

    ice4_stepping_heat = compile_stencil("ice4_stepping_heat", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_stepping.F90", "mode_ice4_stepping", "ice4_stepping_heat"
    )

    FloatFieldsIJK_names = [
        "rv_t",
        "rc_t",
        "rr_t",
        "rs_t",
        "rg_t",
        "exn",
        "th_t",
        "ls_fact",
        "lv_fact",
        "t"
    ]

    f2py_names = {
        "prvt":"rv_t",
        "prct":"rc_t",
        "prrt":"rr_t",
        "prst":"rs_t",
        "prgt":"rg_t",
        "pexn":"exn",
        "ptht":"th_t",
        "zlsfact":"ls_fact",
        "zlvfact":"lv_fact",
        "zzt":"t"
    }

    fexternals = {
        fname: externals[pyname]
        for fname, pyname in {
        "xcpd":"CPD",
        "xcpv":"CPV",
        "xcl":"CL",
        "xci":"CI",
        "xtt":"TT",
        "xlvtt":"LVTT",
        "xlstt":"LSTT"
        }.items()
    }

    # Get random fields + gt4py buffer + fortran reshaping
    GT4Py_FloatFieldsIJK, Fortran_FloatFieldsIJK = get_fields(FloatFieldsIJK_names, f2py_names, gt4py_config, grid)

    ice4_stepping_heat(
        **GT4Py_FloatFieldsIJK,
        domain=grid.shape,
        origin=origin
        )

    result = fortran_stencil(
        **Fortran_FloatFieldsIJK,
        **fortran_packed_dims,
        **fexternals
    )

    zzt = result[0]
    zlsfact = result[1]
    zlvfact = result[2]

    assert_allclose(zzt, GT4Py_FloatFieldsIJK["t"])
    assert_allclose(zlsfact, GT4Py_FloatFieldsIJK["ls_fact"])
    assert_allclose(zlvfact, GT4Py_FloatFieldsIJK["lv_fact"])

