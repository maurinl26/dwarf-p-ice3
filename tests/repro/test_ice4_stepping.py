import pytest

from tests.conftest import get_backends, compile_fortran_stencil
from ifs_physics_common.framework.stencil import compile_stencil
from tests.allocate_random_fields import draw_fields, allocate_fields, allocate_gt4py_fields, allocate_fortran_fields

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
        "plsfact":"ls_fact",
        "plvfact":"lv_fact",
        "pt":"t"
    }

    fexternals ={
        fname: externals[pyname]
        for fname, pyname in {
        "xcpd":"cpd",
        "xcpv":"cpv",
        "xcl":"cl",
        "xci":"ci",
        "xtt":"tt",
        "xlvtt":"lvtt",
        "xlstt":"lstt"
    }.items()
        }

    FloatFieldsIJK = draw_fields(FloatFieldsIJK_names, gt4py_config, grid)
    GT4Py_FloatFieldsIJK = allocate_gt4py_fields(FloatFieldsIJK_names, gt4py_config, grid)
    allocate_fields(GT4Py_FloatFieldsIJK, FloatFieldsIJK)
    Fortran_FloatFieldsIJK = allocate_fortran_fields(f2py_names, FloatFieldsIJK)

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

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_step_limiter(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_mixing_ratio_step_limiter(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_state_update(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_external_tendencies_update(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_tmicro_init(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...

@pytest.mark.skip("Variable initialization")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_tsoft_init(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ldcompute_init(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...

