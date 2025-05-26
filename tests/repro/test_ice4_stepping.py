import pytest
from tests.conftest import get_backends, compile_fortran_stencil
from ifs_physics_common.framework.stencil import compile_stencil

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
        "mode_ice4_rimltc.F90", "mode_ice4_rimltc", "ice4_rimltc"
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

    ...

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

