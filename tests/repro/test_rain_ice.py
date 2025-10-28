import pytest
from tests.conftest import get_backends


@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_rain_ice_init(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_initial_values_saving(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...



@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_precipitation_fraction_sigma(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_precipitation_fraction_liquid_content(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_compute_pdf(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...



@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_total_tendencies(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_correct_negativities(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...



@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_fog_deposition(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...
