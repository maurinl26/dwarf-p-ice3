import pytest
from tests.conftest import get_backends

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_rain_ice_nucleation_pre_processing(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_nucleation(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_rain_ice_nucleation_post_processing(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ...