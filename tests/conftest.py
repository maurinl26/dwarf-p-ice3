import xarray as xr
import pytest
import numpy as np

from ice3.phyex_common.phyex import Phyex

# fixtures
@pytest.fixture(name="ice_adjust_repro_ds", scope="module")
def ice_adjust_ds_fixture():
    ds = xr.open_dataset("./data/ice_adjust.nc", engine="netcdf4")
    return ds

@pytest.fixture(name="sp_dtypes", scope="module")
def sp_dtypes_fixture():
    return {
            "float": np.float32,
            "int": np.int32,
            "bool": np.bool_
        }

@pytest.fixture(name="dp_dtypes", scope="module")
def dp_dtypes_fixtures():
    return {
        "float": np.float64,
        "int": np.int64,
        "bool": np.bool_
    }

@pytest.fixture(name="domain", scope="module")
def domain_fixture():
    return 50, 50, 15

@pytest.fixture(name="origin", scope="module")
def origin_fixture():
    return 0, 0, 0

@pytest.fixture(name="phyex", scope="module")
def phyex_fixture():
    return Phyex("AROME")

@pytest.fixture(name="externals", scope="module")
def externals_fixture(phyex):
    return phyex.to_externals()


################ Fortran for fixtures ##############
@pytest.fixture(name="fortran_dims", scope="module")
def fortran_dims_fixture(grid):
    return {
        "nkt": grid.shape[2],
        "nijt": grid.shape[0] * grid.shape[1],
        "nktb": 1,
        "nkte": grid.shape[2],
        "nijb": 1,
        "nije": grid.shape[0] * grid.shape[1],
    }
    
@pytest.fixture(name="packed_dims", scope="module")
def packed_dims_fixture(grid):
    return {
        "kproma": grid.shape[0] * grid.shape[1] * grid.shape[2],
        "ksize": grid.shape[0] * grid.shape[1] * grid.shape[2]
    }