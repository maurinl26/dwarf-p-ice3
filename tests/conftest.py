import xarray as xr
import pytest
import numpy as np

from ice3.phyex_common.phyex import Phyex

# fixtures
@pytest.fixture(name="ice_adjust_repro_ds", scope="module")
def ice_adjust_ds_fixture():
    ds = xr.open_dataset("./data/ice_adjust.nc", engine="netcdf4")
    return ds

@pytest.fixture(name="rain_ice_repro_ds", scope="module")
def rain_ice_ds_fixture():
    ds = xr.open_dataset("./data/rain_ice.nc", engine="netcdf4")
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

@pytest.fixture(name="tol", scope="module")
def sp_tol_fixture():
    return {
        "atol": 1e-6,
        "rtol": 1e-6,
    }

@pytest.fixture(name="tol", scope="module")
def dp_tol_fixture():
    return {
        "atol": 1e-8,
        "rtol": 1e-10,
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
def fortran_dims_fixture(domain):
    return {
        "nkt": domain[2],
        "nijt": domain[0] * domain[1],
        "nktb": 1,
        "nkte": domain[2],
        "nijb": 1,
        "nije": domain[0] * domain[1],
    }
    
@pytest.fixture(name="packed_dims", scope="module")
def packed_dims_fixture(domain):
    return {
        "kproma": domain[0] * domain[1] * domain[2],
        "ksize": domain[0] * domain[1] * domain[2]
    }
