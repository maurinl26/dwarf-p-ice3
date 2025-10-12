from ifs_physics_common.framework.stencil import compile_stencil
from gt4py.storage import from_array
import numpy as np
from numpy.testing import assert_allclose
import pytest
from ctypes import c_float, c_double

import logging

from ice3.utils.compile_fortran_stencil import compile_fortran_stencil
from ice3.utils.env import BACKEND_LIST




@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", BACKEND_LIST)
def test_ice_adjust(gt4py_config, externals, fortran_dims, precision, backend, grid, origin):
    # Setting backend and precision
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)
    logging.info(f"GT4PyConfig types {gt4py_config.dtypes}")


    fortran_externals = {
        fname: externals[pyname]
        for fname, pyname in externals_mapping.items()
    }

    ice_adjust = compile_stencil("ice_adjust", gt4py_config, externals)

    FloatFieldsIJK_Names = [
        "sigqsat",
        "pabs",
        "sigs",
        "th",
        "exn",
        "exn_ref",
        "rho_dry_ref",
        "t",
        "rv",
        "ri",
        "rc",
        "rr",
        "rs",
        "rg",
        "cf_mf",
        "rc_mf",
        "ri_mf",
        "rv_out",
        "rc_out",
        "ri_out",
        "hli_hri",
        "hli_hcf",
        "hlc_hrc",
        "hlc_hcf",
        "ths",
        "rvs",
        "rcs",
        "ris",
        "cldfr",
        "cph",
        "lv",
        "ls",
        "q1",
        "sigma_rc",
    ]

    from ice3.phyex_common.lookup_table import SRC_1D

    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*grid.shape),
            dtype=(c_float if gt4py_config.dtypes.float == np.float32 else c_double),
            order="F",
        ) for name in FloatFieldsIJK_Names
    }

    state_gt4py = {
        key: from_array(
            FloatFieldsIJK[key],
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        ) for key in FloatFieldsIJK_Names
    }

    ice_adjust(
        **state_gt4py,
        domain=grid.shape,
        origin=origin,
    )

    logging.info(f"Machine precision {np.finfo(float).eps}")


