from ifs_physics_common.framework.stencil import compile_stencil
from gt4py.storage import from_array
import numpy as np
from numpy.testing import assert_allclose
from ctypes import c_float, c_double
import pytest

import logging

from .conftest import NX, NY, NZ
from .conftest import compile_fortran_stencil


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", ["numpy", "gt:cpu_ifirst", "gt:cpu_kfirst"])
def test_thermo(gt4py_config, externals, fortran_dims, precision, backend):
    
    # Setting backend and precision
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)
    logging.info(f"GT4PyConfig types {gt4py_config.dtypes}")

    # Compilation of both gt4py and fortran stencils
    fortran_stencil = compile_fortran_stencil(
        "mode_thermo.F90", "mode_thermo", "latent_heat"
    )
    gt4py_stencil = compile_stencil("thermodynamic_fields", gt4py_config, externals)

    # setting external names with doctor norm
    keys_map = {
        "LVTT": "xlvtt",
        "LSTT": "xlstt",
        "CPV": "xcpv",
        "CI": "xci",
        "CL": "xcl",
        "TT": "xtt",
        "CPD": "xcpd",
    }
    
    externals = {fname: externals[pyname] for pyname, fname in keys_map.items()}

    # setting field names with doctor norm
    Py2F_Names = {
        "rv": "prv",
        "rc": "prc",
        "rr": "prr",
        "th": "pth",
        "ri": "pri",
        "rs": "prs",
        "rg": "prg",
        "exn": "pexn",
        "t": "zt",
        "ls": "zls",
        "lv": "zlv",
        "cph": "zcph",
    }
    
    # intent(out) fields from fortran subroutine
    FieldsNames_Out = ["zt", "zlv", "zls", "zcph"]
    
    F2Py_Names = dict(map(reversed, Py2F_Names.items()))

    # Generating random numpy arrays
    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(NX, NY, NZ),
            dtype=(c_float if gt4py_config.dtypes.float == np.float32 else c_double),
            order="F",
        )
        for name in Py2F_Names.keys()
    }

    # Buffer for GT4Py stencil (using gt4py.storage.from_array)
    GT4Py_FloatFieldsIJK = {
        name: from_array(
            field, dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend
        )
        for name, field in FloatFieldsIJK.items()
    }


    # Reshaping fields for fortran subroutine (I, J, K) to (IJ, K)
    Fortran_Fields = {
        Py2F_Names[pyname]: FloatFieldsIJK[pyname].reshape(
            fortran_dims["nijt"], fortran_dims["nkt"]
        )
        for pyname in Py2F_Names.keys()
    }
    
    # Calling stencils
    gt4py_stencil(**FloatFieldsIJK)

    fortran_output = fortran_stencil(
        krr=6, **Fortran_Fields, **fortran_dims, **externals
    )

    # Unzip output tuple from fmodpy
    Fields_Out = {
        F2Py_Names[name]: fortran_output[i] for i, name in enumerate(FieldsNames_Out)
    }

    # Comparing output means
    logging.info(f"Machine precision {np.finfo(float).eps}")
    for name in Fields_Out.keys():
        logging.info(f"{name} :: Mean gt4py   {GT4Py_FloatFieldsIJK[name][...].mean()}")
        logging.info(f"{name} :: Mean fortran {Fields_Out[name].mean()}")

    # Checking tolerance on output fields
    assert all(
        assert_allclose(
            Fields_Out[key],
            GT4Py_FloatFieldsIJK[key].reshape(
                fortran_dims["nijt"], fortran_dims["nkt"]
            ),
            rtol=1e-6,
        )
        for key in Fields_Out.keys()
    )
