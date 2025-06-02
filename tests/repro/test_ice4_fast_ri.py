import logging

import numpy as np
import pytest
from tests.conftest import compile_fortran_stencil, get_backends
from gt4py.storage import from_array
from ifs_physics_common.framework.stencil import compile_stencil
from numpy.testing import assert_allclose

from tests.allocate_random_fields import get_fields


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_fast_ri(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ldsoft = False

    # Setting backend and precision
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

    ice4_fast_ri = compile_stencil("ice4_fast_ri", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_fast_ri.F90", "mode_ice4_fast_ri", "ice4_fast_ri"
    )

    logging.info(f"Machine precision {np.finfo(np.float32).eps}")
    logging.info(f"Machine precision {np.finfo(np.float32).eps}")

    ldcompute = np.ones(
        grid.shape,
        dtype=bool,
        order="F",
    )

    ldcompute_gt4py = from_array(ldcompute, dtype=bool, backend=gt4py_config.backend)

    fexternals = {
        fname: externals[pyname]
        for fname, pyname in {
            "c_rtmin": "C_RTMIN",
            "i_rtmin": "I_RTMIN",
            "xlbexi": "LBEXI",
            "xlbi": "LBI",
            "x0depi": "O0DEPI",
            "x2depi": "O2DEPI",
            "xdi": "DI",
        }.items()
    }

    f2py_mapping = {
        "prhodref": "rhodref",
        "pai": "ai",
        "pcj": "cj",
        "pcit": "cit",
        "pssi": "ssi",
        "prct": "rct",
        "prit": "rit",
        "prcberi": "rc_beri_tnd",
    }

    GT4Py_FloatFieldsIJK, Fortran_FloatFieldsIJK = get_fields(list(f2py_mapping.values()), f2py_mapping, gt4py_config, grid, )

    ice4_fast_ri(
        **GT4Py_FloatFieldsIJK,
        ldcompute=ldcompute_gt4py,
        ldsoft=ldsoft,
        domain=grid.shape,
        origin=origin,
    )


    logging.info(f"ldsoft fortran {ldsoft}")
    result = fortran_stencil(
        ldsoft=ldsoft,
        ldcompute=ldcompute,
        **Fortran_FloatFieldsIJK,
        **fexternals,
        **fortran_packed_dims,
    )

    rcberi_out = result

    logging.info(f"Mean rc_beri_tnd_gt4py   {GT4Py_FloatFieldsIJK['rc_beri_tnd'].mean()}")
    logging.info(f"Mean rcberi_out          {rcberi_out.mean()}")

    assert_allclose(GT4Py_FloatFieldsIJK['rc_beri_tnd'].ravel(), rcberi_out, 1e-6)


