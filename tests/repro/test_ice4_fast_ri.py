import logging

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array
from numpy.testing import assert_allclose

from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import (CPU_BACKEND, DEBUG_BACKEND, GPU_BACKEND, dp_dtypes,
                            sp_dtypes)


@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
    ],
)
def test_ice4_fast_ri(
    externals, fortran_packed_dims, dtypes, backend, domain, origin, ldsoft = False
):
    # Setting backend and precision
    from ice3.stencils.ice4_fast_ri import ice4_fast_ri

    ice4_fast_ri = stencil(
        backend,
        name="ice4_fast_ri",
        definition=ice4_fast_ri,
        dtypes=dtypes,
        externals=externals,
    )
    fortran_stencil = compile_fortran_stencil(
        fortran_script="mode_ice4_fast_ri.F90",
        fortran_module="mode_ice4_fast_ri",
        fortran_stencil="ice4_fast_ri",
    )

    logging.info(f"Machine precision {np.finfo(np.float32).eps}")
    logging.info(f"Machine precision {np.finfo(np.float32).eps}")

    ldcompute = np.ones(
        domain,
        dtype=bool,
        order="F",
    )

    ldcompute_gt4py = from_array(ldcompute, dtype=np.int32, backend=backend)

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

    GT4Py_FloatFieldsIJK, Fortran_FloatFieldsIJK = get_fields(
        list(f2py_mapping.values()),
        f2py_mapping,
        gt4py_config,
        grid,
    )

    ice4_fast_ri(
        **GT4Py_FloatFieldsIJK,
        ldcompute=ldcompute_gt4py,
        ldsoft=ldsoft,
        domain=domain,
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

    logging.info(
        f"Mean rc_beri_tnd_gt4py   {GT4Py_FloatFieldsIJK['rc_beri_tnd'].mean()}"
    )
    logging.info(f"Mean rcberi_out          {rcberi_out.mean()}")

    assert_allclose(GT4Py_FloatFieldsIJK["rc_beri_tnd"].ravel(), rcberi_out, rtol=1e-6, atol=1e-6)
