import logging
from ctypes import c_float

import numpy as np
from numpy.testing import assert_allclose
from tests.conftest import compile_fortran_stencil

from ice3_gt4py.phyex_common.tables import SRC_1D
from ice3_gt4py.stencils.sigma_rc_dace import sigrc_computation


def test_sigrc_computation_dace(grid):
    I, J, K = grid.shape

    q1 = np.ones((I, J, K), dtype=np.float32)
    inq1 = np.ones((I, J, K), dtype=np.int32)
    sigrc = np.ones((I, J, K), dtype=np.float32)

    sigrc_computation(
        q1=q1, inq1=inq1, src_1d=SRC_1D, sigrc=sigrc, LAMBDA3=0, I=I, J=J, K=K, F=34
    )


def test_sigrc_computation_stencil(
    gt4py_config, externals, fortran_dims, grid, origin
):

    logging.info(f"HLAMBDA3 {externals['LAMBDA3']}")

    I, J, K = grid.shape

    fortran_stencil = compile_fortran_stencil(
        "mode_condensation.F90", "mode_condensation", "sigrc_computation"
    )

    FloatFieldsIJK_Names = ["q1", "sigrc"]
    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*grid.shape),
            dtype=c_float,
            order="F",
        )
        for name in FloatFieldsIJK_Names
    }

    inq1 = np.zeros(grid.shape, dtype=np.int32)

    # dace
    sigrc_computation(
        q1=FloatFieldsIJK["q1"],
        inq1=inq1,
        src_1d=SRC_1D,
        sigrc=FloatFieldsIJK["sigrc"],
        LAMBDA3=0,
        I=I,
        J=J,
        K=K,
        F=34,
    )

    F2Py_Mapping = {"zq1": "q1", "psigrc": "sigrc"}
    Py2F_Mapping = dict(map(reversed, F2Py_Mapping.items()))

    fortran_FloatFieldsIJK = {
        Py2F_Mapping[name]: FloatFieldsIJK[name].reshape(
            grid.shape[0] * grid.shape[1], grid.shape[2]
        )
        for name in FloatFieldsIJK.keys()
    }

    inq1 = np.ones((grid.shape[0] * grid.shape[1], grid.shape[2]))

    result = fortran_stencil(
        inq1=inq1,
        hlambda3=externals["LAMBDA3"],
        **fortran_FloatFieldsIJK,
        **fortran_dims,
    )

    FieldsOut_Names = ["psigrc", "inq1"]

    FieldsOut = {name: result[i] for i, name in enumerate(FieldsOut_Names)}

    logging.info("\n Temporaries")
    logging.info(f"Mean inq1 (dace)   {inq1.mean()}")
    logging.info(f"Mean inq1_out      {FieldsOut['inq1'].mean()}")

    logging.info("\n Outputs")
    logging.info(f"Machine precision {np.finfo(float).eps}")
    logging.info(f"Mean sigrc (dace)    {FloatFieldsIJK["sigrc"].mean()}")
    logging.info(f"Mean psigrc_out      {FieldsOut['psigrc'].mean()}")

    assert_allclose(
        FieldsOut["psigrc"],
        FloatFieldsIJK["sigrc"].reshape(grid.shape[0] * grid.shape[1], grid.shape[2]),
        rtol=1e-6,
    )

