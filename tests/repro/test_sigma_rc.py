import logging
from ctypes import c_float

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ice3.phyex_common.lookup_table import SRC_1D
from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import CPU_BACKEND, DEBUG_BACKEND, GPU_BACKEND


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
    ],
)
def test_sigrc_computation(externals, fortran_dims, domain, origin, precision, backend):
    logging.info(f"HLAMBDA3 {externals['LAMBDA3']}")

    I, J, K = domain

    fortran_stencil = compile_fortran_stencil(
        "mode_condensation.F90", "mode_condensation", "sigrc_computation"
    )

    FloatFieldsIJK_Names = ["q1", "sigrc"]
    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*domain),
            dtype=c_float,
            order="F",
        )
        for name in FloatFieldsIJK_Names
    }

    inq1 = np.zeros(domain, dtype=np.int32)

    from ice3.stencils.sigma_rc_dace import sigrc_computation

    compiled_sdfg = sigrc_computation.to_sdfg().compile()

    # dace
    compiled_sdfg(
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
    logging.info(f"Mean sigrc (dace)    {FloatFieldsIJK['sigrc'].mean()}")
    logging.info(f"Mean psigrc_out      {FieldsOut['psigrc'].mean()}")

    assert_allclose(
        FieldsOut["psigrc"],
        FloatFieldsIJK["sigrc"].reshape(grid.shape[0] * grid.shape[1], grid.shape[2]),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
    ],
)
def test_global_table(backend):
    fortran_global_table = compile_fortran_stencil(
        "mode_condensation.F90", "mode_condensation", "global_table"
    )

    global_table = np.ones((34), dtype=np.float32)
    global_table_out = fortran_global_table(out_table=global_table)

    logging.info(f"GlobalTable[0] : {global_table_out[0]}")
    logging.info(f"GlobalTable[5] : {global_table_out[5]}")
    logging.info(f"GlobalTable[33] : {global_table_out[33]}")

    assert_allclose(global_table_out, SRC_1D, rtol=1e-5)
