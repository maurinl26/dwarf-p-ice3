import logging
from ctypes import c_double, c_float

import numpy as np
import pytest
from gt4py.storage import from_array, ones
from ifs_physics_common.framework.stencil import compile_stencil
from numpy.testing import assert_allclose

from ice3.phyex_common.tables import SRC_1D

from tests.conftest import compile_fortran_stencil, get_backends


def allocate_random_fields(names, gt4py_config, grid, dtype=None, zeros=False):
    dtype = dtype or (c_float if gt4py_config.dtypes.float == np.float32 else c_double)
    arr_func = np.zeros if zeros else np.random.rand
    fields = {name: np.array(arr_func(*grid.shape), dtype=dtype, order="F") for name in names}
    gt4py_buffers = {name: from_array(fields[name], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend) for name in names}
    return fields, gt4py_buffers

@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_condensation(gt4py_config, externals, fortran_dims, precision, backend, grid, origin):
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)
    externals.update({"OCND2": False, "OUSERI": True})
    condensation = compile_stencil("condensation", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil("mode_condensation.F90", "mode_condensation", "condensation")
    dtype = c_float if gt4py_config.dtypes.float == np.float32 else c_double
    sigqsat = np.array(np.random.rand(grid.shape[0], grid.shape[1]), dtype=dtype, order="F")
    main_fields = ["sigrc", "pabs", "sigs", "t", "rv_in", "ri_in", "rc_in", "t_out", "rv_out", "rc_out", "ri_out", "cldfr", "cph", "lv", "ls", "q1"]
    fields, gt4py_buffers = allocate_random_fields(main_fields, gt4py_config, grid, dtype)
    fields["t"] += 300
    temp_fields = ["pv", "piv", "frac_tmp", "qsl", "qsi", "sigma", "cond_tmp", "a", "b", "sbar"]
    temp_fields_dict, temp_gt4py_buffers = allocate_random_fields(temp_fields, gt4py_config, grid, dtype, zeros=True)
    # GT4Py buffers
    sigqsat_gt4py = from_array(sigqsat, dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    # Run stencil
    condensation(
        sigqsat=sigqsat_gt4py,
        pabs=gt4py_buffers["pabs"],
        sigs=gt4py_buffers["sigs"],
        t=gt4py_buffers["t"],
        rv=gt4py_buffers["rv_in"],
        ri=gt4py_buffers["ri_in"],
        rc=gt4py_buffers["rc_in"],
        rv_out=gt4py_buffers["rv_out"],
        rc_out=gt4py_buffers["rc_out"],
        ri_out=gt4py_buffers["ri_out"],
        cldfr=gt4py_buffers["cldfr"],
        cph=gt4py_buffers["cph"],
        lv=gt4py_buffers["lv"],
        ls=gt4py_buffers["ls"],
        q1=gt4py_buffers["q1"],
        pv=temp_gt4py_buffers["pv"],
        piv=temp_gt4py_buffers["piv"],
        frac_tmp=temp_gt4py_buffers["frac_tmp"],
        qsl=temp_gt4py_buffers["qsl"],
        qsi=temp_gt4py_buffers["qsi"],
        sigma=temp_gt4py_buffers["sigma"],
        cond_tmp=temp_gt4py_buffers["cond_tmp"],
        a=temp_gt4py_buffers["a"],
        b=temp_gt4py_buffers["b"],
        sbar=temp_gt4py_buffers["sbar"],
        domain=grid.shape,
        origin=origin
    )
    # Fortran mapping
    F2Py_Mapping = {"ppabs": "pabs", "pt": "t", "prv_in": "rv_in", "prc_in": "rc_in", "pri_in": "ri_in", "psigs": "sigs", "psigqsat": "sigqsat", "plv": "lv", "pls": "ls", "pcph": "cph", "pt_out": "t", "prv_out": "rv_out", "prc_out": "rc_out", "pri_out": "ri_out", "pcldfr": "cldfr", "zq1": "q1", "zpv": "pv", "zpiv": "piv", "zfrac": "frac_tmp", "zqsl": "qsl", "zqsi": "qsi", "zsigma": "sigma", "zcond": "cond_tmp", "za": "a", "zb": "b", "zsbar": "sbar"}
    Py2F_Mapping = dict(map(reversed, F2Py_Mapping.items()))
    all_fields = {**fields, **temp_fields_dict}
    fortran_FloatFieldsIJK = {Py2F_Mapping[name]: all_fields[name].reshape(grid.shape[0]*grid.shape[1], grid.shape[2]) for name in all_fields.keys()}
    logical_keys = {"osigmas": "LSIGMAS", "ocnd2": "OCND2", "ouseri": "OUSERI", "hfrac_ice": "FRAC_ICE_ADJUST", "hcondens": "CONDENS", "lstatnw": "LSTATNW"}
    constant_def = {"xrv": "RV", "xrd": "RD", "xalpi": "ALPI", "xbetai": "BETAI", "xgami": "GAMI", "xalpw": "ALPW", "xbetaw": "BETAW", "xgamw": "GAMW", "xtmaxmix": "TMAXMIX", "xtminmix": "TMINMIX"}
    fortran_externals = {**{fkey: externals[pykey] for fkey, pykey in logical_keys.items()}, **{fkey: externals[pykey] for fkey, pykey in constant_def.items()}}
    result = fortran_stencil(psigsat=sigqsat.reshape(grid.shape[0]*grid.shape[1]), **fortran_FloatFieldsIJK, **fortran_dims, **fortran_externals)
    FieldsOut_Names = ["pt_out", "prv_out", "prc_out", "pri_out", "pcldfr", "zq1", "pv", "piv", "zfrac", "zqsl", "zqsi", "zsigma", "zcond", "za", "zb", "zsbar"]
    FieldsOut = {name: result[i] for i, name in enumerate(FieldsOut_Names)}
    assert_allclose(FieldsOut["pt_out"], gt4py_buffers["t"].reshape(grid.shape[0]*grid.shape[1], grid.shape[2]), rtol=1e-6)
    assert_allclose(FieldsOut["prv_out"], gt4py_buffers["rv_out"].reshape(grid.shape[0]*grid.shape[1], grid.shape[2]), rtol=1e-6)
    assert_allclose(FieldsOut["prc_out"], gt4py_buffers["rc_out"].reshape(grid.shape[0]*grid.shape[1], grid.shape[2]), rtol=1e-6)
    assert_allclose(FieldsOut["pri_out"], gt4py_buffers["ri_out"].reshape(grid.shape[0]*grid.shape[1], grid.shape[2]), rtol=1e-6)
    assert_allclose(FieldsOut["pcldfr"], gt4py_buffers["cldfr"].reshape(grid.shape[0]*grid.shape[1], grid.shape[2]), rtol=1e-6)
    assert_allclose(FieldsOut["zq1"], gt4py_buffers["q1"].reshape(grid.shape[0]*grid.shape[1], grid.shape[2]), rtol=1e-6)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_sigrc_computation(
    gt4py_config, externals, fortran_dims, grid, origin, precision, backend
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
    logging.info(f"Mean sigrc (dace)    {FloatFieldsIJK["sigrc"].mean()}")
    logging.info(f"Mean psigrc_out      {FieldsOut['psigrc'].mean()}")

    assert_allclose(
        FieldsOut["psigrc"],
        FloatFieldsIJK["sigrc"].reshape(grid.shape[0] * grid.shape[1], grid.shape[2]),
        rtol=1e-6,
    )


def test_global_table():
    
    fortran_global_table = compile_fortran_stencil(
        "mode_condensation.F90", 
        "mode_condensation", 
        "global_table"
        )
    
    global_table = np.ones((34), dtype=np.float32)
    global_table_out = fortran_global_table(out_table=global_table)
        
    logging.info(f"GlobalTable[0] : {global_table_out[0]}")
    logging.info(f"GlobalTable[5] : {global_table_out[5]}")
    logging.info(f"GlobalTable[33] : {global_table_out[33]}")
        
    assert_allclose(global_table_out, SRC_1D, rtol=1e-5)
    
   