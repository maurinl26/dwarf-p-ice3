import logging
from ctypes import c_double, c_float

import numpy as np
import pytest
from gt4py.storage import from_array
from ifs_physics_common.framework.stencil import compile_stencil
from numpy.testing import assert_allclose

from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import CPU_BACKENDS, DEBUG_BACKEND, GPU_BACKENDS


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(GPU_BACKENDS, marks=pytest.mark.gpu),
        pytest.param(CPU_BACKENDS, marks=pytest.mark.cpu),
    ]
)
def test_condensation(
    gt4py_config, externals, fortran_dims, precision, backend, domain, origin
):
    # Setting backend and precision
    backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)
    logging.info(f"GT4PyConfig types {gt4py_config.dtypes}")

    externals.update({"OCND2": False})
    externals.update({"OUSERI": True})
    logging.info(f"OCND2 : {externals['OCND2']}")
    condensation = compile_stencil("condensation", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil(
        "mode_condensation.F90", "mode_condensation", "condensation"
    )

    sigqsat = np.array(
        np.random.rand(domain[0], domain[1]),
        dtype=(c_float if gt4py_config.dtypes.float == np.float32 else c_double),
        order="F",
    )

    FloatFieldsIJK_Names = [
        "sigrc",
        "pabs",
        "sigs",
        "t",
        "rv_in",
        "ri_in",
        "rc_in",
        "t_out",
        "rv_out",
        "rc_out",
        "ri_out",
        "cldfr",
        "cph",
        "lv",
        "ls",
        "q1",
    ]

    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*domain.shape),
            dtype=(c_float if gt4py_config.dtypes.float == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_Names
    }

    # Updating temperature
    FloatFieldsIJK["t"] += 300

    sigqsat_gt4py = from_array(
        sigqsat, dtype=gt4py_config.dtypes.float, backend=backend
    )
    pabs_gt4py = from_array(
        FloatFieldsIJK["pabs"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    sigs_gt4py = from_array(
        FloatFieldsIJK["sigs"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    t_gt4py = from_array(
        FloatFieldsIJK["t"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    rv_in_gt4py = from_array(
        FloatFieldsIJK["rv_in"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    ri_in_gt4py = from_array(
        FloatFieldsIJK["ri_in"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    rc_in_gt4py = from_array(
        FloatFieldsIJK["rc_in"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    rv_out_gt4py = from_array(
        FloatFieldsIJK["rv_out"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    rc_out_gt4py = from_array(
        FloatFieldsIJK["rc_out"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    ri_out_gt4py = from_array(
        FloatFieldsIJK["ri_out"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    cldfr_gt4py = from_array(
        FloatFieldsIJK["cldfr"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    cph_gt4py = from_array(
        FloatFieldsIJK["cph"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    lv_gt4py = from_array(
        FloatFieldsIJK["lv"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    ls_gt4py = from_array(
        FloatFieldsIJK["ls"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    q1_gt4py = from_array(
        FloatFieldsIJK["q1"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )

    temporary_FloatFieldsIJK_Names = [
        "pv",
        "piv",
        "frac_tmp",
        "qsl",
        "qsi",
        "sigma",
        "cond_tmp",
        "a",
        "b",
        "sbar",
    ]

    temporary_FloatFieldsIJK = {
        name: np.zeros(
            domain.shape,
            dtype=(c_float if gt4py_config.dtypes.float == np.float32 else c_double),
            order="F",
        )
        for name in temporary_FloatFieldsIJK_Names
    }

    pv_gt4py = from_array(
        temporary_FloatFieldsIJK["pv"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    piv_gt4py = from_array(
        temporary_FloatFieldsIJK["piv"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    frac_tmp_gt4py = from_array(
        temporary_FloatFieldsIJK["frac_tmp"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    qsl_gt4py = from_array(
        temporary_FloatFieldsIJK["qsl"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    qsi_gt4py = from_array(
        temporary_FloatFieldsIJK["qsi"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    sigma_gt4py = from_array(
        temporary_FloatFieldsIJK["sigma"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    cond_tmp_gt4py = from_array(
        temporary_FloatFieldsIJK["cond_tmp"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    a_gt4py = from_array(
        temporary_FloatFieldsIJK["a"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    b_gt4py = from_array(
        temporary_FloatFieldsIJK["b"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )
    sbar_gt4py = from_array(
        temporary_FloatFieldsIJK["sbar"],
        dtype=gt4py_config.dtypes.float,
        backend=backend,
    )

    condensation(
        sigqsat=sigqsat_gt4py,
        pabs=pabs_gt4py,
        sigs=sigs_gt4py,
        t=t_gt4py,
        rv=rv_in_gt4py,
        ri=ri_in_gt4py,
        rc=rc_in_gt4py,
        rv_out=rv_out_gt4py,
        rc_out=rc_out_gt4py,
        ri_out=ri_out_gt4py,
        cldfr=cldfr_gt4py,
        cph=cph_gt4py,
        lv=lv_gt4py,
        ls=ls_gt4py,
        q1=q1_gt4py,
        domain=domain.shape,
        origin=origin,
    )

    logical_keys = {
        "osigmas": "LSIGMAS",
        "ocnd2": "OCND2",
        "ouseri": "OUSERI",
        "hfrac_ice": "FRAC_ICE_ADJUST",
        "hcondens": "CONDENS",
        "lstatnw": "LSTATNW",
    }

    constant_def = {
        "xrv": "RV",
        "xrd": "RD",
        "xalpi": "ALPI",
        "xbetai": "BETAI",
        "xgami": "GAMI",
        "xalpw": "ALPW",
        "xbetaw": "BETAW",
        "xgamw": "GAMW",
        "xtmaxmix": "TMAXMIX",
        "xtminmix": "TMINMIX",
    }

    fortran_externals = {
        **{fkey: externals[pykey] for fkey, pykey in logical_keys.items()},
        **{fkey: externals[pykey] for fkey, pykey in constant_def.items()},
    }

    F2Py_Mapping = {
        "ppabs": "pabs",
        "pt": "t",
        "prv_in": "rv_in",
        "prc_in": "rc_in",
        "pri_in": "ri_in",
        "psigs": "sigs",
        "psigqsat": "sigqsat",
        "plv": "lv",
        "pls": "ls",
        "pcph": "cph",
        "pt_out": "t",
        "prv_out": "rv_out",
        "prc_out": "rc_out",
        "pri_out": "ri_out",
        "pcldfr": "cldfr",
        "zq1": "q1",
        # Temporaries
        "zpv": "pv",
        "zpiv": "piv",
        "zfrac": "frac_tmp",
        "zqsl": "qsl",
        "zqsi": "qsi",
        "zsigma": "sigma",
        "zcond": "cond_tmp",
        "za": "a",
        "zb": "b",
        "zsbar": "sbar",
    }

    Py2F_Mapping = dict(map(reversed, F2Py_Mapping.items()))

    fortran_FloatFieldsIJK = {
        Py2F_Mapping[name]: FloatFieldsIJK[name].reshape(
            domain[0] *domain[1], domain[2]
        )
        for name in FloatFieldsIJK.keys()
    }

    result = fortran_stencil(
        psigsat=sigqsat.reshape(domain.shape[0] * domain.shape[1]),
        **fortran_FloatFieldsIJK,
        **fortran_dims,
        **fortran_externals,
    )

    FieldsOut_Names = [
        "pt_out",
        "prv_out",
        "prc_out",
        "pri_out",
        "pcldfr",
        "zq1",
        "pv",
        "piv",
        "zfrac",
        "zqsl",
        "zqsi",
        "zsigma",
        "zcond",
        "za",
        "zb",
        "zsbar",
    ]

    FieldsOut = {name: result[i] for i, name in enumerate(FieldsOut_Names)}

    logging.info(f"Mean pv_gt4py        {pv_gt4py.mean()}")
    logging.info(f"Mean pv_out          {FieldsOut['pv'].mean()}")

    logging.info(f"Mean piv_gt4py       {piv_gt4py.mean()}")
    logging.info(f"Mean piv             {FieldsOut['piv'].mean()}")

    logging.info(f"Mean frac_tmp_gt4py  {frac_tmp_gt4py.mean()}")
    logging.info(f"Mean zfrac           {FieldsOut['zfrac'].mean()}")

    logging.info(f"Mean qsl_gt4py       {qsl_gt4py.mean()}")
    logging.info(f"Mean zqsl            {FieldsOut['zqsl'].mean()}")

    logging.info(f"Mean qsi_gt4py       {qsi_gt4py.mean()}")
    logging.info(f"Mean zqsi            {FieldsOut['zqsi'].mean()}")

    logging.info(f"Mean sigma_gt4py     {sigma_gt4py.mean()}")
    logging.info(f"Mean zsigma          {FieldsOut['zsigma'].mean()}")

    logging.info(f"Mean cond_tmp_gt4py  {cond_tmp_gt4py.mean()}")
    logging.info(f"Mean zcond           {FieldsOut['zcond'].mean()}")

    logging.info(f"Mean a_gt4py         {a_gt4py.mean()}")
    logging.info(f"Mean za              {FieldsOut['za'].mean()}")

    logging.info(f"Mean b_gt4py         {b_gt4py.mean()}")
    logging.info(f"Mean zb              {FieldsOut['zb'].mean()}")

    logging.info(f"Mean sbar_gt4py      {sbar_gt4py.mean()}")
    logging.info(f"Mean zsbar           {FieldsOut['zsbar'].mean()}")

    logging.info(f"Machine precision {np.finfo(float).eps}")

    logging.info(f"Mean t_gt4py         {t_gt4py.mean()}")
    logging.info(f"Mean pt_out          {FieldsOut['pt_out'].mean()}")

    logging.info(f"Mean rv_gt4py        {rv_out_gt4py.mean()}")
    logging.info(f"Mean prv_out         {FieldsOut['prv_out'].mean()}")

        logging.info(f"Mean rc_out          {rc_out_gt4py.mean()}")
        logging.info(f"Mean prc_out         {FieldsOut["prc_out"].mean()}")

    logging.info(f"Mean ri_out_gt4py    {ri_out_gt4py.mean()}")
    logging.info(f"Mean ri_out          {FieldsOut['pri_out'].mean()}")

    logging.info(f"Mean cldfr_gt4py     {cldfr_gt4py.mean()}")
    logging.info(f"Mean pcldfr          {FieldsOut['pcldfr'].mean()}")

    logging.info(f"Mean q1_gt4py        {q1_gt4py.mean()}")
    logging.info(f"Mean zq1             {FieldsOut['zq1'].mean()}")

    assert_allclose(
        FieldsOut["pt_out"],
domain[0]*domain[1]),
        rtol=1e-6,
    )
    assert_allclose(
        FieldsOut["prv_out"],
        rv_out_gt4py.reshape(domain[0] *domain[1], domain[2]),
        rtol=1e-6,
    )
    assert_allclose(
        FieldsOut["prc_out"],
        rc_out_gt4py.reshape(domain[0] *domain[1], domain[2]),
        rtol=1e-6,
    )
    assert_allclose(
        FieldsOut["pri_out"],
        ri_out_gt4py.reshape(domain[0] *domain[1], domain[2]),
        rtol=1e-6,
    )

    assert_allclose(
        FieldsOut["pcldfr"],
        cldfr_gt4py.reshape(domain[0] *domain[1], domain[2]),
        rtol=1e-6,
    )
    assert_allclose(
        FieldsOut["zq1"],
        q1_gt4py.reshape(domain[0] *domain[1], domain[2]),
        rtol=1e-6,
    )
