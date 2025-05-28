import logging
from ctypes import c_double, c_float

import numpy as np
import pytest
from tests.conftest import compile_fortran_stencil, get_backends
from tests.allocate_random_fields import draw_fields, allocate_gt4py_fields, allocate_fields, allocate_fortran_fields
from gt4py.storage import from_array
from ifs_physics_common.framework.stencil import compile_stencil
from numpy.testing import assert_allclose
import gt4py

@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_nucleation(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    # Setting backend and precision
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

    ice4_nucleation_gt4py = compile_stencil("ice4_nucleation", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_nucleation.F90", "mode_ice4_nucleation", "ice4_nucleation"
    )

    ldcompute = np.array(
        np.random.rand(*grid.shape),
        dtype=bool,
        order="F",
    )

    FloatFieldsIJK_Names = [
        "tht",
        "pabst",
        "rhodref",
        "exn",
        "lsfact",
        "t",
        "rvt",
        "cit",
        "rvheni_mr",
        "ssi",
    ]

    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*grid.shape),
            dtype=(c_float if gt4py_config.dtypes.float == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_Names
    }

    GT4Py_FloatFieldsIJK = {
        name: gt4py.storage.zeros(
            shape=grid.shape,
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        for name in FloatFieldsIJK_Names
    }

    # Allocate
    for name in GT4Py_FloatFieldsIJK.keys():
        GT4Py_FloatFieldsIJK[name] = FloatFieldsIJK[name]


    # Mapping to fortran
    externals_mapping = {
            "xtt": "TT",
            "v_rtmin": "V_RTMIN",
            "xalpw": "ALPW",
            "xbetaw": "BETAW",
            "xgamw": "GAMW",
            "xalpi": "ALPI",
            "xbetai": "BETAI",
            "xgami": "GAMI",
            "xepsilo": "EPSILO",
            "xnu10": "NU10",
            "xnu20": "NU20",
            "xalpha1": "ALPHA1",
            "xalpha2": "ALPHA2",
            "xbeta1": "BETA1",
            "xbeta2": "BETA2",
            "xmnu0": "MNU0",
            "lfeedbackt": "LFEEDBACKT",
        }

    fortran_externals = {
            fkey: externals[pykey] for fkey, pykey in externals_mapping.items()
        }

    f2py_mapping = {
            "ptht": "tht",
            "ppabst": "pabst",
            "prhodref": "rhodref",
            "pexn": "exn",
            "plsfact": "lsfact",
            "pt": "t",
            "prvt": "rvt",
            "pcit": "cit",
            "prvheni_mr": "rvheni_mr",
        }

    fortran_FloatFieldsIJK = {
            name: FloatFieldsIJK[value].ravel() for name, value in f2py_mapping.items()
        }

    # Calls
    ldcompute_gt4py = from_array(
        ldcompute,
        dtype=np.bool_,
        backend=gt4py_config.backend,
    )

    ice4_nucleation_gt4py(
        **GT4Py_FloatFieldsIJK,
        ldcompute=ldcompute_gt4py,
        domain=grid.shape,
        origin=origin,
    )

    result = fortran_stencil(
        ldcompute=ldcompute.ravel(),
        **fortran_FloatFieldsIJK,
        **fortran_packed_dims,
        **fortran_externals,
    )

    cit_out = result[0]
    rvheni_mr_out = result[1]

    logging.info(f"Machine precision {np.finfo(float).eps}")

    logging.info(f"Mean cit_out     {cit_out.mean()}")
    logging.info(f"Mean cit_gt4py   {GT4Py_FloatFieldsIJK['cit'].mean()}")

    logging.info(f"Mean rvheni_mr_out     {rvheni_mr_out.mean()}")
    logging.info(f"Mean rvheni_mr_gt4py   {GT4Py_FloatFieldsIJK['rvheni_mr'].mean()}")

    assert_allclose(cit_out, GT4Py_FloatFieldsIJK['cit'].ravel(), 10e-6)
    assert_allclose(rvheni_mr_out, GT4Py_FloatFieldsIJK['rvheni_mr'].ravel(), 10e-6)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_rimltc(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    # Setting backend and precision
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

    ice4_rimltc_gt4py = compile_stencil("ice4_rimltc", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_rimltc.F90", "mode_ice4_rimltc", "ice4_rimltc"
    )

    ldcompute = np.array(
        np.random.rand(*grid.shape),
        dtype=bool,
        order="F",
    )

    FloatFieldsIJK_Names = [
        "t",
        "exn",
        "lvfact",
        "lsfact",
        "tht",
        "rit",
        "rimltc_mr",
    ]

    FloatFieldsIJK = draw_fields(
        FloatFieldsIJK_Names,
        gt4py_config,
        grid
    )

    GT4Py_FloatFieldsIJK = allocate_gt4py_fields(
        FloatFieldsIJK_Names,
        gt4py_config,
        grid
    )

    allocate_fields(GT4Py_FloatFieldsIJK, FloatFieldsIJK)

    fortran_externals = {"xtt": externals["TT"], "lfeedbackt": externals["LFEEDBACKT"]}

    f2py_mapping = {
        "pexn": "exn",
        "plvfact": "lvfact",
        "plsfact": "lsfact",
        "pt": "t",
        "ptht": "tht",
        "prit": "rit",
        "primltc_mr": "rimltc_mr",
    }

    Fortran_FloatFieldsIJK = allocate_fortran_fields(
        f2py_mapping,
        FloatFieldsIJK,
    )


    ldcompute_gt4py = from_array(
        ldcompute,
        dtype=bool,
        backend=gt4py_config.backend,
    )

    ice4_rimltc_gt4py(
        **GT4Py_FloatFieldsIJK,
        ldcompute=ldcompute_gt4py,
        domain=grid.shape,
        origin=origin,
    )

    result = fortran_stencil(
        ldcompute=ldcompute.ravel(),
        **Fortran_FloatFieldsIJK,
        **fortran_packed_dims,
        **fortran_externals,
    )

    rimltc_mr_out = result[0]

    logging.info(f"Machine precision {np.finfo(float).eps}")
    assert_allclose(rimltc_mr_out, GT4Py_FloatFieldsIJK['rimltc_mr'].ravel(), rtol=10e-6)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_slow(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    # Setting backend and precision
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

    ice4_slow_gt4py = compile_stencil("ice4_slow", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_slow.F90", "mode_ice4_slow", "ice4_slow"
    )

    ldcompute = np.array(
        np.random.rand(*grid.shape),
        dtype=bool,
        order="F",
    )

    FloatFieldsIJK_Names = [
        "rhodref",
        "t",
        "ssi",
        "rvt",
        "rct",
        "rit",
        "rst",
        "rgt",
        "lbdas",
        "lbdag",
        "ai",
        "cj",
        "hli_hcf",
        "hli_hri",
        "rc_honi_tnd",
        "rv_deps_tnd",
        "ri_aggs_tnd",
        "ri_auts_tnd",
        "rv_depg_tnd",
    ]

    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*grid.shape),
            dtype=(c_float if gt4py_config.dtypes.float == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_Names
    }

    GT4Py_FloatFieldsIJK = {
        name: gt4py.storage.zeros(
            shape=grid.shape,
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        for name in FloatFieldsIJK_Names
    }

    # Allocate
    for name in GT4Py_FloatFieldsIJK.keys():
        GT4Py_FloatFieldsIJK[name] = FloatFieldsIJK[name]

    ldcompute_gt4py = from_array(
        ldcompute,
        dtype=bool,
        backend=gt4py_config.backend,
    )

    ldsoft = True

    ice4_slow_gt4py(
        **GT4Py_FloatFieldsIJK,
        ldcompute=ldcompute_gt4py,
        ldsoft=ldsoft,
        domain=grid.shape,
        origin=origin,
    )

    fortran_externals = {
        "xtt": externals["TT"],
        "v_rtmin": externals["V_RTMIN"],
        "c_rtmin": externals["C_RTMIN"],
        "i_rtmin": externals["I_RTMIN"],
        "s_rtmin": externals["S_RTMIN"],
        "g_rtmin": externals["G_RTMIN"],
        "xexiaggs": externals["EXIAGGS"],
        "xfiaggs": externals["FIAGGS"],
        "xcolexis": externals["COLEXIS"],
        "xtimauti": externals["TIMAUTI"],
        "xcriauti": externals["CRIAUTI"],
        "xacriauti": externals["ACRIAUTI"],
        "xbcriauti": externals["BCRIAUTI"],
        "xtexauti": externals["TEXAUTI"],
        "xcexvt": externals["CEXVT"],
        "x0depg": externals["O0DEPG"],
        "x1depg": externals["O1DEPG"],
        "xex1depg": externals["EX1DEPG"],
        "xhon": externals["HON"],
        "xalpha3": externals["ALPHA3"],
        "xex0depg": externals["EX0DEPG"],
        "xbeta3": externals["BETA3"],
        "x0deps": externals["O0DEPS"],
        "x1deps": externals["O1DEPS"],
        "xex1deps": externals["EX1DEPS"],
        "xex0deps": externals["EX0DEPS"],
    }

    F2Py_Mapping = {
        "prhodref": "rhodref",
        "pt": "t",
        "pssi": "ssi",
        "prvt": "rvt",
        "prct": "rct",
        "prit": "rit",
        "prst": "rst",
        "prgt": "rgt",
        "plbdas": "lbdas",
        "plbdag": "lbdag",
        "pai": "ai",
        "pcj": "cj",
        "phli_hcf": "hli_hcf",
        "phli_hri": "hli_hri",
        "prchoni": "rc_honi_tnd",
        "prvdeps": "rv_deps_tnd",
        "priaggs": "ri_aggs_tnd",
        "priauts": "ri_auts_tnd",
        "prvdepg": "rv_depg_tnd",
    }

    Py2F_Mapping = dict(map(reversed, F2Py_Mapping.items()))

    fortran_FloatFieldsIJK = {
        Py2F_Mapping[name]: field.ravel() for name, field in FloatFieldsIJK.items()
    }

    result = fortran_stencil(
        ldsoft=ldsoft,
        ldcompute=ldcompute.ravel(),
        **fortran_FloatFieldsIJK,
        **fortran_externals,
        **fortran_packed_dims,
    )

    prchoni_out = result[0]
    prvdeps_out = result[1]
    priaggs_out = result[2]
    priauts_out = result[3]
    prvdepg_out = result[4]

    logging.info(f"Machine precision {np.finfo(float).eps}")

    logging.info(f"Mean rc_honi_tnd    {GT4Py_FloatFieldsIJK['rc_honi_tnd'].mean()}")
    logging.info(f"Mean rcautr_out     {prchoni_out.mean()}")

    logging.info(f"Mean rv_deps_tnd     {GT4Py_FloatFieldsIJK['rv_deps_tnd'].mean()}")
    logging.info(f"Mean rcautr_out      {prvdeps_out.mean()}")

    logging.info(f"Mean ri_aggs_tnd    {GT4Py_FloatFieldsIJK['ro_aggs_tnd'].mean()}")
    logging.info(f"Mean rcautr_out      {priaggs_out.mean()}")

    logging.info(f"Mean ri_auts_tnd    {GT4Py_FloatFieldsIJK['ri_auts_tnd'].mean()}")
    logging.info(f"Mean priauts_out     {priauts_out.mean()}")

    logging.info(f"Mean rv_depg_tnd    {GT4Py_FloatFieldsIJK['rv_depg_tnd'].mean()}")
    logging.info(f"Mean rcautr_out      {prvdepg_out.mean()}")

    assert_allclose(prchoni_out, rc_honi_tnd.ravel(), 10e-6)
    assert_allclose(prvdeps_out, rv_deps_tnd.ravel(), 10e-6)
    assert_allclose(priaggs_out, ri_aggs_tnd.ravel(), 10e-6)
    assert_allclose(priauts_out, ri_auts_tnd.ravel(), 10e-6)
    assert_allclose(prvdepg_out, rv_depg_tnd.ravel(), 10e-6)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_warm(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    # Setting backend and precision
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

    ice4_warm_gt4py = compile_stencil("ice4_warm", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_warm.F90", "mode_ice4_warm", "ice4_warm"
    )

    logging.info(f"SUBG_RR_EVAP {externals['SUBG_RR_EVAP']}")

    ldcompute = np.array(
        np.random.rand(*grid.shape),
        dtype=bool,
        order="F",
    )

    FloatFieldsIJK_Names = [
        "rhodref",
        "t",
        "pres",
        "tht",
        "lbdar",
        "lbdar_rf",
        "ka",
        "dv",
        "cj",
        "hlc_hcf",
        "hlc_hrc",
        "cf",
        "rf",
        "rvt",
        "rct",
        "rrt",
        "rcautr",
        "rcaccr",
        "rrevav"
    ]

    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*grid.shape),
            dtype=(c_float if gt4py_config.dtypes.float == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_Names
    }

    GT4Py_FloatFieldsIJK = {
        name: gt4py.storage.zeros(
            shape=grid.shape,
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        for name in FloatFieldsIJK_Names
    }

    # Allocate
    for name in GT4Py_FloatFieldsIJK.keys():
        GT4Py_FloatFieldsIJK[name] = FloatFieldsIJK[name]


    ldcompute_gt4py = from_array(ldcompute, dtype=bool, backend=gt4py_config.backend)
    ldsoft = False

    ice4_warm_gt4py(
        **GT4Py_FloatFieldsIJK,
        ldcompute=ldcompute_gt4py,  # boolean field for microphysics computation
        ldsoft=ldsoft,
    )

    fortran_externals = {
        "xalpw": externals["ALPW"],
        "xbetaw": externals["BETAW"],
        "xgamw": externals["GAMW"],
        "xepsilo": externals["EPSILO"],
        "xlvtt": externals["LVTT"],
        "xcpv": externals["CPV"],
        "xcl": externals["CL"],
        "xtt": externals["TT"],
        "xrv": externals["RV"],
        "xcpd": externals["CPD"],
        "xtimautc": externals["TIMAUTC"],
        "xcriautc": externals["CRIAUTC"],
        "xfcaccr": externals["FCACCR"],
        "xexcaccr": externals["EXCACCR"],
        "x0evar": externals["O0EVAR"],
        "x1evar": externals["O1EVAR"],
        "xex0evar": externals["EX0EVAR"],
        "xex1evar": externals["EX1EVAR"],
        "c_rtmin": externals["C_RTMIN"],
        "r_rtmin": externals["R_RTMIN"],
        "xcexvt": externals["CEXVT"],
    }

    f2py_mapping =  {
        "prhodref":"rhodref",
        "pt":"t",
        "ppres":"pres",
        "ptht":"tht",
        "plbdar":"lbdar",
        "plbdar_rf":"lbdar_rf",
        "pka":"ka",
        "pdv":"dv",
        "pcj":"cj",
        "phlc_hcf":"hlc_hcf",
        "phlc_hrc":"hlc_hrc",
        "pcf":"cf",
        "prf":"rf",
        "prvt":"rvt",
        "prct":"rct",
        "prrt":"rrt",
        "prcautr":"rcautr",
        "prcaccr":"rcaccr",
        "prrevav":"rrevav",
    }

    fortran_FloatFieldsIJK = {
        name: FloatFieldsIJK[value].ravel()
        for name, value in f2py_mapping.items()
    }

    result = fortran_stencil(
        ldsoft=ldsoft,
        ldcompute=ldcompute,
        hsubg_rr_evap="none",
        **fortran_FloatFieldsIJK,
        **fortran_externals,
        **fortran_packed_dims,
    )

    rcautr_out = result[0]
    rcaccr_out = result[1]
    rrevav_out = result[2]

    logging.info(f"Machine precision {np.finfo(float).eps}")

    logging.info(f"Mean rcautr          {GT4Py_FloatFieldsIJK['rcautr'].mean()}")
    logging.info(f"Mean rcautr_out      {rcautr_out.mean()}")

    logging.info(f"Mean rcaccr          {GT4Py_FloatFieldsIJK['rcautr'].mean()}")
    logging.info(f"Mean rcaccr_out      {rcaccr_out.mean()}")

    logging.info(f"Mean rrevav          {GT4Py_FloatFieldsIJK['rcautr'].mean()}")
    logging.info(f"Mean rrevav_out      {rrevav_out.mean()}")

    assert_allclose(rcautr_out, GT4Py_FloatFieldsIJK['rcautr'].ravel(), rtol=1e-5)
    assert_allclose(rcaccr_out, GT4Py_FloatFieldsIJK['rcautr'].ravel(), rtol=1e-6)
    assert_allclose(rrevav_out, GT4Py_FloatFieldsIJK['rcautr'].ravel(), rtol=1e-6)

