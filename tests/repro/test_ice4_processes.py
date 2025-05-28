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

    ldcompute_gt4py = from_array(
        ldcompute,
        dtype=np.bool_,
        backend=gt4py_config.backend,
    )

    tht_gt4py = from_array(
        FloatFieldsIJK["tht"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    pabst_gt4py = from_array(
        FloatFieldsIJK["pabst"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rhodref_gt4py = from_array(
        FloatFieldsIJK["rhodref"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    exn_gt4py = from_array(
        FloatFieldsIJK["exn"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    lsfact_gt4py = from_array(
        FloatFieldsIJK["lsfact"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    t_gt4py = from_array(
        FloatFieldsIJK["t"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rvt_gt4py = from_array(
        FloatFieldsIJK["rvt"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    cit_gt4py = from_array(
        FloatFieldsIJK["cit"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rvheni_mr_gt4py = from_array(
        FloatFieldsIJK["rvheni_mr"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    ssi_gt4py = from_array(
        FloatFieldsIJK["ssi"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )

    ice4_nucleation_gt4py(
        ldcompute=ldcompute_gt4py,
        tht=tht_gt4py,
        pabst=pabst_gt4py,
        rhodref=rhodref_gt4py,
        exn=exn_gt4py,
        lsfact=lsfact_gt4py,
        t=t_gt4py,
        rvt=rvt_gt4py,
        cit=cit_gt4py,
        rvheni_mr=rvheni_mr_gt4py,
        ssi=ssi_gt4py,
        domain=grid.shape,
        origin=origin,
    )

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
    logging.info(f"Mean cit_gt4py   {cit_gt4py.mean()}")

    logging.info(f"Mean rvheni_mr_out     {rvheni_mr_out.mean()}")
    logging.info(f"Mean rvheni_mr_gt4py   {rvheni_mr_gt4py.mean()}")

    assert_allclose(cit_out, cit_gt4py.ravel(), 10e-6)
    assert_allclose(rvheni_mr_out, rvheni_mr_gt4py.ravel(), 10e-6)


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

    result = fortran_stencil(
        ldcompute=ldcompute.ravel(),
        **Fortran_FloatFieldsIJK,
        **fortran_packed_dims,
        **fortran_externals,
    )

    rimltc_mr_out = result[0]

    logging.info(f"Machine precision {np.finfo(float).eps}")
    assert_allclose(rimltc_mr_out, rimltc_mr_gt4py.ravel(), rtol=10e-6)


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
            dtype=c_float,
            order="F",
        )
        for name in FloatFieldsIJK_Names
    }

    ldcompute_gt4py = from_array(
        ldcompute,
        dtype=bool,
        backend=gt4py_config.backend,
    )

    rhodref_gt4py = from_array(
        FloatFieldsIJK["rhodref"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    t_gt4py = from_array(
        FloatFieldsIJK["t"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    ssi_gt4py = from_array(
        FloatFieldsIJK["ssi"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rvt_gt4py = from_array(
        FloatFieldsIJK["rvt"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rct_gt4py = from_array(
        FloatFieldsIJK["rct"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rit_gt4py = from_array(
        FloatFieldsIJK["rit"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rst_gt4py = from_array(
        FloatFieldsIJK["rst"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rgt_gt4py = from_array(
        FloatFieldsIJK["rgt"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    lbdas_gt4py = from_array(
        FloatFieldsIJK["lbdas"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    lbdag_gt4py = from_array(
        FloatFieldsIJK["lbdag"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    ai_gt4py = from_array(
        FloatFieldsIJK["ai"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    cj_gt4py = from_array(
        FloatFieldsIJK["cj"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    hli_hcf_gt4py = from_array(
        FloatFieldsIJK["hli_hcf"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    hli_hri_gt4py = from_array(
        FloatFieldsIJK["hli_hri"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rc_honi_tnd_gt4py = from_array(
        FloatFieldsIJK["rc_honi_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rv_deps_tnd_gt4py = from_array(
        FloatFieldsIJK["rv_deps_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    ri_aggs_tnd_gt4py = from_array(
        FloatFieldsIJK["ri_aggs_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    ri_auts_tnd_gt4py = from_array(
        FloatFieldsIJK["ri_auts_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rv_depg_tnd_gt4py = from_array(
        FloatFieldsIJK["rv_depg_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )

    ldsoft = True

    ice4_slow_gt4py(
        ldcompute=ldcompute_gt4py,
        rhodref=rhodref_gt4py,
        t=t_gt4py,
        ssi=ssi_gt4py,
        rvt=rvt_gt4py,
        rct=rct_gt4py,
        rit=rit_gt4py,
        rst=rst_gt4py,
        rgt=rgt_gt4py,
        lbdas=lbdas_gt4py,
        lbdag=lbdag_gt4py,
        ai=ai_gt4py,
        cj=cj_gt4py,
        hli_hcf=hli_hcf_gt4py,
        hli_hri=hli_hri_gt4py,
        rc_honi_tnd=rc_honi_tnd_gt4py,
        rv_deps_tnd=rv_deps_tnd_gt4py,
        ri_aggs_tnd=ri_aggs_tnd_gt4py,
        ri_auts_tnd=ri_auts_tnd_gt4py,
        rv_depg_tnd=rv_depg_tnd_gt4py,
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

    logging.info(f"Mean rcautr_gt4py    {rc_honi_tnd_gt4py.mean()}")
    logging.info(f"Mean rcautr_out      {prchoni_out.mean()}")
    logging.info(
        f"Max abs err rcautr   {max(abs(rc_honi_tnd_gt4py.ravel() - prchoni_out) / abs(prchoni_out))}"
    )

    logging.info(f"Mean rcautr_gt4py    {rv_deps_tnd_gt4py.mean()}")
    logging.info(f"Mean rcautr_out      {prvdeps_out.mean()}")
    logging.info(
        f"Max abs err rcautr   {max(abs(rv_deps_tnd_gt4py.ravel() - prvdeps_out) / abs(prvdeps_out))}"
    )

    logging.info(f"Mean rcautr_gt4py    {ri_aggs_tnd_gt4py.mean()}")
    logging.info(f"Mean rcautr_out      {priaggs_out.mean()}")
    logging.info(
        f"Max abs err rcautr   {max(abs(ri_aggs_tnd_gt4py.ravel() - priaggs_out) / abs(priaggs_out))}"
    )

    logging.info(f"Mean rcautr_gt4py    {ri_auts_tnd_gt4py.mean()}")
    logging.info(f"Mean rcautr_out      {priauts_out.mean()}")
    logging.info(
        f"Max abs err rcautr   {max(abs(ri_auts_tnd_gt4py.ravel() - priauts_out) / abs(priauts_out))}"
    )

    logging.info(f"Mean rcautr_gt4py    {rv_depg_tnd_gt4py.mean()}")
    logging.info(f"Mean rcautr_out      {prvdepg_out.mean()}")
    logging.info(
        f"Max abs err rcautr   {max(abs(rv_depg_tnd_gt4py.ravel() - prvdepg_out) / abs(prvdepg_out))}"
    )

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

    FloatFieldsIJK_names = [
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
            dtype=c_float,
            order="F",
        )
        for name in FloatFieldsIJK_names
    }


    ldcompute_gt4py = from_array(ldcompute, dtype=bool, backend=gt4py_config.backend)
    rhodref_gt4py = from_array(FloatFieldsIJK["rhodref"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    t_gt4py = from_array(FloatFieldsIJK["t"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    pres_gt4py = from_array(FloatFieldsIJK["pres"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    tht_gt4py = from_array(FloatFieldsIJK["tht"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    lbdar_gt4py = from_array(FloatFieldsIJK["lbdar"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    lbdar_rf_gt4py = from_array(FloatFieldsIJK["lbdar_rf"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    ka_gt4py = from_array(FloatFieldsIJK["ka"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    dv_gt4py = from_array(FloatFieldsIJK["dv"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    cj_gt4py = from_array(FloatFieldsIJK["cj"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    hlc_hcf_gt4py = from_array(FloatFieldsIJK["hlc_hcf"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    hlc_hrc_gt4py = from_array(FloatFieldsIJK["hlc_hrc"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    cf_gt4py = from_array(FloatFieldsIJK["cf"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    rf_gt4py = from_array(FloatFieldsIJK["rf"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    rvt_gt4py = from_array(FloatFieldsIJK["rvt"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    rct_gt4py = from_array(FloatFieldsIJK["rct"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    rrt_gt4py = from_array(FloatFieldsIJK["rrt"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    rcautr_gt4py = from_array(FloatFieldsIJK["rcautr"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    rcaccr_gt4py = from_array(FloatFieldsIJK["rcaccr"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)
    rrevav_gt4py = from_array(FloatFieldsIJK["rrevav"], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend)

    ldsoft = False

    ice4_warm_gt4py(
        ldcompute=ldcompute_gt4py,  # boolean field for microphysics computation
        rhodref=rhodref_gt4py,
        t=t_gt4py,  # temperature
        pres=pres_gt4py,
        tht=tht_gt4py,
        lbdar=lbdar_gt4py,  # slope parameter for the rain drop distribution
        lbdar_rf=lbdar_rf_gt4py,  # slope parameter for the rain fraction part
        ka=ka_gt4py,  # thermal conductivity of the air
        dv=dv_gt4py,  # diffusivity of water vapour
        cj=cj_gt4py,  # function to compute the ventilation coefficient
        hlc_hcf=hlc_hcf_gt4py,  # High Cloud Fraction in grid
        hlc_hrc=hlc_hrc_gt4py,  # LWC that is high in grid
        cf=cf_gt4py,  # cloud fraction
        rf=rf_gt4py,  # rain fraction
        rvt=rvt_gt4py,  # water vapour mixing ratio at t
        rct=rct_gt4py,  # cloud water mixing ratio at t
        rrt=rrt_gt4py,  # rain water mixing ratio at t
        rcautr=rcautr_gt4py,  # autoconversion of rc for rr production
        rcaccr=rcaccr_gt4py,  # accretion of r_c for r_r production
        rrevav=rrevav_gt4py,  # evaporation of rr
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

    logging.info(f"Mean rcautr_gt4py    {rcautr_gt4py.mean()}")
    logging.info(f"Mean rcautr_out      {rcautr_out.mean()}")
    logging.info(
        f"Max abs err rcautr   {max(abs(rcautr_gt4py.ravel() - rcautr_out) / abs(rcautr_out))}"
    )

    logging.info(f"Mean rcaccr_gt4py    {rcaccr_gt4py.mean()}")
    logging.info(f"Mean rcaccr_out      {rcaccr_out.mean()}")
    logging.info(
        f"Max abs err rcaccr   {max(abs(rcaccr_gt4py.ravel() - rcaccr_out) / abs(rcaccr_out))}"
    )

    logging.info(f"Mean rrevav_gt4py    {rrevav_gt4py.mean()}")
    logging.info(f"Mean rrevav_out      {rrevav_out.mean()}")
    logging.info(
        f"Max abs err rrevav   {max(abs(rrevav_gt4py.ravel() - rrevav_out) / abs(rrevav_out))}"
    )

    assert_allclose(rcautr_out, rcautr_gt4py.ravel(), rtol=1e-5)
    assert_allclose(rcaccr_out, rcaccr_gt4py.ravel(), rtol=1e-6)
    assert_allclose(rrevav_out, rrevav_gt4py.ravel(), rtol=1e-6)
