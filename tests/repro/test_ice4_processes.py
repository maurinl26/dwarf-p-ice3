from ctypes import c_float

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.next import domain
from gt4py.storage import from_array
from numpy.testing import assert_allclose

from ice3.utils.allocate_random_fields import allocate_random_fields
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
def test_ice4_nucleation(
    backend, externals, fortran_packed_dims, dtypes, grid, origin
):
    from ice3.stencils.ice4_nucleation import ice4_nucleation

    ice4_nucleation_gt4py = stencil(
        name="ice4_nucleation",
        definition=ice4_nucleation,
        dtypes=dtypes,
        externals=externals,
    )
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_nucleation.F90", "mode_ice4_nucleation", "ice4_nucleation"
    )

    ldcompute = np.array(np.random.rand(*grid.shape), dtype=np.bool_, order="F")
    field_names = [
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

    fields, gt4py_buffers = allocate_random_fields(field_names, gt4py_config, grid)
    ldcompute_gt4py = from_array(ldcompute, dtype=np.bool_, backend=backend)
    ice4_nucleation_gt4py(
        ldcompute=ldcompute_gt4py, **gt4py_buffers, domain=domain, origin=origin
    )

    result = fortran_stencil(
        ldcompute=ldcompute.ravel(),
        ptht=fields["tht"].ravel(),
        ppabst=fields["pabst"].ravel(),
        prhodref=fields["rhodref"].ravel(),
        pexn=fields["exn"].ravel(),
        plsfact=fields["lsfact"].ravel(),
        pt=fields["t"].ravel(),
        prvt=fields["rvt"].ravel(),
        pcit=fields["cit"].ravel(),
        prvheni_mr=fields["rvheni_mr"].ravel(),
        xtt=externals["TT"],
        v_rtmin=externals["V_RTMIN"],
        xalpw=externals["ALPW"],
        xbetaw=externals["BETAW"],
        xgamw=externals["GAMW"],
        xalpi=externals["ALPI"],
        xbetai=externals["BETAI"],
        xgami=externals["GAMI"],
        xepsilo=externals["EPSILO"],
        xnu10=externals["NU10"],
        xnu20=externals["NU20"],
        xalpha1=externals["ALPHA1"],
        xalpha2=externals["ALPHA2"],
        xbeta1=externals["BETA1"],
        xbeta2=externals["BETA2"],
        xmnu0=externals["MNU0"],
        lfeedbackt=externals["LFEEDBACKT"],
        **fortran_packed_dims,
    )
    cit_out, rvheni_mr_out = result[0], result[1]
    assert_allclose(cit_out, gt4py_buffers["cit"].ravel(), rtol=10e-6)
    assert_allclose(rvheni_mr_out, gt4py_buffers["rvheni_mr"].ravel(), rtol=10e-6)


@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
    ],
)
def test_ice4_rimltc(backend, domain, dtypes, externals, fortran_packed_dims, origin):
    from ice3.stencils.ice4_rimltc import ice4_rimltc

    ice4_rimltc_gt4py = stencil(
        backend,
        definition=ice4_rimltc,
        name="ice4_rimltc",
        dtypes=dtypes,
        externals=externals,
    )
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_rimltc.F90", "mode_ice4_rimltc", "ice4_rimltc"
    )

    ldcompute = np.array(np.random.rand(*grid.shape), dtype=np.bool_, order="F")
    field_names = ["t", "exn", "lvfact", "lsfact", "tht", "rit", "rimltc_mr"]
    fields, gt4py_buffers = allocate_random_fields(field_names, gt4py_config, grid)
    ldcompute_gt4py = from_array(ldcompute, dtype=np.int32, backend=backend)
    ice4_rimltc_gt4py(
        ldcompute=ldcompute_gt4py, **gt4py_buffers, domain=domain, origin=origin
    )
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
    fortran_FloatFieldsIJK = {
        name: fields[value].ravel() for name, value in f2py_mapping.items()
    }
    result = fortran_stencil(
        ldcompute=ldcompute.ravel(),
        **fortran_FloatFieldsIJK,
        **fortran_packed_dims,
        **fortran_externals,
    )
    rimltc_mr_out = result[0]
    assert_allclose(rimltc_mr_out, gt4py_buffers["rimltc_mr"].ravel(), rtol=10e-6)


@pytest.mark.parametrize("dtypes", [dp_dtypes, sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
    ],
)
def test_ice4_slow(externals, fortran_packed_dims, dtypes, backend, domain, origin):
    from ice3.stencils.ice4_slow import ice4_slow

    ice4_slow_gt4py = stencil(
        backend,
        definition=ice4_slow,
        name="ice4_slow",
        dtypes=dtypes,
        externals=externals,
    )
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_slow.F90", "mode_ice4_slow", "ice4_slow"
    )
    ldcompute = np.array(np.random.rand(*domain), dtype=bool, order="F")
    field_names = [
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
    fields, gt4py_buffers = allocate_random_fields(
        field_names, gt4py_config, grid, dtype=c_float
    )
    ldcompute_gt4py = from_array(ldcompute, dtype=np.bool_, backend=backend)
    ldsoft = True
    ice4_slow_gt4py(
        ldcompute=ldcompute_gt4py,
        ldsoft=ldsoft,
        **gt4py_buffers,
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
    f2py_mapping = {
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
    fortran_FloatFieldsIJK = {
        name: fields[value].ravel() for name, value in f2py_mapping.items()
    }
    (
        prchoni_out, prvdeps_out, priaggs_out, priauts_out, prvdepg_out
    ) = fortran_stencil(
        ldsoft=ldsoft,
        ldcompute=ldcompute.ravel(),
        **fortran_FloatFieldsIJK,
        **fortran_externals,
        **fortran_packed_dims,
    )

    assert_allclose(prchoni_out, gt4py_buffers["rc_honi_tnd"].ravel(), 10e-6)
    assert_allclose(prvdeps_out, gt4py_buffers["rv_deps_tnd"].ravel(), 10e-6)
    assert_allclose(priaggs_out, gt4py_buffers["ri_aggs_tnd"].ravel(), 10e-6)
    assert_allclose(priauts_out, gt4py_buffers["ri_auts_tnd"].ravel(), 10e-6)
    assert_allclose(prvdepg_out, gt4py_buffers["rv_depg_tnd"].ravel(), 10e-6)


@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
    ],
)
def test_ice4_warm(externals, fortran_packed_dims, dtypes, backend, grid, origin):
    from ice3.stencils.ice4_warm import ice4_warm

    ice4_warm_gt4py = stencil(
        backend,
        name="ice4_warm",
        definition=ice4_warm,
        dtypes=dtypes,
        externals=externals,
    )
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_warm.F90", "mode_ice4_warm", "ice4_warm"
    )
    ldcompute = np.array(np.random.rand(*domain), dtype=np.bool_, order="F")
    field_names = [
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
        "rrevav",
    ]
    fields, gt4py_buffers = allocate_random_fields(
        field_names, gt4py_config, grid, dtype=c_float
    )
    ldcompute_gt4py = from_array(ldcompute, dtype=bool, backend=backend)
    ldsoft = False
    ice4_warm_gt4py(ldcompute=ldcompute_gt4py, ldsoft=ldsoft, **gt4py_buffers)
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
    f2py_mapping = {
        "prhodref": "rhodref",
        "pt": "t",
        "ppres": "pres",
        "ptht": "tht",
        "plbdar": "lbdar",
        "plbdar_rf": "lbdar_rf",
        "pka": "ka",
        "pdv": "dv",
        "pcj": "cj",
        "phlc_hcf": "hlc_hcf",
        "phlc_hrc": "hlc_hrc",
        "pcf": "cf",
        "prf": "rf",
        "prvt": "rvt",
        "prct": "rct",
        "prrt": "rrt",
        "prcautr": "rcautr",
        "prcaccr": "rcaccr",
        "prrevav": "rrevav",
    }
    fortran_FloatFieldsIJK = {
        name: fields[value].ravel() for name, value in f2py_mapping.items()
    }
    result = fortran_stencil(
        ldsoft=ldsoft,
        ldcompute=ldcompute,
        hsubg_rr_evap="none",
        **fortran_FloatFieldsIJK,
        **fortran_externals,
        **fortran_packed_dims,
    )
    rcautr_out, rcaccr_out, rrevav_out = result[:3]
    assert_allclose(rcautr_out, gt4py_buffers["rcautr"].ravel(), rtol=1e-5)
    assert_allclose(rcaccr_out, gt4py_buffers["rcaccr"].ravel(), rtol=1e-6)
    assert_allclose(rrevav_out, gt4py_buffers["rrevav"].ravel(), rtol=1e-6)
