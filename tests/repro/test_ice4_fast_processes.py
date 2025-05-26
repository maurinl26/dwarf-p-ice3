import logging
from ctypes import c_double, c_float, c_int

import numpy as np
import pytest
from conftest import compile_fortran_stencil, get_backends
from gt4py.storage import from_array
from ifs_physics_common.framework.stencil import compile_stencil
from numpy.testing import assert_allclose

from ice3_gt4py.phyex_common.xker_raccs import KER_RACCS, KER_RACCSS, KER_SACCRG


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

    FloatFieldsIJK_Names = [
        "rhodref",
        "ai",
        "cj",
        "cit",
        "ssi",
        "rct",
        "rit",
        "rc_beri_tnd",
    ]

    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*grid.shape),
            dtype=(c_float if gt4py_config.dtypes.float == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_Names
    }

    ldcompute_gt4py = from_array(ldcompute, dtype=bool, backend=gt4py_config.backend)
    rhodref_gt4py = from_array(
        FloatFieldsIJK["rhodref"],
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
    cit_gt4py = from_array(
        FloatFieldsIJK["cit"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    ssi_gt4py = from_array(
        FloatFieldsIJK["ssi"],
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
    rc_beri_tnd_gt4py = from_array(
        FloatFieldsIJK["rc_beri_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )

    logging.info(f"IN mean rc_beri_tnd_gt4py {rc_beri_tnd_gt4py.mean()}")

    ice4_fast_ri(
        ldcompute=ldcompute_gt4py,
        rhodref=rhodref_gt4py,
        ai=ai_gt4py,
        cj=cj_gt4py,
        cit=cit_gt4py,
        ssi=ssi_gt4py,
        rct=rct_gt4py,
        rit=rit_gt4py,
        rc_beri_tnd=rc_beri_tnd_gt4py,
        ldsoft=ldsoft,
        domain=grid.shape,
        origin=origin,
    )

    fortran_externals = {
        "c_rtmin": externals["C_RTMIN"],
        "i_rtmin": externals["I_RTMIN"],
        "xlbexi": externals["LBEXI"],
        "xlbi": externals["LBI"],
        "x0depi": externals["O0DEPI"],
        "x2depi": externals["O2DEPI"],
        "xdi": externals["DI"],
    }

    F2Py_Mapping = {
        "prhodref": "rhodref",
        "pai": "ai",
        "pcj": "cj",
        "pcit": "cit",
        "pssi": "ssi",
        "prct": "rct",
        "prit": "rit",
        "prcberi": "rc_beri_tnd",
    }

    Py2F_Mapping = dict(map(reversed, F2Py_Mapping.items()))

    fortran_FloatFieldsIJK = {
        Py2F_Mapping[name]: field.ravel() for name, field in FloatFieldsIJK.items()
    }

    result = fortran_stencil(
        ldsoft=ldsoft,
        ldcompute=ldcompute,
        **fortran_FloatFieldsIJK,
        **fortran_externals,
        **fortran_packed_dims,
    )

    rcberi_out = result[0]

    logging.info(f"Mean rc_beri_tnd_gt4py   {rc_beri_tnd_gt4py.mean()}")
    logging.info(f"Mean rcberi_out          {rcberi_out.mean()}")
    logging.info(
        f"Max abs rtol             {max(abs(rc_beri_tnd_gt4py.ravel() - rcberi_out) / abs(rcberi_out))}"
    )

    assert_allclose(rc_beri_tnd_gt4py.ravel(), rcberi_out, 1e-5)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_fast_rs(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ldsoft = False

    # Setting backend and precision
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

    ice4_fast_rs = compile_stencil("ice4_fast_rs", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_fast_rs.F90", "mode_ice4_fast_rs", "ice4_fast_rs"
    )

    logging.info(f"Machine precision {np.finfo(np.float32).eps}")

    ldcompute = np.ones(
        grid.shape,
        dtype=bool,
        order="F",
    )

    FloatFieldsIJK_names = [
        "rhodref",
        "pres",
        "dv",
        "ka",
        "cj",
        "lbdar",
        "lbdas",
        "t",
        "rvt",
        "rct",
        "rrt",
        "rst",
        "rcrimss",
        "rcrimsg",
        "rsrimcg",
        "rraccss",
        "rraccsg",
        "rsaccrg",
        "rs_mltg_tnd",
        "rc_mltsr_tnd",
        "rs_rcrims_tnd",
        "rs_rcrimss_tnd",
        "rs_rsrimcg_tnd",
        "rs_rraccs_tnd",
        "rs_rraccss_tnd",
        "rs_rsaccrg_tnd",
        "rs_freez1_tnd",
        "rs_freez2_tnd",
    ]

    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*grid.shape),
            dtype=c_float,
            order="F",
        )
        for name in FloatFieldsIJK_names
    }

    index_floor = np.ones(grid.shape, dtype=c_int, order="F")
    index_floor_r = np.ones(grid.shape, dtype=c_int, order="F")
    index_floor_s = np.ones(grid.shape, dtype=c_int, order="F")

    gaminc_rim1 = externals["GAMINC_RIM1"]
    gaminc_rim2 = externals["GAMINC_RIM2"]
    gaminc_rim4 = externals["GAMINC_RIM4"]

    ldcompute_gt4py = from_array(ldcompute, dtype=bool, backend=gt4py_config.backend)
    rhodref_gt4py = from_array(
        FloatFieldsIJK["rhodref"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    pres_gt4py = from_array(
        FloatFieldsIJK["pres"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    dv_gt4py = from_array(
        FloatFieldsIJK["dv"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    ka_gt4py = from_array(
        FloatFieldsIJK["ka"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    cj_gt4py = from_array(
        FloatFieldsIJK["cj"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    lbdar_gt4py = from_array(
        FloatFieldsIJK["lbdar"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    lbdas_gt4py = from_array(
        FloatFieldsIJK["lbdas"],
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
    rct_gt4py = from_array(
        FloatFieldsIJK["rct"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rrt_gt4py = from_array(
        FloatFieldsIJK["rrt"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rst_gt4py = from_array(
        FloatFieldsIJK["rst"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rcrimss_gt4py = from_array(
        FloatFieldsIJK["rcrimss"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rcrimsg_gt4py = from_array(
        FloatFieldsIJK["rcrimsg"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rsrimcg_gt4py = from_array(
        FloatFieldsIJK["rsrimcg"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rraccss_gt4py = from_array(
        FloatFieldsIJK["rraccss"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rraccsg_gt4py = from_array(
        FloatFieldsIJK["rraccsg"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rsaccrg_gt4py = from_array(
        FloatFieldsIJK["rsaccrg"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_mltg_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_mltg_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rc_mltsr_tnd_gt4py = from_array(
        FloatFieldsIJK["rc_mltsr_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_rcrims_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rcrims_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_rcrimss_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rcrimss_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_rsrimcg_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rsaccrg_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_rraccs_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rraccs_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_rraccss_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rraccss_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_rsaccrg_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rsaccrg_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_freez1_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_freez1_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_freez2_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_freez2_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )

    index_floor_gt4py = from_array(
        index_floor, dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend
    )
    index_floor_r_gt4py = from_array(
        index_floor_r, dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend
    )
    index_floor_s_gt4py = from_array(
        index_floor_s, dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend
    )

    gaminc_rim1_gt4py = from_array(
        gaminc_rim1, dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend
    )
    gaminc_rim2_gt4py = from_array(
        gaminc_rim2, dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend
    )
    gaminc_rim4_gt4py = from_array(
        gaminc_rim4, dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend
    )

    ice4_fast_rs(
        ldsoft=ldsoft,
        ldcompute=ldcompute_gt4py,
        rhodref=rhodref_gt4py,
        pres=pres_gt4py,
        dv=dv_gt4py,
        ka=ka_gt4py,
        cj=cj_gt4py,
        lbdar=lbdar_gt4py,
        lbdas=lbdas_gt4py,
        t=t_gt4py,
        rvt=rvt_gt4py,
        rct=rct_gt4py,
        rrt=rrt_gt4py,
        rst=rst_gt4py,
        rcrimss=rcrimss_gt4py,
        rcrimsg=rcrimsg_gt4py,
        rsrimcg=rsrimcg_gt4py,
        rraccss=rraccss_gt4py,
        rraccsg=rraccsg_gt4py,
        rsaccrg=rsaccrg_gt4py,
        rs_mltg_tnd=rs_mltg_tnd_gt4py,
        rc_mltsr_tnd=rc_mltsr_tnd_gt4py,
        rs_rcrims_tnd=rs_rcrimss_tnd_gt4py,
        rs_rcrimss_tnd=rs_rcrimss_tnd_gt4py,
        rs_rsrimcg_tnd=rs_rsrimcg_tnd_gt4py,
        rs_rraccs_tnd=rs_rraccs_tnd_gt4py,
        rs_rraccss_tnd=rs_rraccss_tnd_gt4py,
        rs_rsaccrg_tnd=rs_rsaccrg_tnd_gt4py,
        rs_freez1_tnd=rs_freez1_tnd_gt4py,
        rs_freez2_tnd=rs_freez2_tnd_gt4py,
        gaminc_rim1=gaminc_rim1_gt4py,
        gaminc_rim2=gaminc_rim2_gt4py,
        gaminc_rim4=gaminc_rim4_gt4py,
        ker_raccs=KER_RACCS,
        ker_raccss=KER_RACCSS,
        ker_saccrg=KER_SACCRG,
        index_floor=index_floor_gt4py,
        index_floor_r=index_floor_r_gt4py,
        index_floor_s=index_floor_s_gt4py,
        domain=grid.shape,
        origin=origin,
    )

    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_fast_rs.F90", "mode_ice4_fast_rs", "ice4_fast_rs"
    )

    externals_mapping = {
        "ngaminc": "NGAMINC",
        "nacclbdas": "NACCLBDAS",
        "nacclbdar": "NACCLBDAR",
        "levlimit": "LEVLIMIT",
        "lpack_interp": "LPACK_INTERP",
        "csnowriming": "CSNOWRIMING",
        "xcrimss": "CRIMSS",
        "xexcrimss": "EXCRIMSS",
        "xcrimsg": "CRIMSG",
        "xexcrimsg": "EXCRIMSG",
        "xexsrimcg2": "EXSRIMCG2",
        "xfraccss": "FRACCSS",
        "s_rtmin": "S_RTMIN",
        "c_rtmin": "C_RTMIN",
        "r_rtmin": "R_RTMIN",
        "xepsilo": "EPSILO",
        "xalpi": "ALPI",
        "xbetai": "BETAI",
        "xgami": "GAMI",
        "xtt": "TT",
        "xlvtt": "LVTT",
        "xcpv": "CPV",
        "xci": "CI",
        "xcl": "CL",
        "xlmtt": "LMTT",
        "xestt": "ESTT",
        "xrv": "RV",
        "x0deps": "O0DEPS",
        "x1deps": "O1DEPS",
        "xex0deps": "EX0DEPS",
        "xex1deps": "EX1DEPS",
        "xlbraccs1": "LBRACCS1",
        "xlbraccs2": "LBRACCS2",
        "xlbraccs3": "LBRACCS3",
        "xcxs": "CXS",
        "xsrimcg2": "SRIMCG2",
        "xsrimcg3": "SRIMCG3",
        "xbs": "BS",
        "xlbsaccr1": "LBSACCR1",
        "xlbsaccr2": "LBSACCR2",
        "xlbsaccr3": "LBSACCR3",
        "xfsaccrg": "FSACCRG",
        "xsrimcg": "SRIMCG",
        "xexsrimcg": "EXSRIMCG",
        "xcexvt": "CVEXT",
        "xalpw": "ALPW",
        "xbetaw": "BETAW",
        "xgamw": "GAMW",
        "xfscvmg": "FSCVMG",
    }

    fortran_externals = {
        fkey: externals[pykey] for fkey, pykey in externals_mapping.items()
    }

    fortran_lookup_tables = {
        "xker_raccss": KER_RACCSS,
        "xker_raccs": KER_RACCS,
        "xker_saccrg": KER_SACCRG,
        "xgaminc_rim1": gaminc_rim1_gt4py,
        "xgaminc_rim2": gaminc_rim2_gt4py,
        "xgaminc_rim4": gaminc_rim4_gt4py,
        "xrimintp1": externals["RIMINTP1"],
        "xrimintp2": externals["RIMINTP2"],
        "xaccintp1s": externals["ACCINTP1S"],
        "xaccintp2s": externals["ACCINTP2S"],
        "xaccintp1r": externals["ACCINTP1R"],
        "xaccintp2r": externals["ACCINTP2R"],
    }

    f2py_mapping = {
        "prhodref": "rhodref",
        "ppres": "pres",
        "pdv": "dv",
        "pka":"ka",
        "pcj": "cj",
        "plbdar": "lbdar",
        "plbdas": "lbdas",
        "pt": "t",
        "prvt": "rvt",
        "prct": "rct",
        "prrt": "rrt",
        "prst": "rst",
        "priaggs": "riaggs",
        "prcrimss": "rcrimss",
        "prcrimsg": "rcrimsg",
        "prsrimcg": "rsrimcg",
        "prraccss": "rraccss",
        "prraccsg": "rraccsg",
        "prsaccrg": "rsaccrg",
        "prsmltg": "rsmltg",
        "prcmltsr": "rcmltsr",
        "rs_rcrims_tend": "rs_rcrims_tnd",
        "rs_rcrimss_tend": "rs_rcrimss_tnd",
        "rs_rsrimcg_tend": "rs_rsrimcg_tnd",
        "rs_rraccs_tend": "rs_rraccs_tnd",
        "rs_rraccss_tend": "rs_rraccss_tnd",
        "rs_rsaccrg_tend": "rs_rsaccrg_tnd",
        "rs_freez1_tend": "rs_freez1_tnd",
        "rs_freez2_tend": "rs_freez2_tnd",
    }

    fortran_FloatFieldsIJK = {
        fname: FloatFieldsIJK[pyname]
        for fname, pyname in f2py_mapping.items()
    }

    result = fortran_stencil(
        ldsoft=ldsoft,
        ldcompute=ldcompute,
        **fortran_FloatFieldsIJK,
        **fortran_packed_dims,
        **fortran_externals,
        **fortran_lookup_tables,
    )

    priaggs_out = result[0]
    prcrimss_out = result[1]
    prcrimsg_out = result[2]
    prsrimcg_out = result[3]
    prraccss_out = result[4]
    prraccsg_out = result[5]
    prsaccrg_out = result[6]
    prsmltg_out = result[7]
    prcmltsr_out = result[8]
    prs_tend_out = result[9]

    assert_allclose(priaggs_out, riaggs_gt4py, rtol=1e-6)
    assert_allclose(prcrimss_out, rcrimss_gt4py, rtol=1e-6)
    assert_allclose(prcrimsg_out, rcrimsg_gt4py, rtol=1e-6)
    assert_allclose(prsrimcg_out, rsrimcg_gt4py, rtol=1e-6)
    assert_allclose(prraccss_out, rraccss_gt4py, rtol=1e-6)
    assert_allclose(prraccsg_out, rraccsg_gt4py, rtol=1e-6)
    assert_allclose(prsaccrg_out, rsaccrg_gt4py, rtol=1e-6)
    assert_allclose(prsmltg_out, rs_mltg_tnd_gt4py, rtol=1e-6)
    assert_allclose(prcmltsr_out, rc_mltsr_tnd_gt4py, rtol=1e-6)
    assert_allclose(prs_tend_out, rst, rtol=1e-6)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_fast_rg(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    # Setting backend and precision
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

    ice4_fast_rg = compile_stencil("ice4_fast_rg", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_fast_rg.F90", "mode_ice4_fast_rg", "ice4_fast_rg"
    )

    logging.info(f"Machine precision {np.finfo(np.float32).eps}")

    ldcompute = np.ones(
        grid.shape,
        dtype=bool,
        order="F",
    )

    FloatFieldsIJK_names = {
        "rhodref",
        "pres",
        "dv",
        "ka",
        "cj",
        "lbdar",
        "lbdas",
        "t",
        "rvt",
        "rct",
        "rrt",
        "rst",
        "rcrimss",
        "rcrimsg",
        "rsrimcg",
        "rraccss",
        "rraccsg",
        "rsaccrg",
        "rs_mltg_tnd",
        "rc_mltsr_tnd",
        "rs_rcrims_tnd",
        "rs_rcrimss_tnd",
        "rs_rsrimcg_tnd",
        "rs_rraccs_tnd",
        "rs_rraccss_tnd",
        "rs_rsaccrg_tnd",
        "rs_freez1_tnd",
        "rs_freez2_tnd",
    }

    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*grid.shape),
            dtype=(c_float if gt4py_config.dtypes.float == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_names
    }

    index_floor = np.ones(grid.shape, dtype=c_int, order="F")
    index_floor_r = np.ones(grid.shape, dtype=c_int, order="F")
    index_floor_s = np.ones(grid.shape, dtype=c_int, order="F")

    gaminc_rim1 = externals["GAMINC_RIM1"]
    gaminc_rim2 = externals["GAMINC_RIM2"]
    gaminc_rim4 = externals["GAMINC_RIM4"]

    ldsoft = False

    ldcompute_gt4py = from_array(ldcompute, dtype=bool, backend=gt4py_config.backend)
    rhodref_gt4py = from_array(
        FloatFieldsIJK["rhodref"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    pres_gt4py = from_array(
        FloatFieldsIJK["pres"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    dv_gt4py = from_array(
        FloatFieldsIJK["dv"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    ka_gt4py = from_array(
        FloatFieldsIJK["ka"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    cj_gt4py = from_array(
        FloatFieldsIJK["cj"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    lbdar_gt4py = from_array(
        FloatFieldsIJK["lbdar"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    lbdas_gt4py = from_array(
        FloatFieldsIJK["lbdas"],
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
    rct_gt4py = from_array(
        FloatFieldsIJK["rct"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rrt_gt4py = from_array(
        FloatFieldsIJK["rrt"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rst_gt4py = from_array(
        FloatFieldsIJK["rst"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rcrimss_gt4py = from_array(
        FloatFieldsIJK["rcrimss"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rcrimsg_gt4py = from_array(
        FloatFieldsIJK["rcrimsg"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rsrimcg_gt4py = from_array(
        FloatFieldsIJK["rsrimcg"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rraccss_gt4py = from_array(
        FloatFieldsIJK["rraccss"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rraccsg_gt4py = from_array(
        FloatFieldsIJK["rraccsg"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rsaccrg_gt4py = from_array(
        FloatFieldsIJK["rsaccrg"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_mltg_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_mltg_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rc_mltsr_tnd_gt4py = from_array(
        FloatFieldsIJK["rc_mltsr_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_rcrims_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rcrims_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_rcrimss_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rcrimss_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_rsrimcg_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rsaccrg_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_rraccs_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rraccs_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_rraccss_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rraccss_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_rsaccrg_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rsaccrg_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_freez1_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_freez1_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )
    rs_freez2_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_freez2_tnd"],
        dtype=gt4py_config.dtypes.float,
        backend=gt4py_config.backend,
    )

    index_floor_gt4py = from_array(
        index_floor, dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend
    )
    index_floor_r_gt4py = from_array(
        index_floor_r, dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend
    )
    index_floor_s_gt4py = from_array(
        index_floor_s, dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend
    )

    gaminc_rim1_gt4py = from_array(
        gaminc_rim1, dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend
    )
    gaminc_rim2_gt4py = from_array(
        gaminc_rim2, dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend
    )
    gaminc_rim4_gt4py = from_array(
        gaminc_rim4, dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend
    )

    ice4_fast_rg(
        ldsoft=ldsoft,
        ldcompute=ldcompute_gt4py,
        rhodref=rhodref_gt4py,
        pres=pres_gt4py,
        dv=dv_gt4py,
        ka=ka_gt4py,
        cj=cj_gt4py,
        lbdar=lbdar_gt4py,
        lbdas=lbdas_gt4py,
        t=t_gt4py,
        rvt=rvt_gt4py,
        rct=rct_gt4py,
        rrt=rrt_gt4py,
        rst=rst_gt4py,
        rcrimss=rcrimss_gt4py,
        rcrimsg=rcrimsg_gt4py,
        rsrimcg=rsrimcg_gt4py,
        rraccss=rraccss_gt4py,
        rraccsg=rraccsg_gt4py,
        rsaccrg=rsaccrg_gt4py,
        rs_mltg_tnd=rs_mltg_tnd_gt4py,
        rc_mltsr_tnd=rc_mltsr_tnd_gt4py,
        rs_rcrims_tnd=rs_rcrimss_tnd_gt4py,
        rs_rcrimss_tnd=rs_rcrimss_tnd_gt4py,
        rs_rsrimcg_tnd=rs_rsrimcg_tnd_gt4py,
        rs_rraccs_tnd=rs_rraccs_tnd_gt4py,
        rs_rraccss_tnd=rs_rraccss_tnd_gt4py,
        rs_rsaccrg_tnd=rs_rsaccrg_tnd_gt4py,
        rs_freez1_tnd=rs_freez1_tnd_gt4py,
        rs_freez2_tnd=rs_freez2_tnd_gt4py,
        gaminc_rim1=gaminc_rim1_gt4py,
        gaminc_rim2=gaminc_rim2_gt4py,
        gaminc_rim4=gaminc_rim4_gt4py,
        ker_raccs=KER_RACCS,
        ker_raccss=KER_RACCSS,
        ker_saccrg=KER_SACCRG,
        index_floor=index_floor_gt4py,
        index_floor_r=index_floor_r_gt4py,
        index_floor_s=index_floor_s_gt4py,
        domain=grid.shape,
        origin=origin,
    )

    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_fast_rg.F90", "mode_ice4_fast_rs", "ice4_fast_rs"
    )

    fortran_externals = {
        "ngaminc": externals["NGAMINC"],
        "nacclbdas": externals["NACCLBDAS"],
        "nacclbdar": externals["NACCLBDAR"],
        "levlimit": externals["LEVLIMIT"],
        "lpack_interp": externals["LPACK_INTERP"],
        "csnowriming": externals["CSNOWRIMING"],
        "xcrimss": externals["CRIMSS"],
        "xexcrimss": externals["EXCRIMSS"],
        "xcrimsg": externals["CRIMSG"],
        "xexcrimsg": externals["EXCRIMSG"],
        "xexsrimcg2": externals["EXSRIMCG2"],
        "xfraccss": externals["FRACCSS"],
        "s_rtmin": externals["S_RTMIN"],
        "c_rtmin": externals["C_RTMIN"],
        "r_rtmin": externals["R_RTMIN"],
        "xepsilo": externals["EPSILO"],
        "xalpi": externals["ALPI"],
        "xbetai": externals["BETAI"],
        "xgami": externals["GAMI"],
        "xtt": externals["TT"],
        "xlvtt": externals["LVTT"],
        "xcpv": externals["CPV"],
        "xci": externals["CI"],
        "xcl": externals["CL"],
        "xlmtt": externals["LMTT"],
        "xestt": externals["ESTT"],
        "xrv": externals["RV"],
        "x0deps": externals["O0DEPS"],
        "x1deps": externals["O1DEPS"],
        "xex0deps": externals["EX0DEPS"],
        "xex1deps": externals["EX1DEPS"],
        "xlbraccs1": externals["LBRACCS1"],
        "xlbraccs2": externals["LBRACCS2"],
        "xlbraccs3": externals["LBRACCS3"],
        "xcxs": externals["CXS"],
        "xsrimcg2": externals["SRIMCG2"],
        "xsrimcg3": externals["SRIMCG3"],
        "xbs": externals["BS"],
        "xlbsaccr1": externals["LBSACCR1"],
        "xlbsaccr2": externals["LBSACCR2"],
        "xlbsaccr3": externals["LBSACCR3"],
        "xfsaccrg": externals["FSACCRG"],
        "xsrimcg": externals["SRIMCG"],
        "xexsrimcg": externals["EXSRIMCG"],
        "xcexvt": externals["CVEXT"],
        "xalpw": externals["ALPW"],
        "xbetaw": externals["BETAW"],
        "xgamw": externals["GAMW"],
        "xfscvmg": externals["FSCVMG"],
    }

    fortran_lookup_tables = {
        "xker_raccss": KER_RACCSS,
        "xker_raccs": KER_RACCS,
        "xker_saccrg": KER_SACCRG,
        "xgaminc_rim1": gaminc_rim1_gt4py,
        "xgaminc_rim2": gaminc_rim2_gt4py,
        "xgaminc_rim4": gaminc_rim4_gt4py,
        "xrimintp1": externals["RIMINTP1"],
        "xrimintp2": externals["RIMINTP2"],
        "xaccintp1s": externals["ACCINTP1S"],
        "xaccintp2s": externals["ACCINTP2S"],
        "xaccintp1r": externals["ACCINTP1R"],
        "xaccintp2r": externals["ACCINTP2R"],
    }

    F2Py_Mapping = {
        "prhodref": "rhodref",
        "ppres": "pres",
        "pdv": "dv",
        "pka": "ka",
        "pcj": "cj",
        "plbdar": "lbdar",
        "plbdas": "lbdas",
        "pt": "t",
        "prvt": "rvt",
        "prct": "rct",
        "prrt": "rrt",
        "prst": "rst",
        "priaggs": "riaggs",
        "prcrimss": "rcrimss",
        "prcrimsg": "rcrimsg",
        "prsrimcg": "rsrimcg",
        "prraccss": "rraccss",
        "prraccsg": "rraccsg",
        "prsaccrg": "rsaccrg",
        "prsmltg": "rs_mltg_tnd",
        "prcmltsr": "rc_mltsr_tnd",
        "prs_tend": "rst",
    }

    Py2F_Mapping = dict(map(reversed, F2Py_Mapping.items()))

    fortran_FloatFieldsIJK = {
        Py2F_Mapping[name]: field.ravel() for name, field in FloatFieldsIJK.items()
    }

    result = fortran_stencil(
        ldsoft=ldsoft,
        ldcompute=ldcompute,
        **fortran_FloatFieldsIJK,
        **fortran_packed_dims,
        **fortran_externals,
        **fortran_lookup_tables,
    )

    priaggs_out = result[0]
    prcrimss_out = result[1]
    prcrimsg_out = result[2]
    prsrimcg_out = result[3]
    prraccss_out = result[4]
    prraccsg_out = result[5]
    prsaccrg_out = result[6]
    prsmltg_out = result[7]
    prcmltsr_out = result[8]
    prs_tend_out = result[9]

    assert_allclose(priaggs_out, riaggs_gt4py.ravel(), rtol=1e-6)
    assert_allclose(prcrimss_out, rcrimss_gt4py.ravel(), rtol=1e-6)
    assert_allclose(prcrimsg_out, rcrimsg_gt4py.ravel(), rtol=1e-6)
    assert_allclose(prsrimcg_out, rsrimcg_gt4py.ravel(), rtol=1e-6)
    assert_allclose(prraccss_out, rraccss_gt4py.ravel(), rtol=1e-6)
    assert_allclose(prraccsg_out, rraccsg_gt4py.ravel(), rtol=1e-6)
    assert_allclose(prsaccrg_out, rsaccrg_gt4py.ravel(), rtol=1e-6)
    assert_allclose(prsmltg_out, rs_mltg_tnd_gt4py.ravel(), rtol=1e-6)
    assert_allclose(prcmltsr_out, rc_mltsr_tnd_gt4py.ravel(), rtol=1e-6)
    assert_allclose(prs_tend_out, rst.ravel(), rtol=1e-6)
