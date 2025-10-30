import logging
from ctypes import c_double, c_float, c_int

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array
from numpy.testing import assert_allclose

from ice3.phyex_common.xker_raccs import KER_RACCS, KER_RACCSS, KER_SACCRG
from ice3.utils.allocate_random_fields import allocate_random_fields
from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import CPU_BACKEND, DEBUG_BACKEND, GPU_BACKEND

from src.ice3.utils.env import sp_dtypes, dp_dtypes


@pytest.mark.parametrize("ldsoft", False)
@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
    ],
)
def test_ice4_fast_rs(
    externals, fortran_packed_dims, dtypes, backend, domain, origin, ldsoft
):
    from ice3.stencils.ice4_fast_rs import ice4_fast_rs

    ice4_fast_rs = stencil(
        backend,
        name="ice4_fast_rs",
        definition=ice4_fast_rs,
        dtypes=dtypes,
        externals=externals,
    )
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_fast_processes.F90", "mode_ice4_fast_processes", "ice4_fast_rs"
    )
    field_names = [
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
    fields, gt4py_buffers = allocate_random_fields(
        field_names, gt4py_config, grid, c_float
    )
    ldcompute = np.ones(domain, dtype=bool, order="F")
    ldcompute_gt4py = from_array(ldcompute, dtype=bool, backend=backend)
    ice4_fast_rs(
        ldsoft=ldsoft,
        ldcompute=ldcompute_gt4py,
        **gt4py_buffers,
        domain=domain,
        origin=origin,
    )
    # Fortran externals and mapping
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
    }
    f2py_mapping = {
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
        fname: fields[pyname].ravel()
        for fname, pyname in f2py_mapping.items()
        if pyname in fields
    }
    result = fortran_stencil(
        ldsoft=ldsoft,
        ldcompute=ldcompute,
        **fortran_FloatFieldsIJK,
        **fortran_packed_dims,
        **fortran_externals,
        **fortran_lookup_tables,
    )

    # Output assertions (example for a few fields)
    assert_allclose(result[0], gt4py_buffers["riaggs"].ravel(), rtol=1e-6)
    assert_allclose(result[1], gt4py_buffers["rcrimss"].ravel(), rtol=1e-6)
    assert_allclose(result[2], gt4py_buffers["rcrimsg"].ravel(), rtol=1e-6)
    assert_allclose(result[3], gt4py_buffers["rsrimcg"].ravel(), rtol=1e-6)
    assert_allclose(result[4], gt4py_buffers["rraccss"].ravel(), rtol=1e-6)
    assert_allclose(result[5], gt4py_buffers["rraccsg"].ravel(), rtol=1e-6)
    assert_allclose(result[6], gt4py_buffers["rsaccrg"].ravel(), rtol=1e-6)
    assert_allclose(result[7], gt4py_buffers["rs_mltg_tnd"].ravel(), rtol=1e-6)
    assert_allclose(result[8], gt4py_buffers["rc_mltsr_tnd"].ravel(), rtol=1e-6)
    assert_allclose(result[9], gt4py_buffers["rst"].ravel(), rtol=1e-6)


@pytest.mark.parametrize("ldsoft", False)
@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
    ],
)
def test_ice4_fast_rg(
    gt4py_config, externals, fortran_packed_dims, dtypes, backend, domain, origin
):
    # Setting backend and precision
    from ice3.stencils.ice4_fast_rg import ice4_fast_rg

    print("compile stencil")

    ice4_fast_rg = stencil(
        backend,
        definition=ice4_fast_rg,
        name="ice4_fast_rg",
        externals=externals,
        dtypes=dtypes,
    )

    print("compile fortran stencil")

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
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_names
    }

    index_floor = np.ones(domain, dtype=c_int, order="F")
    index_floor_r = np.ones(domain, dtype=c_int, order="F")
    index_floor_s = np.ones(domain, dtype=c_int, order="F")

    gaminc_rim1 = externals["GAMINC_RIM1"]
    gaminc_rim2 = externals["GAMINC_RIM2"]
    gaminc_rim4 = externals["GAMINC_RIM4"]

    ldsoft = False

    ldcompute_gt4py = from_array(ldcompute, dtype=bool, backend=backend)
    rhodref_gt4py = from_array(
        FloatFieldsIJK["rhodref"],
        dtype=dtypes["float"],
        backend=backend,
    )
    pres_gt4py = from_array(
        FloatFieldsIJK["pres"],
        dtype=dtypes["float"],
        backend=backend,
    )
    dv_gt4py = from_array(
        FloatFieldsIJK["dv"],
        dtype=dtypes["float"],
        backend=backend,
    )
    ka_gt4py = from_array(
        FloatFieldsIJK["ka"],
        dtype=dtypes["float"],
        backend=backend,
    )
    cj_gt4py = from_array(
        FloatFieldsIJK["cj"],
        dtype=dtypes["float"],
        backend=backend,
    )
    lbdar_gt4py = from_array(
        FloatFieldsIJK["lbdar"],
        dtype=dtypes["float"],
        backend=backend,
    )
    lbdas_gt4py = from_array(
        FloatFieldsIJK["lbdas"],
        dtype=dtypes["float"],
        backend=backend,
    )
    t_gt4py = from_array(
        FloatFieldsIJK["t"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rvt_gt4py = from_array(
        FloatFieldsIJK["rvt"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rct_gt4py = from_array(
        FloatFieldsIJK["rct"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rrt_gt4py = from_array(
        FloatFieldsIJK["rrt"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rst_gt4py = from_array(
        FloatFieldsIJK["rst"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rcrimss_gt4py = from_array(
        FloatFieldsIJK["rcrimss"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rcrimsg_gt4py = from_array(
        FloatFieldsIJK["rcrimsg"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rsrimcg_gt4py = from_array(
        FloatFieldsIJK["rsrimcg"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rraccss_gt4py = from_array(
        FloatFieldsIJK["rraccss"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rraccsg_gt4py = from_array(
        FloatFieldsIJK["rraccsg"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rsaccrg_gt4py = from_array(
        FloatFieldsIJK["rsaccrg"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rs_mltg_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_mltg_tnd"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rc_mltsr_tnd_gt4py = from_array(
        FloatFieldsIJK["rc_mltsr_tnd"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rs_rcrims_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rcrims_tnd"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rs_rcrimss_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rcrimss_tnd"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rs_rsrimcg_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rsaccrg_tnd"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rs_rraccs_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rraccs_tnd"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rs_rraccss_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rraccss_tnd"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rs_rsaccrg_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_rsaccrg_tnd"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rs_freez1_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_freez1_tnd"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rs_freez2_tnd_gt4py = from_array(
        FloatFieldsIJK["rs_freez2_tnd"],
        dtype=dtypes["float"],
        backend=backend,
    )

    index_floor_gt4py = from_array(
        index_floor, dtype=dtypes["float"], backend=backend
    )
    index_floor_r_gt4py = from_array(
        index_floor_r, dtype=dtypes["float"], backend=backend
    )
    index_floor_s_gt4py = from_array(
        index_floor_s, dtype=dtypes["float"], backend=backend
    )

    gaminc_rim1_gt4py = from_array(
        gaminc_rim1, dtype=dtypes["float"], backend=backend
    )
    gaminc_rim2_gt4py = from_array(
        gaminc_rim2, dtype=dtypes["float"], backend=backend
    )
    gaminc_rim4_gt4py = from_array(
        gaminc_rim4, dtype=dtypes["float"], backend=backend
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
