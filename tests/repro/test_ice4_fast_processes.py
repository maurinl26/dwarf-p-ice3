import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array
from numpy.testing import assert_allclose

from ice3.phyex_common.xker_raccs import KER_RACCS, KER_RACCSS, KER_SACCRG
from ice3.utils.allocate_random_fields import allocate_random_fields
from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import sp_dtypes, dp_dtypes


@pytest.mark.parametrize("ldsoft", [False])
@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("debug", marks=pytest.mark.debug),
        pytest.param("numpy", marks=pytest.mark.numpy),
        pytest.param("gt:cpu_ifirst", marks=pytest.mark.cpu),
        pytest.param("gt:gpu", marks=pytest.mark.gpu),
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
        "riaggs",
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
        field_names, dtypes, backend, domain
    )
    ldcompute = np.ones(domain, dtype=dtypes["bool"], order="F")
    ldcompute_gt4py = from_array(ldcompute, dtype=dtypes["bool"], backend=backend)
    ice4_fast_rs(
        ldsoft=ldsoft,
        ldcompute=ldcompute_gt4py,
        **gt4py_buffers,
        domain=domain,
        origin=origin,
    )
    # Fortran externals mapping
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

    (
        riaggs,
        rcrimss,
        rcrimcg,
        rcrimsg,
        rraccss,
        rsaccrg,
        rsaccsg,
        rs_mltg_tnd,
        rc_mltsr_tnd,
        rst
    ) = fortran_stencil(
        ldsoft=ldsoft,
        ldcompute=ldcompute,
        **fortran_FloatFieldsIJK,
        **fortran_packed_dims,
        **fortran_externals,
        **fortran_lookup_tables,
    )

    # Output assertions (example for a few fields)
    assert_allclose(riaggs, gt4py_buffers["riaggs"].ravel(), rtol=1e-6)
    assert_allclose(rcrimss, gt4py_buffers["rcrimss"].ravel(), rtol=1e-6)
    assert_allclose(rcrimsg, gt4py_buffers["rcrimsg"].ravel(), rtol=1e-6)
    assert_allclose(rcrimcg, gt4py_buffers["rsrimcg"].ravel(), rtol=1e-6)
    assert_allclose(rraccss, gt4py_buffers["rraccss"].ravel(), rtol=1e-6)
    assert_allclose(rsaccsg, gt4py_buffers["rraccsg"].ravel(), rtol=1e-6)
    assert_allclose(rsaccrg, gt4py_buffers["rsaccrg"].ravel(), rtol=1e-6)
    assert_allclose(rs_mltg_tnd, gt4py_buffers["rs_mltg_tnd"].ravel(), rtol=1e-6)
    assert_allclose(rc_mltsr_tnd, gt4py_buffers["rc_mltsr_tnd"].ravel(), rtol=1e-6)
    assert_allclose(rst, gt4py_buffers["rst"].ravel(), rtol=1e-6)


@pytest.mark.parametrize("ldsoft", [False])
@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("debug", marks=pytest.mark.debug),
        pytest.param("numpy", marks=pytest.mark.numpy),
        pytest.param("gt:cpu_ifirst", marks=pytest.mark.cpu),
        pytest.param("gt:gpu", marks=pytest.mark.gpu),
    ],
)
def test_ice4_fast_rg(
    externals, fortran_packed_dims, dtypes, backend, domain, origin, ldsoft
):
    from ice3.stencils.ice4_fast_rg import ice4_fast_rg

    ice4_fast_rg = stencil(
        backend,
        definition=ice4_fast_rg,
        name="ice4_fast_rg",
        externals=externals,
        dtypes=dtypes,
    )

    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_fast_rg.F90", "mode_ice4_fast_rg", "ice4_fast_rg"
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
        "riaggs",
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
        field_names, dtypes, backend, domain
    )

    ldcompute = np.ones(domain, dtype=dtypes["bool"], order="F")
    ldcompute_gt4py = from_array(ldcompute, dtype=dtypes["bool"], backend=backend)

    # Additional fields specific to ice4_fast_rg
    index_floor = np.ones(domain, dtype=dtypes["int"], order="F")
    index_floor_r = np.ones(domain, dtype=dtypes["int"], order="F")
    index_floor_s = np.ones(domain, dtype=dtypes["int"], order="F")
    
    index_floor_gt4py = from_array(index_floor, dtype=dtypes["int"], backend=backend)
    index_floor_r_gt4py = from_array(index_floor_r, dtype=dtypes["int"], backend=backend)
    index_floor_s_gt4py = from_array(index_floor_s, dtype=dtypes["int"], backend=backend)

    gaminc_rim1_gt4py = from_array(externals["GAMINC_RIM1"], dtype=dtypes["float"], backend=backend)
    gaminc_rim2_gt4py = from_array(externals["GAMINC_RIM2"], dtype=dtypes["float"], backend=backend)
    gaminc_rim4_gt4py = from_array(externals["GAMINC_RIM4"], dtype=dtypes["float"], backend=backend)

    ice4_fast_rg(
        ldsoft=ldsoft,
        ldcompute=ldcompute_gt4py,
        **gt4py_buffers,
        gaminc_rim1=gaminc_rim1_gt4py,
        gaminc_rim2=gaminc_rim2_gt4py,
        gaminc_rim4=gaminc_rim4_gt4py,
        ker_raccs=KER_RACCS,
        ker_raccss=KER_RACCSS,
        ker_saccrg=KER_SACCRG,
        index_floor=index_floor_gt4py,
        index_floor_r=index_floor_r_gt4py,
        index_floor_s=index_floor_s_gt4py,
        domain=domain,
        origin=origin,
    )

    # Fortran externals mapping
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
        "xgaminc_rim1": externals["GAMINC_RIM1"],
        "xgaminc_rim2": externals["GAMINC_RIM2"],
        "xgaminc_rim4": externals["GAMINC_RIM4"],
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

    (
        priaggs_out,
        prcrimss_out,
        prcrimsg_out,
        prsrimcg_out,
        prraccss_out,
        prraccsg_out,
        prsaccrg_out,
        prsmltg_out,
        prcmltsr_out,
        prs_tend_out 
    ) = fortran_stencil(
        ldsoft=ldsoft,
        ldcompute=ldcompute,
        **fortran_FloatFieldsIJK,
        **fortran_packed_dims,
        **fortran_externals,
        **fortran_lookup_tables,
    )


    # assert_allclose(prcrimss_out, rcrimss_gt4py.ravel(), rtol=1e-6)
    assert_allclose(prcrimsg_out, gt4py_buffers[""].ravel(), atol=1e-8, rtol=1e-6)
    assert_allclose(prsrimcg_out, gt4py_buffers[""].ravel(), atol=1e-8, rtol=1e-6)
    assert_allclose(prraccss_out, gt4py_buffers[""].ravel(), atol=1e-8, rtol=1e-6)
    assert_allclose(prraccsg_out, gt4py_buffers[""].ravel(), atol=1e-8, rtol=1e-6)
    assert_allclose(prsaccrg_out, gt4py_buffers[""].ravel(), atol=1e-8, rtol=1e-6)
    assert_allclose(prsmltg_out, gt4py_buffers[""].ravel(), atol=1e-8, rtol=1e-6)
    assert_allclose(prcmltsr_out, gt4py_buffers[""].ravel(), atol=1e-8, rtol=1e-6)
    assert_allclose(prs_tend_out, gt4py_buffers[""].ravel(), atol=1e-8, rtol=1e-6)
