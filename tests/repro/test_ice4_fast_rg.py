from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.config import GT4PyConfig, DataTypes
from ice3_gt4py.phyex_common.phyex import Phyex
from gt4py.storage import from_array
import numpy as np
from numpy.testing import assert_allclose
from pathlib import Path
import fmodpy
import unittest
from ctypes import c_float, c_int
from ice3_gt4py.phyex_common.xker_raccs import KER_RACCS, KER_RACCSS, KER_SACCRG


import logging

from .env import BACKEND, REBUILD, VALIDATE_ARGS, SHAPE


class TestIce4FastRG(unittest.TestCase):
    def test_ice4_fast_rg(self):
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND,
            rebuild=REBUILD,
            validate_args=VALIDATE_ARGS,
            verbose=True,
            dtypes=DataTypes(bool=bool, float=np.float32, int=np.int32),
        )

        logging.info(f"Machine precision {np.finfo(np.float32).eps}")
        logging.info(f"Machine precision {np.finfo(np.float32).eps}")

        phyex_externals = Phyex("AROME").to_externals()
        ice4_fast_rg_gt4py = compile_stencil(
            "ice4_fast_rg", gt4py_config, phyex_externals
        )

        ldcompute = np.ones(
            (SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=bool,
            order="F",
        )
        rhodref = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        
        pres = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        dv = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        ka = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        cj = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        lbdar = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        lbdas = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        t = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rvt = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rct = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rrt = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rst = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rcrimss = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rcrimsg = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rsrimcg = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rraccss = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rraccsg = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rsaccrg = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rs_mltg_tnd = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rc_mltsr_tnd = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rs_rcrims_tnd = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rs_rcrimss_tnd = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rs_rsrimcg_tnd = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rs_rraccs_tnd = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rs_rraccss_tnd = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rs_rsaccrg_tnd = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rs_freez1_tnd = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        rs_freez2_tnd = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
            dtype=c_float,
            order="F",
        )
        index_floor = np.ones((SHAPE[0], SHAPE[1], SHAPE[2]), dtype=c_int, order="F")
        index_floor_r=np.ones((SHAPE[0], SHAPE[1], SHAPE[2]), dtype=c_int, order="F")
        index_floor_s=np.ones((SHAPE[0], SHAPE[1], SHAPE[2]), dtype=c_int, order="F")
        
        gaminc_rim1 = phyex_externals["GAMINC_RIM1"]
        gaminc_rim2 = phyex_externals["GAMINC_RIM2"]
        gaminc_rim4 = phyex_externals["GAMINC_RIM4"]
        
        ldsoft=False

        ldcompute_gt4py = from_array(ldcompute, dtype=bool, backend=BACKEND)
        rhodref_gt4py = from_array(rhodref, dtype=np.float32, backend=BACKEND)
        pres_gt4py = from_array(pres, dtype=np.float32, backend=BACKEND)
        dv_gt4py = from_array(dv, dtype=np.float32, backend=BACKEND)
        ka_gt4py = from_array(ka, dtype=np.float32, backend=BACKEND)
        cj_gt4py = from_array(cj, dtype=np.float32, backend=BACKEND)
        lbdar_gt4py = from_array(lbdar, dtype=np.float32, backend=BACKEND)
        lbdas_gt4py = from_array(lbdas, dtype=np.float32, backend=BACKEND)
        t_gt4py = from_array(t, dtype=np.float32, backend=BACKEND)
        rvt_gt4py = from_array(rvt, dtype=np.float32, backend=BACKEND)
        rct_gt4py = from_array(rct, dtype=np.float32, backend=BACKEND)
        rrt_gt4py = from_array(rrt, dtype=np.float32, backend=BACKEND)
        rst_gt4py = from_array(rst, dtype=np.float32, backend=BACKEND)
        rcrimss_gt4py = from_array(rcrimss, dtype=np.float32, backend=BACKEND)
        rcrimsg_gt4py = from_array(rcrimsg, dtype=np.float32, backend=BACKEND)
        rsrimcg_gt4py = from_array(rsrimcg, dtype=np.float32, backend=BACKEND)
        rraccss_gt4py = from_array(rraccss, dtype=np.float32, backend=BACKEND)
        rraccsg_gt4py = from_array(rraccsg, dtype=np.float32, backend=BACKEND)
        rsaccrg_gt4py = from_array(rsaccrg, dtype=np.float32, backend=BACKEND)
        rs_mltg_tnd_gt4py = from_array(rs_mltg_tnd, dtype=np.float32, backend=BACKEND)
        rc_mltsr_tnd_gt4py = from_array(rc_mltsr_tnd, dtype=np.float32, backend=BACKEND)
        rs_rcrims_tnd_gt4py = from_array(rs_rcrims_tnd, dtype=np.float32, backend=BACKEND)
        rs_rcrimss_tnd_gt4py = from_array(rs_rcrimss_tnd, dtype=np.float32, backend=BACKEND)
        rs_rsrimcg_tnd_gt4py = from_array(rs_rsaccrg_tnd, dtype=np.float32, backend=BACKEND)
        rs_rraccs_tnd_gt4py = from_array(rs_rraccs_tnd, dtype=np.float32, backend=BACKEND)
        rs_rraccss_tnd_gt4py = from_array(rs_rraccss_tnd, dtype=np.float32, backend=BACKEND)
        rs_rsaccrg_tnd_gt4py = from_array(rs_rsaccrg_tnd, dtype=np.float32, backend=BACKEND)
        rs_freez1_tnd_gt4py = from_array(rs_freez1_tnd, dtype=np.float32, backend=BACKEND)
        rs_freez2_tnd_gt4py = from_array(rs_freez2_tnd, dtype=np.float32, backend=BACKEND)
        
        index_floor_gt4py = from_array(index_floor, dtype=np.float32, backend=BACKEND)
        index_floor_r_gt4py = from_array(index_floor_r, dtype=np.float32, backend=BACKEND)
        index_floor_s_gt4py = from_array(index_floor_s, dtype=np.float32, backend=BACKEND)
        
        gaminc_rim1_gt4py = from_array(gaminc_rim1, dtype=np.float32, backend=BACKEND)
        gaminc_rim2_gt4py = from_array(gaminc_rim2, dtype=np.float32, backend=BACKEND)
        gaminc_rim4_gt4py = from_array(gaminc_rim4, dtype=np.float32, backend=BACKEND)

        ice4_fast_rg_gt4py(
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
        )

        fortran_script = "mode_ice4_fast_rg.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)


        result = fortran_script.mode_ice4_fast_rs.ice4_fast_rs(
            kproma=SHAPE[0] * SHAPE[1] * SHAPE[2], 
            ksize=SHAPE[0] * SHAPE[1] * SHAPE[2], 
            ldsoft=ldsoft, 
            ldcompute=ldcompute,
            ngaminc=phyex_externals["NGAMINC"], 
            nacclbdas=phyex_externals["NACCLBDAS"], 
            nacclbdar=phyex_externals["NACCLBDAR"],
            levlimit=phyex_externals["LEVLIMIT"], 
            lpack_interp=phyex_externals["LPACK_INTERP"], 
            csnowriming=phyex_externals["CSNOWRIMING"],
            xcrimss=phyex_externals["CRIMSS"], 
            xexcrimss=phyex_externals["EXCRIMSS"], 
            xcrimsg=phyex_externals["CRIMSG"], 
            xexcrimsg=phyex_externals["EXCRIMSG"], 
            xexsrimcg2=phyex_externals["EXSRIMCG2"],
            xfraccss=phyex_externals["FRACCSS"],
            s_rtmin=phyex_externals["S_RTMIN"], 
            c_rtmin=phyex_externals["C_RTMIN"], 
            r_rtmin=phyex_externals["R_RTMIN"], 
            xepsilo=phyex_externals["EPSILO"], 
            xalpi=phyex_externals["ALPI"],
            xbetai=phyex_externals["BETAI"], 
            xgami=phyex_externals["GAMI"], 
            xtt=phyex_externals["TT"], 
            xlvtt=phyex_externals["LVTT"],
            xcpv=phyex_externals["CPV"],
            xci=phyex_externals["CI"],
            xcl=phyex_externals["CL"], 
            xlmtt=phyex_externals["LMTT"],
            xestt=phyex_externals["ESTT"], 
            xrv=phyex_externals["RV"],
            x0deps=phyex_externals["O0DEPS"], 
            x1deps=phyex_externals["O1DEPS"], 
            xex0deps=phyex_externals["EX0DEPS"], 
            xex1deps=phyex_externals["EX1DEPS"],
            xlbraccs1=phyex_externals["LBRACCS1"], 
            xlbraccs2=phyex_externals["LBRACCS2"], 
            xlbraccs3=phyex_externals["LBRACCS3"],
            xcxs=phyex_externals["CXS"], 
            xsrimcg2=phyex_externals["SRIMCG2"],
            xsrimcg3=phyex_externals["SRIMCG3"], 
            xbs=phyex_externals["BS"],
            xlbsaccr1=phyex_externals["LBSACCR1"],
            xlbsaccr2=phyex_externals["LBSACCR2"], 
            xlbsaccr3=phyex_externals["LBSACCR3"], 
            xfsaccrg=phyex_externals["FSACCRG"],
            xsrimcg=phyex_externals["SRIMCG"], 
            xexsrimcg=phyex_externals["EXSRIMCG"], 
            xcexvt=phyex_externals["CVEXT"],
            xalpw=phyex_externals["ALPW"],
            xbetaw=phyex_externals["BETAW"], 
            xgamw=phyex_externals["GAMW"], 
            xfscvmg=phyex_externals["FSCVMG"],
            xker_raccss=KER_RACCSS, 
            xker_raccs=KER_RACCS,
            xker_saccrg=KER_SACCRG,
            xgaminc_rim1=gaminc_rim1_gt4py, 
            xgaminc_rim2=gaminc_rim2_gt4py, 
            xgaminc_rim4=gaminc_rim4_gt4py,
            xrimintp1=phyex_externals["RIMINTP1"],
            xrimintp2=phyex_externals["RIMINTP2"], 
            xaccintp1s=phyex_externals["ACCINTP1S"], 
            xaccintp2s=phyex_externals["ACCINTP2S"], 
            xaccintp1r=phyex_externals["ACCINTP1R"], 
            xaccintp2r=phyex_externals["ACCINTP2R"], 
            prhodref=rhodref_gt4py, 
            ppres=pres_gt4py, 
            pdv=dv_gt4py, 
            pka=ka_gt4py, 
            pcj=cj_gt4py,
            plbdar=lbdar_gt4py, 
            plbdas=lbdas_gt4py,
            pt=t_gt4py, 
            prvt=rvt_gt4py, 
            prct=rct_gt4py, 
            prrt=rrt_gt4py, 
            prst=rst_gt4py,
            priaggs=riaggs_gt4py,
            prcrimss=rcrimss_gt4py,
            prcrimsg=rcrimsg_gt4py, prsrimcg=rsrimcg_gt4py,
            prraccss=rraccss_gt4py, prraccsg=rraccsg_gt4py, 
            prsaccrg=rsaccrg_gt4py, prsmltg=rs_mltg_tnd_gt4py,
            prcmltsr=rc_mltsr_tnd_gt4py,
            prs_tend=rst_gt4py
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
        
        