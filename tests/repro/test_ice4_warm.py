from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.config import GT4PyConfig
from ice3_gt4py.phyex_common.phyex import Phyex
from gt4py.storage import from_array
import numpy as np
from pathlib import Path
import fmodpy
import unittest
from numpy.testing import assert_allclose

import logging

from .env import BACKEND, REBUILD, VALIDATE_ARGS, SHAPE


class TestIce4Warm(unittest.TestCase):
    def test_ice4_warm(self):
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
        )

        phyex_externals = Phyex("AROME").to_externals()
        ice4_warm_gt4py = compile_stencil("ice4_warm", gt4py_config, phyex_externals)
        
        logging.info(f"SUBG_RR_EVAP {phyex_externals["SUBG_RR_EVAP"]}")
        
        ldcompute = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=bool,
                order="F",
            )
        rhodref = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        
        t = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
        )
        pres = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        tht = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        lbdar = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        lbdar_rf = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        ka = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        dv = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        cj = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        hlc_hcf = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        hlc_hrc = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        cf = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        rf = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
        )
        rvt = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        rct = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        rrt = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        rcautr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        rcaccr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        rrevav = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )

        ldcompute_gt4py = from_array(ldcompute, dtype=bool, backend=BACKEND)
        rhodref_gt4py = from_array(rhodref, dtype=float, backend=BACKEND)
        t_gt4py = from_array(t, dtype=float, backend=BACKEND)
        pres_gt4py = from_array(pres, dtype=float, backend=BACKEND)
        tht_gt4py = from_array(tht, dtype=float, backend=BACKEND)
        lbdar_gt4py = from_array(lbdar, dtype=float, backend=BACKEND)
        lbdar_rf_gt4py = from_array(lbdar_rf, dtype=float, backend=BACKEND)
        ka_gt4py = from_array(ka, dtype=float, backend=BACKEND)
        dv_gt4py = from_array(dv, dtype=float, backend=BACKEND)
        cj_gt4py = from_array(cj, dtype=float, backend=BACKEND)
        hlc_hcf_gt4py = from_array(hlc_hcf, dtype=float, backend=BACKEND)
        hlc_hrc_gt4py = from_array(hlc_hrc, dtype=float, backend=BACKEND)
        cf_gt4py = from_array(cf, dtype=float, backend=BACKEND) 
        rf_gt4py = from_array(rf, dtype=float, backend=BACKEND)
        rvt_gt4py = from_array(rvt, dtype=float, backend=BACKEND)
        rct_gt4py = from_array(rct, dtype=float, backend=BACKEND)
        rrt_gt4py = from_array(rrt, dtype=float, backend=BACKEND)
        rcautr_gt4py = from_array(rcautr, dtype=float, backend=BACKEND)
        rcaccr_gt4py = from_array(rcaccr, dtype=float, backend=BACKEND)
        rrevav_gt4py = from_array(rrevav, dtype=float, backend=BACKEND)

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

        fortran_script = "mode_ice4_warm.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
        
        result = fortran_script.mode_ice4_warm.ice4_warm(
            xalpw=phyex_externals["ALPW"],
            xbetaw=phyex_externals["BETAW"],
            xgamw=phyex_externals["GAMW"],
            xepsilo=phyex_externals["EPSILO"],
            xlvtt=phyex_externals["LVTT"],
            xcpv=phyex_externals["CPV"],
            xcl=phyex_externals["CL"],
            xtt=phyex_externals["TT"],
            xrv=phyex_externals["RV"],
            xcpd=phyex_externals["CPD"],
            xtimautc=phyex_externals["TIMAUTC"],
            xcriautc=phyex_externals["CRIAUTC"],
            xfcaccr=phyex_externals["FCACCR"],
            xexcaccr=phyex_externals["EXCACCR"],
            x0evar=phyex_externals["O0EVAR"],
            x1evar=phyex_externals["O1EVAR"],
            xex0evar=phyex_externals["EX0EVAR"],
            xex1evar=phyex_externals["EX1EVAR"],
            c_rtmin=phyex_externals["C_RTMIN"],
            r_rtmin=phyex_externals["R_RTMIN"],
            xcexvt=phyex_externals["CEXVT"],
            kproma=SHAPE[0] * SHAPE[1] * SHAPE[2],
            ksize=SHAPE[0] * SHAPE[1] * SHAPE[2],
            ldsoft=ldsoft,
            ldcompute=ldcompute,
            hsubg_rr_evap='none',
            prhodref=rhodref.ravel(),
            pt=t.ravel(),
            ppres=pres.ravel(),
            ptht=tht.ravel(),
            plbdar=lbdar.ravel(),
            plbdar_rf=lbdar_rf.ravel(),
            pka=ka.ravel(),
            pdv=dv.ravel(),
            pcj=cj.ravel(),
            phlc_hcf=hlc_hcf.ravel(),
            phlc_hrc=hlc_hrc.ravel(),
            pcf=cf.ravel(),
            prf=rf.ravel(),
            prvt=rvt.ravel(),
            prct=rct.ravel(),
            prrt=rrt.ravel(),
            prcautr=rcautr.ravel(),
            prcaccr=rcaccr.ravel(),
            prrevav=rrevav.ravel(),
        )

        rcautr_out = result[0]
        rcaccr_out = result[1]
        rrevav_out = result[2]

        logging.info(f"Machine precision {np.finfo(float).eps}")

        logging.info(f"Mean rcautr_gt4py    {rcautr_gt4py.mean()}")
        logging.info(f"Mean rcautr_out      {rcautr_out.mean()}")
        logging.info(f"Max abs err rcautr   {max(abs(rcautr_gt4py.ravel() - rcautr_out) / abs(rcautr_out))}")

        logging.info(f"Mean rcaccr_gt4py    {rcaccr_gt4py.mean()}")
        logging.info(f"Mean rcaccr_out      {rcaccr_out.mean()}")
        logging.info(f"Max abs err rcaccr   {max(abs(rcaccr_gt4py.ravel() - rcaccr_out) / abs(rcaccr_out))}")

        logging.info(f"Mean rrevav_gt4py    {rrevav_gt4py.mean()}")
        logging.info(f"Mean rrevav_out      {rrevav_out.mean()}")
        logging.info(f"Max abs err rrevav   {max(abs(rrevav_gt4py.ravel() - rrevav_out) / abs(rrevav_out))}")

        assert_allclose(rcautr_out, rcautr_gt4py.ravel(), rtol=1e-5)
        assert_allclose(rcaccr_out, rcaccr_gt4py.ravel(), rtol=1e-6)
        assert_allclose(rrevav_out, rrevav_gt4py.ravel(), rtol=1e-6)
        

