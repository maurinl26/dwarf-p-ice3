from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.config import GT4PyConfig, DataTypes
from ice3_gt4py.phyex_common.phyex import Phyex
from gt4py.storage import from_array
import numpy as np
from numpy.testing import assert_allclose
from pathlib import Path
import fmodpy
import unittest
from ctypes import c_float

import logging

from .env import BACKEND, REBUILD, VALIDATE_ARGS, SHAPE

class TestCloudFraction(unittest.TestCase):
    
    def test_cloud_fraction(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
        )

        phyex_externals = Phyex("AROME").to_externals()
        cloud_fraction = compile_stencil("cloud_fraction", gt4py_config, phyex_externals)
        
        dt = 50.0
        
        lv = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ls = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        t = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        cph = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rhodref = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        exnref = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rc = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ri = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ths = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rvs = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rcs = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ris = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rc_mf = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ri_mf = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        cf_mf = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        cldfr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rc_tmp = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ri_tmp = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        hlc_hrc = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        hlc_hcf = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        hli_hri = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        hli_hcf = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        
        lv_gt4py = from_array(
            lv,
            dtype=np.float32,
            backend=BACKEND
        )
        ls_gt4py = from_array(
            ls,
            dtype=np.float32,
            backend=BACKEND
        )
        t_gt4py = from_array(
            t,
            dtype=np.float32,
            backend=BACKEND
        )
        cph_gt4py = from_array(
            cph,
            dtype=np.float32,
            backend=BACKEND
        )
        rhodref_gt4py = from_array(
            rhodref,
            dtype=np.float32,
            backend=BACKEND
        )
        exnref_gt4py = from_array(
            exnref,
            dtype=np.float32,
            backend=BACKEND
        )
        rc_gt4py = from_array(
            rc,
            dtype=np.float32,
            backend=BACKEND
        )
        ri_gt4py = from_array(
            ri,
            dtype=np.float32,
            backend=BACKEND
        )
        ths_gt4py = from_array(
            ths,
            dtype=np.float32,
            backend=BACKEND
        )
        rvs_gt4py = from_array(
            rvs,
            dtype=np.float32,
            backend=BACKEND
        )
        rcs_gt4py = from_array(
            rcs,
            dtype=np.float32,
            backend=BACKEND
        )
        ris_gt4py = from_array(
            ris,
            dtype=np.float32,
            backend=BACKEND
        )
        rc_mf_gt4py = from_array(
            rc_mf,
            dtype=np.float32,
            backend=BACKEND
        )
        ri_mf_gt4py = from_array(
            ri_mf,
            dtype=np.float32,
            backend=BACKEND
        )
        cf_mf_gt4py = from_array(
            cf_mf,
            dtype=np.float32,
            backend=BACKEND
        )
        cldfr_gt4py = from_array(
            cldfr,
            dtype=np.float32,
            backend=BACKEND
        )
        rc_tmp_gt4py = from_array(
            rc_tmp,
            dtype=np.float32,
            backend=BACKEND
        )
        ri_tmp_gt4py = from_array(
            ri_tmp,
            dtype=np.float32,
            backend=BACKEND
        )
        hlc_hrc_gt4py = from_array(
            hlc_hrc,
            dtype=np.float32,
            backend=BACKEND
        )
        hlc_hcf_gt4py = from_array(
            hlc_hcf,
            dtype=np.float32,
            backend=BACKEND
        )
        hli_hri_gt4py = from_array(
            hli_hri,
            dtype=np.float32,
            backend=BACKEND
        )
        hli_hcf_gt4py = from_array(
            hli_hcf,
            dtype=np.float32,
            backend=BACKEND
        )
        

        cloud_fraction(
            lv=lv_gt4py,
            ls=ls_gt4py,
            t=t_gt4py,
            cph=cph_gt4py,
            rhodref=rhodref_gt4py,
            exnref=exnref_gt4py,
            rc=rc_gt4py,
            ri=ri_gt4py,
            ths=ths_gt4py,
            rvs=rvs_gt4py,
            rcs=rcs_gt4py,
            ris=ris_gt4py,
            rc_mf=rc_mf_gt4py,
            ri_mf=ri_mf_gt4py,
            cf_mf=cf_mf_gt4py,
            cldfr=cldfr_gt4py,
            rc_tmp=rc_tmp_gt4py,
            ri_tmp=ri_tmp_gt4py,
            hlc_hrc=hlc_hrc_gt4py,
            hlc_hcf=hlc_hcf_gt4py,
            hli_hri=hli_hri_gt4py,
            hli_hcf=hli_hcf_gt4py,
            dt=dt,
        )

        fortran_script = "mode_cloud_fraction.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)

        result = fortran_script.mode_cloud_fraction.cloud_fraction(
            nijt=SHAPE[0] * SHAPE[1], 
            nkt=SHAPE[2],
            nktb=1,
            nkte=SHAPE[2], 
            nijb=1, 
            nije=SHAPE[0]*SHAPE[1],
            lsubg_cond=phyex_externals["LSUBG_COND"],
            xcriautc=phyex_externals["CRIAUTC"], 
            xcriauti=phyex_externals["CRIAUTI"], 
            xacriauti=phyex_externals["ACRIAUTI"], 
            xbcriauti=phyex_externals["BCRIAUTI"], 
            xtt=phyex_externals["TT"],
            csubg_mf_pdf=phyex_externals["CSUBG_MF_PDF"],
            ptstep=dt,
            zrc=rc_tmp.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            zri=ri_tmp.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),
            pexnref=exnref.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            prhodref=rhodref.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),
            pcf_mf=cf_mf.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),   
            prc_mf=rc_mf.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),   
            pri_mf=ri_mf.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),   
            prc=rc.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),    
            pri=ri.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),  
            prvs=rvs.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),    
            prcs=rcs.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),    
            pths=ths.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),    
            pris=ris.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            pcldfr=cldfr.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            phlc_hrc=hlc_hrc.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),
            phlc_hcf=hlc_hcf.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),
            phli_hri=hli_hri.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),
            phli_hcf=hli_hcf.reshape(SHAPE[0]*SHAPE[1], SHAPE[2])
        )
        
        pths_out = result[0] 
        prvs_out=result[1] 
        prcs_out=result[2]
        pris_out=result[3]
        phlc_hrc_out=result[4]
        phlc_hcf_out=result[5]
        phli_hri_out=result[6]
        phli_hcf_out=result[7]
        
        logging.info(f"Machine precision {np.finfo(float).eps}")
        
        logging.info(f"Mean ths_gt4py       {ths_gt4py.mean()}")
        logging.info(f"Mean pths_out        {pths_out.mean()}")
        logging.info(f"Max abs err rcautr   {max(abs(ths_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - pths_out) / abs(pths_out))}")

        logging.info(f"Mean rvs_gt4py       {rvs_gt4py.mean()}")
        logging.info(f"Mean prvs_out        {prvs_out.mean()}")
        logging.info(f"Max abs err rcautr   {max(abs(rvs_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - prvs_out) / abs(prvs_out))}")

        logging.info(f"Mean rcs_gt4py       {rcs_gt4py.mean()}")
        logging.info(f"Mean prcs_out        {prcs_out.mean()}")
        logging.info(f"Max abs err rcautr   {max(abs(rcs_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - prcs_out) / abs(prcs_out))}")

        logging.info(f"Mean ris_gt4py       {ris_gt4py.mean()}")
        logging.info(f"Mean pris_out        {pris_out.mean()}")
        logging.info(f"Max abs err ris      {max(abs(ris_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - pris_out) / abs(pris_out))}")

        logging.info(f"Mean hlc_hrc_gt4py       {hlc_hrc_gt4py.mean()}")
        logging.info(f"Mean phlc_hrc_out        {phlc_hrc_out.mean()}")
        logging.info(f"Max abs err phlc_hrc     {max(abs(phlc_hrc_out.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - phlc_hrc_out) / abs(phlc_hrc_out))}")

        logging.info(f"Mean hlc_hcf_gt4py       {hlc_hcf_gt4py.mean()}")
        logging.info(f"Mean phlc_hcf_out        {phlc_hcf_out.mean()}")
        logging.info(f"Max abs err phlc_hrc     {max(abs(hlc_hcf_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - phlc_hcf_out) / abs(phlc_hcf_out))}")

        logging.info(f"Mean hli_hri_gt4py       {hli_hri_gt4py.mean()}")
        logging.info(f"Mean phli_hri_out        {phli_hri_out.mean()}")
        logging.info(f"Max abs err phlc_hrc     {max(abs(hli_hri_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - phli_hri_out) / abs(phli_hri_out))}")

        logging.info(f"Mean hli_hcf_gt4py       {hli_hcf_gt4py.mean()}")
        logging.info(f"Mean phli_hcf_out        {phli_hcf_out.mean()}")
        logging.info(f"Max abs err phlc_hrc     {max(abs(hli_hcf_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - phli_hcf_out) / abs(phli_hcf_out))}")


        assert_allclose(pths_out, ths_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(prvs_out, rvs_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(prcs_out, rcs_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(pris_out, ris_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        
        assert_allclose(phlc_hrc_out, hlc_hrc_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(phlc_hcf_out, hlc_hcf_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(phli_hri_out, hli_hri_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(phli_hcf_out, hli_hcf_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)

        