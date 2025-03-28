from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.config import GT4PyConfig
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

class TestCondensation(unittest.TestCase):
    
    def test_condensation(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, 
            rebuild=REBUILD, 
            validate_args=VALIDATE_ARGS, 
            verbose=True
        )

        phyex_externals = Phyex("AROME").to_externals()
        condensation = compile_stencil("condensation", gt4py_config, phyex_externals)
        
        sigqsat = np.array(
                np.random.rand(SHAPE[0], SHAPE[1]),
                dtype=c_float,
                order="F",
            )
        sigrc = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        pabs = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        sigs = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        t = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rv_in = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ri_in = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rc_in = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rv_out = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rc_out = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ri_out = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        cldfr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        cph = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
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

        
        
        sigqsat_gt4py = from_array(
            sigqsat,
            dtype=np.float32,
            backend=BACKEND
        )
        pabs_gt4py = from_array(
            pabs,
            dtype=np.float32,
            backend=BACKEND
        )
        sigs_gt4py = from_array(
            sigs,
            dtype=np.float32,
            backend=BACKEND
        )
        t_gt4py = from_array(
            t,
            dtype=np.float32,
            backend=BACKEND
        )
        rv_in_gt4py = from_array(
            rv_in,
            dtype=np.float32,
            backend=BACKEND
        )
        ri_in_gt4py = from_array(
            rv_in,
            dtype=np.float32,
            backend=BACKEND
        )
        rc_in_gt4py = from_array(
            rc_in,
            dtype=np.float32,
            backend=BACKEND
        )
        rv_out_gt4py = from_array(
            rv_out,
            dtype=np.float32,
            backend=BACKEND
        )
        rc_out_gt4py = from_array(
            rc_out,
            dtype=np.float32,
            backend=BACKEND
        )
        ri_out_gt4py = from_array(
            ri_out,
            dtype=np.float32,
            backend=BACKEND
        )
        cldfr_gt4py = from_array(
            cldfr,
            dtype=np.float32,
            backend=BACKEND
        )
        cph_gt4py = from_array(
            cph,
            dtype=np.float32,
            backend=BACKEND
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
        sigrc_gt4py = from_array(
            sigrc,
            dtype=np.float32,
            backend=BACKEND
        )
        

        condensation(
            sigqsat=sigqsat_gt4py,
            pabs=pabs_gt4py,
            sigs=sigs_gt4py, 
            t=t_gt4py,
            rv_in=rv_in_gt4py,
            ri_in=ri_in_gt4py,
            rc_in=rc_in_gt4py,
            rv_out=rv_out_gt4py,
            rc_out=rc_out_gt4py,
            ri_out=ri_out_gt4py,
            cldfr=cldfr_gt4py,
            cph=cph_gt4py,
            lv=lv_gt4py,
            ls=ls_gt4py,
        )

        fortran_script = "mode_condensation.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)

        result = fortran_script.mode_condensation.condensation(
            nijb=1, 
            nije=SHAPE[0] * SHAPE[1], 
            nktb=1, 
            nkte=SHAPE[2], 
            nijt=SHAPE[0] * SHAPE[1], 
            nkt=SHAPE[2],  
            xrv=phyex_externals["RV"], 
            xrd=phyex_externals["RD"], 
            xalpi=phyex_externals["ALPI"], 
            xbetai=phyex_externals["BETAI"], 
            xgami=phyex_externals["GAMI"], 
            xalpw=phyex_externals["ALPW"], 
            xbetaw=phyex_externals["BETAW"], 
            xgamw=phyex_externals["GAMW"],
            osigmas=phyex_externals["LSIGMAS"], 
            ocnd2=phyex_externals["OCND2"],                                        
            hcondens=phyex_externals["HCONDENS"], 
            hlambda3=phyex_externals["LAMBDA3"], 
            lstatnw=phyex_externals["LSTATNW"],                           
            ppabs=pabs.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            pt=t.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),                                             
            pt_out=t.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            prv_in=rv_in.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            prv_out=rv_out.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            prc_in=rc_in.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            prc_out=rc_out.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            pri_in=ri_in.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            pri_out=ri_out.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),     
            psigs=sigs.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            pcldfr=cldfr.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            psigrc=sigrc.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),                                 
            psigqsat=sigqsat.reshape(SHAPE[0]*SHAPE[1]),                                              
            plv=lv.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            pls=ls.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            pcph=cph.reshape(SHAPE[0]*SHAPE[1], SHAPE[2])
        )
        
        pt_out = result[0]  
        prv_out = result[1]
        prc_out = result[2] 
        pri_out = result[3] 
        pcldfr_out = result[4]
        psigrc_out = result[5]
        
    
        logging.info(f"Machine precision {np.finfo(float).eps}")
        
        logging.info(f"Mean t_gt4py       {t_gt4py.mean()}")
        logging.info(f"Mean pt_out        {pt_out.mean()}")
        logging.info(f"Max abs err t      {max(abs(t_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - pt_out) / abs(pt_out))}")

        logging.info(f"Mean rv_gt4py        {rv_out_gt4py.mean()}")
        logging.info(f"Mean prv_out         {prv_out.mean()}")
        logging.info(f"Max abs err rv       {max(abs(rv_out_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - prv_out) / abs(prv_out))}")

        logging.info(f"Mean rc_out          {rc_out_gt4py.mean()}")
        logging.info(f"Mean prc_out         {prc_out.mean()}")
        logging.info(f"Max abs err rc_out   {max(abs(rc_out_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - prc_out) / abs(prc_out))}")

        logging.info(f"Mean ri_out_gt4py    {ri_out_gt4py.mean()}")
        logging.info(f"Mean ri_out          {pri_out.mean()}")
        logging.info(f"Max abs err ri       {max(abs(ri_out_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - pri_out) / abs(pri_out))}")


        logging.info(f"Mean ri_out_gt4py    {cldfr_gt4py.mean()}")
        logging.info(f"Mean ri_out          {pcldfr_out.mean()}")
        logging.info(f"Max abs err ri       {max(abs(cldfr_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - pcldfr_out) / abs(pcldfr_out))}")


        logging.info(f"Mean sigrc           {sigrc_gt4py.mean()}")
        logging.info(f"Mean psigrc          {psigrc_out.mean()}")
        logging.info(f"Max abs err ri       {max(abs(sigrc_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - psigrc_out) / abs(psigrc_out))}")

        assert_allclose(pt_out, t_gt4py.ravel(), rtol=1e-6)
        assert_allclose(prv_out, rv_out_gt4py.ravel(), rtol=1e-6)
        assert_allclose(prc_out, rc_out_gt4py.ravel(), rtol=1e-6)
        assert_allclose(pri_out, ri_out_gt4py.ravel(), rtol=1e-6)
        
        assert_allclose(pcldfr_out, cldfr_gt4py.ravel(), rtol=1e-6)
        assert_allclose(psigrc_out, sigrc_gt4py.ravel(), rtol=1e-6)


        