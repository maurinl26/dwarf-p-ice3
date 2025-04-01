from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.config import GT4PyConfig, DataTypes
from ice3_gt4py.phyex_common.tables import SRC_1D
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
            verbose=False,
            dtypes=DataTypes(
                bool=bool, 
                float=np.float32, 
                int=np.int32)
        )

        phyex_externals = Phyex("AROME").to_externals()
        phyex_externals.update({"OCND2": False})
        logging.info(f"OCND2 : {phyex_externals['OCND2']}")
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
        t = 300 * np.array(
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
        t_out = np.array(
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
        q1 = np.array(
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
        q1_gt4py = from_array(
            q1,
            dtype=np.float32,
            backend=BACKEND
        )
        
        
        # Temporaries 
        pv = np.zeros(
                (SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        piv = np.zeros(
                (SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        frac_tmp = np.zeros(
                (SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        qsl = np.zeros(
                (SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        qsi = np.zeros(
                (SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            ) 
        sigma = np.zeros(
            (SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
        )
        cond_tmp = np.zeros(
            (SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
        )
        a = np.zeros(
            (SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F"
                )
        b = np.zeros(
            (SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F"
                )
        sbar = np.zeros(
            (SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F"
                )
        pv_gt4py = from_array(pv, dtype=np.float32, backend=BACKEND)
        piv_gt4py = from_array(piv, dtype=np.float32, backend=BACKEND)
        frac_tmp_gt4py = from_array(frac_tmp, dtype=np.float32, backend=BACKEND)
        qsl_gt4py = from_array(qsl, dtype=np.float32, backend=BACKEND)
        qsi_gt4py = from_array(qsi, dtype=np.float32, backend=BACKEND)
        sigma_gt4py = from_array(sigma, dtype=np.float32, backend=BACKEND)
        cond_tmp_gt4py = from_array(cond_tmp, dtype=np.float32, backend=BACKEND)
        a_gt4py = from_array(a, dtype=np.float32, backend=BACKEND)
        b_gt4py = from_array(b, dtype=np.float32, backend=BACKEND)
        sbar_gt4py = from_array(sbar, dtype=np.float32, backend=BACKEND)        
        
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
            q1=q1_gt4py,
            # Temporaries
            pv=pv_gt4py,
            piv=piv_gt4py,
            frac_tmp=frac_tmp_gt4py,
            qsl=qsl_gt4py,
            qsi=qsi_gt4py,
            sigma=sigma_gt4py,
            cond_tmp=cond_tmp_gt4py,
            a=a_gt4py,
            b=b_gt4py,
            sbar=sbar_gt4py
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
        
        shapes = {
            "nijb": 1,
            "nije": SHAPE[0] * SHAPE[1],
            "nijt": SHAPE[0] * SHAPE[1],
            "nktb": 1,
            "nkte": SHAPE[2],
            "nkt": SHAPE[2],
        }
        
        logical_keys = {
            "osigmas":phyex_externals["LSIGMAS"], 
            "ocnd2":phyex_externals["OCND2"],      
            "ouseri":True,
            "hfrac_ice":phyex_externals["FRAC_ICE_ADJUST"],                                  
            "hcondens":phyex_externals["CONDENS"], 
            "lstatnw":phyex_externals["LSTATNW"],
        }
        
        constant_def = {
            "xrv":phyex_externals["RV"], 
            "xrd":phyex_externals["RD"], 
            "xalpi":phyex_externals["ALPI"], 
            "xbetai":phyex_externals["BETAI"], 
            "xgami":phyex_externals["GAMI"], 
            "xalpw":phyex_externals["ALPW"], 
            "xbetaw":phyex_externals["BETAW"], 
            "xgamw":phyex_externals["GAMW"],
            "xtmaxmix":phyex_externals["TMAXMIX"],
            "xtminmix":phyex_externals["TMINMIX"],
        }

        result = fortran_script.mode_condensation.condensation(                         
            ppabs=pabs.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            pt=t.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),                                             
            prv_in=rv_in.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            prc_in=rc_in.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            pri_in=ri_in.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            psigs=sigs.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            psigqsat=sigqsat.reshape(SHAPE[0]*SHAPE[1]),                                              
            plv=lv.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            pls=ls.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            pcph=cph.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),
            pt_out=t.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            prv_out=rv_out.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            prc_out=rc_out.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            pri_out=ri_out.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),     
            pcldfr=cldfr.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            zq1=q1.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),
            # Temporaries
            zpv=pv.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),
            zpiv=piv.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),
            zfrac=frac_tmp.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),
            zqsl=qsl.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),
            zqsi=qsi.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),
            zsigma=sigma.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),
            zcond=cond_tmp.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),
            za=a.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),
            zb=b.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),
            zsbar=sbar.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),
            **shapes,
            **logical_keys,
            **constant_def
        )
        
        pt_out = result[0]  
        prv_out = result[1]
        prc_out = result[2] 
        pri_out = result[3] 
        pcldfr_out = result[4]
        zq1_out = result[5]
        
        
        logging.info("Testing temporaries")
        pv_out = result[6]
        piv_out = result[7]
        zfrac_out =result[8]
        zqsl_out = result[9]
        zqsi_out = result[10]
        zsigma_out = result[11]
        zcond_out = result[12]
        za_out = result[13]
        zb_out = result[14]
        zsbar_out = result[15]
        
        logging.info(f"Mean pv_gt4py      {pv_gt4py.mean()}")
        logging.info(f"Mean pv_out        {pv_out.mean()}")

        logging.info(f"Mean piv_gt4py     {piv_gt4py.mean()}")
        logging.info(f"Mean piv_out       {piv_out.mean()}")
        
        logging.info(f"Mean frac_tmp_gt4py {frac_tmp_gt4py.mean()}")
        logging.info(f"Mean zfrac_out      {zfrac_out.mean()}")
        
        logging.info(f"Mean qsl_gt4py     {qsl_gt4py.mean()}")
        logging.info(f"Mean zqsl_out      {zqsl_out.mean()}")
        
        logging.info(f"Mean qsi_gt4py     {qsi_gt4py.mean()}")
        logging.info(f"Mean zqsi_out      {zqsi_out.mean()}")
        
        logging.info(f"Mean sigma_gt4py   {sigma_gt4py.mean()}")
        logging.info(f"Mean zsigma_out    {zsigma_out.mean()}")
        
        logging.info(f"Mean cond_tmp_gt4py  {cond_tmp_gt4py.mean()}")
        logging.info(f"Mean zcond_out       {zcond_out.mean()}")
        
        logging.info(f"Mean a_gt4py         {a_gt4py.mean()}")
        logging.info(f"Mean za_out          {za_out.mean()}")
        
        logging.info(f"Mean b_gt4py         {b_gt4py.mean()}")
        logging.info(f"Mean zb_out          {zb_out.mean()}")
        
        logging.info(f"Mean sbar_gt4py      {sbar_gt4py.mean()}")
        logging.info(f"Mean zsbar_out       {zsbar_out.mean()}")
        
        # assert_allclose(pv_out, pv_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-5)
        # assert_allclose(piv_out, piv_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-5)
        # assert_allclose(zqsl_out, qsl_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-5)
        # assert_allclose(zqsi_out, qsi_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-5)
    
        logging.info(f"Machine precision {np.finfo(float).eps}")
        
        logging.info(f"Mean t_gt4py       {t_gt4py.mean()}")
        logging.info(f"Mean pt_out        {pt_out.mean()}")

        logging.info(f"Mean rv_gt4py        {rv_out_gt4py.mean()}")
        logging.info(f"Mean prv_out         {prv_out.mean()}")

        logging.info(f"Mean rc_out          {rc_out_gt4py.mean()}")
        logging.info(f"Mean prc_out         {prc_out.mean()}")

        logging.info(f"Mean ri_out_gt4py    {ri_out_gt4py.mean()}")
        logging.info(f"Mean ri_out          {pri_out.mean()}")

        logging.info(f"Mean cldfr_gt4py     {cldfr_gt4py.mean()}")
        logging.info(f"Mean pcldfr          {pcldfr_out.mean()}")
        
        logging.info(f"Mean q1_gt4py        {q1_gt4py.mean()}")
        logging.info(f"Mean zq1             {zq1_out.mean()}")

        assert_allclose(pt_out, t_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(prv_out, rv_out_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(prc_out, rc_out_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(pri_out, ri_out_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        
        assert_allclose(pcldfr_out, cldfr_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(zq1_out, q1_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        
    # def test_sigrc_computation(self):
        
    #     logging.info(f"With backend {BACKEND}")
    #     gt4py_config = GT4PyConfig(
    #         backend=BACKEND, 
    #         rebuild=REBUILD, 
    #         validate_args=VALIDATE_ARGS, 
    #         verbose=True
    #     )
        

    #     phyex_externals = Phyex("AROME").to_externals()
    #     logging.info(f"HLAMBDA3 {phyex_externals['LAMBDA3']}")

    #     sigrc_computation = compile_stencil("sigrc", gt4py_config, phyex_externals)
        
    #     q1 = np.array(
    #             np.random.rand(SHAPE[0], SHAPE[1]),
    #             dtype=c_float,
    #             order="F",
    #         )
        
    #     sigrc = np.array(
    #         np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
    #         dtype=c_float,
    #         order="F"
    #     )
        
    #     inq1 = np.ones(
    #         (SHAPE[1], SHAPE[2]),
    #         dtype=np.int32,
    #         order="F"
    #     )
    #     inq1_gt4py = from_array(
    #         inq1, dtype=np.int32, backend=BACKEND
    #     )
        
       
        
    #     q1_gt4py = from_array(
    #         q1,
    #         dtype=np.float32,
    #         backend=BACKEND
    #     )
    #     sigrc_gt4py = from_array(
    #         sigrc,
    #         dtype=np.float32,
    #         backend=BACKEND
    #     )
    #     src_1d_gt4py = from_array(
    #         SRC_1D,
    #         dtype=np.float32,
    #         backend=BACKEND
    #     )
        
    #     sigrc_computation(
    #         q1=q1_gt4py,
    #         sigrc=sigrc_gt4py,
    #         src_1d=src_1d_gt4py,
    #         inq1=inq1_gt4py,
    #     )

    #     fortran_script = "mode_condensation.F90"
    #     current_directory = Path.cwd()
    #     root_directory = current_directory
    #     stencils_directory = Path(
    #         root_directory, "src", "ice3_gt4py", "stencils_fortran"
    #     )
    #     script_path = Path(stencils_directory, fortran_script)

    #     logging.info(f"Fortran script path {script_path}")
    #     fortran_script = fmodpy.fimport(script_path)

    #     result = fortran_script.mode_condensation.sigrc_computation(
    #         nijb=1, 
    #         nije=SHAPE[0] * SHAPE[1], 
    #         nktb=1, 
    #         nkte=SHAPE[2], 
    #         nijt=SHAPE[0] * SHAPE[1], 
    #         nkt=SHAPE[2], 
    #         hlambda3=phyex_externals["LAMBDA3"],
    #         zq1=q1,
    #         src_1d=SRC_1D,
    #     )
        
    #     psigrc_out = result[0]  
    
    #     logging.info(f"Machine precision {np.finfo(float).eps}")
        
    #     logging.info(f"Mean sigrc_gt4py     {sigrc_gt4py.mean()}")
    #     logging.info(f"Mean psi             {psigrc_out.mean()}")

    #     assert_allclose(psigrc_out, sigrc_gt4py.ravel(), rtol=1e-6)


        