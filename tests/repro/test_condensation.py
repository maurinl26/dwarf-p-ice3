from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.config import GT4PyConfig, DataTypes
from ice3_gt4py.phyex_common.tables import SRC_1D
from ice3_gt4py.phyex_common.phyex import Phyex
from gt4py.storage import from_array, ones
import numpy as np
from numpy.testing import assert_allclose
from pathlib import Path
import fmodpy
import unittest
from ctypes import c_float

import logging

from tests.repro.conftest import REBUILD, VALIDATE_ARGS, SHAPE

class TestCondensation(unittest.TestCase):
    
    def setUp(self):
        self.fortran_shapes = {
            "nijb": 1,
            "nije": SHAPE[0] * SHAPE[1],
            "nijt": SHAPE[0] * SHAPE[1],
            "nktb": 1,
            "nkte": SHAPE[2],
            "nkt": SHAPE[2],
        }
        
        logging.info(f"With backend {self.gt4py_config.backend}")
        self.gt4py_config = GT4PyConfig(
            backend=self.gt4py_config.backend, 
            rebuild=REBUILD, 
            validate_args=VALIDATE_ARGS, 
            verbose=False,
            dtypes=DataTypes(
                bool=bool, 
                float=np.float32, 
                int=np.int32)
        )
        
        self.phyex_externals = Phyex("AROME").to_externals()
        
        # Defining fortran routine to catch
        fortran_script = "mode_condensation.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        self.fortran_script = fmodpy.fimport(script_path)
    
    def test_condensation(self):
        
        self.phyex_externals.update({"OCND2": False})
        logging.info(f"OCND2 : {self.phyex_externals['OCND2']}")
        condensation = compile_stencil("condensation", self.gt4py_config, self.phyex_externals)

        FloatFieldsIJK_List = [
            "sigqsat",
            "sigrc", 
            "pabs",
            "sigs", 
            "t",
            "rv_in",
            "ri_in",
            "rc_in",
            "t_out",
            "rv_out",
            "rc_out",
            "ri_out",
            "cldfr",
            "cph",
            "lv",
            "ls",
            "q1",
        ]
        
        FloatFieldsIJK = {
            name: np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            ) for name in FloatFieldsIJK_List
        }
        
        FloatFieldsIJK["t"] += 300
        
        GT4Py_FloatFields_IJK = {
            name: from_array(
            field,
            dtype=np.float32,
            backend=self.gt4py_config.backend
        ) for name, field in FloatFieldsIJK.items()
        }
        
        Temporary_FloatFields_Names = [
            "pv",
            "piv",
            "piv",
            "frac_tmp",
            "qsl",
            "qsi", 
            "sigma",
            "cond_tmp",
            "a",
            "b",
            "sbar" 
        ]
        
        Temporary_FloatFieldsIJK = {
            name: np.zeros(
                (SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            ) for name in Temporary_FloatFields_Names
        }
        
        GT4Py_Temporary_FieldsIJK = {
            name: from_array(field, dtype=np.float32, backend=self.gt4py_config.backend)
            for name, field in Temporary_FloatFieldsIJK.items()
        }
        
        condensation(
            **GT4Py_FloatFields_IJK,
            **GT4Py_Temporary_FieldsIJK
        )
        
        self.phyex_externals.update({"OUSERI": True})
        
        keys_mapping = {
            "osigmas":"LSIGMAS", 
            "ocnd2":"OCND2",      
            "ouseri":"OUSERI",
            "hfrac_ice":"FRAC_ICE_ADJUST",                                  
            "hcondens":"CONDENS", 
            "lstatnw":"LSTATNW",
        }
        
        logical_keys = {
            fkey: self.phyex_externals[pykey]
            for fkey, pykey in keys_mapping.items()
        }

        kst_mapping = {
            "xrv":"RV", 
            "xrd":"RD", 
            "xalpi":"ALPI", 
            "xbetai":"BETAI", 
            "xgami":"GAMI", 
            "xalpw":"ALPW", 
            "xbetaw":"BETAW", 
            "xgamw":"GAMW",
            "xtmaxmix":"TMAXMIX",
            "xtminmix":"TMINMIX",
        }
        
        constant_def = {
            fname: self.phyex_externals[pyname]
            for fname, pyname in kst_mapping.items()
        }
        
        Py2F_Mapping = {
            "ppabs":"pabs", 
            "pt":"t",                                             
            "prv_in":"rv_in", 
            "prc_in":"rc_in", 
            "pri_in":"ri_in", 
            "psigs":"sigs", 
            "psigqsat":"sigqsat",                                              
            "plv":"lv", 
            "pls":"ls", 
            "pcph":"cph",
            "pt_out":"t", 
            "prv_out":"rv_out", 
            "prc_out":"rc_out", 
            "pri_out":"ri_out",     
            "pcldfr":"cldfr", 
            "zq1":"q1",
            "zpv":"pv",
            "zpiv":"piv",
            "zfrac":"frac_tmp",
            "zqsl":"qsl",
            "zqsi":"qsi",
            "zsigma":"sigma",
            "zcond":"cond_tmp",
            "za":"a",
            "zb":"b",
            "zsbar":"sbar",
        }
        
        Fortran_FloatFields_IJK = {
            name: field.reshape(SHAPE[0]*SHAPE[1], SHAPE[2])
            for name, field in FloatFieldsIJK.items()
        }

        result = self.fortran_script.mode_condensation.condensation(                         
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
            **logical_keys,
            **constant_def,
            **self.fortran_shapes,
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
        
        logging.info(f"Temporary outputs")
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

        logging.info(f"\n Ouput fields")
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
        
    def test_sigrc_computation(self):
        
        
        logging.info(f"HLAMBDA3 {self.phyex_externals['LAMBDA3']}")

        sigrc_computation = compile_stencil("sigrc_diagnostic", self.gt4py_config, self.phyex_externals)
        
        IJK_Fields_Names = ["q1", "sigrc"]
        IJK_Fields = {
            name: np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            ) for name in IJK_Fields_Names
        }
        
        GT4Py_IJK_Fields = {
            name: from_array(
            field,
            dtype=np.float32,
            backend=self.gt4py_config.backend
        ) for name, field in IJK_Fields.items()
        }
        
        inq1_gt4py = ones(
            (SHAPE[0], SHAPE[1], SHAPE[2]), 
            dtype=np.int32, 
            backend=self.gt4py_config.backend)
        src_1d_gt4py = from_array(
            SRC_1D,
            dtype=np.float32,
            backend=self.gt4py_config.backend
        )
        
        sigrc_computation(
            src_1d=src_1d_gt4py,
            inq1=inq1_gt4py,
            **GT4Py_IJK_Fields
        )
        
        F2GT4Py_Keys = {
            "zq1": "q1",
            "psigrc": "sigrc"
        }
        
        IJK_FFields = {
            fortran_name: IJK_Fields[gt4py_name].reshape(SHAPE[0]*SHAPE[1], SHAPE[2])
            for fortran_name, gt4py_name in F2GT4Py_Keys.items()
        }

        result = self.fortran_script.mode_condensation.sigrc_computation(
            inq1=np.ones((SHAPE[0]*SHAPE[1], SHAPE[2])),
            hlambda3=self.phyex_externals["LAMBDA3"],
            **IJK_FFields,
            **self.fortran_shapes
        )
        
        psigrc_out = result[0]  
        inq1_out = result[1]
        
        logging.info("\n Temporaries")
        logging.info(f"Mean inq1_gt4py    {inq1_gt4py.mean()}")
        logging.info(f"Mean inq1_out      {inq1_out.mean()}")
    
        logging.info("\n Outputs")
        logging.info(f"Machine precision {np.finfo(float).eps}")
        logging.info(f"Mean sigrc_gt4py     {GT4Py_IJK_Fields["sigrc"].mean()}")
        logging.info(f"Mean psigrc_out      {psigrc_out.mean()}")

        assert_allclose(psigrc_out, GT4Py_IJK_Fields["sigrc"].reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), rtol=1e-6)

        
    def test_global_table(self):
        
        global_table = np.ones((34), dtype=np.float32)
        global_table_out = self.fortran_script.mode_condensation.global_table(out_table=global_table)
        
        logging.info(f"GlobalTable[0] : {global_table_out[0]}")
        logging.info(f"GlobalTable[5] : {global_table_out[5]}")
        logging.info(f"GlobalTable[33] : {global_table_out[33]}")
        
        assert_allclose(global_table_out, SRC_1D, rtol=1e-5)