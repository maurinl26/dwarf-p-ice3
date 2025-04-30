import fmodpy
import logging
from pathlib import Path

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

from tests.conftest import BACKEND, REBUILD, VALIDATE_ARGS, SHAPE

class TestTranscendentalFunctions(unittest.TestCase):
    
    def setUp(self):
        self.fortran_shapes = {
            "nijb": 1,
            "nije": SHAPE[0] * SHAPE[1],
            "nijt": SHAPE[0] * SHAPE[1],
            "nktb": 1,
            "nkte": SHAPE[2],
            "nkt": SHAPE[2],
        }
        
        logging.info(f"With backend {BACKEND}")
        self.gt4py_config = GT4PyConfig(
            backend=BACKEND, 
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
        fortran_script = "mode_sat_mixing_ratio.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)
        
        header = Path(stencils_directory, "c_log.h")
        c_transcendental = Path(stencils_directory, "c_log.F90")

        logging.info(f"Fortran script path {script_path}")
        self.fortran_script = fmodpy.fimport(script_path, dependencies=[header, c_transcendental])
        

    
    def test_saturation_mixing_ratio(self):
        
        sat = compile_stencil("saturation_mixing_ratio", self.gt4py_config, self.phyex_externals)
        
        FloatFieldsIJK_Names = ["pv", "piv", "t"]
        FloatFieldsIJK = {
            name: np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            ) for name in FloatFieldsIJK_Names
        }
        
        FloatFieldsIJK["t"] += 300
        
        GT4Py_FloatFieldsIJK = {
            name: from_array(
            field,
            dtype=np.float32,
            backend=BACKEND
        ) for name, field in FloatFieldsIJK.items()
        }
        
        constant_def = {
            "xalpi":self.phyex_externals["ALPI"], 
            "xbetai":self.phyex_externals["BETAI"], 
            "xgami":self.phyex_externals["GAMI"], 
            "xalpw":self.phyex_externals["ALPW"], 
            "xbetaw":self.phyex_externals["BETAW"], 
            "xgamw":self.phyex_externals["GAMW"],
        }
        
        Py2F_Mapping = {
            "t": "pt",
            "pv": "zpv",
            "piv": "zpiv",
        }
        
        Fort_FloatFieldsIJK = {
            Py2F_Mapping[name]: field.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]) 
            for name, field in FloatFieldsIJK.items()
        }
        
        for name, value in constant_def.items():
            logging.info(f"{name}, {value}")
            
        logging.info(f"t, shape  {GT4Py_FloatFieldsIJK['t'].shape}")
        logging.info(f"pt, shape {Fort_FloatFieldsIJK['pt'].shape}")
        
        logging.info(f"In GT4Py     {GT4Py_FloatFieldsIJK['t'].mean()}")
        logging.info(f"In Fortran   {Fort_FloatFieldsIJK['pt'].mean()}")
        ###### GT4Py call ####
        sat(
            **GT4Py_FloatFieldsIJK
        )
      
        #### Fortran call ####
        result = (
            self.fortran_script
            .mode_transcendental_functions
            .saturation_mixing_ratio(                         
            **Fort_FloatFieldsIJK,
            **constant_def,
            **self.fortran_shapes,
            )
        )
        
        pv_out = result[0]
        piv_out = result[1]
       
        
        logging.info(f"Temporary outputs")
        logging.info(f"Mean pv_gt4py      {GT4Py_FloatFieldsIJK["pv"].mean()}")
        logging.info(f"Mean pv_out        {pv_out.mean()}")

        logging.info(f"Mean piv_gt4py     {GT4Py_FloatFieldsIJK["piv"].mean()}")
        logging.info(f"Mean piv_out       {piv_out.mean()}")
     
        assert_allclose(pv_out, GT4Py_FloatFieldsIJK["pv"].reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(piv_out, GT4Py_FloatFieldsIJK["piv"].reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)

