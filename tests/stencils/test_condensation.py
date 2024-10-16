# -*- coding: utf-8 -*-
from functools import cached_property, partial
import fmodpy
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.grid import ComputationalGrid
import numpy as np
import logging
import sys
from pathlib import Path
from ice3_gt4py.initialisation.utils import initialize_field
from ice3_gt4py.phyex_common.phyex import Phyex
from gt4py.storage import from_array

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.utils.typingx import (
    DataArray,
)

from ice3_gt4py.utils.allocate import allocate
from ice3_gt4py.utils.doctor_norm import field_doctor_norm

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()

class Condensation(ComputationalGridComponent):
    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
        phyex: Phyex,
        fortran_subroutine: str,
        fortran_script: str,
        fortran_module: str,
        gt4py_stencil: str,
    ) -> None:
        super().__init__(
            computational_grid=computational_grid, gt4py_config=gt4py_config
        )
        self.phyex_externals = phyex.to_externals()

        self.compile_fortran_stencil(
            fortran_module=fortran_module,
            fortran_script=fortran_script,
            fortran_subroutine=fortran_subroutine
        )
        
        logging.info(f"{self.phyex_externals['LSUBG_COND'], self.phyex_externals['SUBG_MF_PDF']}")
        
        
        self.compile_gt4py_stencil(gt4py_stencil, self.phyex_externals)

    @cached_property
    def externals(self):
        """Filter phyex externals"""
        return {
            "xrv": self.phyex_externals["RV"], 
            "xrd": self.phyex_externals["RD"], 
            "xalpi": self.phyex_externals["ALPI"], 
            "xbetai": self.phyex_externals["BETAI"], 
            "xgami": self.phyex_externals["GAMI"], 
            "xalpw": self.phyex_externals["ALPW"],
            "xbetaw": self.phyex_externals["BETAW"], 
            "xgamw": self.phyex_externals["GAMW"],
            "hcondens": self.phyex_externals["CONDENS"], 
            "hlambda3": self.phyex_externals["LAMBDA3"],
            "lstatnw": self.phyex_externals["LSTATNW"],
            "ouseri": 0,
            "osigmas": 0,
            "ocnd2": 0
        }

    @cached_property
    def dims(self) -> dict:
        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
        nijt = nit * njt
        return {
            "nijt": nijt, 
            "nkt": nkt,
            "nkte": 0,
            "nktb": nkt - 1,
            "nijb": 0,
            "nije": nijt - 1,
            }

    @cached_property
    def array_shape(self) -> dict:
        return (int(self.dims["nijt"]), int(self.dims["nkt"]))

    @cached_property
    def fields_in(self):
        return {
            "sigqsat": {"grid": (I, J, K), "dtype": "float", "fortran_name": "psigqsat"},
    "pabs": {"grid": (I, J, K), "dtype": "float", "fortran_name": "ppabs"},
    "sigs": {"grid": (I, J, K), "dtype": "float", "fortran_name": "psigs"},
    "t": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pt"},
    "rv_in": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prv_in"},
    "ri_in": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pri_in"},
    "rc_in": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prc_in"},
    "cph": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pcph"},
    "lv": {"grid": (I, J, K), "dtype": "float", "fortran_name": "plv"},
    "ls": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pls"},
        }

    @cached_property
    def fields_out(self):
        return {"rv_out": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prv_out"},
    "rc_out": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prc_out"},
    "ri_out": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pri_out"},
    "cldfr": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pcldfr"},
    "sigrc": {"grid": (I, J, K), "dtype": "float", "fortran_name": "psigrc"},}

    @cached_property
    def fields_inout(self):
        return {}

    #### Compilations ####
    def compile_gt4py_stencil(self, gt4py_stencil: str, externals: dict):
        """Compile GT4Py script given

        Args:
            gt4py_stencil (str): _description_
            externals (dict): _description_
        """
        self.gt4py_stencil = self.compile_stencil(gt4py_stencil, externals)

    def compile_fortran_stencil(self, fortran_script, fortran_module, fortran_subroutine):
        current_directory = Path.cwd()
        logging.info(f"Root directory {current_directory}")
        root_directory = current_directory
        
        stencils_directory = Path(root_directory, "src", "ice3_gt4py", "stencils_fortran")
        script_path = Path(stencils_directory, fortran_script)
        
        logging.info(f"Fortran script path {script_path}")
        self.fortran_script = fmodpy.fimport(script_path)
        fortran_module = getattr(self.fortran_script, fortran_module)
        self.fortran_stencil = getattr(fortran_module, fortran_subroutine)

    
    ##### Calls #####
    def call_fortran_stencil(self, fields: dict):
        """Call fortran stencil on a given field dict

        externals and dims are handled by component attributes itself

        Args:
            fields (dict): dictionnary of numpy arrays
        """
        field_attributes = {**self.fields_in, **self.fields_out, **self.fields_inout}
        state_fortran = dict()
        for key, field in fields.items():
            fortran_name = field_attributes[key]["fortran_name"]
            state_fortran.update({
                fortran_name: field
            })       

        output_fields_tuple = self.fortran_stencil(**state_fortran, **self.dims, **self.externals)
        
        output_fields = dict()
        keys = list({**self.fields_inout, **self.fields_out}.keys())
        for field in output_fields_tuple:
            output_fields.update({keys.pop(0): field})
        
        return output_fields
    

    def call_gt4py_stencil(self, fields: dict):
        """Call gt4py_stencil from a numpy array"""
        state_gt4py = dict() 
        for key, array in fields.items():
            state_gt4py.update({
                key: from_array(array)
                })
        self.gt4py_stencil(**state_gt4py)
        
        return state_gt4py

    