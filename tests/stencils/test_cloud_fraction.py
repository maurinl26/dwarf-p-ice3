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

class CloudFraction(ComputationalGridComponent):
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
        
        logging.info(f"{self.phyex_externals['SUBG_COND'], self.phyex_externals['SUBG_MF_PDF']}")
        
        externals_gt4py = dict()
        for key in ["SUBG_COND", "CRIAUTC", "SUBG_MF_PDF", "CRIAUTI", "ACRIAUTI", "BCRIAUTI", "TT"]:
            externals_gt4py.update({
                key: self.phyex_externals[key]
            })
        
        self.compile_gt4py_stencil(gt4py_stencil, externals_gt4py)

    @cached_property
    def externals(self):
        """Filter phyex externals"""
        return {
            "lsubg_cond": self.phyex_externals["SUBG_COND"],
            "xcriautc":self.phyex_externals["CRIAUTC"],
            "csubg_mf_pdf": self.phyex_externals["SUBG_MF_PDF"],
            "xcriauti": self.phyex_externals["CRIAUTI"],
            "xacriauti": self.phyex_externals["ACRIAUTI"],
            "xbcriauti": self.phyex_externals["BCRIAUTI"],
            "xtt": self.phyex_externals["TT"],
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
            "ptstep": 1
            }

    @cached_property
    def array_shape(self) -> dict:
        return (int(self.dims["nijt"]), int(self.dims["nkt"]))

    @cached_property
    def fields_in(self):
        return {
            "rhodref": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prhodref"},
            "exnref": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pexnref"},
            "rc": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prc"},
            "ri": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pri"},
            "rc_mf": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prc_mf"},
            "ri_mf": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pri_mf"},
            "cf_mf": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pcf_mf"},
            "rc_tmp": {"grid": (I, J, K), "dtype": "float", "fortran_name": "zrc"},
            "ri_tmp": {"grid": (I, J, K), "dtype": "float", "fortran_name": "zri"},
        }

    @cached_property
    def fields_out(self):
        return {
            "cldfr": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pcldfr"},
            "hlc_hrc": {"grid": (I, J, K), "dtype": "float", "fortran_name": "phlc_hrc"},
            "hlc_hcf": {"grid": (I, J, K), "dtype": "float", "fortran_name": "phlc_hcf"},
            "hli_hri": {"grid": (I, J, K), "dtype": "float", "fortran_name": "phli_hri"},
            "hli_hcf": {"grid": (I, J, K), "dtype": "float", "fortran_name": "phli_hcf"},
        }

    @cached_property
    def fields_inout(self):
        return {
            "ths": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pths"},
            "rvs": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prvs"},
            "rcs": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prcs"},
            "ris": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pris"},
        }

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

    