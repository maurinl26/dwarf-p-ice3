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
        externals = phyex.to_externals()

        self.compile_fortran_stencil(
            fortran_module=fortran_module,
            fortran_script=fortran_script,
            fortran_subroutine=fortran_subroutine
        )
        #self.compile_gt4py_stencil(gt4py_stencil)

    @cached_property
    def externals(self, phyex_externals: dict):
        """Filter phyex externals"""
        return {
            "lsubg_cond": phyex_externals["SUBG_COND"],
            "xcriautc": phyex_externals["CRIAUTC"],
            "csubg_mf_pdf": phyex_externals["SUBG_MF_PDF"],
            "xcriauti": phyex_externals["CRIAUTI"],
            "xacriauti": phyex_externals["ACRIAUTI"],
            "xbcriauti": phyex_externals["BCRIAUTI"],
            "xtt": phyex_externals["TT"],
        }

    @cached_property
    def dims(self) -> dict:
        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
        return {"nijt": nit * njt, "nkt": nkt}

    @cached_property
    def array_shape(self) -> dict:
        return (self.dims["nijt"], self.dims["nkt"])

    @cached_property
    def fields_in(self):
        return {
            "lv": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "ls": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "t": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "cph": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "rhodref": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "exnref": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "rc": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "ri": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "rc_mf": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "ri_mf": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "cf_mf": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "rc_tmp": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "ri_tmp": {"grid": (I, J, K), "dtype": "float", "units": ""},
        }

    @cached_property
    def fields_out(self):
        return {
            "hlc_hrc": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "hlc_hcf": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "hli_hri": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "hli_hcf": {"grid": (I, J, K), "dtype": "float", "units": ""},
        }

    @cached_property
    def fields_inout(self):
        return {
            "ths": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "rvs": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "rcs": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "ris": {"grid": (I, J, K), "dtype": "float", "units": ""},
        }

    def compile_fortran_stencil(self, fortran_script, fortran_module, fortran_subroutine):
        current_directory = Path.cwd()
        logging.info(f"Root directory {current_directory}")
        root_directory = current_directory
        
        stencils_directory = Path(root_directory, "src", "ice3_gt4py", "stencils_fortran")
        script_path = Path(stencils_directory, fortran_script)
        
        logging.info(f"Fortran script path {script_path}")
        self.fortran_script = fmodpy.fimport(stencils_directory)
        fortran_module = getattr(self.fortran_script, fortran_module)
        self.fortran_stencil = getattr(fortran_module, fortran_subroutine)
        

    ##### Calls #####
    def call_fortran_stencil(self, fields: dict):
        """Call fortran stencil on a given field dict

        externals and dims are handled by component attributes itself

        Args:
            fields (dict): dictionnary of numpy arrays
        """
        output_fields = self.fortran_stencil(**fields, **self.dims, **self.externals)
        return output_fields

