# -*- coding: utf-8 -*-
from functools import cached_property, partial
import fmodpy
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
from ifs_physics_common.framework.storage import allocate_data_array
import numpy as np
import logging
import sys
from pathlib import Path
from ice3_gt4py.phyex_common.phyex import Phyex



from typing import Literal, Tuple
from ifs_physics_common.utils.typingx import (
    DataArray,
    DataArrayDict,
    NDArrayLikeDict,
)

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()

def allocate_state_latent_heat(
    computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig
) -> NDArrayLikeDict:
    """Allocate field to state keys following type (float, int, bool) and dimensions (2D, 3D).

    Args:
        computational_grid (ComputationalGrid): grid indexes
        gt4py_config (GT4PyConfig): gt4py configuration

    Returns:
        NDArrayLikeDict: dictionnary of field with associated keys for field name
    """

    def _allocate(
        grid_id: Tuple[DimSymbol, ...],
        units: str,
        dtype: Literal["bool", "float", "int"],
    ) -> DataArray:
        return allocate_data_array(
            computational_grid, grid_id, units, gt4py_config=gt4py_config, dtype=dtype
        )

    allocate_b_ij = partial(_allocate, grid_id=(I, J), units="", dtype="bool")
    allocate_b = partial(_allocate, grid_id=(I, J, K), units="", dtype="bool")
    allocate_f = partial(_allocate, grid_id=(I, J, K), units="", dtype="float")
    allocate_h = partial(_allocate, grid_id=(I, J, K - 1 / 2), units="", dtype="float")
    allocate_ij = partial(_allocate, grid_id=(I, J), units="", dtype="float")
    allocate_i_ij = partial(_allocate, grid_id=(I, J), units="", dtype="int")

    return {
        "th": allocate_f(),
        "exn": allocate_f(),
        "rv": allocate_f(),
        "rc": allocate_f(),
        "ri": allocate_f(),
        "rs": allocate_f(),
        "rr": allocate_f(),
        "rg": allocate_f(),
        "t": allocate_f(),
        "lv": allocate_f(),
        "ls": allocate_f(),
        "cph": allocate_f()
    }


class LatentHeat(ComputationalGridComponent):
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
            fortran_subroutine=fortran_subroutine,
        )

        self.compile_gt4py_stencil(gt4py_stencil, self.phyex_externals)

    @cached_property
    def externals(self):
        """Filter phyex externals"""
        return {
            "xlvtt": self.phyex_externals["LVTT"],
            "xlstt": self.phyex_externals["LSTT"],
            "xcl": self.phyex_externals["CL"],
            "xci": self.phyex_externals["CI"],
            "xtt": self.phyex_externals["TT"],
            "xcpv": self.phyex_externals["CPV"],
            "xcpd": self.phyex_externals["CPD"],
            "krr": self.phyex_externals["NRR"],
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
            "th": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pth"},
            "exn": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pexn"},
            "rv": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prv_in"},
            "rc": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prc_in"},
            "ri": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pri_in"},
            "rs": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prs"},
            "rr": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prr"},
            "rg": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prg"},
        }

    @cached_property
    def fields_out(self):
        return {
            "t": {"grid": (I, J, K), "dtype": "float", "fortran_name": "zt"},
            "lv": {"grid": (I, J, K), "dtype": "float", "fortran_name": "zlv"},
            "ls": {"grid": (I, J, K), "dtype": "float", "fortran_name": "zls"},
            "cph": {"grid": (I, J, K), "dtype": "float", "fortran_name": "zcph"},
        }

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

    def compile_fortran_stencil(
        self, fortran_script, fortran_module, fortran_subroutine
    ):
        current_directory = Path.cwd()
        logging.info(f"Root directory {current_directory}")
        root_directory = current_directory

        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
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
            state_fortran.update({fortran_name: field})

        output_fields_tuple = self.fortran_stencil(
            **state_fortran, **self.dims, **self.externals
        )

        output_fields = dict()
        keys = list({**self.fields_inout, **self.fields_out}.keys())
        for field in output_fields_tuple:
            output_fields.update({keys.pop(0): field})

        return output_fields

    def call_gt4py_stencil(self, fields: dict):
        """Call gt4py_stencil from a numpy array"""
        self.gt4py_stencil(**fields)
        return fields
