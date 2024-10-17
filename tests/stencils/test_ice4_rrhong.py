# -*- coding: utf-8 -*-
import datetime
import logging
from functools import cached_property, partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Tuple

import fmodpy
import numpy as np
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol, I, J, K
from ifs_physics_common.framework.storage import allocate_data_array
from ifs_physics_common.utils.typingx import DataArray, NDArrayLikeDict

import ice3_gt4py.stencils
from ice3_gt4py.initialisation.utils import initialize_field
from ice3_gt4py.phyex_common.phyex import Phyex
from ice3_gt4py.utils.allocate import allocate
from stencils.generic_test_component import TestComponent


##### For tests ####
def allocate_state_ice4_rrhong(
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
        "ldcompute": allocate_b(),
        "exn": allocate_f(),
        "ls_fact": allocate_f(),
        "lv_fact": allocate_f(),
        "t": allocate_f(),
        "tht": allocate_f(),
        "rr_t": allocate_f(),
        "rrhong_mr": allocate_f(),
    }


class Ice4RRHONG(TestComponent):
    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
        phyex: Phyex,
        fortran_module: str,
        fortran_subroutine: str,
        fortran_script: str,
        gt4py_stencil: str,
    ) -> None:
        super().__init__(
            computational_grid=computational_grid, 
            gt4py_config=gt4py_config,
            phyex=phyex,
            fortran_script=fortran_script,
            fortran_module=fortran_module,
            fortran_subroutine=fortran_subroutine,
            gt4py_stencil=gt4py_stencil
        )

    @cached_property
    def dims(self):
        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
        kproma = nit * njt * nkt
        ksize = kproma
        return {"kproma": kproma, "ksize": ksize}
    
    @cached_property
    def array_shape(self):
        return (self.dims["ksize"], 1)

    @cached_property
    def fields_in(self):
        return {
            "ldcompute": {"grid": (I, J, K), "dtype": "bool", "fortran_name": "ldcompute"},
            "exn": {"grid": (I, J, K), "dtype": "bool", "fortran_name": "pexn"},
            "lvfact": {"grid": (I, J, K), "dtype": "bool", "fortran_name": "plvfact"},
            "lsfact": {"grid": (I, J, K), "dtype": "bool", "fortran_name": "plsfact"},
            "t": {"grid": (I, J, K), "dtype": "bool", "fortran_name": "pt"},
            "rrt": {"grid": (I, J, K), "dtype": "bool", "fortran_name": "prrt"},
            "tht": {"grid": (I, J, K), "dtype": "bool", "fortran_name": "ptht"},
        }
        
    @cached_property
    def fields_inout(self):
        return {}

    @cached_property
    def fields_out(self):
        return {"rrhong_mr": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prrhong_mr"}}
    
    def call_fortran_stencil(self, fields: dict):
        
        ldcompute = np.ones((self.dims["ksize"]))
        fields.update({"ldcompute": ldcompute})
        
        raveled_fields = dict()
        for name, array in fields.items():
            raveled_fields.update({name: array.reshape(1, -1).ravel()})
        
        return super().call_fortran_stencil(raveled_fields)

