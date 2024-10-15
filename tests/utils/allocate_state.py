# -*- coding: utf-8 -*-
import fmodpy
import numpy as np
import logging
from functools import cached_property
from typing import TYPE_CHECKING
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array
from ice3_gt4py.initialisation.utils import initialize_field
from ice3_gt4py.utils.allocate import allocate
from pathlib import Path

import datetime
from typing import Literal, 
from functools import partial
from ice3_gt4py.phyex_common.phyex import Phyex

from typing import Literal, Tuple
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
from ifs_physics_common.utils.typingx import (
    DataArray,
    DataArrayDict,
    NDArrayLikeDict,
)


class FieldAllocator:
    
    def __init__(self, computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig):
        
        self.computational_grid = computational_grid
        self.gt4py_config = gt4py_config

    def set_allocators(self):
        """Allocate zero array in storage given grid and type characteristics"""
        _allocate_on_computational_grid = partial(
            allocate,
            computational_grid=self.computational_grid,
            gt4py_config=self.gt4py_config,
        )
        
        self.allocate_b_ij = partial(
            _allocate_on_computational_grid, grid_id=(I, J), units="", dtype="bool"
        )
        self.allocate_b_ijk = partial(
            _allocate_on_computational_grid, grid_id=(I, J, K), units="", dtype="bool"
        )

        self.allocate_f_ij = partial(
            _allocate_on_computational_grid, grid_id=(I, J), units="", dtype="float"
        )
        self.allocate_f_ijk = partial(
            _allocate_on_computational_grid, grid_id=(I, J, K), units="", dtype="float"
        )
        self.allocate_f_ijk_h = partial(
            _allocate_on_computational_grid,
            grid_id=(I, J, K - 1 / 2),
            units="",
            dtype="float",
        )

        self.allocate_i_ij = partial(
            _allocate_on_computational_grid, grid_id=(I, J), units="", dtype="int"
        )
        self.allocate_i_ijk = partial(
            _allocate_on_computational_grid, grid_id=(I, J, K), units="", dtype="int"
        )

    def allocate_state(self, field_properties: dict):
        """Allocate GT4Py state"""
        self.set_allocators()
        
        state = dict()
        for field_key, field_attributes in field_properties.items():

            # 3D fields
            if field_attributes["dtype"] == "float" and field_attributes["grid"] == (
                I,
                J,
                K,
            ):
                state.update({field_key: self.allocate_f_ijk()})
            elif field_attributes["dtype"] == "bool" and field_attributes["grid"] == (
                I,
                J,
                K,
            ):
                state.update({field_key: self.allocate_b_ijk()})
            elif field_attributes["dtype"] == "int" and field_attributes["grid"] == (
                I,
                J,
                K,
            ):
                state.update({field_key: self.allocate_i_ijk()})

            # 2D fields
            if field_attributes["dtype"] == "float" and field_attributes["grid"] == (
                I,
                J,
            ):
                state.update({field_key: self.allocate_f_ij()})
            elif field_attributes["dtype"] == "bool" and field_attributes["grid"] == (
                I,
                J,
            ):
                state.update({field_key: self.allocate_b_ij()})
            elif field_attributes["dtype"] == "int" and field_attributes["grid"] == (
                I,
                J,
            ):
                state.update({field_key: self.allocate_i_ij()})

    def allocate_fields(self, fields: dict, field_properties: dict):
        """Allocate fields (as gt4py storage)

        Interface with fortran like fields
        """
        state = self.allocate_state(field_properties)
        for field_key, field_attributes in fields.items():
            initialize_field(state[field_key], fields[field_key][:, np.newaxis])
        return state
