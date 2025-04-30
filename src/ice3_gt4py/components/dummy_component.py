# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from datetime import timedelta
from functools import cached_property
from typing import Dict

from gt4py.storage import from_array

from ifs_physics_common.framework.components import DiagnosticComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from ifs_physics_common.utils.typingx import NDArrayLikeDict, PropertyDict

import sys

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


class DummyComponent(DiagnosticComponent):
    """Implicit Tendency Component for experiments"""

    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
        *,
        enable_checks: bool = True,
    ) -> None:
        super().__init__(
            computational_grid, 
            enable_checks=enable_checks, 
            gt4py_config=gt4py_config
        )

        # aro_filter stands for the parts before 'call ice_adjust' in aro_adjust.f90
        self.multiply_ab2c = self.compile_stencil("multiply_ab2c")


    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "a": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "b": {"grid": (I, J, K), "dtype": "float", "unit": ""}
        }

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        return {
            "c": {"grid": (I, J, K), "dtype": "float", "unit": ""},
        }

    def array_call(
        self,
        state: NDArrayLikeDict,
        timestep: timedelta,
        out_diagnostics: NDArrayLikeDict,
        overwrite_tendencies: Dict[str, bool],
    ) -> None:
        
        
        inputs = {
            name: state[name]
            for name in self._input_properties
        }
        
        diagnostics = {
            name: out_diagnostics[name]
            for name in self._diagnostic_properties
        }
        
        self.multiply_ab2c(
            **inputs,
            **diagnostics,
            origin=(0, 0, 0),
            domain=self.computational_grid.grids[I, J, K].shape,
            validate_args=self.gt4py_config.validate_args,
            exec_info=self.gt4py_config.exec_info,
        )
        
