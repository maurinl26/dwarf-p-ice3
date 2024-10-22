# -*- coding: utf-8 -*-
import logging
import sys
from functools import cached_property

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from repro.generic_test_component import TestComponent
from pathlib import Path
import fmodpy

from ice3_gt4py.phyex_common.phyex import Phyex

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


class MutliplyAB2C(TestComponent):

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
            computational_grid=computational_grid,
            gt4py_config=gt4py_config,
            fortran_script=fortran_script,
            fortran_module=fortran_module,
            fortran_subroutine=fortran_subroutine,
            gt4py_stencil=gt4py_stencil,
            phyex=phyex,
        )

    @cached_property
    def externals(self):
        """Filter phyex externals"""
        return {}

    @cached_property
    def dims(self) -> dict:
        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
        nijt = nit * njt
        return {
            "nijt": nijt,
            "nkt": nkt,
            "nkte": 1,
            "nktb": nkt,
            "nijb": 1,
            "nije": nijt,
        }

    @cached_property
    def array_shape(self) -> dict:
        return (int(self.dims["nijt"]), int(self.dims["nkt"]))

    @cached_property
    def fields_in(self):
        return {
            "a": {"grid": (I, J, K), "dtype": "float", "fortran_name": "a"},
            "b": {"grid": (I, J, K), "dtype": "float", "fortran_name": "b"},
        }

    @cached_property
    def fields_out(self):
        return {
            "c": {"grid": (I, J, K), "dtype": "float", "fortran_name": "c"},
        }

    @cached_property
    def fields_inout(self):
        return {}
    
    def compile_fortran_stencil(self, fortran_script, fortran_module, fortran_subroutine):
        
        current_directory = Path.cwd()
        logging.info(f"Root directory {current_directory}")
        root_directory = current_directory

        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        
        self.fortran_stencil = fmodpy.fimport(script_path).mode_test.multiply_ab2c
    
