# -*- coding: utf-8 -*-
import logging
import sys
import unittest
from functools import cached_property
from pathlib import Path

import fmodpy
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from repro.generic_test_component import TestComponent
from utils.fields_allocation import run_test

from ice3_gt4py.phyex_common.phyex import Phyex

from repro.test_config import default_gt4py_config, test_grid, phyex, default_epsilon


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
    
class DoubleA(TestComponent):

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
        }

    @cached_property
    def array_shape(self) -> dict:
        return (int(self.dims["nijt"]), int(self.dims["nkt"]))

    @cached_property
    def fields_in(self):
        return {
            "a": {"grid": (I, J, K), "dtype": "float", "fortran_name": "a"},
        }

    @cached_property
    def fields_out(self):
        return {
            "c": {"grid": (I, J, K), "dtype": "float", "fortran_name": "c"},
        }

    @cached_property
    def fields_inout(self):
        return {}

class TestDoubleA(unittest.TestCase):
    
    def setUp(self):
        self.component = MutliplyAB2C(
        computational_grid=test_grid,
        phyex=phyex,
        gt4py_config=default_gt4py_config,
        fortran_script="mode_test.F90",
        fortran_module="mode_test",
        fortran_subroutine="double_a",
        gt4py_stencil="double_a",
    )
        
    def test_repro(self):
        """Assert mean absolute error on inout and out fields
        are less than epsilon
        """
        mean_absolute_errors = run_test(self.component)
        for field, diff in mean_absolute_errors.items():
            logging.info(f"Field name : {field}")
            logging.info(f"Epsilon {default_epsilon}")
            self.assertLess(diff, default_epsilon)  

class TestMultiplyAB2C(unittest.TestCase):
    
    def setUp(self):
        self.component = MutliplyAB2C(
        computational_grid=test_grid,
        phyex=phyex,
        gt4py_config=default_gt4py_config,
        fortran_script="mode_test.F90",
        fortran_module="mode_test",
        fortran_subroutine="multiply_ab2c",
        gt4py_stencil="multiply_ab2c",
    )
        
    def test_repro(self):
        """Assert mean absolute error on inout and out fields
        are less than epsilon
        """
        mean_absolute_errors = run_test(self.component)
        for field, diff in mean_absolute_errors.items():
            logging.info(f"Field name : {field}")
            logging.info(f"Epsilon {default_epsilon}")
            self.assertLess(diff, default_epsilon)     

            
def main(out = sys.stderr, verbosity = 2): 
    loader = unittest.TestLoader() 
  
    suite = loader.loadTestsFromModule(sys.modules[__name__]) 
    unittest.TextTestRunner(out, verbosity = verbosity).run(suite) 
      
if __name__ == '__main__': 
    with open('test_multiply_ab2c.out', 'w') as f: 
        main(f) 

