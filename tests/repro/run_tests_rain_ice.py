# -*- coding: utf-8 -*-
import datetime
import logging
import unittest

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from repro.test_ice4_rrhong import Ice4RRHONG

from ice3_gt4py.phyex_common.phyex import Phyex
from utils.fields_allocation import run_test

###### Default config for tests #######
backend = "gt:cpu_ifirst"
rebuild = True
validate_args = True

default_epsilon = 10e-6

phyex = Phyex(program="AROME")

test_grid = ComputationalGrid(50, 1, 1)
dt = datetime.timedelta(seconds=1)

default_gt4py_config = GT4PyConfig(
    backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=False
)

class TestIce4RRHONG(unittest.TestCase):
    
    def setUp(self):
        self.component = Ice4RRHONG(
            computational_grid=test_grid,
            gt4py_config=default_gt4py_config,
            phyex=phyex,
            fortran_script="mode_ice4_rrhong.F90",
            fortran_module="mode_ice4_rrhong",
            fortran_subroutine="ice4_rrhong",
            gt4py_stencil="ice4_rrhong"
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
        

if __name__ == "__main__":
    unittest.main()