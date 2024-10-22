# -*- coding: utf-8 -*-
import datetime
import logging
import unittest

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from repro.test_cloud_fraction import CloudFraction
from repro.test_condensation import Condensation
from repro.test_latent_heat import LatentHeat

from ice3_gt4py.phyex_common.phyex import Phyex
from repro.test_test import MutliplyAB2C
from utils.fields_allocation import run_test
import sys
###### Default config for tests #######
backend = "gt:cpu_ifirst"
rebuild = True
validate_args = True
default_epsilon = 10e-6

phyex = Phyex(program="AROME")

test_grid = ComputationalGrid(50, 1, 15)
dt = datetime.timedelta(seconds=1)

default_gt4py_config = GT4PyConfig(
    backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=False
)

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
        
    def test_repro_latent_heat(self):
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
    with open('testing.out', 'w') as f: 
        main(f) 

