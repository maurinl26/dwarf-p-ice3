# -*- coding: utf-8 -*-
import datetime
import logging
import unittest

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from stencils.test_cloud_fraction import CloudFraction
from stencils.test_condensation import Condensation
from stencils.test_latent_heat import LatentHeat

from ice3_gt4py.phyex_common.phyex import Phyex
from utils.fields_allocation import run_test

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

class TestLatentHeat(unittest.TestCase):
    
    def setUp(self):
        self.component = LatentHeat(
        computational_grid=test_grid,
        phyex=phyex,
        gt4py_config=default_gt4py_config,
        fortran_script="mode_thermo.F90",
        fortran_module="mode_thermo",
        fortran_subroutine="latent_heat",
        gt4py_stencil="thermodynamic_fields",
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
            
class TestCondensation(unittest.TestCase):
    
    def setUp(self):
        self.component = Condensation(
        computational_grid=test_grid,
        phyex=phyex,
        gt4py_config=default_gt4py_config,
        fortran_script="mode_condensation.F90",
        fortran_module="mode_condensation",
        fortran_subroutine="condensation",
        gt4py_stencil="condensation",
    )
  
    def test_repro_condensation(self):
        """Assert mean absolute error on inout and out fields
        are less than epsilon
        """
        mean_absolute_errors = run_test(self.component)
        for field, diff in mean_absolute_errors.items():
            logging.info(f"Field name : {field}")
            logging.info(f"Epsilon {default_epsilon}")
            self.assertLess(diff, default_epsilon)  
            
class TestCloudFraction(unittest.TestCase):
    
    def setUp(self):
        self.component = CloudFraction(
        computational_grid=test_grid,
        phyex=phyex,
        gt4py_config=default_gt4py_config,
        fortran_script="mode_cloud_fraction.F90",
        fortran_module="mode_cloud_fraction",
        fortran_subroutine="cloud_fraction",
        gt4py_stencil="cloud_fraction",
    )
        
    def test_repro_cloud_fraction(self):
        """Assert mean absolute error on inout and out fields
        are less than epsilon
        """
        mean_absolute_errors = run_test(self.component)
        for field, diff in mean_absolute_errors.items():
            logging.info(f"Field name : {field}")
            logging.info(f"Epsilon {default_epsilon}")
            self.assertLess(diff, default_epsilon)  
    

if __name__ == "__main__":
    import logging
    import sys 
    
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logging.getLogger()
    unittest.main()
