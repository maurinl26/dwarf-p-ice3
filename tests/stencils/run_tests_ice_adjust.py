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
from stencils.fields_allocation import run_test

###### Default config for tests #######
backend = "gt:cpu_ifirst"
rebuild = True
validate_args = True

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
        
    def test_reproductibility(self):
        absolute_differences = run_test(self.component)
        
        for field, diff in absolute_differences.items():
            logging.info(f"{field}")
            self.assertLess(diff, 10e-6)   
            
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

        
    def test_reproductibility(self):
        absolute_differences = run_test(self.component)
        
        for field, diff in absolute_differences.items():
            logging.info(f"{field}")
            self.assertLess(diff, 10e-6) 
            
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
        
    def test_reproductibility(self):
        absolute_differences = run_test(self.component)
        
        for field, diff in absolute_differences.items():
            logging.info(f"{field}")
            self.assertLess(diff, 10e-6) 
    

if __name__ == "__main__":
    unittest.main()
