# -*- coding: utf-8 -*-
import logging
import unittest
import sys

from repro.test_ice4_rrhong import Ice4RRHONG

from utils.fields_allocation import run_test
from repro.test_config import phyex, default_epsilon, test_grid, default_gt4py_config


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
        
def main(out = sys.stderr, verbosity = 2): 
    loader = unittest.TestLoader() 
  
    suite = loader.loadTestsFromModule(sys.modules[__name__]) 
    unittest.TextTestRunner(out, verbosity = verbosity).run(suite) 
      
if __name__ == '__main__': 
    with open('test_ice_adjust.out', 'w') as f: 
        main(f) 