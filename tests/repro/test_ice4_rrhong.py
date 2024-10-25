# -*- coding: utf-8 -*-
import logging
from functools import cached_property
import unittest

import numpy as np

from ifs_physics_common.framework.grid import I, J, K

import ice3_gt4py.stencils
from utils.fields_allocation import run_test
from utils.generic_test_component import TestComponent


class Ice4RRHONG(TestComponent):
    
    @cached_property
    def externals(self):
        return {
            "r_rtmin": self.phyex_externals["R_RTMIN"],
            "lfeedbackt": self.phyex_externals["LFEEDBACKT"],
            "xtt": self.phyex_externals["TT"]
        }

    @cached_property
    def dims(self):
        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
        kproma = nit * njt * nkt
        ksize = kproma
        return {"kproma": kproma, "ksize": ksize}
    
    @cached_property
    def array_shape(self):
        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
        kproma = nit * njt * nkt

        return nit*njt, nkt

    @cached_property
    def fields_in(self):
        return {
            "ldcompute": {"grid": (I, J, K), "dtype": "bool", "fortran_name": "ldcompute"},
            "exn": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pexn"},
            "lvfact": {"grid": (I, J, K), "dtype": "float", "fortran_name": "plvfact"},
            "lsfact": {"grid": (I, J, K), "dtype": "float", "fortran_name": "plsfact"},
            "t": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pt"},
            "rrt": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prrt"},
            "tht": {"grid": (I, J, K), "dtype": "float", "fortran_name": "ptht"},
        }
        
    @cached_property
    def fields_inout(self):
        return {}

    @cached_property
    def fields_out(self):
        return {"rrhong_mr": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prrhong_mr"}}
    
    def call_fortran_stencil(self, fields: dict):
        
        
        def packing(fields):
            """Reshaping fields from (nijt, nkt) dims 
            to (nijt*nkt, 1) dims

            Args:
                fields (_type_): _description_

            Returns:
                _type_: _description_
            """
            nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
            nijt = nit * njt
        
            # naïve unpacking, leaving one dimension on k
            return {
                key: field.reshape(nijt*nkt, 1) for key, field in fields.items()
            }
        
        # Handling packing
        new_fields = packing(fields)
        logging.info(f"Field keys : {new_fields.keys()}, new_fields shape : {new_fields['lsfact'].shape}")

        # Handling ldcompute
        ldcompute = np.ones((self.dims["kproma"], 1), dtype="int", order="F")
        new_fields.update({"ldcompute": ldcompute})
        
        return super().call_fortran_stencil(fields)

from repro.default_config import phyex, default_epsilon, test_grid, default_gt4py_config


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
            
    def test_ldcompute_false(self):
        
        # TODO : test
        # self.component.call_gt4py_stencil()
        ...
        
    def test_lfeedbackt_true(self):
        ...
        
    def test_lfeedbackt_false(self):
        ...