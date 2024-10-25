# -*- coding: utf-8 -*-
import logging
from functools import cached_property

import numpy as np
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K

import ice3_gt4py.stencils
from ice3_gt4py.phyex_common.phyex import Phyex
from repro.generic_test_component import TestComponent


class Ice4RRHONG(TestComponent):

    @cached_property
    def dims(self):
        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
        kproma = nit * njt * nkt
        ksize = kproma
        return {"kproma": kproma, "ksize": ksize}
    
    @cached_property
    def array_shape(self):
        return (self.dims["ksize"], 1)

    @cached_property
    def fields_in(self):
        return {
            "ldcompute": {"grid": (I, J, K), "dtype": "bool", "fortran_name": "ldcompute"},
            "exn": {"grid": (I, J, K), "dtype": "bool", "fortran_name": "pexn"},
            "lvfact": {"grid": (I, J, K), "dtype": "bool", "fortran_name": "plvfact"},
            "lsfact": {"grid": (I, J, K), "dtype": "bool", "fortran_name": "plsfact"},
            "t": {"grid": (I, J, K), "dtype": "bool", "fortran_name": "pt"},
            "rrt": {"grid": (I, J, K), "dtype": "bool", "fortran_name": "prrt"},
            "tht": {"grid": (I, J, K), "dtype": "bool", "fortran_name": "ptht"},
        }
        
    @cached_property
    def fields_inout(self):
        return {}

    @cached_property
    def fields_out(self):
        return {"rrhong_mr": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prrhong_mr"}}
    
    def call_fortran_stencil(self, fields: dict):
        
        # Reshape dimension (single lign)
        state_fortran = dict()
        for key, array in fields.items():
            state_fortran.update({key: array.reshape(-1)})
        
        # Add ldcompute as an integer
        ldcompute = np.ones((self.dims["ksize"]), dtype=np.int32, order="F")
        state_fortran.update({"ldcompute": ldcompute}) 
        
        for field_name, array in state_fortran.items():
            logging.info(f"Fortran field name {field_name}, array shape {array.shape}, array type {type(array)}")
                
        return super().call_fortran_stencil(state_fortran)

