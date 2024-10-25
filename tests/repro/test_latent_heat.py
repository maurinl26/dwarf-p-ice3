# -*- coding: utf-8 -*-
import logging
import sys
from functools import cached_property

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from repro.generic_test_component import TestComponent

from ice3_gt4py.phyex_common.phyex import Phyex

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


class LatentHeat(TestComponent):

    @cached_property
    def externals(self):
        """Filter phyex externals"""
        return {
            "xlvtt": self.phyex_externals["LVTT"],
            "xlstt": self.phyex_externals["LSTT"],
            "xcl": self.phyex_externals["CL"],
            "xci": self.phyex_externals["CI"],
            "xtt": self.phyex_externals["TT"],
            "xcpv": self.phyex_externals["CPV"],
            "xcpd": self.phyex_externals["CPD"],
            "krr": self.phyex_externals["NRR"],
        }

    @cached_property
    def dims(self) -> dict:
        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
        nijt = nit * njt
        return {
            "nijt": nijt,
            "nkt": nkt,
            "nkte": 0,
            "nktb": nkt - 1,
            "nijb": 0,
            "nije": nijt - 1,
        }

    @cached_property
    def array_shape(self) -> dict:
        return (int(self.dims["nijt"]), int(self.dims["nkt"]))

    @cached_property
    def fields_in(self):
        return {
            "th": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pth"},
            "exn": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pexn"},
            "rv": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prv_in"},
            "rc": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prc_in"},
            "ri": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pri_in"},
            "rs": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prs"},
            "rr": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prr"},
            "rg": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prg"},
        }

    @cached_property
    def fields_out(self):
        return {
            "t": {"grid": (I, J, K), "dtype": "float", "fortran_name": "zt"},
            "lv": {"grid": (I, J, K), "dtype": "float", "fortran_name": "zlv"},
            "ls": {"grid": (I, J, K), "dtype": "float", "fortran_name": "zls"},
            "cph": {"grid": (I, J, K), "dtype": "float", "fortran_name": "zcph"},
        }

    @cached_property
    def fields_inout(self):
        return {}
    

    def call_gt4py_stencil(self, fields: dict):
        """Call gt4py_stencil from a numpy array"""
        # Overriden method to debug call
        
        for key, value in self.externals.items():
            logging.info(f"External {key}, value : {value}")
        
        return super().call_gt4py_stencil(fields)
