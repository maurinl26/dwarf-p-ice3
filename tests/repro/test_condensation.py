# -*- coding: utf-8 -*-
import logging
import sys
from functools import cached_property
from ice3_gt4py.phyex_common.tables import src_1d
from ifs_physics_common.framework.grid import I, J, K
import unittest

from tests.utils.fields_allocation import run_test
from ice3_gt4py.components.generic_test_component import TestComponent
from utils.allocate_state import allocate_state
from repro.default_config import test_grid, phyex, default_gt4py_config


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


class Condensation(TestComponent):
    @cached_property
    def externals(self):
        """Filter phyex externals"""
        return {
            "xrv": self.phyex_externals["RV"],
            "xrd": self.phyex_externals["RD"],
            "xalpi": self.phyex_externals["ALPI"],
            "xbetai": self.phyex_externals["BETAI"],
            "xgami": self.phyex_externals["GAMI"],
            "xalpw": self.phyex_externals["ALPW"],
            "xbetaw": self.phyex_externals["BETAW"],
            "xgamw": self.phyex_externals["GAMW"],
            "hcondens": self.phyex_externals["CONDENS"],
            "hlambda3": self.phyex_externals["LAMBDA3"],
            "lstatnw": self.phyex_externals["LSTATNW"],
            "ouseri": 0,
            "osigmas": 0,
            "ocnd2": 0,
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
            "sigqsat": {
                "grid": (I, J, K),
                "dtype": "float",
                "fortran_name": "psigqsat",
            },
            "pabs": {"grid": (I, J, K), "dtype": "float", "fortran_name": "ppabs"},
            "sigs": {"grid": (I, J, K), "dtype": "float", "fortran_name": "psigs"},
            "t": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pt"},
            "rv_in": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prv_in"},
            "ri_in": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pri_in"},
            "rc_in": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prc_in"},
            "cph": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pcph"},
            "lv": {"grid": (I, J, K), "dtype": "float", "fortran_name": "plv"},
            "ls": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pls"},
        }

    @cached_property
    def fields_out(self):
        return {
            "t_out": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pt_out"},
            "rv_out": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prv_out"},
            "rc_out": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prc_out"},
            "ri_out": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pri_out"},
            "cldfr": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pcldfr"},
            "sigrc": {"grid": (I, J, K), "dtype": "float", "fortran_name": "psigrc"},
        }

    @cached_property
    def fields_inout(self):
        return {}

    def call_gt4py_stencil(self, fields: dict):
        """Call gt4py_stencil from a numpy array"""

        inq1_field = {"inq1": {"grid": (I, J, K), "dtype": "int"}}
        state = allocate_state(self.computational_grid, self.gt4py_config, inq1_field)
        fields.update(state)
        fields.update({"src_1d": src_1d})

        return super().call_gt4py_stencil(fields)


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
