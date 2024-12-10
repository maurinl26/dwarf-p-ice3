# -*- coding: utf-8 -*-
import logging
from functools import cached_property
import unittest

from ifs_physics_common.framework.grid import I, J, K
from utils.fields_allocation import (
    allocate_gt4py_fields,
    compare_input,
    compare_output,
    draw_fields,
    run_test,
)
from ice3_gt4py.components.generic_test_component import TestComponent

from repro.default_config import default_epsilon, default_gt4py_config, test_grid, phyex


class LatentHeat(TestComponent):
    @cached_property
    def externals(self):
        """Mapping between Fortran externals and GT4Py externals"""
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


class TestLatentHeat(unittest.TestCase):
    def test_repro_latent_heat(self):
        """Assert mean absolute error on inout and out fields
        are less than epsilon
        """

        component = LatentHeat(
            computational_grid=test_grid,
            phyex=phyex,
            gt4py_config=default_gt4py_config,
            fortran_script="mode_thermo.F90",
            fortran_module="mode_thermo",
            fortran_subroutine="latent_heat",
            gt4py_stencil="thermodynamic_fields",
        )

        logging.info(f"\n Start test {component.__class__.__name__}")

        fields = draw_fields(component)
        state_gt4py = allocate_gt4py_fields(component, fields)

        logging.info(f"Compare input  fields")
        compare_input(component, fields, state_gt4py)

        logging.info("Calling fortran field")
        fortran_output_fields = component.call_fortran_stencil(fields)
        logging.info(f"fortran_output_fields {fortran_output_fields.keys()}")

        logging.info("Calling gt4py field")
        gt4py_output_fields = component.call_gt4py_stencil(state_gt4py)

        logging.info("Compare output fields")
        # Compare output is an assertion
        compare_output(
            component=component,
            fortran_fields=fortran_output_fields,
            gt4py_state=gt4py_output_fields,
        )
        logging.info(f"End test {component.__class__.__name__}\n")
