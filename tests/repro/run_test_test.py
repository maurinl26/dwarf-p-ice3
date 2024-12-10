# -*- coding: utf-8 -*-
import logging
import sys
import unittest
from functools import cached_property

from ifs_physics_common.framework.grid import I, J, K
from ice3_gt4py.components.generic_test_component import TestComponent
from utils.fields_allocation import run_test

from repro.default_config import default_gt4py_config, test_grid, phyex, default_epsilon


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()

########### Test Multiplication #################
class MutliplyAB2C(TestComponent):
    @cached_property
    def externals(self):
        """Filter phyex externals"""
        return {}

    @cached_property
    def dims(self) -> dict:
        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
        nijt = nit * njt
        return {
            "nijt": nijt,
            "nkt": nkt,
        }

    @cached_property
    def array_shape(self) -> dict:
        return (int(self.dims["nijt"]), int(self.dims["nkt"]))

    @cached_property
    def fields_in(self):
        return {
            "a": {"grid": (I, J, K), "dtype": "float", "fortran_name": "a"},
            "b": {"grid": (I, J, K), "dtype": "float", "fortran_name": "b"},
        }

    @cached_property
    def fields_out(self):
        return {
            "c": {"grid": (I, J, K), "dtype": "float", "fortran_name": "c"},
        }

    @cached_property
    def fields_inout(self):
        return {}


class DoubleA(TestComponent):
    @cached_property
    def externals(self):
        """Filter phyex externals"""
        return {}

    @cached_property
    def dims(self) -> dict:
        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
        nijt = nit * njt
        return {
            "nijt": nijt,
            "nkt": nkt,
        }

    @cached_property
    def array_shape(self) -> dict:
        return (int(self.dims["nijt"]), int(self.dims["nkt"]))

    @cached_property
    def fields_in(self):
        return {
            "a": {"grid": (I, J, K), "dtype": "float", "fortran_name": "a"},
        }

    @cached_property
    def fields_out(self):
        return {
            "c": {"grid": (I, J, K), "dtype": "float", "fortran_name": "c"},
        }

    @cached_property
    def fields_inout(self):
        return {}


############## Test Multioutput ########################
class Multioutput(TestComponent):
    @cached_property
    def externals(self):
        """Filter phyex externals"""
        return {}

    @cached_property
    def dims(self) -> dict:
        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
        nijt = nit * njt
        return {
            "nijt": nijt,
            "nkt": nkt,
        }

    @cached_property
    def array_shape(self) -> dict:
        return (int(self.dims["nijt"]), int(self.dims["nkt"]))

    @cached_property
    def fields_in(self):
        return {
            "a": {"grid": (I, J, K), "dtype": "float", "fortran_name": "a"},
        }

    @cached_property
    def fields_out(self):
        return {
            "b": {"grid": (I, J, K), "dtype": "float", "fortran_name": "b"},
            "c": {"grid": (I, J, K), "dtype": "float", "fortran_name": "c"},
        }

    @cached_property
    def fields_inout(self):
        return {}


############# 1D Fortran vs 3D GT4Py ###################
class MultiplyOneD(TestComponent):
    @cached_property
    def externals(self):
        """Filter phyex externals"""
        return {}

    @cached_property
    def dims(self) -> dict:
        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape

        # Here we map the packing operation
        nijkt = nit * njt * nkt
        return {
            "nijkt": nijkt,
        }

    @cached_property
    def array_shape(self) -> dict:
        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
        return nit * njt, nkt

    @cached_property
    def fields_in(self):
        return {
            "a": {"grid": (I, J, K), "dtype": "float", "fortran_name": "a"},
        }

    @cached_property
    def fields_out(self):
        return {
            "c": {"grid": (I, J, K), "dtype": "float", "fortran_name": "c"},
        }

    @cached_property
    def fields_inout(self):
        return {}

    def call_fortran_stencil(self, fields: dict):

        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
        nijt = nit * njt

        # naïve unpacking, leaving one dimension on k
        new_fields = {key: field.reshape(nijt, nkt) for key, field in fields.items()}

        return super().call_fortran_stencil(new_fields)


############## Test classes ############################


class TestDoubleA(unittest.TestCase):
    def setUp(self):
        self.component = DoubleA(
            computational_grid=test_grid,
            phyex=phyex,
            gt4py_config=default_gt4py_config,
            fortran_script="mode_test.F90",
            fortran_module="mode_test",
            fortran_subroutine="double_a",
            gt4py_stencil="double_a",
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

    def test_repro(self):
        """Assert mean absolute error on inout and out fields
        are less than epsilon
        """
        mean_absolute_errors = run_test(self.component)
        for field, diff in mean_absolute_errors.items():
            logging.info(f"Field name : {field}")
            logging.info(f"Epsilon {default_epsilon}")
            self.assertLess(diff, default_epsilon)


class TestMultioutput(unittest.TestCase):
    def setUp(self):
        self.component = Multioutput(
            computational_grid=test_grid,
            phyex=phyex,
            gt4py_config=default_gt4py_config,
            fortran_script="mode_test_multioutput.F90",
            fortran_module="mode_test_multioutput",
            fortran_subroutine="multioutput",
            gt4py_stencil="multioutput",
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


class TestMultiplyOneD(unittest.TestCase):
    def setUp(self):
        self.component = MultiplyOneD(
            computational_grid=test_grid,
            phyex=phyex,
            gt4py_config=default_gt4py_config,
            fortran_script="mode_test.F90",
            fortran_module="mode_test",
            fortran_subroutine="mutliply_oned_array",
            gt4py_stencil="double_a",
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


def main(out=sys.stderr, verbosity=2):
    loader = unittest.TestLoader()

    suite = loader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(out, verbosity=verbosity).run(suite)


if __name__ == "__main__":
    with open("test_test.md", "w") as f:
        main(f)
