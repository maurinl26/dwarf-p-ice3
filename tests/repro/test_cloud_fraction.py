# -*- coding: utf-8 -*-
import logging
from functools import cached_property
import unittest

from ifs_physics_common.framework.grid import I, J, K

from utils.fields_allocation import run_test
from ice3_gt4py.components.generic_test_component import TestComponent
from repro.default_config import default_gt4py_config, default_epsilon, phyex, test_grid


class CloudFraction(TestComponent):
    @cached_property
    def externals(self):
        """Filter phyex externals"""

        mapping = {
            "lsubg_cond": "LSUBG_COND",
            "xcriautc": "CRIAUTC",
            "csubg_mf_pdf": "SUBG_MF_PDF",
            "xcriauti": "CRIAUTI",
            "xacriauti": "ACRIAUTI",
            "xbcriauti": "BCRIAUTI",
            "xtt": "TT",
        }

        return {key: self.phyex_externals[value] for key, value in mapping.items()}

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
            "ptstep": 1,
        }

    @cached_property
    def array_shape(self) -> dict:
        return (int(self.dims["nijt"]), int(self.dims["nkt"]))

    @cached_property
    def fields_in(self):
        return {
            "rhodref": {
                "grid": (I, J, K),
                "dtype": "float",
                "fortran_name": "prhodref",
            },
            "exnref": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pexnref"},
            "rc": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prc"},
            "ri": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pri"},
            "rc_mf": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prc_mf"},
            "ri_mf": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pri_mf"},
            "cf_mf": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pcf_mf"},
            "rc_tmp": {"grid": (I, J, K), "dtype": "float", "fortran_name": "zrc"},
            "ri_tmp": {"grid": (I, J, K), "dtype": "float", "fortran_name": "zri"},
        }

    @cached_property
    def fields_out(self):
        return {
            "cldfr": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pcldfr"},
            "hlc_hrc": {
                "grid": (I, J, K),
                "dtype": "float",
                "fortran_name": "phlc_hrc",
            },
            "hlc_hcf": {
                "grid": (I, J, K),
                "dtype": "float",
                "fortran_name": "phlc_hcf",
            },
            "hli_hri": {
                "grid": (I, J, K),
                "dtype": "float",
                "fortran_name": "phli_hri",
            },
            "hli_hcf": {
                "grid": (I, J, K),
                "dtype": "float",
                "fortran_name": "phli_hcf",
            },
        }

    @cached_property
    def fields_inout(self):
        return {
            "ths": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pths"},
            "rvs": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prvs"},
            "rcs": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prcs"},
            "ris": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pris"},
        }


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
