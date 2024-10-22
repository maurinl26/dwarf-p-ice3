# -*- coding: utf-8 -*-
import logging
import sys
from functools import cached_property

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K

from ice3_gt4py.phyex_common.phyex import Phyex
from repro.generic_test_component import TestComponent

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


class CloudFraction(TestComponent):
    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
        phyex: Phyex,
        fortran_subroutine: str,
        fortran_script: str,
        fortran_module: str,
        gt4py_stencil: str,
    ) -> None:
        super().__init__(
            computational_grid=computational_grid,
            gt4py_config=gt4py_config,
            phyex=phyex,
            fortran_script=fortran_script,
            fortran_module=fortran_module,
            fortran_subroutine=fortran_subroutine,
            gt4py_stencil=gt4py_stencil,
        )

    @cached_property
    def externals(self):
        """Filter phyex externals"""
        
        mapping  = {
            "lsubg_cond": "LSUBG_COND",
            "xcriautc": "CRIAUTC",
            "csubg_mf_pdf": "SUBG_MF_PDF",
            "xcriauti": "CRIAUTI",
            "xacriauti": "ACRIAUTI",
            "xbcriauti": "BCRIAUTI",
            "xtt": "TT"
        }
        
        return {
            key: self.phyex_externals[value] for key, value in mapping.items() 
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
