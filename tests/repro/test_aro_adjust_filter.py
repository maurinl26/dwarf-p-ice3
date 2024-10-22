# -*- coding: utf-8 -*-
import logging
import sys
from functools import cached_property
from pathlib import Path

import fmodpy
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from repro.generic_test_component import TestComponent

from ice3_gt4py.phyex_common.phyex import Phyex

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


class AroFilter(TestComponent):
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
            "exnref": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pexnref"},
            "tht": {"grid": (I, J, K), "dtype": "float", "fortran_name": "ptht"},
            "ths": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pths"},
            "rcs": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prcs"},
            "rrs": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prrs"},
            "ris": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pris"},
            "rvs": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prvs"},
            "rgs": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prgs"},
            "rss": {"grid": (I, J, K), "dtype": "float", "fortran_name": "prss"},
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
