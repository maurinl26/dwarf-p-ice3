# -*- coding: utf-8 -*-
import fmodpy
import numpy as np
import logging
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.grid import ComputationalGrid
import datetime

from ice3_gt4py.phyex_common.phyex import Phyex


class TestCondensation(ComputationalGridComponent):
    
    
    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
    ) -> None:
        super().__init__(
            computational_grid=computational_grid, gt4py_config=gt4py_config
        )

        externals = {
            "TT": XTT,
            "R_RTMIN": XRTMIN,
            "LFEEDBACKT": True,
        }

        # aro_filter stands for the parts before 'call ice_adjust' in aro_adjust.f90
        self.cloud_fraction = self.compile_stencil("cloud_fraction", externals)

        self.mode_cloud_fraction = fmodpy.fimport(
            "./src/ice3_gt4py/stencil_fortran/test_ice_adjust/cloud_fraction.F90"
        )

    def test_fortran(self):
        """Call fortran stencil"""

        self.mode_cloud_fraction.cloud_fraction(
            XTT,
            XRTMIN,
            LFEEDBACKT,
            KPROMA,
            KSIZE,
            LDCOMPUTE,
            PEXN,
            PLVFACT,
            PLSFACT,
            PT,
            PRRT,
            PTHT,
            PRRHONG_MR,
        )

    def test_gt4py(self):
        """Call GT4Py stencil"""

        self.cloud_fraction(
            LFEEDBACKT,
            KPROMA,
            KSIZE,
            LDCOMPUTE,
            PEXN,
            PLVFACT,
            PLSFACT,
            PT,
            PRRT,
            PTHT,
            PRRHONG_MR,
        )

    def compare(self):
        """Compare Fortran and Python routines"""

        self.test_fortran()
        logging.info("Fortran called")

        self.test_gt4py()
        logging.info("GT4Py called")

    


class TestCloudFraction(ComputationalGridComponent):
    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
    ) -> None:
        super().__init__(
            computational_grid=computational_grid, gt4py_config=gt4py_config
        )

        externals = {
            "TT": XTT,
            "R_RTMIN": XRTMIN,
            "LFEEDBACKT": True,
        }

        # aro_filter stands for the parts before 'call ice_adjust' in aro_adjust.f90
        self.cloud_fraction = self.compile_stencil("cloud_fraction", externals)

        self.mode_cloud_fraction = fmodpy.fimport(
            "./src/ice3_gt4py/stencil_fortran/test_ice_adjust/cloud_fraction.F90"
        )


    def test_fortran(self):
        """Call fortran stencil"""

        self.mode_cloud_fraction.cloud_fraction(
            XTT,
            XRTMIN,
            LFEEDBACKT,
            KPROMA,
            KSIZE,
            LDCOMPUTE,
            PEXN,
            PLVFACT,
            PLSFACT,
            PT,
            PRRT,
            PTHT,
            PRRHONG_MR,
        )

    def test_gt4py(self):
        """Call GT4Py stencil"""

        self.cloud_fraction(
            LFEEDBACKT,
            KPROMA,
            KSIZE,
            LDCOMPUTE,
            PEXN,
            PLVFACT,
            PLSFACT,
            PT,
            PRRT,
            PTHT,
            PRRHONG_MR,
        )

    def compare(self):
        """Compare Fortran and Python routines"""

        self.test_fortran()
        logging.info("Fortran called")

        self.test_gt4py()
        logging.info("GT4Py called")
