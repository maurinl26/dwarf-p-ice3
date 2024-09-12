# -*- coding: utf-8 -*-
import fmodpy
import numpy as np
import logging
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.grid import ComputationalGrid
import datetime

from ice3_gt4py.phyex_common.phyex import Phyex


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

        # self.KPROMA = 50
        # self.KSIZE = 50

        # self.LDCOMPUTE = np.asfortranarray(np.random.randint(2, size=KPROMA))
        # self.PEXN = np.asfortranarray(np.random.rand(KSIZE))
        # self.PLVFACT = np.asfortranarray(np.random.rand(KSIZE))
        # self.PLSFACT = np.asfortranarray(np.random.rand(KSIZE))
        # self.PT = np.asfortranarray(np.random.rand(KSIZE))
        # self.PRRT = np.asfortranarray(np.random.rand(KSIZE))
        # self.PTHT = np.asfortranarray(np.random.rand(KSIZE))
        # PRRHONG_MR = np.asfortranarray(np.random.rand(KSIZE))

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


if __name__ == "__main__":

    backend = "gt:cpu_ifirst"
    rebuild = True
    validate_args = True

    logging.info("Initializing grid ...")
    grid = ComputationalGrid(50, 1, 15)
    dt = datetime.timedelta(seconds=1)

    ######## Backend and gt4py config #######
    logging.info(f"With backend {backend}")
    gt4py_config = GT4PyConfig(
        backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=True
    )

    # ############### Fortran call ###############
    # mode_ice4_rrhong = fmodpy.fimport(
    #     "./src/ice3_gt4py/stencil_fortran/mode_ice4_rrhong.F90"
    # )

    NIJT, NKT = 50, 15
    PTSTEP = datetime.timedelta(seconds=1)
    OCOMPUTE_SRC = True
    XCRIAUTC = 10e-5
    XCRIAUTI = 10e-5
    XACRIAUTI = 10e-5
    XBCRIAUTI = 10e-5
    XTT = 0
    NKTE = 16
    NKTB = 0
    NIJB = 0
    NIJE = 51
    CSUBG_MF_PDF = "TRIANGLE"

    ZRI = np.asfortranarray(np.random.rand(NIJT, NKT))
    ZRC = np.asfortranarray(np.random.rand(NIJT, NKT))

    PRHODREF = np.asfortranarray(np.random.rand(NIJT, NKT))
    PCF_MF = np.asfortranarray(np.random.rand(NIJT, NKT))
    PRC_MF = np.asfortranarray(np.random.rand(NIJT, NKT))
    PRI_MF = np.asfortranarray(np.random.rand(NIJT, NKT))
    PRC = np.asfortranarray(np.random.rand(NIJT, NKT))
    PRVS = np.asfortranarray(np.random.rand(NIJT, NKT))
    PRCS = np.asfortranarray(np.random.rand(NIJT, NKT))
    PTHS = np.asfortranarray(np.random.rand(NIJT, NKT))
    PSRCS = np.asfortranarray(np.random.rand(NIJT, NKT))
    PCLDFR = np.asfortranarray(np.random.rand(NIJT, NKT))
    PRI = np.asfortranarray(np.random.rand(NIJT, NKT))
    PRIS = np.asfortranarray(np.random.rand(NIJT, NKT))
    PHLC_HRC = np.asfortranarray(np.random.rand(NIJT, NKT))
    PHLC_HCF = np.asfortranarray(np.random.rand(NIJT, NKT))
    PHLI_HRI = np.asfortranarray(np.random.rand(NIJT, NKT))
    PHLI_HCF = np.asfortranarray(np.random.rand(NIJT, NKT))

    cloud_fraction = TestCloudFraction(
        computational_grid=grid,
        gt4py_config=gt4py_config,
    )

    cloud_fraction.compare()
