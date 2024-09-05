# -*- coding: utf-8 -*-
import fmodpy
import numpy as np
import logging
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.grid import ComputationalGrid
import datetime

from ice3_gt4py.phyex_common.phyex import Phyex


class TestIce4RRHONG(ComputationalGridComponent):
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
        self.ice4_rrhong_gt4py = self.compile_stencil("ice4_rrhong", externals)

        self.mode_ice4_rrhong = fmodpy.fimport(
            "./src/ice3_gt4py/stencil_fortran/mode_ice4_rrhong.F90"
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

        self.ice4_rrhong = self.mode_ice4_rrhong.mode_ice4_rrhong.ice4_rrhong(
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

        self.ice4_rrhong_gt4py(
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

    XTT = 0
    XRTMIN = 10e-5
    LFEEDBACKT = False

    KSIZE = 15
    KPROMA = 50
    LDCOMPUTE = np.asfortranarray(np.array([True for _ in range(KPROMA)]))

    PEXN = np.asfortranarray(np.random.rand(KSIZE))
    PLVFACT = np.asfortranarray(np.random.rand(KSIZE))
    PLSFACT = np.asfortranarray(np.random.rand(KSIZE))
    PT = np.asfortranarray(np.random.rand(KSIZE))
    PRRT = np.asfortranarray(np.random.rand(KSIZE))
    PTHT = np.asfortranarray(np.random.rand(KSIZE))
    PRRHONG_MR = np.asfortranarray(np.random.rand(KSIZE))

    ice4_rrhong = TestIce4RRHONG(
        computational_grid=grid,
        gt4py_config=gt4py_config,
    )

    ice4_rrhong.compare()
