# -*- coding: utf-8 -*-
import fmodpy
import numpy as np
import logging
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.grid import ComputationalGrid
import datetime
from ice3_gt4py.initialisation.state_rain_ice import allocate_state_ice4_rrhong
from ice3_gt4py.initialisation.utils import initialize_field
import ice3_gt4py.stencils



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
            "./src/ice3_gt4py/stencils_fortran/mode_ice4_rrhong.F90"
        )

    def test_feedbackt(self):
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
        
        rrhong_mr_fortran = PRRHONG_MR.copy()
        
        state =  allocate_state_ice4_rrhong(self.computational_grid, self.gt4py_config)
        
        logging.info(f"ldcompute shape {LDCOMPUTE.shape}")
        
        state_ice4_rrhong = {
            "ldcompute": initialize_field(state["ldcompute"], LDCOMPUTE),
            "exn": initialize_field(state["exn"], PEXN),
            "ls_fact": initialize_field(state["ls_fact"], PLSFACT),
            "lv_fact": initialize_field(state["lv_fact"], PLVFACT),
            "t": initialize_field(state["t"], PT),
            "tht": initialize_field(state["tht"], PTHT),
            "rr_t": initialize_field(state["rr_t"], PRRT),
            "rrhong_mr": initialize_field(state["rrhong_mr"], PRRHONG_MR)
        }
        
        self.ice4_rrhong_gt4py(
            **state_ice4_rrhong,
        )
        
        rrhong_mr_gt4py = PRRHONG_MR.copy() 
        
        diff = rrhong_mr_gt4py - rrhong_mr_fortran
        assert diff.mean() < 10e-2

    


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

    XTT = 0
    XRTMIN = 10e-5
    LFEEDBACKT = False

    KSIZE = 15
    KPROMA = 50
    LDCOMPUTE = np.asfortranarray(np.array([[True for _ in range(KPROMA)] for _ in range(KSIZE)]))

    PEXN = np.asfortranarray(np.random.rand(KPROMA, KSIZE))
    PLVFACT = np.asfortranarray(np.random.rand(KPROMA, KSIZE))
    PLSFACT = np.asfortranarray(np.random.rand(KPROMA, KSIZE))
    PT = np.asfortranarray(np.random.rand(KPROMA, KSIZE))
    PRRT = np.asfortranarray(np.random.rand(KPROMA, KSIZE))
    PTHT = np.asfortranarray(np.random.rand(KPROMA, KSIZE))
    PRRHONG_MR = np.asfortranarray(np.random.rand(KPROMA, KSIZE))

    ice4_rrhong = TestIce4RRHONG(
        computational_grid=grid,
        gt4py_config=gt4py_config,
    )

    ice4_rrhong.test_feedbackt()
