# -*- coding: utf-8 -*-
import logging
import sys
from datetime import timedelta

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.framework.stencil import compile_stencil
from gt4py.storage import ones, zeros, from_array
import numpy as np
from ice3_gt4py.components.aro_adjust import AroAdjust
from ice3_gt4py.initialisation.state import get_state_ice_adjust
from ice3_gt4py.phyex_common.phyex import Phyex

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()

if __name__ == "__main__":

    BACKEND = "gt:cpu_kfirst"

    ################### Grid #################
    logging.info("Initializing grid ...")
    nx = 100
    ny = 1
    nz = 90
    grid = ComputationalGrid(nx, ny, nz)
    dt = timedelta(seconds=1)

    ################## Phyex #################
    logging.info("Initializing Phyex ...")
    cprogram = "AROME"
    phyex_config = Phyex(cprogram)

    externals = phyex_config.to_externals()

    ######## Backend and gt4py config #######
    logging.info(f"With backend {BACKEND}")
    gt4py_config = GT4PyConfig(
        backend=BACKEND, rebuild=True, validate_args=False, verbose=True
    )

    ############## AroFilter - Compilation ################
    logging.info(f"Compilation for aro_filter")
    aro_filter = compile_stencil("aro_filter", gt4py_config, externals)

    ############## AroFilter - State ####################
    state_filter = {
        "exnref": ones((nx, ny, nz), backend=BACKEND),
        "tht": ones((nx, ny, nz), backend=BACKEND),
        "ths": ones((nx, ny, nz), backend=BACKEND),
        "rcs": ones((nx, ny, nz), backend=BACKEND),
        "rrs": ones((nx, ny, nz), backend=BACKEND),
        "ris": ones((nx, ny, nz), backend=BACKEND),
        "rvs": ones((nx, ny, nz), backend=BACKEND),
        "rgs": ones((nx, ny, nz), backend=BACKEND),
        "rss": ones((nx, ny, nz), backend=BACKEND),
    }

    temporaries_filter = {
        "t_tmp": ones((nx, ny, nz), backend=BACKEND),
        "ls_tmp": ones((nx, ny, nz), backend=BACKEND),
        "lv_tmp": ones((nx, ny, nz), backend=BACKEND),
        "cph_tmp": ones((nx, ny, nz), backend=BACKEND),
        "cor_tmp": ones((nx, ny, nz), backend=BACKEND),
    }

    # timestep
    dt = 1.0
    aro_filter(dt=dt, **state_filter, **temporaries_filter)

    ############## IceAdjust - Compilation ####################
    logging.info(f"Compilation for ice_adjust")
    ice_adjust = compile_stencil("ice_adjust", gt4py_config, externals)

    ############## IceAdjust - State ##########################
    state_ice_adjust = {
        "sigqsat": ones((nx, ny, nz), backend=BACKEND),
        "exn": state_filter["exnref"],
        "exnref": state_filter["exnref"],
        "rhodref": ones((nx, ny, nz), backend=BACKEND),
        "pabs": ones((nx, ny, nz), backend=BACKEND),
        "sigs": ones((nx, ny, nz), backend=BACKEND),
        "cf_mf": ones((nx, ny, nz), backend=BACKEND),
        "rc_mf": ones((nx, ny, nz), backend=BACKEND),
        "ri_mf": ones((nx, ny, nz), backend=BACKEND),
        "th": state_filter["tht"],
        "rv": ones((nx, ny, nz), backend=BACKEND),
        "rc": ones((nx, ny, nz), backend=BACKEND),
        "ri": ones((nx, ny, nz), backend=BACKEND),
        "rr": ones((nx, ny, nz), backend=BACKEND),
        "rs": ones((nx, ny, nz), backend=BACKEND),
        "rg": ones((nx, ny, nz), backend=BACKEND),
        "ths": state_filter["ths"],
        "rvs": state_filter["rvs"],
        "rcs": state_filter["rcs"],
        "ris": state_filter["ris"],
        "cldfr": ones((nx, ny, nz), backend=BACKEND),
        "ifr": ones((nx, ny, nz), backend=BACKEND),
        "hlc_hrc": ones((nx, ny, nz), backend=BACKEND),
        "hlc_hcf": ones((nx, ny, nz), backend=BACKEND),
        "hli_hri": ones((nx, ny, nz), backend=BACKEND),
        "hli_hcf": ones((nx, ny, nz), backend=BACKEND),
        "sigrc": ones((nx, ny, nz), backend=BACKEND),
        "rv_tmp": ones((nx, ny, nz), backend=BACKEND),
        "ri_tmp": ones((nx, ny, nz), backend=BACKEND),
        "rc_tmp": ones((nx, ny, nz), backend=BACKEND),
        "t_tmp": ones((nx, ny, nz), backend=BACKEND),
        "cph": ones((nx, ny, nz), backend=BACKEND),
        "lv": ones((nx, ny, nz), backend=BACKEND),
        "ls": ones((nx, ny, nz), backend=BACKEND),
        "criaut": ones((nx, ny, nz), backend=BACKEND),
        "rt": ones((nx, ny, nz), backend=BACKEND),
        "pv": ones((nx, ny, nz), backend=BACKEND),
        "piv": ones((nx, ny, nz), backend=BACKEND),
        "qsl": ones((nx, ny, nz), backend=BACKEND),
        "qsi": ones((nx, ny, nz), backend=BACKEND),
        "frac_tmp": ones((nx, ny, nz), backend=BACKEND),
        "cond_tmp": ones((nx, ny, nz), backend=BACKEND),
        "a": ones((nx, ny, nz), backend=BACKEND),
        "sbar": ones((nx, ny, nz), backend=BACKEND),
        "sigma": ones((nx, ny, nz), backend=BACKEND),
        "q1": ones((nx, ny, nz), backend=BACKEND),
        "inq1": ones((nx, ny, nz), backend=BACKEND, dtype=np.int64),
    }

    # Global Table
    logging.info("GlobalTable")
    # src_1d = zeros(shape=(34,),  backend=BACKEND, dtype=np.float64),
    src_1d = from_array(
        np.array(
            [
                0.0,
                0.0,
                2.0094444e-04,
                0.316670e-03,
                4.9965648e-04,
                0.785956e-03,
                1.2341294e-03,
                0.193327e-02,
                3.0190963e-03,
                0.470144e-02,
                7.2950651e-03,
                0.112759e-01,
                1.7350994e-02,
                0.265640e-01,
                4.0427860e-02,
                0.610997e-01,
                9.1578111e-02,
                0.135888e00,
                0.1991484,
                0.230756e00,
                0.2850565,
                0.375050e00,
                0.5000000,
                0.691489e00,
                0.8413813,
                0.933222e00,
                0.9772662,
                0.993797e00,
                0.9986521,
                0.999768e00,
                0.9999684,
                0.999997e00,
                1.0000000,
                1.000000,
            ]
        ),
        backend=BACKEND,
    )

    # Timestep
    dt = 1.0
    ice_adjust(dt=dt, src_1d=src_1d, **state_ice_adjust)
