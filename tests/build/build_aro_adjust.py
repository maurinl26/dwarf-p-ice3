# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
import json
import logging
from dataclasses import asdict
from datetime import timedelta
from typing import Dict

from gt4py.storage import from_array, ones

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
import numpy as np

from ice3_gt4py.phyex_common.phyex import Phyex
import sys
from ice3_gt4py.phyex_common.tables import src_1d

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()

""" Sandbox for creation of AroAdjust component """

if __name__ == "__main__":

    BACKEND = "gt:cpu_kfirst"
    from ifs_physics_common.framework.stencil import compile_stencil

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
    gt4py_config.reset_exec_info()

    ############## AroFilter - Compilation ################
    logging.info(f"Compilation for aro_filter")
    aro_filter = compile_stencil("aro_filter", gt4py_config, externals)

    ############## IceAdjust - Compilation ####################
    logging.info(f"Compilation for ice_adjust")
    ice_adjust = compile_stencil("ice_adjust", gt4py_config, externals)

    state = {
        "exnref": ones((nx, ny, nz), backend=BACKEND),
        "th_t": ones((nx, ny, nz), backend=BACKEND),
        "exn": ones((nx, ny, nz), backend=BACKEND),
        "exnref": ones((nx, ny, nz), backend=BACKEND),
        "rhodref": ones((nx, ny, nz), backend=BACKEND),
        "pabs": ones((nx, ny, nz), backend=BACKEND),
        "sigs": ones((nx, ny, nz), backend=BACKEND),
        "cf_mf": ones((nx, ny, nz), backend=BACKEND),
        "rc_mf": ones((nx, ny, nz), backend=BACKEND),
        "ri_mf": ones((nx, ny, nz), backend=BACKEND),
        "th": ones((nx, ny, nz), backend=BACKEND),
        "rv": ones((nx, ny, nz), backend=BACKEND),
        "rc": ones((nx, ny, nz), backend=BACKEND),
        "ri": ones((nx, ny, nz), backend=BACKEND),
        "rr": ones((nx, ny, nz), backend=BACKEND),
        "rs": ones((nx, ny, nz), backend=BACKEND),
        "rg": ones((nx, ny, nz), backend=BACKEND),
        "sigqsat": ones((nx, ny, nz), backend=BACKEND),
        "cldfr": ones((nx, ny, nz), backend=BACKEND),
        "ifr": ones((nx, ny, nz), backend=BACKEND),
        "hlc_hrc": ones((nx, ny, nz), backend=BACKEND),
        "hlc_hcf": ones((nx, ny, nz), backend=BACKEND),
        "hli_hri": ones((nx, ny, nz), backend=BACKEND),
        "hli_hcf": ones((nx, ny, nz), backend=BACKEND),
        "sigrc": ones((nx, ny, nz), backend=BACKEND),
    }

    # sources
    diagnostics = {
        "ths": ones((nx, ny, nz), backend=BACKEND),
        "rcs": ones((nx, ny, nz), backend=BACKEND),
        "rrs": ones((nx, ny, nz), backend=BACKEND),
        "ris": ones((nx, ny, nz), backend=BACKEND),
        "rvs": ones((nx, ny, nz), backend=BACKEND),
        "rgs": ones((nx, ny, nz), backend=BACKEND),
        "rss": ones((nx, ny, nz), backend=BACKEND),
    }

    ############## AroFilter - State ####################
    state_filter = {
        "exnref": state["exnref"],
        "th_t": state["th_t"],
        "ths": diagnostics["ths"],
        "rcs": diagnostics["rcs"],
        "rrs": diagnostics["rrs"],
        "ris": diagnostics["ris"],
        "rvs": diagnostics["rvs"],
        "rgs": diagnostics["rgs"],
        "rss": diagnostics["rss"],
    }

    temporaries_filter = {
        "t_tmp": ones((nx, ny, nz), backend=BACKEND),
        "ls_tmp": ones((nx, ny, nz), backend=BACKEND),
        "lv_tmp": ones((nx, ny, nz), backend=BACKEND),
        "cph_tmp": ones((nx, ny, nz), backend=BACKEND),
        "cor_tmp": ones((nx, ny, nz), backend=BACKEND),
    }

    # timestep
    dt = datetime.timedelta(seconds=1).total_seconds()
    aro_filter(
        **state_filter,
        **temporaries_filter,
        dt=dt,
        origin=(0, 0, 0),
        domain=grid.grids[I, J, K].shape,
        validate_args=True,
        exec_info=gt4py_config.exec_info,
    )

    ############## IceAdjust - State ##########################
    state_ice_adjust = {
        **{
            key: state[key]
            for key in [
                "sigqsat",
                "exn",
                "exnref",
                "rhodref",
                "pabs",
                "sigs",
                "cf_mf",
                "rc_mf",
                "ri_mf",
                "th",
                "rv",
                "rc",
                "rr",
                "ri",
                "rs",
                "rg",
                "cldfr",
                "ifr",
                "hlc_hrc",
                "hlc_hcf",
                "hli_hri",
                "hli_hcf",
                "sigrc",
            ]
        },
        **{
            key: diagnostics[key]
            for key in [
                "ths",
                "rvs",
                "rcs",
                "ris",
            ]
        },
    }

    temporaries_ice_adjust = {
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
    src_1D = from_array(src_1d, backend=BACKEND)

    # Timestep
    dt = datetime.timedelta(seconds=1).total_seconds()
    ice_adjust(
        **state_ice_adjust,
        **temporaries_ice_adjust,
        src_1d=src_1D,
        dt=dt,
        origin=(0, 0, 0),
        domain=grid.grids[I, J, K].shape,
        validate_args=True,
        exec_info=gt4py_config.exec_info,
    )

    logging.info(f"exec_info : {gt4py_config.exec_info}")

    with open("run_aro_adjust.json", "w") as file:
        json.dump(gt4py_config.exec_info, file)
