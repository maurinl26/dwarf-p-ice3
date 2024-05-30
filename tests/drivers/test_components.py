# -*- coding: utf-8 -*-
import json
from pathlib import Path
import subprocess
import numpy as np
import typer
import logging
import datetime
import time
import sys
import xarray as xr

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid

from drivers.core import core
from ice3_gt4py.components.aro_adjust import AroAdjust
from ice3_gt4py.components.ice4_stepping import Ice4Stepping
from ice3_gt4py.components.ice4_tendencies import Ice4Tendencies
from ice3_gt4py.components.ice_adjust import IceAdjust
from ice3_gt4py.components.rain_ice import RainIce
from ice3_gt4py.initialisation.state_aro_adjust import (
    get_constant_state_aro_adjust,
    aro_adjust_fields_keys,
)
from ice3_gt4py.initialisation.state_ice4_tendencies import (
    get_constant_state_ice4_tendencies,
)
from ice3_gt4py.initialisation.state_ice4_stepping import (
    get_constant_state_ice4_stepping,
)
from ice3_gt4py.initialisation.state_ice_adjust import (
    get_state_ice_adjust,
)
from ice3_gt4py.initialisation.state_rain_ice import get_state_rain_ice
from ice3_gt4py.phyex_common.phyex import Phyex
from ice3_gt4py.utils.reader import NetCDFReader

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()

app = typer.Typer()


@app.command()
def run_tendencies(
    backend: str, tracking_file: str, rebuild: bool = True, validate_args: bool = False
):
    """Test Ice4Tendencies component

    Args:
        backend (str): _description_
        tracking_file (str): _description_
        rebuild (bool, optional): _description_. Defaults to True.
        validate_args (bool, optional): _description_. Defaults to False.
    """

    ######## Backend and gt4py config #######
    logging.info(f"With backend {backend}")
    gt4py_config = GT4PyConfig(
        backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=True
    )

    ############# Grid #####################
    logging.info("Initializing grid ...")
    grid = ComputationalGrid(50, 1, 15)
    dt = datetime.timedelta(seconds=1)

    phyex = Phyex(program="AROME")

    ####### Create state for Ice4Tendencies #######
    logging.info("Getting state for Ice4Tendencies")
    state = get_constant_state_ice4_tendencies(grid, gt4py_config=gt4py_config)

    ####### Launch execution ##################
    ######## Instanciation + compilation #####
    logging.info(f"Compilation for RainIce stencils")
    start = time.time()
    stepping = Ice4Stepping(grid, gt4py_config, phyex)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Compilation duration for RainIce : {elapsed_time} s")

    ####### Create state for AroAdjust #######
    logging.info("Getting state for RainIce")
    state = get_constant_state_ice4_stepping(grid, gt4py_config=gt4py_config)
    logging.info(f"Keys : {list(state.keys())}")

    ###### Launching RainIce ###############
    logging.info("Launching RainIce")

    start = time.time()
    tends, diags = stepping(state, dt)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Execution duration for RainIce : {elapsed_time} s")

    logging.info(f"Extracting exec tracking to {tracking_file}")
    with open(tracking_file, "w") as file:
        json.dump(gt4py_config.exec_info, file)


@app.command()
def run_stepping(
    backend: str, tracking_file: str, rebuild: bool = True, validate_args: bool = False
):
    """Test Ice4Stepping component.

    Args:
        backend (str): _description_
        tracking_file (str): _description_
        rebuild (bool, optional): _description_. Defaults to True.
        validate_args (bool, optional): _description_. Defaults to False.
    """

    ##### Grid #####
    logging.info("Initializing grid and timestep ...")
    grid = ComputationalGrid(10000, 1, 15)
    dt = datetime.timedelta(seconds=1)

    ################## Phyex #################
    logging.info("Initializing Phyex in AROME configuration")
    phyex = Phyex(program="AROME")

    ######## Backend and gt4py config #######
    logging.info(f"With backend {backend}")
    gt4py_config = GT4PyConfig(
        backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=True
    )

    ######## Instanciation + compilation #####
    logging.info(f"Compilation for RainIce stencils")
    start = time.time()
    stepping = Ice4Stepping(grid, gt4py_config, phyex)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Compilation duration for RainIce : {elapsed_time} s")

    ####### Create state for AroAdjust #######
    logging.info("Getting state for RainIce")
    state = get_constant_state_ice4_stepping(grid, gt4py_config=gt4py_config)
    logging.info(f"Keys : {list(state.keys())}")

    ###### Launching RainIce ###############
    logging.info("Launching RainIce")

    start = time.time()
    tends, diags = stepping(state, dt)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Execution duration for RainIce : {elapsed_time} s")

    logging.info(f"Extracting exec tracking to {tracking_file}")
    with open(tracking_file, "w") as file:
        json.dump(gt4py_config.exec_info, file)


@app.command()
def run_aro_adjust(backend: str, rebuild: bool = True, validate_args: bool = False):
    """Run aro_adjust component"""

    ##### Grid #####
    logging.info("Initializing grid ...")
    grid = ComputationalGrid(50, 1, 15)
    dt = datetime.timedelta(seconds=1)

    ################## Phyex #################
    logging.info("Initializing Phyex ...")
    phyex = Phyex(program="AROME")

    ######## Backend and gt4py config #######
    logging.info(f"With backend {backend}")
    gt4py_config = GT4PyConfig(
        backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=True
    )

    ######## Instanciation + compilation #####
    logging.info(f"Compilation for AroAdjust stencils")
    start = time.time()
    aro_adjust = AroAdjust(grid, gt4py_config, phyex)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Compilation duration for AroAdjust : {elapsed_time} s")

    ####### Create state for AroAdjust #######
    logging.info("Getting constant state for AroAdjust")
    state = get_constant_state_aro_adjust(
        grid, gt4py_config=gt4py_config, keys=aro_adjust_fields_keys
    )
    logging.info(f"Keys : {list(state.keys())}")

    ###### Launching AroAdjust ###############
    logging.info("Launching AroAdjust")

    start = time.time()
    tends, diags = aro_adjust(state, dt)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Execution duration for AroAdjust : {elapsed_time} s")

    logging.info("Extracting state data to ...")

    with open("run_aro_adjust.json", "w") as file:
        json.dump(gt4py_config.exec_info, file)


if __name__ == "__main__":
    app()
