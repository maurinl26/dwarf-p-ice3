# -*- coding: utf-8 -*-
import datetime
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import typer
import xarray as xr

from ..drivers.core import write_performance_tracking, compare_fields
from ..ice3.components.ice_adjust import IceAdjust
from ..ice3.components.rain_ice import RainIce
from ..ice3.initialisation.state_ice_adjust import get_state_ice_adjust
from ..ice3.initialisation.state_rain_ice import get_state_rain_ice
from ..ice3.phyex_common.phyex import Phyex
from ..ice3.utils.env import ROOT_PATH


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
log = logging.getLogger(__name__)

app = typer.Typer()

######################## GT4Py drivers #######################
@app.command()
def ice_adjust(
    domain: Tuple[int, int, int] = (10000, 1, 50),
    dataset: Path = Path(ROOT_PATH, "data", "ice_adjust.nc"),
    output_path: Path = Path(ROOT_PATH, "data", "ice_adjust_run.nc"),
    tracking_file: Path = Path(ROOT_PATH, "ice_adjust_run.json"),
    backend: str = "gt:cpu_ifirst",
    rebuild: bool = True,
    validate_args: bool = False,
):
    """Run ice_adjust component"""

    ################## Domain ################
    log.info("Initializing grid ...")
    dt = datetime.timedelta(seconds=1)

    ################## Phyex #################
    log.info("Initializing Phyex ...")
    phyex = Phyex("AROME")

    ######## Instanciation + compilation #####
    log.info(f"Compilation for IceAdjust stencils")
    start_time = time.time()
    ice_adjust = IceAdjust()
    elapsed_time = time.time() - start_time
    log.info(f"Compilation duration for IceAdjust : {elapsed_time} s")

    ####### Create state for AroAdjust #######
    log.info("Getting state for IceAdjust")
    # todo : reader to dataset
    ds = xr.load_dataset(dataset)
    state = get_state_ice_adjust(domain, backend, ds)

    # TODO: decorator for tracking
    start = time.time()
    tends, diags = ice_adjust(state, dt)
    stop = time.time()
    elapsed_time = stop - start
    log.info(f"Execution duration for IceAdjust : {elapsed_time} s")

    #################### Write dataset ######################
    xr.Dataset(state).to_netcdf(output_path)

    ############### Compute differences per field ###########
    metrics = compare_fields(dataset, output_path, "ice_adjust")

    ####################### Tracking ########################
    write_performance_tracking(metrics, tracking_file)


@app.command()
def rain_ice(
    domain: Tuple[int, int, int] = (5000, 1, 15),
    dataset: Path = Path(ROOT_PATH, "data", "rain_ice.nc"),
    output_path: Path = Path(ROOT_PATH, "data", "rain_ice_run.nc"),
    tracking_file: Path = Path(ROOT_PATH, "rain_ice_run.json"),
    backend: str = "gt:cpu_ifirst",
    rebuild: bool = True,
    validate_args: bool = False,
):
    """Run aro_rain_ice component"""

    ################## Grid ##################
    log.info("Initializing grid ...")
    dt = datetime.timedelta(seconds=1)

    ################## Phyex #################
    log.info("Initializing Phyex ...")
    phyex = Phyex("AROME")

    ######## Backend and gt4py config #######
    log.info(f"With backend {backend}")

    ######## Instanciation + compilation #####
    log.info(f"Compilation for RainIce stencils")
    start = time.time()
    rain_ice = RainIce()
    stop = time.time()
    elapsed_time = stop - start
    log.info(f"Compilation duration for RainIce : {elapsed_time} s")

    ####### Create state for AroAdjust #######
    log.info("Getting state for RainIce")
    ds = xr.load_dataset(dataset)
    state = get_state_rain_ice(domain, ds)

    ###### Launching RainIce #################
    log.info("Launching RainIce")
    start = time.time()
    tends, diags = rain_ice(state, dt, domain)
    stop = time.time()
    elapsed_time = stop - start
    log.info(f"Execution duration for RainIce : {elapsed_time} s")

    log.info(f"Extracting state data to {output_path}")
    xr.Dataset(state).to_netcdf(output_path)

    ################# Metrics and Performance tracking ############
    metrics = compare_fields(dataset, output_path, "rain_ice")
    write_performance_tracking(metrics, tracking_file)


if __name__ == "__main__":
    app()
