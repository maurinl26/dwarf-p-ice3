# -*- coding: utf-8 -*-
import json
from pathlib import Path
import typer
import logging
import datetime
import time
import sys
import xarray as xr

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid

from ice3_gt4py.components.aro_adjust import AroAdjust
from ice3_gt4py.components.ice_adjust import IceAdjust
from ice3_gt4py.initialisation.state_aro_adjust import (
    get_constant_state_aro_adjust,
    aro_adjust_fields_keys,
)
from ice3_gt4py.initialisation.state_ice_adjust import (
    get_constant_state_ice_adjust,
    get_state_ice_adjust,
    ice_adjust_fields_keys,
)
from ice3_gt4py.phyex_common.phyex import Phyex
from ice3_gt4py.utils.reader import NetCDFReader

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()

app = typer.Typer()


@app.command()
def run_ice_adjust(backend: str, dataset: str, output_path: str, tracking_file: str):
    """Run ice_adjust component"""

    ##### Grid #####
    logging.info("Initializing grid ...")
    nx = 50
    ny = 1
    nz = 15
    grid = ComputationalGrid(nx, ny, nz)
    dt = datetime.timedelta(seconds=1)

    ################## Phyex #################
    logging.info("Initializing Phyex ...")
    cprogram = "AROME"
    phyex = Phyex(cprogram)

    ######## Backend and gt4py config #######
    logging.info(f"With backend {backend}")
    gt4py_config = GT4PyConfig(
        backend=backend, rebuild=True, validate_args=False, verbose=True
    )

    ######## Instanciation + compilation #####
    logging.info(f"Compilation for IceAdjust stencils")
    start = time.time()
    ice_adjust = IceAdjust(grid, gt4py_config, phyex)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Compilation duration for IceAdjust : {elapsed_time} s")

    ####### Create state for AroAdjust #######
    reader = NetCDFReader(Path(dataset))

    logging.info("Getting state for IceAdjust")
    state = get_state_ice_adjust(grid, gt4py_config=gt4py_config, netcdf_reader=reader)
    logging.info(f"Keys : {list(state.keys())}")

    ###### Launching AroAdjust ###############
    logging.info("Launching IceAdjust")

    start = time.time()
    tends, diags = ice_adjust(state, dt)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Execution duration for AroAdjust : {elapsed_time} s")

    logging.info(f"Extracting state data to {output_path}")
    output_fields = xr.Dataset(state)
    for key, field in state.items():
        if key not in ["time"]:
            array = xr.DataArray(
                data=field.data,
                dims=["I", "J", "K"],
                coords={
                    "I": range(nx),
                    "J": range(ny),
                    "K": range(nz),
                },
                name=f"{key}",
            )
            output_fields[key] = array
    output_fields.to_netcdf(Path(output_path))

    logging.info(f"Extracting exec tracking to {tracking_file}")
    with open(tracking_file, "w") as file:
        json.dump(gt4py_config.exec_info, file)


@app.command()
def run_aro_adjust(
    backend: str,
):
    """Run aro_adjust component"""

    ##### Grid #####
    logging.info("Initializing grid ...")
    nx = 100
    ny = 1
    nz = 90
    grid = ComputationalGrid(nx, ny, nz)
    dt = datetime.timedelta(seconds=1)

    ################## Phyex #################
    logging.info("Initializing Phyex ...")
    cprogram = "AROME"
    phyex = Phyex(cprogram)

    ######## Backend and gt4py config #######
    logging.info(f"With backend {backend}")
    gt4py_config = GT4PyConfig(
        backend=backend, rebuild=True, validate_args=False, verbose=True
    )

    ######## Instanciation + compilation #####
    logging.info(f"Compilation for AroAdjust stencils")
    start = time.time()
    aro_adjust = AroAdjust(grid, gt4py_config, phyex)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Compilation duration for AroAdjust : {elapsed_time} s")

    ####### Create state for AroAdjust #######
    logging.info("Getting constant state for IceAdjust")
    state = get_constant_state_aro_adjust(
        grid, gt4py_config=gt4py_config, keys=aro_adjust_fields_keys
    )
    logging.info(f"Keys : {list(state.keys())}")

    ###### Launching AroAdjust ###############
    logging.info("Launching IceAdjust")

    start = time.time()
    tends, diags = aro_adjust(state, dt)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Execution duration for IceAdjust : {elapsed_time} s")

    logging.info("Extracting state data to ...")

    with open("run_aro_adjust.json", "w") as file:
        json.dump(gt4py_config.exec_info, file)


@app.command()
def run_aro_rain_ice():
    """Run aro_rain_ice component"""
    NotImplemented


if __name__ == "__main__":
    app()
