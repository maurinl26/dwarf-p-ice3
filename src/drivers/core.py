# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Tuple
import logging
import datetime
import time
import sys
import xarray as xr

import cupy as cp
import cupy_xarray

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.framework.components import ImplicitTendencyComponent

from ice3_gt4py.phyex_common.phyex import Phyex

from ifs_physics_common.utils.typingx import (
    DataArrayDict,
    NDArrayLikeDict, PropertyDict

)

from ice3_gt4py.utils.reader import NetCDFReader


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


def write_performance_tracking(
    gt4py_config: GT4PyConfig, metrics: dict, tracking_file: str
):
    """Write performance tracking in a log file

    Args:
        gt4py_config (GT4PyConfig): gt4py_config to retrieve exec info
        tracking_file (str): tracking file to write in
    """

    logging.info(f"Extracting exec tracking to {tracking_file}")
    with open(tracking_file, "w") as file:
        json.dump({"performances": gt4py_config.exec_info, "metrics": metrics}, file)


def write_dataset(state: DataArrayDict, shape: Tuple[int], output_path: str):
    """Write output state to netCDF

    Args:
        state (_type_): xr.Dataset
        keys (_type_): keys to write in netCDF
        output_path (_type_): path to write field
    """
    nx, ny, nz = shape

    logging.info(f"Extracting state data to {output_path}")
    output_fields = xr.Dataset(state)
    for key, field in state.items():
        if key not in ["time"]:
            array = xr.DataArray(
                data=field.data[:, :, 1:],
                dims=["I", "J", "K"],
                coords={
                    "I": range(nx),
                    "J": range(ny),
                    "K": range(nz),
                },
                name=f"{key}",
            )
            output_fields[key] = array

        output_fields = output_fields.as_numpy()
    output_fields.to_netcdf(Path(output_path))
    logging.info(f"Data Array written to {output_path}")


def initialize_state(
    component: ImplicitTendencyComponent,
    reader: NetCDFReader,
    grid: ComputationalGrid,
    config: GT4PyConfig,
):
    """_summary_

    Args:
        component (_type_): _description_
        reader (_type_): _description_
        grid (_type_): _description_
        config (_type_): _description_
    """
    # TODO : function to implement
    pass


def core(
    component: ImplicitTendencyComponent,
    gt4py_config: GT4PyConfig,
    grid: ComputationalGrid,
    state: DataArrayDict,
    output_path: str,
    dt: datetime.timedelta,
    tracking_file: str,
):
    """Core for component execution drivers

    Args:
        component (ImplicitTendencyComponent): component to run
        gt4py_config (GT4PyConfig): gt4py_configuration
        state (DataArrayDict): state for the component
        output_path (str): path for output dataset
    """

    nx, ny, nz = grid[("I", "J", "K")].shape

    ################## Phyex #################
    logging.info("Initializing Phyex ...")
    cprogram = "AROME"
    phyex = Phyex(cprogram)

    ######## Instanciation + compilation #####
    logging.info(f"Compilation for AroAdjust stencils")
    start = time.time()
    comp = component(grid, gt4py_config, phyex)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Compilation duration for AroAdjust : {elapsed_time} s")

    ###### Launching AroAdjust ###############
    logging.info("Launching IceAdjust")

    start = time.time()
    tends, diags = comp(state, dt)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Execution duration for IceAdjust : {elapsed_time} s")

    # TODO : replace with write output
    logging.info(f"Extracting state data to {output_path}")
    output_fields = xr.Dataset(state)
    for key, field in state.items():
        if key not in ["time"]:
            array = xr.DataArray(
                data=field.data[:, :, 1:],
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

    with open(f"{tracking_file}", "w") as file:
        json.dump(gt4py_config.exec_info, file)
