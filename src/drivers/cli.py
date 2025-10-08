# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np

from ice3.components.ice_adjust_split import IceAdjustSplit
import typer
import logging
import datetime
import time
import sys
import xarray as xr
from ifs_physics_common.utils.numpyx import to_numpy

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid

from drivers.compare import compare_fields
from drivers.core import write_dataset, write_2d_dataset, write_performance_tracking
from ice3.components.ice_adjust import IceAdjust
from ice3.initialisation.state_ice_adjust import (
    get_state_ice_adjust,
)
from ice3.initialisation.state_rain_ice import get_state_rain_ice
from ice3.phyex_common.phyex import Phyex
from ice3.utils.reader import NetCDFReader

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()

app = typer.Typer()


######################## GT4Py drivers #######################
@app.command()
def ice_adjust(
    backend: str,
    dataset: str,
    output_path: str,
    tracking_file: str,
    rebuild: bool = True,
    validate_args: bool = False,
):
    """Run ice_adjust component"""

    ##### Grid #####
    logging.info("Initializing grid ...")
    nx = 10000
    ny = 1
    nz = 15
    grid = ComputationalGrid(nx, ny, nz)
    dt = datetime.timedelta(seconds=1)

    ################## Phyex #################
    logging.info("Initializing Phyex ...")
    phyex = Phyex("AROME")

    ######## Backend and gt4py config #######
    logging.info(f"With backend {backend}")
    gt4py_config = GT4PyConfig(
        backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=True
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

    ###### Launching IceAdjust ###############
    logging.info("Launching IceAdjust")

    # TODO: decorator for tracking
    start = time.time()
    tends, diags = ice_adjust(state, dt)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Execution duration for IceAdjust : {elapsed_time} s")

    #################### Write dataset ######################
    write_dataset(state, (nx, ny, nz), output_path)

    ############### Compute differences per field ###########
    metrics = compare_fields(dataset, output_path, "ice_adjust")

    ####################### Tracking ########################
    write_performance_tracking(gt4py_config, metrics, tracking_file)


@app.command()
def ice_adjust_split(
    backend: str,
    dataset: str,
    output_path: str,
    tracking_file: str,
    rebuild: bool = True,
    validate_args: bool = False,
):
    """Run ice_adjust splitted version to avoid
    interpolation problems for sigrc
    """

    ##### Grid #####
    logging.info("Initializing grid ...")
    nx = 9472
    ny = 1
    nz = 15
    grid = ComputationalGrid(nx, ny, nz)
    dt = datetime.timedelta(seconds=50)

    ################## Phyex #################
    logging.info("Initializing Phyex ...")
    phyex = Phyex("AROME")

    ######## Backend and gt4py config #######
    logging.info(f"With backend {backend}")
    gt4py_config = GT4PyConfig(
        backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=True
    )

    ######## Instanciation + compilation #####
    logging.info(f"Compilation for IceAdjust stencils")
    start_compilation = time.time()
    ice_adjust_split = IceAdjustSplit(
        grid, gt4py_config, phyex, enable_checks=validate_args
    )
    stop_compilation = time.time()
    elapsed_time = stop_compilation - start_compilation
    logging.info(f"Compilation duration for IceAdjust : {elapsed_time} s")

    ####### Create state for AroAdjust #######
    reader = NetCDFReader(Path(dataset))

    logging.info("Getting state for IceAdjust")
    state = get_state_ice_adjust(grid, gt4py_config=gt4py_config, netcdf_reader=reader)

    # todo : setup check inputs with right interface
    # setup xarray cupy

    ###### Launching IceAdjust ###############
    logging.info("Launching IceAdjust")

    # TODO: decorator for tracking
    start = time.time()
    tends, diags = ice_adjust_split(state, dt)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Execution duration for IceAdjust : {elapsed_time} s")

    ice_adjust_split(state, dt, out_tendencies=tends, out_diagnostics=diags)

    logging.info(f"Diagnostics")
    for name, field in diags.items():
        np_field = to_numpy(field.data[...])
        logging.info(f"Field {name}, mean : {np_field.mean()}")

    logging.info(f"Tendencies")
    for name, field in tends.items():
        np_field = to_numpy(field.data[...])
        logging.info(f"Field {name}, mean : {np_field.mean()}")

    #################### Write dataset ######################
    output_dict = {**state, **diags, **tends}
    write_dataset(output_dict, (nx, ny, nz), output_path)

    ############### Compute differences per field ###########
    # metrics = compare_fields(dataset, output_path, "ice_adjust")

    # ####################### Tracking ########################
    # write_performance_tracking(gt4py_config, metrics, tracking_file)


##################### Fortran drivers #########################
@app.command()
def ice_adjust_fortran(
    dataset: str, output_path: str
):
    """Call and run main_ice_adjust (Fortran)"""
    
    ##### Grid #####
    logging.info("Initializing grid ...")
    nx = 9472
    ny = 1
    nz = 15

    ################## Phyex #################
    logging.info("Initializing Phyex ...")
    phyex = Phyex("AROME")

    ######## Backend and gt4py config #######
    # TODO : remove gt4py config
    gt4py_config = GT4PyConfig(
        backend="numpy", rebuild=False, validate_args=False, verbose=True
    )
    
    #######
    from pyphyex.ice_adjust_wrapper import IceAdjustWrapper
    wrapper = IceAdjustWrapper()

    ####### Create state for AroAdjust #######
    reader = NetCDFReader(Path(dataset))
    
    logging.info("Getting state for IceAdjust")
    
    # TODO : clean state
    state = {
        "prhodj":   reader.get_field("PRHODJ")[0:9472,:],
        "pexnref":  reader.get_field("PEXNREF")[0:9472,:],
        "prhodref": reader.get_field("PRHODREF")[0:9472,:],
        "psigs":    reader.get_field("PSIGS")[0:9472,:],
        "ppabst":   reader.get_field("PPABSM")[0:9472,:], 
        "pexn":     reader.get_field("PEXNREF")[0:9472,:],
        "pcf_mf":   reader.get_field("PCF_MF")[0:9472,:],
        "prc_mf":   reader.get_field("PRC_MF")[0:9472,:],
        "pri_mf":   reader.get_field("PRI_MF")[0:9472,:],
        "pth":      reader.get_field("ZRS")[0:9472,:,0],
        "prv":      reader.get_field("ZRS")[0:9472,:,1],
        "prc":      reader.get_field("ZRS")[0:9472,:,2],
        "prr":      reader.get_field("ZRS")[0:9472,:,3],
        "pri":      reader.get_field("ZRS")[0:9472,:,4],
        "prs":      reader.get_field("ZRS")[0:9472,:,5],
        "prg":      reader.get_field("ZRS")[0:9472,:,6],
        "pths":     reader.get_field("PTHS")[0:9472,:],
        "prvs":     reader.get_field("PRS")[0:9472,:,0],
        "prcs":     reader.get_field("PRS")[0:9472,:,1],
        "pris":     reader.get_field("PRS")[0:9472,:,2], 
        "psigqsat": phyex.nebn.VSIGQSAT * np.ones((nx*ny, nz))    
    }
    
    ###### Launching IceAdjust ###############
    logging.info("Launching IceAdjust")

    start = time.time()
    output = wrapper.ice_adjust(
        nijt=nx * ny,
        nkt=nz, 
        krr=6,
        ptstep=50.0,
        **state
        )
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Execution duration for IceAdjust (Fortran) : {elapsed_time} s")
    
    write_2d_dataset(state, (nx, ny, nz), output_path)

    

@app.command()
def rain_ice(
    backend: str,
    dataset: str,
    output_path: str,
    tracking_file: str,
    rebuild: bool = True,
    validate_args: bool = False,
):
    """Run aro_rain_ice component"""

    ##### Grid #####
    logging.info("Initializing grid ...")
    nx = 5000
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
        backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=True
    )

    ######## Instanciation + compilation #####
    logging.info(f"Compilation for RainIce stencils")
    start = time.time()
    rain_ice = RainIce(grid, gt4py_config, phyex)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Compilation duration for RainIce : {elapsed_time} s")

    ####### Create state for AroAdjust #######
    reader = NetCDFReader(Path(dataset))

    logging.info("Getting state for RainIce")
    state = get_state_rain_ice(grid, gt4py_config=gt4py_config, netcdf_reader=reader)
    logging.info(f"Keys : {list(state.keys())}")

    ###### Launching RainIce ###############
    logging.info("Launching RainIce")

    start = time.time()
    tends, diags = rain_ice(state, dt)
    stop = time.time()
    elapsed_time = stop - start
    logging.info(f"Execution duration for RainIce : {elapsed_time} s")

    logging.info(f"Extracting state data to {output_path}")
    output_fields = xr.Dataset(state)
    for key, field in state.items():
        if key not in ["time"]:
            if key not in ["sea", "town", "inprr", "inprc", "inprg", "inprs"]:
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
            else:
                array = xr.DataArray(
                    data=field.data,
                    dims=["I", "J"],
                    coords={
                        "I": range(nx),
                        "J": range(ny),
                    },
                    name=f"{key}",
                )
                output_fields[key] = array
    output_fields.to_netcdf(Path(output_path))

    ################## Metrics ###################################
    metrics = compare_fields(dataset, output_path, "rain_ice")

    ################### Performance tracking ######################
    write_performance_tracking(gt4py_config, metrics, tracking_file)


@app.command()
def rain_ice_fortran(
    testdir: str, name: str, archfile: str, checkOpt: str, extrapolation_opts: str
):
    """Call and run main_rain_ice (Fortran)"""
    pass


if __name__ == "__main__":
    app()
