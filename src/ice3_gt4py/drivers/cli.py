# -*- coding: utf-8 -*-
import typer
import logging
from datetime import timedelta
import time
import sys

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid

from ice3_gt4py.components.aro_adjust import AroAdjust
from ice3_gt4py.initialisation.state_aro_adjust import get_constant_state_aro_adjust
from ice3_gt4py.phyex_common.phyex import Phyex

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()

app = typer.Typer()


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
    dt = timedelta(seconds=1).total_seconds

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
    logging.info("Getting constant state for AroAdjust")
    state = get_constant_state_aro_adjust(grid, gt4py_config=gt4py_config)

    # out_tendencies =
    # out_diagnostics =
    # overwrite_tendencies =

    ###### Launching AroAdjust ###############
    logging.info("Launching AroAdjust")
    try:
        start = time.time()
        aro_adjust(state, dt)
        stop = time.time()
        elapsed_time = stop - start
        logging.info(f"Execution duration for AroAdjust : {elapsed_time} s")

    except:
        logging.error("Execution failed for AroAdjust")

    else:
        logging.info("Extracting state data to ...")


@app.command()
def run_aro_rain_ice():
    """Run aro_rain_ice component"""
    NotImplemented


if __name__ == "__main__":
    app()
