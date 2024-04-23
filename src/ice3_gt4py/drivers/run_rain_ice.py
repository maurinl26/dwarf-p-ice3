# -*- coding: utf-8 -*-
import logging
import sys
from datetime import timedelta

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.framework.stencil import compile_stencil
from gt4py.storage import ones
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
    dt = timedelta(seconds=2)

    TSTEP = dt.total_seconds()

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

    ######### State and field declaration ######
    global_state = {"ldcompute": ones((nx, ny, nz), backend=BACKEND, dtype=bool)}

    ############## t_micro_init ################
    logging.info("t_micro_init")
    t_micro_init = compile_stencil("ice4_stepping_tmicro_init", gt4py_config, externals)

    state_t_micro_init = {
        "t_micro": ones((nx, ny, nz), backend=BACKEND),
        "ldmicro": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
    }

    t_micro_init(**state_t_micro_init)

    outerloop_counter = 0
    max_outerloop_iterations = 10

    innerloop_counter = 0
    max_innerloop_iterations = 10

    while np.any(state_t_micro_init["t_micro"][...] < TSTEP):

        # Iterations limiter
        if outerloop_counter >= max_outerloop_iterations:
            break

        while np.any(global_state["ldcompute"][...]):

            # Iterations limiter
            if innerloop_counter >= max_innerloop_iterations:
                break

            # 244

            # 249
            ####### ice4_stepping_heat #############
            ice4_stepping_heat = compile_stencil(
                "ice4_stepping_heat", gt4py_config, externals
            )

            state_stepping_heat = {
                "rv_t": ones((nx, ny, nz), backend=BACKEND),
                "rc_t": ones((nx, ny, nz), backend=BACKEND),
                "rr_t": ones((nx, ny, nz), backend=BACKEND),
                "ri_t": ones((nx, ny, nz), backend=BACKEND),
                "rs_t": ones((nx, ny, nz), backend=BACKEND),
                "rg_t": ones((nx, ny, nz), backend=BACKEND),
                "exn": ones((nx, ny, nz), backend=BACKEND),
                "th_t": ones((nx, ny, nz), backend=BACKEND),
                "ls_fact": ones((nx, ny, nz), backend=BACKEND),
                "lv_fact": ones((nx, ny, nz), backend=BACKEND),
                "t": ones((nx, ny, nz), backend=BACKEND),
            }

            ice4_stepping_heat(**state_stepping_heat)

            ####### tendencies #######
            #### TODO : tendencies state + components #####

            ######### ice4_step_limiter ############################
            ice4_step_limiter = compile_stencil(
                "ice4_step_limiter", gt4py_config, externals
            )

            state_step_limiter = {
                "exn": ones((nx, ny, nz), backend=BACKEND),
                "theta_t": ones((nx, ny, nz), backend=BACKEND),
                "theta_a_tnd": ones((nx, ny, nz), backend=BACKEND),
                "theta_b": ones((nx, ny, nz), backend=BACKEND),
                "theta_ext_tnd": ones((nx, ny, nz), backend=BACKEND),
                "rc_t": ones((nx, ny, nz), backend=BACKEND),
                "rr_t": ones((nx, ny, nz), backend=BACKEND),
                "ri_t": ones((nx, ny, nz), backend=BACKEND),
                "rs_t": ones((nx, ny, nz), backend=BACKEND),
                "rg_t": ones((nx, ny, nz), backend=BACKEND),
                "rc_a_tnd": ones((nx, ny, nz), backend=BACKEND),
                "rr_a_tnd": ones((nx, ny, nz), backend=BACKEND),
                "ri_a_tnd": ones((nx, ny, nz), backend=BACKEND),
                "rs_a_tnd": ones((nx, ny, nz), backend=BACKEND),
                "rg_a_tnd": ones((nx, ny, nz), backend=BACKEND),
                "rc_ext_tnd": ones((nx, ny, nz), backend=BACKEND),
                "rr_ext_tnd": ones((nx, ny, nz), backend=BACKEND),
                "ri_ext_tnd": ones((nx, ny, nz), backend=BACKEND),
                "rs_ext_tnd": ones((nx, ny, nz), backend=BACKEND),
                "rg_ext_tnd": ones((nx, ny, nz), backend=BACKEND),
                "rc_b": ones((nx, ny, nz), backend=BACKEND),
                "rr_b": ones((nx, ny, nz), backend=BACKEND),
                "ri_b": ones((nx, ny, nz), backend=BACKEND),
                "rs_b": ones((nx, ny, nz), backend=BACKEND),
                "rg_b": ones((nx, ny, nz), backend=BACKEND),
                "delta_t_micro": ones((nx, ny, nz), backend=BACKEND),
                "t_micro": ones((nx, ny, nz), backend=BACKEND),
                "delta_t_soft": ones((nx, ny, nz), backend=BACKEND),
                "t_soft": ones((nx, ny, nz), backend=BACKEND),
                "ldcompute": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
            }

            ice4_step_limiter(**state_step_limiter)

            # l346 to l388
            ############ ice4_mixing_ratio_step_limiter ############
            ice4_mixing_ratio_step_limiter = compile_stencil(
                "ice4_mixing_ratio_step_limiter"
            )

            state_mixing_ratio_step_limiter = {
                "rc_0r_t": ones((nx, ny, nz), backend=BACKEND),
                "rr_0r_t": ones((nx, ny, nz), backend=BACKEND),
                "ri_0r_t": ones((nx, ny, nz), backend=BACKEND),
                "rs_0r_t": ones((nx, ny, nz), backend=BACKEND),
                "rg_0r_t": ones((nx, ny, nz), backend=BACKEND),
                "rc_t": ones((nx, ny, nz), backend=BACKEND),
                "rr_t": ones((nx, ny, nz), backend=BACKEND),
                "ri_t": ones((nx, ny, nz), backend=BACKEND),
                "rs_t": ones((nx, ny, nz), backend=BACKEND),
                "rg_t": ones((nx, ny, nz), backend=BACKEND),
                "rc_b": ones((nx, ny, nz), backend=BACKEND),
                "rr_b": ones((nx, ny, nz), backend=BACKEND),
                "ri_b": ones((nx, ny, nz), backend=BACKEND),
                "rs_b": ones((nx, ny, nz), backend=BACKEND),
                "rg_b": ones((nx, ny, nz), backend=BACKEND),
                "rc_tnd_a": ones((nx, ny, nz), backend=BACKEND),
                "rr_tnd_a": ones((nx, ny, nz), backend=BACKEND),
                "ri_tnd_a": ones((nx, ny, nz), backend=BACKEND),
                "rs_tnd_a": ones((nx, ny, nz), backend=BACKEND),
                "rg_tnd_a": ones((nx, ny, nz), backend=BACKEND),
                "delta_t_micro": ones((nx, ny, nz), backend=BACKEND),
                "time_threshold_tmp": ones((nx, ny, nz), backend=BACKEND),
                "ldcompute": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
            }

            ice4_mixing_ratio_step_limiter(**state_mixing_ratio_step_limiter)

            # l394 to l404
            # 4.7 new values for next iteration
            ############### ice4_state_update ######################

            ice4_state_update = compile_stencil("state_update", gt4py_config, externals)

            state_state_update = {
                "theta_t": ones((nx, ny, nz), backend=BACKEND),
                "theta_b": ones((nx, ny, nz), backend=BACKEND),
                "theta_tnd_a": ones((nx, ny, nz), backend=BACKEND),
                "rc_t": ones((nx, ny, nz), backend=BACKEND),
                "rr_t": ones((nx, ny, nz), backend=BACKEND),
                "ri_t": ones((nx, ny, nz), backend=BACKEND),
                "rs_t": ones((nx, ny, nz), backend=BACKEND),
                "rg_t": ones((nx, ny, nz), backend=BACKEND),
                "rc_b": ones((nx, ny, nz), backend=BACKEND),
                "rr_b": ones((nx, ny, nz), backend=BACKEND),
                "ri_b": ones((nx, ny, nz), backend=BACKEND),
                "rs_b": ones((nx, ny, nz), backend=BACKEND),
                "rg_b": ones((nx, ny, nz), backend=BACKEND),
                "rc_tnd_a": ones((nx, ny, nz), backend=BACKEND),
                "rr_tnd_a": ones((nx, ny, nz), backend=BACKEND),
                "ri_tnd_a": ones((nx, ny, nz), backend=BACKEND),
                "rs_tnd_a": ones((nx, ny, nz), backend=BACKEND),
                "rg_tnd_a": ones((nx, ny, nz), backend=BACKEND),
                "delta_t_micro": ones((nx, ny, nz), backend=BACKEND),
                "ldmicro": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
                "ci_t": ones((nx, ny, nz), backend=BACKEND),
                "t_micro": ones((nx, ny, nz), backend=BACKEND),
            }

            ice4_state_update(**state_state_update)

            # TODO : next loop
            lsoft = True
            innerloop_counter += 1
        outerloop_counter += 1

    # l440 to l452
    ################ external_tendencies_update ############
    # if ldext_tnd
    externals_tendencies_update = compile_stencil(
        "external_tendencies_update", gt4py_config, externals
    )

    state_external_tendencies_update = {
        "theta_t": ones((nx, ny, nz), backend=BACKEND),
        "theta_tnd_ext": ones((nx, ny, nz), backend=BACKEND),
        "rc_t": ones((nx, ny, nz), backend=BACKEND),
        "rr_t": ones((nx, ny, nz), backend=BACKEND),
        "ri_t": ones((nx, ny, nz), backend=BACKEND),
        "rs_t": ones((nx, ny, nz), backend=BACKEND),
        "rg_t": ones((nx, ny, nz), backend=BACKEND),
        "rc_tnd_ext": ones((nx, ny, nz), backend=BACKEND),
        "rr_tnd_ext": ones((nx, ny, nz), backend=BACKEND),
        "ri_tnd_ext": ones((nx, ny, nz), backend=BACKEND),
        "rs_tnd_ext": ones((nx, ny, nz), backend=BACKEND),
        "rg_tnd_ext": ones((nx, ny, nz), backend=BACKEND),
        "ldmicro": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
    }

    externals_tendencies_update(**state_external_tendencies_update)
