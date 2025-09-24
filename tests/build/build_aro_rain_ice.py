# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import timedelta
from functools import cached_property
from itertools import repeat
from typing import Dict

from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.storage import managed_temporary_storage
from ifs_physics_common.utils.typingx import NDArrayLikeDict, PropertyDict
from ifs_physics_common.utils.f2py import ported_method


from ice3.components.ice4_tendencies import Ice4Tendencies
from ice3.phyex_common.phyex import Phyex


import numpy as np
from gt4py.storage import ones

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

############################################
########## Compile stencils ################
############################################
logging.info("t_micro_init")
t_micro_init = compile_stencil("ice4_stepping_tmicro_init", gt4py_config, externals)

ice4_step_limiter = compile_stencil("ice4_step_limiter", gt4py_config, externals)

ice4_stepping_heat = compile_stencil("ice4_stepping_heat", gt4py_config, externals)

ice4_mixing_ratio_step_limiter = compile_stencil("ice4_mixing_ratio_step_limiter")

ice4_state_update = compile_stencil("state_update", gt4py_config, externals)

externals_tendencies_update = compile_stencil(
    "external_tendencies_update", gt4py_config, externals
)

######### State and field declaration ######
masks = {
    "ldcompute": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
    "ldmicro": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
}

timing = {
    "t_micro": ones((nx, ny, nz), backend=BACKEND),
    "t_soft": ones((nx, ny, nz), backend=BACKEND),
    "delta_t_micro": ones((nx, ny, nz), backend=BACKEND),
    "delta_t_soft": ones((nx, ny, nz), backend=BACKEND),
}

state = {
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

external_tendencies = {
    "theta_ext_tnd": ones((nx, ny, nz), backend=BACKEND),
    "rc_ext_tnd": ones((nx, ny, nz), backend=BACKEND),
    "rr_ext_tnd": ones((nx, ny, nz), backend=BACKEND),
    "ri_ext_tnd": ones((nx, ny, nz), backend=BACKEND),
    "rs_ext_tnd": ones((nx, ny, nz), backend=BACKEND),
    "rg_ext_tnd": ones((nx, ny, nz), backend=BACKEND),
}

increments = {
    "theta_b": ones((nx, ny, nz), backend=BACKEND),
    "rc_b": ones((nx, ny, nz), backend=BACKEND),
    "rr_b": ones((nx, ny, nz), backend=BACKEND),
    "ri_b": ones((nx, ny, nz), backend=BACKEND),
    "rs_b": ones((nx, ny, nz), backend=BACKEND),
    "rg_b": ones((nx, ny, nz), backend=BACKEND),
}

internal_tnd = {
    "theta_a_tnd": ones((nx, ny, nz), backend=BACKEND),
    "rc_a_tnd": ones((nx, ny, nz), backend=BACKEND),
    "rr_a_tnd": ones((nx, ny, nz), backend=BACKEND),
    "ri_a_tnd": ones((nx, ny, nz), backend=BACKEND),
    "rs_a_tnd": ones((nx, ny, nz), backend=BACKEND),
    "rg_a_tnd": ones((nx, ny, nz), backend=BACKEND),
}

initial_state_for_soft_loop = {
    "rc_0r_t": ones((nx, ny, nz), backend=BACKEND),
    "rr_0r_t": ones((nx, ny, nz), backend=BACKEND),
    "ri_0r_t": ones((nx, ny, nz), backend=BACKEND),
    "rs_0r_t": ones((nx, ny, nz), backend=BACKEND),
    "rg_0r_t": ones((nx, ny, nz), backend=BACKEND),
}

############## t_micro_init ################

state_t_micro_init = {
    "t_micro": timing["t_micro"],
    "ldmicro": masks["ldmicro"],
}

t_micro_init(**state_t_micro_init)

outerloop_counter = 0
max_outerloop_iterations = 10

innerloop_counter = 0
max_innerloop_iterations = 10

while np.any(timing["t_micro"][...] < TSTEP):

    # Iterations limiter
    if outerloop_counter >= max_outerloop_iterations:
        break

    while np.any(masks["ldcompute"][...]):

        # Iterations limiter
        if innerloop_counter >= max_innerloop_iterations:
            break

        # 244

        # 249
        ####### ice4_stepping_heat #############

        state_stepping_heat = {
            key: state[key]
            for key in [
                "rv_t",
                "rc_t",
                "rr_t",
                "ri_t",
                "rs_t",
                "rg_t",
                "exn",
                "th_t",
                "ls_fact",
                "lv_fact",
                "t",
            ]
        }

        ice4_stepping_heat(**state_stepping_heat)

        ####### tendencies #######
        #### TODO : tendencies state + components #####

        ######### ice4_step_limiter ############################
        state_step_limiter = {
            **{
                key: state[key]
                for key in ["exn", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t", "th_t"]
            },
            **internal_tnd,
            **external_tendencies,
            **timing,
            "ldcompute": masks["ldcompute"],
        }

        ice4_step_limiter(**state_step_limiter)

        # l346 to l388
        ############ ice4_mixing_ratio_step_limiter ############
        state_mixing_ratio_step_limiter = {
            **{
                key: initial_state_for_soft_loop[key]
                for key in [
                    "rc_0r_t",
                    "rr_0r_t",
                    "ri_0r_t",
                    "rs_0r_t",
                    "rg_0r_t",
                ]
            },
            **{key: state[key] for key in ["rc_t", "rr_t", "ri_t", "rs_t", "rg_t"]},
            **{
                key: increments[key] for key in ["rc_b", "rr_b", "ri_b", "rs_b", "rg_b"]
            },
            **{
                key: internal_tnd[key]
                for key in [
                    "rc_tnd_a",
                    "rr_tnd_a",
                    "ri_tnd_a",
                    "rs_tnd_a",
                    "rg_tnd_a",
                ]
            },
            "delta_t_micro": timing["delta_t_micro"],
            "ldcompute": masks["ldcompute"],
        }

        temporaries_mixing_ratio_step_limiter = {
            "time_threshold_tmp": ones((nx, ny, nz), backend=BACKEND),
        }

        ice4_mixing_ratio_step_limiter(
            **state_mixing_ratio_step_limiter,
            **temporaries_mixing_ratio_step_limiter,
        )

        # l394 to l404
        # 4.7 new values for next iteration
        ############### ice4_state_update ######################
        state_state_update = {
            **{
                key: state[key]
                for key in ["th_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t", "ci_t"]
            },
            **increments,
            **internal_tnd,
            **{key: timing[key] for key in ["delta_t_micro", "t_micro"]},
            "ldmicro": masks["ldmicro"],
        }

        ice4_state_update(**state_state_update)

        # TODO : next loop
        lsoft = True
        innerloop_counter += 1
    outerloop_counter += 1

# l440 to l452
################ external_tendencies_update ############
# if ldext_tnd

state_external_tendencies_update = {
    **{key: state[key] for key in ["th_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t"]},
    **external_tendencies,
    "ldmicro": masks["ldmicro"],
}

externals_tendencies_update(**state_external_tendencies_update)

# ice4_correct_negativities
# ice4_sedimentation
