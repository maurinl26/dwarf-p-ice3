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
import numpy as np


from ice3_gt4py.components.ice4_tendencies import Ice4Tendencies
from ice3_gt4py.phyex_common.phyex import Phyex


class Ice4Stepping(ImplicitTendencyComponent):
    """Component for step computation"""

    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
        phyex: Phyex,
        *,
        enable_checks: bool = True,
    ) -> None:
        super().__init__(
            computational_grid, enable_checks=enable_checks, gt4py_config=gt4py_config
        )

        externals = phyex.to_externals()

        # Stencil collections
        self.tmicro_init = compile_stencil("ice4_stepping_tmicro_init", externals)
        self.tsoft_init = compile_stencil("ice4_stepping_tsoft_init", externals)
        self.ldcompute_init = compile_stencil("ice4_stepping_ldcompute_init", externals)
        self.ice4_stepping_heat = compile_stencil("ice4_stepping_heat", externals)
        self.ice4_step_limiter = compile_stencil("step_limiter", externals)
        self.ice4_mixing_ratio_step_limiter = compile_stencil(
            "mixing_ratio_step_limiter", externals
        )
        self.ice4_state_update = compile_stencil("state_update", externals)
        self.external_tendencies_update = compile_stencil(
            "external_tendencies_update", externals
        )

        # Component for tendency update
        self.ice4_tendencies = Ice4Tendencies(
            self.computational_grid, self.gt4py_config, phyex
        )

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "t_micro": {"grid": (I, J, K), "units": ""},
            "ldmicro": {"grid": (I, J, K), "units": ""},
            "exn": {"grid": (I, J, K), "units": ""},
            "th_t": {"grid": (I, J, K), "units": ""},
            "ls_fact": {"grid": (I, J, K), "units": ""},
            "lv_fact": {"grid": (I, J, K), "units": ""},
            "t": {"grid": (I, J, K), "units": ""},
            "rv_t": {"grid": (I, J, K), "units": ""},
            "rc_t": {"grid": (I, J, K), "units": ""},
            "rr_t": {"grid": (I, J, K), "units": ""},
            "ri_t": {"grid": (I, J, K), "units": ""},
            "rs_t": {"grid": (I, J, K), "units": ""},
            "rg_t": {"grid": (I, J, K), "units": ""},
            # external tendencies
            "theta_ext_tnd": {"grid": (I, J, K), "units": ""},
            "rc_ext_tnd": {"grid": (I, J, K), "units": ""},
            "rr_ext_tnd": {"grid": (I, J, K), "units": ""},
            "ri_ext_tnd": {"grid": (I, J, K), "units": ""},
            "rs_ext_tnd": {"grid": (I, J, K), "units": ""},
            "rg_ext_tnd": {"grid": (I, J, K), "units": ""},
        }

    @cached_property
    def _tendency_properties(self) -> PropertyDict:
        return {}

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        return {}

    @cached_property
    def _temporaries(self) -> PropertyDict:
        return {}

    @ported_method(
        from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
        from_line=214,
        to_line=438,
    )
    def array_call(self, state: NDArrayLikeDict, timestep: timedelta):

        with managed_temporary_storage(
            self.computational_grid,
            *repeat(((I, J, K), "bool"), 1),
            *repeat(((I, J, K), "float"), 17),
            gt4py_config=self.gt4py_config,
        ) as (
            # masks
            ldcompute,
            # intial mixing ratios
            rc_0r_t,
            rr_0r_t,
            ri_0r_t,
            rs_0r_t,
            rg_0r_t,
            # increments
            theta_b,
            rv_b,
            rc_b,
            rr_b,
            ri_b,
            rs_b,
            rg_b,
            # tnd update
            theta_a_tnd,
            rv_a_tnd,
            rc_a_tnd,
            rr_a_tnd,
            ri_a_tnd,
            rs_a_tnd,
            rg_a_tnd,
            # timing
            t_micro,
            delta_t_micro,
            time_threshold_tmp,
        ):
            # Translation note : Ice4Stepping is implemented assuming PARAMI%XTSTEP_TS = 0
            #                   l225 to l229 omitted
            #                   l334 to l341 omitted
            #                   l174 to l178 omitted

            ############## t_micro_init ################

            self.tmicro_init(t_micro, state["ldmicro"])

            outerloop_counter = 0
            max_outerloop_iterations = 10

            # l223 in f90
            while np.any(t_micro[...] < timestep):

                # Translation note XTSTEP_TS == 0 is assumed implying no loops over t_soft
                innerloop_counter = 0
                max_innerloop_iterations = 10

                # Translation note : l230 to l 237 in Fortran
                self.ldcompute_init(ldcompute, t_micro)

                # Iterations limiter
                if outerloop_counter >= max_outerloop_iterations:
                    break

                while np.any(ldcompute[...]):

                    # Iterations limiter
                    if innerloop_counter >= max_innerloop_iterations:
                        break

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

                    self.ice4_stepping_heat(**state_stepping_heat)

                    ####### tendencies #######
                    state_ice4_tendencies = {
                        **{
                            key: state[key]
                            for key in [
                                "pres",
                                "rhodref",
                                "exn",
                                "ls_fact",
                                "lv_fact",
                                "rv_t",
                                "cf",
                                "sigma_rc",
                                "ci_t",
                                "ai",
                                "cj",
                                "ssi",
                                "t",
                                "tht",
                                "rv_t",
                                "rc_t",
                                "rr_t",
                                "ri_t",
                                "rs_t",
                                "rg_t",
                                "hlc_hcf",
                                "hlc_lcf",
                                "hlc_hrc",
                                "hlc_lrc",
                                "hli_hcf",
                                "hli_lcf",
                                "hli_hri",
                                "hli_lri",
                                "fr",
                            ]
                        }
                    }

                    tmps_ice4_tendencies = {
                        "ldcompute": ldcompute,
                        "theta_tnd": theta_a_tnd,
                        "rv_tnd": rv_a_tnd,
                        "rc_tnd": rc_a_tnd,
                        "rr_tnd": rr_a_tnd,
                        "ri_tnd": ri_a_tnd,
                        "rs_tnd": rs_a_tnd,
                        "rg_tnd": rg_a_tnd,
                        "theta_increment": theta_b,
                        "rv_increment": rv_b,
                        "rc_increment": rc_b,
                        "rr_increment": rr_b,
                        "ri_increment": ri_b,
                        "rs_increment": rs_b,
                        "rg_increment": rg_b,
                    }

                    self.ice4_tendencies(
                        ldsoft=lsoft, **state_ice4_tendencies, **tmps_ice4_tendencies
                    )

                    # Translation note : l277 to l283 omitted, no external tendencies in AROME

                    ######### ice4_step_limiter ############################
                    state_step_limiter = {
                        key: state[key]
                        for key in [
                            "exn",
                            "rc_t",
                            "rr_t",
                            "ri_t",
                            "rs_t",
                            "rg_t",
                            "tht",
                        ]
                    }

                    tmps_step_limiter = {
                        "t_micro": t_micro,
                        "delta_t_micro": delta_t_micro,
                        "ldcompute": ldcompute,
                        "theta_a_tnd": theta_a_tnd,
                        "rc_a_tnd": rc_a_tnd,
                        "rr_a_tnd": rr_a_tnd,
                        "ri_a_tnd": ri_a_tnd,
                        "rs_a_tnd": rs_a_tnd,
                        "rg_a_tnd": rg_a_tnd,
                        "theta_b": theta_b,
                        "rc_b": rc_b,
                        "rr_b": rr_b,
                        "ri_b": ri_b,
                        "rs_b": rs_b,
                        "rg_b": rg_b,
                    }

                    self.ice4_step_limiter(**state_step_limiter, **tmps_step_limiter)

                    # l346 to l388
                    ############ ice4_mixing_ratio_step_limiter ############
                    state_mixing_ratio_step_limiter = {
                        key: state[key]
                        for key in ["rc_t", "rr_t", "ri_t", "rs_t", "rg_t"]
                    }

                    temporaries_mixing_ratio_step_limiter = {
                        "ldcompute": ldcompute,
                        "theta_a_tnd": theta_a_tnd,
                        "rc_a_tnd": rc_a_tnd,
                        "rr_a_tnd": rr_a_tnd,
                        "ri_a_tnd": ri_a_tnd,
                        "rs_a_tnd": rs_a_tnd,
                        "rg_a_tnd": rg_a_tnd,
                        "theta_b": theta_b,
                        "rc_b": rc_b,
                        "rr_b": rr_b,
                        "ri_b": ri_b,
                        "rs_b": rs_b,
                        "rg_b": rg_b,
                        "rc_0r_t": rc_0r_t,
                        "rr_0r_t": rr_0r_t,
                        "ri_0r_t": ri_0r_t,
                        "rs_0r_t": rs_0r_t,
                        "rg_0r_t": rg_0r_t,
                        "delta_t_micro": delta_t_micro,
                    }

                    self.ice4_mixing_ratio_step_limiter(
                        **state_mixing_ratio_step_limiter,
                        **temporaries_mixing_ratio_step_limiter,
                    )

                    # l394 to l404
                    # 4.7 new values for next iteration
                    ############### ice4_state_update ######################
                    state_state_update = {
                        key: state[key]
                        for key in [
                            "tht",
                            "rc_t",
                            "rr_t",
                            "ri_t",
                            "rs_t",
                            "rg_t",
                            "ci_t",
                            "ldmicro",
                        ]
                    }

                    tmps_state_update = {
                        "theta_a_tnd": theta_a_tnd,
                        "rc_a_tnd": rc_a_tnd,
                        "rr_a_tnd": rr_a_tnd,
                        "ri_a_tnd": ri_a_tnd,
                        "rs_a_tnd": rs_a_tnd,
                        "rg_a_tnd": rg_a_tnd,
                        "theta_b": theta_b,
                        "rc_b": rc_b,
                        "rr_b": rr_b,
                        "ri_b": ri_b,
                        "rs_b": rs_b,
                        "rg_b": rg_b,
                        "delta_t_micro": delta_t_micro,
                        "t_micro": t_micro,
                    }

                    self.ice4_state_update(**state_state_update, **tmps_state_update)

                    # TODO : next loop
                    lsoft = True
                    innerloop_counter += 1
                outerloop_counter += 1

            # l440 to l452
            ################ external_tendencies_update ############
            # if ldext_tnd

            state_external_tendencies_update = {
                key: state[key]
                for key in [
                    "tht",
                    "rc_t",
                    "rr_t",
                    "ri_t",
                    "rs_t",
                    "rg_t",
                    "ldmicro",
                    "theta_ext_tnd",
                    "rc_ext_tnd",
                    "rr_ext_tnd",
                    "ri_ext_tnd",
                    "rs_ext_tnd",
                    "rg_ext_tnd",
                ]
            }

            self.external_tendencies_update(**state_external_tendencies_update)

            # TODO : add following stencils
            # ice4_correct_negativities
            # ice4_sedimentation
