# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import timedelta
import datetime
from functools import cached_property
from itertools import repeat
from typing import Dict

from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from ifs_physics_common.framework.storage import managed_temporary_storage
from ifs_physics_common.utils.typingx import NDArrayLikeDict, PropertyDict
from ifs_physics_common.utils.f2py import ported_method
from ifs_physics_common.utils.numpyx import to_numpy
import numpy as np
import xarray as xr
import logging


from ice3.components.ice4_tendencies import Ice4Tendencies
from ice3.phyex_common.phyex import Phyex


# todo : merge with ice4_tendencies
class Ice4Stepping(ImplicitTendencyComponent):
    """Component for step computation"""

    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
        phyex: Phyex,
        *,
        enable_checks: bool = False,
    ) -> None:
        super().__init__(
            computational_grid, enable_checks=enable_checks, gt4py_config=gt4py_config
        )

        externals = phyex.to_externals()

        # Switch between numpy and cupy

        # Stencil collections
        self.ice4_stepping_heat = self.compile_stencil("ice4_stepping_heat", externals)
        self.ice4_step_limiter = self.compile_stencil("ice4_step_limiter", externals)
        self.ice4_mixing_ratio_step_limiter = self.compile_stencil(
            "ice4_mixing_ratio_step_limiter", externals
        )
        self.ice4_state_update = self.compile_stencil("state_update", externals)
        self.external_tendencies_update = self.compile_stencil(
            "external_tendencies_update", externals
        )
        self.tmicro_init = self.compile_stencil("ice4_stepping_tmicro_init", externals)
        self.tsoft_init = self.compile_stencil("ice4_stepping_tsoft_init", externals)
        self.ldcompute_init = self.compile_stencil(
            "ice4_stepping_ldcompute_init", externals
        )

        # Component for tendency update
        self.ice4_tendencies = Ice4Tendencies(
            self.computational_grid, self.gt4py_config, phyex
        )


    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "ldmicro": {"grid": (I, J, K), "dtype": "bool", "unit": ""},
            "rhodref": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "pabs_t": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "exn": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "cf": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "sigma_rc": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "ls_fact": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "lv_fact": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "t": {"grid": (I, J, K), "dtype": "float", "unit": ""},
        }

    @cached_property
    def _tendency_properties(self) -> PropertyDict:
        return {}

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        return {
            "ci_t": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "th_t": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rv_t": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rc_t": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rr_t": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "ri_t": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rs_t": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rg_t": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "hlc_hcf": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "hlc_hrc": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "hli_hcf": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "hli_hri": {"grid": (I, J, K), "dtype": "float", "unit": ""},
        }

    @cached_property
    def _temporaries(self) -> PropertyDict:
        return {
            "ldcompute": {"grid": (I, J, K), "dtype": "bool", "unit": ""},
            "ai": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "cj": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "ssi": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "hlc_lcf": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "hlc_lrc": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "hli_lcf": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "hli_lri": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "fr": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            # intial mixing ratios
            "rc_0r_t": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rr_0r_t": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "ri_0r_t": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rs_0r_t": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rg_0r_t": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            # increments
            "theta_b": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rv_b": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rc_b": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rr_b": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "ri_b": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rs_b": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rg_b": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            # tnd update
            "theta_a_tnd": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rv_a_tnd": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rc_a_tnd": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rr_a_tnd": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "ri_a_tnd": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rs_a_tnd": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rg_a_tnd": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            # tendances externes
            "theta_ext_tnd": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rc_ext_tnd": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rr_ext_tnd": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "ri_ext_tnd": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rs_ext_tnd": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "rg_ext_tnd": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            # timing
            "t_micro": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "delta_t_micro": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "time_threshold_tmp": {"grid": (I, J, K), "dtype": "float", "unit": ""},
        }

    @ported_method(
        from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
        from_line=214,
        to_line=438,
    )
    def array_call(
        self,
        state: NDArrayLikeDict,
        timestep: timedelta,
        out_tendencies: NDArrayLikeDict,
        out_diagnostics: NDArrayLikeDict,
        overwrite_tendencies: Dict[str, bool],
    ):

        with managed_temporary_storage(
            self.computational_grid,
            *repeat(((I, J, K), "bool"), 1),
            *repeat(((I, J, K), "float"), 36),
            gt4py_config=self.gt4py_config,
        ) as (
            # masks
            ldcompute,
            ai,
            cj,
            ssi,
            hlc_lcf,
            hlc_lrc,
            hli_lcf,
            hli_lri,
            fr,
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
            # tendances externes
            theta_ext_tnd,
            rc_ext_tnd,
            rr_ext_tnd,
            ri_ext_tnd,
            rs_ext_tnd,
            rg_ext_tnd,
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
            dt = timestep.total_seconds()

            state_tmicro_init = {"ldmicro": state["ldmicro"], "t_micro": t_micro}

            self.tmicro_init(
                **state_tmicro_init,
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,)

            outerloop_counter = 0
            max_outerloop_iterations = 10
            lsoft = False

            # l223 in f90
            _np_t_micro = to_numpy(t_micro)
            while (_np_t_micro < dt).any():

                # Translation note XTSTEP_TS == 0 is assumed implying no loops over t_soft
                innerloop_counter = 0
                max_innerloop_iterations = 10

                # Translation note : l230 to l 237 in Fortran
                self.ldcompute_init(ldcompute, t_micro)

                # Iterations limiter
                if outerloop_counter >= max_outerloop_iterations:
                    break

                while ldcompute.any():

                    # Iterations limiter
                    if innerloop_counter >= max_innerloop_iterations:
                        break

                    ####### ice4_stepping_heat #############
                    state_stepping_heat = {
                        **{
                        key: state[key]
                        for key in [
                            "exn",
                            "ls_fact",
                            "lv_fact",
                            "t",
                        ]
                    },**{
                            "tht": state["th_t"],
                            "rvt": state["rv_t"],
                            "rct": state["rc_t"],
                            "rrt": state["rr_t"],
                            "rit": state["ri_t"],
                            "rst": state["rs_t"],
                            "rgt": state["rg_t"],
                        }
                    }

                    self.ice4_stepping_heat(
                        **state_stepping_heat,
                        origin=(0, 0, 0),
                        domain=self.computational_grid.grids[I, J, K].shape,
                        validate_args=self.gt4py_config.validate_args,
                        exec_info=self.gt4py_config.exec_info,
                        )

                    ####### tendencies #######
                    state_ice4_tendencies = {
                        **{
                            key: state[key]
                            for key in [
                                "rhodref",
                                "exn",
                                "ls_fact",
                                "lv_fact",
                                "rv_t",
                                "cf",
                                "sigma_rc",
                                "ci_t",
                                "t",
                                "hlc_hcf",
                                "hlc_hrc",
                                "hli_hcf",
                                "hli_hri",
                            ]
                        },
                        **{
                              "tht": state["th_t"],
                              "rvt": state["rv_t"],
                              "rct": state["rc_t"],
                              "rrt": state["rr_t"],
                              "rit": state["ri_t"],
                              "rst": state["rs_t"],
                              "rgt": state["rg_t"],
                          },
                        **{"pres": state["pabs_t"]},
                        **{
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
                            "ai": ai,
                            "cj": cj,
                            "ssi": ssi,
                            "hlc_lcf": hlc_lcf,
                            "hlc_lrc": hlc_lrc,
                            "hli_lcf": hli_lcf,
                            "hli_lri": hli_lri,
                            "fr": fr,
                        },
                    }

                    state_tendencies_xr = {
                        **{
                            key: xr.DataArray(
                                data=field,
                                dims=["x", "y", "z"],
                                coords={
                                    "x": range(field.shape[0]),
                                    "y": range(field.shape[1]),
                                    "z": range(field.shape[2]),
                                },
                                name=f"{key}",
                            )
                            for key, field in state_ice4_tendencies.items()
                        },
                        "time": datetime.datetime(year=2024, month=1, day=1),
                    }

                    _, _ = self.ice4_tendencies(
                        ldsoft=True, 
                        state=state_tendencies_xr, 
                        timestep=timestep
                    )

                    # Translation note : l277 to l283 omitted, no external tendencies in AROME
                    # todo: ice4_step_limiter
                    #       ice4_mixing_ratio_step_limiter
                    #       ice4_state_update in one stencil
                    ######### ice4_step_limiter ############################
                    state_step_limiter = {
                        "tht": state["th_t"],
                        "rct": state["rc_t"],
                        "rrt": state["rr_t"],
                        "rit": state["ri_t"],
                        "rst": state["rs_t"],
                        "rgt": state["rg_t"],
                        "exn": state["exn"]
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

                    self.ice4_step_limiter(
                        **state_step_limiter, 
                        **tmps_step_limiter,
                        origin=(0, 0, 0),
                        domain=self.computational_grid.grids[I, J, K].shape,
                        validate_args=self.gt4py_config.validate_args,
                        exec_info=self.gt4py_config.exec_info,
                    )

                    # l346 to l388
                    ############ ice4_mixing_ratio_step_limiter ############
                    state_mixing_ratio_step_limiter = {
                        "rct": state["rc_t"],
                        "rrt": state["rr_t"],
                        "rit": state["ri_t"],
                        "rst": state["rs_t"],
                        "rgt": state["rg_t"],
                        "cit": state["ci_t"],
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
                        origin=(0, 0, 0),
                        domain=self.computational_grid.grids[I, J, K].shape,
                        validate_args=self.gt4py_config.validate_args,
                        exec_info=self.gt4py_config.exec_info,
                    )

                    # l394 to l404
                    # 4.7 new values for next iteration
                    ############### ice4_state_update ######################
                    state_state_update = {
                        "tht": state["th_t"],
                        "rct": state["rc_t"],
                        "rrt": state["rr_t"],
                        "rit": state["ri_t"],
                        "rst": state["rs_t"],
                        "rgt": state["rg_t"],
                        "cit": state["ci_t"],
                        "ldmicro": state["ldmicro"],

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

                    self.ice4_state_update(
                        **state_state_update, 
                        **tmps_state_update,
                        origin=(0, 0, 0),
                        domain=self.computational_grid.grids[I, J, K].shape,
                        validate_args=self.gt4py_config.validate_args,
                        exec_info=self.gt4py_config.exec_info,
                    )

                    # TODO : next loop
                    lsoft = True
                    innerloop_counter += 1
                outerloop_counter += 1

            _np_t_micro = to_numpy(t_micro)
            # l440 to l452
            ################ external_tendencies_update ############
            # if ldext_tnd

            state_external_tendencies_update =  {
                **{
                key: state[key]
                for key in [

                    "ldmicro",
                ]
            }, **{
                    "tht": state["th_t"],
                    "rct": state["rc_t"],
                    "rrt": state["rr_t"],
                    "rit": state["ri_t"],
                    "rst": state["rs_t"],
                    "rgt": state["rg_t"],
                }
            }

            tmps_external_tendencies_update = {
                "theta_tnd_ext": theta_ext_tnd,
                "rc_tnd_ext": rc_ext_tnd,
                "rr_tnd_ext": rr_ext_tnd,
                "ri_tnd_ext": ri_ext_tnd,
                "rs_tnd_ext": rs_ext_tnd,
                "rg_tnd_ext": rg_ext_tnd,
            }

            self.external_tendencies_update(
                **state_external_tendencies_update, 
                **tmps_external_tendencies_update,
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )
