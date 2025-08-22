# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
import logging
import sys
from datetime import timedelta
from functools import cached_property
from itertools import repeat
from typing import Dict

import xarray as xr
from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from ifs_physics_common.framework.storage import managed_temporary_storage
from ifs_physics_common.utils.f2py import ported_method
from ifs_physics_common.utils.typingx import NDArrayLikeDict, PropertyDict

# from ice3_gt4py.components.ice4_stepping import Ice4Stepping
from ice3_gt4py.components.ice4_tendencies import Ice4Tendencies

from ice3_gt4py.phyex_common.param_ice import (
    Sedim,
)
from ice3_gt4py.phyex_common.phyex import Phyex

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


class RainIce(ImplicitTendencyComponent):
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

        self.phyex = phyex
        externals = self.phyex.to_externals()

        # Keys
        SEDIM = self.phyex.param_icen.SEDIM

        # 1. Generalites
        self.rain_ice_thermo = self.compile_stencil("rain_ice_thermo", externals)

        self.rain_ice_mask = self.compile_stencil("rain_ice_mask", externals)

        # 3. Initial values saving
        self.initial_values_saving = self.compile_stencil(
            "initial_values_saving", externals
        )


        # 4.2 Computes precipitation fraction
        self.ice4_precipitation_fraction_sigma = self.compile_stencil(
            "ice4_precipitation_fraction_sigma", externals
        )
        self.ice4_precipitation_fraction_liquid_content = self.compile_stencil(
            "ice4_precipitation_fraction_liquid_content", externals
        )
        self.ice4_compute_pdf = self.compile_stencil("ice4_compute_pdf", externals)
        self.ice4_rainfr_vert = self.compile_stencil("ice4_rainfr_vert", externals)

        # 8. Total tendencies
        # 8.1 Total tendencies limited by available species
        self.total_tendencies = self.compile_stencil(
            "rain_ice_total_tendencies", externals
        )
        # 8.2 Negative corrections
        self.ice4_correct_negativities = self.compile_stencil(
            "ice4_correct_negativities", externals
        )

        # 9. Compute the sedimentation source
        if SEDIM == Sedim.STAT.value:
            self.sedimentation = self.compile_stencil(
                "statistical_sedimentation", externals
            )
        elif SEDIM == Sedim.SPLI.value:
            self.sedimentation = self.compile_stencil("upwind_sedimentation", externals)
        else:
            raise KeyError(
                f"Key not in {[option.name for option in Sedim]} for sedimentation"
            )

        self.rain_fraction_sedimentation = self.compile_stencil(
            "rain_fraction_sedimentation", externals
        )

        # 10 Compute the fog deposition
        self.fog_deposition = self.compile_stencil("fog_deposition", externals)
        
        #####################################################################
        ######################### Stepping ##################################
        #####################################################################
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
        ###################################################################

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "exn": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "dzz": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "t": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "ssi": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "rhodj": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "rhodref": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "pabs_t": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "exnref": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "ci_t": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "cldfr": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "th_t": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "rv_t": {
                "grid": (I, J, K),
                "units": "",
                "irr": 0,
                "dtype": "float",
            },
            "rc_t": {
                "grid": (I, J, K),
                "units": "",
                "irr": 1,
                "dtype": "float",
            },
            "rr_t": {
                "grid": (I, J, K),
                "units": "",
                "irr": 2,
                "dtype": "float",
            },
            "ri_t": {
                "grid": (I, J, K),
                "units": "",
                "irr": 3,
                "dtype": "float",
            },
            "rs_t": {
                "grid": (I, J, K),
                "units": "",
                "irr": 4,
                "dtype": "float",
            },
            "rg_t": {
                "grid": (I, J, K),
                "units": "",
                "irr": 5,
                "dtype": "float",
            },
            "ths": {
                "grid": (I, J, K),
                "units": "",
                "irr": 0,
                "dtype": "float",
            },
            "rvs": {
                "grid": (I, J, K),
                "units": "",
                "irr": 0,
                "dtype": "float",
            },
            "rcs": {
                "grid": (I, J, K),
                "units": "",
                "irr": 1,
                "dtype": "float",
            },
            "rrs": {
                "grid": (I, J, K),
                "units": "",
                "irr": 2,
                "dtype": "float",
            },
            "ris": {
                "grid": (I, J, K),
                "units": "",
                "irr": 3,
                "dtype": "float",
            },
            "rss": {
                "grid": (I, J, K),
                "units": "",
                "irr": 4,
                "dtype": "float",
            },
            "rgs": {
                "grid": (I, J, K),
                "units": "",
                "irr": 5,
                "dtype": "float",
            },
            "fpr_c": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "fpr_r": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "fpr_i": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "fpr_s": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "fpr_g": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "inprc": {"grid": (I, J), "units": "", "dtype": "float"},
            "inprr": {"grid": (I, J), "units": "", "dtype": "float"},
            "inprs": {"grid": (I, J), "units": "", "dtype": "float"},
            "inprg": {"grid": (I, J), "units": "", "dtype": "float"},
            "evap3d": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "indep": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "rainfr": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "sigs": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "pthvrefzikb": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "hlc_hcf": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "hlc_lcf": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "hlc_hrc": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "hlc_lrc": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "hli_hcf": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "hli_lcf": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "hli_hri": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "hli_lri": {"grid": (I, J, K), "units": "", "dtype": "float"},
            # Optional
            "fpr": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "sea": {
                "grid": (I, J),
                "units": "",
                "dtype": "float",
            },
            "town": {
                "grid": (I, J),
                "units": "",
                "dtype": "float",
            },
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

    @cached_property
    def _fortran_names(self) -> PropertyDict:
        return {
        "exn": {"fortran_name": "PEXNREF"},
        "dzz": {"fortran_name": "PDZZ"},
        "t": {"fortran_name": "PT"},
        "ssi": {"fortran_name": "PSSI"},
        "rhodj": {"fortran_name": "PRHODJ"},
        "rhodref": {"fortran_name": "PRHODREF"},
        "pabs_t": {"fortran_name": "PPABSM"},
        "exnref": {"fortran_name": "PEXNREF"},
        "ci_t": {"fortran_name": "PCIT"},
        "cldfr": {"fortran_name": "PCLDFR"},
        "th_t": {"fortran_name": "PTHT"},
        "rv_t": {
            "fortran_name": "PRT",
            "irr": 0,
        },
        "rc_t": {
            "fortran_name": "PRT",
            "irr": 1,
        },
        "rr_t": {
            "fortran_name": "PRT",
            "irr": 2,
        },
        "ri_t": {
            "fortran_name": "PRT",
            "irr": 3,
        },
        "rs_t": {
            "fortran_name": "PRT",
            "irr": 4,
        },
        "rg_t": {
            "fortran_name": "PRT",
            "irr": 5,
        },
        "ths": {
            "fortran_name": "PTHS",
            "irr": 0,
        },
        "rvs": {
            "fortran_name": "PRS",
            "irr": 0,
        },
        "rcs": {
            "fortran_name": "PRS",
            "irr": 1,
        },
        "rrs": {
            "fortran_name": "PRS",
            "irr": 2,
        },
        "ris": {
            "fortran_name": "PRS",
            "irr": 3,
        },
        "rss": {
            "fortran_name": "PRS",
            "irr": 4,
        },
        "rgs": {
            "fortran_name": "PRS",
            "irr": 5,
        },
        "fpr_c": {"fortran_name": "fpr"},
        "fpr_r": {"fortran_name": "fpr"},
        "fpr_i": {"grid": (I, J, K), "units": "", "dtype": "float"},
        "fpr_s": {"grid": (I, J, K), "units": "", "dtype": "float"},
        "fpr_g": {"grid": (I, J, K), "units": "", "dtype": "float"},
        "inprc": {"grid": (I, J), "units": "", "dtype": "float"},
        "inprr": {"grid": (I, J), "units": "", "dtype": "float"},
        "inprs": {"grid": (I, J), "units": "", "dtype": "float"},
        "inprg": {"grid": (I, J), "units": "", "dtype": "float"},
        "evap3d": {"grid": (I, J, K), "units": "", "dtype": "float"},
        "indep": {"grid": (I, J, K), "units": "", "dtype": "float"},
        "rainfr": {"grid": (I, J, K), "units": "", "dtype": "float"},
        "sigs": {"fortran_name": "PSIGS"},
        "pthvrefzikb": {"fortran_name": "pthvrefzikb"},
        "hlc_hcf": {"fortran_name": "PHLC_HRC"},
        "hlc_lcf": {"fortran_name": "PHLC_HCF",},
        "hlc_hrc": {"fortran_name": "PHLC_HRC"},
        "hlc_lrc": {"fortran_name": "PHLC_LRC"},
        "hli_hcf": {"fortran_name": "PHLI_HCF",},
        "hli_lcf": {"fortran_name": "PHL_HCF"},
        "hli_hri": {"fortran_name": "PHLI_HRI",},
        "hli_lri": {"fortran_name": "PHLI_LRI"},
        # Optional
        "fpr": {"fortran_name": None,},
        "sea": {"fortran_name": "PSEA",},
        "town": {"fortran_name": "PTOWN",},
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
            *repeat(((I, J, K), "bool"), 2),
            *repeat(((I, J, K), "float"), 50),
            *repeat(((I, J), "float"), 2),
            gt4py_config=self.gt4py_config,
        ) as (
            ldmicro,
            ldcompute,
            rvheni,
            lv_fact,
            ls_fact,
            wr_th,
            wr_v,
            wr_c,
            wr_r,
            wr_i,
            wr_s,
            wr_g,
            remaining_time,
            delta_t_micro,
            t_micro,
            delta_t_soft,
            t_soft,
            theta_a_tnd,
            rv_a_tnd,
            rc_a_tnd,
            rr_a_tnd,
            ri_a_tnd,
            rs_a_tnd,
            rg_a_tnd,
            theta_b,
            rv_b,
            rc_b,
            rr_b,
            ri_b,
            rs_b,
            rg_b,
            theta_ext_tnd,
            rc_ext_tnd,
            rr_ext_tnd,
            ri_ext_tnd,
            rs_ext_tnd,
            rg_ext_tnd,
            rc_0r_t,
            rr_0r_t,
            ri_0r_t,
            rs_0r_t,
            rg_0r_t,
            ai,
            cj,
            ssi,
            hlc_lcf,
            hlc_lrc,
            hli_lcf,
            hli_lri,
            fr,
            sigma_rc,
            cf,
            w3d,
            inpri,
        ):

            # KEYS
            LSEDIM_AFTER = self.phyex.param_icen.LSEDIM_AFTER
            LDEPOSC = self.phyex.param_icen.LDEPOSC

            # 1. Generalites
            state_rain_ice_thermo = {
                **{
                    key: state[key]
                    for key in [
                        "exn",
                        "th_t",
                        "rv_t",
                        "rc_t",
                        "rr_t",
                        "ri_t",
                        "rs_t",
                        "rg_t",
                    ]
                },
                **{
                    "ls_fact": ls_fact,
                    "lv_fact": lv_fact,
                },
            }
            self.rain_ice_thermo(
                **state_rain_ice_thermo,
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,)
            
            # Compute the mask 
            state_rain_ice_mask = {
                **{
                    key: state[key]
                    for key in [
                        "rc_t",
                        "ri_t",
                        "rr_t",
                        "rs_t",
                        "rg_t"
                    ]
                },
                **{
                    "ldmicro": ldmicro
                }                
            }
            
            self.rain_ice_mask(
                **state_rain_ice_mask,
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info
            )

            # 2. Compute the sedimentation source
            state_sed = {
                key: state[key]
                for key in [
                    "rhodref",
                    "dzz",
                    "pabs_t",
                    "th_t",
                    "rcs",
                    "rrs",
                    "ris",
                    "rss",
                    "rgs",
                    "sea",
                    "town",
                    "fpr_c",
                    "fpr_r",
                    "fpr_i",
                    "fpr_s",
                    "fpr_g",
                    "inprr",
                    "inprc",
                    "inprs",
                    "inprg",
                ]
            }

            tmps_sedim = {"inpri": inpri}

            if not LSEDIM_AFTER:
                self.sedimentation(
                    **state_sed, 
                    **tmps_sedim,
                    origin=(0, 0, 0),
                    domain=self.computational_grid.grids[I, J, K].shape,
                    validate_args=self.gt4py_config.validate_args,
                    exec_info=self.gt4py_config.exec_info,)

            state_initial_values_saving = {
                key: state[key]
                for key in [
                    "th_t",
                    "rv_t",
                    "rc_t",
                    "rr_t",
                    "ri_t",
                    "rs_t",
                    "rg_t",
                    "evap3d",
                    "rainfr",
                ]
            }
            tmps_initial_values_saving = {
                "wr_th": wr_th,
                "wr_v": wr_v,
                "wr_c": wr_c,
                "wr_r": wr_r,
                "wr_i": wr_i,
                "wr_s": wr_s,
                "wr_g": wr_g,
            }
            self.initial_values_saving(
                **state_initial_values_saving, 
                **tmps_initial_values_saving,
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )

            # 5. Tendencies computation
            
            # ice4_stepping handles the tendency update with double while loop
            logging.info("Call to stepping")
            # 5. Tendencies computation
            # Translation note : rain_ice.F90 calls Ice4Stepping inside Ice4Pack packing operations
        
            # Stepping replaced by its stencils + tendencies
            state_tmicro_init = {
                "ldmicro": ldmicro,
                "t_micro": t_micro
                }

            self.tmicro_init(
                **state_tmicro_init,
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
                )

            outerloop_counter = 0
            max_outerloop_iterations = 10
            lsoft = False

            # l223 in f90
            # _np_t_micro = to_numpy(t_micro)            
            dt = timestep.total_seconds()
            
            logging.info("First loop")
            while (t_micro < dt).any():
                
                logging.info(f"type, t_micro {type(t_micro)}, {type(t_micro[0, 0, 0])}")
                logging.info(f"ldcompute, ldcompute {type(ldcompute)}, {type(ldcompute[0, 0, 0])}")                
                logging.info(f"type, th_t {type(state["th_t"])}")

                # Translation note XTSTEP_TS == 0 is assumed implying no loops over t_soft
                innerloop_counter = 0
                max_innerloop_iterations = 10
                
                logging.info(f"ldcompute {ldcompute}")

                # Translation note : l230 to l237 in Fortran
                self.ldcompute_init(ldcompute, t_micro)
                
                logging.info(f"ldcompute {ldcompute}")

                # Iterations limiter
                if outerloop_counter >= max_outerloop_iterations:
                    break

                logging.info("Second loop")
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
                            "t",
                            "th_t",
                            "rv_t",
                            "rc_t",
                            "rr_t",
                            "ri_t",
                            "rs_t",
                            "rg_t"
                        ]
                        },**{
                            "lv_fact": lv_fact,
                            "ls_fact": ls_fact,
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
                                "rv_t",
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
                            "ls_fact": ls_fact,
                            "lv_fact": lv_fact,
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
                            "cf": cf,
                            "sigma_rc": sigma_rc,
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

                    # logging.info("Call tendencies")
                    # _, _ = self.ice4_tendencies(
                    #     state=state_tendencies_xr, 
                    #     timestep=timestep
                    # )
                    
                    logging.info(f"ldcompute {ldcompute}")

                    # Translation note : l277 to l283 omitted, no external tendencies in AROME
                    # TODO : ice4_step_limiter
                    #       ice4_mixing_ratio_step_limiter
                    #       ice4_state_update in one stencil
                    ######### ice4_step_limiter ############################
                    state_step_limiter = {
                        **{
                            key: state[key] 
                        for key in [
                            "th_t",
                            "rc_t",
                            "rr_t",
                            "ri_t",
                            "rs_t",
                            "rg_t",
                            "exn"
                        ]
                        },
                        **{
                            "t_micro": t_micro,
                            "t_soft": t_soft,
                            "delta_t_micro": delta_t_micro,
                            "delta_t_soft": delta_t_soft,
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
                            "theta_ext_tnd": theta_ext_tnd,
                            "rc_ext_tnd": rc_ext_tnd,
                            "rr_ext_tnd": rr_ext_tnd,
                            "ri_ext_tnd": ri_ext_tnd,
                            "rs_ext_tnd": rs_ext_tnd,
                            "rg_ext_tnd": rg_ext_tnd,
                        } 
                    }
                    
                    logging.info("Call step limiter")
                    self.ice4_step_limiter(
                        **state_step_limiter, 
                        origin=(0, 0, 0),
                        domain=self.computational_grid.grids[I, J, K].shape,
                        validate_args=self.gt4py_config.validate_args,
                        exec_info=self.gt4py_config.exec_info,
                    )

                    # l346 to l388
                    ############ ice4_mixing_ratio_step_limiter ############
                    logging.info(f"ldcompute : {ldcompute}")
                    state_mixing_ratio_step_limiter = {
                        **{
                            state[key] for key in [
                                "rc_t",
                                "rr_t",
                                "ri_t",
                                "rs_t",
                                "rg_t",
                                "ci_t"
                            ]
                        },
                        **{
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
                    }
                    
                    logging.info("Call mixing ratio step limiter")
                    self.ice4_mixing_ratio_step_limiter(
                        **state_mixing_ratio_step_limiter,
                        origin=(0, 0, 0),
                        domain=self.computational_grid.grids[I, J, K].shape,
                        validate_args=self.gt4py_config.validate_args,
                        exec_info=self.gt4py_config.exec_info,
                    )

                    # l394 to l404
                    # 4.7 new values for next iteration
                    ############### ice4_state_update ######################
                    state_state_update = {
                        **{
                           key: state[key] for key in [
                                "th_t",
                                "rc_t",
                                "rr_t",
                                "ri_t",
                                "rs_t",
                                "rg_t",
                                "ci_t",
                           ] 
                        },
                        **{
                            "ldmicro": ldmicro,
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
                    }

                    self.ice4_state_update(
                        **state_state_update, 
                        origin=(0, 0, 0),
                        domain=self.computational_grid.grids[I, J, K].shape,
                        validate_args=self.gt4py_config.validate_args,
                        exec_info=self.gt4py_config.exec_info,
                    )

                    # TODO : next loop
                    lsoft = True
                    innerloop_counter += 1
                    logging.info("Loop 2 end")
                    
                outerloop_counter += 1
                
            logging.info("Loop 1 end")

            # l440 to l452
            ################ external_tendencies_update ############
            # if ldext_tnd

            state_external_tendencies_update =  {
                **{
                key: state[key]
                for key in [
                    "th_t",
                    "rc_t",
                    "rr_t",
                    "ri_t",
                    "rs_t",
                    "rg_t",
                    ]
            }, **{
                    "ldmicro": ldmicro,
                    "theta_tnd_ext": theta_ext_tnd,
                    "rc_tnd_ext": rc_ext_tnd,
                    "rr_tnd_ext": rr_ext_tnd,
                    "ri_tnd_ext": ri_ext_tnd,
                    "rs_tnd_ext": rs_ext_tnd,
                    "rg_tnd_ext": rg_ext_tnd,
                }
            }


            self.external_tendencies_update(
                **state_external_tendencies_update, 
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )
            # end of stepping

            # 8. Total tendencies
            # 8.1 Total tendencies limited by available species
            state_total_tendencies = {
                **{
                key: state[key]
                for key in [
                    "exnref",
                    "ths",
                    "rvs",
                    "rcs",
                    "rrs",
                    "ris",
                    "rss",
                    "rgs",
                    "rv_t",
                    "rc_t",
                    "rr_t",
                    "ri_t",
                    "rs_t",
                    "rg_t",
                ]
                },
                **{
                "rvheni": rvheni,
                "ls_fact": ls_fact,
                "lv_fact": lv_fact,
                "wr_th": wr_th,
                "wr_v": wr_v,
                "wr_c": wr_c,
                "wr_r": wr_r,
                "wr_i": wr_i,
                "wr_s": wr_s,
                "wr_g": wr_g,
                }
            }

            self.total_tendencies(
                **state_total_tendencies, 
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,)

            # 8.2 Negative corrections
            state_neg = {
                key: state[key]
                for key in [
                    "th_t",
                    "rv_t",
                    "rc_t",
                    "rr_t",
                    "ri_t",
                    "rs_t",
                    "rg_t",
                ]
            }
            tmps_neg = {"lv_fact": lv_fact, "ls_fact": ls_fact}
            self.ice4_correct_negativities(
                **state_neg, 
                **tmps_neg,
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,)

            # 9. Compute the sedimentation source
            if LSEDIM_AFTER:
                # sedimentation switch is handled in initialisation
                # self.sedimentation is can be either statistical_sedimentation or upwind_sedimentation
                self.sedimentation(**state_sed, **tmps_sedim)

                state_frac_sed = {
                    **{key: state[key] for key in ["rrs", "rss", "rgs"]},
                    **{"wr_r": wr_r, "wr_s": wr_s, "wr_g": wr_g},
                }
                self.rain_fraction_sedimentation(**state_frac_sed)

                state_rainfr = {**{key: state[key] for key in ["prfr", "rr_t", "rs_t"]}}
                self.ice4_rainfr_vert(
                    **state_rainfr,
                    origin=(0, 0, 0),
                    domain=self.computational_grid.grids[I, J, K].shape,
                    validate_args=self.gt4py_config.validate_args,
                    exec_info=self.gt4py_config.exec_info,)

            # 10 Compute the fog deposition
            if LDEPOSC:
                state_fog = {
                    key: state[key]
                    for key in ["rcs", "rc_t", "rhodref", "dzz", "inprc"]
                }
                self.fog_deposition(
                    **state_fog,
                    origin=(0, 0, 0),
                    domain=self.computational_grid.grids[I, J, K].shape,
                    validate_args=self.gt4py_config.validate_args,
                    exec_info=self.gt4py_config.exec_info,)

