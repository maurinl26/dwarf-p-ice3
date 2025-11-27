# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
import logging
from datetime import timedelta
from functools import partial
from itertools import repeat
from typing import Dict, Tuple
from numpy.typing import NDArray

import xarray as xr
from gt4py.cartesian.gtscript import stencil

from ..components.ice4_tendencies import Ice4Tendencies
from ..phyex_common.ice_parameters import Sedim
from ..phyex_common.phyex import Phyex
from ..utils.env import DTYPES, BACKEND
from ..utils.storage import managed_temporaries

log = logging.getLogger(__name__)


class RainIce:
    """Component for step computation"""

    def __init__(
        self,
        phyex: Phyex = Phyex("AROME"),
        backend: str = BACKEND,
        dtypes: Dict = DTYPES,
    ) -> None:

        self.phyex = phyex
        self.backend = backend

        compile_stencil = partial(
            stencil,
            backend=backend,
            dtypes=dtypes,
            externals=phyex.externals
        )

        from ..stencils.ice4_compute_pdf import ice4_compute_pdf
        from ..stencils.ice4_correct_negativities import \
            ice4_correct_negativities
        from ..stencils.ice4_tendencies import ice4_total_tendencies_update
        from ..stencils.precipitation_fraction_liquid_content import \
            ice4_precipitation_fraction_liquid_content
        from ..stencils.rain_ice import (fog_deposition,
                                         ice4_precipitation_fraction_sigma,
                                         ice4_rainfr_vert,
                                         initial_values_saving, rain_ice_mask,
                                         rain_ice_thermo,
                                         rain_ice_total_tendencies,
                                         rain_fraction_sedimentation)
        from ..stencils.sedimentation import (sedimentation_stat,
                                              upwind_sedimentation)

        # List of stencils to compile
        self.external_tendencies_update = compile_stencil(
            definition=ice4_total_tendencies_update,
            name="ice4_total_tendencies_update"
        )


        # 1. Generalites
        self.rain_ice_thermo = compile_stencil(
            definition=rain_ice_thermo,
            name="rain_ice_thermo"
        )

        self.rain_ice_mask = compile_stencil(
            definition=rain_ice_mask,
            name="rain_ice_mask"
        )

        # 3. Initial values saving
        self.initial_values_saving = compile_stencil(
            definition=initial_values_saving,
            name="initial_values_saving"
        )


        # 4.2 Computes precipitation fraction
        self.ice4_precipitation_fraction_sigma = compile_stencil(
            definition=ice4_precipitation_fraction_sigma,
            name="ice4_precipitation_fraction_sigma",
        )
        self.ice4_precipitation_fraction_liquid_content = compile_stencil(
            definition=ice4_precipitation_fraction_liquid_content,
            name="ice4_precipitation_fraction_liquid_content",
        )
        self.ice4_compute_pdf = compile_stencil(
            definition=ice4_compute_pdf,
            name="ice4_compute_pdf"
        )
        self.ice4_rainfr_vert = compile_stencil(
            name="ice4_rainfr_vert",
            definition=ice4_rainfr_vert
            )

        # 8. Total tendencies
        # 8.1 Total tendencies limited by available species
        self.total_tendencies = compile_stencil(
            definition=rain_ice_total_tendencies,
            name="rain_ice_total_tendencies",
        )
        # 8.2 Negative corrections
        self.ice4_correct_negativities = compile_stencil(
            definition=ice4_correct_negativities,
            name="ice4_correct_negativities",
        )

        # 9. Compute the sedimentation source
        match phyex.param_icen.SEDIM:
            case Sedim.STAT.value:
                self.sedimentation = compile_stencil(
                    definition=sedimentation_stat,
                    name="statistical_sedimentation"
                )
            case Sedim.SPLI.value:
                self.sedimentation = compile_stencil(
                    definition=upwind_sedimentation,
                    name="upwind_sedimentation",
                )

        self.rain_fraction_sedimentation = compile_stencil(
            name="rain_fraction_sedimentation",
            definition=rain_fraction_sedimentation
        )

        # 10 Compute the fog deposition
        self.fog_deposition = compile_stencil(
            name="fog_deposition",
            definition=fog_deposition
        )
        
        #####################################################################
        ######################### Stepping ##################################
        #####################################################################
        # Stencil collections
        from ..stencils.ice4_stepping import (external_tendencies_update,
                                              ice4_mixing_ratio_step_limiter,
                                              ice4_step_limiter,
                                              ice4_stepping_heat,
                                              ice4_stepping_init_tsoft,
                                              ice4_stepping_tmicro_init,
                                              ice4_stepping_ldcompute_init,
                                              state_update)
        self.ice4_stepping_heat = compile_stencil(
            name="ice4_stepping_heat",
            definition=ice4_stepping_heat
        )
        self.ice4_step_limiter = compile_stencil(
            name="ice4_step_limiter",
            definition=ice4_step_limiter
        )
        self.ice4_mixing_ratio_step_limiter = compile_stencil(
            name="ice4_mixing_ratio_step_limiter",
            definition=ice4_mixing_ratio_step_limiter
        )
        self.ice4_state_update = compile_stencil(
            name="state_update",
            definition=state_update
          )
        self.external_tendencies_update = compile_stencil(
            name="external_tendencies_update",
            definition=external_tendencies_update
        )
        self.tmicro_init = compile_stencil(
            name="ice4_stepping_tmicro_init",
            definition=ice4_stepping_tmicro_init
        )
        self.tsoft_init = compile_stencil(
            name="ice4_stepping_tsoft_init",
            definition=ice4_stepping_init_tsoft
        )
        self.ldcompute_init = compile_stencil(
            name="ice4_stepping_ldcompute_init",
            definition=ice4_stepping_ldcompute_init
        )

        # Component for tendency update
        self.ice4_tendencies = Ice4Tendencies(
            phyex=phyex,
            backend=backend,
            dtypes=dtypes
        )
        ###################################################################


    # from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    # from_line=214,
    # to_line=438,
    def __call__(
        self,
        state: Dict[str, NDArray],
        timestep: timedelta,
        domain: Tuple[int],
        validate_args: bool = True,
        exec_info: Dict = True
    ):

        with managed_temporaries(
            *repeat((domain, "bool"), 2),
            *repeat((domain, "float"), 50),
            *repeat((domain[0:2], "float"), 2),
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
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,)
            
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
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info
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
                    domain=domain,
                    validate_args=validate_args,
                    exec_info=exec_info,)

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
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
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
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
            )

            outerloop_counter = 0
            max_outerloop_iterations = 10
            lsoft = False

            # l223 in f90
            # _np_t_micro = to_numpy(t_micro)
            dt = timestep.total_seconds()

            log.info("First loop")
            while (t_micro < dt).any():

                log.info(f"type, t_micro {type(t_micro)}, {type(t_micro[0, 0, 0])}")
                log.info(f"ldcompute, ldcompute {type(ldcompute)}, {type(ldcompute[0, 0, 0])}")
                log.info(f"type, th_t {type(state['th_t'])}")

                # Translation note XTSTEP_TS == 0 is assumed implying no loops over t_soft
                innerloop_counter = 0
                max_innerloop_iterations = 10

                log.info(f"ldcompute {ldcompute}")

                # Translation note : l230 to l237 in Fortran
                self.ldcompute_init(
                    ldcompute=ldcompute,
                    t_micro=t_micro,
                    origin=(0, 0, 0),
                    domain=domain,
                    validate_args=validate_args,
                    exec_info=exec_info,
                )

                log.info(f"ldcompute {ldcompute}")

                # Iterations limiter
                if outerloop_counter >= max_outerloop_iterations:
                    break

                log.info("Second loop")
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
                        domain=domain,
                        validate_args=validate_args,
                        exec_info=exec_info,
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
                            "th_t": state["th_t"],
                            "rv_t": state["rv_t"],
                            "rc_t": state["rc_t"],
                            "rr_t": state["rr_t"],
                            "ri_t": state["ri_t"],
                            "rs_t": state["rs_t"],
                            "rg_t": state["rg_t"],
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

                    # log.info("Call tendencies")
                    # _, _ = self.ice4_tendencies(
                    #     ldsoft=lsoft,
                    #     state=state_ice4_tendencies,
                    #     timestep=timestep,
                    #     out_tendencies={},
                    #     out_diagnostics={},
                    #     overwrite_tendencies={},
                    #     domain=domain,
                    #     exec_info=exec_info,
                    #     validate_args=validate_args,
                    # )

                    log.info(f"ldcompute {ldcompute}")

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

                    log.info("Call step limiter")
                    self.ice4_step_limiter(
                        **state_step_limiter, 
                        origin=(0, 0, 0),
                        domain=domain,
                        validate_args=validate_args,
                        exec_info=exec_info,
                    )

                    # l346 to l388
                    ############ ice4_mixing_ratio_step_limiter ############
                    log.info(f"ldcompute : {ldcompute}")
                    state_mixing_ratio_step_limiter = {
                        **{
                            key: state[key] for key in [
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

                    log.info("Call mixing ratio step limiter")
                    self.ice4_mixing_ratio_step_limiter(
                        **state_mixing_ratio_step_limiter,
                        origin=(0, 0, 0),
                        domain=domain,
                        validate_args=validate_args,
                        exec_info=exec_info,
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
                        domain=domain,
                        validate_args=validate_args,
                        exec_info=exec_info,
                    )

                    # TODO : next loop
                    lsoft = True
                    innerloop_counter += 1
                    log.info("Loop 2 end")

                outerloop_counter += 1

            log.info("Loop 1 end")

            # l440 to l452
            ################ external_tendencies_update ############
            # if ldext_tnd

            state_external_tendencies_update = {
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
                },
                **{
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
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
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
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
            )

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
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
            )

            # 9. Compute the sedimentation source
            if LSEDIM_AFTER:
                # sedimentation switch is handled in initialisation
                # self.sedimentation is can be either statistical_sedimentation or upwind_sedimentation
                self.sedimentation(
                    **state_sed,
                    **tmps_sedim,
                    origin=(0, 0, 0),
                    domain=domain,
                    validate_args=validate_args,
                    exec_info=exec_info,
                )

                state_frac_sed = {
                    **{key: state[key] for key in ["rrs", "rss", "rgs"]},
                    **{"wr_r": wr_r, "wr_s": wr_s, "wr_g": wr_g},
                }
                self.rain_fraction_sedimentation(
                    **state_frac_sed,
                    origin=(0, 0, 0),
                    domain=domain,
                    validate_args=validate_args,
                    exec_info=exec_info,
                )

                state_rainfr = {**{key: state[key] for key in ["prfr", "rr_t", "rs_t"]}}
                self.ice4_rainfr_vert(
                    **state_rainfr,
                    origin=(0, 0, 0),
                    domain=domain,
                    validate_args=validate_args,
                    exec_info=exec_info,
                )

            # 10 Compute the fog deposition
            if LDEPOSC:
                state_fog = {
                    key: state[key]
                    for key in ["rcs", "rc_t", "rhodref", "dzz", "inprc"]
                }
                self.fog_deposition(
                    **state_fog,
                    origin=(0, 0, 0),
                    domain=domain,
                    validate_args=validate_args,
                    exec_info=exec_info
                )
