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


from ice3_gt4py.initialisation.state_ice4_tendencies import (
    get_constant_state_ice4_tendencies,
)
from ice3_gt4py.phyex_common.phyex import Phyex


class Ice4Tendencies(ImplicitTendencyComponent):
    """Implicit Tendency Component calling
    ice_adjust : saturation adjustment of temperature and mixing ratios

    ice_adjust stencil is ice_adjust.F90 in PHYEX
    """

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

        # Tendencies
        self.ice4_nucleation = compile_stencil("ice4_nucleation", externals)
        self.ice4_nucleation_post_processing = compile_stencil(
            "ice4_nucleation_post_processing", externals
        )

        self.ice4_rrhong = compile_stencil("ice4_rrhong", externals)
        self.ice4_rrhong_post_processing = compile_stencil(
            "ice4_rrhong_post_processing", externals
        )

        self.ice4_rimltc = compile_stencil("ice4_rimltc", externals)
        self.ice4_rimltc_post_processing = compile_stencil("rimltc_post_processing")

        self.ice4_increment_update = compile_stencil("ice4_increment_update", externals)
        self.ice4_derived_fields = compile_stencil("ice4_derived_fields", externals)

        # TODO: add ice4_compute_pdf
        self.ice4_slope_parameters = compile_stencil("ice4_slope_parameters", externals)

        self.ice4_slow = compile_stencil("ice4_slow", externals)
        self.ice4_warm = compile_stencil("ice4_warm", externals)

        self.ice4_fast_rs = compile_stencil("ice4_fast_rs", externals)

        self.ice4_fast_rg_pre_processing = compile_stencil(
            "ice4_fast_rg_pre_processing", externals
        )
        self.ice4_fast_rg = compile_stencil("ice4_fast_rg", externals)

        self.ice4_fast_ri = compile_stencil("ice4_fast_ri", externals)

        self.ice4_tendencies_update = compile_stencil(
            "ice4_tendencies_update", externals
        )

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "ldcompute": {"grid": (I, J, K), "units": ""},
            "pres": {"grid": (I, J, K), "units": ""},
            "rhodref": {"grid": (I, J, K), "units": ""},
            "exn": {"grid": (I, J, K), "units": ""},
            "ls_fact": {"grid": (I, J, K), "units": ""},
            "lv_fact": {"grid": (I, J, K), "units": ""},
            "rv_t": {"grid": (I, J, K), "units": ""},
            "cf": {"grid": (I, J, K), "units": ""},
            "sigma_rc": {"grid": (I, J, K), "units": ""},
            "ci_t": {"grid": (I, J, K), "units": ""},
            "ai": {"grid": (I, J, K), "units": ""},
            "cj": {"grid": (I, J, K), "units": ""},
            "ssi": {"grid": (I, J, K), "units": ""},
            "t": {"grid": (I, J, K), "units": ""},
            "tht": {"grid": (I, J, K), "units": ""},
            # PVART in f90
            "rv_t": {"grid": (I, J, K), "units": ""},
            "rc_t": {"grid": (I, J, K), "units": ""},
            "rr_t": {"grid": (I, J, K), "units": ""},
            "ri_t": {"grid": (I, J, K), "units": ""},
            "rs_t": {"grid": (I, J, K), "units": ""},
            "rg_t": {"grid": (I, J, K), "units": ""},
            # A in f90
            "theta_tnd": {"grid": (I, J, K), "units": ""},
            "rv_tnd": {"grid": (I, J, K), "units": ""},
            "rc_tnd": {"grid": (I, J, K), "units": ""},
            "rr_tnd": {"grid": (I, J, K), "units": ""},
            "ri_tnd": {"grid": (I, J, K), "units": ""},
            "rs_tnd": {"grid": (I, J, K), "units": ""},
            "rg_tnd": {"grid": (I, J, K), "units": ""},
            # B in f90
            "theta_increment": {"grid": (I, J, K), "units": ""},
            "rv_increment": {"grid": (I, J, K), "units": ""},
            "rc_increment": {"grid": (I, J, K), "units": ""},
            "rr_increment": {"grid": (I, J, K), "units": ""},
            "ri_increment": {"grid": (I, J, K), "units": ""},
            "rs_increment": {"grid": (I, J, K), "units": ""},
            "rg_increment": {"grid": (I, J, K), "units": ""},
            # Slopes
            "lbdar": {"grid": (I, J, K), "units": ""},
            "lbdar_rf": {"grid": (I, J, K), "units": ""},
            "lbdas": {"grid": (I, J, K), "units": ""},
            "lbdag": {"grid": (I, J, K), "units": ""},
            # mr - diagnostics
            "rvheni_mr": {"grid": (I, J, K), "units": ""},
            "rrhong_mr": {"grid": (I, J, K), "units": ""},
            "rimltc_mr": {"grid": (I, J, K), "units": ""},
            "rgsi_mr": {"grid": (I, J, K), "units": ""},
            "rsrimcg_mr": {"grid": (I, J, K), "units": ""},
            # tnd
            "rc_honi_tnd": {"grid": (I, J, K), "units": ""},
            "rv_deps_tnd": {"grid": (I, J, K), "units": ""},
            "ri_aggs_tnd": {"grid": (I, J, K), "units": ""},
            "ri_auts_tnd": {"grid": (I, J, K), "units": ""},
            "rv_depg_tnd": {"grid": (I, J, K), "units": ""},
            "rs_mltg_tnd": {"grid": (I, J, K), "units": ""},
            "rc_mltsr_tnd": {"grid": (I, J, K), "units": ""},
            "rs_rcrims_tnd": {"grid": (I, J, K), "units": ""},
            "rs_rcrimss_tnd": {"grid": (I, J, K), "units": ""},
            "rs_rsrimcg_tnd": {"grid": (I, J, K), "units": ""},
            "rs_rraccs_tnd": {"grid": (I, J, K), "units": ""},
            "rs_rraccss_tnd": {"grid": (I, J, K), "units": ""},
            "rs_rsaccrg_tnd": {"grid": (I, J, K), "units": ""},
            "rs_freez1_tnd": {"grid": (I, J, K), "units": ""},
            "rs_freez2_tnd": {"grid": (I, J, K), "units": ""},
            "rg_rcdry_tnd": {"grid": (I, J, K), "units": ""},
            "rg_ridry_tnd": {"grid": (I, J, K), "units": ""},
            "rg_rsdry_tnd": {"grid": (I, J, K), "units": ""},
            "rg_rrdry_tnd": {"grid": (I, J, K), "units": ""},
            "rg_riwet_tnd": {"grid": (I, J, K), "units": ""},
            "rg_rswet_tnd": {"grid": (I, J, K), "units": ""},
            "rg_freez1_tnd": {"grid": (I, J, K), "units": ""},
            "rg_freez2_tnd": {"grid": (I, J, K), "units": ""},
            "rc_beri_tnd": {"grid": (I, J, K), "units": ""},
            # transformations
            "rgsi": {"grid": (I, J, K), "units": ""},
            "rchoni": {"grid": (I, J, K), "units": ""},
            "rvdeps": {"grid": (I, J, K), "units": ""},
            "riaggs": {"grid": (I, J, K), "units": ""},
            "riauts": {"grid": (I, J, K), "units": ""},
            "rvdepg": {"grid": (I, J, K), "units": ""},
            "rcautr": {"grid": (I, J, K), "units": ""},
            "rcaccr": {"grid": (I, J, K), "units": ""},
            "rrevav": {"grid": (I, J, K), "units": ""},
            "rcberi": {"grid": (I, J, K), "units": ""},
            "rsmltg": {"grid": (I, J, K), "units": ""},
            "rcmltsr": {"grid": (I, J, K), "units": ""},
            "rraccss": {"grid": (I, J, K), "units": ""},  # 13
            "rraccsg": {"grid": (I, J, K), "units": ""},  # 14
            "rsaccrg": {"grid": (I, J, K), "units": ""},  # 15
            "rcrimss": {"grid": (I, J, K), "units": ""},  # 16
            "rcrimsg": {"grid": (I, J, K), "units": ""},  # 17
            "rsrimcg": {"grid": (I, J, K), "units": ""},  # 18
            "ricfrrg": {"grid": (I, J, K), "units": ""},  # 19
            "rrcfrig": {"grid": (I, J, K), "units": ""},  # 20
            "ricfrr": {"grid": (I, J, K), "units": ""},  # 21
            "rcwetg": {"grid": (I, J, K), "units": ""},  # 22
            "riwetg": {"grid": (I, J, K), "units": ""},  # 23
            "rrwetg": {"grid": (I, J, K), "units": ""},  # 24
            "rswetg": {"grid": (I, J, K), "units": ""},  # 25
            "rcdryg": {"grid": (I, J, K), "units": ""},  # 26
            "ridryg": {"grid": (I, J, K), "units": ""},  # 27
            "rrdryg": {"grid": (I, J, K), "units": ""},  # 28
            "rsdryg": {"grid": (I, J, K), "units": ""},  # 29
            "rgmltr": {"grid": (I, J, K), "units": ""},  # 31
            # tnd_update
            "theta_tnd": {"grid": (I, J, K), "units": ""},
            "rv_tnd": {"grid": (I, J, K), "units": ""},
            "rc_tnd": {"grid": (I, J, K), "units": ""},
            "rr_tnd": {"grid": (I, J, K), "units": ""},
            "ri_tnd": {"grid": (I, J, K), "units": ""},
            "rs_tnd": {"grid": (I, J, K), "units": ""},
            "rg_tnd": {"grid": (I, J, K), "units": ""},
            # others
            "hlc_hcf": {"grid": (I, J, K), "units": ""},
            "hlc_lcf": {"grid": (I, J, K), "units": ""},
            "hlc_hrc": {"grid": (I, J, K), "units": ""},
            "hlc_lrc": {"grid": (I, J, K), "units": ""},
            "hli_hcf": {"grid": (I, J, K), "units": ""},
            "hli_lcf": {"grid": (I, J, K), "units": ""},
            "hli_hri": {"grid": (I, J, K), "units": ""},
            "hli_lri": {"grid": (I, J, K), "units": ""},
            "fr": {"grid": (I, J, K), "units": ""},
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

    def array_call(
        self,
        ldsoft: bool,
        state: NDArrayLikeDict,
        timestep: timedelta,
        out_tendencies: NDArrayLikeDict,
        out_diagnostics: NDArrayLikeDict,
        overwrite_tendencies: Dict[str, bool],
    ) -> None:
        with managed_temporary_storage(
            self.computational_grid,
            *repeat(((I, J, K), "float"), 10),
            gt4py_config=self.gt4py_config,
        ) as (
            usw,
            w1,
            w2,
            ssi,
            grim_tmp,
            gacc_tmp,
            zw_tmp,
            zw1_tmp,
            zw2_tmp,
            zw3_tmp,
            index_floor,
            index_float,
            index_floor_s,
            index_float_s,
            index_floor_r,
            index_float_r,
            gdry,
            ldwetg,  # bool, true if graupel grows in wet mode (out)
            lldryg,  # linked to gdry + temporary
            rdryg_init_tmp,
            rwetg_init_tmp,
            index_floor_g,
            index_float_g,
        ):

            ############## ice4_nucleation ################
            self.state_nucleation = {
                "ldcompute": state["ldcompute"],
                **{
                    key: state[key]
                    for key in [
                        "tht",
                        "pabs",
                        "rhodref",
                        "exn",
                        "ls_fact",
                        "t",
                        "rv_t",
                        "ci_t",
                    ]
                },
                "rvheni_mr": state["rvheni_mr"],
            }

            temporaries_nucleation = {
                "usw": {"grid": (I, J, K), "units": ""},
                "w1": {"grid": (I, J, K), "units": ""},
                "w2": {"grid": (I, J, K), "units": ""},
                "ssi": {"grid": (I, J, K), "units": ""},
            }

            # timestep
            self.ice4_nucleation(**state_nucleation, **temporaries_nucleation)

            ############## ice4_nucleation_post_processing ####################

            state_nucleation_pp = {
                **{
                    key: state[key]
                    for key in ["t", "exn", "ls_fact", "lv_fact", "tht", "rv_t", "ri_t"]
                },
                "rvheni_mr": state["rvheni_mr"],
            }

            # Timestep
            self.ice4_nucleation_post_processing(**state_nucleation_pp)

            ########################### ice4_rrhong #################################
            state_rrhong = {
                "ldcompute": state["ldcompute"],
                **{
                    key: state[key]
                    for key in ["t", "exn", "lv_fact", "ls_fact", "tht", "rr_t"]
                },
                "rrhong_mr": state["rrhong_mr"],
            }

            self.ice4_rrhong(**state_rrhong)

            ########################### ice4_rrhong_post_processing #################
            state_rrhong_pp = {
                **{
                    key: state[key]
                    for key in [
                        "t",
                        "exn",
                        "lv_fact",
                        "ls_fact",
                        "tht",
                        "rg_t",
                        "rr_t",
                    ]
                },
                "rrhong_mr": state["rrhong_mr"],
            }

            self.ice4_rrhong_post_processing(**state_rrhong_pp)

            ########################## ice4_rimltc ##################################
            state_rimltc = {
                "ldcompute": state["ldcompute"],
                **{
                    key: state[key]
                    for key in [
                        "t",
                        "exn",
                        "lv_fact",
                        "ls_fact",
                        "tht",
                        "ri_t",
                    ]
                },
                "rimltc_mr": state["rimltc_mr"],
            }

            self.ice4_rimltc(**state_rimltc)

            ####################### ice4_rimltc_post_processing #####################

            state_rimltc_pp = {
                **{
                    key: state[key]
                    for key in ["t", "exn", "lv_fact", "ls_fact", "tht", "rc_t", "ri_t"]
                },
                "rimltc_mr": state["rimltc_mr"],
            }

            self.ice4_rimltc_post_processing(**state_rimltc_pp)

            ######################## ice4_increment_update ##########################
            state_increment_update = {
                **{key: state[key] for key in ["ls_fact", "lv_fact"]},
                **{
                    key: state[key]
                    for key in [
                        "theta_increment",
                        "rv_increment",
                        "rc_increment",
                        "rr_increment",
                        "ri_increment",
                        "rs_increment",
                        "rg_increment",
                    ]
                },  # PB in F90
                **{
                    key: state[key]
                    for key in ["rvheni_mr", "rimltc_mr", "rrhong_mr", "rsrimcg_mr"]
                },
            }

            self.ice4_increment_update(**state_increment_update)

            ######################## ice4_derived_fields ############################
            state_derived_fields = {
                key: state[key]
                for key in [
                    "t",
                    "rhodref",
                    "rv_t",
                    "pres",
                    "ssi",
                    "ka",
                    "dv",
                    "ai",
                    "cj",
                ]
            }

            temporaries_derived_fields = {
                "zw": zw_tmp,
            }

            self.ice4_derived_fields(
                **state_derived_fields, **temporaries_derived_fields
            )

            ######################## ice4_slope_parameters ##########################
            state_slope_parameters = {
                **{key: state[key] for key in ["rhodref", "t", "rr_t", "rs_t", "rg_t"]},
                **{key: state[key] for key in ["lbdar", "lbdar_rf", "lbdas", "lbdag"]},
            }

            self.ice4_slope_parameters(**state_slope_parameters)

            ######################## ice4_slow ######################################
            state_slow = {
                "ldcompute": state["ldcompute"],
                **{
                    key: state[key]
                    for key in [
                        "rhodref",
                        "t",
                        "ssi",
                        "lv_fact",
                        "ls_fact",
                        "rv_t",
                        "rc_t",
                        "ri_t",
                        "rs_t",
                        "rg_t",
                        "ai",
                        "cj",
                        "hli_hcf",
                        "hli_hri",
                    ]
                },
                **{key: state[key] for key in ["lbdas", "lbdag"]},
                **{
                    key: tnd[key]
                    for key in [
                        "rc_honi_tnd",
                        "rv_deps_tnd",
                        "ri_aggs_tnd",
                        "ri_auts_tnd",
                        "rv_depg_tnd",
                    ]
                },
            }

            self.ice4_slow(ldsoft=ldsoft, **state_slow)

            ######################## ice4_warm ######################################
            state_warm = {
                "ldcompute": state["ldcompute"],
                **{
                    key: state[key]
                    for key in [
                        "rhodref",
                        "lv_fact",
                        "t",  # temperature
                        "tht",
                        "pres",
                        "ka",  # thermal conductivity of the air
                        "dv",  # diffusivity of water vapour
                        "cj",  # function to compute the ventilation coefficient
                        "hlc_hcf",  # High Cloud Fraction in grid
                        "hlc_lcf",  # Low Cloud Fraction in grid
                        "hlc_hrc",  # LWC that is high in grid
                        "hlc_lrc",  # LWC that is low in grid
                        "rv_t",  # water vapour mixing ratio at t
                        "rc_t",  # cloud water mixing ratio at t
                        "rr_t",  # rain water mixing ratio at t
                        "cf",
                        "rf",
                    ]
                },
                **{key: state[key] for key in ["lbdar", "lbdar_rf"]},
                **{key: state[key] for key in ["rcautr", "rcaccr", "rrevav"]},
            }

            self.ice4_warm(ldsoft=ldsoft, **state_warm)

            ######################## ice4_fast_rs ###################################
            state_fast_rs = {
                "ldcompute": state["ldcompute"],
                **{
                    key: state[key]
                    for key in [
                        "rhodref",
                        "lv_fact",
                        "ls_fact",
                        "pres",  # absolute pressure at t
                        "dv",  # diffusivity of water vapor in the air
                        "ka",  # thermal conductivity of the air
                        "cj",  # function to compute the ventilation coefficient
                        "t",
                        "rv_t",
                        "rc_t",
                        "rr_t",
                        "rs_t",
                    ]
                },
                **{key: state[key] for key in ["lbdar", "lbdas"]},
                **{
                    key: state[key]
                    for key in [
                        "riaggs",
                        "rcrimss",
                        "rcrimsg",
                        "rsrimcg",
                        "rraccss",
                        "rraccsg",
                        "rsaccrg",
                    ]
                },
                **{
                    key: state[key]
                    for key in [
                        "rs_mltg_tnd",
                        "rc_mltsr_tnd",  # cloud droplet collection onto aggregates
                        "rs_rcrims_tnd",  # extra dimension 8 in Fortran PRS_TEND
                        "rs_rcrimss_tnd",
                        "rs_rsrimcg_tnd",
                        "rs_rraccs_tnd",
                        "rs_rraccss_tnd",
                        "rs_rsaccrg_tnd",
                        "rs_freez1_tnd",
                        "rs_freez2_tnd",
                    ]
                },
            }

            temporaries_fast_rs = {
                "grim_tmp": grim_tmp,
                "gacc_tmp": gacc_tmp,
                "zw_tmp": zw_tmp,
                "zw1_tmp": zw1_tmp,
                "zw2_tmp": zw2_tmp,
                "zw3_tmp": zw3_tmp,
                "index_floor": index_floor,
                "index_float": index_float,
                "index_floor_s": index_floor_s,
                "index_float_s": index_float_s,
                "index_floor_r": index_floor_r,
                "index_float_r": index_float_r,
            }

            # TODO: replace by real values
            gaminc_rim1 = from_array(np.ones((80)), backend=BACKEND)
            gaminc_rim2 = from_array(np.ones((80)), backend=BACKEND)
            gaminc_rim4 = from_array(np.ones((80)), backend=BACKEND)

            ker_raccs = from_array(np.ones((40, 40)), backend=BACKEND)
            ker_raccss = from_array(np.ones((40, 40)), backend=BACKEND)
            ker_saccrg = from_array(np.ones((40, 40)), backend=BACKEND)

            self.ice4_fast_rs(
                ldsoft=ldsoft,
                gaminc_rim1=gaminc_rim1,
                gaminc_rim2=gaminc_rim2,
                gaminc_rim4=gaminc_rim4,
                ker_raccs=ker_raccs,
                ker_raccss=ker_raccss,
                ker_saccrg=ker_saccrg,
                **state_fast_rs,
                **temporaries_fast_rs,
            )

            ######################## ice4_fast_rg_pre_processing ####################
            state_fast_rg_pp = {
                **{
                    key: transformations[key]
                    for key in [
                        "rgsi",
                        "rvdepg",
                        "rsmltg",
                        "rraccsg",
                        "rsaccrg",
                        "rcrimsg",
                        "rsrimcg",
                    ]
                },
                **{key: state[key] for key in ["rgsi_mr", "rrhong_mr", "rsrimcg_mr"]},
            }

            self.ice4_fast_rg_pre_processing(**state_fast_rg_pp)

            ######################## ice4_fast_rg ###################################
            state_fast_rg = {
                "ldcompute": state["ldcompute"],
                **{
                    key: state[key]
                    for key in [
                        "t",
                        "rhodref",
                        "pres",
                        "rv_t",
                        "rr_t",
                        "ri_t",
                        "rg_t",
                        "rc_t",
                        "rs_t",
                        "ci_t",
                        "ka",
                        "dv",
                        "cj",
                    ]
                },
                **{key: state[key] for key in ["lbdar", "lbdas", "lbdag"]},
                **{
                    key: state[key]
                    for key in [
                        "rg_rcdry_tnd",
                        "rg_ridry_tnd",
                        "rg_rsdry_tnd",
                        "rg_rrdry_tnd",
                        "rg_riwet_tnd",
                        "rg_rswet_tnd",
                        "rg_freez1_tnd",
                        "rg_freez2_tnd",
                    ]
                },
                **{
                    key: state[key]
                    for key in ["ricfrrg", "rrcfrig", "ricfrr", "rgmltr"]
                },
            }

            temporaries_fast_rg = {
                "gdry": gdry,
                "ldwetg": ldwetg,  # bool, true if graupel grows in wet mode (out)
                "lldryg": lldryg,  # linked to gdry + temporary
                "rdryg_init_tmp": rdryg_init_tmp,
                "rwetg_init_tmp": rwetg_init_tmp,
                "zw_tmp": zw_tmp,  # ZZW in Fortran
                "index_floor_s": index_floor_s,
                "index_floor_g": index_floor_g,
                "index_floor_r": index_floor_r,
                "index_float_s": index_float_s,
                "index_float_g": index_float_g,
                "index_float_r": index_float_r,
            }

            ker_sdryg = from_array(np.ones((40, 40)), backend=BACKEND)
            ker_rdryg = from_array(np.ones((40, 40)), backend=BACKEND)

            self.ice4_fast_rg(
                ldsoft=ldsoft,
                ker_sdryg=ker_sdryg,
                ker_rdryg=ker_rdryg,
                **state_fast_rg,
                **temporaries_fast_rg,
            )

            ######################## ice4_fast_ri ###################################
            state_fast_ri = {
                "ldcompute": state["ldcompute"],
                **{
                    key: state[key]
                    for key in [
                        "rhodref",
                        "lv_fact",
                        "ls_fact",
                        "ai",
                        "cj",
                        "ci_t",
                        "ssi",
                        "rc_t",
                        "ri_t",
                    ]
                },
                "rc_beri_tnd": state["rc_beri_tnd"],
            }

            self.ice4_fast_ri(ldsoft=ldsoft, **state_fast_ri)

            ######################## ice4_tendencies_update #########################

            state_tendencies_update = {
                **{key: state[key] for key in ["ls_fact", "lv_fact"]},
                **{
                    key: state[key]
                    for key in [
                        "theta_tnd",
                        "rv_tnd",
                        "rc_tnd",
                        "rr_tnd",
                        "ri_tnd",
                        "rs_tnd",
                        "rg_tnd",
                    ]
                },
                **{
                    key: state[key]
                    for key in [
                        "rchoni",
                        "rvdeps",
                        "riaggs",
                        "riauts",
                        "rvdepg",
                        "rcautr",
                        "rcaccr",
                        "rrevav",
                        "rcberi",
                        "rsmltg",
                        "rcmltsr",
                        "rraccss",  # 13
                        "rraccsg",  # 14
                        "rsaccrg",  # 15  # Rain accretion onto the aggregates
                        "rcrimss",  # 16
                        "rcrimsg",  # 17
                        "rsrimcg",  # 18  # Cloud droplet riming of the aggregates
                        "ricfrrg",  # 19
                        "rrcfrig",  # 20
                        "ricfrr",  # 21  # Rain contact freezing
                        "rcwetg",  # 22
                        "riwetg",  # 23
                        "rrwetg",  # 24
                        "rswetg",  # 25  # Graupel wet growth
                        "rcdryg",  # 26
                        "ridryg",  # 27
                        "rrdryg",  # 28
                        "rsdryg",  # 29  # Graupel dry growth
                        "rgmltr",
                    ]  # 31  # Melting of the graupel
                },
                **{
                    key: state[key]
                    for key in ["rvheni_mr", "rrhong_mr", "rimltc_mr", "rsrimcg_mr"]
                },
            }

            self.ice4_tendencies_update(**state_tendencies_update)
