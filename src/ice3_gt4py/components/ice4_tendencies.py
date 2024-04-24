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
        self.nucleation_post_processing = compile_stencil(
            "nucleation_post_processing", externals
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
        return {}

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
        state: NDArrayLikeDict,
        timestep: timedelta,
        out_tendencies: NDArrayLikeDict,
        out_diagnostics: NDArrayLikeDict,
        overwrite_tendencies: Dict[str, bool],
    ) -> None:
        with managed_temporary_storage(
            self.computational_grid,
            *repeat(((I, J, K), "bool"), 1),
            *repeat(((I, J, K), "float"), 20),
            gt4py_config=self.gt4py_config,
        ) as (ldcompute, rv_heni_mr, usw, w1, w2, ssi):
            inputs = {
                name.split("_", maxsplit=1)[1]: state[name]
                for name in self.input_properties
            }

            temporaries = {
                "ldcompute": ldcompute,
                "rv_heni_mr": rv_heni_mr,
                "usw": usw,
                "w1": w1,
                "w2": w2,
                "ssi": ssi,
            }

            self.ice4_nucleation(**inputs, **temporaries)
            # self.nucleation_post_processing()

            # self.ice4_rrhong()
            # self.ice4_rrhong_post_processing()

            # self.ice4_rimltc()
            # self.ice4_rimltc_post_processing()

            # self.ice4_increment_update()
            # self.ice4_derived_fields()

            # TODO: add ice4_compute_pdf
            # self.ice4_slope_parameters()

            # self.ice4_slow()
            # self.ice4_warm()

            # self.ice4_fast_rs()

            # self.ice4_fast_rg_pre_processing()
            # self.ice4_fast_rg()

            # self.ice4_fast_ri()

            # self.ice4_tendencies_update()


if __name__ == "__main__":

    BACKEND = "gt:cpu_kfirst"
    from gt4py.storage import ones, from_array
    import numpy as np

    ################### Grid #################
    logging.info("Initializing grid ...")
    nx = 100
    ny = 1
    nz = 90
    grid = ComputationalGrid(nx, ny, nz)
    dt = timedelta(seconds=1)

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

    ###############################################
    ############# Compilation #####################
    ###############################################
    logging.info(f"Compilation for ice4_nucleation")
    ice4_nucleation = compile_stencil("ice4_nucleation", gt4py_config, externals)

    logging.info(f"Compilation for ice4_nucleation_post_processing")
    ice4_nucleation_post_processing = compile_stencil(
        "ice4_nucleation_post_processing", gt4py_config, externals
    )

    logging.info(f"Compilation for ice4_rrhong")
    ice4_rrhong = compile_stencil("ice4_rrhong", gt4py_config, externals)

    logging.info(f"Compilation for ice4_rrhong_post_processing")
    ice4_rrhong_post_processing = compile_stencil(
        "ice4_rrhong_post_processing", gt4py_config, externals
    )

    logging.info(f"Compilation for ice4_rimltc")
    ice4_rimltc = compile_stencil("ice4_rimltc", gt4py_config, externals)

    logging.info(f"Compilation for ice4_rimltc_post_processing")
    ice4_rimltc_post_processing = compile_stencil(
        "ice4_rimltc_post_processing", gt4py_config, externals
    )

    logging.info(f"Compilation for ice4_increment_update")
    ice4_increment_update = compile_stencil(
        "ice4_increment_update", gt4py_config, externals
    )

    logging.info(f"Compilation for ice4_derived_fields")
    ice4_derived_fields = compile_stencil(
        "ice4_derived_fields", gt4py_config, externals
    )

    logging.info(f"Compilation for ice4_slope_parameters")
    ice4_slope_parameters = compile_stencil(
        "ice4_slope_parameters", gt4py_config, externals
    )

    logging.info(f"Compilation for ice4_slow")
    ice4_slow = compile_stencil("ice4_slow", gt4py_config, externals)

    logging.info(f"Compilation for ice4_warm")
    ice4_warm = compile_stencil("ice4_warm", gt4py_config, externals)

    logging.info(f"Compilation for ice4_fast_rs")
    ice4_fast_rs = compile_stencil("ice4_fast_rs", gt4py_config, externals)

    logging.info(f"Compilation for ice4_fast_rg_pre_processing")
    ice4_fast_rg_pre_processing = compile_stencil(
        "ice4_fast_rg_pre_processing", gt4py_config, externals
    )

    logging.info(f"Compilation for ice4_fast_rg")
    ice4_fast_rg = compile_stencil("ice4_fast_rg", gt4py_config, externals)

    logging.info(f"Compilation for ice4_fast_ri")
    ice4_fast_ri = compile_stencil("ice4_fast_ri", gt4py_config, externals)

    logging.info(f"Compilation for ice4_tendencies_update")
    ice4_tendencies_update = compile_stencil(
        "ice4_tendencies_update", gt4py_config, externals
    )

    ################ Global state #################

    state = get_constant_state_ice4_tendencies(grid, gt4py_config=gt4py_config)

    time_state = {
        "t_micro": ones((nx, ny, nz), backend=BACKEND),
        "t_soft": ones((nx, ny, nz), backend=BACKEND),
    }

    masks = {"ldcompute": ones((nx, ny, nz), backend=BACKEND, dtype=bool)}

    state = {
        "tht": ones((nx, ny, nz), backend=BACKEND),
        "pabs": ones((nx, ny, nz), backend=BACKEND),
        "rhodref": ones((nx, ny, nz), backend=BACKEND),
        "exn": ones((nx, ny, nz), backend=BACKEND),
        "ls_fact": ones((nx, ny, nz), backend=BACKEND),
        "lv_fact": ones((nx, ny, nz), backend=BACKEND),
        "t": ones((nx, ny, nz), backend=BACKEND),
        "rv_t": ones((nx, ny, nz), backend=BACKEND),
        "rc_t": ones((nx, ny, nz), backend=BACKEND),
        "rr_t": ones((nx, ny, nz), backend=BACKEND),
        "ri_t": ones((nx, ny, nz), backend=BACKEND),
        "rs_t": ones((nx, ny, nz), backend=BACKEND),
        "rg_t": ones((nx, ny, nz), backend=BACKEND),
        "ci_t": ones((nx, ny, nz), backend=BACKEND),
        "pres": ones((nx, ny, nz), backend=BACKEND),
        "ssi": ones((nx, ny, nz), backend=BACKEND),  # supersaturation over ice
        "ka": ones((nx, ny, nz), backend=BACKEND),  #
        "dv": ones((nx, ny, nz), backend=BACKEND),
        "ai": ones((nx, ny, nz), backend=BACKEND),
        "cj": ones((nx, ny, nz), backend=BACKEND),
        "hlc_hcf": ones((nx, ny, nz), backend=BACKEND),  # High Cloud Fraction in grid
        "hlc_lcf": ones((nx, ny, nz), backend=BACKEND),  # Low Cloud Fraction in grid
        "hlc_hrc": ones((nx, ny, nz), backend=BACKEND),  # LWC that is high in grid
        "hlc_lrc": ones((nx, ny, nz), backend=BACKEND),
        "hli_hcf": ones((nx, ny, nz), backend=BACKEND),
        "hli_hri": ones((nx, ny, nz), backend=BACKEND),
    }

    slopes = {
        "lbdar": ones((nx, ny, nz), backend=BACKEND),
        "lbdar_rf": ones((nx, ny, nz), backend=BACKEND),
        "lbdas": ones((nx, ny, nz), backend=BACKEND),
        "lbdag": ones((nx, ny, nz), backend=BACKEND),
    }

    increments = {
        "theta_increment": ones((nx, ny, nz), backend=BACKEND),
        "rv_increment": ones((nx, ny, nz), backend=BACKEND),
        "rc_increment": ones((nx, ny, nz), backend=BACKEND),
        "rr_increment": ones((nx, ny, nz), backend=BACKEND),
        "ri_increment": ones((nx, ny, nz), backend=BACKEND),
        "rs_increment": ones((nx, ny, nz), backend=BACKEND),
        "rg_increment": ones((nx, ny, nz), backend=BACKEND),
    }

    transformations = {
        "rgsi": ones((nx, ny, nz), backend=BACKEND),
        "rchoni": ones((nx, ny, nz), backend=BACKEND),
        "rvdeps": ones((nx, ny, nz), backend=BACKEND),
        "riaggs": ones((nx, ny, nz), backend=BACKEND),
        "riauts": ones((nx, ny, nz), backend=BACKEND),
        "rvdepg": ones((nx, ny, nz), backend=BACKEND),
        "rcautr": ones((nx, ny, nz), backend=BACKEND),
        "rcaccr": ones((nx, ny, nz), backend=BACKEND),
        "rrevav": ones((nx, ny, nz), backend=BACKEND),
        "rcberi": ones((nx, ny, nz), backend=BACKEND),
        "rsmltg": ones((nx, ny, nz), backend=BACKEND),
        "rcmltsr": ones((nx, ny, nz), backend=BACKEND),
        "rraccss": ones((nx, ny, nz), backend=BACKEND),  # 13
        "rraccsg": ones((nx, ny, nz), backend=BACKEND),  # 14
        "rsaccrg": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # 15  # Rain accretion onto the aggregates
        "rcrimss": ones((nx, ny, nz), backend=BACKEND),  # 16
        "rcrimsg": ones((nx, ny, nz), backend=BACKEND),  # 17
        "rsrimcg": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # 18  # Cloud droplet riming of the aggregates
        "ricfrrg": ones((nx, ny, nz), backend=BACKEND),  # 19
        "rrcfrig": ones((nx, ny, nz), backend=BACKEND),  # 20
        "ricfrr": ones((nx, ny, nz), backend=BACKEND),  # 21  # Rain contact freezing
        "rcwetg": ones((nx, ny, nz), backend=BACKEND),  # 22
        "riwetg": ones((nx, ny, nz), backend=BACKEND),  # 23
        "rrwetg": ones((nx, ny, nz), backend=BACKEND),  # 24
        "rswetg": ones((nx, ny, nz), backend=BACKEND),  # 25  # Graupel wet growth
        "rcdryg": ones((nx, ny, nz), backend=BACKEND),  # 26
        "ridryg": ones((nx, ny, nz), backend=BACKEND),  # 27
        "rrdryg": ones((nx, ny, nz), backend=BACKEND),  # 28
        "rsdryg": ones((nx, ny, nz), backend=BACKEND),  # 29  # Graupel dry growth
        "rgmltr": ones((nx, ny, nz), backend=BACKEND),  # 31
    }

    diags = {
        "rvheni_mr": ones((nx, ny, nz), backend=BACKEND),
        "rrhong_mr": ones((nx, ny, nz), backend=BACKEND),
        "rimltc_mr": ones((nx, ny, nz), backend=BACKEND),
        "rgsi_mr": ones((nx, ny, nz), backend=BACKEND),
        "rsrimcg_mr": ones((nx, ny, nz), backend=BACKEND),
    }

    tnd = {
        "rc_honi_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rv_deps_tnd": ones((nx, ny, nz), backend=BACKEND),
        "ri_aggs_tnd": ones((nx, ny, nz), backend=BACKEND),
        "ri_auts_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rv_depg_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rs_mltg_tnd": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # conversion-melting of the aggregates
        "rc_mltsr_tnd": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # cloud droplet collection onto aggregates
        "rs_rcrims_tnd": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # extra dimension 8 in Fortran PRS_TEND
        "rs_rcrimss_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rs_rsrimcg_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rs_rraccs_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rs_rraccss_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rs_rsaccrg_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rs_freez1_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rs_freez2_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_rcdry_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_ridry_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_rsdry_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_rrdry_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_riwet_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_rswet_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_freez1_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_freez2_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rc_beri_tnd": ones((nx, ny, nz), backend=BACKEND),
    }

    # Used in state tendencies update
    tnd_update = {
        "theta_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rv_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rc_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rr_tnd": ones((nx, ny, nz), backend=BACKEND),
        "ri_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rs_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_tnd": ones((nx, ny, nz), backend=BACKEND),
    }

    ############## ice4_nucleation ################
    state_nucleation = {
        "ldcompute": masks["ldcompute"],
        **{
            key: state[key]
            for key in ["tht", "pabs", "rhodref", "exn", "ls_fact", "t", "rv_t", "ci_t"]
        },
        "rvheni_mr": diags["rvheni_mr"],
    }

    temporaries_nucleation = {
        "usw": ones((nx, ny, nz), backend=BACKEND),
        "w1": ones((nx, ny, nz), backend=BACKEND),
        "w2": ones((nx, ny, nz), backend=BACKEND),
        "ssi": ones((nx, ny, nz), backend=BACKEND),
    }

    # timestep
    ice4_nucleation(**state_nucleation, **temporaries_nucleation)

    ############## ice4_nucleation_post_processing ####################

    state_nucleation_pp = {
        **{
            key: state[key]
            for key in ["t", "exn", "ls_fact", "lv_fact", "tht", "rv_t", "ri_t"]
        },
        "rvheni_mr": diags["rvheni_mr"],
    }

    # Timestep
    ice4_nucleation_post_processing(**state_nucleation_pp)

    ########################### ice4_rrhong #################################
    state_rrhong = {
        "ldcompute": masks["ldcompute"],
        **{
            key: state[key] for key in ["t", "exn", "lv_fact", "ls_fact", "tht", "rr_t"]
        },
        "rrhong_mr": diags["rrhong_mr"],
    }

    ice4_rrhong(**state_rrhong)

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
        "rrhong_mr": diags["rrhong_mr"],
    }

    ice4_rrhong_post_processing(**state_rrhong_pp)

    ########################## ice4_rimltc ##################################
    state_rimltc = {
        "ldcompute": masks["ldcompute"],
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
        "rimltc_mr": diags["rimltc_mr"],
    }

    ice4_rimltc(**state_rimltc)

    ####################### ice4_rimltc_post_processing #####################

    state_rimltc_pp = {
        **{
            key: state[key]
            for key in ["t", "exn", "lv_fact", "ls_fact", "tht", "rc_t", "ri_t"]
        },
        "rimltc_mr": diags["rimltc_mr"],
    }

    ice4_rimltc_post_processing(**state_rimltc_pp)

    ######################## ice4_increment_update ##########################
    state_increment_update = {
        **{key: state[key] for key in ["ls_fact", "lv_fact"]},
        **increments,
        **{
            key: diags[key]
            for key in ["rvheni_mr", "rimltc_mr", "rrhong_mr", "rsrimcg_mr"]
        },
    }

    ice4_increment_update(**state_increment_update)

    ######################## ice4_derived_fields ############################
    state_derived_fields = {
        key: state[key]
        for key in ["t", "rhodref", "rv_t", "pres", "ssi", "ka", "dv", "ai", "cj"]
    }

    temporaries_derived_fields = {
        "zw": ones((nx, ny, nz), backend=BACKEND),
    }

    ice4_derived_fields(**state_derived_fields, **temporaries_derived_fields)

    ######################## ice4_slope_parameters ##########################
    state_slope_parameters = {
        **{key: state[key] for key in ["rhodref", "t", "rr_t", "rs_t", "rg_t"]},
        **slopes,
    }

    ice4_slope_parameters(**state_slope_parameters)

    ######################## ice4_slow ######################################
    state_slow = {
        "ldcompute": masks["ldcompute"],
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
        **{key: slopes[key] for key in ["lbdas", "lbdag"]},
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

    ice4_slow(ldsoft=False, **state_slow)
    ice4_slow(ldsoft=True, **state_slow)

    ######################## ice4_warm ######################################
    state_warm = {
        "ldcompute": masks["ldcompute"],  # boolean field for microphysics computation
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
            ]
        },
        **{key: slopes[key] for key in ["lbdar", "lbdar_rf"]},
        **{key: transformations[key] for key in ["rcautr", "rcaccr", "rrevav"]},
        "cf": ones((nx, ny, nz), backend=BACKEND),  # cloud fraction
        "rf": ones((nx, ny, nz), backend=BACKEND),  # rain fraction
    }

    ice4_warm(ldsoft=False, **state_warm)
    ice4_warm(ldsoft=True, **state_warm)

    ######################## ice4_fast_rs ###################################
    state_fast_rs = {
        "ldcompute": masks["ldcompute"],
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
        **{key: slopes[key] for key in ["lbdar", "lbdas"]},
        **{
            key: transformations[key]
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
            key: tnd[key]
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
        "grim_tmp": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
        "gacc_tmp": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
        "zw_tmp": ones((nx, ny, nz), backend=BACKEND),
        "zw1_tmp": ones((nx, ny, nz), backend=BACKEND),
        "zw2_tmp": ones((nx, ny, nz), backend=BACKEND),
        "zw3_tmp": ones((nx, ny, nz), backend=BACKEND),
        "index_floor": ones((nx, ny, nz), backend=BACKEND, dtype=np.int64),
        "index_float": ones((nx, ny, nz), backend=BACKEND),
        "index_floor_s": ones((nx, ny, nz), backend=BACKEND, dtype=np.int64),
        "index_float_s": ones((nx, ny, nz), backend=BACKEND),
        "index_floor_r": ones((nx, ny, nz), backend=BACKEND, dtype=np.int64),
        "index_float_r": ones((nx, ny, nz), backend=BACKEND),
    }

    # TODO: replace by real values
    gaminc_rim1 = from_array(np.ones((80)), backend=BACKEND)
    gaminc_rim2 = from_array(np.ones((80)), backend=BACKEND)
    gaminc_rim4 = from_array(np.ones((80)), backend=BACKEND)

    ker_raccs = from_array(np.ones((40, 40)), backend=BACKEND)
    ker_raccss = from_array(np.ones((40, 40)), backend=BACKEND)
    ker_saccrg = from_array(np.ones((40, 40)), backend=BACKEND)

    ice4_fast_rs(
        ldsoft=False,
        gaminc_rim1=gaminc_rim1,
        gaminc_rim2=gaminc_rim2,
        gaminc_rim4=gaminc_rim4,
        ker_raccs=ker_raccs,
        ker_raccss=ker_raccss,
        ker_saccrg=ker_saccrg,
        **state_fast_rs,
        **temporaries_fast_rs,
    )
    ice4_fast_rs(
        ldsoft=True,
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
        **{key: diags[key] for key in ["rgsi_mr", "rrhong_mr", "rsrimcg_mr"]},
    }

    ice4_fast_rg_pre_processing(**state_fast_rg_pp)

    ######################## ice4_fast_rg ###################################
    state_fast_rg = {
        "ldcompute": masks["ldcompute"],
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
        **{key: slopes[key] for key in ["lbdar", "lbdas", "lbdag"]},
        **{
            key: tnd[key]
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
            key: transformations[key]
            for key in ["ricfrrg", "rrcfrig", "ricfrr", "rgmltr"]
        },
    }

    temporaries_fast_rg = {
        "gdry": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
        "ldwetg": ones(
            (nx, ny, nz), backend=BACKEND, dtype=np.int64
        ),  # bool, true if graupel grows in wet mode (out)
        "lldryg": ones(
            (nx, ny, nz), backend=BACKEND, dtype=np.int64
        ),  # linked to gdry + temporary
        "rdryg_init_tmp": ones((nx, ny, nz), backend=BACKEND),
        "rwetg_init_tmp": ones((nx, ny, nz), backend=BACKEND),
        "zw_tmp": ones((nx, ny, nz), backend=BACKEND),  # ZZW in Fortran
        "index_floor_s": ones((nx, ny, nz), backend=BACKEND, dtype=np.int64),
        "index_floor_g": ones((nx, ny, nz), backend=BACKEND, dtype=np.int64),
        "index_floor_r": ones((nx, ny, nz), backend=BACKEND, dtype=np.int64),
        "index_float_s": ones((nx, ny, nz), backend=BACKEND),
        "index_float_g": ones((nx, ny, nz), backend=BACKEND),
        "index_float_r": ones((nx, ny, nz), backend=BACKEND),
    }

    ker_sdryg = from_array(np.ones((40, 40)), backend=BACKEND)
    ker_rdryg = from_array(np.ones((40, 40)), backend=BACKEND)

    ice4_fast_rg(
        ldsoft=False,
        ker_sdryg=ker_sdryg,
        ker_rdryg=ker_rdryg,
        **state_fast_rg,
        **temporaries_fast_rg,
    )
    ice4_fast_rg(
        ldsoft=True,
        ker_sdryg=ker_sdryg,
        ker_rdryg=ker_rdryg,
        **state_fast_rg,
        **temporaries_fast_rg,
    )

    ######################## ice4_fast_ri ###################################
    state_fast_ri = {
        "ldcompute": masks["ldcompute"],
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
        "rc_beri_tnd": tnd["rc_beri_tnd"],
    }

    ice4_fast_ri(ldsoft=False, **state_fast_ri)
    ice4_fast_ri(ldsoft=True, **state_fast_ri)

    ######################## ice4_tendencies_update #########################

    state_tendencies_update = {
        **{key: state[key] for key in ["ls_fact", "lv_fact"]},
        **tnd_update,
        **{
            key: transformations[key]
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
            key: diags[key]
            for key in ["rvheni_mr", "rrhong_mr", "rimltc_mr", "rsrimcg_mr"]
        },
    }

    ice4_tendencies_update(**state_tendencies_update)
