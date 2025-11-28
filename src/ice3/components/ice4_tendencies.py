# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from datetime import timedelta
from functools import cached_property, partial
from itertools import repeat
from typing import Dict, Tuple
from numpy.typing import NDArray

from gt4py.storage import from_array
from gt4py.cartesian.gtscript import stencil

from ..phyex_common.phyex import Phyex
from ..phyex_common.xker_raccs import KER_RACCS, KER_RACCSS, KER_SACCRG
from ..phyex_common.xker_rdryg import KER_RDRYG
from ..phyex_common.xker_sdryg import KER_SDRYG
from ..stencils.ice4_compute_pdf import ice4_compute_pdf
from ..utils.env import DTYPES, BACKEND
from ..utils.storage import managed_temporaries

log = logging.getLogger(__name__)


class Ice4Tendencies:
    """Implicit Tendency Component calling
    ice_adjust : saturation adjustment of temperature and mixing ratios

    ice_adjust stencil is ice_adjust.F90 in PHYEX
    """

    def __init__(
        self,
        phyex: Phyex = Phyex("AROME"),
        dtypes: Dict = DTYPES,
        backend: str = BACKEND,
    ) -> None:

        self.backend = backend

        compile_stencil = partial(
            stencil,
            backend=backend,
            externals=phyex.externals,
            dtypes=dtypes
        )

        self.gaminc_rim1 = phyex.rain_ice_param.GAMINC_RIM1
        self.gaminc_rim2 = phyex.rain_ice_param.GAMINC_RIM2
        self.gaminc_rim4 = phyex.rain_ice_param.GAMINC_RIM4

        # Tendencies
        from ..stencils.ice4_nucleation import ice4_nucleation
        from ..stencils.ice4_tendencies import (
            ice4_nucleation_post_processing, ice4_rrhong_post_processing, ice4_rimltc_post_processing,
            ice4_slope_parameters, ice4_fast_rg_pre_post_processing, ice4_total_tendencies_update,
            ice4_increment_update, ice4_derived_fields
        )
        from ..stencils.ice4_rrhong import ice4_rrhong
        from ..stencils.ice4_rimltc import ice4_rimltc
        from ..stencils.ice4_slow import ice4_slow
        from ..stencils.ice4_warm import ice4_warm
        from ..stencils.ice4_fast_rs import ice4_fast_rs
        from ..stencils.ice4_fast_ri import ice4_fast_ri
        from ..stencils.ice4_fast_rg import ice4_fast_rg

        self.ice4_nucleation = compile_stencil(
            name="ice4_nucleation",
            definition=ice4_nucleation
        )
        self.ice4_nucleation_post_processing = compile_stencil(
            name="ice4_nucleation_post_processing",
            definition=ice4_nucleation_post_processing
        )
        self.ice4_rrhong = compile_stencil(
            name="ice4_rrhong",
            definition=ice4_rrhong
        )
        self.ice4_rrhong_post_processing = compile_stencil(
            name="ice4_rrhong_post_processing",
            definition=ice4_rrhong_post_processing
        )
        self.ice4_rimltc = compile_stencil(
            name="ice4_rimltc",
            definition=ice4_rimltc

        )
        self.ice4_rimltc_post_processing = compile_stencil(
            name="ice4_rimltc_post_processing",
            definition=ice4_rimltc_post_processing
        )

        self.ice4_increment_update = compile_stencil(
            name="ice4_increment_update",
            definition=ice4_increment_update
        )
        self.ice4_derived_fields = compile_stencil(
            name="ice4_derived_fields",
            definition=ice4_derived_fields
        )

        # TODO: add ice4_compute_pdf
        self.ice4_slope_parameters = compile_stencil(
            name="ice4_slope_parameters",
            definition=ice4_slope_parameters
        )
        self.ice4_slow = compile_stencil(
            name="ice4_slow",
            definition=ice4_slow
        )
        self.ice4_warm = compile_stencil(
            name="ice4_warm",
            definition=ice4_warm
        )
        self.ice4_fast_rs = compile_stencil(
            name="ice4_fast_rs",
            definition=ice4_fast_rs
        )
        self.ice4_fast_rg_pre_processing = compile_stencil(
            name="ice4_fast_rg_pre_processing",
            definition=ice4_fast_rg_pre_post_processing
        )
        self.ice4_fast_rg = compile_stencil(
            name="ice4_fast_rg",
            definition=ice4_fast_rg
        )
        self.ice4_fast_ri = compile_stencil(
            name="ice4_fast_ri",
            definition=ice4_fast_ri
        )
        self.ice4_total_tendencies_update = compile_stencil(
            name="ice4_total_tendencies_update",
            definition=ice4_total_tendencies_update
        )

        self.ice4_compute_pdf = compile_stencil(
            name="ice4_compute_pdf",
            definition=ice4_compute_pdf
        )


    def __call__(
        self,
        ldsoft: bool,
        state: Dict[str, NDArray],
        timestep: timedelta,
        out_tendencies: Dict[str, NDArray],
        out_diagnostics: Dict[str, NDArray],
        overwrite_tendencies: Dict[str, bool],
        domain: Tuple,
        exec_info: Dict,
        validate_args: bool = True
    ) -> None:

        # todo : replace managed context
        with managed_temporaries(
            *repeat((domain, "float"), 63),
            *repeat((domain, "int"), 4),
        ) as (
            # mr
            rvheni_mr,
            rrhong_mr,
            rimltc_mr,
            rgsi_mr,
            rsrimcg_mr,
            # slopes
            lbdar,
            lbdar_rf,
            lbdas,
            lbdag,
            # tnd
            rc_honi_tnd,
            rv_deps_tnd,
            ri_aggs_tnd,
            ri_auts_tnd,
            rv_depg_tnd,
            rs_mltg_tnd,
            rc_mltsr_tnd,
            rs_rcrims_tnd,
            rs_rcrimss_tnd,
            rs_rsrimcg_tnd,
            rs_rraccs_tnd,
            rs_rraccss_tnd,
            rs_rsaccrg_tnd,
            rs_freez1_tnd,
            rs_freez2_tnd,
            rg_rcdry_tnd,
            rg_ridry_tnd,
            rg_rsdry_tnd,
            rg_rrdry_tnd,
            rg_riwet_tnd,
            rg_rswet_tnd,
            rg_freez1_tnd,
            rg_freez2_tnd,
            rc_beri_tnd,
            # transfos
            rgsi,
            rchoni,
            rvdeps,
            riaggs,
            riauts,
            rvdepg,
            rcautr,
            rcaccr,
            rrevav,
            rcberi,
            rsmltg,
            rcmltsr,
            rraccss,  # 13
            rraccsg,  # 14
            rsaccrg,  # 15
            rcrimss,  # 16
            rcrimsg,  # 17
            rsrimcg,  # 18
            ricfrrg,  # 19
            rrcfrig,  # 20
            ricfrr,  # 21
            rcwetg,  # 22
            riwetg,  # 23
            rrwetg,  # 24
            rswetg,  # 25
            rcdryg,  # 26
            ridryg,  # 27
            rrdryg,  # 28
            rsdryg,  # 29
            rgmltr,  # 31
            index_floor,
            index_floor_r,
            index_floor_s,
            index_floor_g,
        ):

            ############## ice4_nucleation ################
            state_nucleation = {
                **{
                    key: state[key]
                    for key in [
                        "ldcompute",
                        "th_t",
                        "rhodref",
                        "exn",
                        "ls_fact",
                        "t",
                        "rv_t",
                        "ci_t",
                        "ssi",
                    ]
                },
                **{"pabs_t": state["pres"]},
            }

            temporaries_nucleation = {
                "rvheni_mr": rvheni_mr,
            }

            # timestep
            self.ice4_nucleation(
                **state_nucleation, 
                **temporaries_nucleation,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
                )

            ############## ice4_nucleation_post_processing ####################

            state_nucleation_pp = {
                **{
                    key: state[key]
                    for key in [
                        "t",
                        "exn",
                        "ls_fact",
                        "lv_fact",
                        "th_t",
                        "rv_t",
                        "ri_t",
                    ]
                },
            }

            tmps_nucleation_pp = {"rvheni_mr": rvheni_mr}

            # Timestep
            self.ice4_nucleation_post_processing(
                **state_nucleation_pp, 
                **tmps_nucleation_pp,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
            )

            ########################### ice4_rrhong #################################
            state_rrhong = {
                "ldcompute": state["ldcompute"],
                **{
                    key: state[key]
                    for key in ["t", "exn", "lv_fact", "ls_fact", "th_t", "rr_t"]
                },
            }

            tmps_rrhong = {"rrhong_mr": rrhong_mr}

            self.ice4_rrhong(
                **state_rrhong, 
                **tmps_rrhong,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
            )

            ########################### ice4_rrhong_post_processing #################
            state_rrhong_pp = {
                **{
                    key: state[key]
                    for key in [
                        "t",
                        "exn",
                        "lv_fact",
                        "ls_fact",
                        "th_t",
                        "rg_t",
                        "rr_t",
                    ]
                },
            }

            self.ice4_rrhong_post_processing(
                **state_rrhong_pp, 
                **tmps_rrhong,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,)

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
                        "th_t",
                        "ri_t",
                    ]
                },
            }

            tmps_rimltc = {"rimltc_mr": rimltc_mr}

            self.ice4_rimltc(
                **state_rimltc, 
                **tmps_rimltc,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,)

            ####################### ice4_rimltc_post_processing #####################

            state_rimltc_pp = {
                **{
                    key: state[key]
                    for key in [
                        "t",
                        "exn",
                        "lv_fact",
                        "ls_fact",
                        "th_t",
                        "rc_t",
                        "ri_t",
                    ]
                },
            }

            self.ice4_rimltc_post_processing(
                **state_rimltc_pp, 
                **tmps_rimltc,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,)

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
            }

            tmps_increment_update = {
                "rvheni_mr": rvheni_mr,
                "rimltc_mr": rimltc_mr,
                "rrhong_mr": rrhong_mr,
                "rsrimcg_mr": rsrimcg_mr,
            }

            self.ice4_increment_update(
                **state_increment_update, 
                **tmps_increment_update,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
            )

            ######################## ice4_compute_pdf ###############################
            state_compute_pdf = {
                key: state[key]
                for key in [
                    "ldcompute",
                    "rhodref",
                    "rc_t",
                    "ri_t",
                    "cf",
                    "t",
                    "sigma_rc",
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

            self.ice4_compute_pdf(
                **state_compute_pdf,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,)

            # l263 to l278 omitted because LLRFR is False in AROME

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

            self.ice4_derived_fields(
                **state_derived_fields,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,)

            ######################## ice4_slope_parameters ##########################
            state_slope_parameters = {
                **{key: state[key] for key in ["rhodref", "t", "rr_t", "rs_t", "rg_t"]},
            }

            tmps_slopes = {
                "lbdar": lbdar,
                "lbdar_rf": lbdar_rf,
                "lbdas": lbdas,
                "lbdag": lbdag,
            }

            self.ice4_slope_parameters(
                **state_slope_parameters, 
                **tmps_slopes,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,)

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
            }

            tmps_slow = {
                "lbdas": lbdas,
                "lbdag": lbdag,
                "rc_honi_tnd": rc_honi_tnd,
                "rv_deps_tnd": rv_deps_tnd,
                "ri_aggs_tnd": ri_aggs_tnd,
                "ri_auts_tnd": ri_auts_tnd,
                "rv_depg_tnd": rv_depg_tnd,
            }

            self.ice4_slow(
                ldsoft=ldsoft, 
                **state_slow, 
                **tmps_slow,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,)

            ######################## ice4_warm ######################################
            state_warm = {
                "ldcompute": state["ldcompute"],
                **{
                    key: state[key]
                    for key in [
                        "rhodref",
                        "lv_fact",
                        "t",  # temperature
                        "th_t",
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
            }

            tmps_warm = {
                "lbdar": lbdar,
                "lbdar_rf": lbdar_rf,
                "rcautr": rcautr,
                "rcaccr": rcaccr,
                "rrevav": rrevav,
            }

            self.ice4_warm(
                ldsoft=ldsoft, 
                **state_warm, 
                **tmps_warm,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,)

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
            }

            temporaries_fast_rs = {
                "lbdar": lbdar,
                "lbdar_rf": lbdar_rf,
                "rs_mltg_tnd": rs_mltg_tnd,
                "rc_mltsr_tnd": rc_mltsr_tnd,
                "rs_rcrims_tnd": rs_rcrims_tnd,  # extra dimension 8 in Fortran PRS_TEND
                "rs_rcrimss_tnd": rs_rcrimss_tnd,
                "rs_rsrimcg_tnd": rs_rsrimcg_tnd,
                "rs_rraccs_tnd": rs_rraccs_tnd,
                "rs_rraccss_tnd": rs_rraccss_tnd,
                "rs_rsaccrg_tnd": rs_rsaccrg_tnd,
                "rs_freez1_tnd": rs_freez1_tnd,
                "rs_freez2_tnd": rs_freez2_tnd,
                "riaggs": riaggs,
                "rcrimss": rcrimss,
                "rcrimsg": rcrimsg,
                "rsrimcg": rsrimcg,
                "rraccss": rraccss,
                "rraccsg": rraccsg,
                "rsaccrg": rsaccrg,
                "index_floor": index_floor,
                "index_floor_r": index_floor_r,
                "index_floor_s": index_floor_s,
            }

            gaminc_rim1 = from_array(
                self.gaminc_rim1, backend=self.backend
            )
            gaminc_rim2 = from_array(
                self.gaminc_rim2, backend=self.backend
            )
            gaminc_rim4 = from_array(
                self.gaminc_rim4, backend=self.backend
            )

            ker_raccs = from_array(KER_RACCS, backend=self.backend)
            ker_raccss = from_array(
                KER_RACCSS, backend=self.backend
            )
            ker_saccrg = from_array(KER_SACCRG, backend=self.backend)

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
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
            )

            ######################## ice4_fast_rg_pre_processing ####################
            state_fast_rg_pp = {
                **{
                    key: state[key]
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
            }

            tmps_fast_rg_pp = {
                "rgsi_mr": rgsi_mr,
                "rrhong_mr": rrhong_mr,
                "rsrimcg_mr": rsrimcg_mr,
            }

            self.ice4_fast_rg_pre_processing(
                **state_fast_rg_pp, 
                **tmps_fast_rg_pp,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,)

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
            }

            temporaries_fast_rg = {
                "lbdar": lbdar,
                "lbdas": lbdas,
                "lbdag": lbdag,
                "rg_rcdry_tnd": rg_rcdry_tnd,
                "rg_ridry_tnd": rg_ridry_tnd,
                "rg_rsdry_tnd": rg_rsdry_tnd,
                "rg_rrdry_tnd": rg_rrdry_tnd,
                "rg_riwet_tnd": rg_riwet_tnd,
                "rg_rswet_tnd": rg_rswet_tnd,
                "rg_freez1_tnd": rg_freez1_tnd,
                "rg_freez2_tnd": rg_freez2_tnd,
                "ricfrrg": ricfrrg,
                "rrcfrig": rrcfrig,
                "ricfrr": ricfrr,
                "rgmltr": rgmltr,
                "index_floor_s": index_floor_s,
                "index_floor_g": index_floor_g,
                "index_floor_r": index_floor_r,
            }

            ker_sdryg = from_array(KER_SDRYG, backend=self.backend)
            ker_rdryg = from_array(KER_RDRYG, backend=self.backend)

            self.ice4_fast_rg(
                ldsoft=ldsoft,
                ker_sdryg=ker_sdryg,
                ker_rdryg=ker_rdryg,
                **state_fast_rg,
                **temporaries_fast_rg,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
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
            }

            tmps_fast_ri = {
                "rc_beri_tnd": rc_beri_tnd,
            }

            self.ice4_fast_ri(
                ldsoft=ldsoft, 
                **state_fast_ri, 
                **tmps_fast_ri,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,)

            ######################## ice4_total_tendencies_update #########################

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
            }

            tmps_tnd_update = {
                "rvheni_mr": rvheni_mr,
                "rrhong_mr": rrhong_mr,
                "rimltc_mr": rimltc_mr,
                "rsrimcg_mr": rsrimcg_mr,
                "rchoni": rchoni,
                "rvdeps": rvdeps,
                "riaggs": riaggs,
                "riauts": riauts,
                "rvdepg": rvdepg,
                "rcautr": rcautr,
                "rcaccr": rcaccr,
                "rrevav": rrevav,
                "rcberi": rcberi,
                "rsmltg": rsmltg,
                "rcmltsr": rcmltsr,
                "rraccss": rraccss,  # 13
                "rraccsg": rraccsg,  # 14
                "rsaccrg": rsaccrg,  # 15  # Rain accretion onto the aggregates
                "rcrimss": rcrimss,  # 16
                "rcrimsg": rcrimsg,  # 17
                "rsrimcg": rsrimcg,  # 18  # Cloud droplet riming of the aggregates
                "ricfrrg": ricfrrg,  # 19
                "rrcfrig": rrcfrig,  # 20
                "ricfrr": ricfrr,  # 21  # Rain contact freezing
                "rcwetg": rcwetg,  # 22
                "riwetg": riwetg,  # 23
                "rrwetg": rrwetg,  # 24
                "rswetg": rswetg,  # 25  # Graupel wet growth
                "rcdryg": rcdryg,  # 26
                "ridryg": ridryg,  # 27
                "rrdryg": rrdryg,  # 28
                "rsdryg": rsdryg,  # 29  # Graupel dry growth
                "rgmltr": rgmltr,
            }

            self.ice4_total_tendencies_update(
                **state_tendencies_update, 
                **tmps_tnd_update,
                rwetgh=state.get("rwetgh", 0.0),
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
            )
