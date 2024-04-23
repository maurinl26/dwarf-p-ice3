# -*- coding: utf-8 -*-
import logging
import sys
from datetime import timedelta

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.framework.stencil import compile_stencil
from gt4py.storage import ones, zeros, from_array
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

    ################ Global state #################
    global_state = {
        "tht": ones((nx, ny, nz), backend=BACKEND),
        "pabs": ones((nx, ny, nz), backend=BACKEND),
        "rhodref": ones((nx, ny, nz), backend=BACKEND),
        "exn": ones((nx, ny, nz), backend=BACKEND),
        "ls_fact": ones((nx, ny, nz), backend=BACKEND),
        "t": ones((nx, ny, nz), backend=BACKEND),
        "rv_t": ones((nx, ny, nz), backend=BACKEND),
    }

    ############## ice4_nucleation ################
    logging.info(f"Compilation for ice4_nucleation")
    ice4_nucleation = compile_stencil("ice4_nucleation", gt4py_config, externals)

    state_nucleation = {
        "ldcompute": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
        "tht": ones((nx, ny, nz), backend=BACKEND),
        "pabs": ones((nx, ny, nz), backend=BACKEND),
        "rhodref": ones((nx, ny, nz), backend=BACKEND),
        "exn": ones((nx, ny, nz), backend=BACKEND),
        "ls_fact": ones((nx, ny, nz), backend=BACKEND),
        "t": ones((nx, ny, nz), backend=BACKEND),
        "rv_t": ones((nx, ny, nz), backend=BACKEND),
        "ci_t": ones((nx, ny, nz), backend=BACKEND),
        "rv_heni_mr": ones((nx, ny, nz), backend=BACKEND),
        "usw": ones((nx, ny, nz), backend=BACKEND),
        "w1": ones((nx, ny, nz), backend=BACKEND),
        "w2": ones((nx, ny, nz), backend=BACKEND),
        "ssi": ones((nx, ny, nz), backend=BACKEND),
    }

    # timestep
    ice4_nucleation(**state_nucleation)

    ############## ice4_nucleation_post_processing ####################
    logging.info(f"Compilation for ice4_nucleation_post_processing")
    ice4_nucleation_post_processing = compile_stencil(
        "ice4_nucleation_post_processing", gt4py_config, externals
    )

    state_nucleation_pp = {
        "t": ones((nx, ny, nz), backend=BACKEND),
        "exn": ones((nx, ny, nz), backend=BACKEND),
        "ls_fact": ones((nx, ny, nz), backend=BACKEND),
        "lv_fact": ones((nx, ny, nz), backend=BACKEND),
        "tht": ones((nx, ny, nz), backend=BACKEND),
        "rv_t": ones((nx, ny, nz), backend=BACKEND),
        "ri_t": ones((nx, ny, nz), backend=BACKEND),
        "rvheni_mr": ones((nx, ny, nz), backend=BACKEND),
    }

    # Timestep
    ice4_nucleation_post_processing(**state_nucleation_pp)

    ########################### ice4_rrhong #################################
    logging.info(f"Compilation for ice4_rrhong")
    ice4_rrhong = compile_stencil("ice4_rrhong", gt4py_config, externals)

    state_rrhong = {
        "ldcompute": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
        "t": ones((nx, ny, nz), backend=BACKEND),
        "exn": ones((nx, ny, nz), backend=BACKEND),
        "lv_fact": ones((nx, ny, nz), backend=BACKEND),
        "ls_fact": ones((nx, ny, nz), backend=BACKEND),
        "tht": ones((nx, ny, nz), backend=BACKEND),
        "rrhong_mr": ones((nx, ny, nz), backend=BACKEND),
        "rr_t": ones((nx, ny, nz), backend=BACKEND),
    }

    ice4_rrhong(**state_rrhong)

    ########################### ice4_rrhong_post_processing #################
    logging.info(f"Compilation for ice4_rrhong_post_processing")
    ice4_rrhong_post_processing = compile_stencil(
        "ice4_rrhong_post_processing", gt4py_config, externals
    )

    state_rrhong_pp = {
        "t": ones((nx, ny, nz), backend=BACKEND),
        "exn": ones((nx, ny, nz), backend=BACKEND),
        "lv_fact": ones((nx, ny, nz), backend=BACKEND),
        "ls_fact": ones((nx, ny, nz), backend=BACKEND),
        "tht": ones((nx, ny, nz), backend=BACKEND),
        "rg_t": ones((nx, ny, nz), backend=BACKEND),
        "rr_t": ones((nx, ny, nz), backend=BACKEND),
        "rrhong_mr": ones((nx, ny, nz), backend=BACKEND),
    }

    ice4_rrhong_post_processing(**state_rrhong_pp)

    ########################## ice4_rimltc ##################################
    logging.info(f"Compilation for ice4_rimltc")
    ice4_rimltc = compile_stencil("ice4_rimltc", gt4py_config, externals)

    state_rimltc = {
        "ldcompute": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
        "t": ones((nx, ny, nz), backend=BACKEND),
        "exn": ones((nx, ny, nz), backend=BACKEND),
        "lv_fact": ones((nx, ny, nz), backend=BACKEND),
        "ls_fact": ones((nx, ny, nz), backend=BACKEND),
        "tht": ones((nx, ny, nz), backend=BACKEND),
        "ri_t": ones((nx, ny, nz), backend=BACKEND),
        "rimltc_mr": ones((nx, ny, nz), backend=BACKEND),
    }

    ice4_rimltc(**state_rimltc)

    ####################### ice4_rimltc_post_processing #####################
    logging.info(f"Compilation for ice4_rimltc_post_processing")
    ice4_rimltc_post_processing = compile_stencil(
        "ice4_rimltc_post_processing", gt4py_config, externals
    )

    state_rimltc_pp = {
        "t": ones((nx, ny, nz), backend=BACKEND),
        "exn": ones((nx, ny, nz), backend=BACKEND),
        "lv_fact": ones((nx, ny, nz), backend=BACKEND),
        "ls_fact": ones((nx, ny, nz), backend=BACKEND),
        "tht": ones((nx, ny, nz), backend=BACKEND),
        "rc_t": ones((nx, ny, nz), backend=BACKEND),
        "ri_t": ones((nx, ny, nz), backend=BACKEND),
        "rimltc_mr": ones((nx, ny, nz), backend=BACKEND),
    }

    ice4_rimltc_post_processing(**state_rimltc_pp)

    ######################## ice4_increment_update ##########################
    logging.info(f"Compilation for ice4_increment_update")
    ice4_increment_update = compile_stencil(
        "ice4_increment_update", gt4py_config, externals
    )

    state_increment_update = {
        "ls_fact": ones((nx, ny, nz), backend=BACKEND),
        "lv_fact": ones((nx, ny, nz), backend=BACKEND),
        "theta_increment": ones((nx, ny, nz), backend=BACKEND),
        "rv_increment": ones((nx, ny, nz), backend=BACKEND),
        "rc_increment": ones((nx, ny, nz), backend=BACKEND),
        "rr_increment": ones((nx, ny, nz), backend=BACKEND),
        "ri_increment": ones((nx, ny, nz), backend=BACKEND),
        "rs_increment": ones((nx, ny, nz), backend=BACKEND),
        "rg_increment": ones((nx, ny, nz), backend=BACKEND),
        "rvheni_mr": ones((nx, ny, nz), backend=BACKEND),
        "rimltc_mr": ones((nx, ny, nz), backend=BACKEND),
        "rrhong_mr": ones((nx, ny, nz), backend=BACKEND),
        "rsrimcg_mr": ones((nx, ny, nz), backend=BACKEND),
    }

    ice4_increment_update(**state_increment_update)

    ######################## ice4_derived_fields ############################
    logging.info(f"Compilation for ice4_derived_fields")
    ice4_derived_fields = compile_stencil(
        "ice4_derived_fields", gt4py_config, externals
    )

    state_derived_fields = {
        "t": ones((nx, ny, nz), backend=BACKEND),
        "rhodref": ones((nx, ny, nz), backend=BACKEND),
        "pres": ones((nx, ny, nz), backend=BACKEND),
        "ssi": ones((nx, ny, nz), backend=BACKEND),
        "ka": ones((nx, ny, nz), backend=BACKEND),
        "dv": ones((nx, ny, nz), backend=BACKEND),
        "ai": ones((nx, ny, nz), backend=BACKEND),
        "cj": ones((nx, ny, nz), backend=BACKEND),
        "zw": ones((nx, ny, nz), backend=BACKEND),
        "rv_t": ones((nx, ny, nz), backend=BACKEND),
    }

    ice4_derived_fields(**state_derived_fields)

    ######################## ice4_slope_parameters ##########################
    logging.info(f"Compilation for ice4_slope_parameters")
    ice4_slope_parameters = compile_stencil(
        "ice4_slope_parameters", gt4py_config, externals
    )

    state_slope_parameters = {
        "rhodref": ones((nx, ny, nz), backend=BACKEND),
        "t": ones((nx, ny, nz), backend=BACKEND),
        "rr_t": ones((nx, ny, nz), backend=BACKEND),
        "rs_t": ones((nx, ny, nz), backend=BACKEND),
        "rg_t": ones((nx, ny, nz), backend=BACKEND),
        "lbdar": ones((nx, ny, nz), backend=BACKEND),
        "lbdar_rf": ones((nx, ny, nz), backend=BACKEND),
        "lbdas": ones((nx, ny, nz), backend=BACKEND),
        "lbdag": ones((nx, ny, nz), backend=BACKEND),
    }

    ice4_slope_parameters(**state_slope_parameters)

    ######################## ice4_slow ######################################
    logging.info(f"Compilation for ice4_slow")
    ice4_slow = compile_stencil("ice4_slow", gt4py_config, externals)

    state_slow = {
        "ldcompute": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
        "rhodref": ones((nx, ny, nz), backend=BACKEND),
        "t": ones((nx, ny, nz), backend=BACKEND),
        "ssi": ones((nx, ny, nz), backend=BACKEND),
        "lv_fact": ones((nx, ny, nz), backend=BACKEND),
        "ls_fact": ones((nx, ny, nz), backend=BACKEND),
        "rv_t": ones((nx, ny, nz), backend=BACKEND),
        "rc_t": ones((nx, ny, nz), backend=BACKEND),
        "ri_t": ones((nx, ny, nz), backend=BACKEND),
        "rs_t": ones((nx, ny, nz), backend=BACKEND),
        "rg_t": ones((nx, ny, nz), backend=BACKEND),
        "lbdas": ones((nx, ny, nz), backend=BACKEND),
        "lbdag": ones((nx, ny, nz), backend=BACKEND),
        "ai": ones((nx, ny, nz), backend=BACKEND),
        "cj": ones((nx, ny, nz), backend=BACKEND),
        "hli_hcf": ones((nx, ny, nz), backend=BACKEND),
        "hli_hri": ones((nx, ny, nz), backend=BACKEND),
        "rc_honi_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rv_deps_tnd": ones((nx, ny, nz), backend=BACKEND),
        "ri_aggs_tnd": ones((nx, ny, nz), backend=BACKEND),
        "ri_auts_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rv_depg_tnd": ones((nx, ny, nz), backend=BACKEND),
    }

    ice4_slow(ldsoft=False, **state_slow)
    ice4_slow(ldsoft=True, **state_slow)

    ######################## ice4_warm ######################################
    logging.info(f"Compilation for ice4_warm")
    ice4_warm = compile_stencil("ice4_warm", gt4py_config, externals)

    state_warm = {
        "ldcompute": ones(
            (nx, ny, nz), backend=BACKEND, dtype=bool
        ),  # boolean field for microphysics computation
        "rhodref": ones((nx, ny, nz), backend=BACKEND),
        "lv_fact": ones((nx, ny, nz), backend=BACKEND),
        "t": ones((nx, ny, nz), backend=BACKEND),  # temperature
        "pres": ones((nx, ny, nz), backend=BACKEND),
        "tht": ones((nx, ny, nz), backend=BACKEND),
        "lbda_r": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # slope parameter for the rain drop distribution
        "lbda_r_rf": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # slope parameter for the rain fraction part
        "ka": ones((nx, ny, nz), backend=BACKEND),  # thermal conductivity of the air
        "dv": ones((nx, ny, nz), backend=BACKEND),  # diffusivity of water vapour
        "cj": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # function to compute the ventilation coefficient
        "hlc_hcf": ones((nx, ny, nz), backend=BACKEND),  # High Cloud Fraction in grid
        "hlc_lcf": ones((nx, ny, nz), backend=BACKEND),  # Low Cloud Fraction in grid
        "hlc_hrc": ones((nx, ny, nz), backend=BACKEND),  # LWC that is high in grid
        "hlc_lrc": ones((nx, ny, nz), backend=BACKEND),  # LWC that is low in grid
        "cf": ones((nx, ny, nz), backend=BACKEND),  # cloud fraction
        "rf": ones((nx, ny, nz), backend=BACKEND),  # rain fraction
        "rv_t": ones((nx, ny, nz), backend=BACKEND),  # water vapour mixing ratio at t
        "rc_t": ones((nx, ny, nz), backend=BACKEND),  # cloud water mixing ratio at t
        "rr_t": ones((nx, ny, nz), backend=BACKEND),  # rain water mixing ratio at t
        "rc_autr": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # autoconversion of rc for rr production
        "rc_accr": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # accretion of r_c for r_r production
        "rr_evap": ones((nx, ny, nz), backend=BACKEND),  # evaporation of rr
    }

    ice4_warm(ldsoft=False, **state_warm)
    ice4_warm(ldsoft=True, **state_warm)

    ######################## ice4_fast_rs ###################################
    logging.info(f"Compilation for ice4_fast_rs")
    ice4_fast_rs = compile_stencil("ice4_fast_rs", gt4py_config, externals)

    state_fast_rs = {
        "ldcompute": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
        "rhodref": ones((nx, ny, nz), backend=BACKEND),
        "lv_fact": ones((nx, ny, nz), backend=BACKEND),
        "ls_fact": ones((nx, ny, nz), backend=BACKEND),
        "pres": ones((nx, ny, nz), backend=BACKEND),  # absolute pressure at t
        "dv": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # diffusivity of water vapor in the air
        "ka": ones((nx, ny, nz), backend=BACKEND),  # thermal conductivity of the air
        "cj": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # function to compute the ventilation coefficient
        "lbda_r": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # Slope parameter for rain distribution
        "lbda_s": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # Slope parameter for snow distribution
        "t": ones((nx, ny, nz), backend=BACKEND),  # Temperature
        "rv_t": ones((nx, ny, nz), backend=BACKEND),  # vapour m.r. at t
        "rc_t": ones((nx, ny, nz), backend=BACKEND),
        "rr_t": ones((nx, ny, nz), backend=BACKEND),
        "rs_t": ones((nx, ny, nz), backend=BACKEND),
        "ri_aggs": ones((nx, ny, nz), backend=BACKEND),  # ice aggregation on snow
        "rc_rimss_out": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # cloud droplet riming of the aggregates
        "rc_rimsg_out": ones((nx, ny, nz), backend=BACKEND),
        "rs_rimcg_out": ones((nx, ny, nz), backend=BACKEND),
        "rr_accss_out": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # rain accretion onto the aggregates
        "rr_accsg_out": ones((nx, ny, nz), backend=BACKEND),
        "rs_accrg_out": ones((nx, ny, nz), backend=BACKEND),
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
    )

    ######################## ice4_fast_rg_pre_processing ####################
    logging.info(f"Compilation for ice4_fast_rg_pre_processing")
    ice4_fast_rg_pre_processing = compile_stencil(
        "ice4_fast_rg_pre_processing", gt4py_config, externals
    )

    state_fast_rg_pp = {
        "rgsi": ones((nx, ny, nz), backend=BACKEND),
        "rgsi_mr": ones((nx, ny, nz), backend=BACKEND),
        "rvdepg": ones((nx, ny, nz), backend=BACKEND),
        "rsmltg": ones((nx, ny, nz), backend=BACKEND),
        "rraccsg": ones((nx, ny, nz), backend=BACKEND),
        "rsaccrg": ones((nx, ny, nz), backend=BACKEND),
        "rcrimsg": ones((nx, ny, nz), backend=BACKEND),
        "rsrimcg": ones((nx, ny, nz), backend=BACKEND),
        "rrhong_mr": ones((nx, ny, nz), backend=BACKEND),
        "rsrimcg_mr": ones((nx, ny, nz), backend=BACKEND),
    }

    ice4_fast_rg_pre_processing(**state_fast_rg_pp)

    ######################## ice4_fast_rg ###################################
    logging.info(f"Compilation for ice4_fast_rg")
    ice4_fast_rg = compile_stencil("ice4_fast_rg", gt4py_config, externals)

    state_fast_rg = {
        "ldcompute": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
        "t": ones((nx, ny, nz), backend=BACKEND),
        "rhodref": ones((nx, ny, nz), backend=BACKEND),
        "pres": ones((nx, ny, nz), backend=BACKEND),
        "rv_t": ones((nx, ny, nz), backend=BACKEND),
        "rr_t": ones((nx, ny, nz), backend=BACKEND),
        "ri_t": ones((nx, ny, nz), backend=BACKEND),
        "rg_t": ones((nx, ny, nz), backend=BACKEND),
        "rc_t": ones((nx, ny, nz), backend=BACKEND),
        "rs_t": ones((nx, ny, nz), backend=BACKEND),
        "ci_t": ones((nx, ny, nz), backend=BACKEND),
        "ka": ones((nx, ny, nz), backend=BACKEND),
        "dv": ones((nx, ny, nz), backend=BACKEND),
        "cj": ones((nx, ny, nz), backend=BACKEND),
        "lbdar": ones((nx, ny, nz), backend=BACKEND),
        "lbdas": ones((nx, ny, nz), backend=BACKEND),
        "lbdag": ones((nx, ny, nz), backend=BACKEND),
        "ricfrrg": ones((nx, ny, nz), backend=BACKEND),
        "rrcfrig": ones((nx, ny, nz), backend=BACKEND),
        "ricfrr": ones((nx, ny, nz), backend=BACKEND),
        "rg_rcdry_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_ridry_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_rsdry_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_rrdry_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_riwet_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_rswet_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_freez1_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_freez2_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_mltr": ones((nx, ny, nz), backend=BACKEND),
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
        ldsoft=False, ker_sdryg=ker_sdryg, ker_rdryg=ker_rdryg, **state_fast_rg
    )
    ice4_fast_rg(ldsoft=True, ker_sdryg=ker_sdryg, ker_rdryg=ker_rdryg, **state_fast_rg)

    ######################## ice4_fast_ri ###################################
    logging.info(f"Compilation for ice4_fast_ri")
    ice4_fast_ri = compile_stencil("ice4_fast_ri", gt4py_config, externals)

    state_fast_ri = {
        "ldcompute": ones((nx, ny, nz), backend=BACKEND, dtype=bool),
        "rhodref": ones((nx, ny, nz), backend=BACKEND),
        "lv_fact": ones((nx, ny, nz), backend=BACKEND),
        "ls_fact": ones((nx, ny, nz), backend=BACKEND),
        "ai": ones((nx, ny, nz), backend=BACKEND),
        "cj": ones((nx, ny, nz), backend=BACKEND),
        "ci_t": ones((nx, ny, nz), backend=BACKEND),
        "ssi": ones((nx, ny, nz), backend=BACKEND),
        "rc_t": ones((nx, ny, nz), backend=BACKEND),
        "ri_t": ones((nx, ny, nz), backend=BACKEND),
        "rc_beri_tnd": ones((nx, ny, nz), backend=BACKEND),
    }

    ice4_fast_ri(ldsoft=False, **state_fast_ri)
    ice4_fast_ri(ldsoft=True, **state_fast_ri)

    ######################## ice4_tendencies_update #########################
    logging.info(f"Compilation for ice4_tendencies_update")
    ice4_tendencies_update = compile_stencil(
        "ice4_tendencies_update", gt4py_config, externals
    )

    state_tendencies_update = {
        "ls_fact": ones((nx, ny, nz), backend=BACKEND),
        "lv_fact": ones((nx, ny, nz), backend=BACKEND),
        "theta_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rv_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rc_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rr_tnd": ones((nx, ny, nz), backend=BACKEND),
        "ri_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rs_tnd": ones((nx, ny, nz), backend=BACKEND),
        "rg_tnd": ones((nx, ny, nz), backend=BACKEND),
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
        "rgmltr": ones((nx, ny, nz), backend=BACKEND),  # 31  # Melting of the graupel
        "rvheni_mr": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # 43  # heterogeneous nucleation mixing ratio change
        "rrhong_mr": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # 44  # Spontaneous freezing mixing ratio change
        "rimltc_mr": ones(
            (nx, ny, nz), backend=BACKEND
        ),  # 45  # Cloud ce melting mixing ratio change
        "rsrimcg_mr": ones((nx, ny, nz), backend=BACKEND),
    }

    ice4_tendencies_update(**state_tendencies_update)
