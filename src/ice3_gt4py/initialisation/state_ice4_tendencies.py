# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Dict

from gt4py.storage import ones
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array

from ice3_gt4py.initialisation.state import initialize_state_with_constant
from ice3_gt4py.initialisation.utils import initialize_field

if TYPE_CHECKING:
    from typing import Literal, Tuple

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
    from ifs_physics_common.utils.typingx import (
        DataArray,
        DataArrayDict,
        NDArrayLikeDict,
    )

############################## Ice4Tendencies #################################
def allocate_state_ice4_tendencies(
    computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig
) -> NDArrayLikeDict:
    """Allocate field to state keys following type (float, int, bool) and dimensions (2D, 3D).

    Args:
        computational_grid (ComputationalGrid): grid indexes
        gt4py_config (GT4PyConfig): gt4py configuration

    Returns:
        NDArrayLikeDict: dictionnary of field with associated keys for field name
    """

    def _allocate(
        grid_id: Tuple[DimSymbol, ...],
        units: str,
        dtype: Literal["bool", "float", "int"],
    ) -> DataArray:
        return allocate_data_array(
            computational_grid, grid_id, units, gt4py_config=gt4py_config, dtype=dtype
        )

    allocate_b_ij = partial(_allocate, grid_id=(I, J), units="", dtype="bool")
    allocate_b = partial(_allocate, grid_id=(I, J, K), units="", dtype="bool")
    allocate_f = partial(_allocate, grid_id=(I, J, K), units="", dtype="float")
    allocate_f_h = partial(
        _allocate, grid_id=(I, J, K - 1 / 2), units="", dtype="float"
    )
    allocate_f_ij = partial(_allocate, grid_id=(I, J), units="", dtype="float")
    allocate_i_ij = partial(_allocate, grid_id=(I, J), units="", dtype="int")

    time_state = {
        "t_micro": allocate_f(),
        "t_soft": allocate_f(),
    }

    masks = {"ldcompute": allocate_b(), "ldmicro": allocate_b()}

    state = {
        "tht": allocate_f(),
        "pabs": allocate_f(),
        "rhodref": allocate_f(),
        "exn": allocate_f(),
        "ls_fact": allocate_f(),
        "lv_fact": allocate_f(),
        "t": allocate_f(),
        "rv_t": allocate_f(),
        "rc_t": allocate_f(),
        "rr_t": allocate_f(),
        "ri_t": allocate_f(),
        "rs_t": allocate_f(),
        "rg_t": allocate_f(),
        "ci_t": allocate_f(),
        "pres": allocate_f(),
        "ssi": allocate_f(),  # supersaturation over ice
        "ka": allocate_f(),  #
        "dv": allocate_f(),
        "ai": allocate_f(),
        "cj": allocate_f(),
        "hlc_hcf": allocate_f(),  # High Cloud Fraction in grid
        "hlc_lcf": allocate_f(),  # Low Cloud Fraction in grid
        "hlc_hrc": allocate_f(),  # LWC that is high in grid
        "hlc_lrc": allocate_f(),
        "hli_hcf": allocate_f(),
        "hli_hri": allocate_f(),
    }

    slopes = {
        "lbdar": allocate_f(),
        "lbdar_rf": allocate_f(),
        "lbdas": allocate_f(),
        "lbdag": allocate_f(),
    }

    increments = {
        "theta_increment": allocate_f(),
        "rv_increment": allocate_f(),
        "rc_increment": allocate_f(),
        "rr_increment": allocate_f(),
        "ri_increment": allocate_f(),
        "rs_increment": allocate_f(),
        "rg_increment": allocate_f(),
    }

    transformations = {
        "rgsi": allocate_f(),
        "rchoni": allocate_f(),
        "rvdeps": allocate_f(),
        "riaggs": allocate_f(),
        "riauts": allocate_f(),
        "rvdepg": allocate_f(),
        "rcautr": allocate_f(),
        "rcaccr": allocate_f(),
        "rrevav": allocate_f(),
        "rcberi": allocate_f(),
        "rsmltg": allocate_f(),
        "rcmltsr": allocate_f(),
        "rraccss": allocate_f(),  # 13
        "rraccsg": allocate_f(),  # 14
        "rsaccrg": allocate_f(),  # 15  # Rain accretion onto the aggregates
        "rcrimss": allocate_f(),  # 16
        "rcrimsg": allocate_f(),  # 17
        "rsrimcg": allocate_f(),  # 18  # Cloud droplet riming of the aggregates
        "ricfrrg": allocate_f(),  # 19
        "rrcfrig": allocate_f(),  # 20
        "ricfrr": allocate_f(),  # 21  # Rain contact freezing
        "rcwetg": allocate_f(),  # 22
        "riwetg": allocate_f(),  # 23
        "rrwetg": allocate_f(),  # 24
        "rswetg": allocate_f(),  # 25  # Graupel wet growth
        "rcdryg": allocate_f(),  # 26
        "ridryg": allocate_f(),  # 27
        "rrdryg": allocate_f(),  # 28
        "rsdryg": allocate_f(),  # 29  # Graupel dry growth
        "rgmltr": allocate_f(),  # 31
    }

    diags = {
        "rvheni_mr": allocate_f(),
        "rrhong_mr": allocate_f(),
        "rimltc_mr": allocate_f(),
        "rgsi_mr": allocate_f(),
        "rsrimcg_mr": allocate_f(),
    }

    tnd = {
        "rc_honi_tnd": allocate_f(),
        "rv_deps_tnd": allocate_f(),
        "ri_aggs_tnd": allocate_f(),
        "ri_auts_tnd": allocate_f(),
        "rv_depg_tnd": allocate_f(),
        "rs_mltg_tnd": allocate_f(),  # conversion-melting of the aggregates
        "rc_mltsr_tnd": allocate_f(),  # cloud droplet collection onto aggregates
        "rs_rcrims_tnd": allocate_f(),  # extra dimension 8 in Fortran PRS_TEND
        "rs_rcrimss_tnd": allocate_f(),
        "rs_rsrimcg_tnd": allocate_f(),
        "rs_rraccs_tnd": allocate_f(),
        "rs_rraccss_tnd": allocate_f(),
        "rs_rsaccrg_tnd": allocate_f(),
        "rs_freez1_tnd": allocate_f(),
        "rs_freez2_tnd": allocate_f(),
        "rg_rcdry_tnd": allocate_f(),
        "rg_ridry_tnd": allocate_f(),
        "rg_rsdry_tnd": allocate_f(),
        "rg_rrdry_tnd": allocate_f(),
        "rg_riwet_tnd": allocate_f(),
        "rg_rswet_tnd": allocate_f(),
        "rg_freez1_tnd": allocate_f(),
        "rg_freez2_tnd": allocate_f(),
        "rc_beri_tnd": allocate_f(),
    }

    # Used in state tendencies update
    tnd_update = {
        "theta_tnd": allocate_f(),
        "rv_tnd": allocate_f(),
        "rc_tnd": allocate_f(),
        "rr_tnd": allocate_f(),
        "ri_tnd": allocate_f(),
        "rs_tnd": allocate_f(),
        "rg_tnd": allocate_f(),
    }

    return {
        **time_state,
        **masks,
        **state,
        # TODO set following as temporaries
        **slopes,
        **increments,
        **transformations,
        **diags,
        **tnd,
        **tnd_update,
    }


def get_constant_state_ice4_tendencies(
    computational_grid: ComputationalGrid, *, gt4py_config: GT4PyConfig
) -> DataArrayDict:
    """Create state dictionnary with allocation of tables and setup to a constant value.

    Args:
        computational_grid (ComputationalGrid): grid indexes
        gt4py_config (GT4PyConfig): configuration for gt4py

    Returns:
        DataArrayDict: initialized dictionnary of state
    """
    state = allocate_state_ice4_tendencies(
        computational_grid, gt4py_config=gt4py_config
    )
    initialize_state_with_constant(state, 0.5, gt4py_config, list(state.keys()))
    return state
