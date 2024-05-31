# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np

from ice3_gt4py.initialisation.state import initialize_state_with_constant
from ice3_gt4py.initialisation.state_ice_adjust import allocate_state_ice_adjust

if TYPE_CHECKING:

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid
    from ifs_physics_common.utils.typingx import (
        DataArrayDict,
    )


def get_constant_state_ice_adjust(
    computational_grid: ComputationalGrid,
    *,
    gt4py_config: GT4PyConfig,
    keys: Dict[list],
) -> DataArrayDict:
    """Create state dictionnary with allocation of tables and setup to a constant value.

    Args:
        computational_grid (ComputationalGrid): grid indexes
        gt4py_config (GT4PyConfig): configuration for gt4py

    Returns:
        DataArrayDict: initialized dictionnary of state
    """
    state = allocate_state_ice_adjust(computational_grid, gt4py_config=gt4py_config)
    initialize_state_with_constant(state, 0.5, gt4py_config, keys)
    return state
