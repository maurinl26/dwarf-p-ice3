# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
from functools import partial
from typing import TYPE_CHECKING, Dict, List

from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array
from ifs_physics_common.framework.components import ImplicitTendencyComponent


from ice3_gt4py.initialisation.state import initialize_state_with_constant

if TYPE_CHECKING:
    from typing import Literal, Tuple

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
    from ifs_physics_common.utils.typingx import (
        DataArray,
        DataArrayDict,
        NDArrayLikeDict,
    )

############################## AroAdjust #################################
def allocate_state(
    computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig, component: ImplicitTendencyComponent
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
    
    return {
        "time": datetime.datetime(year=2024, month=1, day=1),
        **{
            
            field_name: partial(_allocate, grid_id=properties["grid"], units=properties["units"], dtype=properties["dtype"])
            for field_name, properties in component.input_properties.items()
        }
    }


def get_constant_state_aro_adjust(
    computational_grid: ComputationalGrid,*, gt4py_config: GT4PyConfig, keys: List[str], component: ImplicitTendencyComponent
) -> DataArrayDict:
    """Create state dictionnary with allocation of tables and setup to a constant value.

    Args:
        computational_grid (ComputationalGrid): grid indexes
        gt4py_config (GT4PyConfig): configuration for gt4py

    Returns:
        DataArrayDict: initialized dictionnary of state
    """
    state = allocate_state(computational_grid, gt4py_config=gt4py_config, component=component)
    initialize_state_with_constant(state, 0.5, gt4py_config, keys)
    return state
