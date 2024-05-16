# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Dict

from gt4py.storage import ones


if TYPE_CHECKING:
    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.utils.typingx import (
        DataArrayDict,
    )


################################## Utils ##########################################


def initialize_state_with_constant(
    state: DataArrayDict, C: float, gt4py_config: GT4PyConfig, keys: Dict[list]
) -> None:
    """Initialize fields of state dictionnary with a constant field.

    Args:
        state (DataArrayDict): dictionnary of state
        C (float): constant value for initialization
        gt4py_config (GT4PyConfig): configuration of gt4py
    """

    for name in keys:
        state[name][...] = C * ones(state[name].shape, backend=gt4py_config.backend)
