# -*- coding: utf-8 -*-
from ifs_physics_common.framework.storage import allocate_data_array
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
from ifs_physics_common.utils.typingx import (
    DataArray,
)

from gt4py.storage import from_array
from typing import Literal, Tuple
import numpy as np

