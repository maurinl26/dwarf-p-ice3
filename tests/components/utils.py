from ifs_physics_common.utils.typingx import NDArrayLike
from ifs_physics_common.framework.storage import zeros
from typing import Tuple, Literal

import gt4py
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from collections.abc import Hashable
from typing import Optional

def ones(
    computational_grid: ComputationalGrid,
    grid_id: Hashable,
    data_shape: Optional[Tuple[int, ...]] = None,
    *,
    gt4py_config: GT4PyConfig,
    dtype: Literal["bool", "float", "int"],
) -> NDArrayLike:
    """
    Create an array defined over the grid ``grid_id`` of ``computational_grid``
    and fill it with zeros.

    Relying on GT4Py utilities to optimally allocate memory based on the chosen backend.
    """
    grid = computational_grid.grids[grid_id]
    data_shape = data_shape or ()
    shape = grid.storage_shape + data_shape
    dtype = gt4py_config.dtypes.dict()[dtype]
    return gt4py.storage.ones(shape, dtype, backend=gt4py_config.backend)