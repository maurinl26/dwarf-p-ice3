from datetime import timedelta
import pytest
from ifs_physics_common.framework.grid import I,J,K
from typing import Tuple, Literal

import gt4py
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from collections.abc import Hashable
from typing import Optional
from ifs_physics_common.utils.typingx import NDArrayLike
from ifs_physics_common.framework.storage import zeros

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


def test_dummy_component(computational_grid, gt4py_config):
    
    from ice3_gt4py.components.dummy_component import DummyComponent

    
    dummy_component = DummyComponent(
        computational_grid=computational_grid,
        gt4py_config=gt4py_config,
        enable_checks=False
    )
    
    grid_shape = computational_grid.grids[(I, J, K)].shape
        
    # Buffer
    state = {
        "a": ones(
            computational_grid=computational_grid,
            grid_id=(I, J, K),
            data_shape=grid_shape,
            gt4py_config=gt4py_config,
            dtype="float"
            ),
        "b": ones(
            computational_grid=computational_grid,
            grid_id=(I, J, K),
            data_shape=grid_shape,
            gt4py_config=gt4py_config,
            dtype="float"
            ),
    }
    
    dt = timedelta(seconds=50.0)
    
    out_diags = {
        "c": zeros(computational_grid=computational_grid,
            grid_id=(I, J, K),
            data_shape=grid_shape,
            gt4py_config=gt4py_config,
            dtype="float")
    }
    
    out_diags = dummy_component(
        state=state,
        timestep=dt,
    )
    
    assert out_diags["c"].mean() == 1.0