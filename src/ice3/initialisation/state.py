# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
from functools import partial
from typing import TYPE_CHECKING, Dict, Literal, Tuple

import numpy as np
from gt4py.storage import ones
from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
from ifs_physics_common.framework.storage import allocate_data_array
from ifs_physics_common.utils.typingx import DataArray, DataArrayDict, NDArrayLikeDict

from ice3_gt4py.initialisation.utils import initialize_field
from ice3_gt4py.utils.reader import NetCDFReader
import logging

if TYPE_CHECKING:
    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.utils.typingx import DataArrayDict


def allocate_state(
    computational_grid: ComputationalGrid,
    gt4py_config: GT4PyConfig,
    component: ImplicitTendencyComponent,
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
            field_name: partial(
                _allocate,
                grid_id=properties["grid"],
                units=properties["units"],
                dtype=properties["dtype"],
            )()
            for field_name, properties in component.input_properties.items()
        },
    }


def get_constant_state(
    computational_grid: ComputationalGrid,
    *,
    gt4py_config: GT4PyConfig,
    component: ImplicitTendencyComponent,
) -> DataArrayDict:
    """Create state dictionnary with allocation of tables and setup to a constant value.

    Args:
        computational_grid (ComputationalGrid): grid indexes
        gt4py_config (GT4PyConfig): configuration for gt4py

    Returns:
        DataArrayDict: initialized dictionnary of state
    """
    state = allocate_state(
        computational_grid, gt4py_config=gt4py_config, component=component
    )
    initialize_state_with_constant(state, 1, gt4py_config, component._input_properties)
    return state


def get_state(
    computational_grid: ComputationalGrid,
    *,
    gt4py_config: GT4PyConfig,
    netcdf_reader: NetCDFReader,
    component: ImplicitTendencyComponent,
) -> DataArrayDict:
    """Create a state with reproductibility data set.

    Args:
        computational_grid (ComputationalGrid): grid
        gt4py_config (GT4PyConfig): config for gt4py
        keys (Dict[keys]): field names

    Returns:
        DataArrayDict: dictionnary of data array containing reproductibility data
    """
    state = allocate_state(
        computational_grid, gt4py_config=gt4py_config, component=component
    )
    initialize_state(state, netcdf_reader, component)
    return state


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


def initialize_state(
    state: DataArrayDict,
    netcdf_reader: NetCDFReader,
    component: ImplicitTendencyComponent,
) -> None:
    """Initialize fields of state dictionnary with a constant field.

    Args:
        state (DataArrayDict): dictionnary of state
        gt4py_config (GT4PyConfig): configuration of gt4py
    """
    dims = netcdf_reader.get_dims()
    n_IJ, n_K = dims["IJ"], dims["K"]

    for key, attributes in component._input_properties.items():

        fortran_name = (
            attributes["fortran_name"] if "fortran_name" in attributes.keys() else None
        )
        irr = attributes["irr"] if "irr" in attributes.keys() else None
        grid = attributes["grid"]

        logging.info(f"name={key}, fortran_name={fortran_name}, dim={irr}")
        if fortran_name is not None:
            if fortran_name in ["ZRS", "PRS", "PRT"]:
                buffer = netcdf_reader.get_field(fortran_name)[:, :, irr]
            else:
                buffer = netcdf_reader.get_field(fortran_name)
        else:
            if len(grid) == 3:
                buffer = np.zeros((n_IJ, n_K))
            elif len(grid) == 2:
                buffer = np.zeros((n_IJ))
            else:
                raise ValueError("Data must be 2d or 3d")

        logging.info(f"name = {key}, buffer.shape = {buffer.shape}")
        initialize_field(state[key], buffer)
