import numpy as np

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import Grid
from ctypes import c_float, c_double
from typing import List, Dict
import gt4py


def draw_fields(names: List[str], gt4py_config: GT4PyConfig, grid: Grid):
    return {
        name: np.array(
            np.random.rand(*grid.shape),
            dtype=(c_float if gt4py_config.dtypes.float == np.float32 else c_double),
            order="F",
        )
        for name in names
    }

def allocate_gt4py_fields(names: List[str], gt4py_config: GT4PyConfig, grid: Grid):
    return {
        name: gt4py.storage.zeros(
            shape=grid.shape,
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        for name in names
    }

def allocate_fields(fields, buffer):
    # Allocate
    for name in fields.keys():
        fields[name] = buffer[name]

    # return gt4py_fields

def allocate_fortran_fields(
        f2py_names: Dict[str, str],
        buffer: Dict[str, np.ndarray]
):
    return {
        fname: buffer[pyname].ravel()
        for fname, pyname in f2py_names.items()
    }
