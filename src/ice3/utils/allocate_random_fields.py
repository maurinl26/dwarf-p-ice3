import numpy as np

from gt4py.storage import from_array
from ctypes import c_float, c_double
from typing import List, Dict, Tuple
import gt4py


def allocate_random_fields(names, dtypes, backend, domain):
    dtype = (c_float if dtypes["float"] == np.float32 else c_double)
    fields = {name: np.array(np.random.rand(*domain), dtype=dtype["float"], order="F") for name in names}
    gt4py_buffers = {name: from_array(fields[name], dtype=dtypes["float"], backend=backend) for name in names}
    return fields, gt4py_buffers


def draw_fields(names: List[str], dtypes: Dict[str, type], domain: Tuple[int]):
    return {
        name: np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in names
    }

def allocate_gt4py_fields(names: List[str], domain: Tuple[int], dtypes: Dict[str, type], backend: str):
    return {
        name: gt4py.storage.zeros(
            shape=domain,
            dtype=dtypes["float"],
            backend=backend
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
