import numpy as np
from typing import List, Dict, Tuple
from numpy.typing import NDArray

from gt4py.storage import from_array, zeros
from ctypes import c_float, c_double



def allocate_random_fields(
        names: List[str],
        dtypes: Dict[str, type],
        backend: str,
        domain: Tuple[int, ...]) -> Tuple[NDArray, NDArray]:
    dtype = (c_float if dtypes["float"] == np.float32 else c_double)
    fields = {name: np.array(np.random.rand(*domain), dtype=dtype["float"], order="F") for name in names}
    gt4py_buffers = {name: from_array(fields[name], dtype=dtypes["float"], backend=backend) for name in names}
    return fields, gt4py_buffers


def draw_fields(names: List[str], dtypes: Dict[str, type], domain: Tuple[int, ...]) -> Dict[str, NDArray]:
    return {
        name: np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in names
    }

def allocate_gt4py_fields(
        names: List[str],
        domain: Tuple[int, ...],
        dtypes: Dict[str, type],
        backend: str) -> Dict[str, NDArray]:
    return {
        name: zeros(
            shape=domain,
            dtype=dtypes["float"],
            backend=backend
        )
        for name in names
    }

def allocate_fortran_fields(
        f2py_names: Dict[str, str],
        buffer: Dict[str, np.ndarray]
) -> Dict[str, NDArray]:
    return {
        fname: buffer[pyname].ravel()
        for fname, pyname in f2py_names.items()
    }
