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
    """Allocate random fields for testing purposes with both NumPy and GT4Py storage.
    
    Creates random field data in Fortran order (column-major) and wraps them in GT4Py
    storage objects for the specified backend. This is useful for initializing test
    data that needs to be compatible with both NumPy operations and GT4Py stencils.

    Args:
        names (List[str]): List of field names to create
        dtypes (Dict[str, type]): Dictionary mapping type names to numpy dtypes,
            must contain "float" key
        backend (str): GT4Py backend name (e.g., "gt:cpu_ifirst", "gt:gpu")
        domain (Tuple[int, ...]): Shape of the fields to create (e.g., (ni, nj, nk))

    Returns:
        Tuple[Dict[str, NDArray], Dict[str, NDArray]]: A tuple containing:
            - Dictionary of NumPy arrays with random data
            - Dictionary of GT4Py storage objects wrapping the NumPy arrays
    """
    dtype = (c_float if dtypes["float"] == np.float32 else c_double)
    fields = {name: np.array(np.random.rand(*domain), dtype=dtype["float"], order="F") for name in names}
    gt4py_buffers = {name: from_array(fields[name], dtype=dtypes["float"], backend=backend) for name in names}
    return fields, gt4py_buffers


def draw_fields(names: List[str], dtypes: Dict[str, type], domain: Tuple[int, ...]) -> Dict[str, NDArray]:
    """Generate random NumPy arrays for multiple fields.
    
    Creates a dictionary of random field data in Fortran order (column-major) with
    values between 0 and 1. This is a simpler alternative to allocate_random_fields
    when GT4Py storage is not needed.

    Args:
        names (List[str]): List of field names to create
        dtypes (Dict[str, type]): Dictionary mapping type names to numpy dtypes,
            must contain "float" key
        domain (Tuple[int, ...]): Shape of the fields to create (e.g., (ni, nj, nk))

    Returns:
        Dict[str, NDArray]: Dictionary mapping field names to NumPy arrays with random data
    """
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
    """Allocate zero-initialized GT4Py storage fields.
    
    Creates GT4Py storage objects initialized to zero for the specified backend.
    This is useful for allocating output fields or intermediate storage that will
    be populated by stencil computations.

    Args:
        names (List[str]): List of field names to create
        domain (Tuple[int, ...]): Shape of the fields to create (e.g., (ni, nj, nk))
        dtypes (Dict[str, type]): Dictionary mapping type names to numpy dtypes,
            must contain "float" key
        backend (str): GT4Py backend name (e.g., "gt:cpu_ifirst", "gt:gpu")

    Returns:
        Dict[str, NDArray]: Dictionary mapping field names to zero-initialized GT4Py storage objects
    """
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
    """Prepare field data for Fortran interoperability via f2py.
    
    Converts GT4Py or NumPy arrays to flattened (1D) arrays suitable for passing to
    Fortran subroutines through f2py. The mapping allows for renaming fields to match
    Fortran naming conventions (e.g., applying doctor norm prefixes).

    Args:
        f2py_names (Dict[str, str]): Mapping from Fortran field names to Python field names,
            e.g., {"prc_t": "rc_t", "pri_t": "ri_t"}
        buffer (Dict[str, np.ndarray]): Dictionary of NumPy or GT4Py arrays to be passed
            to Fortran

    Returns:
        Dict[str, NDArray]: Dictionary mapping Fortran field names to flattened 1D arrays
            ready for f2py function calls
    """
    return {
        fname: buffer[pyname].ravel()
        for fname, pyname in f2py_names.items()
    }
