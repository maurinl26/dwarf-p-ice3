"""Environment configuration for GT4Py backend and precision settings.

This module configures the GT4Py backend and numerical precision for the ice3 microphysics
package. It reads environment variables to determine:
- Which GT4Py backend to use (CPU, GPU, etc.)
- The numerical precision (single or double precision)

The configuration is used throughout the package to ensure consistent stencil compilation
and data type usage.

Environment Variables:
    GT_BACKEND: GT4Py backend name (default: "gt:cpu_ifirst")
        Examples: "gt:cpu_ifirst", "gt:cpu_kfirst", "gt:gpu", "numpy", etc.
    PRECISION: Numerical precision (default: "single")
        Options: "single" (float32), "double" (float64)

Module Constants:
    ROOT_PATH: Project root directory path
    BACKEND: Configured GT4Py backend
    DTYPES: Dictionary mapping type names to numpy dtypes
    compile_stencil: Pre-configured stencil decorator with backend and dtypes
"""

from gt4py.cartesian.stencil_object import StencilObject


from typing import Callable


import os
import logging
import numpy as np
from pathlib import Path

# Project root directory (2 levels up from this file)
ROOT_PATH = Path(__file__).parents[2]

# Single precision data types
sp_dtypes = {
    "float": np.float32,
    "int": np.int32,
    "bool": np.bool_
}

# Double precision data types
dp_dtypes = {
    "float": np.float64,
    "int": np.int64,
    "bool": np.bool_
}


############# Set BACKEND ##############
try:
    BACKEND = os.environ["GT_BACKEND"]
    logging.info(f"Backend {BACKEND}")
except KeyError:
    logging.warning("Backend not found")
    BACKEND = "gt:cpu_ifirst"


############ Set DTYPES ###############
try:
    PRECISION = os.environ["PRECISION"]
    match PRECISION:
        case "single":
            DTYPES = sp_dtypes
        case "double":
            DTYPES = dp_dtypes
        case _:
            DTYPES = sp_dtypes
except KeyError:
    DTYPES = sp_dtypes


######## Consistent stencil compilation ##########
from functools import partial
from gt4py.cartesian.gtscript import stencil

# Pre-configured stencil decorator with backend and dtypes set
# Use this instead of the raw @stencil decorator to ensure consistency
compile_stencil: Callable[..., StencilObject] = partial(stencil, backend=BACKEND, dtypes=DTYPES)
