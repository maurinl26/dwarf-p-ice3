import os
import logging
import numpy as np
from pathlib import Path

ROOT_PATH = Path(__file__).parents[2]

sp_dtypes = {
    "float": np.float32,
    "int": np.int32,
    "bool": np.bool_
}

dp_dtypes = {
    "float": np.float64,
    "int": np.int64,
    "bool": np.bool_
}

BACKEND_LIST = ["numpy", "gt:cpu_ifirst", "gt:gpu", "dace:cpu", "dace:gpu"]

DEBUG_BACKEND = "numpy"
CPU_BACKEND = "dace:cpu"
GPU_BACKEND = "dace:gpu"


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

compile_stencil = partial(stencil, backend=BACKEND, dtypes=DTYPES)

