import os
import logging
import numpy as np

BACKEND_LIST = ["numpy", "gt:cpu_ifirst", "gt:gpu", "dace:cpu", "dace:gpu"]

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
DEBUG_BACKEND = "numpy"
CPU_BACKEND = "dace:cpu"
GPU_BACKEND = "dace:gpu"

try:
    BACKEND = os.environ["GT_BACKEND"]
    logging.info(f"Backend {BACKEND}")
except KeyError:
    logging.warning("Backend not found")
    BACKEND = "gt:cpu_ifirst"


