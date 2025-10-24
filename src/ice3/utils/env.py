from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
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
CPU_BACKENDS = "gt:cpu_kfirst"
GPU_BACKENDS = "gt:gpu"

try:
    BACKEND = os.environ["GT_BACKEND"]
    logging.info(f"Backend {BACKEND}")
except KeyError:
    logging.warning("Backend not found")
    BACKEND = "gt:cpu_ifirst"





