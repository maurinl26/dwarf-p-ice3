from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
import os
import logging


try:
    BACKEND = os.environ("GT_BACKEND")
    logging.info(f"Backend {BACKEND}")
except KeyError:
    logging.warning("Backend not found")
    BACKEND = "gt:cpu_ifirst"



REBUILD = True
VALIDATE_ARGS = True
SHAPE = (50, 50, 15)

DEFAULT_GT4PY_CONFIG = GT4PyConfig(
            backend=BACKEND, 
            rebuild=REBUILD, 
            validate_args=VALIDATE_ARGS, 
            verbose=True
        )

TEST_GRID = ComputationalGrid(SHAPE[0], SHAPE[1], SHAPE[2])

