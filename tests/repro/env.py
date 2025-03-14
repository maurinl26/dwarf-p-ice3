from ifs_physics_common.framework.config import GT4PyConfig
from pathlib import Path

BACKEND = "gt:cpu_kfirst"
REBUILD = True
VALIDATE_ARGS = True
SHAPE = (50, 50, 15)

DEFAULT_GT4PY_CONFIG = GT4PyConfig(
            backend=BACKEND, 
            rebuild=REBUILD, 
            validate_args=VALIDATE_ARGS, 
            verbose=True
        )

