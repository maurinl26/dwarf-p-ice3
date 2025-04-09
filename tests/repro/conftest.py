from ifs_physics_common.framework.config import GT4PyConfig
from pathlib import Path
import fmodpy 
import logging
import pytest
import numpy as np

from ifs_physics_common.framework.config import GT4PyConfig, DataTypes
from ice3_gt4py.phyex_common.phyex import Phyex

# TODO : rework as fixtures
BACKEND = "gt:cpu_ifirst"
REBUILD = True
VALIDATE_ARGS = True
SHAPE = (50, 50, 15)
NX, NY, NZ = SHAPE

DEFAULT_GT4PY_CONFIG = GT4PyConfig(
            backend=BACKEND, 
            rebuild=REBUILD, 
            validate_args=VALIDATE_ARGS, 
            verbose=True
        )

def compile_fortran_stencil(
    fortran_script: str, fortran_module: str, fortran_stencil: str
):
    """Compile fortran stencil wrapped in a fortran file + module with fmodpy

    Args:
        fortran_script (str): _description_
        fortran_module (str): _description_
        fortran_stencil (str): _description_

    Returns:
        _type_: _description_
    """
    #### Fortran subroutine
    file_path = Path(__file__)
    root_directory = file_path.parent.parent.parent
    stencils_directory = Path(root_directory, "src", "ice3_gt4py", "stencils_fortran")
    script_path = Path(stencils_directory, fortran_script)

    logging.info(f"Fortran script path {script_path}")
    fortran_script = fmodpy.fimport(script_path)
    mode = getattr(fortran_script, fortran_module)
    return getattr(mode, fortran_stencil)

# fixtures
@pytest.fixture(name="gt4py_config", scope="module")
def gt4py_config_fixture():
    return GT4PyConfig(
        backend=BACKEND,
        rebuild=REBUILD,
        validate_args=VALIDATE_ARGS,
        verbose=True,
        dtypes=DataTypes(bool=bool, float=np.float32, int=np.int32),
    )


@pytest.fixture(name="externals", scope="module")
def externals_fixture():
    return Phyex("AROME").to_externals()


@pytest.fixture(name="fortran_dims", scope="module")
def fortran_dims_fixture():
    return {
        "nkt": NZ,
        "nijt": NX * NY,
        "nktb": 1,
        "nkte": NZ,
        "nijb": 1,
        "nije": NX * NY,
    }