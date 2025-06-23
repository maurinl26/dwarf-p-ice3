from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from pathlib import Path
import fmodpy 
import logging
import pytest

from ifs_physics_common.framework.config import GT4PyConfig
from ice3_gt4py.phyex_common.phyex import Phyex


def get_backends(gpu: bool = False):
    backends = ["numpy", "gt:cpu_ifirst", "gt:cpu_kfirst", "dace:cpu"]
    if gpu:
        backends += ["gt:gpu","dace:gpu"]
    return backends
    

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
    root_directory = file_path.parent.parent
    stencils_directory = Path(root_directory, "src", "ice3_gt4py", "stencils_fortran")
    script_path = Path(stencils_directory, fortran_script)

    logging.info(f"Fortran script path {script_path}")
    fortran_script = fmodpy.fimport(script_path)
    mode = getattr(fortran_script, fortran_module)
    return getattr(mode, fortran_stencil)

# fixtures
@pytest.fixture(name="domain", scope="module")
def domain_fixture():
    return (50, 50, 15)

@pytest.fixture(name="computational_grid", scope="module")
def computational_grid_fixture(domain):
    return ComputationalGrid(*domain)

@pytest.fixture(name="grid", scope="module")
def grid_fixture(computational_grid):
    return computational_grid.grids[(I, J, K)]

@pytest.fixture(name="origin", scope="module")
def origin_fixture():
    return (0, 0, 0) 

@pytest.fixture(name="gt4py_config", scope="module")
def gt4py_config_fixture():
    return GT4PyConfig(
        backend="numpy"
    )


@pytest.fixture(name="externals", scope="module")
def externals_fixture():
    return Phyex("AROME").to_externals()


@pytest.fixture(name="fortran_dims", scope="module")
def fortran_dims_fixture(grid):
    return {
        "nkt": grid.shape[2],
        "nijt": grid.shape[0] * grid.shape[1],
        "nktb": 1,
        "nkte": grid.shape[2],
        "nijb": 1,
        "nije": grid.shape[0] * grid.shape[1],
    }
    
@pytest.fixture(name="fortran_packed_dims", scope="module")
def packed_dims_fixture(grid):
    return {
        "kproma": grid.shape[0] * grid.shape[1] * grid.shape[2],
        "ksize": grid.shape[0] * grid.shape[1] * grid.shape[2]
    }