import datetime

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid

from ice3_gt4py.phyex_common.phyex import Phyex

### Fortran dims
NIT = 50
NJT = 1
NKT = 15
NIJT = NIT * NJT

# Fortran indexing from 1 to end index
NKB, NKE = 1, NKT

# Fortran indexing from 1 to end index
NIJB, NIJE = 1, NIJT

backend = "gt:cpu_ifirst"
rebuild = True
validate_args = True

phyex = Phyex(program="AROME")

test_grid = ComputationalGrid(50, 1, 15)
dt = datetime.timedelta(seconds=1)

default_gt4py_config = GT4PyConfig(
    backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=False
)

#### Default error margin ####
default_epsilon = 10e-5
