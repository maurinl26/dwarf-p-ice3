import pytest
import numpy as np
from gt4py.cartesian.gtscript import stencil
from gt4py.next.program_processors.runners.double_roundtrip import backend


def test_ice_adjust(externals):

    from ice3.stencils.ice_adjust import ice_adjust
    
    dtypes = {
        float: np.float64,
        int: np.int32
    }

    ice_adjust_gt4py = stencil(
        backend,
        ice_adjust,
        name="ice_adjust",
        dtypes=dtypes,
        externals=externals
    )



    ice_adjust_gt4py()
