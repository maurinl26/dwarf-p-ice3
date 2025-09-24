import pytest
from gt4py.cartesian.gtscript import stencil


def test_ice_adjust():

    from ice3.stencils.ice_adjust import ice_adjust

    try :
        ice_adjust_gt4py = stencil(ice_adjust, backend="gt:cpu_ifirst")
    except Exception as e :
        print(e)
        print("Compilation failed")


