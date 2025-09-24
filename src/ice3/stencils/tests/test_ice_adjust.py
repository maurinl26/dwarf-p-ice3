from gt4py.cartesian.gtscript import stencil


def test_ice_adjust():

    from ice3.stencils.ice_adjust import ice_adjust

    ice_adjust_gt4py = stencil(ice_adjust, backend="gt:cpu_ifirst")


    pass
