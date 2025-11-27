# -*- coding: utf-8 -*-
from gt4py.cartesian.gtscript import computation, stencil, Field
from gt4py.storage import zeros, from_array
import pytest
from ice3.functions.interp_micro import index_micro2d_dry_s
from ice3.utils.env import dp_dtypes, sp_dtypes
import numpy as np

def stencil_index_micro2d_dry_s(
    lbdas: Field["float"],
    weight_s: Field["float"],
    index_s: Field["int"]
):

    with computation(PARALLEL), interval(...):
        index_s, weight_s = index_micro2d_dry_s(lbdas[0,0,0])

@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("debug", marks=pytest.mark.debug),
        pytest.param("numpy", marks=pytest.mark.numpy),
        pytest.param("gt:cpu_ifirst", marks=pytest.mark.cpu),
        pytest.param("gt:gpu", marks=pytest.mark.gpu),
    ],
)
def test_index_micro_dry_s(
    backend,
    dtypes,
    externals,
    domain,
    origin
):

    stencil_dry_s = stencil(
        definition=stencil_index_micro2d_dry_s,
        name="dry_s",
        backend=backend,
        dtypes=dtypes,
        externals=externals
    )


    lbdas = np.ones(domain, dtype=dtypes["float"])

    lbdas_gt4py = from_array(lbdas, backend=backend, dtype=dtypes["float"], aligned_index=origin)
    index_s_gt4py = zeros(domain, backend=backend, dtype=dtypes["int"])
    weight_s_gt4py = zeros(domain, backend=backend, dtype=dtypes["float"])


    stencil_dry_s(
        lbdas=lbdas_gt4py,
        index_s=index_s_gt4py,
        weight_s=weight_s_gt4py,
        domain=domain,
        origin=origin,
    )

    assert index_s_gt4py.dtype == dtypes["int"]