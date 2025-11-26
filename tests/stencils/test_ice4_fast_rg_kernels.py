# -*- coding: utf-8 -*-
from gt4py.cartesian.gtscript import PARALLEL, GlobalTable, computation, Field, stencil
from gt4py.storage import from_array, zeros
import pytest
import numpy as np

from ice3.utils.env import dp_dtypes, sp_dtypes
from ice3.functions.interp_micro import index_micro2d_dry_g, index_micro2d_dry_s, index_micro2d_dry_r


def stencil_kernel1_ice4_fast_rg(
    ldsoft: "bool",
    gdry: "bool",
    lbdas: Field["float"],
    lbdag: Field["float"],
    ker_sdryg: GlobalTable["float", (40, 40)],
):

    with computation(PARALLEL), interval(...):
        if (not ldsoft) and gdry:
            index_floor_s, index_float_s = index_micro2d_dry_s(lbdas)
            index_floor_g, index_float_g = index_micro2d_dry_g(lbdag)
            zw_tmp = index_float_g * (
                index_float_s * ker_sdryg.A[index_floor_g + 1, index_floor_s + 1]
                + (1 - index_float_s) * ker_sdryg.A[index_floor_g + 1, index_floor_s]
            ) + (1 - index_float_g) * (
                index_float_s * ker_sdryg.A[index_floor_g, index_floor_s + 1]
                + (1 - index_float_s) * ker_sdryg.A[index_floor_g, index_floor_s]
            )


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
def test_kernel1_ice4_fast_rg(externals, dtypes, backend, domain, origin):

    from ice3.phyex_common.xker_sdryg import KER_SDRYG

    kernel1 = stencil(
        definition=stencil_kernel2_ice4_fast_rg,
        name="kernel1",
        backend=backend,
        dtypes=dtypes,
        externals=externals
    )

    ldsoft = dtypes["bool"](False)
    gdry = dtypes["bool"](True)

    lbdas = np.ones(domain, dtype=dtypes["float"])
    lbdag = np.ones(domain, dtype=dtypes["float"])

    lbdas_gt4py = from_array(lbdas, dtypes=dtypes["float"], backend=backend, aligned_index=origin)
    lbdag_gt4py = from_array(lbdag, dtypes=dtypes["float"], backend=backend, aligned_index=origin)

    ker_rdryg = from_array(KER_SDRYG, dtypes=dtypes["float"], backend=backend, aligned_index=origin)

    kernel1(
        ldsoft=ldsoft,
        gdry=gdry,
        lbdas=lbdas_gt4py,
        lbdag=lbdag_gt4py,
        ker_sdryg=ker_rdryg,
        domain=domain,
        origin=origin
    )




def stencil_kernel2_ice4_fast_rg(
    ldsoft: "bool",
    lbdag: Field["float"],
    lbdar: Field["float"],
    ker_rdryg: GlobalTable["float", (40, 40)]
):

    with computation(PARALLEL), interval(...):
        if not ldsoft:
            index_floor_g, index_float_g = index_micro2d_dry_g(lbdag)
            index_floor_r, index_float_r = index_micro2d_dry_r(lbdar)
            zw_tmp = index_float_r * (
                index_float_g * ker_rdryg.A[index_floor_r + 1, index_floor_g + 1]
                + (1 - index_float_g) * ker_rdryg.A[index_floor_r + 1, index_floor_g]
            ) + (1 - index_float_r) * (
                index_float_g * ker_rdryg.A[index_floor_r, index_floor_g + 1]
                + (1 - index_float_g) * ker_rdryg.A[index_floor_r, index_floor_g]
            )
    

    
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
def test_kernel2_ice4_fast_rg(externals, dtypes, backend, domain, origin):

    from ice3.phyex_common.xker_sdryg import KER_SDRYG
    
    kernel2 = stencil(
        definition=stencil_kernel1_ice4_fast_rg,
        name="kernel1",
        backend=backend,
        dtypes=dtypes,
        externals=externals
    )

    ldsoft = dtypes["bool"](False)
    gdry = dtypes["bool"](True)

    lbdas = np.ones(domain, dtype=dtypes["float"])
    lbdag = np.ones(domain, dtype=dtypes["float"])

    lbdas_gt4py = from_array(lbdas, dtypes=dtypes["float"], backend=backend, aligned_index=origin)
    lbdag_gt4py = from_array(lbdag, dtypes=dtypes["float"], backend=backend, aligned_index=origin)

    ker_rdryg = from_array(KER_SDRYG, dtypes=dtypes["float"], backend=backend, aligned_index=origin)

    kernel2(
        ldsoft=ldsoft,
        gdry=gdry,
        lbdas=lbdas_gt4py,
        lbdag=lbdag_gt4py,
        ker_sdryg=ker_rdryg,
        domain=domain,
        origin=origin
    )




 