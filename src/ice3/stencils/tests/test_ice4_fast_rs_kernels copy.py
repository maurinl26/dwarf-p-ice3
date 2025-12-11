# -*- coding: utf-8 -*-
"""
Test suite for ICE4 fast rain-snow interaction kernels.

This module tests the bilinear interpolation of microphysical lookup tables
used in the ICE4 scheme for computing rain-snow accretion interactions.
The kernels perform 2D interpolation on pre-computed tables (KER_RACCSS,
KER_RACCS, KER_SACCRG) to determine collection efficiencies between
hydrometeor species in accretion processes.
"""

from gt4py.cartesian.gtscript import Field, GlobalTable, stencil
from gt4py.storage import from_array, zeros


from ice3.functions.interp_micro import index_micro2d_acc_r, index_micro2d_acc_s

import pytest
import numpy as np

from ice3.utils.env import dp_dtypes, sp_dtypes


def stencil_kernel_ice4_fast_rs(
    ldsoft: "bool",
    gacc_tmp: "bool",
    lbdar: Field["float"],
    lbdas: Field["float"],
    zw1_tmp: Field["float"],
    zw2_tmp: Field["float"],
    zw3_tmp: Field["float"],
    ker_raccss: GlobalTable["float", (40, 40)],
    ker_raccs: GlobalTable["float", (40, 40)],
    ker_saccrg: GlobalTable["float", (40, 40)],
):
    """
    Stencil for rain-snow accretion kernel interpolation.

    This stencil performs bilinear interpolation on three lookup tables
    (KER_RACCSS, KER_RACCS, KER_SACCRG) to compute collection efficiencies
    for different rain-snow interaction processes based on their slope
    parameters (lambda).

    Parameters
    ----------
    ldsoft : bool
        Flag for soft hail parameterization (when False, uses standard graupel).
    gacc_tmp : bool
        Flag indicating if accretion conditions are met.
    lbdar : Field[float]
        Slope parameter field for rain particle size distribution.
    lbdas : Field[float]
        Slope parameter field for snow particle size distribution.
    zw1_tmp : Field[float]
        Output field for interpolated RACCSS kernel values
        (rain accretion on snow to produce snow).
    zw2_tmp : Field[float]
        Output field for interpolated RACCS kernel values
        (rain accretion on snow).
    zw3_tmp : Field[float]
        Output field for interpolated SACCRG kernel values
        (snow accretion on rain to produce graupel).
    ker_raccss : GlobalTable[float, (40, 40)]
        Pre-computed kernel table for rain accretion on snow (dry).
    ker_raccs : GlobalTable[float, (40, 40)]
        Pre-computed kernel table for rain accretion on snow.
    ker_saccrg : GlobalTable[float, (40, 40)]
        Pre-computed kernel table for snow accretion on rain.

    Notes
    -----
    - The bilinear interpolation uses floor and fractional indices from
      index_micro2d_acc_r and index_micro2d_acc_s functions to interpolate
      between the four nearest table values for each kernel.
    - LDPACK conditional logic (lines 159-223) has been removed in translation.
    - Only processes with gacc_tmp=True and ldsoft=False are computed.
    - Tendencies rs_rraccs_tnd, rs_rraccss_tnd, and rs_rsaccrg_tnd are
      initialized to zero but not computed in this kernel excerpt.
    """
    with computation(PARALLEL), interval(...):
        # Translation note : LDPACK is False l159 to l223 removed in interp_micro.func.h
        #                                    l226 to l266 kept

        if (not ldsoft) and gacc_tmp:
            rs_rraccs_tnd = 0
            rs_rraccss_tnd = 0
            rs_rsaccrg_tnd = 0

            index_floor_r, index_float_r = index_micro2d_acc_r(lbdar)
            index_floor_s, index_float_s = index_micro2d_acc_s(lbdas)
            zw1_tmp = index_float_s * (
                index_float_r * ker_raccss.A[index_floor_s + 1, index_floor_r + 1]
                + (1 - index_float_r) * ker_raccss.A[index_floor_s + 1, index_floor_r]
            ) + (1 - index_float_s) * (
                index_float_r * ker_raccss.A[index_floor_s, index_floor_r + 1]
                + (1 - index_float_r) * ker_raccss.A[index_floor_s, index_floor_r]
            )

            zw2_tmp = index_float_s * (
                index_float_r * ker_raccs.A[index_floor_s + 1, index_floor_r + 1]
                + (1 - index_float_r) * ker_raccs.A[index_floor_s + 1, index_floor_r]
            ) + (1 - index_float_s) * (
                index_float_r * ker_raccs.A[index_floor_s, index_floor_r + 1]
                + (1 - index_float_r) * ker_raccs.A[index_floor_s, index_floor_r]
            )

            zw3_tmp = index_float_s * (
                index_float_r * ker_saccrg.A[index_floor_s + 1, index_floor_r + 1]
                + (1 - index_float_r) * ker_saccrg.A[index_floor_s + 1, index_floor_r]
            ) + (1 - index_float_s) * (
                index_float_r * ker_saccrg.A[index_floor_s, index_floor_r + 1]
                + (1 - index_float_r) * ker_saccrg.A[index_floor_s, index_floor_r]
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
def test_kernels_ice4_fast_rs(dtypes, backend, domain, origin, externals):
    """
    Test rain-snow accretion kernel interpolation.

    This test validates the bilinear interpolation of three lookup tables
    (KER_RACCSS, KER_RACCS, KER_SACCRG) across different computational
    backends and data types. It verifies that the stencil can successfully
    access and interpolate the tables for computing rain-snow collection
    efficiencies in various accretion processes.

    Parameters
    ----------
    dtypes : dict
        Dictionary containing data type specifications (dp_dtypes or sp_dtypes).
    backend : str
        The computational backend to use (debug, numpy, gt:cpu_ifirst, gt:gpu).
    domain : tuple
        Domain size for the fields, provided by pytest fixture.
    origin : tuple
        Origin coordinates for the fields, provided by pytest fixture.
    externals : dict
        Dictionary of external compile-time parameters for gt4py stencils.

    Notes
    -----
    The test uses:
    - Unit slope parameters (lbdar=1.0, lbdas=1.0)
    - Standard conditions (ldsoft=False, gacc_tmp=True)
    - Three output fields (zw1_tmp, zw2_tmp, zw3_tmp) initialized to zero

    The test verifies basic interpolation functionality without checking
    specific numerical values, focusing on successful execution across
    different backends.
    """
    from ice3.phyex_common.xker_raccs import KER_RACCS, KER_RACCSS, KER_SACCRG

    kernel = stencil(
        definition=stencil_kernel_ice4_fast_rs,
        name="kernel",
        backend=backend,
        dtypes=dtypes,
        externals=externals,
    )

    ldsoft = dtypes["bool"](False)
    gacc_tmp = dtypes["bool"](True)

    lbdar = np.ones(domain, dtype=dtypes["float"])
    lbdas = np.ones(domain, dtype=dtypes["float"])

    lbdar_gt4py = from_array(
        lbdar, dtype=dtypes["float"], backend=backend, aligned_index=origin
    )
    lbdas_gt4py = from_array(
        lbdas, dtype=dtypes["float"], backend=backend, aligned_index=origin
    )

    ker_saccrg = from_array(
        KER_SACCRG, dtype=dtypes["float"], backend=backend, aligned_index=origin
    )
    ker_raccs = from_array(
        KER_RACCS, dtype=dtypes["float"], backend=backend, aligned_index=origin
    )
    ker_raccss = from_array(
        KER_RACCSS, dtype=dtypes["float"], backend=backend, aligned_index=origin
    )

    zw1_tmp_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    zw2_tmp_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    zw3_tmp_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)

    kernel(
        ldsoft=ldsoft,
        gacc_tmp=gacc_tmp,
        lbdar=lbdar_gt4py,
        lbdas=lbdas_gt4py,
        zw1_tmp=zw1_tmp_gt4py,
        zw2_tmp=zw2_tmp_gt4py,
        zw3_tmp=zw3_tmp_gt4py,
        ker_raccs=ker_raccs,
        ker_raccss=ker_raccss,
        ker_saccrg=ker_saccrg,
        domain=domain,
        origin=origin,
    )
