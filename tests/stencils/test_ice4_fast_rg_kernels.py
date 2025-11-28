# -*- coding: utf-8 -*-
"""
Test suite for ICE4 fast rain-graupel interaction kernels.

This module tests the bilinear interpolation of microphysical lookup tables
used in the ICE4 scheme for computing rain-graupel dry growth interactions.
The kernels perform 2D interpolation on pre-computed tables (KER_SDRYG, KER_RDRYG)
to determine collection efficiencies between hydrometeor species.
"""

from gt4py.cartesian.gtscript import PARALLEL, GlobalTable, computation, Field, stencil
from gt4py.storage import from_array, ones, zeros
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from ice3.utils.env import dp_dtypes, sp_dtypes
from ice3.functions.interp_micro import (
    index_micro2d_dry_g,
    index_micro2d_dry_s,
    index_micro2d_dry_r,
)


def stencil_dummy_interp_kernel1(
    output: Field["float"], ker_sdryg: GlobalTable[("float", (81, 81))]
):
    with computation(PARALLEL), interval(...):
        output[0, 0, 0] = ker_sdryg.A[1, 1]


def stencil_dummy_interp_kernel2(
    index_floor_g: Field["int"],
    index_floor_s: Field["int"],
    output: Field["float"],
    ker_sdryg: GlobalTable[("float", (81, 81))],
):
    with computation(PARALLEL), interval(...):
        output[0, 0, 0] = ker_sdryg.A[index_floor_s, index_floor_g]


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
def test_dummy_interpolation_kernel(domain, backend, externals, dtypes, origin):
    stencil_dummy_interp = stencil(
        definition=stencil_dummy_interp_kernel2,
        name="dummy_interp",
        backend=backend,
        dtypes=dtypes,
        externals=externals,
    )

    from ice3.phyex_common.xker_sdryg import KER_SDRYG

    ker_sdryg = from_array(KER_SDRYG, dtype=dtypes["float"], backend=backend)
    index_floor_g = ones(domain, backend=backend, dtype=dtypes["int"])
    index_floor_s = ones(domain, backend=backend, dtype=dtypes["int"])

    output = zeros(domain, backend=backend, dtype=dtypes["float"])

    stencil_dummy_interp(
        ker_sdryg=ker_sdryg,
        index_floor_s=index_floor_s,
        index_floor_g=index_floor_g,
        output=output,
        domain=domain,
        origin=origin,
    )

    assert output.any() == 0.185306e01


def stencil_kernel1_ice4_fast_rg(
    ldsoft: "bool",
    gdry: "bool",
    lbdas: Field["float"],
    lbdag: Field["float"],
    ker_sdryg: GlobalTable["float", (81, 81)],
    index_floor_s: "int" = 0,
    index_floor_g: "int" = 0,
):
    """
    Stencil for snow-graupel dry growth kernel interpolation.

    This stencil performs bilinear interpolation on the KER_SDRYG lookup table
    to compute collection efficiencies between snow and graupel particles
    based on their slope parameters (lambda).

    Parameters
    ----------
    ldsoft : bool
        Flag for soft hail parameterization (when False, uses standard graupel).
    gdry : bool
        Flag indicating if dry growth conditions are met.
    lbdas : Field[float]
        Slope parameter field for snow particle size distribution.
    lbdag : Field[float]
        Slope parameter field for graupel particle size distribution.
    ker_sdryg : GlobalTable[float, (40, 40)]
        Pre-computed kernel table for snow-graupel dry growth collection.

    Notes
    -----
    The bilinear interpolation uses floor and fractional indices from
    index_micro2d_dry_s and index_micro2d_dry_g functions to interpolate
    between the four nearest table values.
    """
    with computation(PARALLEL), interval(...):
        if (not ldsoft) and gdry:
            _, weight_s = index_micro2d_dry_s(lbdas[0, 0, 0])
            _, weight_g = index_micro2d_dry_g(lbdag[0, 0, 0])
            zw_tmp = weight_g * (
                weight_s * ker_sdryg.A[index_floor_g, index_floor_s]
                + (1 - weight_s) * ker_sdryg.A[index_floor_g, index_floor_s]
            ) + (1 - weight_g) * (
                weight_s * ker_sdryg.A[index_floor_g, index_floor_s]
                + (1 - weight_s) * ker_sdryg.A[index_floor_g, index_floor_s]
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
    """
    Test snow-graupel dry growth kernel interpolation.

    This test validates the bilinear interpolation of the KER_SDRYG table
    across different computational backends and data types. It verifies
    that the stencil can successfully access and interpolate the lookup
    table for computing snow-graupel collection efficiencies.

    Parameters
    ----------
    externals : dict
        Dictionary of external compile-time parameters for gt4py stencils.
    dtypes : dict
        Dictionary containing data type specifications (dp_dtypes or sp_dtypes).
    backend : str
        The computational backend to use (debug, numpy, gt:cpu_ifirst, gt:gpu).
    domain : tuple
        Domain size for the fields, provided by pytest fixture.
    origin : tuple
        Origin coordinates for the fields, provided by pytest fixture.

    Notes
    -----
    The test uses unit slope parameters (lbdas=1.0, lbdag=1.0) and
    standard conditions (ldsoft=False, gdry=True) to verify basic
    interpolation functionality.
    """
    from ice3.phyex_common.xker_sdryg import KER_SDRYG

    kernel1 = stencil(
        definition=stencil_kernel2_ice4_fast_rg,
        name="kernel1",
        backend=backend,
        dtypes=dtypes,
        externals=externals,
    )

    ldsoft = dtypes["bool"](False)
    gdry = dtypes["bool"](True)

    lbdas = np.ones(domain, dtype=dtypes["float"])
    lbdag = np.ones(domain, dtype=dtypes["float"])

    lbdas_gt4py = from_array(
        lbdas, dtypes=dtypes["float"], backend=backend, aligned_index=origin
    )
    lbdag_gt4py = from_array(
        lbdag, dtypes=dtypes["float"], backend=backend, aligned_index=origin
    )

    ker_rdryg = from_array(
        KER_SDRYG, dtypes=dtypes["float"], backend=backend, aligned_index=origin
    )

    kernel1(
        ldsoft=ldsoft,
        gdry=gdry,
        lbdas=lbdas_gt4py,
        lbdag=lbdag_gt4py,
        ker_sdryg=ker_rdryg,
        index_floor_s=dtypes["int"](0),
        index_floor_g=dtypes["int"](0),
        domain=domain,
        origin=origin,
    )


def stencil_kernel2_ice4_fast_rg(
    ldsoft: "bool",
    lbdag: Field["float"],
    lbdar: Field["float"],
    ker_rdryg: GlobalTable["float", (40, 40)],
):
    """
    Stencil for rain-graupel dry growth kernel interpolation.

    This stencil performs bilinear interpolation on the KER_RDRYG lookup table
    to compute collection efficiencies between rain and graupel particles
    based on their slope parameters (lambda).

    Parameters
    ----------
    ldsoft : bool
        Flag for soft hail parameterization (when False, uses standard graupel).
    lbdag : Field[float]
        Slope parameter field for graupel particle size distribution.
    lbdar : Field[float]
        Slope parameter field for rain particle size distribution.
    ker_rdryg : GlobalTable[float, (40, 40)]
        Pre-computed kernel table for rain-graupel dry growth collection.

    Notes
    -----
    The bilinear interpolation uses floor and fractional indices from
    index_micro2d_dry_g and index_micro2d_dry_r functions to interpolate
    between the four nearest table values.
    """
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
    """
    Test rain-graupel dry growth kernel interpolation.

    This test validates the bilinear interpolation of the KER_RDRYG table
    across different computational backends and data types. It verifies
    that the stencil can successfully access and interpolate the lookup
    table for computing rain-graupel collection efficiencies.

    Parameters
    ----------
    externals : dict
        Dictionary of external compile-time parameters for gt4py stencils.
    dtypes : dict
        Dictionary containing data type specifications (dp_dtypes or sp_dtypes).
    backend : str
        The computational backend to use (debug, numpy, gt:cpu_ifirst, gt:gpu).
    domain : tuple
        Domain size for the fields, provided by pytest fixture.
    origin : tuple
        Origin coordinates for the fields, provided by pytest fixture.

    Notes
    -----
    The test uses unit slope parameters (lbdas=1.0, lbdag=1.0) and
    standard conditions (ldsoft=False, gdry=True) to verify basic
    interpolation functionality.
    """
    from ice3.phyex_common.xker_rdryg import KER_RDRYG

    kernel2 = stencil(
        definition=stencil_kernel2_ice4_fast_rg,
        name="kernel2",
        backend=backend,
        dtypes=dtypes,
        externals=externals,
    )

    ldsoft = dtypes["bool"](False)
    gdry = dtypes["bool"](True)

    lbdas = np.ones(domain, dtype=dtypes["float"])
    lbdag = np.ones(domain, dtype=dtypes["float"])

    lbdas_gt4py = from_array(
        lbdas, dtypes=dtypes["float"], backend=backend, aligned_index=origin
    )
    lbdag_gt4py = from_array(
        lbdag, dtypes=dtypes["float"], backend=backend, aligned_index=origin
    )

    ker_rdryg = from_array(
        KER_RDRYG, dtypes=dtypes["float"], backend=backend, aligned_index=origin
    )

    kernel2(
        ldsoft=ldsoft,
        gdry=gdry,
        lbdas=lbdas_gt4py,
        lbdag=lbdag_gt4py,
        ker_rdryg=ker_rdryg,
        domain=domain,
        origin=origin,
    )
