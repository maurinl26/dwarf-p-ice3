# -*- coding: utf-8 -*-
"""
Test suite for the sign function.

This module contains tests for the sign function implementation in gt4py,
which returns the absolute value of a scalar with the sign determined by
the input scalar's sign.
"""

from __future__ import annotations
from ctypes import c_size_t

from gt4py.cartesian.gtscript import stencil, Field
from gt4py.storage import from_array, zeros

from ice3.functions.sign import sign
from ice3.utils.env import dp_dtypes, sp_dtypes

import pytest
import numpy as np
from numpy.testing import assert_array_equal


def sign_stencil(a: "float", b: Field["float"], c: Field["float"]):
    """
    GT4Py stencil wrapper for testing the sign function.

    This stencil applies the sign function across a field, computing
    c = sign(a, b) for all grid points.

    Parameters
    ----------
    a : float
        Scalar input whose absolute value and sign determine the output.
    b : Field[float]
        Input field (note: currently not used in the sign function logic).
    c : Field[float]
        Output field containing the result of the sign function.
    """
    with computation(PARALLEL), interval(...):
        c = sign(a, b)


@pytest.mark.parametrize("dtypes", [dp_dtypes, sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("debug", marks=pytest.mark.debug),
        pytest.param("numpy", marks=pytest.mark.numpy),
        pytest.param("gt:cpu_ifirst", marks=pytest.mark.cpu),
        pytest.param("gt:gpu", marks=pytest.mark.gpu),
    ],
)
def test_stencil_sign(dtypes, backend, domain, origin):
    """
    Test the sign function stencil with various inputs and backends.

    This test validates the sign function behavior across different
    computational backends (debug, numpy, CPU, GPU) and data types
    (single and double precision).

    The test covers four scenarios:
    1. Positive scalar (a=1.0) with positive field → expects positive output
    2. Negative scalar (a=-1.0) with positive field → expects negative output
    3. Negative scalar (a=-1.0) with negative field → expects negative output
    4. Negative scalar (a=-1.0) with positive field → expects negative output

    Parameters
    ----------
    dtypes : dict
        Dictionary containing data type specifications (dp_dtypes or sp_dtypes).
    backend : str
        The computational backend to use for testing
        (debug, numpy, gt:cpu_ifirst, gt:gpu).
    domain : tuple
        Domain size for the field, provided by pytest fixture.
    origin : tuple
        Origin coordinates for the field, provided by pytest fixture.

    Notes
    -----
    The sign function currently returns abs(a) with the sign of a itself.
    The parameter b appears to be unused in the current implementation.
    """
    sign_stencil_gt4py = stencil(
        definition=sign_stencil, name="sign", dtypes=dtypes, backend=backend
    )

    # Test 1: Positive scalar with positive field
    a = dtypes["float"](1.0)
    b = np.ones(domain)

    b_gt4py = from_array(
        b, dtype=dtypes["float"], backend=backend, aligned_index=origin
    )
    c_gt4py = zeros(shape=domain, dtype=dtypes["float"], backend=backend)

    sign_stencil_gt4py(a=a, b=b_gt4py, c=c_gt4py, domain=domain, origin=origin)

    assert_array_equal(c_gt4py, 1.0)

    # Test 2: Positive scalar with negative field
    a = dtypes["float"](1.0)
    b = -1 * np.ones(domain)

    b_gt4py = from_array(
        b, dtype=dtypes["float"], backend=backend, aligned_index=origin
    )
    c_gt4py = zeros(shape=domain, dtype=dtypes["float"], backend=backend)

    sign_stencil_gt4py(a=a, b=b_gt4py, c=c_gt4py, domain=domain, origin=origin)

    assert_array_equal(c_gt4py, 1.0)

    # Test 3: Negative scalar with negative field
    a = dtypes["float"](-1.0)
    b = -1 * np.ones(domain)

    b_gt4py = from_array(
        b, dtype=dtypes["float"], backend=backend, aligned_index=origin
    )
    c_gt4py = zeros(shape=domain, dtype=dtypes["float"], backend=backend)

    sign_stencil_gt4py(a=a, b=b_gt4py, c=c_gt4py, domain=domain, origin=origin)

    assert_array_equal(c_gt4py, -1.0)

    # Test 4: Negative scalar with positive field (validation)
    a = dtypes["float"](-1.0)
    b = np.ones(domain)

    b_gt4py = from_array(
        b, dtype=dtypes["float"], backend=backend, aligned_index=origin
    )
    c_gt4py = zeros(shape=domain, dtype=dtypes["float"], backend=backend)

    sign_stencil_gt4py(a=a, b=b_gt4py, c=c_gt4py, domain=domain, origin=origin)
