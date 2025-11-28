# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import stencil, Field
from gt4py.storage import ones
from numpy import logical_and
import pytest

from ice3.utils.env import DTYPES, ALL_BACKENDS


def stencil_inner(ldcompute: Field["bool"], logical_switch: "bool"):
    with computation(PARALLEL), interval(...):
        ldcompute = logical_switch


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_inner_stencil_loop(backend, domain, origin):
    inner_stencil = stencil(
        definition=stencil_inner, name="stencil_inner", backend=backend, dtypes=DTYPES
    )

    ldcompute = ones(shape=domain, backend=backend, dtype=DTYPES["bool"])

    # Loop to test
    i = 0
    while ldcompute.any():
        if i < 10:
            inner_stencil(
                ldcompute=ldcompute, logical_switch=True, domain=domain, origin=origin
            )
        else:
            inner_stencil(
                ldcompute=ldcompute, logical_switch=False, domain=domain, origin=origin
            )
        i += 1

    # Output test
    assert i == 11
    print("While loop validates")
