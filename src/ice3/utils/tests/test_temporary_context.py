# -*- coding: utf-8 -*-
from gt4py.cartesian.gtscript import IJK, IJ, K
from ice3.utils.env import dp_dtypes, sp_dtypes
from ice3.utils.storage import managed_temporaries
import pytest


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
def test_managed_temporaries_context(backend, origin, domain, dtypes):

    with managed_temporaries(
        [
            (IJK, "float"),
            (IJK, "int"),
            (IJK, "bool"),
            (IJ, "float"),
            (IJ, "int"),
            (IJ, "bool"),
            (K, "float"), 
            (K, "int"),
            (K, "bool")
        ],
        dtypes=dtypes,
        backend=backend,
        domain=domain
    ) as (
        field_3d_float,
        field_3d_int,
        field_3d_bool,
        field_2d_float,
        field_2d_int,
        field_2d_bool, 
        vertical_slice_float,
        vertical_slice_int,
        vertical_slice_bool
    ):

        print(f"3DFieldFloat : {field_3d_float}")
        print(f"3DFieldInt   : {field_3d_int}")
        print(f"3DFieldBool  : {field_3d_bool}")
        print(f"2DFieldFloat : {field_2d_float}")
        print(f"2DFieldInt   : {field_2d_int}")
        print(f"2DFieldBool  : {field_2d_bool}")
        print(f"VerticalSliceFloat   : {vertical_slice_float}")
        print(f"VerticalSliceInt   : {vertical_slice_int}")
        print(f"VerticalSliceBool  : {vertical_slice_bool}")

