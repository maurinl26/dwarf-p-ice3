from ctypes import c_double, c_float

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array
from numpy.testing import assert_allclose

from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import (CPU_BACKEND, DEBUG_BACKEND, GPU_BACKEND, dp_dtypes,
                            sp_dtypes)


@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
    ],
)
def test_condensation(dtypes, externals, fortran_dims, backend, domain, origin):
    # Setting backend and precision

    externals.update({"OCND2": False, "OUSERI": True})

    from ice3.stencils.condensation import condensation

    condensation_stencil = stencil(
        backend, name="condensation", definition=condensation, dtypes=dtypes
    )
    fortran_stencil = compile_fortran_stencil(
        "mode_condensation.F90", "mode_condensation", "condensation"
    )

    sigqsat = np.array(
        np.random.rand(domain[0], domain[1]),
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )

    FloatFieldsIJK_Names = [
        "sigrc",
        "pabs",
        "sigs",
        "t",
        "rv_in",
        "ri_in",
        "rc_in",
        "t_out",
        "rv_out",
        "rc_out",
        "ri_out",
        "cldfr",
        "cph",
        "lv",
        "ls",
        "q1",
    ]

    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*domain.shape),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_Names
    }

    # Updating temperature
    FloatFieldsIJK["t"] += 300

    sigqsat_gt4py = from_array(sigqsat, dtype=dtypes["float"], backend=backend)
    pabs_gt4py = from_array(
        FloatFieldsIJK["pabs"],
        dtype=dtypes["float"],
        backend=backend,
    )
    sigs_gt4py = from_array(
        FloatFieldsIJK["sigs"],
        dtype=dtypes["float"],
        backend=backend,
    )
    t_gt4py = from_array(
        FloatFieldsIJK["t"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rv_in_gt4py = from_array(
        FloatFieldsIJK["rv_in"],
        dtype=dtypes["float"],
        backend=backend,
    )
    ri_in_gt4py = from_array(
        FloatFieldsIJK["ri_in"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rc_in_gt4py = from_array(
        FloatFieldsIJK["rc_in"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rv_out_gt4py = from_array(
        FloatFieldsIJK["rv_out"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rc_out_gt4py = from_array(
        FloatFieldsIJK["rc_out"],
        dtype=dtypes["float"],
        backend=backend,
    )
    ri_out_gt4py = from_array(
        FloatFieldsIJK["ri_out"],
        dtype=dtypes["float"],
        backend=backend,
    )
    cldfr_gt4py = from_array(
        FloatFieldsIJK["cldfr"],
        dtype=dtypes["float"],
        backend=backend,
    )
    cph_gt4py = from_array(
        FloatFieldsIJK["cph"],
        dtype=dtypes["float"],
        backend=backend,
    )
    lv_gt4py = from_array(
        FloatFieldsIJK["lv"],
        dtype=dtypes["float"],
        backend=backend,
    )
    ls_gt4py = from_array(
        FloatFieldsIJK["ls"],
        dtype=dtypes["float"],
        backend=backend,
    )
    q1_gt4py = from_array(
        FloatFieldsIJK["q1"],
        dtype=dtypes["float"],
        backend=backend,
    )

    temporary_FloatFieldsIJK_Names = [
        "pv",
        "piv",
        "frac_tmp",
        "qsl",
        "qsi",
        "sigma",
        "cond_tmp",
        "a",
        "b",
        "sbar",
    ]

    temporary_FloatFieldsIJK = {
        name: np.zeros(
            domain.shape,
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in temporary_FloatFieldsIJK_Names
    }

    pv_gt4py = from_array(
        temporary_FloatFieldsIJK["pv"],
        dtype=dtypes["float"],
        backend=backend,
    )
    piv_gt4py = from_array(
        temporary_FloatFieldsIJK["piv"],
        dtype=dtypes["float"],
        backend=backend,
    )
    frac_tmp_gt4py = from_array(
        temporary_FloatFieldsIJK["frac_tmp"],
        dtype=dtypes["float"],
        backend=backend,
    )
    qsl_gt4py = from_array(
        temporary_FloatFieldsIJK["qsl"],
        dtype=dtypes["float"],
        backend=backend,
    )
    qsi_gt4py = from_array(
        temporary_FloatFieldsIJK["qsi"],
        dtype=dtypes["float"],
        backend=backend,
    )
    sigma_gt4py = from_array(
        temporary_FloatFieldsIJK["sigma"],
        dtype=dtypes["float"],
        backend=backend,
    )
    cond_tmp_gt4py = from_array(
        temporary_FloatFieldsIJK["cond_tmp"],
        dtype=dtypes["float"],
        backend=backend,
    )
    a_gt4py = from_array(
        temporary_FloatFieldsIJK["a"],
        dtype=dtypes["float"],
        backend=backend,
    )
    b_gt4py = from_array(
        temporary_FloatFieldsIJK["b"],
        dtype=dtypes["float"],
        backend=backend,
    )
    sbar_gt4py = from_array(
        temporary_FloatFieldsIJK["sbar"],
        dtype=dtypes["float"],
        backend=backend,
    )

    condensation_stencil(
        sigqsat=sigqsat_gt4py,
        pabs=pabs_gt4py,
        sigs=sigs_gt4py,
        t=t_gt4py,
        rv=rv_in_gt4py,
        ri=ri_in_gt4py,
        rc=rc_in_gt4py,
        rv_out=rv_out_gt4py,
        rc_out=rc_out_gt4py,
        ri_out=ri_out_gt4py,
        cldfr=cldfr_gt4py,
        cph=cph_gt4py,
        lv=lv_gt4py,
        ls=ls_gt4py,
        q1=q1_gt4py,
        domain=domain.shape,
        origin=origin,
    )

    logical_keys = {
        "osigmas": "LSIGMAS",
        "ocnd2": "OCND2",
        "ouseri": "OUSERI",
        "hfrac_ice": "FRAC_ICE_ADJUST",
        "hcondens": "CONDENS",
        "lstatnw": "LSTATNW",
    }

    constant_def = {
        "xrv": "RV",
        "xrd": "RD",
        "xalpi": "ALPI",
        "xbetai": "BETAI",
        "xgami": "GAMI",
        "xalpw": "ALPW",
        "xbetaw": "BETAW",
        "xgamw": "GAMW",
        "xtmaxmix": "TMAXMIX",
        "xtminmix": "TMINMIX",
    }

    fortran_externals = {
        **{fkey: externals[pykey] for fkey, pykey in logical_keys.items()},
        **{fkey: externals[pykey] for fkey, pykey in constant_def.items()},
    }

    F2Py_Mapping = {
        "ppabs": "pabs",
        "pt": "t",
        "prv_in": "rv_in",
        "prc_in": "rc_in",
        "pri_in": "ri_in",
        "psigs": "sigs",
        "psigqsat": "sigqsat",
        "plv": "lv",
        "pls": "ls",
        "pcph": "cph",
        "pt_out": "t",
        "prv_out": "rv_out",
        "prc_out": "rc_out",
        "pri_out": "ri_out",
        "pcldfr": "cldfr",
        "zq1": "q1",
        # Temporaries
        "zpv": "pv",
        "zpiv": "piv",
        "zfrac": "frac_tmp",
        "zqsl": "qsl",
        "zqsi": "qsi",
        "zsigma": "sigma",
        "zcond": "cond_tmp",
        "za": "a",
        "zb": "b",
        "zsbar": "sbar",
    }

    Py2F_Mapping = dict(map(reversed, F2Py_Mapping.items()))

    fortran_FloatFieldsIJK = {
        Py2F_Mapping[name]: FloatFieldsIJK[name].reshape(
            domain[0] * domain[1], domain[2]
        )
        for name in FloatFieldsIJK.keys()
    }

    result = fortran_stencil(
        psigsat=sigqsat.reshape(domain.shape[0] * domain.shape[1]),
        **fortran_FloatFieldsIJK,
        **fortran_dims,
        **fortran_externals,
    )

    FieldsOut_Names = [
        "pt_out",
        "prv_out",
        "prc_out",
        "pri_out",
        "pcldfr",
        "zq1",
        "pv",
        "piv",
        "zfrac",
        "zqsl",
        "zqsi",
        "zsigma",
        "zcond",
        "za",
        "zb",
        "zsbar",
    ]

    FieldsOut = {name: result[i] for i, name in enumerate(FieldsOut_Names)}

    assert_allclose(
        FieldsOut["pt_out"], t_out.reshape(domain[0] * domain[1]), rtol=1e-6, atol=1e-6
    )
    assert_allclose(
        FieldsOut["prv_out"],
        rv_out_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert_allclose(
        FieldsOut["prc_out"],
        rc_out_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
    )
    assert_allclose(
        FieldsOut["pri_out"],
        ri_out_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
    )

    assert_allclose(
        FieldsOut["pcldfr"],
        cldfr_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
    )
    assert_allclose(
        FieldsOut["zq1"],
        q1_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
    )
