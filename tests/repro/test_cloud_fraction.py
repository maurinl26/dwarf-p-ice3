import logging
from ctypes import c_double, c_float

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array, zeros
from numpy.testing import assert_allclose

from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import (CPU_BACKEND, DEBUG_BACKEND, GPU_BACKEND, dp_dtypes,
                            sp_dtypes)


@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
    ],
)
def test_thermo(benchmark, dtypes, externals, fortran_dims, backend, domain, origin):
    # Compilation of both gt4py and fortran stencils
    from ice3.stencils.cloud_fraction import thermodynamic_fields

    thermo_stencil = stencil(
        backend,
        definition=thermodynamic_fields,
        name="thermo",
        externals=externals,
        dtypes=dtypes,
    )

    fortran_stencil = compile_fortran_stencil(
        "mode_thermo.F90", "mode_thermo", "latent_heat"
    )

    F2Py_Mapping = {
        "prv": "rv",
        "prc": "rc",
        "pri": "ri",
        "prr": "rr",
        "prs": "rs",
        "prg": "rg",
        "pth": "th",
        "pexn": "exn",
        "zt": "t",
        "zls": "ls",
        "zlv": "lv",
        "zcph": "cph",
    }

    Py2F_Mapping = dict(map(reversed, F2Py_Mapping.items()))

    externals_mapping = {
        "xlvtt": "LVTT",
        "xlstt": "LSTT",
        "xcpv": "CPV",
        "xci": "CI",
        "xcl": "CL",
        "xtt": "TT",
        "xcpd": "CPD",
    }

    fortran_externals = {
        fname: externals[pyname] for fname, pyname in externals_mapping.items()
    }

    FloatFieldsIJK_Names = [
        "th",
        "exn",
        "rv",
        "rc",
        "rr",
        "ri",
        "rs",
        "rg",
        "lv",
        "ls",
        "cph",
        "t",
    ]

    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_Names
    }

    th_gt4py = from_array(
        FloatFieldsIJK["th"],
        dtype=dtypes["float"],
        backend=backend,
    )
    exn_gt4py = from_array(
        FloatFieldsIJK["exn"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rv_gt4py = from_array(
        FloatFieldsIJK["rv"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rc_gt4py = from_array(
        FloatFieldsIJK["rc"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rr_gt4py = from_array(
        FloatFieldsIJK["rr"],
        dtype=dtypes["float"],
        backend=backend,
    )
    ri_gt4py = from_array(
        FloatFieldsIJK["ri"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rs_gt4py = from_array(
        FloatFieldsIJK["rs"],
        dtype=dtypes["float"],
        backend=backend,
    )
    rg_gt4py = from_array(
        FloatFieldsIJK["rg"],
        dtype=dtypes["float"],
        backend=backend,
    )

    lv_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    ls_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    cph_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    t_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)

    Fortran_FloatFieldsIJK = {
        Py2F_Mapping[name]: field.reshape(domain[0] * domain[1], domain[2])
        for name, field in FloatFieldsIJK.items()
    }

    def run_thermo():
        thermo_stencil(
            th=th_gt4py,
            exn=exn_gt4py,
            rv=rv_gt4py,
            rc=rc_gt4py,
            rr=rr_gt4py,
            ri=ri_gt4py,
            rs=rs_gt4py,
            rg=rg_gt4py,
            lv=lv_gt4py,
            ls=ls_gt4py,
            cph=cph_gt4py,
            t=t_gt4py,
            domain=domain,
            origin=origin,
        )

        return (t_gt4py, ls_gt4py, lv_gt4py, cph_gt4py)

    benchmark(run_thermo)

    zt, zlv, zls, zcph = fortran_stencil(
        krr=6,
        **Fortran_FloatFieldsIJK,
        **fortran_externals,
        **fortran_dims,
    )

    logging.info(f"Machine dtypes {np.finfo(float).eps}")

    assert_allclose(zt, t_gt4py.reshape(domain[0] * domain[1], domain[2]), rtol=1e-6)
    assert_allclose(zlv, lv_gt4py.reshape(domain[0] * domain[1], domain[2]), rtol=1e-6)
    assert_allclose(zls, ls_gt4py.reshape(domain[0] * domain[1], domain[2]), rtol=1e-6)
    assert_allclose(
        zcph, cph_gt4py.reshape(domain[0] * domain[1], domain[2]), rtol=1e-6
    )


@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
    ],
)
def test_cloud_fraction_1(
    benchmark, externals, fortran_dims, dtypes, backend, domain, origin
):
    externals.update({"LSUBG_COND": True})

    from ice3.stencils.cloud_fraction import cloud_fraction_1

    cloud_fraction_1 = stencil(
        backend,
        definition=cloud_fraction_1,
        name="cloud_fraction_1",
        externals=externals,
    )

    dt = dtypes["float"](50.0)

    FloatFieldsIJK_Names = [
        "lv",
        "ls",
        "cph",
        "exnref",
        "rc",
        "ri",
        "ths",
        "rvs",
        "rcs",
        "ris",
        "rc_tmp",
        "ri_tmp",
    ]

    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_Names
    }

    lv_gt4py = from_array(
        FloatFieldsIJK["lv"],
        backend=backend,
        dtype=dtypes["float"],
    )
    ls_gt4py = from_array(
        FloatFieldsIJK["ls"],
        backend=backend,
        dtype=dtypes["float"],
    )
    cph_gt4py = from_array(
        FloatFieldsIJK["cph"],
        backend=backend,
        dtype=dtypes["float"],
    )
    exnref_gt4py = from_array(
        FloatFieldsIJK["exnref"],
        backend=backend,
        dtype=dtypes["float"],
    )
    rc_gt4py = from_array(
        FloatFieldsIJK["rc"],
        backend=backend,
        dtype=dtypes["float"],
    )
    ri_gt4py = from_array(
        FloatFieldsIJK["ri"],
        backend=backend,
        dtype=dtypes["float"],
    )
    ths_gt4py = from_array(
        FloatFieldsIJK["ths"],
        backend=backend,
        dtype=dtypes["float"],
    )
    rvs_gt4py = from_array(
        FloatFieldsIJK["rvs"],
        backend=backend,
        dtype=dtypes["float"],
    )
    rcs_gt4py = from_array(
        FloatFieldsIJK["rcs"],
        backend=backend,
        dtype=dtypes["float"],
    )
    ris_gt4py = from_array(
        FloatFieldsIJK["ris"],
        backend=backend,
        dtype=dtypes["float"],
    )
    rc_tmp_gt4py = from_array(
        FloatFieldsIJK["rc_tmp"],
        backend=backend,
        dtype=dtypes["float"],
    )
    ri_tmp_gt4py = from_array(
        FloatFieldsIJK["ri_tmp"],
        backend=backend,
        dtype=dtypes["float"],
    )

    def run_cloud_fraction_1():
        cloud_fraction_1(
            lv=lv_gt4py,
            ls=ls_gt4py,
            cph=cph_gt4py,
            exnref=exnref_gt4py,
            rc=rc_gt4py,
            ri=ri_gt4py,
            ths=ths_gt4py,
            rvs=rvs_gt4py,
            rcs=rcs_gt4py,
            ris=ris_gt4py,
            rc_tmp=rc_tmp_gt4py,
            ri_tmp=ri_tmp_gt4py,
            dt=dt,
            domain=domain,
            origin=origin,
        )
        return (ths_gt4py, rvs_gt4py, rcs_gt4py, ris_gt4py, rc_tmp_gt4py, ri_tmp_gt4py)

    benchmark(run_cloud_fraction_1)

    fortran_stencil = compile_fortran_stencil(
        "mode_cloud_fraction_split.F90", "mode_cloud_fraction_split", "cloud_fraction_1"
    )
    logging.info(f"SUBG_MF_PDF  : {externals['SUBG_MF_PDF']}")
    logging.info(f"LSUBG_COND   : {externals['LSUBG_COND']}")

    F2Py_Mapping = {
        "zrc": "rc_tmp",
        "zri": "ri_tmp",
        "pexnref": "exnref",
        "zcph": "cph",
        "zlv": "lv",
        "zls": "ls",
        "prc": "rc",
        "pri": "ri",
        "prvs": "rvs",
        "prcs": "rcs",
        "pths": "ths",
        "pris": "ris",
    }

    Py2F_Mapping = dict(map(reversed, F2Py_Mapping.items()))

    Fortran_FloatFieldsIJK = {
        Py2F_Mapping[name]: field.reshape(domain[0] * domain[1], domain[2])
        for name, field in FloatFieldsIJK.items()
    }

    result = fortran_stencil(ptstep=dt, **Fortran_FloatFieldsIJK, **fortran_dims)

    FieldsOut_Names = ["pths", "prvs", "prcs", "pris"]

    FieldsOut = {name: result[i] for i, name in enumerate(FieldsOut_Names)}

    logging.info(f"Machine dtypes {np.finfo(float).eps}")

    logging.info(f"Mean ths_gt4py       {ths_gt4py.mean()}")
    logging.info(f"Mean pths_out        {FieldsOut['pths'].mean()}")

    logging.info(f"Mean rvs_gt4py       {rvs_gt4py.mean()}")
    logging.info(f"Mean prvs_out        {FieldsOut['prvs'].mean()}")

    logging.info(f"Mean rcs_gt4py       {rcs_gt4py.mean()}")
    logging.info(f"Mean prcs_out        {FieldsOut['prcs'].mean()}")

    logging.info(f"Mean ris_gt4py       {ris_gt4py.mean()}")
    logging.info(f"Mean pris_out        {FieldsOut['pris'].mean()}")

    assert_allclose(
        FieldsOut["pths"],
        ths_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
    )
    assert_allclose(
        FieldsOut["prvs"],
        rvs_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
    )
    assert_allclose(
        FieldsOut["prcs"],
        rcs_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
    )
    assert_allclose(
        FieldsOut["pris"],
        ris_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
    )


@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
    ],
)
def test_cloud_fraction_2(
    benchmark, dtypes, externals, fortran_dims, backend, domain, origin
):
    externals["LSUBG_COND"] = True
    externals.update({"SUBG_MF_PDF": 0})

    from ice3.stencils.cloud_fraction import cloud_fraction_2

    cloud_fraction_2 = stencil(
        backend,
        definition=cloud_fraction_2,
        name="cloud_fraction_1",
        externals=externals,
    )
    fortran_stencil = compile_fortran_stencil(
        "mode_cloud_fraction_split.F90", "mode_cloud_fraction_split", "cloud_fraction_2"
    )

    dt = dtypes["float"](50.0)

    FloatFieldsIJK_Names = [
        "rhodref",
        "exnref",
        "t",
        "cph",
        "lv",
        "ls",
        "ths",
        "rvs",
        "rcs",
        "ris",
        "rc_mf",
        "ri_mf",
        "cf_mf",
        "cldfr",
        "hlc_hrc",
        "hlc_hcf",
        "hli_hri",
        "hli_hcf",
    ]

    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_Names
    }

    rhodref_gt4py = from_array(
        FloatFieldsIJK["rhodref"],
        backend=backend,
        dtype=dtypes["float"],
    )
    exnref_gt4py = from_array(
        FloatFieldsIJK["exnref"],
        backend=backend,
        dtype=dtypes["float"],
    )
    t_gt4py = from_array(
        FloatFieldsIJK["t"],
        backend=backend,
        dtype=dtypes["float"],
    )
    cph_gt4py = from_array(
        FloatFieldsIJK["cph"],
        backend=backend,
        dtype=dtypes["float"],
    )
    lv_gt4py = from_array(
        FloatFieldsIJK["lv"],
        backend=backend,
        dtype=dtypes["float"],
    )
    ls_gt4py = from_array(
        FloatFieldsIJK["ls"],
        backend=backend,
        dtype=dtypes["float"],
    )
    ths_gt4py = from_array(
        FloatFieldsIJK["ths"],
        backend=backend,
        dtype=dtypes["float"],
    )
    rvs_gt4py = from_array(
        FloatFieldsIJK["rvs"],
        backend=backend,
        dtype=dtypes["float"],
    )
    rcs_gt4py = from_array(
        FloatFieldsIJK["rcs"],
        backend=backend,
        dtype=dtypes["float"],
    )
    ris_gt4py = from_array(
        FloatFieldsIJK["ris"],
        backend=backend,
        dtype=dtypes["float"],
    )
    rc_mf_gt4py = from_array(
        FloatFieldsIJK["rc_mf"],
        backend=backend,
        dtype=dtypes["float"],
    )
    ri_mf_gt4py = from_array(
        FloatFieldsIJK["ri_mf"],
        backend=backend,
        dtype=dtypes["float"],
    )
    cf_mf_gt4py = from_array(
        FloatFieldsIJK["cf_mf"],
        backend=backend,
        dtype=dtypes["float"],
    )
    cldfr_gt4py = from_array(
        FloatFieldsIJK["cldfr"],
        backend=backend,
        dtype=dtypes["float"],
    )
    hlc_hrc_gt4py = from_array(
        FloatFieldsIJK["hlc_hrc"],
        backend=backend,
        dtype=dtypes["float"],
    )
    hlc_hcf_gt4py = from_array(
        FloatFieldsIJK["hlc_hcf"],
        backend=backend,
        dtype=dtypes["float"],
    )
    hli_hri_gt4py = from_array(
        FloatFieldsIJK["hli_hri"],
        backend=backend,
        dtype=dtypes["float"],
    )
    hli_hcf_gt4py = from_array(
        FloatFieldsIJK["hli_hcf"],
        backend=backend,
        dtype=dtypes["float"],
    )

    def run_cloud_fraction_2():
        cloud_fraction_2(
            rhodref=rhodref_gt4py,
            exnref=exnref_gt4py,
            t=t_gt4py,
            cph=cph_gt4py,
            lv=lv_gt4py,
            ls=ls_gt4py,
            ths=ths_gt4py,
            rvs=rvs_gt4py,
            rcs=rcs_gt4py,
            ris=ris_gt4py,
            rc_mf=rc_mf_gt4py,
            ri_mf=ri_mf_gt4py,
            cf_mf=cf_mf_gt4py,
            cldfr=cldfr_gt4py,
            hlc_hrc=hlc_hrc_gt4py,
            hlc_hcf=hlc_hcf_gt4py,
            hli_hri=hli_hri_gt4py,
            hli_hcf=hli_hcf_gt4py,
            dt=dt,
            domain=domain,
            origin=origin,
        )
        return (hlc_hrc_gt4py, hlc_hcf_gt4py, hli_hri_gt4py, hli_hcf_gt4py)

    benchmark(run_cloud_fraction_2)

    logging.info(f"SUBG_MF_PDF  : {externals['SUBG_MF_PDF']}")
    logging.info(f"LSUBG_COND   : {externals['LSUBG_COND']}")

    keys_mapping = {
        "xcriautc": "CRIAUTC",
        "xcriauti": "CRIAUTI",
        "xacriauti": "ACRIAUTI",
        "xbcriauti": "BCRIAUTI",
        "xtt": "TT",
        "csubg_mf_pdf": "SUBG_MF_PDF",
        "lsubg_cond": "LSUBG_COND",
    }

    fortran_externals = {key: externals[value] for key, value in keys_mapping.items()}

    logging.info(f"csubg_mf_pdf : {fortran_externals['csubg_mf_pdf']}")
    from ice3.phyex_common.ice_parameters import SubGridMassFluxPDF

    logging.info(
        f"csubg_mf_pdf : {SubGridMassFluxPDF(fortran_externals['csubg_mf_pdf'])}"
    )
    logging.info(f"lsubg_cond   : {fortran_externals['lsubg_cond']}")

    F2Py_Mapping = {
        "pexnref": "exnref",
        "prhodref": "rhodref",
        "zcph": "cph",
        "zlv": "lv",
        "zls": "ls",
        "zt": "t",
        "pcf_mf": "cf_mf",
        "prc_mf": "rc_mf",
        "pri_mf": "ri_mf",
        "pths": "ths",
        "prvs": "rvs",
        "prcs": "rcs",
        "pris": "ris",
        "pcldfr": "cldfr",
        "phlc_hrc": "hlc_hrc",
        "phlc_hcf": "hlc_hcf",
        "phli_hri": "hli_hri",
        "phli_hcf": "hli_hcf",
    }

    Py2F_Mapping = dict(map(reversed, F2Py_Mapping.items()))

    Fortran_FloatFieldsIJK = {
        Py2F_Mapping[name]: field.reshape(domain[0] * domain[1], domain[2])
        for name, field in FloatFieldsIJK.items()
    }

    (
        pths_out,
        prvs_out,
        prcs_out,
        pris_out,
        pcldfr_out,
        phlc_hrc_out,
        phlc_hcf_out,
        phli_hri_out,
        phli_hcf_out,
    ) = fortran_stencil(
        ptstep=dt, **Fortran_FloatFieldsIJK, **fortran_dims, **fortran_externals
    )

    logging.info(f"Machine dtypes {np.finfo(float).eps}")

    logging.info(f"Mean cldfr_gt4py     {cldfr_gt4py.mean()}")
    logging.info(f"Mean pcldfr_out      {pcldfr_out.mean()}")

    logging.info(f"Mean hlc_hrc_gt4py   {hlc_hrc_gt4py.mean()}")
    logging.info(f"Mean phlc_hrc_out    {phlc_hrc_out.mean()}")

    logging.info(f"Mean hlc_hcf_gt4py   {hlc_hcf_gt4py.mean()}")
    logging.info(f"Mean phlc_hcf_out    {phlc_hcf_out.mean()}")

    logging.info(f"Mean hli_hri_gt4py   {hli_hri_gt4py.mean()}")
    logging.info(f"Mean phli_hri_out    {phli_hri_out.mean()}")

    logging.info(f"Mean hli_hcf_gt4py   {hli_hcf_gt4py.mean()}")
    logging.info(f"Mean phli_hcf        {phli_hcf_out.mean()}")

    assert_allclose(
        pcldfr_out,
        cldfr_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
    )
    assert_allclose(
        phlc_hcf_out,
        hlc_hcf_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
    )
    assert_allclose(
        phlc_hrc_out,
        hlc_hrc_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
    )
    assert_allclose(
        phli_hri_out,
        hli_hri_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
    )
    assert_allclose(
        phli_hcf_out,
        hli_hcf_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
    )
