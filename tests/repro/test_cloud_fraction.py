import logging
from ctypes import c_double, c_float

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array, zeros
from numpy.testing import assert_allclose
from gt4py.eve.codegen import JinjaTemplate as as_jinja

from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import dp_dtypes, sp_dtypes


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
        prv= FloatFieldsIJK["rv"],
        prc=FloatFieldsIJK["rc"],
        pri=FloatFieldsIJK["ri"],
        prr=FloatFieldsIJK["rr"],
        prs=FloatFieldsIJK["rs"],
        prg=FloatFieldsIJK["rg"],
        pth=FloatFieldsIJK["th"],
        pexn=FloatFieldsIJK["exn"],
        xlvtt=externals["LVTT"],
        xlstt=externals["LSTT"],
        xcpv=externals["CPV"],
        xci=externals["CI"],
        xcl=externals["CL"],
        xtt=externals["TT"],
        xcpd=externals["CPD"],
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
        pytest.param("debug", marks=pytest.mark.debug),
        pytest.param("numpy", marks=pytest.mark.numpy),
        pytest.param("gt:cpu_ifirst", marks=pytest.mark.cpu),
        pytest.param("gt:gpu", marks=pytest.mark.gpu),
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


    fortran_stencil = compile_fortran_stencil(
        "mode_cloud_fraction_split.F90", "mode_cloud_fraction_split", "cloud_fraction_1"
    )
    logging.info(f"SUBG_MF_PDF  : {externals['SUBG_MF_PDF']}")
    logging.info(f"LSUBG_COND   : {externals['LSUBG_COND']}")


    (pths_out, prvs_out, prcs_out, pris_out)\
        = fortran_stencil(ptstep=dt,
                            zrc=FloatFieldsIJK["rc_tmp"].reshape(domain[0]*domain[1], domain[2]),
                            zri=FloatFieldsIJK["ri_tmp"].reshape(domain[0]*domain[1], domain[2]),
                            pexnref=FloatFieldsIJK["exnref"].reshape(domain[0]*domain[1], domain[2]),
                            zcph=FloatFieldsIJK["cph"].reshape(domain[0]*domain[1], domain[2]),
                            zlv=FloatFieldsIJK["lv"].reshape(domain[0]*domain[1], domain[2]),
                            zls=FloatFieldsIJK["ls"].reshape(domain[0]*domain[1], domain[2]),
                            prc=FloatFieldsIJK["rc"].reshape(domain[0]*domain[1], domain[2]),
                            pri=FloatFieldsIJK["ri"].reshape(domain[0]*domain[1], domain[2]),
                            prvs=FloatFieldsIJK["rvs"].reshape(domain[0]*domain[1], domain[2]),
                            prcs=FloatFieldsIJK["rcs"].reshape(domain[0]*domain[1], domain[2]),
                            pths=FloatFieldsIJK["ths"].reshape(domain[0]*domain[1], domain[2]),
                            pris=FloatFieldsIJK["ris"].reshape(domain[0]*domain[1], domain[2]),
                            **fortran_dims)


    logging.info(f"Machine dtypes {np.finfo(float).eps}")

    assert_allclose(
        pths_out,
        ths_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
        atol=1e-8
    )
    assert_allclose(
        prvs_out,
        rvs_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
        atol=1e-8
    )
    assert_allclose(
        prcs_out,
        rcs_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
        atol=1e-8
    )
    assert_allclose(
        pris_out,
        ris_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=1e-6,
        atol=1e-8
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
        "mode_cloud_fraction_split.F90",
        "mode_cloud_fraction_split",
        "cloud_fraction_2"
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

    logging.info(f"SUBG_MF_PDF  : {externals['SUBG_MF_PDF']}")
    logging.info(f"LSUBG_COND   : {externals['LSUBG_COND']}")
    logging.info(f"csubg_mf_pdf : {fortran_externals['csubg_mf_pdf']}")

    fortran_externals = {key: externals[value] for key, value in keys_mapping.items()}

    from ice3.phyex_common.ice_parameters import SubGridMassFluxPDF

    logging.info(
        f"csubg_mf_pdf : {SubGridMassFluxPDF(fortran_externals['csubg_mf_pdf'])}"
    )
    logging.info(f"lsubg_cond   : {fortran_externals['lsubg_cond']}")

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
        ptstep=dt,
        pexnref=FloatFieldsIJK["exnref"].reshape(domain[0] * domain[1], domain[2]),
        prhodref=FloatFieldsIJK["rhodref"].reshape(domain[0] * domain[1], domain[2]),
        zcph=FloatFieldsIJK["cph"].reshape(domain[0] * domain[1], domain[2]),
        zlv=FloatFieldsIJK["lv"].reshape(domain[0] * domain[1], domain[2]),
        zls=FloatFieldsIJK["ls"].reshape(domain[0] * domain[1], domain[2]),
        zt=FloatFieldsIJK[""].reshape(domain[0] * domain[1], domain[2]),
        pcf_mf=FloatFieldsIJK["cf_mf"].reshape(domain[0] * domain[1], domain[2]),
        prc_mf=FloatFieldsIJK["rc_mf"].reshape(domain[0] * domain[1], domain[2]),
        pri_mf=FloatFieldsIJK["ri_mf"].reshape(domain[0] * domain[1], domain[2]),
        pths=FloatFieldsIJK["ths"].reshape(domain[0] * domain[1], domain[2]),
        prvs=FloatFieldsIJK["rvs"].reshape(domain[0] * domain[1], domain[2]),
        prcs=FloatFieldsIJK["rcs"].reshape(domain[0] * domain[1], domain[2]),
        pris=FloatFieldsIJK["ris"].reshape(domain[0] * domain[1], domain[2]),
        pcldfr=FloatFieldsIJK["cldfr"].reshape(domain[0] * domain[1], domain[2]),
        phlc_hrc=FloatFieldsIJK["hlc_hrc"].reshape(domain[0] * domain[1], domain[2]),
        phlc_hcf=FloatFieldsIJK["hlc_hcf"].reshape(domain[0] * domain[1], domain[2]),
        phli_hri=FloatFieldsIJK["hli_hrc"].reshape(domain[0] * domain[1], domain[2]),
        phli_hcf=FloatFieldsIJK["hli_hcf"].reshape(domain[0] * domain[1], domain[2]),
        xcriautc=externals["CRIAUTC"],
        xcriauti=externals["CRIAUTI"],
        xacriauti=externals["ACRIAUTI"],
        xbcriauti=externals["BCRIAUTI"],
        xtt=externals["TT"],
        csubg_mf_pdf=externals["SUBG_MF_PDF"],
        lsubg_cond=externals["LSUBG_COND"],
        **fortran_dims,
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
