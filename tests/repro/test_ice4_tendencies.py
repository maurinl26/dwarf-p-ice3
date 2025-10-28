import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array
from numpy.testing import assert_allclose

from ice3.utils.allocate_random_fields import allocate_random_fields
from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import (CPU_BACKEND, DEBUG_BACKEND, GPU_BACKEND, dp_dtypes,
                            sp_dtypes)


@pytest.mark.parametrize("dtypes", [dp_dtypes, sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
    ],
)
def test_ice4_nucleation_post_processing(
    gt4py_config, externals, fortran_packed_dims, dtypes, backend, domain, origin
):
    from ice3.stencils.ice4_tendencies import ice4_nucleation_post_processing

    ice4_nucleation_post_processing_gt4py = stencil(
        backend,
        definition=ice4_nucleation_post_processing,
        name="ice4_nucleation_post_processing",
        dtypes=dtypes,
        externals=externals,
    )
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90",
        "mode_ice4_tendencies",
        "ice4_nucleation_post_processing",
    )
    field_names = ["t", "exn", "lsfact", "tht", "rvt", "rit", "rvheni_mr"]
    fields, gt4py_buffers = allocate_random_fields(field_names, gt4py_config, domain)
    ice4_nucleation_post_processing_gt4py(
        **gt4py_buffers, domain=domain, origin=(0, 0, 0)
    )
    f2py_mapping = {
        "plsfact": "lsfact",
        "pexn": "exn",
        "ptht": "tht",
        "prit": "rit",
        "prvt": "rvt",
        "zt": "t",
        "prvheni_mr": "rvheni_mr",
    }
    fortran_FloatFieldsIJK = {
        name: fields[value].ravel() for name, value in f2py_mapping.items()
    }
    result = fortran_stencil(**fortran_FloatFieldsIJK, **fortran_packed_dims)
    tht_out, rvt_out, rit_out, zt_out = result[:4]
    assert_allclose(tht_out, gt4py_buffers["tht"].ravel(), rtol=1e-3)
    assert_allclose(rit_out, gt4py_buffers["rit"].ravel(), rtol=1e-3)
    assert_allclose(zt_out, gt4py_buffers["t"].ravel(), rtol=1e-3)
    assert_allclose(rvt_out, gt4py_buffers["rvt"].ravel(), rtol=1e-3)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
    ],
)
def test_ice4_rrhong_post_processing(
    externals, fortran_packed_dims, dtypes, domain, origin, backend
):
    from ice3.stencils.ice4_tendencies import ice4_rrhong_post_processing

    ice4_rrhong_post_processing_gt4py = stencil(
        backend,
        name="ice4_rrhong_post_processing",
        definition=ice4_rrhong_post_processing,
        dtypes=dtypes,
        externals=externals,
    )

    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90",
        "mode_ice4_tendencies",
        "ice4_rrhong_post_processing",
    )
    field_names = ["t", "exn", "lsfact", "lvfact", "tht", "rrt", "rgt", "rrhong_mr"]
    fields, gt4py_buffers = allocate_random_fields(field_names, gt4py_config, domain)
    ice4_rrhong_post_processing_gt4py(**gt4py_buffers, domain=domain, origin=(0, 0, 0))
    f2py_mapping = {
        "plsfact": "lsfact",
        "plvfact": "lvfact",
        "pexn": "exn",
        "prrhong_mr": "rrhong_mr",
        "ptht": "tht",
        "pt": "t",
        "prrt": "rrt",
        "prgt": "rgt",
    }
    fortran_FloatFieldsIJK = {
        name: fields[value].ravel() for name, value in f2py_mapping.items()
    }
    result = fortran_stencil(**fortran_FloatFieldsIJK, **fortran_packed_dims)
    tht_out, t_out, rrt_out, rgt_out = result[:4]
    assert_allclose(tht_out, gt4py_buffers["tht"].ravel(), rtol=1e-3)
    assert_allclose(t_out, gt4py_buffers["t"].ravel(), rtol=1e-3)
    assert_allclose(rrt_out, gt4py_buffers["rrt"].ravel(), rtol=1e-3)
    assert_allclose(rgt_out, gt4py_buffers["rgt"].ravel(), rtol=1e-3)


@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
    ],
)
def test_ice4_rimltc_post_processing(
    backend, externals, fortran_packed_dims, dtypes, domain, origin
):
    from ice3.stencils.ice4_tendencies import ice4_rimltc_post_processing

    ice4_rimltc_post_processing_gt4py = stencil(
        backend,
        name="ice4_rimltc_post_processing",
        definition=ice4_rimltc_post_processing,
        dtypes=dtypes,
        externals=externals,
    )
    fortran_stencil = compile_fortran_stencil(
        fortran_script="mode_ice4_tendencies.F90",
        fortran_module="mode_ice4_tendencies",
        fortran_stencil="ice4_rimltc_post_processing",
    )

    field_names = ["t", "exn", "lsfact", "lvfact", "rimltc_mr", "tht", "rct", "rit"]
    fields, gt4py_buffers = allocate_random_fields(field_names, gt4py_config, domain)
    ice4_rimltc_post_processing_gt4py(**gt4py_buffers, domain=domain, origin=(0, 0, 0))
    f2py_mapping = {
        "plsfact": "lsfact",
        "plvfact": "lvfact",
        "pexn": "exn",
        "primltc_mr": "rimltc_mr",
        "ptht": "tht",
        "pt": "t",
        "prit": "rit",
        "prct": "rct",
    }

    fortran_FloatFieldsIJK = {
        name: fields[value].ravel() for name, value in f2py_mapping.items()
    }

    result = fortran_stencil(**fortran_FloatFieldsIJK, **fortran_packed_dims)
    tht_out, t_out, rct_out, rit_out = result[:4]
    assert_allclose(tht_out, gt4py_buffers["tht"].ravel(), rtol=1e-3)
    assert_allclose(t_out, gt4py_buffers["t"].ravel(), rtol=1e-3)
    assert_allclose(rct_out, gt4py_buffers["rct"].ravel(), rtol=1e-3)
    assert_allclose(rit_out, gt4py_buffers["rit"].ravel(), rtol=1e-3)


@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKENDS, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKENDS, marks=pytest.mark.gpu),
    ],
)
def test_ice4_fast_rg_pre_processing(
    externals, fortran_packed_dims, dtypes, backend, domain, origin
):
    from ice3.stencils.ice4_tendencies import ice4_fast_rg_pre_processing

    ice4_fast_rg_pre_processing_gt4py = stencil(
        backend,
        name="ice4_fast_rg_pre_processing",
        definition=ice4_fast_rg_pre_processing,
        dtypes=dtypes,
        externals=externals,
    )
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90",
        "mode_ice4_tendencies",
        "ice4_fast_rg_pre_processing",
    )

    field_names = [
        "rgsi",
        "rgsi_mr",
        "rvdepg",
        "rsmltg",
        "rraccsg",
        "rsaccrg",
        "rcrimsg",
        "rsrimcg",
        "rrhong_mr",
        "rsrimcg_mr",
    ]
    fields, gt4py_buffers = allocate_random_fields(field_names, gt4py_config, domain)
    ice4_fast_rg_pre_processing_gt4py(**gt4py_buffers, domain=domain, origin=(0, 0, 0))
    result = fortran_stencil(
        rvdepg=fields["rvdepg"],
        rsmltg=fields["rsmltg"],
        rraccsg=fields["rraccsg"],
        rsaccrg=fields["rsaccrg"],
        rcrimsg=fields["rcrimsg"],
        rsrimcg=fields["rsrimcg"],
        rrhong_mr=fields["rrhong_mr"],
        rsrimcg_mr=fields["rsrimcg_mr"],
        zgrsi=fields["rgsi"],
        zrgsi_mr=fields["rgsi_mr"],
        **fortran_packed_dims,
    )
    zrgsi_out, zrgsi_mr_out = result[:2]
    assert_allclose(zrgsi_out, gt4py_buffers["rgsi"].ravel(), rtol=1e-3)
    assert_allclose(zrgsi_mr_out, gt4py_buffers["rgsi_mr"].ravel(), rtol=1e-3)


@pytest.mark.parametrize("dtypes", [dp_dtypes, sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKENDS, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKENDS, marks=pytest.mark.gpu),
    ],
)
def test_ice4_increment_update(
    externals, fortran_packed_dims, dtypes, backend, domain, origin
):
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90", "mode_ice4_tendencies", "ice4_increment_update"
    )

    from ice3.stencils.ice4_tendencies import ice4_increment_update

    ice4_increment_update_gt4py = stencil(
        backend,
        name="ice4_increment_update",
        definition=ice4_increment_update,
        dtypes=dtypes,
        externals=externals,
    )
    field_names = [
        "lsfact",
        "lvfact",
        "theta_increment",
        "rv_increment",
        "rc_increment",
        "rr_increment",
        "ri_increment",
        "rs_increment",
        "rg_increment",
        "rvheni_mr",
        "rimltc_mr",
        "rrhong_mr",
        "rsrimcg_mr",
    ]
    fields, gt4py_buffers = allocate_random_fields(field_names, gt4py_config, domain)
    ice4_increment_update_gt4py(**gt4py_buffers, domain=domain, origin=(0, 0, 0))
    result = fortran_stencil(
        plsfact=fields["lsfact"].ravel(),
        plvfact=fields["lvfact"].ravel(),
        prvheni_mr=fields["rvheni_mr"].ravel(),
        primltc_mr=fields["rimltc_mr"].ravel(),
        prrhong_mr=fields["rrhong_mr"].ravel(),
        prsrimcg_mr=fields["rsrimcg_mr"].ravel(),
        pth_inst=fields["theta_increment"].ravel(),
        prv_inst=fields["rv_increment"].ravel(),
        prc_inst=fields["rc_increment"].ravel(),
        prr_inst=fields["rr_increment"].ravel(),
        pri_inst=fields["ri_increment"].ravel(),
        prs_inst=fields["rs_increment"].ravel(),
        prg_inst=fields["rg_increment"].ravel(),
        **fortran_packed_dims,
    )
    (
        pth_inst_out,
        prv_inst_out,
        prc_inst_out,
        prr_inst_out,
        pri_inst_out,
        prs_inst_out,
        prg_inst_out,
    ) = result[:7]
    assert_allclose(pth_inst_out, gt4py_buffers["theta_increment"].ravel(), rtol=1e-3)
    assert_allclose(prv_inst_out, gt4py_buffers["rv_increment"].ravel(), rtol=1e-3)
    assert_allclose(prc_inst_out, gt4py_buffers["rc_increment"].ravel(), rtol=1e-3)
    assert_allclose(pri_inst_out, gt4py_buffers["ri_increment"].ravel(), rtol=1e-3)
    assert_allclose(prr_inst_out, gt4py_buffers["rr_increment"].ravel(), rtol=1e-3)
    assert_allclose(prs_inst_out, gt4py_buffers["rs_increment"].ravel(), rtol=1e-3)
    assert_allclose(prg_inst_out, gt4py_buffers["rg_increment"].ravel(), rtol=1e-3)


@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
    ],
)
def test_ice4_derived_fields(
    domain, externals, fortran_packed_dims, dtypes, backend, grid, origin
):
    from ice3.stencils.ice4_tendencies import ice4_derived_fields

    ice4_derived_fields_gt4py = stencil(
        backend,
        definition=ice4_derived_fields,
        name="ice4_derived_fields",
        dtypes=dtypes,
        externals=externals,
    )

    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90", "mode_ice4_tendencies", "ice4_derived_fields"
    )

    t = 300 * np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rhodref = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    pres = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    ssi = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    ka = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    dv = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    ai = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    cj = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rvt = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    zw = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )

    t_gt4py = from_array(t, dtype=dtypes["float"], backend=backend)
    rhodref_gt4py = from_array(rhodref, dtype=dtypes["float"], backend=backend)
    pres_gt4py = from_array(pres, dtype=dtypes["float"], backend=backend)
    ssi_gt4py = from_array(ssi, dtype=dtypes["float"], backend=backend)
    ka_gt4py = from_array(ka, dtype=dtypes["float"], backend=backend)
    dv_gt4py = from_array(dv, dtype=dtypes["float"], backend=backend)
    ai_gt4py = from_array(ai, dtype=dtypes["float"], backend=backend)
    cj_gt4py = from_array(cj, dtype=dtypes["float"], backend=backend)
    rvt_gt4py = from_array(rvt, dtype=dtypes["float"], backend=backend)
    zw_gt4py = from_array(zw, dtype=dtypes["float"], backend=backend)

    ice4_derived_fields_gt4py(
        t=t_gt4py,
        rhodref=rhodref_gt4py,
        pres=pres_gt4py,
        ssi=ssi_gt4py,
        ka=ka_gt4py,
        ai=ai_gt4py,
        cj=cj_gt4py,
        rvt=rvt_gt4py,
        dv=dv,
        zw=zw_gt4py,
        domain=domain,
        origin=(0, 0, 0),
    )

    (
        zzw_out,
        pssi_out,
        zka_out,
        zai_out,
        zdv_out,
        zcj_out,
    ) = fortran_stencil(
        xalpi=externals["ALPI"],
        xbetai=externals["BETAI"],
        xgami=externals["GAMI"],
        xepsilo=externals["EPSILO"],
        xrv=externals["RV"],
        xci=externals["CI"],
        xlstt=externals["LSTT"],
        xcpv=externals["CPV"],
        xp00=externals["P00"],
        xscfac=externals["SCFAC"],
        xtt=externals["TT"],
        zt=t.ravel(),
        prvt=rvt.ravel(),
        ppres=pres.ravel(),
        prhodref=rhodref.ravel(),
        zzw=zw.ravel(),
        pssi=ssi.ravel(),
        zka=ka.ravel(),
        zai=ai.ravel(),
        zdv=dv.ravel(),
        zcj=cj.ravel(),
        **fortran_packed_dims,
    )

    assert_allclose(pssi_out, ssi_gt4py.ravel(), rtol=1e-3)
    assert_allclose(zka_out, ka_gt4py.ravel(), rtol=1e-3)
    assert_allclose(zai_out, ai_gt4py.ravel(), rtol=1e-3)
    assert_allclose(zdv_out, dv_gt4py.ravel(), rtol=1e-3)
    assert_allclose(zcj_out, cj_gt4py.ravel(), rtol=1e-3)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
    ],
)
def test_ice4_slope_parameters(
    gt4py_config, externals, fortran_packed_dims, dtypes, backend, grid, origin
):
    from ice3.stencils.ice4_tendencies import ice4_slope_parameters

    ice4_slope_parameters_gt4py = stencil(
        backend,
        name="ice4_slope_parameters",
        definition=ice4_slope_parameters,
        dtypes=dtypes,
        externals=externals,
    )

    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90", "mode_ice4_tendencies", "ice4_slope_parameters"
    )

    rhodref = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    t = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rrt = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rst = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rgt = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    lbdar = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    lbdar_rf = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    lbdas = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    lbdag = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )

    rhodref_gt4py = from_array(rhodref, dtype=dtypes["float"], backend=backend)
    t_gt4py = from_array(t, dtype=dtypes["float"], backend=backend)
    rrt_gt4py = from_array(rrt, dtype=dtypes["float"], backend=backend)
    rst_gt4py = from_array(rst, dtype=dtypes["float"], backend=backend)
    rgt_gt4py = from_array(rgt, dtype=dtypes["float"], backend=backend)
    lbdar_gt4py = from_array(lbdar, dtype=dtypes["float"], backend=backend)
    lbdar_rf_gt4py = from_array(lbdar_rf, dtype=dtypes["float"], backend=backend)
    lbdas_gt4py = from_array(lbdas, dtype=dtypes["float"], backend=backend)
    lbdag_gt4py = from_array(lbdag, dtype=dtypes["float"], backend=backend)

    ice4_slope_parameters_gt4py(
        rhodref=rhodref_gt4py,
        t=t_gt4py,
        rrt=rrt_gt4py,
        rst=rst_gt4py,
        rgt=rgt_gt4py,
        lbdar=lbdar_gt4py,
        lbdar_rf=lbdar_rf_gt4py,
        lbdas=lbdas_gt4py,
        lbdag=lbdag_gt4py,
    )

    (zlbdar_out, zlbdar_rf_out, zlbdas_out, zlbdag_out) = fortran_stencil(
        xlbr=externals["LBR"],
        xlbexr=externals["LBEXR"],
        xlbg=externals["LBG"],
        xlbdas_min=externals["LBDAS_MIN"],
        xlbdas_max=externals["LBDAS_MAX"],
        xtrans_mp_gammas=externals["TRANS_MP_GAMMAS"],
        xlbs=externals["LBS"],
        xlbexs=externals["LBEXS"],
        xlbexg=externals["LBEXG"],
        r_rtmin=externals["R_RTMIN"],
        s_rtmin=externals["S_RTMIN"],
        g_rtmin=externals["G_RTMIN"],
        lsnow_t=externals["LSNOW_T"],
        prrt=rrt.ravel(),
        prhodref=rhodref.ravel(),
        prst=rst.ravel(),
        prgt=rgt.ravel(),
        zt=t.ravel(),
        zlbdag=lbdag.ravel(),
        zlbdas=lbdas.ravel(),
        zlbdar=lbdar.ravel(),
        zlbdar_rf=lbdar_rf.ravel(),
        **fortran_packed_dims,
    )

    assert_allclose(zlbdar_out, lbdar_gt4py.ravel(), rtol=1e-3)
    assert_allclose(zlbdar_rf_out, lbdar_rf_gt4py.ravel(), rtol=1e-3)
    assert_allclose(zlbdas_out, lbdas_gt4py.ravel(), rtol=1e-3)
    assert_allclose(zlbdag_out, lbdag_gt4py.ravel(), rtol=1e-3)


@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(DEBUG_BACKEND, marks=pytest.mark.debug),
        pytest.param(CPU_BACKEND, marks=pytest.mark.cpu),
        pytest.param(GPU_BACKEND, marks=pytest.mark.gpu),
    ],
)
def test_ice4_total_tendencies_update(
    gt4py_config, externals, fortran_packed_dims, dtypes, backend, domain, origin
):
    from ice3.stencils.ice4_tendencies import ice4_total_tendencies_update

    ice4_total_tendencies_update = stencil(
        backend,
        name="ice4_total_tendencies_update",
        definition=ice4_total_tendencies_update,
        dtypes=dtypes,
        externals=externals,
    )

    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90",
        "mode_ice4_tendencies",
        "ice4_total_tendencies_update",
    )

    lsfact = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    lvfact = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    th_tnd = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rv_tnd = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rc_tnd = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rr_tnd = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    ri_tnd = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rs_tnd = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rg_tnd = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rchoni = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rvdeps = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    riaggs = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    riauts = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rvdepg = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rcautr = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rcaccr = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rrevav = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rcberi = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rsmltg = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rcmltsr = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rraccss = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rraccsg = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rsaccrg = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rcrimss = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rcrimsg = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rsrimcg = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    ricfrrg = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rrcfrig = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    ricfrr = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rcwetg = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    riwetg = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rrwetg = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rswetg = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rcdryg = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    ridryg = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rrdryg = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rsdryg = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rgmltr = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )
    rwetgh = np.array(
        np.random.rand(*domain),
        dtype=dtypes["float"],
        order="F",
    )

    lsfact_gt4py = from_array(lsfact, dtype=dtypes["float"], backend=backend)
    lvfact_gt4py = from_array(lvfact, dtype=dtypes["float"], backend=backend)
    th_tnd_gt4py = from_array(th_tnd, dtype=dtypes["float"], backend=backend)
    rv_tnd_gt4py = from_array(rv_tnd, dtype=dtypes["float"], backend=backend)
    rc_tnd_gt4py = from_array(rc_tnd, dtype=dtypes["float"], backend=backend)
    rr_tnd_gt4py = from_array(rr_tnd, dtype=dtypes["float"], backend=backend)
    ri_tnd_gt4py = from_array(ri_tnd, dtype=dtypes["float"], backend=backend)
    rs_tnd_gt4py = from_array(rs_tnd, dtype=dtypes["float"], backend=backend)
    rg_tnd_gt4py = from_array(rg_tnd, dtype=dtypes["float"], backend=backend)

    rchoni_gt4py = from_array(rchoni, dtype=dtypes["float"], backend=backend)
    rvdeps_gt4py = from_array(rvdeps, dtype=dtypes["float"], backend=backend)
    riaggs_gt4py = from_array(riaggs, dtype=dtypes["float"], backend=backend)
    riauts_gt4py = from_array(riauts, dtype=dtypes["float"], backend=backend)
    rvdepg_gt4py = from_array(rvdepg, dtype=dtypes["float"], backend=backend)
    rcautr_gt4py = from_array(rcautr, dtype=dtypes["float"], backend=backend)
    rcaccr_gt4py = from_array(rcaccr, dtype=dtypes["float"], backend=backend)
    rrevav_gt4py = from_array(rrevav, dtype=dtypes["float"], backend=backend)
    rcberi_gt4py = from_array(rcberi, dtype=dtypes["float"], backend=backend)
    rsmltg_gt4py = from_array(rsmltg, dtype=dtypes["float"], backend=backend)
    rcmltsr_gt4py = from_array(rcmltsr, dtype=dtypes["float"], backend=backend)
    rraccss_gt4py = from_array(rraccss, dtype=dtypes["float"], backend=backend)
    rraccsg_gt4py = from_array(rraccsg, dtype=dtypes["float"], backend=backend)
    rsaccrg_gt4py = from_array(rsaccrg, dtype=dtypes["float"], backend=backend)
    rcrimss_gt4py = from_array(rcrimss, dtype=dtypes["float"], backend=backend)
    rcrimsg_gt4py = from_array(rcrimsg, dtype=dtypes["float"], backend=backend)
    rsrimcg_gt4py = from_array(rsrimcg, dtype=dtypes["float"], backend=backend)
    ricfrrg_gt4py = from_array(ricfrrg, dtype=dtypes["float"], backend=backend)
    rrcfrig_gt4py = from_array(rrcfrig, dtype=dtypes["float"], backend=backend)
    ricfrr_gt4py = from_array(ricfrr, dtype=dtypes["float"], backend=backend)
    rcwetg_gt4py = from_array(rcwetg, dtype=dtypes["float"], backend=backend)
    riwetg_gt4py = from_array(riwetg, dtype=dtypes["float"], backend=backend)
    rrwetg_gt4py = from_array(rrwetg, dtype=dtypes["float"], backend=backend)
    rswetg_gt4py = from_array(rswetg, dtype=dtypes["float"], backend=backend)
    rcdryg_gt4py = from_array(rcdryg, dtype=dtypes["float"], backend=backend)
    ridryg_gt4py = from_array(ridryg, dtype=dtypes["float"], backend=backend)
    rrdryg_gt4py = from_array(rrdryg, dtype=dtypes["float"], backend=backend)
    rsdryg_gt4py = from_array(rsdryg, dtype=dtypes["float"], backend=backend)
    rgmltr_gt4py = from_array(rgmltr, dtype=dtypes["float"], backend=backend)
    rwetgh_gt4py = from_array(rwetgh, dtype=dtypes["float"], backend=backend)

    ice4_total_tendencies_update(
        lsfact=lsfact_gt4py,
        lvfact=lvfact_gt4py,
        th_tnd=th_tnd_gt4py,
        rv_tnd=rv_tnd_gt4py,
        rc_tnd=rc_tnd_gt4py,
        rr_tnd=rr_tnd_gt4py,
        ri_tnd=ri_tnd_gt4py,
        rs_tnd=rs_tnd_gt4py,
        rg_tnd=rg_tnd_gt4py,
        rchoni=rchoni_gt4py,
        rvdeps=rvdeps_gt4py,
        riaggs=riaggs_gt4py,
        riauts=riauts_gt4py,
        rvdepg=rvdepg_gt4py,
        rcautr=rcautr_gt4py,
        rcaccr=rcaccr_gt4py,
        rrevav=rrevav_gt4py,
        rcberi=rcberi_gt4py,
        rsmltg=rsmltg_gt4py,
        rcmltsr=rcmltsr_gt4py,
        rraccss=rraccss_gt4py,
        rraccsg=rraccsg_gt4py,
        rsaccrg=rsaccrg_gt4py,
        rcrimss=rcrimss_gt4py,
        rcrimsg=rcrimsg_gt4py,
        rsrimcg=rsrimcg_gt4py,
        ricfrrg=ricfrrg_gt4py,
        rrcfrig=rrcfrig_gt4py,
        ricfrr=ricfrr_gt4py,
        rcwetg=rcwetg_gt4py,
        riwetg=riwetg_gt4py,
        rrwetg=rrwetg_gt4py,
        rswetg=rswetg_gt4py,
        rcdryg=rcdryg_gt4py,
        ridryg=ridryg_gt4py,
        rrdryg=rrdryg_gt4py,
        rsdryg=rsdryg_gt4py,
        rgmltr=rgmltr_gt4py,
        rwetgh=rwetgh_gt4py,
        domain=domain,
        origin=(0, 0, 0),
    )

    (
        pth_tnd_out,
        prv_tnd_out,
        prc_tnd_out,
        prr_tnd_out,
        pri_tnd_out,
        prs_tnd_out,
        prg_tnd_out,
    ) = fortran_stencil(
        plsfact=lsfact.ravel(),
        plvfact=lvfact.ravel(),
        pth_tnd=th_tnd.ravel(),
        prv_tnd=rv_tnd.ravel(),
        prc_tnd=rc_tnd.ravel(),
        prr_tnd=rr_tnd.ravel(),
        pri_tnd=ri_tnd.ravel(),
        prs_tnd=rs_tnd.ravel(),
        prg_tnd=rg_tnd.ravel(),
        rchoni=rchoni.ravel(),
        rvdeps=rvdeps.ravel(),
        riaggs=riaggs.ravel(),
        riauts=riauts.ravel(),
        rvdepg=rvdepg.ravel(),
        rcautr=rcautr.ravel(),
        rcaccr=rcaccr.ravel(),
        rrevav=rrevav.ravel(),
        rcberi=rcberi.ravel(),
        rsmltg=rsmltg.ravel(),
        rcmltsr=rcmltsr.ravel(),
        rraccss=rraccss.ravel(),
        rraccsg=rraccsg.ravel(),
        rsaccrg=rsaccrg.ravel(),
        rcrimss=rcrimss.ravel(),
        rcrimsg=rcrimsg.ravel(),
        rsrimcg=rsrimcg.ravel(),
        ricfrrg=ricfrrg.ravel(),
        rrcfrig=rrcfrig.ravel(),
        ricfrr=ricfrr.ravel(),
        rcwetg=rcwetg.ravel(),
        riwetg=riwetg.ravel(),
        rrwetg=rrwetg.ravel(),
        rswetg=rswetg.ravel(),
        rcdryg=rcdryg.ravel(),
        ridryg=ridryg.ravel(),
        rrdryg=rrdryg.ravel(),
        rsdryg=rsdryg.ravel(),
        rgmltr=rgmltr.ravel(),
        rwetgh=rwetgh.ravel(),
        **fortran_packed_dims,
    )

    assert_allclose(pth_tnd_out, th_tnd_gt4py.ravel(), rtol=1e-3)
    assert_allclose(prc_tnd_out, rc_tnd_gt4py.ravel(), rtol=1e-3)
    assert_allclose(prr_tnd_out, rr_tnd_gt4py.ravel(), rtol=1e-3)
    assert_allclose(pri_tnd_out, ri_tnd_gt4py.ravel(), rtol=1e-3)
    assert_allclose(prs_tnd_out, rs_tnd_gt4py.ravel(), rtol=1e-3)
    assert_allclose(prg_tnd_out, rg_tnd_gt4py.ravel(), rtol=1e-3)
