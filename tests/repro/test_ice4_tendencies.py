import logging

import numpy as np
import pytest
from conftest import get_backends, compile_fortran_stencil
from gt4py.storage import from_array
from ifs_physics_common.framework.stencil import compile_stencil
from numpy.testing import assert_allclose

from ice3_gt4py.phyex_common.phyex import Phyex


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_nucleation_post_processing(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    # Setting backend and precision
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

    ice4_nucleation_post_processing_gt4py = compile_stencil("ice4_nucleation_post_processing", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90", "mode_ice4_tendencies", "ice4_nucleation_post_processing"
    )

    FloatFieldsIJK_names = [
        "t",
        "exn",
        "lsfact",
        "tht",
        "rvt",
        "rit",
        "rvheni_mr",
    ]

    FloatFieldsIJK = {
        name: np.array(
        np.random.rand(*grid.shape),
        dtype=gt4py_config.dtypes.float,
        order="F",
    ) for name in FloatFieldsIJK_names
    }

    t_gt4py = from_array(FloatFieldsIJK["t"], dtype=gt4py_config.dtypes.float, backend=backend)
    exn_gt4py = from_array(FloatFieldsIJK["exn"], dtype=gt4py_config.dtypes.float, backend=backend)
    lsfact_gt4py = from_array(FloatFieldsIJK["lsfact"], dtype=gt4py_config.dtypes.float, backend=backend)
    tht_gt4py = from_array(FloatFieldsIJK["tht"], dtype=gt4py_config.dtypes.float, backend=backend)
    rvt_gt4py = from_array(FloatFieldsIJK["rvt"], dtype=gt4py_config.dtypes.float, backend=backend)
    rit_gt4py = from_array(FloatFieldsIJK["rit"], dtype=gt4py_config.dtypes.float, backend=backend)
    rvheni_mr_gt4py = from_array(FloatFieldsIJK["rvheni_mr"], dtype=gt4py_config.dtypes.float, backend=backend)

    ice4_nucleation_post_processing_gt4py(
        t=t_gt4py,
        exn=exn_gt4py,
        lsfact=lsfact_gt4py,
        tht=tht_gt4py,
        rvt=rvt_gt4py,
        rit=rit_gt4py,
        rvheni_mr=rvheni_mr_gt4py,
        domain=grid.shape,
        origin=(0, 0, 0)
    )

    f2py_mapping = {
        "plsfact": "lsfact",
        "pexn": "exn",
        "ptht": "tht",
        "prit": "rit",
        "prvt": "rvt",
        "zt": "t",
        "prvheni_mr": "rvheni_mr"
    }

    fortran_FloatFieldsIJK = {
        name: FloatFieldsIJK[value].ravel()
        for name, value in f2py_mapping.items()
    }

    result = fortran_stencil(
        **fortran_FloatFieldsIJK,
        **fortran_packed_dims
    )

    tht_out = result[0]
    rvt_out = result[1]
    rit_out = result[2]
    zt_out = result[3]

    logging.info(f"Mean tht_gt4py {tht_gt4py.mean()}")
    logging.info(f"Mean   tht_out {tht_out.mean()}")
    logging.info(
        f"Max abs rtol {max(abs(tht_gt4py.ravel() - tht_out) / abs(tht_out))}"
    )

    logging.info(f"Mean rit_gt4py {rit_gt4py.mean()}")
    logging.info(f"Mean   rit_out {rit_out.mean()}")
    logging.info(
        f"Max abs rtol {max(abs(rit_gt4py.ravel() - rit_out) / abs(rit_out))}"
    )

    logging.info(f"Mean rvt_gt4py {rvt_gt4py.mean()}")
    logging.info(f"Mean   rvt_out {rvt_out.mean()}")
    logging.info(
        f"Max abs rtol {max(abs(rvt_gt4py.ravel() - rvt_out) / abs(rvt_out))}"
    )

    logging.info(f"Mean t_gt4py {t_gt4py.mean()}")
    logging.info(f"Mean  zt_out {zt_out.mean()}")
    logging.info(f"Max abs rtol {max(abs(t_gt4py.ravel() - zt_out) / abs(zt_out))}")

    logging.info(f"Machine precision {np.finfo(gt4py_config.dtypes.float).eps}")

    assert_allclose(tht_out, tht_gt4py.ravel(), rtol=1e-3)
    assert_allclose(rit_out, rit_gt4py.ravel(), rtol=1e-3)
    assert_allclose(zt_out, t_gt4py.ravel(), rtol=1e-3)
    assert_allclose(rvt_out, rvt_gt4py.ravel(), rtol=1e-3)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_rrhong_post_processing(gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin):

        # Setting backend and precision
        gt4py_config.backend = backend
        gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

        ice4_rrhong_post_processing_gt4py = compile_stencil("ice4_rrhong_post_processing", gt4py_config, externals)
        fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90", "mode_ice4_tendencies", "ice4_rrhong_post_processing"
        )

        FloatFieldsIJK_names = [
            "t",
            "exn",
            "lsfact",
            "lvfact",
            "tht",
            "rrt",
            "rgt",
            "rrhong_mr"
        ]

        FloatFieldsIJK =  {
            name: np.array(
                np.random.rand(*grid),
                dtype=gt4py_config.dtypes.float,
                order="F",
            ) for name in FloatFieldsIJK_names
        }

        t_gt4py = from_array(FloatFieldsIJK["t"], dtype=gt4py_config.dtypes.float, backend=backend)
        exn_gt4py = from_array(FloatFieldsIJK["exn"], dtype=gt4py_config.dtypes.float, backend=backend)
        lsfact_gt4py = from_array(FloatFieldsIJK["lsfact"], dtype=gt4py_config.dtypes.float, backend=backend)
        lvfact_gt4py = from_array(FloatFieldsIJK["lvfact"], dtype=gt4py_config.dtypes.float, backend=backend)
        tht_gt4py = from_array(FloatFieldsIJK["tht"], dtype=gt4py_config.dtypes.float, backend=backend)
        rrt_gt4py = from_array(FloatFieldsIJK["rrt"], dtype=gt4py_config.dtypes.float, backend=backend)
        rgt_gt4py = from_array(FloatFieldsIJK["rgt"], dtype=gt4py_config.dtypes.float, backend=backend)
        rrhong_mr_gt4py = from_array(FloatFieldsIJK["rrhong_mr"], dtype=gt4py_config.dtypes.float, backend=backend)

        ice4_rrhong_post_processing_gt4py(
            t=t_gt4py,
            exn=exn_gt4py,
            lsfact=lsfact_gt4py,
            lvfact=lvfact_gt4py,
            tht=tht_gt4py,
            rrt=rrt_gt4py,
            rgt=rgt_gt4py,
            rrhong_mr=rrhong_mr_gt4py,
            domain=grid.shape,
            origin=(0, 0, 0)
        )

        f2py_mapping = {
            "p"+name: name for name in [
                "lsfact",
                "lvfact",
                "exn",
                "rrhong_mr",
                "tht",
                "t",
                "rrt",
                "rgt"
            ]
        }

        fortran_FloatFieldsIJK = {
            name: FloatFieldsIJK[value].ravel()
            for name, value in f2py_mapping.items()
        }

        result = fortran_stencil(
            **fortran_FloatFieldsIJK,
            **fortran_packed_dims
        )

        tht_out = result[0]
        t_out = result[1]
        rrt_out = result[2]
        rgt_out = result[3]

        logging.info(f"Mean tht_gt4py {tht_gt4py.mean()}")
        logging.info(f"Mean tht_out {tht_out.mean()}")
        logging.info(
            f"Max abs rtol {max(abs(tht_gt4py.ravel() - tht_out) / abs(tht_out))}"
        )

        logging.info(f"Mean t_gt4py {t_gt4py.mean()}")
        logging.info(f"Mean t_out {t_out.mean()}")
        logging.info(f"Max abs rtol {max(abs(t_gt4py.ravel() - t_out) / abs(t_out))}")

        logging.info(f"Mean rrt_gt4py {rrt_gt4py.mean()}")
        logging.info(f"Mean rrt_out {rrt_out.mean()}")
        logging.info(
            f"Max abs rtol {max(abs(rrt_gt4py.ravel() - rrt_out) / abs(rrt_out))}"
        )

        logging.info(f"Mean rgt_gt4py {rgt_gt4py.mean()}")
        logging.info(f"Mean rgt_out {rgt_out.mean()}")
        logging.info(
            f"Max abs rtol {max(abs(rgt_gt4py.ravel() - rgt_out) / abs(rgt_out))}"
        )

        logging.info(f"Machine precision {np.finfo(gt4py_config.dtypes.float).eps}")

        assert_allclose(tht_out, tht_gt4py.ravel(), rtol=1e-3)
        assert_allclose(t_out, t_gt4py.ravel(), rtol=1e-3)
        assert_allclose(rrt_out, rrt_gt4py.ravel(), rtol=1e-3)
        assert_allclose(rgt_out, rgt_gt4py.ravel(), rtol=1e-3)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_rimltc_post_processing(gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin):
        # Setting backend and precision
        gt4py_config.backend = backend
        gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

        ice4_rimltc_post_processing_gt4py = compile_stencil("ice4_rimltc_post_processing", gt4py_config, externals)
        fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90", "mode_ice4_tendencies", "ice4_rimltc_post_processing"
        )

        FloatFieldsIJK_names = [
            "t",
            "exn",
            "lsfact",
            "lvfact",
            "rimltc_mr",
            "tht",
            "rct",
            "rit",
        ]

        FloatFieldsIJK = {
            name: np.array(
            np.random.rand(*grid),
            dtype=gt4py_config.dtypes.float,
            order="F",
        ) for name in FloatFieldsIJK_names
        }

        t_gt4py = from_array(FloatFieldsIJK["t"], dtype=gt4py_config.dtypes.float, backend=backend)
        exn_gt4py = from_array(FloatFieldsIJK["exn"], dtype=gt4py_config.dtypes.float, backend=backend)
        lsfact_gt4py = from_array(FloatFieldsIJK["lsfact"], dtype=gt4py_config.dtypes.float, backend=backend)
        lvfact_gt4py = from_array(FloatFieldsIJK["lvfact"], dtype=gt4py_config.dtypes.float, backend=backend)
        tht_gt4py = from_array(FloatFieldsIJK["tht"], dtype=gt4py_config.dtypes.float, backend=backend)
        rimltc_mr_gt4py = from_array(FloatFieldsIJK["rimltc_mr"], dtype=gt4py_config.dtypes.float, backend=backend)
        rct_gt4py = from_array(FloatFieldsIJK["rct"], dtype=gt4py_config.dtypes.float, backend=backend)
        rit_gt4py = from_array(FloatFieldsIJK["rit"], dtype=gt4py_config.dtypes.float, backend=backend)

        ice4_rimltc_post_processing_gt4py(
            t=t_gt4py,
            exn=exn_gt4py,
            lsfact=lsfact_gt4py,
            lvfact=lvfact_gt4py,
            rimltc_mr=rimltc_mr_gt4py,
            tht=tht_gt4py,
            rct=rct_gt4py,
            rit=rit_gt4py,
            domain=grid.shape,
            origin=(0, 0, 0)
        )

        f2py_mapping = {
            "p"+name: name for name in [
                "lsfact",
                "lvfact",
                "exn",
                "rimltc_mr",
                "tht",
                "t",
                "rit",
                "rct",
            ]
        }

        fortran_FloatFieldsIJK = {
            name: FloatFieldsIJK[value].ravel()
            for name, value in f2py_mapping.items()
        }

        result = fortran_stencil(
            **fortran_FloatFieldsIJK,
            **fortran_packed_dims
        )

        tht_out = result[0]
        t_out = result[1]
        rct_out = result[2]
        rit_out = result[3]

        logging.info(f"Mean tht_gt4py {tht_gt4py.mean()}")
        logging.info(f"Mean   tht_out {tht_out.mean()}")

        logging.info(f"Mean t_gt4py {t_gt4py.mean()}")
        logging.info(f"Mean   t_out {t_out.mean()}")

        logging.info(f"Mean rct_gt4py {rct_gt4py.mean()}")
        logging.info(f"Mean   rct_out {rct_out.mean()}")

        logging.info(f"Mean rit_gt4py {rit_gt4py.mean()}")
        logging.info(f"Mean   rit_out {rit_out.mean()}")

        logging.info(f"Machine precision {np.finfo(gt4py_config.dtypes.float).eps}")

        assert_allclose(tht_out, tht_gt4py.ravel(), rtol=1e-3)
        assert_allclose(t_out, t_gt4py.ravel(), rtol=1e-3)
        assert_allclose(rct_out, rct_gt4py.ravel(), rtol=1e-3)
        assert_allclose(rit_out, rit_gt4py.ravel(), rtol=1e-3)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_fast_rg_pre_processing(gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin):
        # Setting backend and precision
        gt4py_config.backend = backend
        gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

        fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90", "mode_ice4_tendencies", "ice4_fast_rg_pre_processing"
        )

        ice4_fast_rg_pre_processing_gt4py = compile_stencil(
            "ice4_fast_rg_pre_processing", gt4py_config, externals
        )



        rgsi = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rgsi_mr = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rvdepg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rsmltg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rraccsg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rsaccrg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rcrimsg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rsrimcg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rrhong_mr = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rsrimcg_mr = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )

        rgsi_gt4py = from_array(rgsi, dtype=gt4py_config.dtypes.float, backend=backend)
        rgsi_mr_gt4py = from_array(rgsi_mr, dtype=gt4py_config.dtypes.float, backend=backend)
        rvdepg_gt4py = from_array(rvdepg, dtype=gt4py_config.dtypes.float, backend=backend)
        rsmltg_gt4py = from_array(rsmltg, dtype=gt4py_config.dtypes.float, backend=backend)
        rraccsg_gt4py = from_array(rraccsg, dtype=gt4py_config.dtypes.float, backend=backend)
        rsaccrg_gt4py = from_array(rsaccrg, dtype=gt4py_config.dtypes.float, backend=backend)
        rcrimsg_gt4py = from_array(rcrimsg, dtype=gt4py_config.dtypes.float, backend=backend)
        rsrimcg_gt4py = from_array(rsrimcg, dtype=gt4py_config.dtypes.float, backend=backend)
        rrhong_mr_gt4py = from_array(rrhong_mr, dtype=gt4py_config.dtypes.float, backend=backend)
        rsrimcg_mr_gt4py = from_array(rsrimcg_mr, dtype=gt4py_config.dtypes.float, backend=backend)

        ice4_fast_rg_pre_processing_gt4py(
            rgsi=rgsi_gt4py,
            rgsi_mr=rgsi_mr_gt4py,
            rvdepg=rvdepg_gt4py,
            rsmltg=rsmltg_gt4py,
            rraccsg=rraccsg_gt4py,
            rsaccrg=rsaccrg_gt4py,
            rcrimsg=rcrimsg_gt4py,
            rsrimcg=rsrimcg_gt4py,
            rrhong_mr=rrhong_mr_gt4py,
            rsrimcg_mr=rsrimcg_mr_gt4py,
            domain=grid.shape,
            origin=(0, 0, 0)
        )

        result = fortran_stencil(
            rvdepg=rvdepg,
            rsmltg=rsmltg,
            rraccsg=rraccsg,
            rsaccrg=rsaccrg,
            rcrimsg=rcrimsg,
            rsrimcg=rsrimcg,
            rrhong_mr=rrhong_mr,
            rsrimcg_mr=rsrimcg,
            zgrsi=rgsi,
            zrgsi_mr=rgsi_mr,
            **fortran_packed_dims
        )

        zrgsi_out = result[0]
        zrgsi_mr_out = result[1]

        logging.info(f"Mean zrgsi_gt4py {rgsi_gt4py.mean()}")
        logging.info(f"Mean zrgsi_out   {zrgsi_out.mean()}")
        logging.info(
            f"Max abs rtol     {max(abs(rgsi_gt4py.ravel() - zrgsi_out) / abs(zrgsi_out))}"
        )

        logging.info(f"Mean rgsi_mr_gt4py   {rgsi_gt4py.mean()}")
        logging.info(f"Mean rgsi_mr_out     {zrgsi_mr_out.mean()}")
        logging.info(
            f"Max abs rtol {max(abs(rgsi_mr_gt4py.ravel() - zrgsi_mr_out) / abs(zrgsi_mr_out))}"
        )

        logging.info(f"Machine precision {np.finfo(gt4py_config.dtypes.float).eps}")

        assert_allclose(zrgsi_out, rgsi_gt4py.ravel(), rtol=1e-3)
        assert_allclose(zrgsi_mr_out, rgsi_mr_gt4py.ravel(), rtol=1e-3)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_increment_update(gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin):
        # Setting backend and precision
        gt4py_config.backend = backend
        gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

        fortran_stencil = compile_fortran_stencil(
            "mode_ice4_tendencies.F90", "mode_ice4_tendencies", "ice4_increment_update"
        )
  
        ice4_increment_update_gt4py = compile_stencil(
            "ice4_increment_update", gt4py_config, externals
        )

        lsfact = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )

        lvfact = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        theta_increment = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rv_increment = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rc_increment = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rr_increment = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        ri_increment = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rs_increment = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rg_increment = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rvheni_mr = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rimltc_mr = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rrhong_mr = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rsrimcg_mr = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )

        lsfact_gt4py = from_array(lsfact, dtype=gt4py_config.dtypes.float, backend=backend)
        lvfact_gt4py = from_array(lvfact, dtype=gt4py_config.dtypes.float, backend=backend)
        theta_increment_gt4py = from_array(
            theta_increment, dtype=gt4py_config.dtypes.float, backend=backend
        )
        rv_increment_gt4py = from_array(rv_increment, dtype=gt4py_config.dtypes.float, backend=backend)
        rc_increment_gt4py = from_array(rc_increment, dtype=gt4py_config.dtypes.float, backend=backend)
        rr_increment_gt4py = from_array(rr_increment, dtype=gt4py_config.dtypes.float, backend=backend)
        ri_increment_gt4py = from_array(ri_increment, dtype=gt4py_config.dtypes.float, backend=backend)
        rs_increment_gt4py = from_array(rs_increment, dtype=gt4py_config.dtypes.float, backend=backend)
        rg_increment_gt4py = from_array(rg_increment, dtype=gt4py_config.dtypes.float, backend=backend)
        rvheni_mr_gt4py = from_array(rvheni_mr, dtype=gt4py_config.dtypes.float, backend=backend)
        rimltc_mr_gt4py = from_array(rimltc_mr, dtype=gt4py_config.dtypes.float, backend=backend)
        rsrimcg_mr_gt4py = from_array(rsrimcg_mr, dtype=gt4py_config.dtypes.float, backend=backend)
        rrhong_mr_gt4py = from_array(rrhong_mr, dtype=gt4py_config.dtypes.float, backend=backend)

        ice4_increment_update_gt4py(
            lsfact=lsfact_gt4py,
            lvfact=lvfact_gt4py,
            theta_increment=theta_increment_gt4py,
            rv_increment=rv_increment_gt4py,
            rc_increment=rc_increment_gt4py,
            rr_increment=rr_increment_gt4py,
            ri_increment=ri_increment_gt4py,
            rs_increment=rs_increment_gt4py,
            rg_increment=rg_increment_gt4py,
            rimltc_mr=rimltc_mr_gt4py,
            rrhong_mr=rrhong_mr_gt4py,
            rsrimcg_mr=rsrimcg_mr_gt4py,
            rvheni_mr=rvheni_mr_gt4py,
            domain=grid.shape,
            origin=(0, 0, 0)
        )

      
        result = fortran_stencil(
            plsfact=lsfact.ravel(),
            plvfact=lvfact.ravel(),
            prvheni_mr=rvheni_mr.ravel(),
            primltc_mr=rimltc_mr.ravel(),
            prrhong_mr=rrhong_mr.ravel(),
            prsrimcg_mr=rsrimcg_mr.ravel(),
            pth_inst=theta_increment.ravel(),
            prv_inst=rv_increment.ravel(),
            prc_inst=rc_increment.ravel(),
            prr_inst=rr_increment.ravel(),
            pri_inst=ri_increment.ravel(),
            prs_inst=rs_increment.ravel(),
            prg_inst=rg_increment.ravel(),
            **fortran_packed_dims
        )

        pth_inst_out = result[0]
        prv_inst_out = result[1]
        prc_inst_out = result[2]
        prr_inst_out = result[3]
        pri_inst_out = result[4]
        prs_inst_out = result[5]
        prg_inst_out = result[6]

        logging.info(f"Mean tht_incr_gt4py {theta_increment_gt4py.mean()}")
        logging.info(f"Mean   tht_inst_out {pth_inst_out.mean()}")

        logging.info(f"Mean rv_incr_gt4py {rv_increment_gt4py.mean()}")
        logging.info(f"Mean   rv_inst_out {prv_inst_out.mean()}")

        logging.info(f"Mean rct_incr_gt4py {rc_increment_gt4py.mean()}")
        logging.info(f"Mean   rct_inst_out {prc_inst_out.mean()}")

        logging.info(f"Mean rrt_incr_gt4py {rr_increment_gt4py.mean()}")
        logging.info(f"Mean   rrt_inst_out {prr_inst_out.mean()}")

        logging.info(f"Mean rit_incr_gt4py {ri_increment_gt4py.mean()}")
        logging.info(f"Mean   rit_inst_out {pri_inst_out.mean()}")

        logging.info(f"Mean rst_incr_gt4py {rs_increment_gt4py.mean()}")
        logging.info(f"Mean   rst_inst_out {prs_inst_out.mean()}")

        logging.info(f"Mean rit_incr_gt4py {rg_increment_gt4py.mean()}")
        logging.info(f"Mean   rit_inst_out {prg_inst_out.mean()}")

        logging.info(f"Machine precision {np.finfo(gt4py_config.dtypes.float).eps}")

        assert_allclose(pth_inst_out, theta_increment_gt4py.ravel(), rtol=1e-3)
        assert_allclose(prv_inst_out, rv_increment_gt4py.ravel(), rtol=1e-3)
        assert_allclose(prc_inst_out, rc_increment_gt4py.ravel(), rtol=1e-3)
        assert_allclose(pri_inst_out, ri_increment_gt4py.ravel(), rtol=1e-3)
        assert_allclose(prr_inst_out, rr_increment_gt4py.ravel(), rtol=1e-3)
        assert_allclose(prs_inst_out, rs_increment_gt4py.ravel(), rtol=1e-3)
        assert_allclose(prg_inst_out, rg_increment_gt4py.ravel(), rtol=1e-3)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_derived_fields(gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin):
        # Setting backend and precision
        gt4py_config.backend = backend
        gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

        fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90", "mode_ice4_tendencies", "ice4_derived_fields"
        )

        ice4_derived_fields_gt4py = compile_stencil(
            "ice4_derived_fields",
            gt4py_config,
            externals,
        )

        t = 300 * np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rhodref = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        pres = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        ssi = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        ka = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        dv = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        ai = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        cj = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rvt = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        zw = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )

        t_gt4py = from_array(t, dtype=gt4py_config.dtypes.float, backend=backend)
        rhodref_gt4py = from_array(rhodref, dtype=gt4py_config.dtypes.float, backend=backend)
        pres_gt4py = from_array(pres, dtype=gt4py_config.dtypes.float, backend=backend)
        ssi_gt4py = from_array(ssi, dtype=gt4py_config.dtypes.float, backend=backend)
        ka_gt4py = from_array(ka, dtype=gt4py_config.dtypes.float, backend=backend)
        dv_gt4py = from_array(dv, dtype=gt4py_config.dtypes.float, backend=backend)
        ai_gt4py = from_array(ai, dtype=gt4py_config.dtypes.float, backend=backend)
        cj_gt4py = from_array(cj, dtype=gt4py_config.dtypes.float, backend=backend)
        rvt_gt4py = from_array(rvt, dtype=gt4py_config.dtypes.float, backend=backend)
        zw_gt4py = from_array(zw, dtype=gt4py_config.dtypes.float, backend=backend)

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
            domain=grid.shape,
            origin=(0, 0, 0)
        )


        result = fortran_stencil(
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
            **fortran_packed_dims
        )

        zzw_out = result[0]
        pssi_out = result[1]
        zka_out = result[2]
        zai_out = result[3]
        zdv_out = result[4]
        zcj_out = result[5]

        logging.info(f"Mean zw_gt4py {zw_gt4py.mean()}")
        logging.info(f"Mean  zzw_out {zzw_out.mean()}")
        logging.info(
            f"Max rtol err zw {max(abs(zw_gt4py.ravel() - zzw_out) / abs(zzw_out))}"
        )

        logging.info(f"Mean ssi_gt4py {ssi_gt4py.mean()}")
        logging.info(f"Mean  pssi_out {pssi_out.mean()}")
        logging.info(
            f"Max rtol err ssi {max(abs(ssi_gt4py.ravel() - pssi_out) / abs(pssi_out))}"
        )

        logging.info(f"Mean ka_gt4py {ka_gt4py.mean()}")
        logging.info(f"Mean  zka_out {zka_out.mean()}")
        logging.info(
            f"Max rtol err ka {max(abs(ka_gt4py.ravel() - zka_out) / abs(zka_out))}"
        )

        logging.info(f"Mean ai_gt4py {ai_gt4py.mean()}")
        logging.info(f"Mean  zai_out {zai_out.mean()}")
        logging.info(
            f"Max rtol err ai {max(abs(ai_gt4py.ravel() - zai_out) / abs(zai_out))}"
        )

        logging.info(f"Mean dv_gt4py {dv_gt4py.mean()}")
        logging.info(f"Mean  zdv_out {zdv_out.mean()}")
        logging.info(
            f"Max rtol err dv {max(abs(dv_gt4py.ravel() - zdv_out) / abs(zdv_out))}"
        )

        logging.info(f"Mean cj_gt4py {cj_gt4py.mean()}")
        logging.info(f"Mean  zcj_out {zcj_out.mean()}")
        logging.info(
            f"Max rtol err cj {max(abs(cj_gt4py.ravel() - zcj_out) / abs(zcj_out))}"
        )

        logging.info(f"Machine precision {np.finfo(gt4py_config.dtypes.float).eps}")

        assert_allclose(pssi_out, ssi_gt4py.ravel(), rtol=1e-3)
        assert_allclose(zka_out, ka_gt4py.ravel(), rtol=1e-3)
        assert_allclose(zai_out, ai_gt4py.ravel(), rtol=1e-3)
        assert_allclose(zdv_out, dv_gt4py.ravel(), rtol=1e-3)
        assert_allclose(zcj_out, cj_gt4py.ravel(), rtol=1e-3)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_slope_parameters(gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin):
        # Setting backend and precision
        gt4py_config.backend = backend
        gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

        ice4_slope_parameters_gt4py = compile_stencil("ice4_slope_parameters", gt4py_config, externals)
        fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90", "mode_ice4_tendencies", "ice4_slope_parameters"
        )

        rhodref = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        t = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rrt = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rst = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rgt = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        lbdar = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        lbdar_rf = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        lbdas = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        lbdag = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )

        rhodref_gt4py = from_array(rhodref, dtype=gt4py_config.dtypes.float, backend=backend)
        t_gt4py = from_array(t, dtype=gt4py_config.dtypes.float, backend=backend)
        rrt_gt4py = from_array(rrt, dtype=gt4py_config.dtypes.float, backend=backend)
        rst_gt4py = from_array(rst, dtype=gt4py_config.dtypes.float, backend=backend)
        rgt_gt4py = from_array(rgt, dtype=gt4py_config.dtypes.float, backend=backend)
        lbdar_gt4py = from_array(lbdar, dtype=gt4py_config.dtypes.float, backend=backend)
        lbdar_rf_gt4py = from_array(lbdar_rf, dtype=gt4py_config.dtypes.float, backend=backend)
        lbdas_gt4py = from_array(lbdas, dtype=gt4py_config.dtypes.float, backend=backend)
        lbdag_gt4py = from_array(lbdag, dtype=gt4py_config.dtypes.float, backend=backend)

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


        result = fortran_stencil(
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
            **fortran_packed_dims
        )

        zlbdar_out = result[0]
        zlbdar_rf_out = result[1]
        zlbdas_out = result[2]
        zlbdag_out = result[3]

        logging.info(f"Mean lbdar_gt4py {lbdar_gt4py.mean()}")
        logging.info(f"Mean zlbdar_out {zlbdar_out.mean()}")
        logging.info(
            f"Max rtol err ssi {max(abs(lbdar_gt4py.ravel() - zlbdar_out) / abs(zlbdar_out))}"
        )

        logging.info(f"Mean lbdar_rf_gt4py {lbdar_rf_gt4py.mean()}")
        logging.info(f"Mean zlbdar_rf_out {zlbdar_rf_out.mean()}")
        logging.info(
            f"Max abs err ssi {max(abs(lbdar_rf_gt4py.ravel() - zlbdar_rf_out) / abs(zlbdar_out))}"
        )

        logging.info(f"Mean lbdas_gt4py {lbdas_gt4py.mean()}")
        logging.info(f"Mean lbdas_out {zlbdas_out.mean()}")
        logging.info(
            f"Max abs err ssi {max(abs(lbdas_gt4py.ravel() - zlbdas_out) / abs(zlbdas_out))}"
        )

        logging.info(f"Mean lbdag_gt4py {lbdag_gt4py.mean()}")
        logging.info(f"Mean lbdag_out {zlbdag_out.mean()}")
        logging.info(
            f"Max abs err ssi {max(abs(lbdag_gt4py.ravel() - zlbdag_out) / abs(zlbdag_out))}"
        )

        logging.info(f"Machine precision {np.finfo(gt4py_config.dtypes.float).eps}")

        assert_allclose(zlbdar_out, lbdar_gt4py.ravel(), rtol=1e-3)
        assert_allclose(zlbdar_rf_out, lbdar_rf_gt4py.ravel(), rtol=1e-3)
        assert_allclose(zlbdas_out, lbdas_gt4py.ravel(), rtol=1e-3)
        assert_allclose(zlbdag_out, lbdag_gt4py.ravel(), rtol=1e-3)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_ice4_total_tendencies_update(gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin):
        # Setting backend and precision
        gt4py_config.backend = backend
        gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)

        fortran_stencil = compile_fortran_stencil(
            "mode_ice4_tendencies.F90", "mode_ice4_tendencies", "ice4_total_tendencies_update"
        )
        ice4_total_tendencies_update = compile_stencil(
            "ice4_total_tendencies_update", gt4py_config, externals
        )

        lsfact = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        lvfact = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        th_tnd = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rv_tnd = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rc_tnd = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rr_tnd = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        ri_tnd = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rs_tnd = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rg_tnd = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rchoni = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rvdeps = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        riaggs = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        riauts = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rvdepg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rcautr = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rcaccr = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rrevav = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rcberi = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rsmltg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rcmltsr = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rraccss = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rraccsg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rsaccrg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rcrimss = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rcrimsg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rsrimcg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        ricfrrg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rrcfrig = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        ricfrr = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rcwetg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        riwetg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rrwetg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rswetg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rcdryg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        ridryg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rrdryg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rsdryg = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rgmltr = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )
        rwetgh = np.array(
            np.random.rand(*grid.shape),
            dtype=gt4py_config.dtypes.float,
            order="F",
        )

        lsfact_gt4py = from_array(lsfact, dtype=gt4py_config.dtypes.float, backend=backend)
        lvfact_gt4py = from_array(lvfact, dtype=gt4py_config.dtypes.float, backend=backend)
        th_tnd_gt4py = from_array(th_tnd, dtype=gt4py_config.dtypes.float, backend=backend)
        rv_tnd_gt4py = from_array(rv_tnd, dtype=gt4py_config.dtypes.float, backend=backend)
        rc_tnd_gt4py = from_array(rc_tnd, dtype=gt4py_config.dtypes.float, backend=backend)
        rr_tnd_gt4py = from_array(rr_tnd, dtype=gt4py_config.dtypes.float, backend=backend)
        ri_tnd_gt4py = from_array(ri_tnd, dtype=gt4py_config.dtypes.float, backend=backend)
        rs_tnd_gt4py = from_array(rs_tnd, dtype=gt4py_config.dtypes.float, backend=backend)
        rg_tnd_gt4py = from_array(rg_tnd, dtype=gt4py_config.dtypes.float, backend=backend)

        rchoni_gt4py = from_array(rchoni, dtype=gt4py_config.dtypes.float, backend=backend)
        rvdeps_gt4py = from_array(rvdeps, dtype=gt4py_config.dtypes.float, backend=backend)
        riaggs_gt4py = from_array(riaggs, dtype=gt4py_config.dtypes.float, backend=backend)
        riauts_gt4py = from_array(riauts, dtype=gt4py_config.dtypes.float, backend=backend)
        rvdepg_gt4py = from_array(rvdepg, dtype=gt4py_config.dtypes.float, backend=backend)
        rcautr_gt4py = from_array(rcautr, dtype=gt4py_config.dtypes.float, backend=backend)
        rcaccr_gt4py = from_array(rcaccr, dtype=gt4py_config.dtypes.float, backend=backend)
        rrevav_gt4py = from_array(rrevav, dtype=gt4py_config.dtypes.float, backend=backend)
        rcberi_gt4py = from_array(rcberi, dtype=gt4py_config.dtypes.float, backend=backend)
        rsmltg_gt4py = from_array(rsmltg, dtype=gt4py_config.dtypes.float, backend=backend)
        rcmltsr_gt4py = from_array(rcmltsr, dtype=gt4py_config.dtypes.float, backend=backend)
        rraccss_gt4py = from_array(rraccss, dtype=gt4py_config.dtypes.float, backend=backend)
        rraccsg_gt4py = from_array(rraccsg, dtype=gt4py_config.dtypes.float, backend=backend)
        rsaccrg_gt4py = from_array(rsaccrg, dtype=gt4py_config.dtypes.float, backend=backend)
        rcrimss_gt4py = from_array(rcrimss, dtype=gt4py_config.dtypes.float, backend=backend)
        rcrimsg_gt4py = from_array(rcrimsg, dtype=gt4py_config.dtypes.float, backend=backend)
        rsrimcg_gt4py = from_array(rsrimcg, dtype=gt4py_config.dtypes.float, backend=backend)
        ricfrrg_gt4py = from_array(ricfrrg, dtype=gt4py_config.dtypes.float, backend=backend)
        rrcfrig_gt4py = from_array(rrcfrig, dtype=gt4py_config.dtypes.float, backend=backend)
        ricfrr_gt4py = from_array(ricfrr, dtype=gt4py_config.dtypes.float, backend=backend)
        rcwetg_gt4py = from_array(rcwetg, dtype=gt4py_config.dtypes.float, backend=backend)
        riwetg_gt4py = from_array(riwetg, dtype=gt4py_config.dtypes.float, backend=backend)
        rrwetg_gt4py = from_array(rrwetg, dtype=gt4py_config.dtypes.float, backend=backend)
        rswetg_gt4py = from_array(rswetg, dtype=gt4py_config.dtypes.float, backend=backend)
        rcdryg_gt4py = from_array(rcdryg, dtype=gt4py_config.dtypes.float, backend=backend)
        ridryg_gt4py = from_array(ridryg, dtype=gt4py_config.dtypes.float, backend=backend)
        rrdryg_gt4py = from_array(rrdryg, dtype=gt4py_config.dtypes.float, backend=backend)
        rsdryg_gt4py = from_array(rsdryg, dtype=gt4py_config.dtypes.float, backend=backend)
        rgmltr_gt4py = from_array(rgmltr, dtype=gt4py_config.dtypes.float, backend=backend)
        rwetgh_gt4py = from_array(rwetgh, dtype=gt4py_config.dtypes.float, backend=backend)

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
            domain=grid.shape,
            origin=(0, 0, 0)
        )



        result = fortran_stencil(
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
            **fortran_packed_dims
        )

        pth_tnd_out = result[0]
        prv_tnd_out = result[1]
        prc_tnd_out = result[2]
        prr_tnd_out = result[3]
        pri_tnd_out = result[4]
        prs_tnd_out = result[5]
        prg_tnd_out = result[6]

        logging.info("Test ice4 total tendencies")
        logging.info(f"Mean th_tnd_gt4py {th_tnd_gt4py.mean()}")
        logging.info(f"Mean th_tnd_out {pth_tnd_out.mean()}")
        logging.info(
            f"Max rtol err th_tnd {max(abs(th_tnd_gt4py.ravel() - pth_tnd_out) / abs(pth_tnd_out))}"
        )

        logging.info(f"Mean rv_tnd_gt4py {rv_tnd_gt4py.mean()}")
        logging.info(f"Mean rv_tnd_out {prv_tnd_out.mean()}")
        logging.info(
            f"Max rtol err rv_tnd {max(abs(rv_tnd_gt4py.ravel() - prv_tnd_out) / abs(prv_tnd_out))}"
        )

        logging.info(f"Mean rc_tnd_gt4py {rc_tnd_gt4py.mean()}")
        logging.info(f"Mean prc_tnd_out {prc_tnd_out.mean()}")
        logging.info(
            f"Max rtol err rc_tnd {max(abs(rc_tnd_gt4py.ravel() - prc_tnd_out) / abs(prc_tnd_out))}"
        )

        logging.info(f"Mean rr_tnd_gt4py {rr_tnd_gt4py.mean()}")
        logging.info(f"Mean prr_tnd_out {prr_tnd_out.mean()}")
        logging.info(
            f"Max rtol err rr {max(abs(rr_tnd_gt4py.ravel() - prr_tnd_out) / abs(prr_tnd_out))}"
        )

        logging.info(f"Mean ri_tnd_gt4py {ri_tnd_gt4py.mean()}")
        logging.info(f"Mean pri_tnd_out {pri_tnd_out.mean()}")
        logging.info(
            f"Max rtol err ri_tnd {max(abs(ri_tnd_gt4py.ravel() - pri_tnd_out) / abs(pri_tnd_out))}"
        )

        logging.info(f"Mean rs_tnd_gt4py {rs_tnd_gt4py.mean()}")
        logging.info(f"Mean prs_tnd_out {prs_tnd_out.mean()}")
        logging.info(
            f"Max rtol err rs_tnd {max(abs(rs_tnd_gt4py.ravel() - prs_tnd_out) / abs(prs_tnd_out))}"
        )

        logging.info(f"Mean rg_tnd_gt4py {rg_tnd_gt4py.mean()}")
        logging.info(f"Mean prg_tnd_out {prg_tnd_out.mean()}")
        logging.info(
            f"Max rtol err rg_tnd {max(abs(rg_tnd_gt4py.ravel() - prg_tnd_out) / abs(prg_tnd_out))}"
        )

        logging.info(f"Machine precision {np.finfo(gt4py_config.dtypes.float).eps}")

        assert_allclose(pth_tnd_out, th_tnd_gt4py.ravel(), rtol=1e-3)
        assert_allclose(prc_tnd_out, rc_tnd_gt4py.ravel(), rtol=1e-3)
        assert_allclose(prr_tnd_out, rr_tnd_gt4py.ravel(), rtol=1e-3)
        assert_allclose(pri_tnd_out, ri_tnd_gt4py.ravel(), rtol=1e-3)
        assert_allclose(prs_tnd_out, rs_tnd_gt4py.ravel(), rtol=1e-3)
        assert_allclose(prg_tnd_out, rg_tnd_gt4py.ravel(), rtol=1e-3)

    