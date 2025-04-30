from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.config import GT4PyConfig, DataTypes
from ice3_gt4py.phyex_common.phyex import Phyex
from gt4py.storage import from_array
import numpy as np
from pathlib import Path
import fmodpy
import unittest
from numpy.testing import assert_allclose

import logging

from conftest import BACKEND, REBUILD, VALIDATE_ARGS, SHAPE


class TestIce4Tendencies(unittest.TestCase):
    
    def test_ice4_nucleation_post_processing(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
        )

        phyex_externals = Phyex("AROME").to_externals()
        ice4_nucleation_post_processing_gt4py = compile_stencil(
            "ice4_nucleation_post_processing", 
            gt4py_config, 
            phyex_externals
        )
        
        
        t = 300 * np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )        
        exn = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        lsfact = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        tht = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rvt = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rit = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rvheni_mr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        
        
        t_gt4py = from_array(t, dtype=np.float64, backend=BACKEND)
        exn_gt4py = from_array(exn, dtype=np.float64, backend=BACKEND)
        lsfact_gt4py = from_array(lsfact, dtype=np.float64, backend=BACKEND) 
        tht_gt4py = from_array(tht, dtype=np.float64, backend=BACKEND)
        rvt_gt4py = from_array(rvt, dtype=np.float64, backend=BACKEND)
        rit_gt4py = from_array(rit, dtype=np.float64, backend=BACKEND)
        rvheni_mr_gt4py = from_array(rvheni_mr, dtype=np.float64, backend=BACKEND)
        
        ice4_nucleation_post_processing_gt4py(
            t=t_gt4py,
            exn=exn_gt4py,
            lsfact=lsfact_gt4py,
            tht=tht_gt4py,
            rvt=rvt_gt4py,
            rit=rit_gt4py,
            rvheni_mr=rvheni_mr_gt4py,
        )
        
        fortran_script = "mode_ice4_tendencies.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
        
        result = fortran_script.mode_ice4_tendencies.ice4_nucleation_post_processing(
            kproma=SHAPE[0]*SHAPE[1]*SHAPE[2],
            ksize=SHAPE[0]*SHAPE[1]*SHAPE[2],
            plsfact=lsfact.ravel(),
            pexn=exn.ravel(),
            tht=tht.ravel(),
            prit=rit.ravel(),
            prvt=rvt.ravel(),
            zt=t.ravel(),
            rvheni_mr=rvheni_mr.ravel()
        )

        tht_out = result[0]
        rvt_out = result[1]
        rit_out = result[2]
        zt_out = result[3]
        
        logging.info(f"Mean tht_gt4py {tht_gt4py.mean()}")
        logging.info(f"Mean   tht_out {tht_out.mean()}")
        logging.info(f"Max abs rtol {max(abs(tht_gt4py.ravel() - tht_out) / abs(tht_out))}")
        
        logging.info(f"Mean rit_gt4py {rit_gt4py.mean()}")
        logging.info(f"Mean   rit_out {rit_out.mean()}")
        logging.info(f"Max abs rtol {max(abs(rit_gt4py.ravel() - rit_out) / abs(rit_out))}")
        
        logging.info(f"Mean rvt_gt4py {rvt_gt4py.mean()}")
        logging.info(f"Mean   rvt_out {rvt_out.mean()}")
        logging.info(f"Max abs rtol {max(abs(rvt_gt4py.ravel() - rvt_out) / abs(rvt_out))}")

        logging.info(f"Mean t_gt4py {t_gt4py.mean()}")
        logging.info(f"Mean  zt_out {zt_out.mean()}")
        logging.info(f"Max abs rtol {max(abs(t_gt4py.ravel() - zt_out) / abs(zt_out))}")
        
        logging.info(f"Machine precision {np.finfo(np.float64).eps}")

        assert_allclose(tht_out, tht_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(rit_out, rit_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(zt_out, t_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(rvt_out, rvt_gt4py.ravel(),  rtol=1e-3)
        
    
    def test_ice4_rrhong_post_processing(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
        )

        phyex_externals = Phyex("AROME").to_externals()
        ice4_rrhong_post_processing_gt4py = compile_stencil(
            "ice4_rrhong_post_processing", 
            gt4py_config, 
            phyex_externals
        )
        
        
        t = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        exn = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        lsfact = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        lvfact = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        tht = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rrt = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rgt = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rrhong_mr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        
        
        t_gt4py = from_array(t, dtype=np.float64, backend=BACKEND)
        exn_gt4py = from_array(exn, dtype=np.float64, backend=BACKEND)
        lsfact_gt4py = from_array(lsfact, dtype=np.float64, backend=BACKEND) 
        lvfact_gt4py = from_array(lvfact, dtype=np.float64, backend=BACKEND)
        tht_gt4py = from_array(tht, dtype=np.float64, backend=BACKEND)
        rrt_gt4py = from_array(rrt, dtype=np.float64, backend=BACKEND)
        rgt_gt4py = from_array(rgt, dtype=np.float64, backend=BACKEND)
        rrhong_mr_gt4py = from_array(rrhong_mr, dtype=np.float64, backend=BACKEND)
        
        ice4_rrhong_post_processing_gt4py(
            t=t_gt4py,
            exn=exn_gt4py,
            lsfact=lsfact_gt4py,
            lvfact=lvfact_gt4py,
            tht=tht_gt4py,
            rrt=rrt_gt4py,
            rgt=rgt_gt4py,
            rrhong_mr=rrhong_mr_gt4py,
        )
        
        fortran_script = "mode_ice4_tendencies.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
        
        result = fortran_script.mode_ice4_tendencies.ice4_rrhong_post_processing(
            kproma=SHAPE[0]*SHAPE[1]*SHAPE[2],
            ksize=SHAPE[0]*SHAPE[1]*SHAPE[2],
            plsfact=lsfact.ravel(), 
            plvfact=lvfact.ravel(), 
            pexn=exn.ravel(), 
            prrhong_mr=rrhong_mr.ravel(),
            ptht=tht.ravel(),
            pt=t.ravel(),
            prrt=rrt.ravel(),
            prgt=rgt.ravel()
        )

        tht_out = result[0]
        t_out = result[1]
        rrt_out = result[2]
        rgt_out = result[3]
        
        logging.info(f"Mean tht_gt4py {tht_gt4py.mean()}")
        logging.info(f"Mean tht_out {tht_out.mean()}")
        logging.info(f"Max abs rtol {max(abs(tht_gt4py.ravel() - tht_out) / abs(tht_out))}")

        logging.info(f"Mean t_gt4py {t_gt4py.mean()}")
        logging.info(f"Mean t_out {t_out.mean()}")
        logging.info(f"Max abs rtol {max(abs(t_gt4py.ravel() - t_out) / abs(t_out))}")
        
        
        logging.info(f"Mean rrt_gt4py {rrt_gt4py.mean()}")
        logging.info(f"Mean rrt_out {rrt_out.mean()}")
        logging.info(f"Max abs rtol {max(abs(rrt_gt4py.ravel() - rrt_out) / abs(rrt_out))}")
        
        logging.info(f"Mean rgt_gt4py {rgt_gt4py.mean()}")
        logging.info(f"Mean rgt_out {rgt_out.mean()}")
        logging.info(f"Max abs rtol {max(abs(rgt_gt4py.ravel() - rgt_out) / abs(rgt_out))}")

        logging.info(f"Machine precision {np.finfo(np.float64).eps}")

        assert_allclose(tht_out, tht_gt4py[...].ravel(),  rtol=1e-3)
        assert_allclose(t_out, t_gt4py[...].ravel(),  rtol=1e-3)
        assert_allclose(rrt_out, rrt_gt4py[...].ravel(),  rtol=1e-3)
        assert_allclose(rgt_out, rgt_gt4py[...].ravel(),  rtol=1e-3)
        

    def test_ice4_rimltc_post_processing(self):
        
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
        )

        phyex_externals = Phyex("AROME").to_externals()
        ice4_rimltc_post_processing_gt4py = compile_stencil(
            "ice4_rimltc_post_processing", 
            gt4py_config, 
            phyex_externals
        )
        
        
        t = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        
        exn = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        lsfact = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        lvfact = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rimltc_mr = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        tht = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rct = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rit = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        
        
        t_gt4py = from_array(t, dtype=np.float64, backend=BACKEND)
        exn_gt4py = from_array(exn, dtype=np.float64, backend=BACKEND)
        lsfact_gt4py = from_array(lsfact, dtype=np.float64, backend=BACKEND) 
        lvfact_gt4py = from_array(lvfact, dtype=np.float64, backend=BACKEND)
        tht_gt4py = from_array(tht, dtype=np.float64, backend=BACKEND)
        rimltc_mr_gt4py = from_array(rimltc_mr, dtype=np.float64, backend=BACKEND)
        rct_gt4py = from_array(rct, dtype=np.float64, backend=BACKEND)
        rit_gt4py = from_array(rit, dtype=np.float64, backend=BACKEND)
        
        ice4_rimltc_post_processing_gt4py(
            t=t_gt4py,
            exn=exn_gt4py,
            lsfact=lsfact_gt4py,
            lvfact=lvfact_gt4py,
            rimltc_mr=rimltc_mr_gt4py,
            tht=tht_gt4py,
            rct=rct_gt4py,
            rit=rit_gt4py
        )
        
        fortran_script = "mode_ice4_tendencies.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
        
        result = fortran_script.mode_ice4_tendencies.ice4_rimltc_post_processing(
            kproma=SHAPE[0]*SHAPE[1]*SHAPE[2],
            ksize=SHAPE[0]*SHAPE[1]*SHAPE[2],
            plsfact=lsfact.ravel(), 
            plvfact=lvfact.ravel(), 
            pexn=exn.ravel(), 
            primltc_mr=rimltc_mr.ravel(),
            ptht=tht.ravel(),
            pt=t.ravel(),
            prit=rit.ravel(),
            prct=rct.ravel()
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

        logging.info(f"Machine precision {np.finfo(np.float64).eps}")

        assert_allclose(tht_out, tht_gt4py[...].ravel(),  rtol=1e-3)
        assert_allclose(t_out, t_gt4py[...].ravel(),  rtol=1e-3)
        assert_allclose(rct_out, rct_gt4py[...].ravel(),  rtol=1e-3)
        assert_allclose(rit_out, rit_gt4py[...].ravel(),  rtol=1e-3)
        

    def test_ice4_rimltc_post_processing(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
        )

        phyex_externals = Phyex("AROME").to_externals()
        ice4_rimltc_post_processing_gt4py = compile_stencil(
            "ice4_rimltc_post_processing", 
            gt4py_config, 
            phyex_externals
        )
         
        t = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            ) 
        exn = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        lsfact = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        lvfact = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rimltc_mr = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        tht = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rct = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rit = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        
        
        t_gt4py = from_array(t, dtype=np.float64, backend=BACKEND)
        exn_gt4py = from_array(exn, dtype=np.float64, backend=BACKEND)
        lsfact_gt4py = from_array(lsfact, dtype=np.float64, backend=BACKEND) 
        lvfact_gt4py = from_array(lvfact, dtype=np.float64, backend=BACKEND)
        tht_gt4py = from_array(tht, dtype=np.float64, backend=BACKEND)
        rimltc_mr_gt4py = from_array(rimltc_mr, dtype=np.float64, backend=BACKEND)
        rct_gt4py = from_array(rct, dtype=np.float64, backend=BACKEND)
        rit_gt4py = from_array(rit, dtype=np.float64, backend=BACKEND)
        
        ice4_rimltc_post_processing_gt4py(
            t=t_gt4py,
            exn=exn_gt4py,
            lsfact=lsfact_gt4py,
            lvfact=lvfact_gt4py,
            rimltc_mr=rimltc_mr_gt4py,
            tht=tht_gt4py,
            rct=rct_gt4py,
            rit=rit_gt4py
        )
        
        fortran_script = "mode_ice4_tendencies.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
        
        result = fortran_script.mode_ice4_tendencies.ice4_rimltc_post_processing(
            kproma=SHAPE[0]*SHAPE[1]*SHAPE[2],
            ksize=SHAPE[0]*SHAPE[1]*SHAPE[2],
            plsfact=lsfact.ravel(), 
            plvfact=lvfact.ravel(), 
            pexn=exn.ravel(), 
            primltc_mr=rimltc_mr.ravel(),
            ptht=tht.ravel(),
            pt=t.ravel(),
            prit=rit.ravel(),
            prct=rct.ravel()
        )

        tht_out = result[0]
        t_out = result[1]
        rct_out = result[2]
        rit_out = result[3]
        
        logging.info(f"Mean tht_gt4py {tht_gt4py.mean()}")
        logging.info(f"Mean tht_out {tht_out.mean()}")
        logging.info(f"Max abs rtol {max(abs(tht_gt4py.ravel() - tht_out) / abs(tht_out))}")
        
        logging.info(f"Mean t_gt4py {t_gt4py.mean()}")
        logging.info(f"Mean t_out {t_out.mean()}")
        logging.info(f"Max abs rtol {max(abs(t_gt4py.ravel() - t_out) / abs(t_out))}")

        
        logging.info(f"Mean rct_gt4py {rct_gt4py.mean()}")
        logging.info(f"Mean rct_out {rct_out.mean()}")
        logging.info(f"Max abs rtol {max(abs(rct_gt4py.ravel() - rct_out) / abs(rct_out))}")

        
        logging.info(f"Mean rit_gt4py {rit_gt4py.mean()}")
        logging.info(f"Mean rit_out {rit_out.mean()}")
        logging.info(f"Max abs rtol {max(abs(rit_gt4py.ravel() - rit_out) / abs(rit_out))}")

        logging.info(f"Machine precision {np.finfo(np.float64).eps}")

        assert_allclose(tht_out, tht_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(t_out, t_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(rct_out, rct_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(rit_out, rit_gt4py.ravel(),  rtol=1e-3)
        
    
    def test_ice4_fast_rg_pre_processing(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
        )

        phyex_externals = Phyex("AROME").to_externals()
        ice4_rimltc_post_processing_gt4py = compile_stencil(
            "ice4_rimltc_post_processing", 
            gt4py_config, 
            phyex_externals
        )
         
        rgsi = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            ) 
        rgsi_mr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            ) 
        rvdepg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            ) 
        rsmltg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            ) 
        rraccsg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            ) 
        rsaccrg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            ) 
        rcrimsg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            ) 
        rsrimcg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            ) 
        rrhong_mr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            ) 
        rsrimcg_mr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            ) 
        
        rgsi_gt4py  = from_array(rgsi, dtype=np.float64, backend=BACKEND)
        rgsi_mr_gt4py  = from_array(rgsi_mr, dtype=np.float64, backend=BACKEND)
        rvdepg_gt4py  = from_array(rvdepg, dtype=np.float64, backend=BACKEND)
        rsmltg_gt4py  = from_array(rsmltg, dtype=np.float64, backend=BACKEND)
        rraccsg_gt4py  = from_array(rraccsg, dtype=np.float64, backend=BACKEND)
        rsaccrg_gt4py  = from_array(rsaccrg, dtype=np.float64, backend=BACKEND)
        rcrimsg_gt4py  = from_array(rcrimsg, dtype=np.float64, backend=BACKEND)
        rsrimcg_gt4py  = from_array(rsrimcg, dtype=np.float64, backend=BACKEND)
        rrhong_mr_gt4py  = from_array(rrhong_mr, dtype=np.float64, backend=BACKEND)
        rsrimcg_mr_gt4py  = from_array(rsrimcg_mr, dtype=np.float64, backend=BACKEND)
        
        ice4_rimltc_post_processing_gt4py(
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
        )
        
        fortran_script = "mode_ice4_tendencies.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, 
            "src", 
            "ice3_gt4py", 
            "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
        
        result = fortran_script.mode_ice4_tendencies.ice4_fast_rg_pre_processing(
            kproma=SHAPE[0]*SHAPE[1]*SHAPE[2],
            ksize=SHAPE[0]*SHAPE[1]*SHAPE[2],
            rvdepg=rvdepg, 
            rsmltg=rsmltg, 
            rraccsg=rraccsg, 
            rsaccrg=rsaccrg, 
            rcrimsg=rcrimsg, 
            rsrimcg=rsrimcg,
            rrhong_mr=rrhong_mr, 
            rsrimcg_mr=rsrimcg,
            zgrsi=rgsi, 
            zrgsi_mr=rgsi_mr
        )

        zrgsi_out = result[0]
        zrgsi_mr_out = result[1]
        
        logging.info(f"Mean zrgsi_gt4py {rgsi_gt4py.mean()}")
        logging.info(f"Mean zrgsi_out   {zrgsi_out.mean()}")
        logging.info(f"Max abs rtol     {max(abs(rgsi_gt4py.ravel() - zrgsi_out) / abs(zrgsi_out))}")
        
        logging.info(f"Mean rgsi_mr_gt4py   {rgsi_gt4py.mean()}")
        logging.info(f"Mean rgsi_mr_out     {zrgsi_mr_out.mean()}")
        logging.info(f"Max abs rtol {max(abs(rgsi_mr_gt4py.ravel() - zrgsi_mr_out) / abs(zrgsi_mr_out))}")
        
        logging.info(f"Machine precision {np.finfo(np.float64).eps}")

        assert_allclose(zrgsi_out, rgsi_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(zrgsi_mr_out, rgsi_mr_gt4py.ravel(),  rtol=1e-3)
      
    
    def test_ice4_increment_update(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
        )

        phyex_externals = Phyex("AROME").to_externals()
        ice4_increment_update_gt4py = compile_stencil(
            "ice4_increment_update", 
            gt4py_config, 
            phyex_externals
        )
        
        
        lsfact = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        
        lvfact = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        theta_increment = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rv_increment = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rc_increment = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rr_increment = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        ri_increment = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rs_increment = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rg_increment = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rvheni_mr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rimltc_mr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rrhong_mr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rsrimcg_mr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        
        
        lsfact_gt4py = from_array(lsfact, dtype=np.float64, backend=BACKEND)
        lvfact_gt4py = from_array(lvfact, dtype=np.float64, backend=BACKEND)
        theta_increment_gt4py = from_array(theta_increment, dtype=np.float64, backend=BACKEND) 
        rv_increment_gt4py = from_array(rv_increment, dtype=np.float64, backend=BACKEND)
        rc_increment_gt4py = from_array(rc_increment, dtype=np.float64, backend=BACKEND)
        rr_increment_gt4py = from_array(rr_increment, dtype=np.float64, backend=BACKEND)
        ri_increment_gt4py = from_array(ri_increment, dtype=np.float64, backend=BACKEND)
        rs_increment_gt4py = from_array(rs_increment, dtype=np.float64, backend=BACKEND)
        rg_increment_gt4py = from_array(rg_increment, dtype=np.float64, backend=BACKEND)
        rvheni_mr_gt4py = from_array(rvheni_mr, dtype=np.float64, backend=BACKEND)
        rimltc_mr_gt4py = from_array(rimltc_mr, dtype=np.float64, backend=BACKEND)
        rsrimcg_mr_gt4py = from_array(rsrimcg_mr, dtype=np.float64, backend=BACKEND)
        rrhong_mr_gt4py = from_array(rrhong_mr, dtype=np.float64, backend=BACKEND)
        
        
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
            rvheni_mr=rvheni_mr_gt4py
        )
        
        fortran_script = "mode_ice4_tendencies.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
        
        result = fortran_script.mode_ice4_tendencies.ice4_increment_update(
            kproma=SHAPE[0]*SHAPE[1]*SHAPE[2],
            ksize=SHAPE[0]*SHAPE[1]*SHAPE[2],
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
            prg_inst=rg_increment.ravel()
        )

        pth_inst_out=result[0]
        prv_inst_out=result[1]
        prc_inst_out=result[2]
        prr_inst_out=result[3]
        pri_inst_out=result[4]
        prs_inst_out=result[5]
        prg_inst_out=result[6]
        
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
        
        logging.info(f"Machine precision {np.finfo(np.float64).eps}")

        assert_allclose(pth_inst_out, theta_increment_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(prv_inst_out, rv_increment_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(prc_inst_out, rc_increment_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(pri_inst_out, ri_increment_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(prr_inst_out, rr_increment_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(prs_inst_out, rs_increment_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(prg_inst_out, rg_increment_gt4py.ravel(),  rtol=1e-3)

        
    def test_ice4_derived_fields(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
        )

        phyex_externals = Phyex("AROME").to_externals()
        ice4_derived_fields_gt4py = compile_stencil(
            "ice4_derived_fields", 
            gt4py_config, 
            phyex_externals,
        )
        
        t = 300 * np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rhodref = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        pres = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        ssi = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        ka = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        dv = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        ai = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        cj = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rvt = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        zw = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        
        t_gt4py = from_array(t, dtype=np.float64, backend=BACKEND)
        rhodref_gt4py = from_array(rhodref, dtype=np.float64, backend=BACKEND)
        pres_gt4py = from_array(pres, dtype=np.float64, backend=BACKEND) 
        ssi_gt4py = from_array(ssi, dtype=np.float64, backend=BACKEND)
        ka_gt4py = from_array(ka, dtype=np.float64, backend=BACKEND)
        dv_gt4py = from_array(dv, dtype=np.float64, backend=BACKEND)
        ai_gt4py = from_array(ai, dtype=np.float64, backend=BACKEND)
        cj_gt4py = from_array(cj, dtype=np.float64, backend=BACKEND)
        rvt_gt4py = from_array(rvt, dtype=np.float64, backend=BACKEND)
        zw_gt4py = from_array(zw, dtype=np.float64, backend=BACKEND)
        
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
            zw=zw_gt4py
        )
        
        fortran_script = "mode_ice4_tendencies.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
        
        result = fortran_script.mode_ice4_tendencies.ice4_derived_fields(
            kproma=SHAPE[0]*SHAPE[1]*SHAPE[2],
            ksize=SHAPE[0]*SHAPE[1]*SHAPE[2],
            xalpi=phyex_externals["ALPI"],
            xbetai=phyex_externals["BETAI"],
            xgami=phyex_externals["GAMI"],
            xepsilo=phyex_externals["EPSILO"],
            xrv=phyex_externals["RV"],
            xci=phyex_externals["CI"],
            xlstt=phyex_externals["LSTT"],
            xcpv=phyex_externals["CPV"],
            xp00=phyex_externals["P00"],
            xscfac=phyex_externals["SCFAC"],
            xtt=phyex_externals["TT"],
            zt=t.ravel(),
            prvt=rvt.ravel(),
            ppres=pres.ravel(),
            prhodref=rhodref.ravel(),
            zzw=zw.ravel(),
            pssi=ssi.ravel(),
            zka=ka.ravel(),
            zai=ai.ravel(),
            zdv=dv.ravel(),
            zcj=cj.ravel()
        )

        zzw_out=result[0]
        pssi_out=result[1]
        zka_out=result[2]
        zai_out=result[3]
        zdv_out=result[4]
        zcj_out=result[5]
        
        logging.info(f"Mean zw_gt4py {zw_gt4py.mean()}")
        logging.info(f"Mean  zzw_out {zzw_out.mean()}")
        logging.info(f"Max rtol err zw {max(abs(zw_gt4py.ravel() - zzw_out) / abs(zzw_out))}")
        
        logging.info(f"Mean ssi_gt4py {ssi_gt4py.mean()}")
        logging.info(f"Mean  pssi_out {pssi_out.mean()}")
        logging.info(f"Max rtol err ssi {max(abs(ssi_gt4py.ravel() - pssi_out) / abs(pssi_out))}")
        
        logging.info(f"Mean ka_gt4py {ka_gt4py.mean()}")
        logging.info(f"Mean  zka_out {zka_out.mean()}")
        logging.info(f"Max rtol err ka {max(abs(ka_gt4py.ravel() - zka_out) / abs(zka_out))}")

        logging.info(f"Mean ai_gt4py {ai_gt4py.mean()}")
        logging.info(f"Mean  zai_out {zai_out.mean()}")
        logging.info(f"Max rtol err ai {max(abs(ai_gt4py.ravel() - zai_out) / abs(zai_out))}")
        
        logging.info(f"Mean dv_gt4py {dv_gt4py.mean()}")
        logging.info(f"Mean  zdv_out {zdv_out.mean()}")
        logging.info(f"Max rtol err dv {max(abs(dv_gt4py.ravel() - zdv_out) / abs(zdv_out))}")

        logging.info(f"Mean cj_gt4py {cj_gt4py.mean()}")
        logging.info(f"Mean  zcj_out {zcj_out.mean()}")
        logging.info(f"Max rtol err cj {max(abs(cj_gt4py.ravel() - zcj_out) / abs(zcj_out))}")

        logging.info(f"Machine precision {np.finfo(np.float64).eps}")

        assert_allclose(pssi_out, ssi_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(zka_out, ka_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(zai_out, ai_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(zdv_out, dv_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(zcj_out, cj_gt4py.ravel(),  rtol=1e-3)

        
    def test_ice4_slope_parameters(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
        )

        phyex_externals = Phyex("AROME").to_externals()
        ice4_slope_parameters = compile_stencil(
            "ice4_slope_parameters", 
            gt4py_config, 
            phyex_externals
        )
        
        rhodref = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        t = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rrt = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rst = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rgt = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        lbdar = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        lbdar_rf = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        lbdas = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        lbdag = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        
        
        
        rhodref_gt4py = from_array(rhodref, dtype=np.float64, backend=BACKEND)
        t_gt4py = from_array(t, dtype=np.float64, backend=BACKEND)
        rrt_gt4py = from_array(rrt, dtype=np.float64, backend=BACKEND) 
        rst_gt4py = from_array(rst, dtype=np.float64, backend=BACKEND)
        rgt_gt4py = from_array(rgt, dtype=np.float64, backend=BACKEND)
        lbdar_gt4py = from_array(lbdar, dtype=np.float64, backend=BACKEND)
        lbdar_rf_gt4py = from_array(lbdar_rf, dtype=np.float64, backend=BACKEND)
        lbdas_gt4py = from_array(lbdas, dtype=np.float64, backend=BACKEND)
        lbdag_gt4py = from_array(lbdag, dtype=np.float64, backend=BACKEND)

        
        ice4_slope_parameters(
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
            
        
        fortran_script = "mode_ice4_tendencies.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
        
        result = fortran_script.mode_ice4_tendencies.ice4_slope_parameters(
            kproma=SHAPE[0]*SHAPE[1]*SHAPE[2],
            ksize=SHAPE[0]*SHAPE[1]*SHAPE[2],
            xlbr=phyex_externals["LBR"],
            xlbexr=phyex_externals["LBEXR"],
            xlbg=phyex_externals["LBG"],
            xlbdas_min=phyex_externals["LBDAS_MIN"],
            xlbdas_max=phyex_externals["LBDAS_MAX"], 
            xtrans_mp_gammas=phyex_externals["TRANS_MP_GAMMAS"],
            xlbs=phyex_externals["LBS"], 
            xlbexs=phyex_externals["LBEXS"], 
            xlbexg=phyex_externals["LBEXG"],
            r_rtmin=phyex_externals["R_RTMIN"], 
            s_rtmin=phyex_externals["S_RTMIN"], 
            g_rtmin=phyex_externals["G_RTMIN"],
            lsnow_t=phyex_externals["LSNOW_T"], 
            prrt=rrt.ravel(), 
            prhodref=rhodref.ravel(), 
            prst=rst.ravel(), 
            prgt=rgt.ravel(), 
            zt=t.ravel(),
            zlbdag=lbdag.ravel(), 
            zlbdas=lbdas.ravel(), 
            zlbdar=lbdar.ravel(), 
            zlbdar_rf=lbdar_rf.ravel()
        )
        
        zlbdar_out = result[0]
        zlbdar_rf_out = result[1]
        zlbdas_out = result[2]
        zlbdag_out = result[3]
        
        logging.info(f"Mean lbdar_gt4py {lbdar_gt4py.mean()}")
        logging.info(f"Mean zlbdar_out {zlbdar_out.mean()}")
        logging.info(f"Max rtol err ssi {max(abs(lbdar_gt4py.ravel() - zlbdar_out) / abs(zlbdar_out))}")
        
        logging.info(f"Mean lbdar_rf_gt4py {lbdar_rf_gt4py.mean()}")
        logging.info(f"Mean zlbdar_rf_out {zlbdar_rf_out.mean()}")
        logging.info(f"Max abs err ssi {max(abs(lbdar_rf_gt4py.ravel() - zlbdar_rf_out) /  abs(zlbdar_out))}")
        
        logging.info(f"Mean lbdas_gt4py {lbdas_gt4py.mean()}")
        logging.info(f"Mean lbdas_out {zlbdas_out.mean()}")
        logging.info(f"Max abs err ssi {max(abs(lbdas_gt4py.ravel() - zlbdas_out) / abs(zlbdas_out))}")

        
        logging.info(f"Mean lbdag_gt4py {lbdag_gt4py.mean()}")
        logging.info(f"Mean lbdag_out {zlbdag_out.mean()}")
        logging.info(f"Max abs err ssi {max(abs(lbdag_gt4py.ravel() - zlbdag_out) / abs(zlbdag_out))}")

        
        logging.info(f"Machine precision {np.finfo(np.float64).eps}")

        assert_allclose(zlbdar_out, lbdar_gt4py.ravel(),   rtol=1e-3)
        assert_allclose(zlbdar_rf_out, lbdar_rf_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(zlbdas_out, lbdas_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(zlbdag_out, lbdag_gt4py.ravel(),  rtol=1e-3)
        
        
    def test_ice4_total_tendencies_update(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
        )

        phyex_externals = Phyex("AROME").to_externals()
        ice4_total_tendencies_update = compile_stencil(
            "ice4_total_tendencies_update", 
            gt4py_config, 
            phyex_externals
        )
        
        lsfact = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        lvfact = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        th_tnd = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rv_tnd = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )
        rc_tnd = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rr_tnd = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        ri_tnd = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rs_tnd = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rg_tnd = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rchoni = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rvdeps = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        riaggs = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        riauts = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rvdepg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rcautr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rcaccr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rrevav = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rcberi = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rsmltg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rcmltsr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rraccss = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rraccsg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rsaccrg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rcrimss = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rcrimsg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rsrimcg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        ricfrrg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rrcfrig = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        ricfrr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rcwetg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        riwetg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rrwetg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rswetg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rcdryg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        ridryg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rrdryg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rsdryg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rgmltr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
            )
        rwetgh = np.array(
            np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float64,
                order="F",
        )

        
        lsfact_gt4py = from_array(lsfact, dtype=np.float64, backend=BACKEND)
        lvfact_gt4py = from_array(lvfact, dtype=np.float64, backend=BACKEND)
        th_tnd_gt4py = from_array(th_tnd, dtype=np.float64, backend=BACKEND) 
        rv_tnd_gt4py = from_array(rv_tnd, dtype=np.float64, backend=BACKEND)
        rc_tnd_gt4py = from_array(rc_tnd, dtype=np.float64, backend=BACKEND)
        rr_tnd_gt4py = from_array(rr_tnd, dtype=np.float64, backend=BACKEND)
        ri_tnd_gt4py = from_array(ri_tnd, dtype=np.float64, backend=BACKEND)
        rs_tnd_gt4py = from_array(rs_tnd, dtype=np.float64, backend=BACKEND)
        rg_tnd_gt4py = from_array(rg_tnd, dtype=np.float64, backend=BACKEND)
        
        rchoni_gt4py = from_array(rchoni, dtype=np.float64, backend=BACKEND)
        rvdeps_gt4py = from_array(rvdeps, dtype=np.float64, backend=BACKEND)
        riaggs_gt4py = from_array(riaggs, dtype=np.float64, backend=BACKEND)
        riauts_gt4py = from_array(riauts, dtype=np.float64, backend=BACKEND)
        rvdepg_gt4py = from_array(rvdepg, dtype=np.float64, backend=BACKEND)
        rcautr_gt4py = from_array(rcautr, dtype=np.float64, backend=BACKEND)
        rcaccr_gt4py = from_array(rcaccr, dtype=np.float64, backend=BACKEND)
        rrevav_gt4py = from_array(rrevav, dtype=np.float64, backend=BACKEND)
        rcberi_gt4py = from_array(rcberi, dtype=np.float64, backend=BACKEND)
        rsmltg_gt4py = from_array(rsmltg, dtype=np.float64, backend=BACKEND)
        rcmltsr_gt4py = from_array(rcmltsr, dtype=np.float64, backend=BACKEND)
        rraccss_gt4py = from_array(rraccss, dtype=np.float64, backend=BACKEND)
        rraccsg_gt4py = from_array(rraccsg, dtype=np.float64, backend=BACKEND)
        rsaccrg_gt4py = from_array(rsaccrg, dtype=np.float64, backend=BACKEND)
        rcrimss_gt4py = from_array(rcrimss, dtype=np.float64, backend=BACKEND)
        rcrimsg_gt4py = from_array(rcrimsg, dtype=np.float64, backend=BACKEND)
        rsrimcg_gt4py = from_array(rsrimcg, dtype=np.float64, backend=BACKEND)
        ricfrrg_gt4py = from_array(ricfrrg, dtype=np.float64, backend=BACKEND)
        rrcfrig_gt4py = from_array(rrcfrig, dtype=np.float64, backend=BACKEND)
        ricfrr_gt4py = from_array(ricfrr, dtype=np.float64, backend=BACKEND)
        rcwetg_gt4py = from_array(rcwetg, dtype=np.float64, backend=BACKEND)
        riwetg_gt4py = from_array(riwetg, dtype=np.float64, backend=BACKEND)
        rrwetg_gt4py = from_array(rrwetg, dtype=np.float64, backend=BACKEND)
        rswetg_gt4py = from_array(rswetg, dtype=np.float64, backend=BACKEND)
        rcdryg_gt4py = from_array(rcdryg, dtype=np.float64, backend=BACKEND)
        ridryg_gt4py = from_array(ridryg, dtype=np.float64, backend=BACKEND)
        rrdryg_gt4py = from_array(rrdryg, dtype=np.float64, backend=BACKEND)
        rsdryg_gt4py = from_array(rsdryg, dtype=np.float64, backend=BACKEND)
        rgmltr_gt4py = from_array(rgmltr, dtype=np.float64, backend=BACKEND)
        rwetgh_gt4py = from_array(rwetgh, dtype=np.float64, backend=BACKEND)
        
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
            rwetgh=rwetgh_gt4py
        )
        
        fortran_script = "mode_ice4_tendencies.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
        
        result = fortran_script.mode_ice4_tendencies.ice4_total_tendencies_update(
            kproma=SHAPE[0]*SHAPE[1]*SHAPE[2],
            ksize=SHAPE[0]*SHAPE[1]*SHAPE[2],
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
        logging.info(f"Max rtol err th_tnd {max(abs(th_tnd_gt4py.ravel() - pth_tnd_out) / abs(pth_tnd_out))}")
        
        logging.info(f"Mean rv_tnd_gt4py {rv_tnd_gt4py.mean()}")
        logging.info(f"Mean rv_tnd_out {prv_tnd_out.mean()}")
        logging.info(f"Max rtol err rv_tnd {max(abs(rv_tnd_gt4py.ravel() - prv_tnd_out) / abs(prv_tnd_out))}")

        
        logging.info(f"Mean rc_tnd_gt4py {rc_tnd_gt4py.mean()}")
        logging.info(f"Mean prc_tnd_out {prc_tnd_out.mean()}")
        logging.info(f"Max rtol err rc_tnd {max(abs(rc_tnd_gt4py.ravel() - prc_tnd_out) / abs(prc_tnd_out))}")
        
        logging.info(f"Mean rr_tnd_gt4py {rr_tnd_gt4py.mean()}")
        logging.info(f"Mean prr_tnd_out {prr_tnd_out.mean()}")
        logging.info(f"Max rtol err rr {max(abs(rr_tnd_gt4py.ravel() - prr_tnd_out) / abs(prr_tnd_out))}")
        
        logging.info(f"Mean ri_tnd_gt4py {ri_tnd_gt4py.mean()}")
        logging.info(f"Mean pri_tnd_out {pri_tnd_out.mean()}")
        logging.info(f"Max rtol err ri_tnd {max(abs(ri_tnd_gt4py.ravel() - pri_tnd_out) / abs(pri_tnd_out))}")
        
        logging.info(f"Mean rs_tnd_gt4py {rs_tnd_gt4py.mean()}")
        logging.info(f"Mean prs_tnd_out {prs_tnd_out.mean()}")
        logging.info(f"Max rtol err rs_tnd {max(abs(rs_tnd_gt4py.ravel() - prs_tnd_out) / abs(prs_tnd_out))}")
        
        logging.info(f"Mean rg_tnd_gt4py {rg_tnd_gt4py.mean()}")
        logging.info(f"Mean prg_tnd_out {prg_tnd_out.mean()}")
        logging.info(f"Max rtol err rg_tnd {max(abs(rg_tnd_gt4py.ravel() - prg_tnd_out) / abs(prg_tnd_out))}")
        
        logging.info(f"Machine precision {np.finfo(np.float64).eps}")

        assert_allclose(pth_tnd_out, th_tnd_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(prc_tnd_out, rc_tnd_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(prr_tnd_out, rr_tnd_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(pri_tnd_out, ri_tnd_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(prs_tnd_out, rs_tnd_gt4py.ravel(),  rtol=1e-3)
        assert_allclose(prg_tnd_out, rg_tnd_gt4py.ravel(),  rtol=1e-3)
        
        