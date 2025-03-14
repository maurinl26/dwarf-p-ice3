from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.config import GT4PyConfig
from ice3_gt4py.phyex_common.phyex import Phyex
from gt4py.storage import from_array
import numpy as np
from pathlib import Path
import fmodpy
import unittest

import logging

BACKEND = "gt:cpu_ifirst"
REBUILD = True
VALIDATE_ARGS = True
SHAPE = (50, 50, 15)

class TestIce4Rimltc(unittest.TestCase):
    
    def test_ice4_rimltc(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
        backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
    )
    
        phyex_externals = Phyex("AROME").to_externals()
        ice4_rimltc_gt4py = compile_stencil("ice4_rimltc", gt4py_config, phyex_externals)
        
        ldcompute = from_array(
            data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.bool_,
        backend=BACKEND
        )
        t= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=np.float64,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        exn= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=np.float64,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        lv_fact= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=np.float64,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        ls_fact= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=np.float64,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        tht= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=np.float64,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # theta at time t
        ri_t= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=np.float64,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # rain water mixing ratio at t
        rimltc_mr= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=np.float64,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        ice4_rimltc_gt4py(
            ldcompute=ldcompute,
            t=t,
            exn=exn,
            lv_fact=lv_fact,
            ls_fact=ls_fact,
            tht=tht,  
            ri_t=ri_t,  
            rimltc_mr=rimltc_mr,
        )
    

        fortran_script = "mode_ice4_rimltc.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
    
        result = fortran_script.mode_ice4_rimltc.ice4_rimltc(
                xtt=phyex_externals["TT"],
                lfeedbackt=phyex_externals["LFEEDBACKT"],
                kproma=SHAPE[0]*SHAPE[1],
                ksize=SHAPE[2],
                ldcompute=ldcompute,
                pexn=exn, 
                plvfact=lv_fact,
                plsfact=ls_fact,
                pt=t, 
                ptht=tht, 
                prit=ri_t, 
                primltc_mr=rimltc_mr
    )
        
        RIMLTC_MR = result[0]
        
        logging.info(f"Machine precision {np.finfo(np.float64).eps}")
        assert np.allclose(RIMLTC_MR, rimltc_mr, 10e-8)
        
    def test_ice4_rimltc_post_processing(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
        backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
    )
    
        phyex_externals = Phyex("AROME").to_externals()
        ice4_rimltc_post_processing_gt4py = compile_stencil(
            "ice4_rimltc_post_processing", 
            gt4py_config, phyex_externals
            )
        
        ldcompute = from_array(
            data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.bool_,
        backend=BACKEND
        )
        
        t = from_array(
            data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=np.float64,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
        ) 
        exn = from_array(
            data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=np.float64,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
        ) 
        ls_fact = from_array(
            data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=np.float64,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
        ) 
        lv_fact = from_array(
            data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=np.float64,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
        ) 
        tht = from_array(
            data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=np.float64,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
        ) 
        rc_t = from_array(
            data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=np.float64,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
        ) 
        ri_t = from_array(
            data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=np.float64,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
        ) 
        rimltc_mr = from_array(
            data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=np.float64,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
        ) 
    
        ice4_rimltc_post_processing_gt4py(
            t=t,
            exn=exn,
            ls_fact=ls_fact,
            lv_fact=lv_fact,
            tht=tht,
            rc_t=rc_t,
            ri_t=ri_t,
            rimltc_mr=rimltc_mr,
        )
    

        fortran_script = "mode_ice4_rimltc.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
        
        print(f"Shape {ls_fact.ravel().shape}")
    
        result = fortran_script.mode_ice4_rimltc.ice4_rimltc_post_processing(
                ksize=SHAPE[0]*SHAPE[1]*SHAPE[2],
                plsfact=ls_fact.ravel(),
                plvfact=lv_fact.ravel(),
                primltc_mr=rimltc_mr.ravel(),
                ptht=tht.ravel(),
                prit=ri_t.ravel(),
                prct=rc_t.ravel(),
                pt=t.ravel(),
                pexn=exn.ravel()
    )
        
        PRIMLTC_MR = result[0]
        PTHT = result[1]
        PRIT = result[2]
        PRCT = result[3]
        PT = result[4]
        PEXN = result[5]
        
        logging.info(f"Machine precision {np.finfo(np.float64).eps}")
        assert np.allclose(PRIMLTC_MR, rimltc_mr.ravel(), 10e-8)
        assert np.allclose(PTHT, tht.ravel(), 10e-8)
        assert np.allclose(PRIT, ri_t.ravel(), 10e-8)
        assert np.allclose(PRCT, rc_t.ravel(), 10e-8)
        assert np.allclose(PT, t.ravel(), 10e-8)
        assert np.allclose(PEXN, exn.ravel(), 10e-8)
        
        
        
    
    
