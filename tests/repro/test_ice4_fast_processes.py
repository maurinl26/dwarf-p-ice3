from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.config import GT4PyConfig, DataTypes
from ice3_gt4py.phyex_common.phyex import Phyex
from gt4py.storage import from_array
import numpy as np
from numpy.testing import assert_allclose
from pathlib import Path
import fmodpy
import unittest

import logging

from .env import BACKEND, REBUILD, VALIDATE_ARGS, SHAPE


class TestIce4Fast(unittest.TestCase):
    
    def test_ice4_fast_ri(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, 
            rebuild=REBUILD, 
            validate_args=VALIDATE_ARGS, 
            verbose=True,
            dtypes=DataTypes(bool=bool, float=np.float32, int=np.int32)
        )
        
        logging.info(f"Machine precision {np.finfo(np.float32).eps}")
        logging.info(f"Machine precision {np.finfo(np.float32).eps}")


        phyex_externals = Phyex("AROME").to_externals()
        ice4_fast_ri_gt4py = compile_stencil(
            "ice4_fast_ri", 
            gt4py_config, 
            phyex_externals
        )
        
        
        ldcompute = np.ones(
                (SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=bool,
                order="F",
            )
        rhodref = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float32,
                order="F",
            )
        ai= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float32,
                order="F",
            )
        cj= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float32,
                order="F",
            )
        cit= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float32,
                order="F",
            )
        ssi= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float32,
                order="F",
            )
        rct= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float32,
                order="F",
            )
        rit= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float32,
                order="F",
            )
        rc_beri_tnd = np.zeros(
                (SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=np.float32,
                order="F",
            )
        ldsoft = False
        
        ldcompute_gt4py = from_array(ldcompute, dtype=bool, backend=BACKEND)
        rhodref_gt4py = from_array(rhodref, dtype=np.float32, backend=BACKEND)
        ai_gt4py = from_array(ai, dtype=np.float32, backend=BACKEND)
        cj_gt4py = from_array(cj, dtype=np.float32, backend=BACKEND)
        cit_gt4py = from_array(cit, dtype=np.float32, backend=BACKEND) 
        ssi_gt4py = from_array(ssi, dtype=np.float32, backend=BACKEND)
        rct_gt4py = from_array(rct, dtype=np.float32, backend=BACKEND)
        rit_gt4py = from_array(rit, dtype=np.float32, backend=BACKEND)
        rc_beri_tnd_gt4py = from_array(rc_beri_tnd, dtype=np.float32, backend=BACKEND)
        
        logging.info(f"IN mean rc_beri_tnd_gt4py {rc_beri_tnd_gt4py.mean()}")
        
        ice4_fast_ri_gt4py(
            ldcompute=ldcompute_gt4py,
            rhodref=rhodref_gt4py,
            ai=ai_gt4py,
            cj=cj_gt4py,
            cit=cit_gt4py,
            ssi=ssi_gt4py,
            rct=rct_gt4py,
            rit=rit_gt4py, 
            rc_beri_tnd=rc_beri_tnd_gt4py,
            ldsoft=ldsoft
        )
        
        fortran_script = "mode_ice4_fast_ri.F90"
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
        
        logging.info(f"IN mean rc_beri_tnd {rc_beri_tnd.mean()}")
                
        result = fortran_script.mode_ice4_fast_ri.ice4_fast_ri(
            kproma=SHAPE[0]*SHAPE[1]*SHAPE[2],
            ksize=SHAPE[0]*SHAPE[1]*SHAPE[2],
            c_rtmin=phyex_externals["C_RTMIN"],
            i_rtmin=phyex_externals["I_RTMIN"],
            xlbexi=phyex_externals["LBEXI"],
            xlbi=phyex_externals["LBI"],
            x0depi=phyex_externals["O0DEPI"],
            x2depi=phyex_externals["O2DEPI"],
            xdi=phyex_externals["DI"],
            ldsoft=ldsoft,
            ldcompute=ldcompute.ravel(),
            prhodref=rhodref.ravel(),
            pai=ai.ravel(),
            pcj=cj.ravel(),
            pcit=cit.ravel(),
            pssi=ssi.ravel(),
            prct=rct.ravel(),
            prit=rit.ravel(),
            prcberi=rc_beri_tnd.ravel()
        )

        rcberi_out = result[0]
        
        logging.info(f"Mean rc_beri_tnd_gt4py   {rc_beri_tnd_gt4py.mean()}")
        logging.info(f"Mean rcberi_out          {rcberi_out.mean()}")
        logging.info(f"Max abs rtol             {max(abs(rc_beri_tnd_gt4py.ravel() - rcberi_out) / abs(rcberi_out))}")
        

        assert_allclose(rc_beri_tnd_gt4py.ravel(), rcberi_out, 1e-5)

