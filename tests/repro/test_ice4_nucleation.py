from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.config import GT4PyConfig
from ice3_gt4py.phyex_common.phyex import Phyex
from gt4py.storage import from_array
import numpy as np
from pathlib import Path
import fmodpy
import unittest
from numpy.testing import assert_allclose
import logging
from .conftest import BACKEND, VALIDATE_ARGS, REBUILD, SHAPE


class TestIce4Nucleation(unittest.TestCase):
    def test_ice4_nucleation(self):
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, 
            rebuild=REBUILD, 
            validate_args=VALIDATE_ARGS, 
            verbose=True
        )

        phyex_externals = Phyex("AROME").to_externals()
        ice4_nucleation_gt4py = compile_stencil("ice4_nucleation", gt4py_config, phyex_externals)

        ldcompute = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=bool,
                order="F",
            )
        tht = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        pabst = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        rhodref = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        exn = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        lsfact = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        t = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        rvt = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        cit = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        rvheni_mr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        ssi = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        ldcompute_gt4py = from_array(ldcompute,
            dtype=np.bool_,
            backend=BACKEND,
        )  
        tht_gt4py = from_array(tht,
            dtype=float,
            backend=BACKEND,
        )  
        pabst_gt4py = from_array(
            pabst,
            dtype=float,
            backend=BACKEND,
        )  
        rhodref_gt4py = from_array(
            rhodref,
            dtype=float,
            backend=BACKEND,
        )  
        exn_gt4py = from_array(
            exn,
            dtype=float,
            backend=BACKEND,
        )  
        lsfact_gt4py = from_array(
            lsfact,
            dtype=float,
            backend=BACKEND,
        )  
        t_gt4py = from_array(
            t,
            dtype=float,
            backend=BACKEND,
        )  
        rvt_gt4py = from_array(
            rvt,
            dtype=float,
            backend=BACKEND,
        )  
        cit_gt4py = from_array(
            cit,
            dtype=float,
            backend=BACKEND,
        )  
        rvheni_mr_gt4py = from_array(
            rvheni_mr,
            dtype=float,
            backend=BACKEND,
        )  
        ssi_gt4py = from_array(
            ssi,
            dtype=float,
            backend=BACKEND,
        )  
        

        ice4_nucleation_gt4py(
            ldcompute=ldcompute_gt4py,
            tht=tht_gt4py,
            pabst=pabst_gt4py,
            rhodref=rhodref_gt4py,
            exn=exn_gt4py,
            lsfact=lsfact_gt4py,
            t=t_gt4py,
            rvt=rvt_gt4py,
            cit=cit_gt4py,
            rvheni_mr=rvheni_mr_gt4py,
            ssi=ssi_gt4py,
        )

        fortran_script = "mode_ice4_nucleation.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)

        result = fortran_script.mode_ice4_nucleation.ice4_nucleation(
                xtt=phyex_externals["TT"], 
                v_rtmin=phyex_externals["V_RTMIN"],
                xalpw=phyex_externals["ALPW"], 
                xbetaw=phyex_externals["BETAW"], 
                xgamw=phyex_externals["GAMW"],
                xalpi=phyex_externals["ALPI"], 
                xbetai=phyex_externals["BETAI"],
                xgami=phyex_externals["GAMI"], 
                xepsilo=phyex_externals["EPSILO"],
                xnu10=phyex_externals["NU10"], 
                xnu20=phyex_externals["NU20"], 
                xalpha1=phyex_externals["ALPHA1"], 
                xalpha2=phyex_externals["ALPHA2"], 
                xbeta1=phyex_externals["BETA1"], 
                xbeta2=phyex_externals["BETA2"],
                xmnu0=phyex_externals["MNU0"],
                lfeedbackt=phyex_externals["LFEEDBACKT"],
                ksize=SHAPE[0]*SHAPE[1]*SHAPE[2],
                kproma=SHAPE[0]*SHAPE[1]*SHAPE[2],
                ldcompute=ldcompute.ravel(),
                ptht=tht.ravel(),
                ppabst=pabst.ravel(),
                prhodref=rhodref.ravel(), 
                pexn=exn.ravel(), 
                plsfact=lsfact.ravel(),
                pt=t.ravel(),  
                prvt=rvt.ravel(), 
                pcit=cit.ravel(),               # INOUT
        )

        cit_out = result[0]
        rvheni_mr_out = result[1]

        logging.info(f"Machine precision {np.finfo(float).eps}")
        
        assert_allclose(cit_out, cit_gt4py.ravel(), 10e-6)
        assert_allclose(rvheni_mr_out, rvheni_mr_gt4py.ravel(), 10e-6)

