from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.config import GT4PyConfig
from ice3_gt4py.phyex_common.phyex import Phyex
from gt4py.storage import from_array
import numpy as np
from numpy.testing import assert_allclose
from pathlib import Path
import fmodpy
import unittest

import logging

from .conftest import BACKEND, REBUILD, VALIDATE_ARGS, SHAPE


class TestIce4Rimltc(unittest.TestCase):
    def test_ice4_rimltc(self):
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
        )

        phyex_externals = Phyex("AROME").to_externals()
        ice4_rimltc_gt4py = compile_stencil(
            "ice4_rimltc", gt4py_config, phyex_externals
        )

        ldcompute = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=bool,
                order="F",
            )
        t = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        exn = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        lvfact = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        lsfact = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        tht = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
        rit = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )

        rimltc_mr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=float,
                order="F",
            )
            
        ldcompute_gt4py = from_array(
            ldcompute,
            dtype=bool,
            backend=BACKEND,
        )
        t_gt4py = from_array(
            t,
            dtype=float,
            backend=BACKEND,
        )
        exn_gt4py = from_array(
            exn,
            dtype=float,
            backend=BACKEND,
        )
        lvfact_gt4py = from_array(
            lvfact,
            dtype=float,
            backend=BACKEND,
        )
        lsfact_gt4py = from_array(
            lsfact,
            dtype=float,
            backend=BACKEND,
        )
        tht_gt4py = from_array(
            tht,
            dtype=float,
            backend=BACKEND,
        )  # theta at time t
        rit_gt4py = from_array(
            rit,
            dtype=float,
            backend=BACKEND,
        )  # rain water mixing ratio at t
        rimltc_mr_gt4py = from_array(
            rimltc_mr,
            dtype=float,
            backend=BACKEND,
        )
        
        
        ice4_rimltc_gt4py(
            ldcompute=ldcompute_gt4py,
            t=t_gt4py,
            exn=exn_gt4py,
            lvfact=lvfact_gt4py,
            lsfact=lsfact_gt4py,
            tht=tht_gt4py,
            rit=rit_gt4py,
            rimltc_mr=rimltc_mr_gt4py,
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
            kproma=SHAPE[0] * SHAPE[1] * SHAPE[2],
            ksize=SHAPE[0] * SHAPE[1] * SHAPE[2],
            ldcompute=ldcompute.ravel(),
            pexn=exn.ravel(),
            plvfact=lvfact.ravel(),
            plsfact=lsfact.ravel(),
            pt=t.ravel(),
            ptht=tht.ravel(),
            prit=rit.ravel(),
            primltc_mr=rimltc_mr.ravel(),
        )

        rimltc_mr_out = result[0]

        logging.info(f"Machine precision {np.finfo(float).eps}")
        assert_allclose(rimltc_mr_out, rimltc_mr_gt4py.ravel(), rtol=10e-6)

    