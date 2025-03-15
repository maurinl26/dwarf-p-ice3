from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.config import GT4PyConfig
from ice3_gt4py.phyex_common.phyex import Phyex
from gt4py.storage import from_array
import numpy as np
from pathlib import Path
import fmodpy
import unittest
from numpy.testing import assert_allclose
from ctypes import c_float, c_int

import logging
from .env import DEFAULT_GT4PY_CONFIG, SHAPE, BACKEND


class TestICE4Slow(unittest.TestCase):
    def test_ice4_slow(self):

        phyex_externals = Phyex("AROME").to_externals()
        ice4_slow_gt4py = compile_stencil("ice4_slow", DEFAULT_GT4PY_CONFIG, phyex_externals)

        ldcompute = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=bool,
                order="F",
            )
         
        rhodref = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        t = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ssi = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        rvt = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )

        rct = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rit = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rst = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rgt = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        lbdas = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        lbdag = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ai = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        cj = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        hli_hcf = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        hli_hri = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rc_honi_tnd = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rv_deps_tnd = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ri_aggs_tnd = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ri_auts_tnd = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rv_depg_tnd = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )

        ldcompute_gt4py = from_array(ldcompute,
            dtype=bool,
            backend=BACKEND,
        )
        rhodref_gt4py = from_array(rhodref,
            dtype=float,
            backend=BACKEND,
        )
        t_gt4py = from_array(t,
            dtype=float,
            backend=BACKEND,
        )
        ssi_gt4py = from_array(ssi,
            dtype=float,
            backend=BACKEND,
        )
        
        rvt_gt4py = from_array(rvt,
            dtype=float,
            backend=BACKEND,
        )

        rct_gt4py = from_array(rct,
            dtype=float,
            backend=BACKEND,
        )
        rit_gt4py = from_array(rit,
            dtype=float,
            backend=BACKEND,
        )
        rst_gt4py = from_array(rst,
            dtype=float,
            backend=BACKEND,
        )
        rgt_gt4py = from_array(rgt,
            dtype=float,
            backend=BACKEND,
        )
        lbdas_gt4py = from_array(lbdas,
            dtype=float,
            backend=BACKEND,
        )
        lbdag_gt4py = from_array(lbdag,
            dtype=float,
            backend=BACKEND,
        )
        ai_gt4py = from_array(ai,
            dtype=float,
            backend=BACKEND,
        )
        cj_gt4py = from_array(cj,
            dtype=float,
            backend=BACKEND,
        )
        hli_hcf_gt4py = from_array(hli_hcf,
            dtype=float,
            backend=BACKEND,
        )
        hli_hri_gt4py = from_array(hli_hri,
            dtype=float,
            backend=BACKEND,
        )
        rc_honi_tnd_gt4py = from_array(rc_honi_tnd,
            dtype=float,
            backend=BACKEND,
        )
        rv_deps_tnd_gt4py = from_array(rv_deps_tnd,
            dtype=float,
            backend=BACKEND,
        )
        ri_aggs_tnd_gt4py = from_array(ri_aggs_tnd,
            dtype=float,
            backend=BACKEND,
        )
        ri_auts_tnd_gt4py = from_array(ri_auts_tnd,
            dtype=float,
            backend=BACKEND,
        )
        rv_depg_tnd_gt4py = from_array(rv_depg_tnd,
            dtype=float,
            backend=BACKEND,
        )

        ldsoft = True

        ice4_slow_gt4py(
            ldcompute=ldcompute_gt4py,
            rhodref=rhodref_gt4py,
            t=t_gt4py,
            ssi=ssi_gt4py,
            rvt=rvt_gt4py,
            rct=rct_gt4py,
            rit=rit_gt4py,
            rst=rst_gt4py,
            rgt=rgt_gt4py,
            lbdas=lbdas_gt4py,
            lbdag=lbdag_gt4py,
            ai=ai_gt4py,
            cj=cj_gt4py,
            hli_hcf=hli_hcf_gt4py,
            hli_hri=hli_hri_gt4py,
            rc_honi_tnd=rc_honi_tnd_gt4py,
            rv_deps_tnd=rv_deps_tnd_gt4py,
            ri_aggs_tnd=ri_aggs_tnd_gt4py,
            ri_auts_tnd=ri_auts_tnd_gt4py,
            rv_depg_tnd=rv_depg_tnd_gt4py,
            ldsoft=ldsoft,
        )

        fortran_script = "mode_ice4_slow.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)

        result = fortran_script.mode_ice4_slow.ice4_slow(
            xtt=phyex_externals["TT"],
            v_rtmin=phyex_externals["V_RTMIN"],
            c_rtmin=phyex_externals["C_RTMIN"],
            i_rtmin=phyex_externals["I_RTMIN"],
            s_rtmin=phyex_externals["S_RTMIN"],
            g_rtmin=phyex_externals["G_RTMIN"],
            xexiaggs=phyex_externals["EXIAGGS"],
            xfiaggs=phyex_externals["FIAGGS"],
            xcolexis=phyex_externals["COLEXIS"],
            xtimauti=phyex_externals["TIMAUTI"],
            xcriauti=phyex_externals["CRIAUTI"],
            xacriauti=phyex_externals["ACRIAUTI"],
            xbcriauti=phyex_externals["BCRIAUTI"],
            xtexauti=phyex_externals["TEXAUTI"],
            xcexvt=phyex_externals["CEXVT"],
            x0depg=phyex_externals["O0DEPG"],
            x1depg=phyex_externals["O1DEPG"],
            xex1depg=phyex_externals["EX1DEPG"],
            xhon=phyex_externals["HON"],
            xalpha3=phyex_externals["ALPHA3"],
            xex0depg=phyex_externals["EX0DEPG"],
            xbeta3=phyex_externals["BETA3"],
            x0deps=phyex_externals["O0DEPS"],
            x1deps=phyex_externals["O1DEPS"],
            xex1deps=phyex_externals["EX1DEPS"],
            xex0deps=phyex_externals["EX0DEPS"],
            kproma=SHAPE[0] * SHAPE[1] * SHAPE[2],
            ksize=SHAPE[0] * SHAPE[1] * SHAPE[2],
            ldcompute=ldcompute.ravel(),
            prhodref=rhodref.ravel(),
            pt=t.ravel(),
            pssi=ssi.ravel(),
            prvt=rvt.ravel(),
            prct=rct.ravel(),
            prit=rit.ravel(),
            prst=rst.ravel(),
            prgt=rgt.ravel(),
            plbdas=lbdas.ravel(),
            plbdag=lbdag.ravel(),
            pai=ai.ravel(),
            pcj=cj.ravel(),
            phli_hcf=hli_hcf.ravel(),
            phli_hri=hli_hri.ravel(),
            prchoni=rc_honi_tnd.ravel(),
            prvdeps=rv_deps_tnd.ravel(),
            priaggs=ri_aggs_tnd.ravel(),
            priauts=ri_auts_tnd.ravel(),
            prvdepg=rv_depg_tnd.ravel(),
            ldsoft=ldsoft,
        )

        prchoni_out = result[0]
        prvdeps_out = result[1]
        priaggs_out = result[2]
        priauts_out = result[3]
        prvdepg_out = result[4]

        logging.info(f"Machine precision {np.finfo(float).eps}")
        
        logging.info(f"Mean rcautr_gt4py    {rc_honi_tnd_gt4py.mean()}")
        logging.info(f"Mean rcautr_out      {prchoni_out.mean()}")
        logging.info(f"Max abs err rcautr   {max(abs(rc_honi_tnd_gt4py.ravel() - prchoni_out) / abs(prchoni_out))}")

        logging.info(f"Mean rcautr_gt4py    {rv_deps_tnd_gt4py.mean()}")
        logging.info(f"Mean rcautr_out      {prvdeps_out.mean()}")
        logging.info(f"Max abs err rcautr   {max(abs(rv_deps_tnd_gt4py.ravel() - prvdeps_out) / abs(prvdeps_out))}")

        logging.info(f"Mean rcautr_gt4py    {ri_aggs_tnd_gt4py.mean()}")
        logging.info(f"Mean rcautr_out      {priaggs_out.mean()}")
        logging.info(f"Max abs err rcautr   {max(abs(ri_aggs_tnd_gt4py.ravel() - priaggs_out) / abs(priaggs_out))}")

        logging.info(f"Mean rcautr_gt4py    {ri_auts_tnd_gt4py.mean()}")
        logging.info(f"Mean rcautr_out      {priauts_out.mean()}")
        logging.info(f"Max abs err rcautr   {max(abs(ri_auts_tnd_gt4py.ravel() - priauts_out) / abs(priauts_out))}")

        logging.info(f"Mean rcautr_gt4py    {rv_depg_tnd_gt4py.mean()}")
        logging.info(f"Mean rcautr_out      {prvdepg_out.mean()}")
        logging.info(f"Max abs err rcautr   {max(abs(rv_depg_tnd_gt4py.ravel() - prvdepg_out) / abs(prvdepg_out))}")


        assert_allclose(prchoni_out, rc_honi_tnd.ravel(), 10e-6)
        assert_allclose(prvdeps_out, rv_deps_tnd.ravel(), 10e-6)
        assert_allclose(priaggs_out, ri_aggs_tnd.ravel(), 10e-6)
        assert_allclose(priauts_out, ri_auts_tnd.ravel(), 10e-6)
        assert_allclose(prvdepg_out, rv_depg_tnd.ravel(), 10e-6)

