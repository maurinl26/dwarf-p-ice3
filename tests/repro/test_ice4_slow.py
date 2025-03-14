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

class TestICE4Slow(unittest.TestCase):
    
    def test_ice4_slow(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
        backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
    )
    
        phyex_externals = Phyex("AROME").to_externals()
    
        ice4_slow_gt4py = compile_stencil("ice4_slow", gt4py_config, phyex_externals)
    
        ldcompute = from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.bool_,
        backend=BACKEND
    )
        rhodref= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        t= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        ssi= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        lv_fact= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        ls_fact= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        rv_t= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        rc_t= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        ri_t= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        rs_t= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        rg_t= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        lbdas= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        lbdag= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        ai= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        cj= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        hli_hcf= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        hli_hri= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        rc_honi_tnd= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        rv_deps_tnd= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        ri_aggs_tnd= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        ri_auts_tnd= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        rv_depg_tnd= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=float,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
    
        ldsoft = True
    
        ice4_slow_gt4py(
        ldcompute=ldcompute,
        rhodref=rhodref,
        t=t,
        ssi=ssi,
        lv_fact=lv_fact,
        ls_fact=ls_fact,
        rv_t=rv_t,
        rc_t=rc_t,
        ri_t=ri_t,
        rs_t=rs_t,
        rg_t=rg_t,
        lbdas=lbdas,
        lbdag=lbdag,
        ai=ai,
        cj=cj,
        hli_hcf=hli_hcf,
        hli_hri=hli_hri,
        rc_honi_tnd=rc_honi_tnd,
        rv_deps_tnd=rv_deps_tnd,
        ri_aggs_tnd=ri_aggs_tnd,
        ri_auts_tnd=ri_auts_tnd,
        rv_depg_tnd=rv_depg_tnd,    
        ldsoft=ldsoft
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
        kproma=SHAPE[0]*SHAPE[1],
        ksize=SHAPE[2],
        ldcompute=ldcompute[...],
        prhodref=rhodref[...],
        pt=t[...],
        pssi=ssi[...],
        plvfact=lv_fact[...],
        plsfact=ls_fact[...],
        prvt=rv_t[...],
        prct=rc_t[...],
        prit=ri_t[...],
        prst=rs_t[...],
        prgt=rg_t[...],
        plbdas=lbdas[...],
        plbdag=lbdag[...],
        pai=ai[...],
        pcj=cj[...],
        phli_hcf=hli_hcf[...],
        phli_hri=hli_hri[...],
        prchoni=rc_honi_tnd[...],
        prvdeps=rv_deps_tnd[...],
        priaggs=ri_aggs_tnd[...],
        priauts=ri_auts_tnd[...],
        prvdepg=rv_depg_tnd[...],    
        ldsoft=ldsoft
    )
        
        PRCHONI = result[0]
        PRVDEPS = result[1]
        PRIAGGS = result[2]
        PRIAUTS = result[3] 
        PRVDEPG = result[4]
    
        logging.info(f"Machine precision {np.finfo(np.float64).eps}")
    
        logging.info(f"PRCHONI mean {PRCHONI.mean()}, rc_honi_tnd mean : {rc_honi_tnd.mean()} ")
        logging.info(np.allclose(PRCHONI, rc_honi_tnd, 10e-8))
        assert np.allclose(PRCHONI, rc_honi_tnd, 10e-8)
        assert np.allclose(PRVDEPS, rv_deps_tnd, 10e-8)
        assert np.allclose(PRIAGGS, ri_aggs_tnd, 10e-8)
        assert np.allclose(PRIAUTS, ri_auts_tnd, 10e-8)
        assert np.allclose(PRVDEPG, rv_depg_tnd, 10e-8)    
    
    
