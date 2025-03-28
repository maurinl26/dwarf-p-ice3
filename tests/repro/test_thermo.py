from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.config import GT4PyConfig, DataTypes
from ice3_gt4py.phyex_common.phyex import Phyex
from gt4py.storage import from_array
import numpy as np
from numpy.testing import assert_allclose
from pathlib import Path
import fmodpy
import unittest
from ctypes import c_float

import logging

from .env import BACKEND, REBUILD, VALIDATE_ARGS, SHAPE

class TestThermo(unittest.TestCase):
    
    def test_thermo(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, 
            rebuild=REBUILD, 
            validate_args=VALIDATE_ARGS, 
            verbose=True,
            dtypes=DataTypes(
                bool=bool, 
                float=np.float32, 
                int=np.int32)
        )

        phyex_externals = Phyex("AROME").to_externals()
        thermo_fields = compile_stencil("thermodynamic_fields", gt4py_config, phyex_externals)
        
        dt = 50.0
        
        th = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        exn = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rv = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rc = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rr = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ri = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rs = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rg = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        lv = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ls = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        cph = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        t = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        th_gt4py = from_array(
            th,
            dtype=np.float32,
            backend=BACKEND
        )
        exn_gt4py = from_array(
            exn,
            dtype=np.float32,
            backend=BACKEND
        )
        rv_gt4py = from_array(
            rv,
            dtype=np.float32,
            backend=BACKEND
        )
        rc_gt4py = from_array(
            rc,
            dtype=np.float32,
            backend=BACKEND
        )
        rr_gt4py = from_array(
            rr,
            dtype=np.float32,
            backend=BACKEND
        )
        ri_gt4py = from_array(
            ri,
            dtype=np.float32,
            backend=BACKEND
        )
        rs_gt4py = from_array(
            rs,
            dtype=np.float32,
            backend=BACKEND
        )
        rg_gt4py = from_array(
            rg,
            dtype=np.float32,
            backend=BACKEND
        )
        lv_gt4py = from_array(
            lv,
            dtype=np.float32,
            backend=BACKEND
        )
        ls_gt4py = from_array(
            ls,
            dtype=np.float32,
            backend=BACKEND
        )
        cph_gt4py = from_array(
            cph,
            dtype=np.float32,
            backend=BACKEND
        )
        t_gt4py = from_array(
            t,
            dtype=np.float32,
            backend=BACKEND
        )

        thermo_fields(
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
        )

        fortran_script = "mode_thermo.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)

        result = fortran_script.mode_thermo.latent_heat(
            nkt=SHAPE[2], 
            nijt=SHAPE[0] * SHAPE[1], 
            nktb=0, 
            nkte=SHAPE[2], 
            nijb=0, 
            nije=SHAPE[0] * SHAPE[1],
            xlvtt=phyex_externals["LVTT"], 
            xlstt=phyex_externals["LSTT"],
            xcpv=phyex_externals["CPV"], 
            xci=phyex_externals["CI"], 
            xcl=phyex_externals["CL"], 
            xtt=phyex_externals["TT"], 
            xcpd=phyex_externals["CPD"], 
            krr=6,
            prv_in=rv.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            prc_in=rc.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            pri_in=ri.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            prr=rr.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            prs=rs.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            prg=rg.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),
            pth=th.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            pexn=exn.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),
            zt=t.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            zls=ls.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            zlv=lv.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            zcph=cph.reshape(SHAPE[0]*SHAPE[1], SHAPE[2])
        )
        
        zt_out = result[0]
        zlv_out = result[1]
        zls_out = result[2]
        zcph_out = result[3]
        
        logging.info(f"Machine precision {np.finfo(float).eps}")
        
        logging.info(f"Mean t_gt4py         {t_gt4py.mean()}")
        logging.info(f"Mean zt_out          {zt_out.mean()}")
        logging.info(f"Max abs err zt       {max(abs(t_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - zt_out) / abs(zt_out))}")

        logging.info(f"Mean lv_gt4py        {lv_gt4py.mean()}")
        logging.info(f"Mean zlv_out         {zlv_out.mean()}")
        logging.info(f"Max abs err rcautr   {max(abs(lv_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - zlv_out) / abs(zlv_out))}")

        logging.info(f"Mean ls_gt4py        {ls_gt4py.mean()}")
        logging.info(f"Mean zls_out         {zls_out.mean()}")
        logging.info(f"Max abs err ls       {max(abs(ls_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - zls_out) / abs(zls_out))}")

        logging.info(f"Mean cph_gt4py       {cph_gt4py.mean()}")
        logging.info(f"Mean cph_out         {zcph_out.mean()}")
        logging.info(f"Max abs err ris      {max(abs(cph_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]) - zcph_out) / abs(zcph_out))}")

        assert_allclose(zt_out, t_gt4py, rtol=1e-6)
        assert_allclose(zlv_out, lv_gt4py, rtol=1e-6)
        assert_allclose(zls_out, ls_gt4py, rtol=1e-6)
        assert_allclose(zcph_out, cph_gt4py, rtol=1e-6)
        
        