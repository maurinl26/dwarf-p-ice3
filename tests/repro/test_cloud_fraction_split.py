from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.config import GT4PyConfig, DataTypes
from ice3_gt4py.phyex_common.phyex import Phyex
from gt4py.storage import from_array
import numpy as np
from numpy.testing import assert_allclose
from pathlib import Path
import fmodpy
import unittest
import pytest
from ctypes import c_float, c_double

import logging 

from .conftest import BACKEND, REBUILD, VALIDATE_ARGS, SHAPE, compile_fortran_stencil


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", ["numpy", "gt:cpu_ifirst", "gt:cpu_kfirst"])
def test_thermo(gt4py_config, externals, fortran_dims, precision, backend, grid):
    
         # Setting backend and precision
        gt4py_config.backend = backend
        gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)
        logging.info(f"GT4PyConfig types {gt4py_config.dtypes}")
        
        F2Py_Mapping = {
            "prv":"rv", 
            "prc":"rc", 
            "pri":"ri", 
            "prr":"rr", 
            "prs":"rs", 
            "prg":"rg",
            "pth":"th", 
            "pexn":"exn",
            "zt":"t", 
            "zls":"ls", 
            "zlv":"lv", 
            "zcph":"cph",
        }
        
        Py2F_Mapping =  dict(map(reversed, F2Py_Mapping.items()))

        externals_mapping = {
            "xlvtt":"LVTT", 
            "xlstt":"LSTT",
            "xcpv":"CPV", 
            "xci":"CI", 
            "xcl":"CL", 
            "xtt":"TT", 
            "xcpd":"CPD",
        }
        
        fortran_externals = {
            fname: externals[pyname]
            for fname, pyname in externals_mapping.items()
        }
        
        
        # Compilation of both gt4py and fortran stencils
        fortran_stencil = compile_fortran_stencil(
        "mode_thermo.F90", "mode_thermo", "latent_heat"
        )
        thermo_fields = compile_stencil("thermodynamic_fields", gt4py_config, externals)
        
        
        FloatFieldsIJK_Names = [
            "th",
        "exn",
        "rv",
        "rc",
        "rr",
        "ri",
        "rs",
        "rg",
        "lv",
        "ls",
        "cph",
        "t", 
        ]
        
        FloatFieldsIJK = {
            name: np.array(
                np.random.rand(*grid.shape),
                dtype=(c_float if gt4py_config.dtypes.float == np.float32 else c_double),
                order="F",
            ) for name in FloatFieldsIJK_Names
        }
        
        

        
        th_gt4py = from_array(
            FloatFieldsIJK["th"],
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        exn_gt4py = from_array(
            FloatFieldsIJK["exn"],
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        rv_gt4py = from_array(
            FloatFieldsIJK["rv"],
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        rc_gt4py = from_array(
            FloatFieldsIJK["rc"],
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        rr_gt4py = from_array(
            FloatFieldsIJK["rr"],
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        ri_gt4py = from_array(
            FloatFieldsIJK["ri"],
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        rs_gt4py = from_array(
            FloatFieldsIJK["rs"],
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        rg_gt4py = from_array(
            FloatFieldsIJK["rg"],
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        lv_gt4py = from_array(
            FloatFieldsIJK["lv"],
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        ls_gt4py = from_array(
            FloatFieldsIJK["ls"],
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        cph_gt4py = from_array(
            FloatFieldsIJK["cph"],
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        t_gt4py = from_array(
            FloatFieldsIJK["t"],
            dtype=gt4py_config.dtypes.float,
            backend=gt4py_config.backend
        )
        
        Fortran_FloatFieldsIJK = {
            Py2F_Mapping[name]: field.reshape(grid.shape[0]*grid.shape[1], grid.shape[2])
            for name, field in FloatFieldsIJK.items()
        }

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


        result = fortran_stencil(
            krr=6,
            **Fortran_FloatFieldsIJK,
            **fortran_externals,
            **fortran_dims
        )
        
        Fields_OutNames = ['zt', 'zlv', 'zls', 'zcph']
        Fields_Out = {
            name: result[i] for i, name in enumerate(Fields_OutNames)
        }
        
        logging.info(f"Machine precision {np.finfo(float).eps}")
                    
        
        logging.info(f"Mean t_gt4py         {t_gt4py.mean()}")
        logging.info(f"Mean zt_out          {Fields_Out[Py2F_Mapping['t']].mean()}")

        logging.info(f"Mean lv_gt4py        {lv_gt4py.mean()}")
        logging.info(f"Mean zlv_out         {Fields_Out['zlv'].mean()}")

        logging.info(f"Mean ls_gt4py        {ls_gt4py.mean()}")
        logging.info(f"Mean zls_out         {Fields_Out['zls'].mean()}")

        logging.info(f"Mean cph_gt4py       {cph_gt4py.mean()}")
        logging.info(f"Mean cph_out         {Fields_Out['zcph'].mean()}")

        assert_allclose(Fields_Out['zt'], t_gt4py.reshape(grid.shape[0] * grid.shape[1], grid.shape[2]), rtol=1e-6)
        assert_allclose(Fields_Out['zlv'], lv_gt4py.reshape(grid.shape[0] * grid.shape[1], grid.shape[2]), rtol=1e-6)
        assert_allclose(Fields_Out['zls'], ls_gt4py.reshape(grid.shape[0] * grid.shape[1], grid.shape[2]), rtol=1e-6)
        assert_allclose(Fields_Out['zcph'], cph_gt4py.reshape(grid.shape[0] * grid.shape[1], grid.shape[2]), rtol=1e-6)

class TestCloudFraction(unittest.TestCase):
    
    def test_cloud_fraction_1(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, 
            rebuild=REBUILD, 
            validate_args=VALIDATE_ARGS, 
            verbose=True,
            dtypes=DataTypes(bool=bool, float=np.float32, int=np.int32)
        )

        phyex_externals = Phyex("AROME").to_externals()
        phyex_externals["LSUBG_COND"] = True       
        cloud_fraction_1 = compile_stencil("cloud_fraction_1", gt4py_config, phyex_externals)
        
        dt = np.float32(50.0)
        
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
        exnref = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rc = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ri = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ths = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rvs = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rcs = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ris = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rc_tmp = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ri_tmp = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        dt = np.float32(50.0)
        
        
        lv_gt4py = from_array(lv,
                backend=BACKEND, dtype=np.float32
            )
        ls_gt4py = from_array(ls,
                backend=BACKEND,dtype=np.float32
            )
        cph_gt4py = from_array(cph,
                backend=BACKEND,dtype=np.float32
            )
        exnref_gt4py = from_array(exnref,
                backend=BACKEND,dtype=np.float32
            )
        rc_gt4py = from_array(rc,
                backend=BACKEND,dtype=np.float32
            )
        ri_gt4py = from_array(ri,
                backend=BACKEND,dtype=np.float32
            )
        ths_gt4py = from_array(ths,
                backend=BACKEND,dtype=np.float32
            )
        rvs_gt4py = from_array(rvs,
                backend=BACKEND,dtype=np.float32
            )
        rcs_gt4py = from_array(rcs,
                backend=BACKEND,dtype=np.float32
            )
        ris_gt4py = from_array(ris,
                backend=BACKEND,dtype=np.float32
            )
        rc_tmp_gt4py = from_array(rc_tmp,
                backend=BACKEND,dtype=np.float32
            )
        ri_tmp_gt4py = from_array(ri_tmp,
                backend=BACKEND,dtype=np.float32
            )

        cloud_fraction_1(
            lv=lv_gt4py,
            ls=ls_gt4py,
            cph=cph_gt4py,
            exnref=exnref_gt4py,
            rc=rc_gt4py,
            ri=ri_gt4py,
            ths=ths_gt4py,
            rvs=rvs_gt4py,
            rcs=rcs_gt4py,
            ris=ris_gt4py,
            rc_tmp=rc_tmp_gt4py,
            ri_tmp=ri_tmp_gt4py,
            dt=dt,
        )

        fortran_script = "mode_cloud_fraction_split.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
        
        logging.info(f"SUBG_MF_PDF  : {phyex_externals["SUBG_MF_PDF"]}")
        logging.info(f"LSUBG_COND   : {phyex_externals["LSUBG_COND"]}")

        result = fortran_script.mode_cloud_fraction_split.cloud_fraction_1(
            nijt=SHAPE[0] * SHAPE[1], 
            nkt=SHAPE[2],
            nktb=1,
            nkte=SHAPE[2], 
            nijb=1, 
            nije=SHAPE[0]*SHAPE[1],
            ptstep=dt,
            zrc=rc_tmp.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            zri=ri_tmp.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),
            pexnref=exnref.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
            zcph=cph.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),
            zlv=lv.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),
            zls=ls.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),
            prc=rc.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),    
            pri=ri.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),  
            prvs=rvs.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),    
            prcs=rcs.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),    
            pths=ths.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]),    
            pris=ris.reshape(SHAPE[0]*SHAPE[1], SHAPE[2]), 
        )
        
        pths_out = result[0] 
        prvs_out=result[1] 
        prcs_out=result[2]
        pris_out=result[3]
        
        logging.info(f"Machine precision {np.finfo(float).eps}")
        
        logging.info(f"Mean ths_gt4py       {ths_gt4py.mean()}")
        logging.info(f"Mean pths_out        {pths_out.mean()}")

        logging.info(f"Mean rvs_gt4py       {rvs_gt4py.mean()}")
        logging.info(f"Mean prvs_out        {prvs_out.mean()}")

        logging.info(f"Mean rcs_gt4py       {rcs_gt4py.mean()}")
        logging.info(f"Mean prcs_out        {prcs_out.mean()}")

        logging.info(f"Mean ris_gt4py       {ris_gt4py.mean()}")
        logging.info(f"Mean pris_out        {pris_out.mean()}")
        
        assert_allclose(pths_out, ths_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(prvs_out, rvs_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(prcs_out, rcs_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(pris_out, ris_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)

    def test_cloud_fraction_2(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
            backend=BACKEND, 
            rebuild=REBUILD, 
            validate_args=VALIDATE_ARGS, 
            verbose=True,
            dtypes=DataTypes(bool=bool, float=np.float32, int=np.int32)
        )

        phyex_externals = Phyex("AROME").to_externals()
        phyex_externals["LSUBG_COND"] = True       
        cloud_fraction_2 = compile_stencil("cloud_fraction_2", gt4py_config, phyex_externals)
        
        dt = np.float32(50.0)
        
        rhodref = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        exnref= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        t= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        cph= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        lv= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        ls= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        ths = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        rvs = np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        rcs= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        ris= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        rc_mf= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        ri_mf= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        cf_mf= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        cldfr= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        hlc_hrc= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        hlc_hcf= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        hli_hri= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
        hli_hcf= np.array(
                np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                dtype=c_float,
                order="F",
            )
        
    
        rhodref_gt4py = from_array(rhodref, backend=BACKEND, dtype=np.float32)
        exnref_gt4py = from_array(exnref, backend=BACKEND, dtype=np.float32)
        t_gt4py = from_array(t, backend=BACKEND, dtype=np.float32)
        cph_gt4py = from_array(cph, backend=BACKEND, dtype=np.float32)
        lv_gt4py = from_array(lv, backend=BACKEND, dtype=np.float32)
        ls_gt4py = from_array(ls, backend=BACKEND, dtype=np.float32)
        ths_gt4py = from_array(ths, backend=BACKEND, dtype=np.float32)
        rvs_gt4py = from_array(rvs, backend=BACKEND, dtype=np.float32)
        rcs_gt4py = from_array(rcs, backend=BACKEND, dtype=np.float32)
        ris_gt4py = from_array(ris, backend=BACKEND, dtype=np.float32)
        rc_mf_gt4py = from_array(rc_mf, backend=BACKEND, dtype=np.float32)
        ri_mf_gt4py = from_array(ri_mf, backend=BACKEND, dtype=np.float32)
        cf_mf_gt4py = from_array(cf_mf, backend=BACKEND, dtype=np.float32)
        cldfr_gt4py = from_array(cldfr, backend=BACKEND, dtype=np.float32)
        hlc_hrc_gt4py = from_array(hlc_hrc, backend=BACKEND, dtype=np.float32)
        hlc_hcf_gt4py = from_array(hlc_hcf, backend=BACKEND, dtype=np.float32)
        hli_hri_gt4py = from_array(hli_hri, backend=BACKEND, dtype=np.float32)
        hli_hcf_gt4py = from_array(hli_hcf, backend=BACKEND, dtype=np.float32)
        
        cloud_fraction_2(
            rhodref=rhodref_gt4py,
            exnref=exnref_gt4py,
            t=t_gt4py,
            cph=cph_gt4py,
            lv=lv_gt4py,
            ls=ls_gt4py,
            ths=ths_gt4py,
            rvs=rvs_gt4py,
            rcs=rcs_gt4py,
            ris=ris_gt4py,
            rc_mf=rc_mf_gt4py,
            ri_mf=ri_mf_gt4py,
            cf_mf=cf_mf_gt4py,
            cldfr=cldfr_gt4py,
            hlc_hrc=hlc_hrc_gt4py,
            hlc_hcf=hlc_hcf_gt4py,
            hli_hri=hli_hri_gt4py,
            hli_hcf=hli_hcf_gt4py,
            dt=dt
        )

        fortran_script = "mode_cloud_fraction_split.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
        
        logging.info(f"SUBG_MF_PDF  : {phyex_externals["SUBG_MF_PDF"]}")
        logging.info(f"LSUBG_COND   : {phyex_externals["LSUBG_COND"]}")

        result = fortran_script.mode_cloud_fraction_split.cloud_fraction_2(
            nijt=SHAPE[0] * SHAPE[1], 
            nkt=SHAPE[2],                              
            nkte=SHAPE[2], 
            nktb=1,                                   
            nijb=1, 
            nije=SHAPE[0] * SHAPE[1],                                   
            xcriautc=phyex_externals["CRIAUTC"], 
            xcriauti=phyex_externals["CRIAUTI"], 
            xacriauti=phyex_externals["ACRIAUTI"], 
            xbcriauti=phyex_externals["BCRIAUTI"], 
            xtt=phyex_externals["TT"],
            csubg_mf_pdf=phyex_externals["SUBG_MF_PDF"], 
            lsubg_cond=phyex_externals["LSUBG_COND"],                             
            ptstep=dt,                                       
            pexnref=exnref.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), 
            prhodref=rhodref.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),
            zcph=cph.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),                      
            zlv=lv.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), 
            zls=ls.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), 
            zt=t.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),                                 
            pcf_mf=cf_mf.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), 
            prc_mf=rc_mf.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), 
            pri_mf=ri_mf.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),                                                             
            pths=ths.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), 
            prvs=rvs.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), 
            prcs=rcs.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), 
            pris=ris.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),                       
            pcldfr=cldfr.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]),                                       
            phlc_hrc=hlc_hrc.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), 
            phlc_hcf=hlc_hcf.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), 
            phli_hri=hli_hri.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), 
            phli_hcf=hli_hcf.reshape(SHAPE[0] * SHAPE[1], SHAPE[2])
        )
        
        pcldfr_out = result[0] 
        phlc_hrc_out = result[1] 
        phlc_hcf_out = result[2]
        phli_hri_out = result[3]
        phli_hcf_out = result[4]
        
        logging.info(f"Machine precision {np.finfo(float).eps}")
        
        logging.info(f"Mean cldfr_gt4py     {cldfr_gt4py.mean()}")
        logging.info(f"Mean pthpcldfr_out   {pcldfr_out.mean()}")

        logging.info(f"Mean hlc_hrc_gt4py   {hlc_hrc_gt4py.mean()}")
        logging.info(f"Mean phlc_hrc_out    {phlc_hrc_out.mean()}")

        logging.info(f"Mean hlc_hcf_gt4py   {hlc_hcf_gt4py.mean()}")
        logging.info(f"Mean phlc_hcf_out    {phlc_hcf_out.mean()}")
        
        logging.info(f"Mean hli_hri_gt4py   {hli_hri_gt4py.mean()}")
        logging.info(f"Mean phli_hri_out    {phli_hri_out.mean()}")

        logging.info(f"Mean hli_hcf_gt4py   {hli_hcf_gt4py.mean()}")
        logging.info(f"Mean phli_hcf        {phli_hcf_out.mean()}")
        
        assert_allclose(pcldfr_out, cldfr_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(phlc_hcf_out, hlc_hcf_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(phlc_hrc_out, hlc_hrc_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(phli_hri_out, hli_hri_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)
        assert_allclose(phli_hcf_out, hli_hcf_gt4py.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), rtol=1e-6)

        
   