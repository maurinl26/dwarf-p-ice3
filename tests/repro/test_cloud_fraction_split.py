from ifs_physics_common.framework.stencil import compile_stencil
from gt4py.storage import from_array
import numpy as np
from numpy.testing import assert_allclose
import pytest
from ctypes import c_float, c_double

import logging 

from conftest import compile_fortran_stencil, get_backends 


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_thermo(gt4py_config, externals, fortran_dims, precision, backend, grid, origin):
    
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
            domain=grid.shape,
            origin=origin
        )


        result = fortran_stencil(
            krr=6,
            **Fortran_FloatFieldsIJK,
            **fortran_externals,
            **fortran_dims,
        )
        
        Fields_OutNames = ['zt', 'zlv', 'zls', 'zcph']
        Fields_Out = {
            name: result[i] for i, name in enumerate(Fields_OutNames)
        }
        
        logging.info(f"Machine precision {np.finfo(float).eps}")
        
        for fname in Fields_OutNames:
            logging.info(f"{F2Py_Mapping[fname]} :: Mean gt4py      {FloatFieldsIJK[F2Py_Mapping[fname]].mean()}")
            logging.info(f"{F2Py_Mapping[fname]} :: Mean fortran    {Fields_Out[fname].mean()}")
        
        assert_allclose(Fields_Out['zt'], t_gt4py.reshape(grid.shape[0] * grid.shape[1], grid.shape[2]), rtol=1e-6)
        assert_allclose(Fields_Out['zlv'], lv_gt4py.reshape(grid.shape[0] * grid.shape[1], grid.shape[2]), rtol=1e-6)
        assert_allclose(Fields_Out['zls'], ls_gt4py.reshape(grid.shape[0] * grid.shape[1], grid.shape[2]), rtol=1e-6)
        assert_allclose(Fields_Out['zcph'], cph_gt4py.reshape(grid.shape[0] * grid.shape[1], grid.shape[2]), rtol=1e-6)
        
        
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_cloud_fraction_1(gt4py_config, externals, fortran_dims, precision, backend, grid, origin):
    
         # Setting backend and precision
        gt4py_config.backend = backend
        gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)
        logging.info(f"GT4PyConfig types {gt4py_config.dtypes}")

        externals["LSUBG_COND"] = True       
        cloud_fraction_1 = compile_stencil("cloud_fraction_1", gt4py_config, externals)
        
        dt = gt4py_config.dtypes.float(50.0)
        
        FloatFieldsIJK_Names = [
             "lv",
            "ls",
            "cph",
            "exnref",
            "rc",
            "ri",
            "ths",
            "rvs",
            "rcs",
            "ris",
            "rc_tmp",
            "ri_tmp",
        ]
        
        FloatFieldsIJK = {
            name:np.array(
                np.random.rand(*grid.shape),
                dtype=(c_float if gt4py_config.dtypes.float == np.float32 else c_double),
                order="F",
            ) for name in FloatFieldsIJK_Names
        }
        
        lv_gt4py = from_array(FloatFieldsIJK["lv"],
                backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float
            )
        ls_gt4py = from_array(FloatFieldsIJK["ls"],
                backend=gt4py_config.backend,dtype=gt4py_config.dtypes.float
            )
        cph_gt4py = from_array(FloatFieldsIJK["cph"],
                backend=gt4py_config.backend,dtype=gt4py_config.dtypes.float
            )
        exnref_gt4py = from_array(FloatFieldsIJK["exnref"],
                backend=gt4py_config.backend,dtype=gt4py_config.dtypes.float
            )
        rc_gt4py = from_array(FloatFieldsIJK["rc"],
                backend=gt4py_config.backend,dtype=gt4py_config.dtypes.float
            )
        ri_gt4py = from_array(FloatFieldsIJK["ri"],
                backend=gt4py_config.backend,dtype=gt4py_config.dtypes.float
            )
        ths_gt4py = from_array(FloatFieldsIJK["ths"],
                backend=gt4py_config.backend,dtype=gt4py_config.dtypes.float
            )
        rvs_gt4py = from_array(FloatFieldsIJK["rvs"],
                backend=gt4py_config.backend,dtype=gt4py_config.dtypes.float
            )
        rcs_gt4py = from_array(FloatFieldsIJK["rcs"],
                backend=gt4py_config.backend,dtype=gt4py_config.dtypes.float
            )
        ris_gt4py = from_array(FloatFieldsIJK["ris"],
                backend=gt4py_config.backend,dtype=gt4py_config.dtypes.float
            )
        rc_tmp_gt4py = from_array(FloatFieldsIJK["rc_tmp"],
                backend=gt4py_config.backend,dtype=gt4py_config.dtypes.float
            )
        ri_tmp_gt4py = from_array(FloatFieldsIJK["ri_tmp"],
                backend=gt4py_config.backend,dtype=gt4py_config.dtypes.float
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
            domain=grid.shape,
            origin=origin
        )
        
        fortran_stencil = compile_fortran_stencil("mode_cloud_fraction_split.F90", "mode_cloud_fraction_split", "cloud_fraction_1")
        logging.info(f"SUBG_MF_PDF  : {externals["SUBG_MF_PDF"]}")
        logging.info(f"LSUBG_COND   : {externals["LSUBG_COND"]}")
        
        F2Py_Mapping = {
            "zrc":"rc_tmp", 
            "zri":"ri_tmp",
            "pexnref":"exnref", 
            "zcph":"cph",
            "zlv":"lv",
            "zls":"ls",
            "prc":"rc",    
            "pri":"ri",  
            "prvs":"rvs",    
            "prcs":"rcs",    
            "pths":"ths",    
            "pris":"ris", 
        }
        
        Py2F_Mapping = dict(map(reversed, F2Py_Mapping.items()))
        
        Fortran_FloatFieldsIJK = {
            Py2F_Mapping[name]: field.reshape(grid.shape[0]*grid.shape[1], grid.shape[2])
            for name, field in FloatFieldsIJK.items()
        }

        result = fortran_stencil(
            ptstep=dt,
            **Fortran_FloatFieldsIJK,
            **fortran_dims
        )
        
        FieldsOut_Names = ["pths", "prvs", "prcs", "pris"]
        
        FieldsOut = {
            name: result[i] for i, name in enumerate(FieldsOut_Names)
        }
        
        logging.info(f"Machine precision {np.finfo(float).eps}")
        
        logging.info(f"Mean ths_gt4py       {ths_gt4py.mean()}")
        logging.info(f"Mean pths_out        {FieldsOut['pths'].mean()}")

        logging.info(f"Mean rvs_gt4py       {rvs_gt4py.mean()}")
        logging.info(f"Mean prvs_out        {FieldsOut['prvs'].mean()}")

        logging.info(f"Mean rcs_gt4py       {rcs_gt4py.mean()}")
        logging.info(f"Mean prcs_out        {FieldsOut['prcs'].mean()}")

        logging.info(f"Mean ris_gt4py       {ris_gt4py.mean()}")
        logging.info(f"Mean pris_out        {FieldsOut['pris'].mean()}")
        
        assert_allclose(FieldsOut["pths"], ths_gt4py.reshape(grid.shape[0]*grid.shape[1], grid.shape[2]), rtol=1e-6)
        assert_allclose(FieldsOut["prvs"], rvs_gt4py.reshape(grid.shape[0]*grid.shape[1], grid.shape[2]), rtol=1e-6)
        assert_allclose(FieldsOut["prcs"], rcs_gt4py.reshape(grid.shape[0]*grid.shape[1], grid.shape[2]), rtol=1e-6)
        assert_allclose(FieldsOut["pris"], ris_gt4py.reshape(grid.shape[0]*grid.shape[1], grid.shape[2]), rtol=1e-6)
        

@pytest.mark.parametrize("subg_mf_pdf", [0, 1])
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_cloud_fraction_2(gt4py_config, externals, fortran_dims, precision, backend, grid, origin, subg_mf_pdf):
        
        # Setting backend and precision
        gt4py_config.backend = backend
        gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)
        
        logging.info(f"GT4PyConfig types {gt4py_config.dtypes}")
        externals["LSUBG_COND"] = True 
        externals.update({
            "SUBG_MF_PDF": subg_mf_pdf
        })      
        
        # Fortran and GT4Py stencils compilation
        cloud_fraction_2 = compile_stencil("cloud_fraction_2", gt4py_config, externals)
        fortran_stencil = compile_fortran_stencil("mode_cloud_fraction_split.F90", "mode_cloud_fraction_split", "cloud_fraction_2")
        
        dt = gt4py_config.dtypes.float(50.0)
        
        FloatFieldsIJK_Names = [
            "rhodref",
            "exnref",
            "t",
            "cph",
            "lv",
            "ls",
            "ths",
            "rvs",
            "rcs",
            "ris",
            "rc_mf",
            "ri_mf",
            "cf_mf",
            "cldfr",
            "hlc_hrc",
            "hlc_hcf",
            "hli_hri",
            "hli_hcf",
        ]
        
        FloatFieldsIJK = {
            name: np.array(
                np.random.rand(*grid.shape),
                dtype=(c_float if gt4py_config.dtypes.float == np.float32 else c_double),
                order="F",
            ) for name in FloatFieldsIJK_Names
        }
        
        rhodref_gt4py = from_array(FloatFieldsIJK["rhodref"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        exnref_gt4py = from_array(FloatFieldsIJK["exnref"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        t_gt4py = from_array(FloatFieldsIJK["t"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        cph_gt4py = from_array(FloatFieldsIJK["cph"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        lv_gt4py = from_array(FloatFieldsIJK["lv"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        ls_gt4py = from_array(FloatFieldsIJK["ls"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        ths_gt4py = from_array(FloatFieldsIJK["ths"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        rvs_gt4py = from_array(FloatFieldsIJK["rvs"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        rcs_gt4py = from_array(FloatFieldsIJK["rcs"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        ris_gt4py = from_array(FloatFieldsIJK["ris"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        rc_mf_gt4py = from_array(FloatFieldsIJK["rc_mf"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        ri_mf_gt4py = from_array(FloatFieldsIJK["ri_mf"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        cf_mf_gt4py = from_array(FloatFieldsIJK["cf_mf"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        cldfr_gt4py = from_array(FloatFieldsIJK["cldfr"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        hlc_hrc_gt4py = from_array(FloatFieldsIJK["hlc_hrc"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        hlc_hcf_gt4py = from_array(FloatFieldsIJK["hlc_hcf"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        hli_hri_gt4py = from_array(FloatFieldsIJK["hli_hri"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        hli_hcf_gt4py = from_array(FloatFieldsIJK["hli_hcf"], backend=gt4py_config.backend, dtype=gt4py_config.dtypes.float)
        
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
            dt=dt,
            domain=grid.shape,
            origin=origin
        )
        
        logging.info(f"SUBG_MF_PDF  : {externals["SUBG_MF_PDF"]}")
        logging.info(f"LSUBG_COND   : {externals["LSUBG_COND"]}")
        
        keys_mapping = {
            "xcriautc":"CRIAUTC", 
            "xcriauti":"CRIAUTI", 
            "xacriauti":"ACRIAUTI", 
            "xbcriauti":"BCRIAUTI", 
            "xtt":"TT",
            "csubg_mf_pdf":"SUBG_MF_PDF", 
            "lsubg_cond":"LSUBG_COND",
        }
        
        fortran_externals = {
            key: externals[value]
            for key, value in keys_mapping.items()
        }
        
        logging.info(f"csubg_mf_pdf : {fortran_externals['csubg_mf_pdf']}")
        
        from ice3_gt4py.phyex_common.param_ice import SubGridMassFluxPDF
        logging.info(f"csubg_mf_pdf : {SubGridMassFluxPDF(fortran_externals['csubg_mf_pdf'])}")
        logging.info(f"lsubg_cond   : {fortran_externals['lsubg_cond']}")
        
        F2Py_Mapping = {
            "pexnref":"exnref", 
            "prhodref":"rhodref",
            "zcph":"cph",                      
            "zlv":"lv", 
            "zls":"ls", 
            "zt":"t",                                 
            "pcf_mf":"cf_mf", 
            "prc_mf":"rc_mf", 
            "pri_mf":"ri_mf",                                                             
            "pths":"ths", 
            "prvs":"rvs", 
            "prcs":"rcs", 
            "pris":"ris",                       
            "pcldfr":"cldfr",                                       
            "phlc_hrc":"hlc_hrc", 
            "phlc_hcf":"hlc_hcf", 
            "phli_hri":"hli_hri", 
            "phli_hcf":"hli_hcf",
        }
        
        Py2F_Mapping =  dict(map(reversed, F2Py_Mapping.items()))

        Fortran_FloatFieldsIJK = {
            Py2F_Mapping[name]: field.reshape(grid.shape[0]*grid.shape[1], grid.shape[2])
            for name, field in FloatFieldsIJK.items()
        }
        

        result = fortran_stencil(                                                          
            ptstep=dt,                                       
            **Fortran_FloatFieldsIJK,
            **fortran_dims,
            **fortran_externals
        )
        
        pths_out = result[0]
        prvs_out = result[1]
        prcs_out = result[2]
        pris_out = result[3]
        
        pcldfr_out = result[4] 
        phlc_hrc_out = result[5] 
        phlc_hcf_out = result[6]
        phli_hri_out = result[7]
        phli_hcf_out = result[8]
        
        logging.info(f"Machine precision {np.finfo(float).eps}")
        
        logging.info(f"Mean cldfr_gt4py     {cldfr_gt4py.mean()}")
        logging.info(f"Mean pcldfr_out      {pcldfr_out.mean()}")

        logging.info(f"Mean hlc_hrc_gt4py   {hlc_hrc_gt4py.mean()}")
        logging.info(f"Mean phlc_hrc_out    {phlc_hrc_out.mean()}")

        logging.info(f"Mean hlc_hcf_gt4py   {hlc_hcf_gt4py.mean()}")
        logging.info(f"Mean phlc_hcf_out    {phlc_hcf_out.mean()}")
        
        logging.info(f"Mean hli_hri_gt4py   {hli_hri_gt4py.mean()}")
        logging.info(f"Mean phli_hri_out    {phli_hri_out.mean()}")

        logging.info(f"Mean hli_hcf_gt4py   {hli_hcf_gt4py.mean()}")
        logging.info(f"Mean phli_hcf        {phli_hcf_out.mean()}")
        
        assert_allclose(pcldfr_out, cldfr_gt4py.reshape(grid.shape[0]*grid.shape[1], grid.shape[2]), rtol=1e-6)
        assert_allclose(phlc_hcf_out, hlc_hcf_gt4py.reshape(grid.shape[0]*grid.shape[1], grid.shape[2]), rtol=1e-6)
        assert_allclose(phlc_hrc_out, hlc_hrc_gt4py.reshape(grid.shape[0]*grid.shape[1], grid.shape[2]), rtol=1e-6)
        assert_allclose(phli_hri_out, hli_hri_gt4py.reshape(grid.shape[0]*grid.shape[1], grid.shape[2]), rtol=1e-6)
        assert_allclose(phli_hcf_out, hli_hcf_gt4py.reshape(grid.shape[0]*grid.shape[1], grid.shape[2]), rtol=1e-6)

   