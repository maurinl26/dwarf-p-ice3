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

class TestICE4Warm(unittest.TestCase):
    
    def test_ice4_warm(self):
        
        logging.info(f"With backend {BACKEND}")
        gt4py_config = GT4PyConfig(
        backend=BACKEND, rebuild=REBUILD, validate_args=VALIDATE_ARGS, verbose=True
    )
    
        phyex_externals = Phyex("AROME").to_externals()
        ice4_warm_gt4py = compile_stencil("ice4_warm", gt4py_config, phyex_externals)
    
        ldcompute = from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.bool_,
        backend=BACKEND
    )
        ldcompute= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.bool_,
        backend=BACKEND
    )  # boolean field for microphysics computation
        rhodref= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        lv_fact= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        t= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # temperature
        pres= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        tht= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )
        lbdar= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # slope parameter for the rain drop distribution
        lbdar_rf= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # slope parameter for the rain fraction part
        ka= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # thermal conductivity of the air
        dv= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # diffusivity of water vapour
        cj= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # function to compute the ventilation coefficient
        hlc_hcf= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # High Cloud Fraction in grid
        hlc_lcf= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # Low Cloud Fraction in grid
        hlc_hrc= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # LWC that is high in grid
        hlc_lrc= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # LWC that is low in grid
        cf= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # cloud fraction
        rf= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # rain fraction
        rv_t= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # water vapour mixing ratio at t
        rc_t= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # cloud water mixing ratio at t
        rr_t= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # rain water mixing ratio at t
        rcautr= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # autoconversion of rc for rr production
        rcaccr= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # accretion of r_c for r_r production
        rrevav= from_array(
        data=np.array(
                    np.random.rand(SHAPE[0], SHAPE[1], SHAPE[2]),
                    dtype=bool,
                    order="F",
                ),
        dtype=np.float64,
        backend=BACKEND
    )  # evaporation of rr
        
        ldsoft = True
    
        ice4_warm_gt4py(
            ldcompute=ldcompute,  # boolean field for microphysics computation
            rhodref=rhodref,
            lv_fact=lv_fact,
            t=t,  # temperature
            pres=pres,
            tht=tht,
            lbdar=lbdar,  # slope parameter for the rain drop distribution
            lbdar_rf=lbdar_rf,  # slope parameter for the rain fraction part
            ka=ka,  # thermal conductivity of the air
            dv=dv,  # diffusivity of water vapour
            cj=cj,  # function to compute the ventilation coefficient
            hlc_hcf=hlc_hcf,  # High Cloud Fraction in grid
            hlc_lcf=hlc_lcf,  # Low Cloud Fraction in grid
            hlc_hrc=hlc_hrc,  # LWC that is high in grid
            hlc_lrc=hlc_lrc,  # LWC that is low in grid
            cf=cf,  # cloud fraction
            rf=rf,  # rain fraction
            rv_t=rv_t, # water vapour mixing ratio at t
            rc_t=rc_t,  # cloud water mixing ratio at t
            rr_t=rr_t, # rain water mixing ratio at t
            rcautr = rcautr, # autoconversion of rc for rr production
            rcaccr = rcaccr, # accretion of r_c for r_r production
            rrevav = rrevav,  # evaporation of rr
            ldsoft=ldsoft
        )
        
        fortran_script = "mode_ice4_warm.F90"
        current_directory = Path.cwd()
        root_directory = current_directory
        stencils_directory = Path(
            root_directory, "src", "ice3_gt4py", "stencils_fortran"
        )
        script_path = Path(stencils_directory, fortran_script)

        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
    
        result = fortran_script.mode_ice4_warm.ice4_warm(
            xalpw=phyex_externals["ALPW"], 
            xbetaw=phyex_externals["BETAW"], 
            xgamw=phyex_externals["GAMW"], 
            xepsilo=phyex_externals["EPSILO"],
            xlvtt=phyex_externals["LVTT"],
            xcpv=phyex_externals["CPV"],
            xcl=phyex_externals["CL"],
            xtt=phyex_externals["TT"], 
            xrv=phyex_externals["RV"],
            xcpd=phyex_externals["CPD"],
            xtimautc=phyex_externals["TIMAUTC"], 
            xcriautc=phyex_externals["CRIAUTC"],
            xfcaccr=phyex_externals["FCACCR"], 
            xexcaccr=phyex_externals["EXCACCR"],
            x0evar=phyex_externals["O0EVAR"], 
            x1evar=phyex_externals["O1EVAR"], 
            xex0evar=phyex_externals["EX0EVAR"], 
            xex1evar=phyex_externals["EX1EVAR"],
            c_rtmin=phyex_externals["C_RTMIN"], 
            r_rtmin=phyex_externals["R_RTMIN"], 
            xcexvt=phyex_externals["CEXVT"],
            kproma=SHAPE[0]*SHAPE[1], 
            ksize=SHAPE[2],
            ldsoft=ldsoft,
            ldcompute=ldcompute,
            hsubg_rc_rr_accr=phyex_externals["SUBG_RC_RR_ACCR"], 
            hsubg_rr_evap=phyex_externals["SUBG_RR_EVAP"], 
            prhodref=rhodref, 
            plvfact=lv_fact,
            pt=t,
            ppres=pres,    
            ptht=tht,     
            plbdar=lbdar, 
            plbdar_rf=lbdar_rf,
            pka=ka,
            pdv=dv, 
            pcj=cj, 
            phlc_hcf=hlc_hcf,
            phlc_lcf=hlc_lcf, 
            phlc_hrc=hlc_hrc, 
            phlc_lrc=hlc_lrc, 
            pcf=cf, 
            prf=rf,
            prvt=rv_t,
            prct=rc_t,
            prrt=rr_t,
            prcautr=rcautr,
            prcaccr=rcaccr,
            prrevav=rrevav 
    )
        
        RCAUTR = result[0]
        RCACCR = result[1]
        RREVAV = result[2]
    
        logging.info(f"Machine precision {np.finfo(np.float64).eps}")
    
        logging.info(f"PRCHONI mean {RCAUTR.mean()}, rc_honi_tnd mean : {rcautr.mean()} ")
        assert np.allclose(RCAUTR, rcautr, 10e-8)
        assert np.allclose(RCACCR, rcaccr, 10e-8)
        assert np.allclose(RREVAV, rrevav, 10e-8)
    
    
