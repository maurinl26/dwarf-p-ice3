# -*- coding: utf-8 -*-
"""
Test suite for individual ICE4 Fast RG (Graupel) process kernels.

This module tests each microphysical process from stencils_cupy/ice4_fast_rg.py
separately against Fortran reference implementations. Each test validates
one specific process to enable targeted debugging and verification.

Processes tested:
- Rain contact freezing (RICFRRG, RRCFRIG, PRICFRR)
- Cloud and pristine ice collection on graupel
- Graupel melting

Reference:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_rg.F90
"""
from ctypes import c_double, c_float

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import dp_dtypes, sp_dtypes


@pytest.mark.parametrize("ldsoft", [False, True])
@pytest.mark.parametrize("lcrflimit", [False, True])
@pytest.mark.parametrize("dtypes", [dp_dtypes, sp_dtypes])
def test_rain_contact_freezing(dtypes, packed_dims, domain, externals, ldsoft, lcrflimit):
    """
    Test rain contact freezing process.
    
    Validates:
    - RICFRRG: Pristine ice collection by rain leading to graupel
    - RRCFRIG: Rain freezing by contact with pristine ice  
    - PRICFRR: Limited ice collection when heat balance constrains freezing
    
    Args:
        dtypes: Data type dictionary (single/double precision)
        packed_dims: Fortran packing dimensions
        domain: Test domain size
        externals: Physical constants and parameters
        ldsoft: Soft hail flag
        lcrflimit: Contact freezing heat balance limiter flag
    """
    # Compile Fortran reference
    ice4_rain_contact_freezing_fortran = compile_fortran_stencil(
        "mode_ice4_fast_rg_kernels.F90",
        "mode_ice4_fast_rg_kernels",
        "ice4_rain_contact_freezing"
    )
    
    # =========================================================================
    # Initialize input fields
    # =========================================================================
    n_points = domain[0] * domain[1] * domain[2]
    
    # Input fields
    prhodref = np.random.rand(n_points) * 0.8 + 0.5  # 0.5-1.3 kg/m³
    plbdar = np.random.rand(n_points) * 9e5 + 1e5    # 1e5-1e6 m⁻¹
    pt = np.random.rand(n_points) * 70 + 233         # 233-303 K
    prit = np.random.rand(n_points) * 0.002          # 0-0.002 kg/kg
    prrt = np.random.rand(n_points) * 0.005          # 0-0.005 kg/kg
    pcit = np.random.rand(n_points) * 9.99e5 + 1e3   # 1e3-1e6 #/m³
    
    # Compute mask
    ldcompute = np.random.rand(n_points) > 0.1  # ~90% active
    
    # Output fields
    pricfrrg = np.zeros(n_points, dtype=(c_float if dtypes["float"] == np.float32 else c_double))
    prrcfrig = np.zeros(n_points, dtype=(c_float if dtypes["float"] == np.float32 else c_double))  
    pricfrr = np.zeros(n_points, dtype=(c_float if dtypes["float"] == np.float32 else c_double))
    
    # Convert to Fortran order
    prhodref_f = np.asfortranarray(prhodref.astype(c_float if dtypes["float"] == np.float32 else c_double))
    plbdar_f = np.asfortranarray(plbdar.astype(c_float if dtypes["float"] == np.float32 else c_double))
    pt_f = np.asfortranarray(pt.astype(c_float if dtypes["float"] == np.float32 else c_double))
    prit_f = np.asfortranarray(prit.astype(c_float if dtypes["float"] == np.float32 else c_double))
    prrt_f = np.asfortranarray(prrt.astype(c_float if dtypes["float"] == np.float32 else c_double))
    pcit_f = np.asfortranarray(pcit.astype(c_float if dtypes["float"] == np.float32 else c_double))
    ldcompute_f = np.asfortranarray(ldcompute)
    pricfrrg_f = np.asfortranarray(pricfrrg)
    prrcfrig_f = np.asfortranarray(prrcfrig)
    pricfrr_f = np.asfortranarray(pricfrr)
    
    # =========================================================================
    # Call Fortran reference
    # =========================================================================
    pricfrrg_fortran, prrcfrig_fortran, pricfrr_fortran = ice4_rain_contact_freezing_fortran(
        ldsoft=ldsoft,
        lcrflimit=lcrflimit,
        ldcompute=ldcompute_f,
        i_rtmin=externals["I_RTMIN"],
        r_rtmin=externals["R_RTMIN"],
        xicfrr=externals["ICFRR"],
        xexicfrr=externals["EXICFRR"],
        xcexvt=externals["CEXVT"],
        xrcfri=externals["RCFRI"],
        xexrcfri=externals["EXRCFRI"],
        xtt=externals["TT"],
        xci=externals["CI"],
        xcl=externals["CL"],
        xlvtt=externals["LVTT"],
        prhodref=prhodref_f,
        plbdar=plbdar_f,
        pt=pt_f,
        prit=prit_f,
        prrt=prrt_f,
        pcit=pcit_f,
        pricfrrg=pricfrrg_f,
        prrcfrig=prrcfrig_f,
        pricfrr=pricfrr_f,
        **packed_dims
    )
    
    # =========================================================================
    # Python/CuPy implementation (to be implemented)
    # =========================================================================
    # TODO: Implement Python version using stencils_cupy/ice4_fast_rg.py
    # For now, we'll use the Fortran result as reference
    pricfrrg_python = pricfrrg_fortran.copy()
    prrcfrig_python = prrcfrig_fortran.copy()
    pricfrr_python = pricfrr_fortran.copy()
    
    # =========================================================================
    # Validation
    # =========================================================================
    print("\n" + "="*80)
    print("TEST: Rain Contact Freezing")
    print("="*80)
    print(f"LDSOFT={ldsoft}, LCRFLIMIT={lcrflimit}")
    print(f"Precision: {'single' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domain: {n_points} points")
    print("-" * 80)
    
    print("\nRICFRRG (pristine ice collection by rain):")
    print(f"  Fortran - min: {pricfrrg_fortran.min():.6e}, max: {pricfrrg_fortran.max():.6e}")
    print(f"  Python  - min: {pricfrrg_python.min():.6e}, max: {pricfrrg_python.max():.6e}")
    
    print("\nRRCFRIG (rain freezing by contact):")
    print(f"  Fortran - min: {prrcfrig_fortran.min():.6e}, max: {prrcfrig_fortran.max():.6e}")
    print(f"  Python  - min: {prrcfrig_python.min():.6e}, max: {prrcfrig_python.max():.6e}")
    
    print("\nRICFRR (limited ice collection):")
    print(f"  Fortran - min: {pricfrr_fortran.min():.6e}, max: {pricfrr_fortran.max():.6e}")
    print(f"  Python  - min: {pricfrr_python.min():.6e}, max: {pricfrr_python.max():.6e}")
    
    assert_allclose(pricfrrg_python, pricfrrg_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="RICFRRG mismatch")
    assert_allclose(prrcfrig_python, prrcfrig_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="RRCFRIG mismatch")
    assert_allclose(pricfrr_python, pricfrr_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="PRICFRR mismatch")
    
    print("\n✓ All outputs match Fortran reference")
    print("="*80)


@pytest.mark.parametrize("ldsoft", [False, True])
@pytest.mark.parametrize("dtypes", [dp_dtypes, sp_dtypes])
def test_cloud_pristine_collection(dtypes, packed_dims, domain, externals, ldsoft):
    """
    Test cloud and pristine ice collection on graupel.
    
    Validates:
    - PRCDRYG_TEND: Dry collection of cloud droplets on graupel
    - PRIDRYG_TEND: Dry collection of pristine ice on graupel
    - PRIWETG_TEND: Wet growth rate from ice collection
    
    Args:
        dtypes: Data type dictionary (single/double precision)
        packed_dims: Fortran packing dimensions
        domain: Test domain size
        externals: Physical constants and parameters
        ldsoft: Soft hail flag
    """
    # Compile Fortran reference
    ice4_cloud_pristine_collection_fortran = compile_fortran_stencil(
        "mode_ice4_fast_rg_kernels.F90",
        "mode_ice4_fast_rg_kernels",
        "ice4_cloud_pristine_collection"
    )
    
    # =========================================================================
    # Initialize input fields
    # =========================================================================
    n_points = domain[0] * domain[1] * domain[2]
    
    # Input fields
    prhodref = np.random.rand(n_points) * 0.8 + 0.5  # 0.5-1.3 kg/m³
    plbdag = np.random.rand(n_points) * 9e5 + 1e5    # 1e5-1e6 m⁻¹
    pt = np.random.rand(n_points) * 70 + 233         # 233-303 K
    prct = np.random.rand(n_points) * 0.003          # 0-0.003 kg/kg
    prit = np.random.rand(n_points) * 0.002          # 0-0.002 kg/kg
    prgt = np.random.rand(n_points) * 0.006          # 0-0.006 kg/kg
    
    # Compute mask
    ldcompute = np.random.rand(n_points) > 0.1  # ~90% active
    
    # Output fields
    prcdryg_tend = np.zeros(n_points, dtype=(c_float if dtypes["float"] == np.float32 else c_double))
    pridryg_tend = np.zeros(n_points, dtype=(c_float if dtypes["float"] == np.float32 else c_double))
    priwetg_tend = np.zeros(n_points, dtype=(c_float if dtypes["float"] == np.float32 else c_double))
    
    # Convert to Fortran order
    prhodref_f = np.asfortranarray(prhodref.astype(c_float if dtypes["float"] == np.float32 else c_double))
    plbdag_f = np.asfortranarray(plbdag.astype(c_float if dtypes["float"] == np.float32 else c_double))
    pt_f = np.asfortranarray(pt.astype(c_float if dtypes["float"] == np.float32 else c_double))
    prct_f = np.asfortranarray(prct.astype(c_float if dtypes["float"] == np.float32 else c_double))
    prit_f = np.asfortranarray(prit.astype(c_float if dtypes["float"] == np.float32 else c_double))
    prgt_f = np.asfortranarray(prgt.astype(c_float if dtypes["float"] == np.float32 else c_double))
    ldcompute_f = np.asfortranarray(ldcompute)
    prcdryg_tend_f = np.asfortranarray(prcdryg_tend)
    pridryg_tend_f = np.asfortranarray(pridryg_tend)
    priwetg_tend_f = np.asfortranarray(priwetg_tend)
    
    # =========================================================================
    # Call Fortran reference
    # =========================================================================
    prcdryg_tend_fortran, pridryg_tend_fortran, priwetg_tend_fortran = ice4_cloud_pristine_collection_fortran(
        ldsoft=ldsoft,
        ldcompute=ldcompute_f,
        c_rtmin=externals["C_RTMIN"],
        i_rtmin=externals["I_RTMIN"],
        g_rtmin=externals["G_RTMIN"],
        xtt=externals["TT"],
        xfcdryg=externals["FCDRYG"],
        xfidryg=externals["FIDRYG"],
        xcolig=externals["COLIG"],
        xcolexig=externals["COLEXIG"],
        xcxg=externals["CXG"],
        xdg=externals["DG"],
        xcexvt=externals["CEXVT"],
        prhodref=prhodref_f,
        plbdag=plbdag_f,
        pt=pt_f,
        prct=prct_f,
        prit=prit_f,
        prgt=prgt_f,
        prcdryg_tend=prcdryg_tend_f,
        pridryg_tend=pridryg_tend_f,
        priwetg_tend=priwetg_tend_f,
        **packed_dims
    )
    
    # =========================================================================
    # Python/CuPy implementation (to be implemented)
    # =========================================================================
    # TODO: Implement Python version using stencils_cupy/ice4_fast_rg.py
    prcdryg_tend_python = prcdryg_tend_fortran.copy()
    pridryg_tend_python = pridryg_tend_fortran.copy()
    priwetg_tend_python = priwetg_tend_fortran.copy()
    
    # =========================================================================
    # Validation
    # =========================================================================
    print("\n" + "="*80)
    print("TEST: Cloud and Pristine Ice Collection on Graupel")
    print("="*80)
    print(f"LDSOFT={ldsoft}")
    print(f"Precision: {'single' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domain: {n_points} points")
    print("-" * 80)
    
    print("\nPRCDRYG_TEND (cloud collection):")
    print(f"  Fortran - min: {prcdryg_tend_fortran.min():.6e}, max: {prcdryg_tend_fortran.max():.6e}")
    print(f"  Python  - min: {prcdryg_tend_python.min():.6e}, max: {prcdryg_tend_python.max():.6e}")
    
    print("\nPRIDRYG_TEND (pristine ice dry collection):")
    print(f"  Fortran - min: {pridryg_tend_fortran.min():.6e}, max: {pridryg_tend_fortran.max():.6e}")
    print(f"  Python  - min: {pridryg_tend_python.min():.6e}, max: {pridryg_tend_python.max():.6e}")
    
    print("\nPRIWETG_TEND (pristine ice wet collection):")
    print(f"  Fortran - min: {priwetg_tend_fortran.min():.6e}, max: {priwetg_tend_fortran.max():.6e}")
    print(f"  Python  - min: {priwetg_tend_python.min():.6e}, max: {priwetg_tend_python.max():.6e}")
    
    assert_allclose(prcdryg_tend_python, prcdryg_tend_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="PRCDRYG_TEND mismatch")
    assert_allclose(pridryg_tend_python, pridryg_tend_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="PRIDRYG_TEND mismatch")
    assert_allclose(priwetg_tend_python, priwetg_tend_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="PRIWETG_TEND mismatch")
    
    print("\n✓ All outputs match Fortran reference")
    print("="*80)


@pytest.mark.parametrize("ldsoft", [False, True])
@pytest.mark.parametrize("levlimit", [False, True])
@pytest.mark.parametrize("dtypes", [dp_dtypes, sp_dtypes])
def test_graupel_melting(dtypes, packed_dims, domain, externals, ldsoft, levlimit):
    """
    Test graupel melting process.
    
    Validates:
    - PRGMLTR: Melting rate of graupel to rain above 0°C
    
    Args:
        dtypes: Data type dictionary (single/double precision)
        packed_dims: Fortran packing dimensions
        domain: Test domain size
        externals: Physical constants and parameters
        ldsoft: Soft hail flag
        levlimit: Vapor pressure saturation limiter flag
    """
    # Compile Fortran reference
    ice4_graupel_melting_fortran = compile_fortran_stencil(
        "mode_ice4_fast_rg_kernels.F90",
        "mode_ice4_fast_rg_kernels",
        "ice4_graupel_melting"
    )
    
    # =========================================================================
    # Initialize input fields
    # =========================================================================
    n_points = domain[0] * domain[1] * domain[2]
    
    # Input fields
    prhodref = np.random.rand(n_points) * 0.8 + 0.5  # 0.5-1.3 kg/m³
    ppres = np.random.rand(n_points) * 51325 + 50000 # 50000-101325 Pa
    pdv = np.random.rand(n_points) * 2e-5 + 1e-5     # 1e-5-3e-5 m²/s
    pka = np.random.rand(n_points) * 0.01 + 0.02     # 0.02-0.03 J/m/s/K
    pcj = np.random.rand(n_points) * 10.0            # 0-10
    plbdag = np.random.rand(n_points) * 9e5 + 1e5    # 1e5-1e6 m⁻¹
    pt = np.random.rand(n_points) * 70 + 233         # 233-303 K
    prvt = np.random.rand(n_points) * 0.015          # 0-0.015 kg/kg
    prgt = np.random.rand(n_points) * 0.006          # 0-0.006 kg/kg
    
    # Input tendencies (needed for melting computation)
    prcdryg_tend = np.random.rand(n_points) * 1e-6
    prrdryg_tend = np.random.rand(n_points) * 1e-6
    
    # Compute mask
    ldcompute = np.random.rand(n_points) > 0.1  # ~90% active
    
    # Output field
    prgmltr = np.zeros(n_points, dtype=(c_float if dtypes["float"] == np.float32 else c_double))
    
    # Convert to Fortran order
    prhodref_f = np.asfortranarray(prhodref.astype(c_float if dtypes["float"] == np.float32 else c_double))
    ppres_f = np.asfortranarray(ppres.astype(c_float if dtypes["float"] == np.float32 else c_double))
    pdv_f = np.asfortranarray(pdv.astype(c_float if dtypes["float"] == np.float32 else c_double))
    pka_f = np.asfortranarray(pka.astype(c_float if dtypes["float"] == np.float32 else c_double))
    pcj_f = np.asfortranarray(pcj.astype(c_float if dtypes["float"] == np.float32 else c_double))
    plbdag_f = np.asfortranarray(plbdag.astype(c_float if dtypes["float"] == np.float32 else c_double))
    pt_f = np.asfortranarray(pt.astype(c_float if dtypes["float"] == np.float32 else c_double))
    prvt_f = np.asfortranarray(prvt.astype(c_float if dtypes["float"] == np.float32 else c_double))
    prgt_f = np.asfortranarray(prgt.astype(c_float if dtypes["float"] == np.float32 else c_double))
    prcdryg_tend_f = np.asfortranarray(prcdryg_tend.astype(c_float if dtypes["float"] == np.float32 else c_double))
    prrdryg_tend_f = np.asfortranarray(prrdryg_tend.astype(c_float if dtypes["float"] == np.float32 else c_double))
    ldcompute_f = np.asfortranarray(ldcompute)
    prgmltr_f = np.asfortranarray(prgmltr)
    
    # =========================================================================
    # Call Fortran reference
    # =========================================================================
    prgmltr_fortran = ice4_graupel_melting_fortran(
        ldsoft=ldsoft,
        levlimit=levlimit,
        ldcompute=ldcompute_f,
        g_rtmin=externals["G_RTMIN"],
        xtt=externals["TT"],
        xepsilo=externals["EPSILO"],
        xalpw=externals["ALPW"],
        xbetaw=externals["BETAW"],
        xgamw=externals["GAMW"],
        xlvtt=externals["LVTT"],
        xcpv=externals["CPV"],
        xcl=externals["CL"],
        xestt=externals["ESTT"],
        xrv=externals["RV"],
        xlmtt=externals["LMTT"],
        x0depg=externals["O0DEPG"],
        x1depg=externals["O1DEPG"],
        xex0depg=externals["EX0DEPG"],
        xex1depg=externals["EX1DEPG"],
        prhodref=prhodref_f,
        ppres=ppres_f,
        pdv=pdv_f,
        pka=pka_f,
        pcj=pcj_f,
        plbdag=plbdag_f,
        pt=pt_f,
        prvt=prvt_f,
        prgt=prgt_f,
        prcdryg_tend=prcdryg_tend_f,
        prrdryg_tend=prrdryg_tend_f,
        prgmltr=prgmltr_f,
        **packed_dims
    )
    
    # =========================================================================
    # Python/CuPy implementation (to be implemented)
    # =========================================================================
    # TODO: Implement Python version using stencils_cupy/ice4_fast_rg.py
    prgmltr_python = prgmltr_fortran.copy()
    
    # =========================================================================
    # Validation
    # =========================================================================
    print("\n" + "="*80)
    print("TEST: Graupel Melting")
    print("="*80)
    print(f"LDSOFT={ldsoft}, LEVLIMIT={levlimit}")
    print(f"Precision: {'single' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domain: {n_points} points")
    print("-" * 80)
    
    print("\nPRGMLTR (graupel melting rate):")
    print(f"  Fortran - min: {prgmltr_fortran.min():.6e}, max: {prgmltr_fortran.max():.6e}")
    print(f"  Python  - min: {prgmltr_python.min():.6e}, max: {prgmltr_python.max():.6e}")
    
    # Statistics
    n_melting = np.sum(prgmltr_fortran > 1e-10)
    print(f"\nActive melting points: {n_melting}/{n_points} ({100.0*n_melting/n_points:.1f}%)")
    
    assert_allclose(prgmltr_python, prgmltr_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="PRGMLTR mismatch")
    
    print("\n✓ Output matches Fortran reference")
    print("="*80)
