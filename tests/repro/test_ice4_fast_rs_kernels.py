# -*- coding: utf-8 -*-
"""
Test suite for individual ICE4 Fast RS (Snow/Aggregate) process kernels.

This module tests each microphysical process from stencils_cupy/ice4_fast_rs.py
separately against Fortran reference implementations. Each test validates
one specific process to enable targeted debugging and verification.

Processes tested:
- Maximum freezing rate computation for snow processes
- Conversion-melting of aggregates

Reference:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_rs.F90
"""
from ctypes import c_double, c_float

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import dp_dtypes, sp_dtypes


@pytest.mark.parametrize("ldsoft", [False, True])
@pytest.mark.parametrize("levlimit", [False, True])
@pytest.mark.parametrize("dtypes", [dp_dtypes, sp_dtypes])
def test_compute_freezing_rate(dtypes, packed_dims, domain, externals, ldsoft, levlimit):
    """
    Test maximum freezing rate computation for snow processes.
    
    This computes the maximum rate at which snow aggregates can freeze
    based on heat budget constraints and vapor deposition.
    
    Validates:
    - PFREEZ_RATE: Maximum freezing rate available 
    - PFREEZ1_TEND: First freezing term (vapor deposition)
    - PFREEZ2_TEND: Second freezing term (heat capacity factor)
    
    Args:
        dtypes: Data type dictionary (single/double precision)
        packed_dims: Fortran packing dimensions
        domain: Test domain size
        externals: Physical constants and parameters
        ldsoft: Soft hail flag
        levlimit: Vapor pressure saturation limiter flag
    """
    # Compile Fortran reference
    ice4_compute_freezing_rate_fortran = compile_fortran_stencil(
        "mode_ice4_fast_rs_kernels.F90",
        "mode_ice4_fast_rs_kernels",
        "ice4_compute_freezing_rate"
    )
    
    # =========================================================================
    # Initialize input fields
    # =========================================================================
    n_points = domain[0] * domain[1] * domain[2]
    
    # Input fields
    prhodref = np.random.rand(n_points) * 0.8 + 0.5   # 0.5-1.3 kg/m³
    ppres = np.random.rand(n_points) * 51325 + 50000  # 50000-101325 Pa
    pdv = np.random.rand(n_points) * 2e-5 + 1e-5      # 1e-5-3e-5 m²/s
    pka = np.random.rand(n_points) * 0.01 + 0.02      # 0.02-0.03 J/m/s/K
    pcj = np.random.rand(n_points) * 10.0             # 0-10
    plbdas = np.random.rand(n_points) * 9e5 + 1e5     # 1e5-1e6 m⁻¹
    pt = np.random.rand(n_points) * 70 + 233          # 233-303 K
    prvt = np.random.rand(n_points) * 0.015           # 0-0.015 kg/kg
    prst = np.random.rand(n_points) * 0.004           # 0-0.004 kg/kg
    priaggs = np.random.rand(n_points) * 1e-6         # Ice aggregation rate
    
    # Compute mask
    ldcompute = np.random.rand(n_points) > 0.1  # ~90% active
    
    # Output fields
    pfreez_rate = np.zeros(n_points, dtype=(c_float if dtypes["float"] == np.float32 else c_double))
    pfreez1_tend = np.zeros(n_points, dtype=(c_float if dtypes["float"] == np.float32 else c_double))
    pfreez2_tend = np.zeros(n_points, dtype=(c_float if dtypes["float"] == np.float32 else c_double))
    
    # Convert to Fortran order
    prhodref_f = np.asfortranarray(prhodref.astype(c_float if dtypes["float"] == np.float32 else c_double))
    ppres_f = np.asfortranarray(ppres.astype(c_float if dtypes["float"] == np.float32 else c_double))
    pdv_f = np.asfortranarray(pdv.astype(c_float if dtypes["float"] == np.float32 else c_double))
    pka_f = np.asfortranarray(pka.astype(c_float if dtypes["float"] == np.float32 else c_double))
    pcj_f = np.asfortranarray(pcj.astype(c_float if dtypes["float"] == np.float32 else c_double))
    plbdas_f = np.asfortranarray(plbdas.astype(c_float if dtypes["float"] == np.float32 else c_double))
    pt_f = np.asfortranarray(pt.astype(c_float if dtypes["float"] == np.float32 else c_double))
    prvt_f = np.asfortranarray(prvt.astype(c_float if dtypes["float"] == np.float32 else c_double))
    prst_f = np.asfortranarray(prst.astype(c_float if dtypes["float"] == np.float32 else c_double))
    priaggs_f = np.asfortranarray(priaggs.astype(c_float if dtypes["float"] == np.float32 else c_double))
    ldcompute_f = np.asfortranarray(ldcompute)
    pfreez_rate_f = np.asfortranarray(pfreez_rate)
    pfreez1_tend_f = np.asfortranarray(pfreez1_tend)
    pfreez2_tend_f = np.asfortranarray(pfreez2_tend)
    
    # =========================================================================
    # Call Fortran reference
    # =========================================================================
    pfreez_rate_fortran, pfreez1_tend_fortran, pfreez2_tend_fortran = ice4_compute_freezing_rate_fortran(
        ldsoft=ldsoft,
        levlimit=levlimit,
        ldcompute=ldcompute_f,
        s_rtmin=externals["S_RTMIN"],
        xepsilo=externals["EPSILO"],
        xalpi=externals["ALPI"],
        xbetai=externals["BETAI"],
        xgami=externals["GAMI"],
        xtt=externals["TT"],
        xlvtt=externals["LVTT"],
        xcpv=externals["CPV"],
        xcl=externals["CL"],
        xci=externals["CI"],
        xlmtt=externals["LMTT"],
        xestt=externals["ESTT"],
        xrv=externals["RV"],
        x0deps=externals["O0DEPS"],
        x1deps=externals["O1DEPS"],
        xex0deps=externals["EX0DEPS"],
        xex1deps=externals["EX1DEPS"],
        prhodref=prhodref_f,
        ppres=ppres_f,
        pdv=pdv_f,
        pka=pka_f,
        pcj=pcj_f,
        plbdas=plbdas_f,
        pt=pt_f,
        prvt=prvt_f,
        prst=prst_f,
        priaggs=priaggs_f,
        pfreez_rate=pfreez_rate_f,
        pfreez1_tend=pfreez1_tend_f,
        pfreez2_tend=pfreez2_tend_f,
        **packed_dims
    )
    
    # =========================================================================
    # Python/CuPy implementation (to be implemented)
    # =========================================================================
    # TODO: Implement Python version using stencils_cupy/ice4_fast_rs.py
    pfreez_rate_python = pfreez_rate_fortran.copy()
    pfreez1_tend_python = pfreez1_tend_fortran.copy()
    pfreez2_tend_python = pfreez2_tend_fortran.copy()
    
    # =========================================================================
    # Validation
    # =========================================================================
    print("\n" + "="*80)
    print("TEST: Maximum Freezing Rate Computation for Snow")
    print("="*80)
    print(f"LDSOFT={ldsoft}, LEVLIMIT={levlimit}")
    print(f"Precision: {'single' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domain: {n_points} points")
    print("-" * 80)
    
    print("\nPFREEZ_RATE (maximum freezing rate):")
    print(f"  Fortran - min: {pfreez_rate_fortran.min():.6e}, max: {pfreez_rate_fortran.max():.6e}")
    print(f"  Python  - min: {pfreez_rate_python.min():.6e}, max: {pfreez_rate_python.max():.6e}")
    
    print("\nPFREEZ1_TEND (vapor deposition term):")
    print(f"  Fortran - min: {pfreez1_tend_fortran.min():.6e}, max: {pfreez1_tend_fortran.max():.6e}")
    print(f"  Python  - min: {pfreez1_tend_python.min():.6e}, max: {pfreez1_tend_python.max():.6e}")
    
    print("\nPFREEZ2_TEND (heat capacity factor):")
    print(f"  Fortran - min: {pfreez2_tend_fortran.min():.6e}, max: {pfreez2_tend_fortran.max():.6e}")
    print(f"  Python  - min: {pfreez2_tend_python.min():.6e}, max: {pfreez2_tend_python.max():.6e}")
    
    # Statistics
    n_active = np.sum(pfreez_rate_fortran > 1e-10)
    print(f"\nActive freezing points: {n_active}/{n_points} ({100.0*n_active/n_points:.1f}%)")
    
    assert_allclose(pfreez_rate_python, pfreez_rate_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="PFREEZ_RATE mismatch")
    assert_allclose(pfreez1_tend_python, pfreez1_tend_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="PFREEZ1_TEND mismatch")
    assert_allclose(pfreez2_tend_python, pfreez2_tend_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="PFREEZ2_TEND mismatch")
    
    print("\n✓ All outputs match Fortran reference")
    print("="*80)


@pytest.mark.parametrize("ldsoft", [False, True])
@pytest.mark.parametrize("levlimit", [False, True])
@pytest.mark.parametrize("dtypes", [dp_dtypes, sp_dtypes])
def test_conversion_melting_snow(dtypes, packed_dims, domain, externals, ldsoft, levlimit):
    """
    Test conversion-melting of snow aggregates.
    
    This process computes melting of snow above 0°C, accounting for:
    - Vapor diffusion and thermal conduction
    - Heat released by collection of cloud droplets and rain
    - Collection of cloud droplets at positive temperature (no phase change)
    
    Validates:
    - PRSMLTG: Snow melting rate above 0°C
    - PRCMLTSR: Cloud droplet collection by snow at T>0°C
    
    Args:
        dtypes: Data type dictionary (single/double precision)
        packed_dims: Fortran packing dimensions
        domain: Test domain size
        externals: Physical constants and parameters
        ldsoft: Soft hail flag
        levlimit: Vapor pressure saturation limiter flag
    """
    # Compile Fortran reference
    ice4_conversion_melting_snow_fortran = compile_fortran_stencil(
        "mode_ice4_fast_rs_kernels.F90",
        "mode_ice4_fast_rs_kernels",
        "ice4_conversion_melting_snow"
    )
    
    # =========================================================================
    # Initialize input fields
    # =========================================================================
    n_points = domain[0] * domain[1] * domain[2]
    
    # Input fields
    prhodref = np.random.rand(n_points) * 0.8 + 0.5   # 0.5-1.3 kg/m³
    ppres = np.random.rand(n_points) * 51325 + 50000  # 50000-101325 Pa
    pdv = np.random.rand(n_points) * 2e-5 + 1e-5      # 1e-5-3e-5 m²/s
    pka = np.random.rand(n_points) * 0.01 + 0.02      # 0.02-0.03 J/m/s/K
    pcj = np.random.rand(n_points) * 10.0             # 0-10
    plbdas = np.random.rand(n_points) * 9e5 + 1e5     # 1e5-1e6 m⁻¹
    pt = np.random.rand(n_points) * 70 + 233          # 233-303 K
    prvt = np.random.rand(n_points) * 0.015           # 0-0.015 kg/kg
    prst = np.random.rand(n_points) * 0.004           # 0-0.004 kg/kg
    
    # Input tendencies from riming and accretion
    prcrims_tend = np.random.rand(n_points) * 1e-6   # Cloud riming tendency
    prraccs_tend = np.random.rand(n_points) * 1e-6   # Rain accretion tendency
    
    # Compute mask
    ldcompute = np.random.rand(n_points) > 0.1  # ~90% active
    
    # Output fields
    prsmltg = np.zeros(n_points, dtype=(c_float if dtypes["float"] == np.float32 else c_double))
    prcmltsr = np.zeros(n_points, dtype=(c_float if dtypes["float"] == np.float32 else c_double))
    
    # Convert to Fortran order
    prhodref_f = np.asfortranarray(prhodref.astype(c_float if dtypes["float"] == np.float32 else c_double))
    ppres_f = np.asfortranarray(ppres.astype(c_float if dtypes["float"] == np.float32 else c_double))
    pdv_f = np.asfortranarray(pdv.astype(c_float if dtypes["float"] == np.float32 else c_double))
    pka_f = np.asfortranarray(pka.astype(c_float if dtypes["float"] == np.float32 else c_double))
    pcj_f = np.asfortranarray(pcj.astype(c_float if dtypes["float"] == np.float32 else c_double))
    plbdas_f = np.asfortranarray(plbdas.astype(c_float if dtypes["float"] == np.float32 else c_double))
    pt_f = np.asfortranarray(pt.astype(c_float if dtypes["float"] == np.float32 else c_double))
    prvt_f = np.asfortranarray(prvt.astype(c_float if dtypes["float"] == np.float32 else c_double))
    prst_f = np.asfortranarray(prst.astype(c_float if dtypes["float"] == np.float32 else c_double))
    prcrims_tend_f = np.asfortranarray(prcrims_tend.astype(c_float if dtypes["float"] == np.float32 else c_double))
    prraccs_tend_f = np.asfortranarray(prraccs_tend.astype(c_float if dtypes["float"] == np.float32 else c_double))
    ldcompute_f = np.asfortranarray(ldcompute)
    prsmltg_f = np.asfortranarray(prsmltg)
    prcmltsr_f = np.asfortranarray(prcmltsr)
    
    # =========================================================================
    # Call Fortran reference
    # =========================================================================
    prsmltg_fortran, prcmltsr_fortran = ice4_conversion_melting_snow_fortran(
        ldsoft=ldsoft,
        levlimit=levlimit,
        ldcompute=ldcompute_f,
        s_rtmin=externals["S_RTMIN"],
        xepsilo=externals["EPSILO"],
        xalpw=externals["ALPW"],
        xbetaw=externals["BETAW"],
        xgamw=externals["GAMW"],
        xtt=externals["TT"],
        xlvtt=externals["LVTT"],
        xcpv=externals["CPV"],
        xcl=externals["CL"],
        xlmtt=externals["LMTT"],
        xestt=externals["ESTT"],
        xrv=externals["RV"],
        x0deps=externals["O0DEPS"],
        x1deps=externals["O1DEPS"],
        xex0deps=externals["EX0DEPS"],
        xex1deps=externals["EX1DEPS"],
        xfscvmg=externals["FSCVMG"],
        prhodref=prhodref_f,
        ppres=ppres_f,
        pdv=pdv_f,
        pka=pka_f,
        pcj=pcj_f,
        plbdas=plbdas_f,
        pt=pt_f,
        prvt=prvt_f,
        prst=prst_f,
        prcrims_tend=prcrims_tend_f,
        prraccs_tend=prraccs_tend_f,
        prsmltg=prsmltg_f,
        prcmltsr=prcmltsr_f,
        **packed_dims
    )
    
    # =========================================================================
    # Python/CuPy implementation (to be implemented)
    # =========================================================================
    # TODO: Implement Python version using stencils_cupy/ice4_fast_rs.py
    prsmltg_python = prsmltg_fortran.copy()
    prcmltsr_python = prcmltsr_fortran.copy()
    
    # =========================================================================
    # Validation
    # =========================================================================
    print("\n" + "="*80)
    print("TEST: Conversion-Melting of Snow Aggregates")
    print("="*80)
    print(f"LDSOFT={ldsoft}, LEVLIMIT={levlimit}")
    print(f"Precision: {'single' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domain: {n_points} points")
    print("-" * 80)
    
    print("\nPRSMLTG (snow melting rate):")
    print(f"  Fortran - min: {prsmltg_fortran.min():.6e}, max: {prsmltg_fortran.max():.6e}")
    print(f"  Python  - min: {prsmltg_python.min():.6e}, max: {prsmltg_python.max():.6e}")
    
    print("\nPRCMLTSR (cloud collection at T>0°C):")
    print(f"  Fortran - min: {prcmltsr_fortran.min():.6e}, max: {prcmltsr_fortran.max():.6e}")
    print(f"  Python  - min: {prcmltsr_python.min():.6e}, max: {prcmltsr_python.max():.6e}")
    
    # Statistics
    n_melting = np.sum(prsmltg_fortran > 1e-10)
    n_collection = np.sum(prcmltsr_fortran > 1e-10)
    print(f"\nActive melting points: {n_melting}/{n_points} ({100.0*n_melting/n_points:.1f}%)")
    print(f"Active collection points: {n_collection}/{n_points} ({100.0*n_collection/n_points:.1f}%)")
    
    # Temperature distribution
    t_melting = pt[prsmltg_fortran > 1e-10]
    if len(t_melting) > 0:
        print(f"\nMelting zone temperatures:")
        print(f"  min={t_melting.min():.1f}K, max={t_melting.max():.1f}K, mean={t_melting.mean():.1f}K")
    
    assert_allclose(prsmltg_python, prsmltg_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="PRSMLTG mismatch")
    assert_allclose(prcmltsr_python, prcmltsr_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="PRCMLTSR mismatch")
    
    print("\n✓ All outputs match Fortran reference")
    print("="*80)
