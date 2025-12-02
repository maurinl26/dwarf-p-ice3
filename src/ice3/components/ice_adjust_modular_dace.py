"""
ICE_ADJUST Modular Component - DaCe Implementation

This component reproduces the microphysical adjustment scheme ICE_ADJUST from PHYEX
using DaCe stencils for GPU acceleration.

Reference:
    PHYEX-IAL_CY50T1/micro/ice_adjust.F90
    PHYEX-IAL_CY50T1/micro/condensation.F90

Architecture:
    1. thermodynamic_fields : Compute T, Lv, Ls, Cph
    2. condensation         : CB02 scheme, produce rc_out, ri_out
    3. cloud_fraction_1     : Microphysical sources, conservation
    4. cloud_fraction_2     : Final cloud fraction, autoconversion
"""
from __future__ import annotations

import logging
from typing import Tuple, Dict
import numpy as np
from numpy.typing import NDArray

from ..phyex_common.phyex import Phyex
from ..utils.env import DTYPES, BACKEND

log = logging.getLogger(__name__)


class IceAdjustModularDaCe:
    """
    Modular microphysical adjustment component using DaCe.
    
    DaCe implementation of ICE_ADJUST reproducing the PHYEX reference
    using separated stencils from condensation.py and cloud_fraction.py.
    
    Execution sequence:
        1. thermodynamic_fields : T = θ×Π, Lv, Ls, Cph
        2. condensation         : CB02 scheme → rc_tmp, ri_tmp
        3. cloud_fraction_1     : Compute sources, water/energy conservation
        4. cloud_fraction_2     : Cloud fraction, autoconversion
    
    Args:
        phyex: PHYEX configuration (AROME by default)
        dtypes: Data types (float32/float64)
        backend: Backend for DaCe (cpu, gpu)
    
    Example:
        >>> phyex = Phyex("AROME")
        >>> ice_adjust = IceAdjustModularDaCe(phyex)
        >>> ice_adjust(
        ...     sigqsat, exn, exnref, rhodref, pabs, sigs,
        ...     cf_mf, rc_mf, ri_mf, th, rv, rc, rr, ri, rs, rg,
        ...     cldfr, hlc_hrc, hlc_hcf, hli_hri, hli_hcf, sigrc,
        ...     ths, rvs, rcs, ris, timestep, domain
        ... )
    """

    def __init__(
        self,
        phyex: Phyex = Phyex("AROME"),
        dtypes: Dict = DTYPES,
        backend: str = "cpu",
    ) -> None:
        """
        Initialize the modular ICE_ADJUST component with DaCe.
        """
        self.phyex = phyex
        self.dtypes = dtypes
        self.backend = backend

        # Import DaCe stencils
        from ..stencils_dace.cloud_fraction import (
            thermodynamic_fields,
            cloud_fraction_1,
            cloud_fraction_2,
        )
        from ..stencils_dace.condensation import condensation

        # Store stencil references
        self.thermodynamic_fields_stencil = thermodynamic_fields
        self.condensation_stencil = condensation
        self.cloud_fraction_1_stencil = cloud_fraction_1
        self.cloud_fraction_2_stencil = cloud_fraction_2

        # Extract constants from phyex
        self.NRR = phyex.param_icen.NRR
        self.CPD = phyex.constants.CPD
        self.CPV = phyex.constants.CPV
        self.CL = phyex.constants.XCL
        self.CI = phyex.constants.XCI
        
        self.LSUBG_COND = phyex.nebn.LSUBG_COND
        self.SUBG_MF_PDF = phyex.param_icen.SUBG_MF_PDF
        self.CRIAUTC = phyex.param_icen.CRIAUTC
        self.CRIAUTI = phyex.param_icen.CRIAUTI
        self.ACRIAUTI = phyex.param_icen.ACRIAUTI
        self.BCRIAUTI = phyex.param_icen.BCRIAUTI
        self.TT = phyex.constants.XTT
        
        self.CONDENS = phyex.nebn.CONDENS
        self.FRAC_ICE_ADJUST = phyex.nebn.FRAC_ICE_ADJUST
        self.LSIGMAS = phyex.nebn.LSIGMAS
        self.LSTATNW = phyex.nebn.LSTATNW
        self.OCND2 = False
        self.RD = phyex.constants.RD
        self.RV = phyex.constants.RV
        self.TMAXMIX = phyex.param_icen.XTMAXMIX
        self.TMINMIX = phyex.param_icen.XTMINMIX

        log.info("="*70)
        log.info("IceAdjustModularDaCe - PHYEX Configuration")
        log.info("="*70)
        log.info(f"Backend          : {backend}")
        log.info(f"Precision        : {dtypes['float']}")
        log.info(f"SUBG_COND        : {self.LSUBG_COND}")
        log.info(f"SUBG_MF_PDF      : {self.SUBG_MF_PDF}")
        log.info(f"SIGMAS           : {self.LSIGMAS}")
        log.info(f"CONDENS (scheme) : {self.CONDENS}")
        log.info(f"FRAC_ICE_ADJUST  : {self.FRAC_ICE_ADJUST}")
        log.info("="*70)

    def __call__(
        self,
        sigqsat: NDArray,
        exn: NDArray,
        exnref: NDArray,
        rhodref: NDArray,
        pabs: NDArray,
        sigs: NDArray,
        cf_mf: NDArray,
        rc_mf: NDArray,
        ri_mf: NDArray,
        th: NDArray,
        rv: NDArray,
        rc: NDArray,
        rr: NDArray,
        ri: NDArray,
        rs: NDArray,
        rg: NDArray,
        cldfr: NDArray,
        hlc_hrc: NDArray,
        hlc_hcf: NDArray,
        hli_hri: NDArray,
        hli_hcf: NDArray,
        sigrc: NDArray,
        ths: NDArray,
        rvs: NDArray,
        rcs: NDArray,
        ris: NDArray,
        timestep: float,
        domain: Tuple[int, ...],
    ):
        """
        Execute the complete ICE_ADJUST microphysical adjustment sequence.
        
        Execution sequence corresponding to ice_adjust.F90:
        
        1. THERMODYNAMIC_FIELDS (ice_adjust.F90, l.450-473)
           - Compute temperature T = θ × Π
           - Latent heats Lv, Ls (T-dependent)
           - Specific heat Cph (function of hydrometeors)
        
        2. CONDENSATION (condensation.F90, l.186-575)
           - CB02 scheme (Chaboureau & Bechtold 2002)
           - Compute subgrid variability
           - Produce condensates rc_out, ri_out
           - Initial cloud fraction
        
        3. CLOUD_FRACTION_1 (ice_adjust.F90, l.278-312)
           - Compute microphysical sources drc/dt, dri/dt
           - Total water conservation
           - Thermal adjustment of potential temperature
        
        4. CLOUD_FRACTION_2 (ice_adjust.F90, l.313-419)
           - Final cloud fraction with mass flux
           - Liquid autoconversion (cloud → rain)
           - Ice autoconversion (ice → snow)
        """
        ni, nj, nk = domain
        
        # Create temporary fields
        dtype = self.dtypes['float']
        t = np.zeros((ni, nj, nk), dtype=dtype)
        lv = np.zeros((ni, nj, nk), dtype=dtype)
        ls = np.zeros((ni, nj, nk), dtype=dtype)
        cph = np.zeros((ni, nj, nk), dtype=dtype)
        rv_out = np.zeros((ni, nj, nk), dtype=dtype)
        rc_out = np.zeros((ni, nj, nk), dtype=dtype)
        ri_out = np.zeros((ni, nj, nk), dtype=dtype)
        q1 = np.zeros((ni, nj, nk), dtype=dtype)
        pv_out = np.zeros((ni, nj, nk), dtype=dtype)
        piv_out = np.zeros((ni, nj, nk), dtype=dtype)
        frac_out = np.zeros((ni, nj, nk), dtype=dtype)
        qsl_out = np.zeros((ni, nj, nk), dtype=dtype)
        qsi_out = np.zeros((ni, nj, nk), dtype=dtype)
        sigma_out = np.zeros((ni, nj, nk), dtype=dtype)
        cond_out = np.zeros((ni, nj, nk), dtype=dtype)
        a_out = np.zeros((ni, nj, nk), dtype=dtype)
        b_out = np.zeros((ni, nj, nk), dtype=dtype)
        sbar_out = np.zeros((ni, nj, nk), dtype=dtype)

        log.debug("Step 1/4: Computing thermodynamic fields")
        # ================================================================
        # STEP 1: THERMODYNAMIC FIELDS
        # Reference: ice_adjust.F90, lines 450-473
        # ================================================================
        self.thermodynamic_fields_stencil(
            th, exn, rv, rc, rr, ri, rs, rg,
            lv, ls, cph, t,
            self.NRR, self.CPD, self.CPV, self.CL, self.CI
        )

        log.debug("Step 2/4: Condensation (CB02 scheme)")
        # ================================================================
        # STEP 2: CONDENSATION
        # Reference: condensation.F90, lines 186-575
        # CB02 scheme from Chaboureau & Bechtold (2002)
        # ================================================================
        self.condensation_stencil(
            sigqsat, pabs, sigs, t, rv, ri, rc,
            rv_out, rc_out, ri_out, cldfr, cph, lv, ls, q1,
            pv_out, piv_out, frac_out, qsl_out, qsi_out,
            sigma_out, cond_out, a_out, b_out, sbar_out,
            self.CONDENS, self.FRAC_ICE_ADJUST, self.LSIGMAS, 
            self.LSTATNW, self.OCND2, self.RD, self.RV, 
            self.TMAXMIX, self.TMINMIX
        )

        log.debug("Step 3/4: Computing microphysical sources")
        # ================================================================
        # STEP 3: MICROPHYSICAL SOURCES
        # Reference: ice_adjust.F90, lines 278-312
        # Water conservation and thermal adjustment
        # ================================================================
        self.cloud_fraction_1_stencil(
            lv, ls, cph, exnref, rc, ri,
            ths, rvs, rcs, ris,
            rc_out, ri_out,
            np.float32(timestep)
        )

        log.debug("Step 4/4: Cloud fraction and autoconversion")
        # ================================================================
        # STEP 4: CLOUD FRACTION AND AUTOCONVERSION
        # Reference: ice_adjust.F90, lines 313-419
        # Final cloud fraction computation, autoconversion
        # ================================================================
        self.cloud_fraction_2_stencil(
            rhodref, exnref, t, cph, lv, ls,
            ths, rvs, rcs, ris,
            rc_mf, ri_mf, cf_mf,
            cldfr, hlc_hrc, hlc_hcf, hli_hri, hli_hcf,
            np.float32(timestep),
            self.LSUBG_COND, self.SUBG_MF_PDF,
            self.CRIAUTC, self.CRIAUTI, 
            self.ACRIAUTI, self.BCRIAUTI, self.TT
        )

        log.debug("ICE_ADJUST modular DaCe: complete sequence finished")

    def __repr__(self) -> str:
        """Text representation of the component."""
        return (
            f"IceAdjustModularDaCe(backend={self.backend}, "
            f"dtype={self.dtypes['float']}, "
            f"SUBG_COND={self.LSUBG_COND})"
        )
