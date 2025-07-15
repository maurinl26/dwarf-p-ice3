# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import dace

# Stands for CSUBG_MF_PDF in modd_param_icen.F90
# Default to NONE
class SubGridMassFluxPDF(Enum):
    NONE = 0
    TRIANGLE = 1


# Stands for CSUBG_RC_RR_ACCR in modd_param_icen.F90
# Default to NONE
class SubgRRRCAccr(Enum):
    NONE = 0
    PRFR = 1


# Stands for CSUBG_RR_EVAP in modd_param_icen.F90
# Default to NONE
class SubgRREvap(Enum):
    NONE = 0
    CLFR = 1
    PRFR = 2


# Stands for CSUBG_PR_PDF in modd_param_icen.F90
# Default to SIGM
class SubgPRPDF(Enum):
    SIGM = 0
    HLCRECTPDF = 1
    HLCISOTRIPDF = 2
    HLCTRIANGPDF = 3
    HLCQUADRAPDF = 4


# Stands for CSUBG_AUCV_RC in modd_param_icen.F90
# Default to NONE
class SubgAucvRc(Enum):
    NONE = 0
    PDF = 1
    ADJU = 2
    CLFR = 3
    SIGM = 4


# Stands for CSUBG_AUCV_RI in modd_param_icen.F90
# Default to NONE
class SubgAucvRi(Enum):
    NONE = 0
    CLFR = 1
    ADJU = 2


class SnowRiming(Enum):
    M90 = 0
    OLD = 1


class Sedim(Enum):
    SPLI = 0
    STAT = 1


@dataclass
class ParamIce:
    """
    Parameters for ice processes
    Default values are taken from modd_param_icen.F90

    hprogram: Literal["AROME", "MESO-NH", "LMDZ"]

    lwarm: bool             # Formation of rain by warm processes
    lsedic: bool            # Enable the droplets sedimentation
    ldeposc: bool           # Enable cloud droplets deposition

    vdeposc: float          # Droplet deposition velocity

    pristine_ice:           # Pristine ice type PLAT, COLU, or BURO
    sedim: str              # Sedimentation calculation mode

    lred: bool              # To use modified ice3/ice4 - to reduce time step dependency
    lfeedbackt: bool        # Feedback on temperature taken into account when True
    levlimit: bool          # Water vapour limited by saturation when True
    lnullwetg: bool         # Graupel wet growth activated with null rate when True to allow water shedding
    lwetgpost: bool         # Graupel wet growth activated with positive telmperature when True (to allow water shedding)

    snow_riming: Literal["OLD", "M90"]      # OLD or M90 for Murakami 1990 formulation
    frac_m90: float                         # Fraction used in M90 formulation

    nmaxiter_micro: int     # max number of iteration for time or mixing ratio splitting
    mrstep: float           # max mixing ratio step for mixing ratio splitting

    lconvhg: bool           # True to allow conversion from hail to graupel
    lcrflimit: bool         # True to limit rain contact freezing

    tstep_ts: float         # approximative time step when for use of time splitting version

    subg_rc_rr_accr: str    # subgrid rc-rr accretion
    subg_rr_evap: str       # subgrid rr evaporation
    subg_pr_pdf: str        # pdf for subgrid precipitation
    subg_aucv_rc: str       # type of subgrid rc->rr autoconv. method
    subg_aucv_ri: str       # type of subgrid ri->rs autoconv. method
    subg_mf_pdf: str        # PDF to use for MF cloud autoconversions

    ladj_before: bool       # must we perform an adjustment before rain_ice call
    ladj_after: bool        # must we perform an adjustment after rain_ice call
    lsedim_after: bool      # sedimentation done before (.FALSE.) or after (.TRUE.) microphysics

    split_maxcfl: float     # Maximum CFL number allowed for SPLIT scheme
    lsnow_t: bool           # Snow parameterization from Wurtz (2021)

    # Translation note : lpack_interp and lpack_micro are for Fortran code GPU optimization (no use in GT4Py)
    lpack_interp: bool      # switch to pack arrays before interpolation functions
    lpack_micro: bool       # switch to pack arrays beafore computing the process tendencies

    npromicro: int          # Fortran - Size of cache-blocking bloc

    lcriauti: bool          # switch to compute acriauti and bcriauti

    criauti_nam: float      # minimum value for ice to snow autoconversion
    acriauti_nam:           # A parameter for ice to snow autoconversion power law
    brcriauti_nam: float    # B parameter for ice to snow autoconvserion power law
    t0criauti_nam: float    # threshold temperature for ice to snow autoconversion
    criautc_nam: float      # threshold for liquid cloud to rain autoconversion 10**(at+b)
    rdepsred_nam: float     # tuning factor of sublimation on ice
    rdepgred_nam: float     # tuning factor of sublimation on graupel
    lcond2: bool            # logical switch to separate liquid and ice
    frmin_nam: np.ndarray = field(init=False)
    """

    HPROGRAM: Literal["AROME", "MESO-NH", "LMDZ"]

    LWARM: bool = field(default=True)  # Formation of rain by warm processes
    LSEDIC: bool = field(default=True)  # Enable the droplets sedimentation
    LDEPOSC: bool = field(
        default=False
    )  # Enable cloud droplets deposition on vegetation

    VDEPOSC: float = field(default=0.02)  # Droplet deposition velocity

    PRISTINE_ICE: Literal["PLAT", "COLU", "BURO"] = field(
        default="PLAT"
    )  # Pristine ice type PLAT, COLU, or BURO
    SEDIM: int = field(default=Sedim.SPLI.value)  # Sedimentation calculation mode

    # To use modified ice3/ice4 - to reduce time step dependency
    LRED: bool = field(default=True)
    LFEEDBACKT: bool = field(default=True)
    LEVLIMIT: bool = field(default=True)
    LNULLWETG: bool = field(default=True)
    LWETGPOST: bool = field(default=True)

    SNOW_RIMING: int = field(default=SnowRiming.M90.value)

    FRAC_M90: float = field(default=0.1)
    NMAXITER_MICRO: int = field(default=5)
    MRSTEP: float = field(default=5e-5)

    LCONVHG: bool = field(default=False)
    LCRFLIMIT: bool = field(default=True)

    TSTEP_TS: float = field(default=0)

    SUBG_RC_RR_ACCR: int = field(
        default=SubgRRRCAccr.NONE.value
    )  # subgrid rc-rr accretion
    SUBG_RR_EVAP: int = field(default=SubgRREvap.NONE.value)  # subgrid rr evaporation
    SUBG_PR_PDF: int = field(
        default=SubgPRPDF.SIGM.value
    )  # pdf for subgrid precipitation
    SUBG_AUCV_RC: int = field(
        default=SubgAucvRc.NONE.value
    )  # type of subgrid rc->rr autoconv. method
    SUBG_AUCV_RI: int = field(
        default=SubgAucvRi.NONE.value
    )  # type of subgrid ri->rs autoconv. method

    # PDF to use for MF cloud autoconversions
    SUBG_MF_PDF: int = field(default=SubGridMassFluxPDF.TRIANGLE.value)

    # key for adjustment before rain_ice call
    LADJ_BEFORE: bool = field(default=True)

    # key for adjustment after rain_ice call
    LADJ_AFTER: bool = field(default=True)

    # switch to perform sedimentation
    # before (.FALSE.)
    # or after (.TRUE.) microphysics
    LSEDIM_AFTER: bool = field(default=False)

    # Maximum CFL number allowed for SPLIT scheme
    SPLIT_MAXCFL: float = field(default=0.8)

    # Snow parameterization from Wurtz (2021)
    LSNOW_T: bool = field(default=False)

    LPACK_INTERP: bool = field(default=True)
    LPACK_MICRO: bool = field(default=True)
    LCRIAUTI: bool = field(default=True)

    NPROMICRO: int = field(default=0)

    CRIAUTI_NAM: float = field(default=0.2e-4)
    ACRIAUTI_NAM: float = field(default=0.06)
    BRCRIAUTI_NAM: float = field(default=-3.5)
    T0CRIAUTI_NAM: float = field(init=False)
    CRIAUTC_NAM: float = field(default=0.5e-3)
    RDEPSRED_NAM: float = field(default=1)
    RDEPGRED_NAM: float = field(default=1)
    LCOND2: bool = field(default=False)

    # TODO : replace frmin_nam by a global table
    # FRMIN_NAM: dace.float64[41] = field(init=False)

    def __post_init__(self):
        self.T0CRIAUTI_NAM = (np.log10(self.CRIAUTI_NAM) - self.BRCRIAUTI_NAM) / 0.06
        self.set_frmin_nam()

        if self.HPROGRAM == "AROME":
            self.LCONVHG = True
            self.LADJ_BEFORE = True
            self.LADJ_AFTER = False
            self.LRED = False
            self.SEDIM = Sedim.STAT.value
            self.MRSTEP = 0
            self.SUBG_AUCV_RC = SubgAucvRc.PDF.value

        elif self.HPROGRAM == "LMDZ":
            self.SUBG_AUCV_RC = SubgAucvRc.PDF.value
            self.SEDIM = Sedim.STAT.value
            self.NMAXITER_MICRO = 1
            self.CRIAUTC_NAM = 0.001
            self.CRIAUTI_NAM = 0.0002
            self.T0CRIAUTI_NAM = -5
            self.LRED = True
            self.LCONVHG = True
            self.LADJ_BEFORE = True
            self.LADJ_AFTER = True

    def set_frmin_nam(self):
        tmp_frmin_nam = np.empty(41)
        tmp_frmin_nam[1:6] = 0
        tmp_frmin_nam[7:9] = 1.0
        tmp_frmin_nam[10] = 10.0
        tmp_frmin_nam[11] = 1.0
        tmp_frmin_nam[12] = 0.0
        tmp_frmin_nam[13] = 1.0e-15
        tmp_frmin_nam[14] = 120.0
        tmp_frmin_nam[15] = 1.0e-4
        tmp_frmin_nam[16:20] = 0.0
        tmp_frmin_nam[21:22] = 1.0
        tmp_frmin_nam[23] = 0.5
        tmp_frmin_nam[24] = 1.5
        tmp_frmin_nam[25] = 30.0
        tmp_frmin_nam[26:38] = 0.0
        tmp_frmin_nam[39] = 0.25
        tmp_frmin_nam[40] = 0.15

        self.FRMIN_NAM = tmp_frmin_nam
