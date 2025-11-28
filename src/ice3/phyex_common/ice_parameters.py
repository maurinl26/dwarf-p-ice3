# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np

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


# from_file="PHYEX/src/common/aux/modd_param_icen.F90"
@dataclass
class IceParameters:
    """
    Configuration parameters for ICE3/ICE4 microphysics schemes.
    
    This dataclass contains all tunable parameters, switches, and options
    controlling the behavior of the ICE3 and ICE4 bulk microphysics schemes.
    Default values are taken from PHYEX/modd_param_icen.F90 and can be
    customized for different atmospheric models (AROME, Meso-NH, LMDZ).
    
    Attributes
    ----------
    
    **Model Configuration**
    
    HPROGRAM : {"AROME", "MESO-NH", "LMDZ"}
        Target atmospheric model. Sets appropriate default parameters in __post_init__.
    
    **Warm Cloud Processes**
    
    LWARM : bool
        Enable warm rain processes (autoconversion, accretion). Default: True.
    LSEDIC : bool
        Enable cloud droplet sedimentation. Default: True.
    LDEPOSC : bool
        Enable fog deposition on vegetation (surface process). Default: False.
    VDEPOSC : float
        Fog deposition velocity (m/s). Default: 0.02.
    
    **Ice Crystal Properties**
    
    PRISTINE_ICE : {"PLAT", "COLU", "BURO"}
        Pristine ice crystal habit:
        - "PLAT": Plate crystals
        - "COLU": Column crystals  
        - "BURO": Bullet rosette crystals
        Default: "PLAT".
    
    **Numerical Schemes**
    
    SEDIM : int
        Sedimentation scheme:
        - Sedim.SPLI (0): Time-split sedimentation
        - Sedim.STAT (1): Statistical sedimentation
        Default: SPLI.
    
    LRED : bool
        Use modified ICE3/ICE4 to reduce time step dependency. Default: True.
    
    NMAXITER_MICRO : int
        Maximum iterations for time/mixing ratio splitting. Default: 5.
    
    MRSTEP : float
        Maximum mixing ratio step for splitting (kg/kg). Default: 5×10⁻⁵.
        Set to 0 to disable mixing ratio splitting.
    
    SPLIT_MAXCFL : float
        Maximum CFL number for upwind sedimentation scheme. Default: 0.8.
    
    **Thermodynamic Options**
    
    LFEEDBACKT : bool
        Include temperature feedback during iterations. Default: True.
    
    LEVLIMIT : bool
        Limit water vapor by saturation (prevent supersaturation). Default: True.
    
    LCOND2 : bool
        Separate liquid and ice condensation processes. Default: False.
    
    **Graupel Growth Modes**
    
    LNULLWETG : bool
        Allow graupel wet growth with null rate (enables water shedding). Default: True.
    
    LWETGPOST : bool
        Allow graupel wet growth at T > 0°C (for water shedding). Default: True.
    
    LCONVHG : bool
        Allow conversion from hail to graupel (7-category scheme only). Default: False.
        Set to True for AROME and LMDZ.
    
    **Snow Riming Parameterization**
    
    SNOW_RIMING : int
        Snow riming formulation:
        - SnowRiming.M90 (0): Murakami (1990) formulation
        - SnowRiming.OLD (1): Original formulation
        Default: M90.
    
    FRAC_M90 : float
        Fraction parameter for M90 riming formulation. Default: 0.1.
    
    LSNOW_T : bool
        Use temperature-dependent snow parameterization (Wurtz, 2021). Default: False.
    
    **Contact Freezing**
    
    LCRFLIMIT : bool
        Limit rain contact freezing rate by available latent heat. Default: True.
    
    **Subgrid-Scale Parameterizations**
    
    SUBG_RC_RR_ACCR : int
        Subgrid cloud-rain accretion method:
        - SubgRRRCAccr.NONE (0): No subgrid treatment
        - SubgRRRCAccr.PRFR (1): Precipitation fraction method
        Default: NONE.
    
    SUBG_RR_EVAP : int
        Subgrid rain evaporation method:
        - SubgRREvap.NONE (0): Grid-box mean
        - SubgRREvap.CLFR (1): Cloud fraction
        - SubgRREvap.PRFR (2): Precipitation fraction
        Default: NONE.
    
    SUBG_PR_PDF : int
        PDF for subgrid precipitation:
        - SubgPRPDF.SIGM (0): Sigma method
        - SubgPRPDF.HLCRECTPDF (1): HLC rectangular PDF
        - SubgPRPDF.HLCISOTRIPDF (2): HLC isotropic triangular PDF
        - SubgPRPDF.HLCTRIANGPDF (3): HLC triangular PDF
        - SubgPRPDF.HLCQUADRAPDF (4): HLC quadratic PDF
        Default: SIGM.
    
    SUBG_AUCV_RC : int
        Subgrid liquid autoconversion method:
        - SubgAucvRc.NONE (0): Grid-box mean
        - SubgAucvRc.PDF (1): PDF-based (default for AROME/LMDZ)
        - SubgAucvRc.ADJU (2): Adjustment-based
        - SubgAucvRc.CLFR (3): Cloud fraction
        - SubgAucvRc.SIGM (4): Sigma method
        Default: NONE.
    
    SUBG_AUCV_RI : int
        Subgrid ice autoconversion method:
        - SubgAucvRi.NONE (0): Grid-box mean
        - SubgAucvRi.CLFR (1): Cloud fraction
        - SubgAucvRi.ADJU (2): Adjustment-based
        Default: NONE.
    
    SUBG_MF_PDF : int
        PDF for mass flux cloud autoconversion:
        - SubGridMassFluxPDF.NONE (0): No PDF
        - SubGridMassFluxPDF.TRIANGLE (1): Triangular PDF (default)
        Default: TRIANGLE.
    
    **Saturation Adjustment**
    
    LADJ_BEFORE : bool
        Perform saturation adjustment before microphysics. Default: True.
    
    LADJ_AFTER : bool
        Perform saturation adjustment after microphysics. Default: True.
        Set to False for AROME (adjustment done separately).
    
    LSEDIM_AFTER : bool
        Perform sedimentation after (True) or before (False) microphysics. Default: False.
    
    **Autoconversion Thresholds**
    
    CRIAUTI_NAM : float
        Minimum ice mixing ratio for ice→snow autoconversion (kg/kg). Default: 2×10⁻⁵.
    
    ACRIAUTI_NAM : float
        'A' parameter in ice→snow power law: rate = 10^(A×(T-T₀)+B). Default: 0.06.
    
    BRCRIAUTI_NAM : float
        'B' parameter in ice→snow power law. Default: -3.5.
    
    T0CRIAUTI_NAM : float
        Threshold temperature for ice→snow autoconversion (°C).
        Computed from other parameters in __post_init__.
    
    CRIAUTC_NAM : float
        Threshold for cloud→rain autoconversion (kg/kg). Default: 5×10⁻⁴.
        Set to 1×10⁻³ for LMDZ.
    
    LCRIAUTI : bool
        Compute ACRIAUTI and BCRIAUTI parameters. Default: True.
    
    **Deposition Tuning**
    
    RDEPSRED_NAM : float
        Tuning factor for vapor deposition on snow (dimensionless). Default: 1.0.
    
    RDEPGRED_NAM : float
        Tuning factor for vapor deposition on graupel (dimensionless). Default: 1.0.
    
    **Fortran Optimization (Not Used in GT4Py)**
    
    LPACK_INTERP : bool
        Pack arrays before interpolation (Fortran GPU optimization). Default: True.
    
    LPACK_MICRO : bool
        Pack arrays before process calculations (Fortran GPU optimization). Default: True.
    
    NPROMICRO : int
        Cache-blocking block size (Fortran optimization). Default: 0.
    
    **Time Stepping**
    
    TSTEP_TS : float
        Approximate time step for time-splitting version (s). Default: 0.
    
    **Tuning Arrays**
    
    FRMIN_NAM : np.ndarray
        Array of 41 tuning parameters for various processes.
        Initialized in set_frmin_nam() method.
    
    Notes
    -----
    Model-Specific Defaults:
    
    **AROME Configuration:**
    - LCONVHG = True
    - LADJ_BEFORE = True, LADJ_AFTER = False
    - LRED = False
    - SEDIM = STAT
    - MRSTEP = 0 (no mixing ratio splitting)
    - SUBG_AUCV_RC = PDF
    
    **LMDZ Configuration:**
    - SUBG_AUCV_RC = PDF
    - SEDIM = STAT
    - NMAXITER_MICRO = 1
    - CRIAUTC_NAM = 1×10⁻³
    - CRIAUTI_NAM = 2×10⁻⁴
    - T0CRIAUTI_NAM = -5°C
    - LRED = True
    - LCONVHG = True
    - LADJ_BEFORE = True, LADJ_AFTER = True
    
    **Meso-NH Configuration:**
    - Uses default values
    
    Source Reference
    ----------------
    PHYEX/src/common/aux/modd_param_icen.F90
    
    References
    ----------
    Murakami, M., 1990: Numerical modeling of dynamical and microphysical
    evolution of an isolated convective cloud. J. Meteor. Soc. Japan, 68, 107-128.
    
    Wurtz, J., 2021: Amélioration et évaluation de la paramétrisation des
    hydrométéores mixtes dans le modèle AROME. PhD thesis, Université Toulouse III.
    
    Examples
    --------
    >>> # AROME configuration
    >>> params_arome = IceParameters(HPROGRAM="AROME")
    >>> print(params_arome.SUBG_AUCV_RC)  # PDF method
    1
    
    >>> # Meso-NH with custom settings
    >>> params_mnh = IceParameters(
    ...     HPROGRAM="MESO-NH",
    ...     LSNOW_T=True,  # Enable Wurtz snow param
    ...     CRIAUTC_NAM=1e-3  # Higher autoconversion threshold
    ... )
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
    FRMIN_NAM: np.ndarray = field(init=False)

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
