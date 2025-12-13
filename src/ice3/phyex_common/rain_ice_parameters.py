# -*- coding: utf-8 -*-
from math import gamma, log
from numpy.typing import NDArray
import logging
import numpy as np
import cython

from ice3.phyex_common.constants import Constants
from ice3.phyex_common.incomplete_gamma import generalized_incomplete_gamma
from ice3.phyex_common.ice_parameters import IceParameters
from ice3.phyex_common.rain_ice_descriptors import RainIceDescriptors

logging.getLogger()


# from_file="PHYEX/src/common/aux/modd_rain_ice_paramn.F90"
@cython.cclass
class RainIceParameters:
    """
    Microphysical rate constants for ICE3/ICE4 bulk schemes.
    
    This dataclass contains all computed rate constants, collection efficiencies,
    and kernel coefficients used in the ICE3 and ICE4 microphysics schemes.
    Values are computed in __post_init__ from fundamental constants (Constants),
    hydrometeor descriptors (RainIceDescriptors), and user parameters (IceParameters).
    
    The constants are organized by process type:
    - Sedimentation (SED)
    - Nucleation (HEN, HON)  
    - Vapor deposition/Evaporation (DEP, EVA)
    - Autoconversion (AUT)
    - Aggregation (AGG)
    - Accretion (ACC)
    - Riming (RIM)
    - Contact freezing (CFR)
    - Dry/wet graupel growth (DRY/WET)
    
    Attributes
    ----------
    
    **Dependencies (Required for Initialization)**
    
    cst : Constants
        Physical constants dataclass.
    rid : RainIceDescriptors
        Hydrometeor properties (masses, fall speeds, size distributions).
    parami : IceParameters
        User-configurable microphysics parameters.
    
    **Sedimentation Constants (SED)**
    
    FSEDC_1, FSEDC_2 : float
        Cloud droplet sedimentation factors for bimodal distribution (m/s).
    FSEDR : float
        Rain sedimentation pre-factor (m^EXSEDR/s).
    EXSEDR : float
        Rain sedimentation exponent (dimensionless).
    FSEDI : float
        Ice crystal sedimentation pre-factor.
    EXCSEDI, EXRSEDI : float
        Ice crystal sedimentation exponents for concentration and mixing ratio.
    FSEDS : float
        Snow sedimentation pre-factor (m^EXSEDS/s).
    EXSEDS : float
        Snow sedimentation exponent (dimensionless).
    FSEDG : float
        Graupel sedimentation pre-factor (m^EXSEDG/s).
    EXSEDG : float
        Graupel sedimentation exponent (dimensionless).
    
    **Heterogeneous Ice Nucleation (HENI)**
    
    NU10 : float
        Contact nucleation parameter (m⁻³·s⁻¹), habit-dependent.
        Multiplied by 50 for plates, 1250 for columns, 850 for bullet rosettes.
    ALPHA1, BETA1 : float
        Immersion freezing temperature coefficients (dimensionless).
        Formula: N_frz = NU10 × exp(-ALPHA1 × (273.15 - T)^BETA1)
    NU20 : float
        Deposition nucleation parameter (m⁻³·s⁻¹), habit-dependent.
    ALPHA2, BETA2 : float
        Deposition nucleation temperature coefficients (dimensionless).
    MNU0 : float
        Mass of newly nucleated ice crystal (kg). Default: 6.88×10⁻¹³.
    
    **Homogeneous Ice Nucleation (HON)**
    
    ALPHA3, BETA3 : float
        Homogeneous freezing temperature coefficients.
    HON : float
        Homogeneous freezing rate coefficient (m³/s).
        Computed from Fletcher's formula for droplet size distribution.
    
    **Vapor Deposition and Evaporation**
    
    SCFAC : float
        Schmidt number factor for ventilation (dimensionless).
    
    Rain Evaporation (EVAR):
    O0EVAR, O1EVAR : float
        Evaporation coefficients without and with ventilation (m²/s).
    EX0EVAR, EX1EVAR : float
        Exponents for evaporation formulas (dimensionless).
    
    Ice Deposition (DEPI):
    O0DEPI, O2DEPI : float
        Deposition coefficients on ice crystals (m²/s).
    
    Snow Deposition (DEPS):
    O0DEPS, O1DEPS : float
        Deposition coefficients on snow (m²/s).
    EX0DEPS, EX1DEPS : float
        Exponents for snow deposition.
    RDEPSRED : float
        Tuning factor for snow deposition (dimensionless). Default: 1.0.
    
    Graupel Deposition (DEPG):
    O0DEPG, O1DEPG : float
        Deposition coefficients on graupel (m²/s).
    EX0DEPG, EX1DEPG : float
        Exponents for graupel deposition.
    RDEPGRED : float
        Tuning factor for graupel deposition (dimensionless). Default: 1.0.
    
    **Autoconversion (AUT)**
    
    Ice → Snow (IAUTI):
    TIMAUTI : float
        Time constant at T = T_t (s). Default: 1×10⁻³.
    TEXAUTI : float
        Temperature factor (K⁻¹). Default: 0.015.
    CRIAUTI : float
        Threshold ice mixing ratio (kg/kg).
    T0CRIAUTI : float
        Threshold temperature (°C).
    ACRIAUTI, BCRIAUTI : float
        Power law coefficients: rate = 10^(A×(T-T₀)+B).
    
    Cloud → Rain (AUTC):
    TIMAUTC : float
        Time constant (s). Default: 1×10⁻³.
    CRIAUTC : float
        Threshold cloud mixing ratio (kg/kg).
    
    **Snow Aggregation (AGG)**
    
    COLIS : float
        Ice-snow collection efficiency (dimensionless). Default: 0.25.
    COLEXIS : float
        Temperature factor for collection efficiency (K⁻¹). Default: 0.05.
    FIAGGS : float
        Aggregation rate pre-factor (m³·kg⁻¹·s⁻¹).
    EXIAGGS : float
        Aggregation rate exponent (dimensionless).
    
    **Cloud-Rain Accretion (ACC)**
    
    FCACCR : float
        Accretion rate pre-factor (m³·s⁻¹).
    EXCACCR : float
        Accretion rate exponent (dimensionless).
    
    **Snow Riming (RIM)**
    
    DCSLIM : float
        Diameter threshold for riming mode transition (m). Default: 7 mm.
    COLCS : float
        Cloud-snow collection efficiency (dimensionless). Default: 1.0.
    
    Small Snow Riming (CRIMSS):
    CRIMSS : float
        Riming coefficient for small aggregates (m³·s⁻¹).
    EXCRIMSS : float
        Exponent for small aggregate riming.
    
    Large Snow Riming (CRIMSG):
    CRIMSG : float
        Riming coefficient for large aggregates (m³·s⁻¹).
    EXCRIMSG : float
        Exponent for large aggregate riming.
    
    Snow→Graupel Conversion (RSRIMCG):
    SRIMCG, SRIMCG2, SRIMCG3 : float
        Conversion rate coefficients (various units).
    EXSRIMCG, EXSRIMCG2 : float
        Conversion rate exponents.
    
    **Riming Lookup Tables**
    
    NGAMINC : int
        Number of table values (80).
    GAMINC_BOUND_MIN, GAMINC_BOUND_MAX : float
        Lambda bounds for tables (m⁻¹).
    RIMINTP1, RIMINTP2 : float
        Interpolation coefficients.
    GAMINC_RIM1, GAMINC_RIM2, GAMINC_RIM4 : NDArray
        Incomplete gamma function tables for riming calculations (80 values each).
    
    **Rain-Snow Accretion (RACCS, RACCSS)**
    
    FRACCSS : float
        Pre-factor for accretion rate (kg·m⁻⁵·s⁻¹).
    LBRACCS1, LBRACCS2, LBRACCS3 : float
        Lambda-dependent coefficients for small snow accretion.
    FSACCRG : float
        Pre-factor for conversion to graupel (m³·s⁻¹).
    LBSACCR1, LBSACCR2, LBSACCR3 : float
        Lambda-dependent coefficients for large snow accretion.
    
    ACCLBDAS_MIN, ACCLBDAS_MAX : float
        Snow lambda range for kernel tables (m⁻¹).
    ACCLBDAR_MIN, ACCLBDAR_MAX : float
        Rain lambda range for kernel tables (m⁻¹).
    NACCLBDAS, NACCLBDAR : int
        Number of table points (40 each).
    ACCINTP1S, ACCINTP2S, ACCINTP1R, ACCINTP2R : float
        Interpolation coefficients for kernel tables.
    
    **Snow Melting**
    
    FSCVMG : float
        Conversion-melting factor (dimensionless). Default: 2.0.
    
    **Rain Contact Freezing (CFR)**
    
    COLIR : float
        Ice-rain collection efficiency (dimensionless). Default: 1.0.
    EXRCFRI, RCFRI : float
        Rain contact freezing rate coefficients.
    EXICFRR, ICFRR : float
        Ice-rain freezing coefficients.
    
    **Graupel Dry Growth (DRY)**
    
    Cloud Collection:
    FCDRYG : float
        Cloud droplet collection geometric factor (π/4).
    
    Ice Collection:
    COLIG : float
        Ice-graupel collection efficiency. Default: 0.01.
    COLEXIG : float
        Temperature factor for ice-graupel. Default: 0.1.
    FIDRYG, FIDRYG2 : float
        Ice collection rate coefficients.
    EXFIDRYG : float
        Ice collection exponent.
    
    Snow Collection:
    COLSG : float
        Snow-graupel collection efficiency. Default: 0.01.
    COLEXSG : float
        Temperature factor for snow-graupel. Default: 0.1.
    FSDRYG : float
        Snow collection coefficient.
    LBSDRYG1, LBSDRYG2, LBSDRYG3 : float
        Lambda-dependent coefficients.
    
    Rain Collection:
    FRDRYG : float
        Rain collection coefficient.
    LBRDRYG1, LBRDRYG2, LBRDRYG3 : float
        Lambda-dependent coefficients.
    
    Dry Growth Kernel Tables:
    DRYLBDAR_MIN, DRYLBDAR_MAX : float
        Rain lambda bounds (m⁻¹).
    DRYLBDAS_MIN, DRYLBDAS_MAX : float
        Snow lambda bounds (m⁻¹).
    DRYLBDAG_MIN, DRYLBDAG_MAX : float
        Graupel lambda bounds (m⁻¹).
    NDRYLBDAR, NDRYLBDAS, NDRYLBDAG : int
        Table dimensions (40, 80, 40).
    DRYINTP1R, DRYINTP2R, etc. : float
        Interpolation coefficients.
    
    **Precomputed Kernels**
    
    ker_saccrg, ker_raccs, ker_raccss : NDArray
        Snow-graupel and rain-snow accretion kernels (40×40).
    ker_rdryg, ker_sdryg : NDArray
        Rain and snow dry growth kernels on graupel (40×40 or 40×80).
    
    Notes
    -----
    All constants are computed in __post_init__ based on:
    - Hydrometeor size distributions (N(D) = N₀ × D^ν × exp(-λD^α))
    - Fall speed relations (V = A × D^B × (ρ/ρ₀)^C)
    - Collection kernels from kinetic equations
    - Moment integrals using gamma functions
    
    The formulas account for:
    - Air density corrections (rho00 factor)
    - Ventilation effects (enhanced mass/heat transfer)
    - Temperature dependencies
    - Particle habit variations
    
    Source Reference
    ----------------
    PHYEX/src/common/aux/modd_rain_ice_paramn.F90
    PHYEX/src/common/micro/mode_ini_rain_ice.F90
    
    See Also
    --------
    Constants : Physical constants
    RainIceDescriptors : Hydrometeor properties
    IceParameters : User configuration
    """

    # Constants dependencies
    cst: Constants
    rid: RainIceDescriptors
    parami: IceParameters

    # Parameters for microphysical sources and transformations
    FSEDC_1: cython.double  # Constants for sedimentation fluxes of C
    FSEDC_2: cython.double
    FSEDR: cython.double  # Constants for sedimentation
    EXSEDR: cython.double
    FSEDI: cython.double
    EXCSEDI: cython.double
    EXRSEDI: cython.double
    FSEDS: cython.double
    EXSEDS: cython.double
    FSEDG: cython.double
    EXSEDG: cython.double

    # Constants for heterogeneous ice nucleation HEN
    NU10: cython.double
    ALPHA1: cython.double
    BETA1: cython.double
    NU20: cython.double
    ALPHA2: cython.double
    BETA2: cython.double
    MNU0: cython.double  # Mass of nucleated ice crystal

    # Constants for homogeneous ice nucleation HON
    ALPHA3: cython.double
    BETA3: cython.double
    HON: cython.double

    # Constants for raindrop and evaporation EVA
    SCFAC: cython.double
    O0EVAR: cython.double
    O1EVAR: cython.double
    EX0EVAR: cython.double
    EX1EVAR: cython.double
    O0DEPI: cython.double  # deposition DEP on I
    O2DEPI: cython.double
    O0DEPS: cython.double  # on S
    O1DEPS: cython.double
    EX0DEPS: cython.double
    EX1DEPS: cython.double
    RDEPSRED: cython.double
    O0DEPG: cython.double  # on G
    O1DEPG: cython.double
    EX0DEPG: cython.double
    EX1DEPG: cython.double
    RDEPGRED: cython.double

    # Constants for pristine ice autoconversion : AUT
    TIMAUTI: cython.double  # Time constant at T=T_t
    TEXAUTI: cython.double
    CRIAUTI: cython.double
    T0CRIAUTI: cython.double
    ACRIAUTI: cython.double
    BCRIAUTI: cython.double

    # Constants for snow aggregation : AGG
    COLIS: cython.double  # Collection efficiency of I + S
    COLEXIS: cython.double  # Temperature factor of the I+S collection efficiency
    FIAGGS: cython.double
    EXIAGGS: cython.double

    # Constants for cloud droplet autoconversion AUT
    TIMAUTC: cython.double
    CRIAUTC: cython.double

    # Constants for cloud droplets accretion on raindrops : ACC
    FCACCR: cython.double
    EXCACCR: cython.double

    # Constants for the riming of the aggregates : RIM
    DCSLIM: cython.double
    COLCS: cython.double
    EXCRIMSS: cython.double
    CRIMSS: cython.double
    EXCRIMSG: cython.double
    CRIMSG: cython.double

    EXSRIMCG: cython.double
    EXSRIMCG2: cython.double
    SRIMCG: cython.double
    SRIMCG2: cython.double
    SRIMCG3: cython.double

    GAMINC_BOUND_MIN: cython.double
    GAMINC_BOUND_MAX: cython.double

    RIMINTP1: cython.double
    RIMINTP2: cython.double

    NGAMINC: cython.int  # Number of tab. Lbda_s

    GAMINC_RIM1: object  # NDArray
    GAMINC_RIM2: object  # NDArray
    GAMINC_RIM4: object  # NDArray

    # Constants for the accretion
    FRACCSS: cython.double
    LBRACCS1: cython.double
    LBRACCS2: cython.double
    LBRACCS3: cython.double
    FSACCRG: cython.double
    LBSACCR1: cython.double
    LBSACCR2: cython.double
    LBSACCR3: cython.double
    ACCLBDAS_MIN: cython.double
    ACCLBDAS_MAX: cython.double
    ACCLBDAR_MIN: cython.double
    ACCLBDAR_MAX: cython.double
    ACCINTP1S: cython.double
    ACCINTP2S: cython.double
    ACCINTP1R: cython.double
    ACCINTP2R: cython.double

    # number of values in global tables (per axis)
    NACCLBDAS: cython.int
    NACCLBDAR: cython.int

    # 7.3 Constant for the conversion-melting rate
    FSCVMG: cython.double

    # Constants for rain contact freezing
    COLIR: cython.double
    EXRCFRI: cython.double
    RCFRI: cython.double
    EXICFRR: cython.double
    ICFRR: cython.double

    # Constants for the dry growth of the graupel : DRY
    FCDRYG: cython.double
    COLIG: cython.double
    COLEXIG: cython.double
    FIDRYG: cython.double
    FIDRYG2: cython.double
    EXFIDRYG: cython.double
    COLSG: cython.double
    COLEXSG: cython.double
    FSDRYG: cython.double
    LBSDRYG1: cython.double
    LBSDRYG2: cython.double
    LBSDRYG3: cython.double
    FRDRYG: cython.double
    LBRDRYG1: cython.double
    LBRDRYG2: cython.double
    LBRDRYG3: cython.double
    DRYLBDAR_MIN: cython.double
    DRYLBDAR_MAX: cython.double
    DRYLBDAS_MIN: cython.double
    DRYLBDAS_MAX: cython.double
    DRYLBDAG_MIN: cython.double
    DRYLBDAG_MAX: cython.double
    DRYINTP1R: cython.double
    DRYINTP2R: cython.double
    DRYINTP1S: cython.double
    DRYINTP2S: cython.double
    DRYINTP1G: cython.double
    DRYINTP2G: cython.double

    NDRYLBDAR: cython.int
    NDRYLBDAS: cython.int
    NDRYLBDAG: cython.int

    # Kernel arrays
    ker_saccrg: object  # NDArray
    ker_raccs: object  # NDArray
    ker_raccss: object  # NDArray
    ker_rdryg: object  # NDArray
    ker_sdryg: object  # NDArray

    def __init__(self, cst: Constants, rid: RainIceDescriptors, parami: IceParameters):
        """Initialize RainIceParameters with dependencies."""
        self.cst = cst
        self.rid = rid
        self.parami = parami

        # Initialize all float attributes that will be computed
        self.FSEDC_1 = 0.0
        self.FSEDC_2 = 0.0
        self.FSEDR = 0.0
        self.EXSEDR = 0.0
        self.FSEDI = 0.0
        self.EXCSEDI = 0.0
        self.EXRSEDI = 0.0
        self.FSEDS = 0.0
        self.EXSEDS = 0.0
        self.FSEDG = 0.0
        self.EXSEDG = 0.0

        # Heterogeneous ice nucleation
        self.NU10 = 0.0
        self.ALPHA1 = 4.5
        self.BETA1 = 0.6
        self.NU20 = 0.0
        self.ALPHA2 = 12.96
        self.BETA2 = 0.639
        self.MNU0 = 6.88e-13

        # Homogeneous ice nucleation
        self.ALPHA3 = -3.075
        self.BETA3 = 81.00356
        self.HON = 0.0

        # Raindrop and evaporation
        self.SCFAC = 0.0
        self.O0EVAR = 0.0
        self.O1EVAR = 0.0
        self.EX0EVAR = 0.0
        self.EX1EVAR = 0.0
        self.O0DEPI = 0.0
        self.O2DEPI = 0.0
        self.O0DEPS = 0.0
        self.O1DEPS = 0.0
        self.EX0DEPS = 0.0
        self.EX1DEPS = 0.0
        self.RDEPSRED = 0.0
        self.O0DEPG = 0.0
        self.O1DEPG = 0.0
        self.EX0DEPG = 0.0
        self.EX1DEPG = 0.0
        self.RDEPGRED = 0.0

        # Pristine ice autoconversion
        self.TIMAUTI = 1e-3
        self.TEXAUTI = 0.015
        self.CRIAUTI = 0.0
        self.T0CRIAUTI = 0.0
        self.ACRIAUTI = 0.0
        self.BCRIAUTI = 0.0

        # Snow aggregation
        self.COLIS = 0.25
        self.COLEXIS = 0.05
        self.FIAGGS = 0.0
        self.EXIAGGS = 0.0

        # Cloud droplet autoconversion
        self.TIMAUTC = 1e-3
        self.CRIAUTC = 0.0

        # Cloud droplets accretion on raindrops
        self.FCACCR = 0.0
        self.EXCACCR = 0.0

        # Riming of the aggregates
        self.DCSLIM = 0.007
        self.COLCS = 1.0
        self.EXCRIMSS = 0.0
        self.CRIMSS = 0.0
        self.EXCRIMSG = 0.0
        self.CRIMSG = 0.0
        self.EXSRIMCG = 0.0
        self.EXSRIMCG2 = 0.0
        self.SRIMCG = 0.0
        self.SRIMCG2 = 0.0
        self.SRIMCG3 = 0.0

        self.GAMINC_BOUND_MIN = 0.0
        self.GAMINC_BOUND_MAX = 0.0
        self.RIMINTP1 = 0.0
        self.RIMINTP2 = 0.0
        self.NGAMINC = 0

        self.GAMINC_RIM1 = None
        self.GAMINC_RIM2 = None
        self.GAMINC_RIM4 = None

        # Accretion
        self.FRACCSS = 0.0
        self.LBRACCS1 = 0.0
        self.LBRACCS2 = 0.0
        self.LBRACCS3 = 0.0
        self.FSACCRG = 0.0
        self.LBSACCR1 = 0.0
        self.LBSACCR2 = 0.0
        self.LBSACCR3 = 0.0
        self.ACCLBDAS_MIN = 5e1
        self.ACCLBDAS_MAX = 5e5
        self.ACCLBDAR_MIN = 1e3
        self.ACCLBDAR_MAX = 1e7
        self.ACCINTP1S = 0.0
        self.ACCINTP2S = 0.0
        self.ACCINTP1R = 0.0
        self.ACCINTP2R = 0.0

        self.NACCLBDAS = 40
        self.NACCLBDAR = 40

        # Conversion-melting rate
        self.FSCVMG = 2.0

        # Rain contact freezing
        self.COLIR = 1.0
        self.EXRCFRI = 0.0
        self.RCFRI = 0.0
        self.EXICFRR = 0.0
        self.ICFRR = 0.0

        # Dry growth of the graupel
        self.FCDRYG = 0.0
        self.COLIG = 0.01
        self.COLEXIG = 0.1
        self.FIDRYG = 0.0
        self.FIDRYG2 = 0.0
        self.EXFIDRYG = 0.0
        self.COLSG = 0.01
        self.COLEXSG = 0.1
        self.FSDRYG = 0.0
        self.LBSDRYG1 = 0.0
        self.LBSDRYG2 = 0.0
        self.LBSDRYG3 = 0.0
        self.FRDRYG = 0.0
        self.LBRDRYG1 = 0.0
        self.LBRDRYG2 = 0.0
        self.LBRDRYG3 = 0.0
        self.DRYLBDAR_MIN = 1e3
        self.DRYLBDAR_MAX = 1e7
        self.DRYLBDAS_MIN = 2.5e1
        self.DRYLBDAS_MAX = 2.5e9
        self.DRYLBDAG_MIN = 1e3
        self.DRYLBDAG_MAX = 1e7
        self.DRYINTP1R = 0.0
        self.DRYINTP2R = 0.0
        self.DRYINTP1S = 0.0
        self.DRYINTP2S = 0.0
        self.DRYINTP1G = 0.0
        self.DRYINTP2G = 0.0

        self.NDRYLBDAR = 40
        self.NDRYLBDAS = 80
        self.NDRYLBDAG = 40

        # Kernel arrays - will be initialized in __post_init__
        self.ker_saccrg = None
        self.ker_raccs = None
        self.ker_raccss = None
        self.ker_rdryg = None
        self.ker_sdryg = None

        # Call post_init to compute all derived values
        self.__post_init__()

    def __post_init__(self):
        # 4. CONSTANTS FOR THE SEDIMENTATION
        # 4.1 Exponent of the fall-speed air density correction

        e = 0.5 * np.exp(
            self.cst.ALPW - self.cst.BETAW / 293.15 - self.cst.GAMW * log(293.15)
        )
        rv = (self.cst.RD_RV) * e / (101325 - e)
        rho00 = 101325 * (1 + rv) / (self.cst.RD + rv * self.cst.RV) / 293.15

        # 4.2    Constants for sedimentation
        self.FSEDC_1, self.FSEDC_2 = (
            (
                gamma(self.rid.NUC + (self.rid.DC + 3) / self.rid.ALPHAC)
                / gamma(self.rid.NUC + 3 / self.rid.ALPHAC)
                * rho00**self.rid.CEXVT
            ),
            (
                gamma(self.rid.NUC2 + (self.rid.DC + 3) / self.rid.ALPHAC2)
                / gamma(self.rid.NUC2 + 3 / self.rid.ALPHAC2)
                * rho00**self.rid.CEXVT
            ),
        )

        momg = lambda alpha, nu, p: gamma(nu + p / alpha) / gamma(nu)

        self.EXSEDR = (self.rid.BR + self.rid.DR + 1.0) / (self.rid.BR + 1.0)
        self.FSEDR = (
            self.rid.CR
            + self.rid.AR
            + self.rid.CCR
            * momg(self.rid.ALPHAR, self.rid.NUR, self.rid.BR)
            * (
                self.rid.AR
                * self.rid.CCR
                * momg(self.rid.ALPHAR, self.rid.NUR, self.rid.BR) ** (-self.EXSEDR)
                * rho00**self.rid.CEXVT
            )
        )

        self.EXRSEDI = (self.rid.BI + self.rid.DI) / self.rid.BI
        self.EXCSEDI = 1 - self.EXRSEDI
        self.FSEDI = (
            (4 * 900 * self.cst.PI) ** (-self.EXCSEDI)
            * self.rid.C_I
            * self.rid.AI
            * momg(self.rid.ALPHAI, self.rid.NUI, self.rid.BI + self.rid.DI)
            * (
                (self.rid.AI * momg(self.rid.ALPHAI, self.rid.NUI, self.rid.BI))
                ** (-self.EXRSEDI)
                * rho00**self.rid.CEXVT
            )
        )

        self.EXSEDS = (self.rid.BS + self.rid.DS - self.rid.CXS) / (
            self.rid.BS - self.rid.CXS
        )
        self.SEDS = (
            self.rid.CS
            * self.rid.A_S
            * self.rid.CCS
            * momg(self.rid.ALPHAS, self.rid.NUS, self.rid.BS + self.rid.DS)
            * (
                self.rid.A_S
                * self.rid.CCS
                * momg(self.rid.ALPHAS, self.rid.NUS, self.rid.BS)
            )
            ** (-self.EXSEDS)
            * rho00**self.rid.CEXVT
        )

        # if self.parami.LRED:
        self.EXSEDS = self.rid.DS - self.rid.BS
        self.FSEDS = (
            self.rid.CS
            * momg(self.rid.ALPHAS, self.rid.NUS, self.rid.BS + self.rid.DS)
            / momg(self.rid.ALPHAS, self.rid.NUS, self.rid.BS)
            * rho00**self.rid.CEXVT
        )

        self.EXSEDG = (self.rid.BG + self.rid.DG - self.rid.CXG) / (
            self.rid.BG - self.rid.CXG
        )
        self.FSEDG = (
            self.rid.CG
            * self.rid.AG
            * self.rid.CCG
            * momg(self.rid.ALPHAG, self.rid.NUG, self.rid.BG + self.rid.DG)
            * (
                self.rid.AG
                * self.rid.CCG
                * momg(self.rid.ALPHAG, self.rid.NUG, self.rid.BG)
            )
            ** (-self.EXSEDG)
            * rho00
            * self.rid.CEXVT
        )

        # 5. Constants for the slow cold processes
        fact_nucl = 0
        if self.parami.PRISTINE_ICE == "PLAT":
            fact_nucl = 1.0  # Plates
        elif self.parami.PRISTINE_ICE == "COLU":
            fact_nucl = 25.0  # Columns
        elif self.parami.PRISTINE_ICE == "BURO":
            fact_nucl = 17.0  # Bullet rosettes

        self.NU10 = 50 * fact_nucl
        self.NU20 = 1000 * fact_nucl

        self.HON = (self.cst.PI / 6) * ((2 * 3 * 4 * 5 * 6) / (2 * 3)) * (1.1e5) ** (-3)

        # 5.2 Constants for vapor deposition on ice
        self.SCFAC = (0.63 ** (1 / 3)) * np.sqrt((rho00) ** self.rid.CEXVT)
        self.O0DEPI = (
            (4 * self.cst.PI)
            * self.rid.C1I
            * self.rid.F0I
            * momg(self.rid.ALPHAI, self.rid.NUI, 1)
        )
        self.O2DEPI = (
            (4 * self.cst.PI)
            * self.rid.C1I
            * self.rid.F2I
            * self.rid.C_I
            * momg(self.rid.ALPHAI, self.rid.NUI, self.rid.DI + 2.0)
        )

        # Translation note: #ifdef REPRO48 l588 to l591 kept in mode_ini_rain_ice.F90
        #                                  l593 to l596 removed
        self.O0DEPS = (
            self.rid.NS
            * (4 * self.cst.PI)
            * self.rid.CCS
            * self.rid.C1S
            * self.rid.F0S
            * momg(self.rid.ALPHAS, self.rid.NUS, 1)
        )
        self.O1DEPS = (
            self.rid.NS
            * (4 * self.cst.PI)
            * self.rid.CCS
            * self.rid.C1S
            * self.rid.F1S
            * np.sqrt(self.rid.CS)
            * momg(self.rid.ALPHAS, self.rid.NUS, 0.5 * self.rid.DS + 1.5)
        )
        self.EX0DEPS = self.rid.CXS - 1.0
        self.EX1DEPS = self.rid.CXS - 0.5 * (self.rid.DS + 3.0)
        self.RDEPSRED = self.parami.RDEPSRED_NAM

        self.O0DEPG = (
            (4 * self.cst.PI)
            * self.rid.CCG
            * self.rid.C1G
            * self.rid.F0G
            * momg(self.rid.ALPHAG, self.rid.NUG, 1)
        )
        self.O1DEPG = (
            (4 * self.cst.PI)
            * self.rid.CCG
            * self.rid.C1G
            * self.rid.F1G
            * np.sqrt(self.rid.CG)
            * momg(self.rid.ALPHAG, self.rid.NUG, 0.5 * self.rid.DG + 1.5)
        )
        self.EX0DEPG = self.rid.CXG - 1.0
        self.EX1DEPG = self.rid.CXG - 0.5 * (self.rid.DG + 3.0)
        self.RDEPGRED = self.parami.RDEPGRED_NAM

        # 5.3 Constants for pristine ice autoconversion
        self.CRIAUTI = self.parami.CRIAUTI_NAM
        if self.parami.LCRIAUTI == 1:
            self.T0CRIAUTI = self.parami.T0CRIAUTI_NAM
            tcrio = -40
            crio = 1.25e-6
            self.BCRIAUTI = (
                -(np.log10(self.CRIAUTI) - np.log10(crio) * self.T0CRIAUTI / tcrio)
                * tcrio
                / (self.T0CRIAUTI - tcrio)
            )
            self.ACRIAUTI = (np.log10(crio) - self.BCRIAUTI) / tcrio

        else:
            self.ACRIAUTI = self.parami.ACRIAUTI_NAM
            self.BCRIAUTI = self.parami.BRCRIAUTI_NAM
            self.T0CRIAUTI = (np.log10(self.CRIAUTI) - self.BCRIAUTI) / 0.06

        # 5.4 Constants for snow aggregation
        # Translation note: #ifdef REPRO48 l655 to l656 kept in mode_ini_rain_ice.F90
        #                                  l658 to l659 removed
        self.FIAGGS = (
            (self.cst.PI / 4)
            * self.COLIS
            * self.rid.CCS
            * self.rid.CS
            * (rho00**self.rid.CEXVT)
            * momg(self.rid.ALPHAS, self.rid.NUS, self.rid.DS + 2.0)
        )
        self.EXIAGGS = self.rid.CXS - self.rid.DS - 2.0

        # 6. Constants for the slow warm processes
        # 6.1 Constants for the accretion of cloud droplets autoconversion
        self.CRIAUTC = self.parami.CRIAUTC_NAM

        # 6.2 Constants for the accretion of cloud droplets by raindrops
        self.FCACCR = (
            (self.cst.PI / 4)
            * self.rid.CCR
            * self.rid.CR
            * (rho00**self.rid.CEXVT)
            * momg(self.rid.ALPHAR, self.rid.NUR, self.rid.DR + 2.0)
        )
        self.EXCACCR = -self.rid.DR - 3.0

        # 6.3 Constants for the evaporation of rain drops
        self.O0EVAR = (
            (4.0 * self.cst.PI)
            * self.rid.CCR
            * self.rid.CR
            * self.rid.F0R
            * momg(self.rid.ALPHAR, self.rid.NUR, 1)
        )
        self.O1EVAR = (
            (4.0 * self.cst.PI)
            * self.rid.CCR
            * self.rid.C1R
            * self.rid.F1R
            * momg(self.rid.ALPHAR, self.rid.NUR, 0.5 * self.rid.DR + 1.5)
        )
        self.EX0EVAR = -2.0
        self.EX1EVAR = -1.0 - 0.5 * (self.rid.DR + 3.0)

        # 7. Constants for the fast cold processes for the aggregateDS
        # 7.1 Constants for the riming of the aggregates

        # Translation note: #ifdef REPRO48 l712 and l713 kept in mode_ini_rain_ice.F90
        #                                  l715 and l716 removed

        # Translation note: #ifdef REPRO48 l721 to l725 kept in mode_ini_rain_ice.F90
        #                                  l727 to l731 removed

        self.EXCRIMSS = -self.rid.DS - 2.0
        self.CRIMSS = (
            self.rid.NS
            * (self.cst.PI / 4)
            * self.COLCS
            * self.rid.CS
            * (rho00**self.rid.CEXVT)
            * momg(self.rid.ALPHAS, self.rid.NUS, self.rid.DS + 2)
        )

        self.EXCRIMSG = self.EXCRIMSS
        self.CRIMSG = self.CRIMSS

        self.SRIMCG = (
            self.rid.CCS
            * self.rid.A_S
            * momg(self.rid.ALPHAS, self.rid.NUS, self.rid.BS)
        )
        self.EXSRIMCG = self.rid.CXS - self.rid.BS
        self.SRIMCG2 = (
            self.rid.CCS
            * self.rid.AG
            * momg(self.rid.ALPHAS, self.rid.NUS, self.rid.BS)
        )
        self.SRIMCG3 = self.parami.FRAC_M90
        self.EXSRIMCG2 = self.rid.CXS - self.rid.BG

        self.NGAMINC = 80
        self.GAMINC_BOUND_MIN = 1e-1
        self.GAMINC_BOUND_MAX = 1e7
        zrate = np.exp(
            log(self.GAMINC_BOUND_MAX / self.GAMINC_BOUND_MIN) / (self.NGAMINC - 1)
        )

        self.RIMINTP1 = self.rid.ALPHAS / log(zrate)
        self.RIMINTP2 = 1 + self.RIMINTP1 * log(
            self.DCSLIM / (self.GAMINC_BOUND_MIN) ** (1 / self.rid.ALPHAS)
        )

        # init GAMINC_RIM1, GAMINC_RIM2, GAMINC_RIM4
        self.init_gaminc_rim_tables()

        # 7.2 Constants for the accretion of raindrops

        # Translation note: #ifdef REPRO48 l763 kept
        #                                  l765 removed

        self.FRACCSS = (
            ((self.cst.PI**2) / 24)
            * self.rid.CCS
            * self.rid.CCR
            * self.cst.RHOLW
            * (rho00**self.rid.CEXVT)
        )

        self.LBRACCS1 = momg(self.rid.ALPHAS, self.rid.NUS, 2) * momg(
            self.rid.ALPHAR, self.rid.NUR, 3
        )
        self.LBRACCS2 = (
            2
            * momg(self.rid.ALPHAS, self.rid.NUS, 1)
            * momg(self.rid.ALPHAR, self.rid.NUR, 4)
        )
        self.LBRACCS3 = momg(self.rid.ALPHAR, self.rid.NUR, 5)

        # Translation note : #ifdef REPRO48 l773 kept
        #                                   l775 removed
        self.FSACCRG = (
            (self.cst.PI / 4)
            * self.rid.A_S
            * self.rid.CCS
            * self.rid.CCR
            * (rho00**self.rid.CEXVT)
        )

        self.LBSACCR1 = momg(self.rid.ALPHAR, self.rid.NUR, 2) * momg(
            self.rid.ALPHAS, self.rid.NUS, self.rid.BS
        )
        self.LBSACCR2 = momg(self.rid.ALPHAR, self.rid.NUR, 1) * momg(
            self.rid.ALPHAS, self.rid.NUS, self.rid.BS + 1
        )
        self.LBSACCR3 = momg(self.rid.ALPHAS, self.rid.NUS, self.rid.BS + 2)

        # Defining the ranges for the computation of kernels
        zrate = log(self.ACCLBDAS_MAX / self.ACCLBDAS_MIN) / (self.NACCLBDAS - 1)
        self.ACCINTP1S = 1 / zrate
        self.ACCINTP2S = 1 - log(self.ACCLBDAS_MIN) / zrate

        zrate = log(self.ACCLBDAR_MAX / self.ACCLBDAR_MIN) / (self.NACCLBDAR - 1)
        self.ACCINTP1R = 1 / zrate
        self.ACCINTP2R = 1 - log(self.ACCLBDAR_MIN) / zrate

        # Translation note : l800 to 902 -> computation of the kernels

        # 8 Constants for the fast cold processes for the graupel

        # 8.1 Constants for the rain contact freezing
        xr = -1
        self.EXRCFRI = -self.rid.DR - 5 + xr
        self.RCFRI = (
            ((self.cst.PI**2) / 24)
            * self.rid.CCR
            * self.cst.RHOLW
            * self.COLIR
            * self.rid.CR
            * (rho00 * self.rid.CEXVT)
            * momg(self.rid.ALPHAR, self.rid.NUR, self.rid.DR + 5)
        )

        self.EXICFRR = -self.rid.DR - 2 + xr
        self.ICFRR = (
            (self.cst.PI / 4)
            * self.COLIR
            * self.rid.CR
            * (rho00**self.rid.CEXVT)
            * self.rid.CCR
            * momg(self.rid.ALPHAR, self.rid.NUR, self.rid.DR + 2)
        )

        # 8.2 Constants for the dry growth of the graupel

        # 8.2.1 Constants for the cloud droplets collection by the graupel
        self.FCDRYG = self.cst.PI / 4

        # 8.2.2 Constants for the cloud ice collection by the graupel
        self.FIDRYG = (
            (self.cst.PI / 4)
            * self.COLIG
            * self.rid.CCG
            * (rho00**self.rid.CEXVT)
            * momg(self.rid.ALPHAG, self.rid.NUG, self.rid.DG + 2)
        )
        self.EXFIDRYG = (self.rid.CXG - self.rid.DG - 2) / (self.rid.CXG - self.rid.BG)
        self.FIDRYG2 = (
            self.FIDRYG
            * self.COLIG
            * (
                self.rid.AG
                * self.rid.CCG
                * momg(self.rid.ALPHAG, self.rid.NUG, self.rid.BG)
            )
        )

        # 8.2.3 Constants for the aggregate collection by the graupel
        # Translation note : #ifdef REPRO48 l973 kept
        #                                   l975 removed
        self.FSDRYG = (
            (self.cst.PI**2 / 24)
            * self.rid.CCG
            * self.rid.CCR
            * self.cst.RHOLW
            * (rho00**self.rid.CEXVT)
        )
        self.LBSDRYG1 = momg(self.rid.ALPHAG, self.rid.NUG, 2) * momg(
            self.rid.ALPHAS, self.rid.NUS, self.rid.BS
        )
        self.LBSDRYG2 = (
            2
            * momg(self.rid.ALPHAG, self.rid.NUG, 1)
            * momg(self.rid.ALPHAS, self.rid.NUS, self.rid.BS + 1)
        )
        self.LBSDRYG3 = momg(self.rid.ALPHAS, self.rid.NUS, self.rid.BS + 2)

        # 8.2.4 Constants for the raindrop collection by the graupel
        self.FRDRYG = (
            (self.cst.PI**2 / 24)
            * self.rid.CCG
            * self.rid.CCG
            * self.cst.RHOLW
            * (rho00**self.rid.CEXVT)
        )
        self.LBRDRYG1 = momg(self.rid.ALPHAG, self.rid.NUG, 3) * momg(
            self.rid.ALPHAR, self.rid.NUR, 3
        )
        self.LBRDRYG2 = (
            2
            * momg(self.rid.ALPHAG, self.rid.NUG, 1)
            * momg(self.rid.ALPHAR, self.rid.NUR, 4)
        )
        self.LBRDRYG3 = momg(self.rid.ALPHAR, self.rid.NUR, 5)

        # Notice: one magnitude of lambda discretized over 10 points
        zrate = log(self.DRYLBDAR_MAX / self.DRYLBDAR_MIN) / (self.NDRYLBDAR - 1)
        self.DRYINTP1R = 1 / zrate
        self.DRYINTP2R = 1 - log(self.DRYLBDAR_MIN) / zrate

        zrate = log(self.DRYLBDAS_MAX / self.DRYLBDAS_MIN) / (self.NDRYLBDAS - 1)
        self.DRYINTP1S = 1 / zrate
        self.DRYINTP2S = 1 - log(self.DRYLBDAS_MIN) / zrate

        zrate = log(self.DRYLBDAG_MAX / self.DRYLBDAG_MIN) / (self.NDRYLBDAG - 1)
        self.DRYINTP1G = 1 / zrate
        self.DRYINTP2G = 1 - log(self.DRYLBDAG_MIN) / zrate

        # Translation note : l1018 to l1152 -> computation of the kernels

        # Tranlsation note : l1154 to l1160 skipped

        # Tranlsation note : 9. Constants for hailstones not implemented
        #                    l1162 to l1481 removed

        # kernels
        self.ker_saccrg = self.get_kernel("saccrg")
        self.ker_raccs = self.get_kernel("raccs")
        self.ker_raccss = self.get_kernel("raccss")
        self.ker_rdryg = self.get_kernel("rdryg")
        self.ker_sdryg = self.get_kernel("sdryg")

    # from_file="PHYEX/src/common/micro/mode_ini_rain_ice.F90"
    def init_gaminc_rim_tables(self):
        """Compute generalized incomplete gamma tables for riming"""

        zrate = np.exp(
            log(self.GAMINC_BOUND_MAX / self.GAMINC_BOUND_MIN) / (self.NGAMINC - 1)
        )

        try:
            logging.info(
                f"a factor {self.rid.NUS + (2 + self.rid.DS) / self.rid.ALPHAS}, GAMINC * zrate = {self.GAMINC_BOUND_MIN * zrate}"
            )

            GAMINC_RIM1 = np.array(
                [
                    generalized_incomplete_gamma(
                        self.rid.NUS + (2 + self.rid.DS) / self.rid.ALPHAS,
                        self.GAMINC_BOUND_MIN * zrate * j1,
                    )
                    for j1 in range(1, self.NGAMINC + 1)
                ]
            )
            GAMINC_RIM2 = np.array(
                [
                    generalized_incomplete_gamma(
                        self.rid.NUS + self.rid.BS / self.rid.ALPHAS,
                        self.GAMINC_BOUND_MIN * zrate * j1,
                    )
                    for j1 in range(1, self.NGAMINC + 1)
                ]
            )
            GAMINC_RIM4 = np.array(
                [
                    generalized_incomplete_gamma(
                        self.rid.NUS + self.rid.BS / self.rid.ALPHAS,
                        self.GAMINC_BOUND_MIN * zrate * j1,
                    )
                    for j1 in range(1, self.NGAMINC + 1)
                ]
            )

        except ValueError as e:
            logging.info(
                f"Value error while computing generalized_incomplete_gamma : {e}"
            )
            GAMINC_RIM1 = np.ones(80)
            GAMINC_RIM2 = np.ones(80)
            GAMINC_RIM4 = np.ones(80)

        finally:
            self.GAMINC_RIM1 = GAMINC_RIM1
            self.GAMINC_RIM2 = GAMINC_RIM2
            self.GAMINC_RIM4 = GAMINC_RIM4

    def get_kernel(
        self, kernel: str
    ):
        """Load kernels for convolutions as numpy arrays

        Args:
            kernel (str): kernel to load

        Raises:
            KeyError: Error if name is not in kernel names (saccrg, raccs, raccss, rdryg, sdryg)

        Returns:
            _type_: python indented kernel (from 0 to n-1, instead 1 to n in fortran)
        """

        if kernel == "saccrg":
            from ice3.phyex_common.xker_raccs import KER_SACCRG

            return KER_SACCRG[1:, 1:]

        elif kernel == "raccs":
            from ice3.phyex_common.xker_raccs import KER_RACCS

            return KER_RACCS[1:, 1:]

        elif kernel == "raccss":
            from ice3.phyex_common.xker_raccs import KER_RACCSS

            return KER_RACCSS[1:, 1:]

        elif kernel == "rdryg":
            from ice3.phyex_common.xker_rdryg import KER_RDRYG

            return KER_RDRYG[1:, 1:]

        elif kernel == "sdryg":
            from ice3.phyex_common.xker_sdryg import KER_SDRYG

            return KER_SDRYG[1:, 1:]

        else:
            raise KeyError(f"{kernel} not found in GlobalTables")


@cython.cclass
class CloudPar:
    """Declaration of the model-n dependant Microphysic constants

    Args:
        nsplitr (int): Number of required small time step integration
            for rain sedimentation computation
        nsplitg (int): Number of required small time step integration
            for ice hydrometeor sedimentation computation

    """

    nsplitr: cython.int
    nsplitg: cython.int

    def __init__(self, nsplitr: int, nsplitg: int):
        """Initialize CloudPar with split numbers."""
        self.nsplitr = nsplitr
        self.nsplitg = nsplitg
