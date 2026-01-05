# -*- coding: utf-8 -*-
from math import gamma, sqrt
import numpy as np
import cython

from ice3.phyex_common.constants import Constants
from ice3.phyex_common.ice_parameters import IceParameters


# from_file="PHYEX/src/common/aux/modd_rain_ice_descrn.F90"
@cython.cclass
class RainIceDescriptors:
    """
    Hydrometeor physical properties and size distribution parameters.
    
    This dataclass defines the fundamental properties of each hydrometeor
    species (cloud, rain, ice, snow, graupel) including mass-diameter
    relations, terminal fall speeds, particle size distributions, and
    ventilation coefficients. These descriptors are used throughout the
    microphysics calculations.
    
    **Key Relationships:**
    
    Mass-Diameter: m(D) = A_x × D^B_x
    Fall Speed: v(D) = C_x × D^D_x × (ρ/ρ₀)^CEXVT
    Size Distribution: N(D) = N₀ × D^ν × exp(-λ×D^α)
    Number Concentration: N = CC_x × λ^CX_x
    Slope Parameter: λ = LB_x × (r_x × ρ_ref)^LBEX_x
    
    Attributes
    ----------
    
    **Dependencies**
    
    cst : Constants
        Physical constants.
    parami : IceParameters
        User configuration parameters.
    
    **General Parameters**
    
    CEXVT : float
        Air density exponent for fall speed correction. Default: 0.4.
        Accounts for v ∝ (ρ₀/ρ)^CEXVT.
    
    RTMIN : np.ndarray
        Minimum allowed mixing ratios for all species (41 values, kg/kg).
    
    **Cloud Droplets (subscript C)**
    
    Mass-Diameter Relations:
    AC : float
        Mass prefactor (kg/m^BC). Default: 524 (assumes spherical, ρ=1000 kg/m³).
    BC : float
        Mass exponent (dimensionless). Default: 3.0 (spherical).
    
    Fall Speed Relations:
    CC : float
        Fall speed prefactor (m^(1-DC)/s). Default: 842.
    DC : float
        Fall speed exponent (dimensionless). Default: 2.0.
    
    Size Distribution (Bimodal over land/sea):
    ALPHAC : float
        Shape parameter for gamma distribution over land. Default: 1.0.
    NUC : float
        Scale parameter over land. Default: 3.0 (narrow distribution).
    ALPHAC2 : float
        Shape parameter over sea. Default: 1.0.
    NUC2 : float
        Scale parameter over sea. Default: 1.0 (broader distribution).
    
    LBEXC : float
        Slope parameter exponent (computed).
    LBC_1, LBC_2 : float
        Slope parameter prefactors for land and sea (computed).
    
    Diagnostic Concentrations:
    CONC_LAND : float
        Droplet concentration over land (m⁻³). Default: 3×10⁸.
    CONC_SEA : float
        Over sea (m⁻³). Default: 1×10⁸.
    CONC_URBAN : float
        Over urban areas (m⁻³). Default: 5×10⁸.
    
    **Rain Drops (subscript R)**
    
    Mass-Diameter:
    AR : float
        Mass prefactor (kg/m^BR). Default: 524.
    BR : float
        Mass exponent. Default: 3.0 (spherical).
    
    Fall Speed (Marshall-Palmer):
    CR : float
        Fall speed prefactor (m^(1-DR)/s). Default: 842.
    DR : float
        Fall speed exponent. Default: 0.8.
    
    Size Distribution:
    CCR : float
        Number concentration prefactor (m⁻⁴). Default: 8×10⁻⁶.
    ALPHAR : float
        Shape parameter. Default: 3.0.
    NUR : float
        Scale parameter. Default: 1.0 (exponential).
    LBEXR : float
        Slope exponent (computed).
    LBR : float
        Slope prefactor (computed, m⁻¹).
    
    Ventilation Coefficients:
    F0R : float
        Non-ventilated coefficient. Default: 1.0.
    F1R : float
        Ventilated coefficient. Default: 0.26.
    C1R : float
        Thermodynamic shape parameter. Default: 0.5.
    
    **Ice Crystals (subscript I) - Habit Dependent**
    
    Properties depend on PRISTINE_ICE setting:
    
    Plates (PLAT):
    AI : float
        Mass prefactor. Default: 0.82 kg/m^BI.
    BI : float
        Mass exponent. Default: 2.5.
    C_I : float
        Fall speed prefactor. Default: 800 m^(1-DI)/s.
    DI : float
        Fall speed exponent. Default: 1.0.
    C1I : float
        Shape parameter. Default: 1/π.
    
    Columns (COLU):
    AI = 2.14×10⁻³, BI = 1.7, C_I = 2.1×10⁵, DI = 1.585, C1I = 0.8
    
    Bullet Rosettes (BURO):
    AI = 44.0, BI = 3.0, C_I = 4.3×10⁵, DI = 1.663, C1I = 0.5
    
    Size Distribution:
    ALPHAI : float
        Shape parameter. Default: 1.0 (exponential).
    NUI : float
        Scale parameter. Default: 1.0.
    LBEXI, LBI : float
        Slope parameters (computed).
    
    Ventilation:
    F0I : float
        Non-ventilated. Default: 1.0.
    F2I : float
        Ventilated (enhanced diffusion). Default: 0.14.
    
    **Snow/Aggregates (subscript S)**
    
    Mass-Diameter:
    A_S : float
        Mass prefactor (kg/m^BS). Default: 0.02 (low density aggregates).
    BS : float
        Mass exponent. Default: 1.9.
    
    Fall Speed:
    CS : float
        Fall speed prefactor. Default: 5.1 m^(1-DS)/s.
    DS : float
        Fall speed exponent. Default: 0.27.
    FVELOS : float
        Additional fall speed factor. Default: 0.097.
        Set to 25.14 if LSNOW_T=True (Wurtz parameterization).
    
    Size Distribution:
    CCS : float
        Number concentration prefactor. Default: 5.0.
    CXS : float
        Concentration-slope relation exponent. Default: 1.0.
    ALPHAS, NUS : float
        Shape parameters. Default: 1.0, 1.0.
        Modified if LSNOW_T=True: ALPHAS=0.214, NUS=43.7.
    NS : float
        Number to mass conversion factor (computed).
    LBEXS, LBS : float
        Slope parameters (computed).
    TRANS_MP_GAMMAS : float
        Gamma function transformation coefficient. Default: 1.0.
    
    Ventilation:
    F0S, F1S : float
        Ventilation coefficients. Default: 0.86, 0.28.
    C1S : float
        Shape parameter (computed as 1/π).
    
    **Graupel (subscript G)**
    
    Mass-Diameter:
    AG : float
        Mass prefactor (kg/m^BG). Default: 19.6.
    BG : float
        Mass exponent. Default: 2.8.
    
    Fall Speed:
    CG : float
        Fall speed prefactor. Default: 124 m^(1-DG)/s.
    DG : float
        Fall speed exponent. Default: 0.66.
    
    Size Distribution:
    CCG : float
        Number concentration prefactor. Default: 5×10⁵.
    CXG : float
        Concentration-slope exponent. Default: -0.5.
    ALPHAG, NUG : float
        Shape parameters. Default: 1.0, 1.0 (exponential).
    LBEXG, LBG : float
        Slope parameters (computed).
    
    Ventilation:
    F0G, F1G : float
        Ventilation coefficients. Default: 0.86, 0.28.
    C1G : float
        Shape parameter. Default: 0.5.
    
    **Minimum Mixing Ratio Thresholds**
    
    V_RTMIN : float
        Vapor threshold (kg/kg). Default: 1×10⁻²⁰.
    C_RTMIN : float
        Cloud threshold. Default: 1×10⁻²⁰.
    R_RTMIN : float
        Rain threshold. Default: 1×10⁻²⁰.
    I_RTMIN : float
        Ice threshold. Default: 1×10⁻²⁰.
    S_RTMIN : float
        Snow threshold. Default: 1×10⁻¹⁵.
    G_RTMIN : float
        Graupel threshold. Default: 1×10⁻¹⁵.
    
    **Maximum Slope Parameters**
    
    LBDAR_MAX, LBDAS_MAX, LBDAG_MAX : float
        Maximum λ values (m⁻¹). Default: 1×10⁵.
    LBDAS_MIN : float
        Minimum λ for snow (m⁻¹). Default: 1×10⁻¹⁰.
    
    **Statistical Sedimentation Parameters**
    
    GAC, GC : float
        Gamma function values for cloud (land).
    GAC2, GC2 : float
        Gamma function values for cloud (sea).
    RAYDEF0 : float
        Default radius factor (computed).
    
    Notes
    -----
    
    **Mass-Diameter Relations:**
    The mass m relates to maximum dimension D as m = A × D^B.
    - B=3: Spherical particles (cloud, rain)
    - B<3: Non-spherical aggregates (snow, some ice)
    - B>3: Rimed/dense particles
    
    **Fall Speed Relations:**
    Terminal velocity v = C × D^D × (ρ₀/ρ)^CEXVT
    - D→1: Stokes regime (small particles)
    - D→0.5: Intermediate regime
    - D→0: Best number regime (large particles)
    
    **Size Distributions:**
    Generalized gamma: N(D) = N₀ × D^ν × exp(-λ×D^α)
    - α=1, ν=0: Exponential (Marshall-Palmer for rain)
    - α=3, ν=0: Gamma distribution
    - Higher ν: Broader distributions
    
    **Ventilation Coefficients:**
    Enhanced mass/heat transfer for falling particles:
    - F0: Motionless particle (diffusion only)
    - F1: Enhanced by particle motion (Re^0.5 dependence)
    - F2: Further enhanced (Re dependence)
    
    **Habit Dependencies:**
    Ice crystal properties (A_I, B_I, C_I, D_I, C1I) depend on 
    the dominant crystal habit (plates, columns, or bullet rosettes),
    which affects mass, area, and fall speed.
    
    Source Reference
    ----------------
    PHYEX/src/common/aux/modd_rain_ice_descrn.F90
    
    References
    ----------
    Marshall, J.S., and W.M. Palmer, 1948: The distribution of raindrops
    with size. J. Meteor., 5, 165-166.
    
    Locatelli, J.D., and P.V. Hobbs, 1974: Fall speeds and masses of
    solid precipitation particles. J. Geophys. Res., 79, 2185-2197.
    
    Field, P.R., et al., 2005: Snow size distribution parameterization
    for midlatitude and tropical ice clouds. J. Atmos. Sci., 62, 4446-4468.
    
    Examples
    --------
    >>> from ice3.phyex_common.constants import Constants
    >>> from ice3.phyex_common.ice_parameters import IceParameters
    >>> cst = Constants()
    >>> param = IceParameters(HPROGRAM="AROME", PRISTINE_ICE="PLAT")
    >>> rid = RainIceDescriptors(cst=cst, parami=param)
    >>> print(f"Rain mass law: m = {rid.AR} × D^{rid.BR}")
    Rain mass law: m = 524 × D^3.0
    >>> print(f"Ice slope parameter prefactor: {rid.LBI:.2e} m^-1")
    Ice slope parameter prefactor: ...
    """

    cst: Constants
    parami: IceParameters

    CEXVT: cython.double  # Air density fall speed correction

    RTMIN: object  # numpy array
    # Min values allowed for mixing ratios

    # Cloud droplet charact.
    AC: cython.double
    BC: cython.double
    CC: cython.double
    DC: cython.double

    # Rain drop charact
    AR: cython.double
    BR: cython.double
    CR: cython.double
    DR: cython.double
    CCR: cython.double
    F0R: cython.double
    F1R: cython.double
    C1R: cython.double

    # ar, br -> mass - diameter power law
    # cr, dr -> terminal speed velocity - diameter powerlaw
    # f0, f1, f2 -> ventilation coefficients
    # C1 ?

    # Cloud ice charact
    AI: cython.double
    BI: cython.double
    C_I: cython.double
    DI: cython.double
    F0I: cython.double
    F2I: cython.double
    C1I: cython.double

    # Snow/agg charact.
    A_S: cython.double
    BS: cython.double
    CS: cython.double
    DS: cython.double
    CCS: cython.double
    CXS: cython.double
    F0S: cython.double
    F1S: cython.double
    C1S: cython.double

    # Graupel charact.
    AG: cython.double
    BG: cython.double
    CG: cython.double
    DG: cython.double
    CCG: cython.double
    CXG: cython.double
    F0G: cython.double
    F1G: cython.double
    C1G: cython.double

    # Cloud droplet distribution parameters

    # Over land
    ALPHAC: cython.double  # Gamma law of the Cloud droplet (here volume-like distribution)
    NUC: cython.double  # Gamma law with little dispersion

    # Over sea
    ALPHAC2: cython.double
    NUC2: cython.double

    LBEXC: cython.double
    LBC_1: cython.double
    LBC_2: cython.double

    # Rain drop distribution parameters
    ALPHAR: cython.double  # Gamma law of the Cloud droplet (here volume-like distribution)
    NUR: cython.double  # Gamma law with little dispersion
    LBEXR: cython.double
    LBR: cython.double

    # Cloud ice distribution parameters
    ALPHAI: cython.double  # Exponential law
    NUI: cython.double  # Exponential law
    LBEXI: cython.double
    LBI: cython.double

    # Snow/agg. distribution parameters
    ALPHAS: cython.double
    NUS: cython.double
    LBEXS: cython.double
    LBS: cython.double
    NS: cython.double

    # Graupel distribution parameters
    ALPHAG: cython.double
    NUG: cython.double
    LBEXG: cython.double
    LBG: cython.double

    FVELOS: cython.double  # factor for snow fall speed after Thompson
    TRANS_MP_GAMMAS: cython.double  # coefficient to convert lambda for gamma functions
    LBDAR_MAX: cython.double  # Max values allowed for the shape parameters (rain,snow,graupeln)
    LBDAS_MAX: cython.double
    LBDAG_MAX: cython.double
    LBDAS_MIN: cython.double

    V_RTMIN: cython.double
    C_RTMIN: cython.double
    R_RTMIN: cython.double
    I_RTMIN: cython.double
    S_RTMIN: cython.double
    G_RTMIN: cython.double

    CONC_SEA: cython.double  # Diagnostic concentration of droplets over sea
    CONC_LAND: cython.double  # Diagnostic concentration of droplets over land
    CONC_URBAN: cython.double  # Diagnostic concentration of droplets over urban area

    # Statistical sedimentation
    GAC: cython.double
    GC: cython.double
    GAC2: cython.double
    GC2: cython.double

    RAYDEF0: cython.double

    def __init__(self, cst: Constants, parami: IceParameters):
        """Initialize RainIceDescriptors with default values based on constants and parameters."""
        self.cst = cst
        self.parami = parami

        self.CEXVT = 0.4

        self.RTMIN = np.zeros(41)

        # Cloud droplet charact.
        self.AC = 524.0
        self.BC = 3.0
        self.CC = 842.0
        self.DC = 2.0

        # Rain drop charact
        self.AR = 524.0
        self.BR = 3.0
        self.CR = 842.0
        self.DR = 0.8
        self.CCR = 8e-6
        self.F0R = 1.0
        self.F1R = 0.26
        self.C1R = 0.5

        # Cloud ice charact - will be set in __post_init__
        self.AI = 0.0
        self.BI = 0.0
        self.C_I = 0.0
        self.DI = 0.0
        self.F0I = 1.00
        self.F2I = 0.14
        self.C1I = 0.0

        # Snow/agg charact.
        self.A_S = 0.02
        self.BS = 1.9
        self.CS = 5.1
        self.DS = 0.27
        self.CCS = 5.0
        self.CXS = 1.0
        self.F0S = 0.86
        self.F1S = 0.28
        self.C1S = 0.0

        # Graupel charact.
        self.AG = 19.6
        self.BG = 2.8
        self.CG = 124.0
        self.DG = 0.66
        self.CCG = 5e5
        self.CXG = -0.5
        self.F0G = 0.86
        self.F1G = 0.28
        self.C1G = 1.0 / 2.0

        # Cloud droplet distribution parameters
        self.ALPHAC = 1.0
        self.NUC = 3.0

        # Over sea
        self.ALPHAC2 = 1.0
        self.NUC2 = 1.0

        # Will be computed in __post_init__
        self.LBEXC = 0.0
        self.LBC_1 = 0.0
        self.LBC_2 = 0.0

        # Rain drop distribution parameters
        self.ALPHAR = 3.0
        self.NUR = 1.0
        self.LBEXR = 0.0
        self.LBR = 0.0

        # Cloud ice distribution parameters
        self.ALPHAI = 1.0
        self.NUI = 1.0
        self.LBEXI = 0.0
        self.LBI = 0.0

        # Snow/agg. distribution parameters
        self.ALPHAS = 1.0
        self.NUS = 1.0
        self.LBEXS = 0.0
        self.LBS = 0.0
        self.NS = 0.0

        # Graupel distribution parameters
        self.ALPHAG = 1.0
        self.NUG = 1.0
        self.LBEXG = 0.0
        self.LBG = 0.0

        self.FVELOS = 0.097
        self.TRANS_MP_GAMMAS = 1.0
        self.LBDAR_MAX = 1e5
        self.LBDAS_MAX = 1e5
        self.LBDAG_MAX = 1e5
        self.LBDAS_MIN = 1e-10

        self.V_RTMIN = 1e-20
        self.C_RTMIN = 1e-20
        self.R_RTMIN = 1e-20
        self.I_RTMIN = 1e-20
        self.S_RTMIN = 1e-15
        self.G_RTMIN = 1e-15

        self.CONC_SEA = 1e8
        self.CONC_LAND = 3e8
        self.CONC_URBAN = 5e8

        # Statistical sedimentation - will be computed
        self.GAC = 0.0
        self.GC = 0.0
        self.GAC2 = 0.0
        self.GC2 = 0.0

        self.RAYDEF0 = 0.0

        # Call post_init to compute derived values
        self.__post_init__()

    def __post_init__(self):
        # 2.2    Ice crystal characteristics
        if self.parami.PRISTINE_ICE == "PLAT":
            self.AI = 0.82
            self.BI = 2.5
            self.C_I = 800.0
            self.DI = 1.0
            self.C1I = 1.0 / self.cst.PI

        elif self.parami.PRISTINE_ICE == "COLU":
            self.AI = 2.14e-3
            self.BI = 1.7
            self.C_I = 2.1e5
            self.DI = 1.585
            self.C1I = 0.8

        elif self.parami.PRISTINE_ICE == "BURO":
            self.AI = 44.0
            self.BI = 3.0
            self.C_I = 4.3e5
            self.DI = 1.663
            self.C1I = 0.5

        if self.parami.LSNOW_T == 1:
            self.CS = 5.1
            self.DS = 0.27
            self.FVELOS = 25.14

            self.ALPHAS = 0.214
            self.NUS = 43.7
            self.TRANS_MP_GAMMAS = sqrt(
                (gamma(self.NUS + 2 / self.ALPHAS) * gamma(self.NUS + 4 / self.ALPHAS))
                / (
                    8
                    * gamma(self.NUS + 1 / self.ALPHAS)
                    * gamma(self.NUS + 3 / self.ALPHAS)
                )
            )

        self.C1S = 1 / self.cst.PI

        self.LBEXC = 1 / self.BC
        self.LBEXR = 1 / (-1 - self.BR)
        self.LBEXI = 1 / -self.BI
        self.LBEXS = 1 / (self.CXS - self.BS)
        self.LBEXG = 1 / (self.CXG - self.BG)

        # 3.4 Constant for shape parameter
        momg = lambda alpha, nu, p: gamma(nu + p / alpha) / gamma(nu)

        gamc = momg(self.ALPHAC, self.NUC, 3)
        gamc2 = momg(self.ALPHAC2, self.NUC2, 3)
        self.LBC_1, self.LBC_2 = (self.AR * gamc, self.AR * gamc2)

        self.LBR = (self.AR * self.CCR * momg(self.ALPHAR, self.NUR, self.BR)) ** (
            -self.LBEXR
        )
        self.LBI = (self.AI * self.C_I * momg(self.ALPHAI, self.NUI, self.BI)) ** (
            -self.LBEXI
        )
        self.LBS = (self.A_S * self.CCS * momg(self.ALPHAS, self.NUS, self.BS)) ** (
            -self.LBEXS
        )
        self.LBG = (self.AG * self.CCG * momg(self.ALPHAG, self.NUG, self.BG)) ** (
            -self.LBEXG
        )

        self.NS = 1.0 / (self.A_S * momg(self.ALPHAS, self.NUS, self.BS))

        self.GAC = gamma(self.NUC + 1 / self.ALPHAC)
        self.GC = gamma(self.NUC)
        self.GAC2 = gamma(self.NUC2 + 1 / self.ALPHAC2)
        self.GC2 = gamma(self.NUC2)
        self.RAYDEF0 = max(1, 0.5 * (self.GAC / self.GC))


@cython.cclass
class CloudPar:
    """Declaration of the model-n dependant Microphysic constants

    Args:
        nsplitr (int): Number of required small time step integration
            for rain sedimentation computation
        nsplitg (int): Number of required small time step integration
            for ice hydrometeor sedimentation computation

    """

    NSPLITR: cython.int
    NSPLITG: cython.int

    def __init__(self, NSPLITR: int, NSPLITG: int):
        """Initialize CloudPar with split numbers."""
        self.NSPLITR = NSPLITR
        self.NSPLITG = NSPLITG
