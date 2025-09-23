# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from math import gamma, sqrt
from typing import List, Tuple

import numpy as np
from ifs_physics_common.utils.f2py import ported_class

from ice3_gt4py.phyex_common.constants import Constants
from ice3_gt4py.phyex_common.param_ice import ParamIce


@ported_class(from_file="PHYEX/src/common/aux/modd_rain_ice_descrn.F90")
@dataclass
class RainIceDescr:
    """Declaration of the microphysical descriptove constants for use in the warm and cold schemes.

    m(D)    = XAx * D ** Bx         : Mass-MaxDim relationship
    v(D)    = XCx * D ** Dx         : Fallspeed-MaxDim relationship
    N(Lbda) = XCCx * Lbda ** CXx    : NumberConc-Slopeparam relationship
    f0x, f1x, f2x                   : Ventilation factors
    c1x                             : Shape parameter for deposition

    and

    alphax, nux                        : Generalized GAMMA law
    Lbda = XLBx * (r_x*rho_dref)**XLBEXx : Slope parameter of the
                                                distribution law


    """

    cst: Constants
    parami: ParamIce

    CEXVT: float = 0.4  # Air density fall speed correction

    RTMIN: np.ndarray = field(default_factory=lambda: np.zeros(41))
    # Min values allowed for mixing ratios

    # Cloud droplet charact.
    AC: float = field(default=524)
    BC: float = field(default=3.0)
    CC: float = field(default=842)
    DC: float = field(default=2)

    # Rain drop charact
    AR: float = field(default=524)
    BR: float = field(default=3.0)
    CR: float = field(default=842)
    DR: float = field(default=0.8)
    CCR: float = field(default=8e-6)
    F0R: float = field(default=1.0)
    F1R: float = field(default=0.26)
    C1R: float = field(default=0.5)

    # ar, br -> mass - diameter power law
    # cr, dr -> terminal speed velocity - diameter powerlaw
    # f0, f1, f2 -> ventilation coefficients
    # C1 ?

    # Cloud ice charact
    AI: float = field(init=False)
    BI: float = field(init=False)
    C_I: float = field(init=False)
    DI: float = field(init=False)
    F0I: float = field(default=1.00)
    F2I: float = field(default=0.14)
    C1I: float = field(init=False)

    # Snow/agg charact.
    A_S: float = field(default=0.02)
    BS: float = field(default=1.9)
    CS: float = field(default=5.1)
    DS: float = field(default=0.27)
    CCS: float = field(default=5.0)  # not lsnow
    CXS: float = field(default=1.0)
    F0S: float = field(default=0.86)
    F1S: float = field(default=0.28)
    C1S: float = field(init=False)

    # Graupel charact.
    AG: float = field(default=19.6)
    BG: float = field(default=2.8)
    CG: float = field(default=124)
    DG: float = field(default=0.66)
    CCG: float = field(default=5e5)
    CXG: float = field(default=-0.5)
    F0G: float = field(default=0.86)
    F1G: float = field(default=0.28)
    C1G: float = field(default=1 / 2)

    # Cloud droplet distribution parameters

    # Over land
    ALPHAC: float = (
        1.0  # Gamma law of the Cloud droplet (here volume-like distribution)
    )
    NUC: float = 3.0  # Gamma law with little dispersion

    # Over sea
    ALPHAC2: float = 1.0
    NUC2: float = 1.0

    LBEXC: float = field(init=False)
    LBC_1: float = field(init=False)
    LBC_2: float = field(init=False)

    # Rain drop distribution parameters
    ALPHAR: float = (
        3.0  # Gamma law of the Cloud droplet (here volume-like distribution)
    )
    NUR: float = 1.0  # Gamma law with little dispersion
    LBEXR: float = field(init=False)
    LBR: float = field(init=False)

    # Cloud ice distribution parameters
    ALPHAI: float = 1.0  # Exponential law
    NUI: float = 1.0  # Exponential law
    LBEXI: float = field(init=False)
    LBI: float = field(init=False)

    # Snow/agg. distribution parameters
    ALPHAS: float = field(default=1.0)
    NUS: float = field(default=1.0)
    LBEXS: float = field(init=False)
    LBS: float = field(init=False)
    NS: float = field(init=False)

    # Graupel distribution parameters
    ALPHAG: float = 1.0
    NUG: float = 1.0
    LBEXG: float = field(init=False)
    LBG: float = field(init=False)

    FVELOS: float = field(default=0.097)  # factor for snow fall speed after Thompson
    TRANS_MP_GAMMAS: float = field(
        default=1
    )  # coefficient to convert lambda for gamma functions
    LBDAR_MAX: float = field(
        default=1e5
    )  # Max values allowed for the shape parameters (rain,snow,graupeln)
    LBDAS_MAX: float = field(default=1e5)
    LBDAG_MAX: float = field(default=1e5)
    LBDAS_MIN: float = field(default=1e-10)

    V_RTMIN: float = field(default=1e-20)
    C_RTMIN: float = field(default=1e-20)
    R_RTMIN: float = field(default=1e-20)
    I_RTMIN: float = field(default=1e-20)
    S_RTMIN: float = field(default=1e-15)
    G_RTMIN: float = field(default=1e-15)

    CONC_SEA: float = 1e8  # Diagnostic concentration of droplets over sea
    CONC_LAND: float = 3e8  # Diagnostic concentration of droplets over land
    CONC_URBAN: float = 5e8  # Diagnostic concentration of droplets over urban area

    # Statistical sedimentation
    GAC: float = field(init=False)
    GC: float = field(init=False)
    GAC2: float = field(init=False)
    GC2: float = field(init=False)

    RAYDEF0: float = field(init=False)

    def __post_init__(self):
        # 2.2    Ice crystal characteristics
        if self.parami.PRISTINE_ICE == "PLAT":
            self.AI = 0.82
            self.BI = 2.5
            self.C_I = 800
            self.DI = 1.0
            self.C1I = 1 / self.cst.PI

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

        if self.parami.LSNOW_T:
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


@dataclass
class CloudPar:
    """Declaration of the model-n dependant Microphysic constants

    Args:
        nsplitr (int): Number of required small time step integration
            for rain sedimentation computation
        nsplitg (int): Number of required small time step integration
            for ice hydrometeor sedimentation computation

    """

    NSPLITR: int
    NSPLITG: int
