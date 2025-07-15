# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from math import gamma, log
from numpy.typing import NDArray
import logging

import numpy as np

from ice3_gt4py.phyex_common.constants import Constants
from ice3_gt4py.phyex_common.gamma_inc import generalized_incomplete_gamma
from ice3_gt4py.phyex_common.param_ice import ParamIce
from ice3_gt4py.phyex_common.rain_ice_descr import RainIceDescr


@dataclass
class RainIceParam:
    """Constants for RainIce parametrization and source computation

    # Constants dependencies
    cst: Constants
    rid: RainIceDescr
    parami: ParamIce

    # Parameters for microphysical sources and transformations
    fsedc: Tuple[float]     # Constants for sedimentation fluxes of C
    fsedr: float            # Constants for sedimentation
    exsedr: float
    fsedi: float
    excsedi: float
    exrsedi: float
    fseDS: float
    exseDS: float
    fsedg: float
    exsedg: float

    # Constants for heterogeneous ice nucleation HEN
    nu10: float
    alpha1: float = 4.5
    beta1: float = 0.6
    nu20: float
    alpha2: float = 12.96
    beta2: float = 0.639
    mnu0: float = 6.88e-13  # Mass of nucleated ice crystal

    # Constants for homogeneous ice nucleation HON
    alpha3: float = -3.075
    beta3: float = 81.00356
    hon: float

    # Constants for raindrop and evaporation EVA
    scfac: float
    o0evar: float
    o1evar: float
    ex0evar: float
    ex1evar: float
    o0dePI: float           # deposition DEP on Ice
    o2dePI: float
    o0deps: float           # on Snow
    o1deps: float
    ex0deps: float
    ex1deps: float
    rdepsred: float
    o0depg: float           # on Graupel
    o1depg: float
    ex0depg: float
    ex1depg: float
    rdepgred: float

    # Constants for pristine ice autoconversion : AUT
    timauti: float = 1e-3   # Time constant at T=T_t
    texauti: float = 0.015
    criauti: float
    t0criauti: float
    acriauti: float
    bcriauti: float

    # Constants for snow aggregation : AGG
    colis: float = 0.25     # Collection efficiency of I + S
    colexis: float = 0.05   # Temperature factor of the I+S collection efficiency
    fiaggs: float
    exiaggs: float

    # Constants for cloud droplet autoconversion AUT
    timautc: float = 1e-3
    criautc: float

    # Constants for cloud droplets accretion on raindrops : ACC
    fcaccr: float
    excaccr: float

    # Constants for the riming of the aggregates : RIM
    dcslim: float = 0.007
    colcs: float = 1.0
    excrimss: float
    crimss: float
    excrimsg: float
    crimsg: float

    excrimsg: float
    crimsg: float
    exsrimcg: float
    exsrimcg2: float
    srimcg3: float

    gaminc_bound_min: float
    gaminc_bound_max: float
    rimintp1: float
    rimintp2: float

    ngaminc: int   # Number of tab. Lbda_s

    """

    # Constants dependencies
    cst: Constants
    rid: RainIceDescr
    parami: ParamIce

    # Parameters for microphysical sources and transformations
    FSEDC_1: float = field(init=False)  # Constants for sedimentation fluxes of C
    FSEDC_2: float = field(init=False)
    FSEDR: float = field(init=False)  # Constants for sedimentation
    EXSEDR: float = field(init=False)
    FSEDI: float = field(init=False)
    EXCSEDI: float = field(init=False)
    EXRSEDI: float = field(init=False)
    FSEDS: float = field(init=False)
    EXSEDS: float = field(init=False)
    FSEDG: float = field(init=False)
    EXSEDG: float = field(init=False)

    # Constants for heterogeneous ice nucleation HEN
    NU10: float = field(init=False)
    ALPHA1: float = 4.5
    BETA1: float = 0.6
    NU20: float = field(init=False)
    ALPHA2: float = 12.96
    BETA2: float = 0.639
    MNU0: float = 6.88e-13  # Mass of nucleated ice crystal

    # Constants for homogeneous ice nucleation HON
    ALPHA3: float = -3.075
    BETA3: float = 81.00356
    HON: float = field(init=False)

    # Constants for raindrop and evaporation EVA
    SCFAC: float = field(init=False)
    O0EVAR: float = field(init=False)
    O1EVAR: float = field(init=False)
    EX0EVAR: float = field(init=False)
    EX1EVAR: float = field(init=False)
    O0DEPI: float = field(init=False)  # deposition DEP on I
    O2DEPI: float = field(init=False)
    O0DEPS: float = field(init=False)  # on S
    O1DEPS: float = field(init=False)
    EX0DEPS: float = field(init=False)
    EX1DEPS: float = field(init=False)
    RDEPSRED: float = field(init=False)
    O0DEPG: float = field(init=False)  # on G
    O1DEPG: float = field(init=False)
    EX0DEPG: float = field(init=False)
    EX1DEPG: float = field(init=False)
    RDEPGRED: float = field(init=False)

    # Constants for pristine ice autoconversion : AUT
    TIMAUTI: float = 1e-3  # Time constant at T=T_t
    TEXAUTI: float = 0.015
    CRIAUTI: float = field(init=False)
    T0CRIAUTI: float = field(init=False)
    ACRIAUTI: float = field(init=False)
    BCRIAUTI: float = field(init=False)

    # Constants for snow aggregation : AGG
    COLIS: float = 0.25  # Collection efficiency of I + S
    COLEXIS: float = 0.05  # Temperature factor of the I+S collection efficiency
    FIAGGS: float = field(init=False)
    EXIAGGS: float = field(init=False)

    # Constants for cloud droplet autoconversion AUT
    TIMAUTC: float = 1e-3
    CRIAUTC: float = field(init=False)

    # Constants for cloud droplets accretion on raindrops : ACC
    FCACCR: float = field(init=False)
    EXCACCR: float = field(init=False)

    # Constants for the riming of the aggregates : RIM
    DCSLIM: float = 0.007
    COLCS: float = 1.0
    EXCRIMSS: float = field(init=False)
    CRIMSS: float = field(init=False)
    EXCRIMSG: float = field(init=False)
    CRIMSG: float = field(init=False)

    EXCRIMSG: float = field(init=False)
    CRIMSG: float = field(init=False)
    EXSRIMCG: float = field(init=False)
    EXSRIMCG2: float = field(init=False)
    SRIMCG: float = field(init=False)
    SRIMCG2: float = field(init=False)
    SRIMCG3: float = field(init=False)

    GAMINC_BOUND_MIN: float = field(init=False)
    GAMINC_BOUND_MAX: float = field(init=False)

    RIMINTP1: float = field(init=False)
    RIMINTP2: float = field(init=False)

    NGAMINC: int = field(init=False)  # Number of tab. Lbda_s

    # GAMINC_RIM1: NDArray = field(init=False)
    # GAMINC_RIM2: NDArray = field(init=False)
    # GAMINC_RIM4: NDArray = field(init=False)

    # Constants for the accretion
    FRACCSS: float = field(init=False)
    LBRACCS1: float = field(init=False)
    LBRACCS2: float = field(init=False)
    LBRACCS3: float = field(init=False)
    FSACCRG: float = field(init=False)
    LBSACCR1: float = field(init=False)
    LBSACCR2: float = field(init=False)
    LBSACCR3: float = field(init=False)
    ACCLBDAS_MIN: float = field(default=5e1)
    ACCLBDAS_MAX: float = field(default=5e5)
    ACCLBDAR_MIN: float = field(default=1e3)
    ACCLBDAR_MAX: float = field(default=1e7)
    ACCINTP1S: float = field(init=False)
    ACCINTP2S: float = field(init=False)
    ACCINTP1R: float = field(init=False)
    ACCINTP2R: float = field(init=False)

    # number of values in global tables (per axis)
    NACCLBDAS: int = field(default=40)
    NACCLBDAR: int = field(default=40)

    # 7.3 Constant for the conversion-melting rate
    FSCVMG: float = field(default=2)

    # Constants for rain contact freezing
    COLIR: float = field(default=1.0)
    EXRCFRI: float = field(init=False)
    RCFRI: float = field(init=False)
    EXICFRR: float = field(init=False)
    ICFRR: float = field(init=False)

    # Constants for the dry growth of the graupel : DRY
    FCDRYG: float = field(init=False)
    COLIG: float = field(default=0.01)
    COLEXIG: float = field(default=0.1)
    FIDRYG: float = field(init=False)
    FIDRYG2: float = field(init=False)
    EXFIDRYG: float = field(init=False)
    COLSG: float = field(default=0.01)
    COLEXSG: float = field(default=0.1)
    FSDRYG: float = field(init=False)
    LBSDRYG1: float = field(init=False)
    LBSDRYG2: float = field(init=False)
    LBSDRYG3: float = field(init=False)
    FRDRYG: float = field(init=False)
    LBRDRYG1: float = field(init=False)
    LBRDRYG2: float = field(init=False)
    LBRDRYG3: float = field(init=False)
    DRYLBDAR_MIN: float = field(default=1e3)
    DRYLBDAR_MAX: float = field(default=1e7)
    DRYLBDAS_MIN: float = field(default=2.5e1)
    DRYLBDAS_MAX: float = field(default=2.5e9)
    DRYLBDAG_MIN: float = field(default=1e3)
    DRYLBDAG_MAX: float = field(default=1e7)
    DRYINTP1R: float = field(init=False)
    DRYINTP2R: float = field(init=False)
    DRYINTP1S: float = field(init=False)
    DRYINTP2S: float = field(init=False)
    DRYINTP1G: float = field(init=False)
    DRYINTP2G: float = field(init=False)

    NDRYLBDAR: int = field(default=40)
    NDRYLBDAS: int = field(default=80)
    NDRYLBDAG: int = field(default=40)

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
        if self.parami.LCRIAUTI:
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
        # self.init_gaminc_rim_tables()

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
        # self.ker_saccrg = self.get_kernel("saccrg")
        # self.ker_raccs = self.get_kernel("raccs")
        # self.ker_raccss = self.get_kernel("raccss")
        # self.ker_rdryg = self.get_kernel("rdryg")
        # self.ker_sdryg = self.get_kernel("sdryg")


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
            from ice3_gt4py.phyex_common.xker_raccs import KER_SACCRG

            return KER_SACCRG[1:, 1:]

        elif kernel == "raccs":
            from ice3_gt4py.phyex_common.xker_raccs import KER_RACCS

            return KER_RACCS[1:, 1:]

        elif kernel == "raccss":
            from ice3_gt4py.phyex_common.xker_raccs import KER_RACCSS

            return KER_RACCSS[1:, 1:]

        elif kernel == "rdryg":
            from ice3_gt4py.phyex_common.xker_rdryg import KER_RDRYG

            return KER_RDRYG[1:, 1:]

        elif kernel == "sdryg":
            from ice3_gt4py.phyex_common.xker_sdryg import KER_SDRYG

            return KER_SDRYG[1:, 1:]

        else:
            raise KeyError(f"{kernel} not found in GlobalTables")


@dataclass
class CloudPar:
    """Declaration of the model-n dependant Microphysic constants

    Args:
        nsplitr (int): Number of required small time step integration
            for rain sedimentation computation
        nsplitg (int): Number of required small time step integration
            for ice hydrometeor sedimentation computation

    """

    nsplitr: int
    nsplitg: int
