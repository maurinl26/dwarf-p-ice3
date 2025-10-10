# -*- coding: utf-8 -*-
from dataclasses import dataclass, field, InitVar
import cython
from enum import Enum
from typing import Literal

class FracIceAdjust(Enum):
    """Enumeration for ice fraction adjustments modes

    T in case of AROME

    """
    T = 0
    O = 1
    N = 2
    S = 3


class FracIceShallow(Enum):
    """Enumeration of ice fraction for shallow mass fluxes

    T in case of AROME
    """
    T = 0
    S = 1


class Condens(Enum):
    """Enumeration for condensation variance

    HCONDENS in .F90
    CB02 for AROME
    """
    CB02 = 0
    GAUS = 1


class Lambda3(Enum):
    """LAMBDA3 in AROME

    CB by default in AROME
    """
    CB = 0


########### PHYEX/src/common/aux/modd_nebn.F90 #############
@dataclass
@cython.cclass
class Neb:
    """Declaration of

    Args:
        tminmix (InitVar[float]): minimum temperature for mixed phase
        tmaxmix (InitVar[float]): maximum temperature for mixed phase
        hgt_qs (InitVar[float]): switch for height dependant VQSIGSAT
        frac_ice_adjust (str): ice fraction for adjustments
        frac_ice_shallow (str): ice fraction for shallow_mf
        vsigqsat (InitVar[float]): coeff applied to qsat variance contribution
        condens (str): subgrid condensation PDF
        lambda3 (str): lambda3 choice for subgrid cloud scheme
        statnw (InitVar[bool]): updated full statistical cloud scheme
        sigmas (InitVar[bool]): switch for using sigma_s from turbulence scheme
        subg_cond (InitVar[bool]): switch for subgrid condensation

    """

    HPROGRAM: Literal["AROME", "MESO-NH", "LMDZ"]

    TMINMIX: InitVar[float] = 273.16
    TMAXMIX: InitVar[float] = 253.16
    LHGT_QS: InitVar[bool] = False
    FRAC_ICE_ADJUST: InitVar[int] = FracIceAdjust.S.value
    FRAC_ICE_SHALLOW: InitVar[int] = FracIceShallow.S.value
    VSIGQSAT: InitVar[float] = 0.02
    CONDENS: InitVar[int] = Condens.CB02.value
    LAMBDA3: InitVar[int] = Lambda3.CB.value
    LSTATNW: InitVar[bool] = False
    LSIGMAS: InitVar[bool] = True
    LSUBG_COND: InitVar[bool] = False

    def __post_init__(self):
        if self.HPROGRAM == "AROME":
            self.FRAC_ICE_ADJUST = FracIceAdjust.T.value
            self.FRAC_ICE_SHALLOW = FracIceShallow.T.value
            self.VSIGQSAT = 0.02
            self.LSIGMAS = True
            self.LSUBG_COND = True

        elif self.HPROGRAM == "LMDZ":
            self.LSUBG_COND = True
