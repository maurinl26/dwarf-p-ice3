# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from ifs_physics_common.utils.f2py import ported_class


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


@ported_class(from_file="PHYEX/src/common/aux/modd_nebn.F90")
@dataclass
class Neb:
    """Declaration of

    Args:
        tminmix (float): minimum temperature for mixed phase
        tmaxmix (float): maximum temperature for mixed phase
        hgt_qs (float): switch for height dependant VQSIGSAT
        frac_ice_adjust (str): ice fraction for adjustments
        frac_ice_shallow (str): ice fraction for shallow_mf
        vsigqsat (float): coeff applied to qsat variance contribution
        condens (str): subgrid condensation PDF
        lambda3 (str): lambda3 choice for subgrid cloud scheme
        statnw (bool): updated full statistical cloud scheme
        sigmas (bool): switch for using sigma_s from turbulence scheme
        subg_cond (bool): switch for subgrid condensation

    """

    HPROGRAM: Literal["AROME", "MESO-NH", "LMDZ"]

    TMINMIX: float = field(default=273.16)
    TMAXMIX: float = field(default=253.16)
    HGT_QS: float = field(default=False)
    FRAC_ICE_ADJUST: FracIceAdjust = field(default="S")
    FRAC_ICE_SHALLOW: str = field(default="S")
    VSIGQSAT: float = field(default=0.02)
    CONDENS: str = field(default="CB02")
    LAMBDA3: str = field(default="CB")
    STATNW: bool = field(default=False)
    SIGMAS: bool = field(default=True)
    SUBG_COND: bool = field(default=False)

    def __post_init__(self):
        if self.HPROGRAM == "AROME":
            self.FRAC_ICE_ADJUST = FracIceAdjust.S.value
            self.FRAC_ICE_SHALLOW = FracIceShallow.T.value
            self.VSIGQSAT = 0
            self.SIGMAS = False
            self.SUBG_COND = True

        elif self.HPROGRAM == "LMDZ":
            self.SUBG_COND = True
