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

    hprogram: Literal["AROME", "MESO-NH", "LMDZ"]

    tminmix: float = field(default=273.16)  
    tmaxmix: float = field(default=253.16)  
    hgt_qs: float = field(default=False)  
    frac_ice_adjust: FracIceAdjust = field(default="S") 
    frac_ice_shallow: str = field(default="S")  
    vsigqsat: float = field(default=0.02)  
    condens: str = field(default="CB02")  
    lambda3: str = field(default="CB")  
    statnw: bool = field(default=False)  
    sigmas: bool = field(default=True)  
    subg_cond: bool = field(default=False)  

    def __post_init__(self):
        if self.hprogram == "AROME":
            self.frac_ice_adjust = FracIceAdjust.S.value
            self.frac_ice_shallow = FracIceShallow.T.value
            self.vsigqsat = 0
            self.sigmas = False

        elif self.hprogram == "LMDZ":
            self.subg_cond = True
