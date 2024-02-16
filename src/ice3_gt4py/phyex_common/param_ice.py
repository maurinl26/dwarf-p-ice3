# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from ifs_physics_common.utils.f2py import ported_class
from enum import Enum

from ice3_gt4py.drivers.namel2config import Namparar


class SubGridMassFluxPDF(Enum):

    NONE = 0
    TRIANGLE = 1


@ported_class(from_file="PHYEX/src/common/aux/modd_param_icen.F90")
@dataclass
class ParamIce:
    """

    hprogram: Literal["AROME", "MESO-NH", "LMDZ"]

    lwarm: bool             # Formation of rain by warm processes
    lsedic: bool            # Enable the droplets sedimentation
    ldeposc: bool           # Enable cloud droplets deposition

    vdeposc: float          # Droplet deposition velocity

    pristine_ice:           # Pristine ice type PLAT, COLU, or BURO
    sedim: str              # Sedimentation calculation mode

    lred: bool              # To use modified ice3/ice4 - to reduce time step dependency

    subg_rc_rr_accr: str    # subgrid rc-rr accretion
    subg_rr_evap: str       # subgrid rr evaporation
    subg_rr_pdf: str        # pdf for subgrid precipitation
    subg_aucv_rc: str       # type of subgrid rc->rr autoconv. method
    subg_aucv_ri: str       # type of subgrid ri->rs autoconv. method
    subg_mf_pdf: str        # PDF to use for MF cloud autoconversions

    ladj_before: bool       # must we perform an adjustment before rain_ice call
    ladj_after: bool        # must we perform an adjustment after rain_ice call
    lsedim_after: bool      # sedimentation done before (.FALSE.) or after (.TRUE.) microphysics

    split_maxcfl: float     # Maximum CFL number allowed for SPLIT scheme
    lsnow_t: bool           # Snow parameterization from Wurtz (2021)

    lpack_interp: bool
    lpack_micro: bool
    lcriauti: bool

    npromicro: int

    criauti_nam: float = field(default=0.2e-4)
    acriauti_nam: float = field(default=0.06)
    brcriauti_nam: float = field(default=-3.5)
    t0criauti_nam: float = field(init=False)
    criautc_nam: float = field(default=0.5e-3)
    rdepsred_nam: float = field(default=1)
    rdepgred_nam: float = field(default=1)
    lcond2: bool = field(default=False)
    frmin_nam: np.ndarray = field(init=False)
    """

    hprogram: Literal["AROME", "MESO-NH", "LMDZ"]

    lwarm: bool = field(default=True)  # Formation of rain by warm processes
    lsedic: bool = field(default=True)  # Enable the droplets sedimentation
    ldeposc: bool = field(default=False)  # Enable cloud droplets deposition

    vdeposc: float = field(default=0.02)  # Droplet deposition velocity

    pristine_ice: str = field(default="PLAT")  # Pristine ice type PLAT, COLU, or BURO
    sedim: str = field(default="SPLI")  # Sedimentation calculation mode

    # To use modified ice3/ice4 - to reduce time step dependency
    lred: bool = field(default=True)

    subg_rc_rr_accr: str = field(default="NONE")  # subgrid rc-rr accretion
    subg_rr_evap: str = field(default="NONE")  # subgrid rr evaporation
    subg_rr_pdf: str = field(default="SIGM")  # pdf for subgrid precipitation
    subg_aucv_rc: str = field(default="NONE")  # type of subgrid rc->rr autoconv. method
    subg_aucv_ri: str = field(default="NONE")  # type of subgrid ri->rs autoconv. method

    # PDF to use for MF cloud autoconversions
    subg_mf_pdf: int = field(default=SubGridMassFluxPDF.TRIANGLE.value)

    # key for adjustment before rain_ice call
    ladj_before: bool = field(default=True)

    # key for adjustment after rain_ice call
    ladj_after: bool = field(default=True)

    # switch to perform sedimentation
    # before (.FALSE.)
    # or after (.TRUE.) microphysics
    lsedim_after: bool = field(default=False)

    # Maximum CFL number allowed for SPLIT scheme
    split_maxcfl: float = field(default=0.8)

    # Snow parameterization from Wurtz (2021)
    lsnow_t: bool = field(default=False)

    lpack_interp: bool = field(default=True)
    lpack_micro: bool = field(default=True)
    lcriauti: bool = field(default=True)

    npromicro: int = field(default=0)

    criauti_nam: float = field(default=0.2e-4)
    acriauti_nam: float = field(default=0.06)
    brcriauti_nam: float = field(default=-3.5)
    t0criauti_nam: float = field(init=False)
    criautc_nam: float = field(default=0.5e-3)
    rdepsred_nam: float = field(default=1)
    rdepgred_nam: float = field(default=1)
    lcond2: bool = field(default=False)
    frmin_nam: np.ndarray = field(init=False)

    def __post_init__(self):
        self.t0criauti_nam = (np.log10(self.criauti_nam) - self.brcriauti_nam) / 0.06

        self.frmin_nam = np.empty(41)
        self.frmin_nam[1:6] = 0
        self.frmin_nam[7:9] = 1.0
        self.frmin_nam[10] = 10.0
        self.frmin_nam[11] = 1.0
        self.frmin_nam[12] = 0.0
        self.frmin_nam[13] = 1.0e-15
        self.frmin_nam[14] = 120.0
        self.frmin_nam[15] = 1.0e-4
        self.frmin_nam[16:20] = 0.0
        self.frmin_nam[21:22] = 1.0
        self.frmin_nam[23] = 0.5
        self.frmin_nam[24] = 1.5
        self.frmin_nam[25] = 30.0
        self.frmin_nam[26:38] = 0.0
        self.frmin_nam[39] = 0.25
        self.frmin_nam[40] = 0.15

        if self.hprogram == "AROME":
            self.lconvhg = True
            self.ladj_before = True
            self.ladj_after = False
            self.lred = False
            self.sedim = "STAT"
            self.mrstep = 0
            self.subg_aucv_rc = "PDF"

        elif self.hprogram == "LMDZ":
            self.subg_aucv_rc = "PDF"
            self.sedim = "STAT"
            self.nmaxiter_micro = 1
            self.criautc_nam = 0.001
            self.criauti_nam = 0.0002
            self.t0criauti_nam = -5
            self.lred = True
            self.lconvhg = True
            self.ladj_before = True
            self.ladj_after = True
