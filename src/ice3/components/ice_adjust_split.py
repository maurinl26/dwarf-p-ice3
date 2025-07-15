# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass

import dace
import numpy as np

from dace.dtypes import float64, compiletime
from ice3_gt4py.stencils.condensation_split import condensation
from ice3_gt4py.stencils.sigma_rc_dace import sigrc_computation
from ice3_gt4py.stencils.cloud_fraction_split import thermodynamic_fields, cloud_fraction_1, cloud_fraction_2
from ice3_gt4py.phyex_common.tables import SRC_1D

from ice3_gt4py.utils.typingx import dtype_float, dtype_int, FloatFieldIJK
from ice3_gt4py.utils.dims import I, J, K

class IceAdjustState:
    RD: float
    RV: float

    FRAC_ICE_ADJUST: int
    TMAXMIX: float
    TMINMIX: float

    OCND2: bool
    LSIGMAS: bool
    LSTATNW: bool
    LSUBG_COND: bool

    CONDENS: int
    NRR: int

    CPV: float
    CPD: float
    CL: float
    CI: float

    SUBG_MF_PDF: int
    CRIAUTC: float
    CRIAUTI: float
    ACRIAUTI: float
    BCRIAUTI: float
    TT: float
    LAMBDA3: bool

    def __init__(self, phyext: dict):
        self.RD = phyext["RD"]
        self.RV = phyext["RV"]

        self.FRAC_ICE_ADJUST = phyext["FRAC_ICE_ADJUST"]
        self.TMAXMIX = phyext["TMAXMIX"]
        self.TMINMIX = phyext["TMINMIX"]

        self.OCND2 = phyext["OCND2"]
        self.LSIGMAS = phyext["LSIGMAS"]
        self.LSTATNW = phyext["LSTATNW"]
        self.LSUBG_COND = phyext["LSUBG_COND"]

        self.CONDENS = phyext["CONDENS"]
        self.NRR = phyext["NRR"]

        self.CPV = phyext["CPV"]
        self.CPD = phyext["CPD"]
        self.CL = phyext["CL"]
        self.CI = phyext["CI"]

        self.SUBG_MF_PDF = phyext["SUBG_MF_PDF"]
        self.CRIAUTC = phyext["CRIAUTC"]
        self.CRIAUTI = phyext["CRIAUTI"]
        self.ACRIAUTI = phyext["ACRIAUTI"]
        self.BCRIAUTI = phyext["BCRIAUTI"]
        self.TT = phyext["TT"]
        self.LAMBDA3 = phyext["LAMBDA3"]


@dace.program
def ice_adjust(
    th: float64[I, J, K],
    exn: float64[I, J, K],
    rv: float64[I, J, K],
    rc: float64[I, J, K],
    rr: float64[I, J, K],
    ri: float64[I, J, K],
    rs: float64[I, J, K],
    rg: float64[I, J, K],
    NRR: compiletime,
    CPD: compiletime,
    CPV: compiletime,
    CL: compiletime,
    CI: compiletime,
):

    cph = np.ndarray([I, J, K], dtype=dtype_float)
    lv = np.ndarray([I, J, K], dtype=dtype_float)
    ls = np.ndarray([I, J, K], dtype=dtype_float)
    t = np.ndarray([I, J, K], dtype=dtype_float)

    thermodynamic_fields(
        th[:,:,:],
        exn[:,:,:],
        rv[:,:,:],
        rc[:,:,:],
        rr[:,:,:],
        ri[:,:,:],
        rs[:,:,:],
        rg[:,:,:],
        cph[:,:,:],
        lv[:,:,:],
        ls[:,:,:],
        t[:,:,:],
        NRR=NRR,
        CPD=CPD,
        CPV=CPV,
        CL=CL,
        CI=CI,
    )
