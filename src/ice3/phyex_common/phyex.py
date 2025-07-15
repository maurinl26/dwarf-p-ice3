# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass, field
from typing import Literal, Tuple
from enum import Enum

from ice3_gt4py.phyex_common.constants import Constants
from ice3_gt4py.phyex_common.nebn import Neb
from ice3_gt4py.phyex_common.rain_ice_param import ParamIce, RainIceDescr, RainIceParam


class Boundary(Enum):
    PRESCRIBED = 0
    CYCL = 1


@dataclass
class Phyex:
    """Data class for physics parametrizations

    Args:
        program (Literal): Switch between Meso-NH and AROME
        timestep (float): timestep for physical parametrizations

        cst (Constants): Physical constants description
        param_icen (ParamIce): Control parameters for microphysics
        rain_ice_descrn (RainIceDescr): Microphysical descriptive constants

        tstep (float): time step employed for physics
        itermax (int): number of iterations for ice adjust

        lmfconv (bool): use convective mass flux in the condensation scheme
        compute_src (bool): compute s'r'
        khalo (int): size of the halo for parallel distribution (in turb)
        program (str): Name of the model
        nomixlg (bool): turbulence for lagrangian variables
        ocean (bool): ocean version of the turbulent scheme
        couples (bool): ocean atmo LES interactive coupling
        blowsnow (bool): blowsnow
        rsnow (float): blowing factor
        lbcx (Tuple[str]): boundary conditions
        lbcy (Tuple[str]): boundary conditions
        ibm (bool): run with ibm$
        flyer (bool): MesoNH flyer diagnostic
        diag_in_run (bool): LES diagnostics
        o2d (bool): 2D version of turbulence
    """

    program: Literal["AROME", "MESO-NH"]
    timestep: float = field(default=1)

    cst: Constants = field(init=False)
    param_icen: ParamIce = field(init=False)
    rain_ice_descrn: RainIceDescr = field(init=False)
    rain_ice_param: RainIceParam = field(init=False)
    nebn: Neb = field(init=False)

    ITERMAX: int = field(default=1)
    TSTEP: float = field(default=45)
    INV_TSTEP: float = field(init=False)
    NRR: float = field(default=6)

    # Miscellaneous terms
    LMFCONV: bool = field(default=True)
    COMPUTE_SRC: bool = field(default=True)
    KHALO: int = field(default=1)
    PROGRAM: str = field(default="AROME")
    NOMIXLG: bool = field(default=False)
    OCEAN: bool = field(default=False)
    DEEPOC: bool = field(default=False)
    COUPLES: bool = field(default=False)
    BLOWSNOW: bool = field(default=False)
    RSNOW: float = field(default=1.0)
    LBCX: Tuple[int] = field(default=(Boundary.CYCL.value, Boundary.CYCL.value))
    LBCY: Tuple[int] = field(default=(Boundary.CYCL.value, Boundary.CYCL.value))
    IBM: bool = field(default=False)
    FLYER: bool = field(default=False)
    DIAG_IN_RUN: bool = field(default=False)
    O2D: bool = field(default=False)

    # flat: bool
    # tbuconf: TBudgetConf

    def __post_init__(self):
        self.cst = Constants()
        self.param_icen = ParamIce(self.PROGRAM)
        self.nebn = Neb(self.PROGRAM)
        self.rain_ice_descrn = RainIceDescr(self.cst, self.param_icen)
        self.rain_ice_param = RainIceParam(
            self.cst, self.rain_ice_descrn, self.param_icen
        )
        
        self.INV_TSTEP = 1 / self.TSTEP

    def to_externals(self):
        externals = {}
        externals.update(asdict(self.cst))
        externals.update(asdict(self.param_icen))
        externals.update(asdict(self.rain_ice_descrn))
        externals.update(asdict(self.rain_ice_param))
        externals.update(asdict(self.nebn))
        externals.update({"TSTEP": self.TSTEP, "NRR": self.NRR, "INV_TSTEP": self.INV_TSTEP})

        return externals
