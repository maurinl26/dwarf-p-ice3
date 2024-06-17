# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from f90nml import read, Namelist


@dataclass
class Namparar:
    nml_file_path: str

    cfrac_ice_adjust: str = field(init=False)
    cfrac_ice_shallow: str = field(init=False)
    cmicro: str = field(init=False)
    csedim: str = field(init=False)
    csnowriming: bool = field(init=False)
    lcrflimit: bool = field(init=False)
    lcriauti: bool = field(init=False)
    levlimit: bool = field(init=False)
    lfeedbackt: bool = field(init=False)
    lfprec3d: bool = field(init=False)
    lnullwetg: bool = field(init=False)
    lnullweth: bool = field(init=False)
    lolsmc: bool = field(init=False)
    losigmas: bool = field(init=False)
    losedic: bool = field(init=False)
    losubg_cond: bool = field(init=False)
    lsedim_after: bool = field(init=False)
    lwetgpost: bool = field(init=False)
    lwethpost: bool = field(init=False)
    nmaxiter_micro: int = field(init=False)
    nprintfr: int = field(init=False)
    nptp: int = field(init=False)
    rcriautc: float = field(init=False)
    rcrauti: float = field(init=False)
    rt0criauti: float = field(init=False)
    vsigqsat: float = field(init=False)
    xfracm90: float = field(init=False)
    xmrstep: float = field(init=False)
    xsplit_maxcfl: float = field(init=False)
    xstep_ts: float = field(init=False)

    def __post_init__(self):
        """Read namelist file and allocate attributes values"""

        with open(self.nml_file_path) as nml_file:
            nml = read(nml_file)
            nml_namparar = nml.get("NAMPARAR")

            self.allocate_namelist_values(nml_namparar)

    def allocate_namelist_values(self, nml_namparar: Namelist):
        """Allocate values of dataclass attributes with namelist fields
        Args:
            nml_namparar (_type_): namelist &NAMPARAR
        """

        for field in self.__dataclass_fields__:
            setattr(self, field, nml_namparar.get(field))
