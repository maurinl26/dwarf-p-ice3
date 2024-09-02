# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from f90nml import read, Namelist


@dataclass
class Namparar:
    nml_file_path: str

    CFRAC_ICE_ADJUST: str = field(init=False)
    CFRAC_ICE_SHALLOW: str = field(init=False)
    CMICRO: str = field(init=False)
    CSEDIM: str = field(init=False)
    CSNOWRIMING: bool = field(init=False)
    LCRFLIMIT: bool = field(init=False)
    LCRIAUTI: bool = field(init=False)
    LEVLIMIT: bool = field(init=False)
    LFEEDBACKT: bool = field(init=False)
    LFPREC3D: bool = field(init=False)
    LNULLWETG: bool = field(init=False)
    LNULLWETH: bool = field(init=False)
    LOLSMC: bool = field(init=False)
    LOSIGMAS: bool = field(init=False)
    LOSEDIC: bool = field(init=False)
    LOSUBG_COND: bool = field(init=False)
    LSEDIM_AFTER: bool = field(init=False)
    LWETGPOST: bool = field(init=False)
    LWETHPOST: bool = field(init=False)
    NMAXITER_MICRO: int = field(init=False)
    NPRINTFR: int = field(init=False)
    NPTP: int = field(init=False)
    RCRIAUTC: float = field(init=False)
    RCRAUTI: float = field(init=False)
    RT0CRIAUTI: float = field(init=False)
    VSIGQSAT: float = field(init=False)
    XFRACM90: float = field(init=False)
    XMRSTEP: float = field(init=False)
    XSPLIT_MAXCFL: float = field(init=False)
    XSTEP_TS: float = field(init=False)

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
