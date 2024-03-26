# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from enum import Enum


class VerticalLevelOrder(Enum):
    """Specify order of index on vertical levels

    SPACE_TO_GROUND for AROME-like indexing
    GROUND_TO_PACE for Meso-NH like indexing
    """

    SPACE_TO_GROUND = -1
    GROUND_TO_SPACE = 1


@dataclass
class DIMPhyex:
    """Specify index boundaries for PHYEX domain

    Not used in dwarf-ice3-gt4py but reproduced for translation support

    # x dimension
    nit: int  # Array dim
    nib: int = field(init=False)  # First index
    nie: int = field(init=False)  # Last index

    # y dimension
    njt: int
    njb: int = field(init=False)
    nje: int = field(init=False)

    # z dimension
    nkt: int  # Array total dimension on z (nz)
    nkles: int  # Total physical k dimension

    nka: int  # Near ground array index
    nku: int  # Uppest atmosphere array index

    nkb: int  # Near ground physical array index
    nke: int  # Uppest atmosphere physical array index

    nktb: int  # smaller index for the physical domain
    nkte: int  # greater index for the physical domain

    nibc: int
    njbc: int
    niec: int
    nijt: int = field(init=False)  # horizontal packing
    nijb: int = field(init=False)  # first index for horizontal packing
    nije: int = field(init=False)  # last index for horizontal packing
    """

    # x dimension
    NIT: int  # Array dim

    NIB: int = field(init=False)  # First index
    NIE: int = field(init=False)  # Last index

    # y dimension
    NJT: int
    NJB: int = field(init=False)
    NJE: int = field(init=False)

    # z dimension
    VERTICAL_LEVEL_ORDER: VerticalLevelOrder

    # TODO: remove nkl (FORTRAN implementation) to use VerticalLevelOrder
    NKL: int  # Order of the vertical levels
    # 1 : Meso NH order (bottom to top)
    # -1 : AROME order (top to bottom)

    NKT: int  # Array total dimension on z (nz)
    NKLES: int  # Total physical k dimension

    NKA: int  # Near ground array index
    NKU: int  # Uppest atmosphere array index

    NKB: int  # Near ground physical array index
    NKE: int  # Uppest atmosphere physical array index

    NKTB: int  # smaller index for the physical domain
    NKTE: int  # greater index for the physical domain

    NIBC: int
    NJBC: int
    NIEC: int
    NIJT: int = field(init=False)  # horizontal packing
    NIJB: int = field(init=False)  # first index for horizontal packing
    NIJE: int = field(init=False)  # last index for horizontal packing

    def __post_init__(self):
        self.NIB, self.NIE = 0, self.NIT - 1  # python like indexing
        self.NJB, self.NJE = 0, self.NJT - 1

        self.NIJT = self.NIT * self.NJT
        self.NIJB, self.NIJE = 0, self.NIJT - 1
