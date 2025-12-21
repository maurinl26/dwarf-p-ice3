# -*- coding: utf-8 -*-
from enum import Enum
import cython

class VerticalLevelOrder(Enum):
    """Specify order of index on vertical levels

    SPACE_TO_GROUND for AROME-like indexing
    GROUND_TO_PACE for Meso-NH like indexing
    """

    SPACE_TO_GROUND = -1
    GROUND_TO_SPACE = 1


@cython.cclass
class PhyexDimensions:
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
    NIT: cython.int  # Array dim

    NIB: cython.int  # First index
    NIE: cython.int  # Last index

    # y dimension
    NJT: cython.int
    NJB: cython.int
    NJE: cython.int

    # z dimension
    VERTICAL_LEVEL_ORDER: VerticalLevelOrder

    # TODO: remove nkl (FORTRAN implementation) to use VerticalLevelOrder
    NKL: cython.int  # Order of the vertical levels
    # 1 : Meso NH order (bottom to top)
    # -1 : AROME order (top to bottom)

    NKT: cython.int  # Array total dimension on z (nz)
    NKLES: cython.int  # Total physical k dimension

    NKA: cython.int  # Near ground array index
    NKU: cython.int  # Uppest atmosphere array index

    NKB: cython.int  # Near ground physical array index
    NKE: cython.int  # Uppest atmosphere physical array index

    NKTB: cython.int  # smaller index for the physical domain
    NKTE: cython.int  # greater index for the physical domain

    NIBC: cython.int
    NJBC: cython.int
    NIEC: cython.int
    NIJT: cython.int  # horizontal packing
    NIJB: cython.int  # first index for horizontal packing
    NIJE: cython.int  # last index for horizontal packing

    def __init__(
        self,
        NIT: int,
        NJT: int,
        VERTICAL_LEVEL_ORDER: VerticalLevelOrder,
        NKL: int,
        NKT: int,
        NKLES: int,
        NKA: int,
        NKU: int,
        NKB: int,
        NKE: int,
        NKTB: int,
        NKTE: int,
        NIBC: int,
        NJBC: int,
        NIEC: int,
    ):
        """Initialize PhyexDimensions with all required parameters."""
        self.NIT = NIT
        self.NJT = NJT
        self.VERTICAL_LEVEL_ORDER = VERTICAL_LEVEL_ORDER
        self.NKL = NKL
        self.NKT = NKT
        self.NKLES = NKLES
        self.NKA = NKA
        self.NKU = NKU
        self.NKB = NKB
        self.NKE = NKE
        self.NKTB = NKTB
        self.NKTE = NKTE
        self.NIBC = NIBC
        self.NJBC = NJBC
        self.NIEC = NIEC

        # Initialize computed fields
        self.NIB = 0
        self.NIE = 0
        self.NJB = 0
        self.NJE = 0
        self.NIJT = 0
        self.NIJB = 0
        self.NIJE = 0

        # Call post_init to compute derived values
        self.__post_init__()

    def __post_init__(self):
        self.NIB, self.NIE = 0, self.NIT - 1  # python like indexing
        self.NJB, self.NJE = 0, self.NJT - 1

        self.NIJT = self.NIT * self.NJT
        self.NIJB, self.NIJE = 0, self.NIJT - 1
