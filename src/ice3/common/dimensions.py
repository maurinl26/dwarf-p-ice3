"""
Dimensions module - GT4Py translation of MODD_DIMPHYEX.

This module defines the dimension structure used in the physics calculations,
translated from the Fortran MODD_DIMPHYEX module.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DimPhyex:
    """
    Dimension structure equivalent to DIMPHYEX_t in Fortran.
    
    This class holds all the dimension information needed for the physics
    calculations, including array bounds and computational indices.
    """
    
    # X direction dimensions
    NIT: int  # Array total dimension
    NIB: int  # First inner mass point index
    NIE: int  # Last inner mass point index
    
    # Y direction dimensions
    NJT: int  # Array total dimension
    NJB: int  # First inner mass point index
    NJE: int  # Last inner mass point index
    
    # Z direction dimensions
    NKL: int   # Order of vertical levels (1: ground to space, -1: space to ground)
    NKT: int   # Array total dimension
    NKLES: int # Number of vertical levels for LES diagnostics
    NKA: int   # Near ground array index
    NKU: int   # Uppermost atmosphere array index
    NKB: int   # Near ground physical array index
    NKE: int   # Uppermost physical atmosphere array index
    NKTB: int  # Smaller index of physical domain
    NKTE: int  # Greater index of physical domain
    
    # Computational indices
    NIBC: int  # Computational indices used in DO LOOP
    NJBC: int  # = NIB/NJC/NIE/NJE in all schemes
    NIEC: int  # except in turbulence where external HALO points must be
    NJEC: int  # included so NIBC=NJBC=1 and NIEC/NJEC=NIT/NJT
    
    # Horizontal packing
    NIJT: int  # NIT*NJT for horizontal packing
    NIJB: int  # First horizontal inner mass point index
    NIJE: int  # Last horizontal inner mass point index
    
    # LES dimensions
    NLESMASK: int   # Number of LES masks
    NLES_TIMES: int # Number of LES time data storage
    
    def __post_init__(self):
        """Validate and compute derived dimensions."""
        # Validate basic constraints
        assert self.NIB >= 1 and self.NIE <= self.NIT
        assert self.NJB >= 1 and self.NJE <= self.NJT
        assert self.NKL in [-1, 1], "NKL must be 1 (ground to space) or -1 (space to ground)"
        
        # Compute derived horizontal dimensions if not provided
        if not hasattr(self, '_computed'):
            if self.NIJT == 0:
                self.NIJT = self.NIT * self.NJT
            if self.NIJB == 0:
                self.NIJB = (self.NJB - 1) * self.NIT + self.NIB
            if self.NIJE == 0:
                self.NIJE = (self.NJE - 1) * self.NIT + self.NIE
            
            self._computed = True
    
    @classmethod
    def create_simple(cls, ni: int, nj: int, nk: int, 
                     halo: int = 1, jpvext: int = 1) -> 'DimPhyex':
        """
        Create a simple DimPhyex structure with standard settings.
        
        Args:
            ni: Number of points in x direction (without halo)
            nj: Number of points in y direction (without halo)  
            nk: Number of points in z direction (without vertical extension)
            halo: Horizontal halo size (default: 1)
            jpvext: Vertical extension (default: 1)
            
        Returns:
            DimPhyex instance with computed dimensions
        """
        # Total dimensions including halo/extensions
        nit = ni + 2 * halo
        njt = nj + 2 * halo
        nkt = nk + 2 * jpvext
        
        # Physical bounds
        nib = 1 + halo
        nie = ni + halo
        njb = 1 + halo
        nje = nj + halo
        
        # Vertical bounds (assuming ground to space ordering)
        nkl = 1
        nka = 1
        nku = nkt
        nkb = 1 + jpvext
        nke = nk + jpvext
        nktb = nkb
        nkte = nke
        
        # Computational bounds (same as physical for most schemes)
        nibc = nib
        njbc = njb
        niec = nie
        njec = nje
        
        return cls(
            NIT=nit, NIB=nib, NIE=nie,
            NJT=njt, NJB=njb, NJE=nje,
            NKL=nkl, NKT=nkt, NKLES=nk,
            NKA=nka, NKU=nku, NKB=nkb, NKE=nke,
            NKTB=nktb, NKTE=nkte,
            NIBC=nibc, NJBC=njbc, NIEC=niec, NJEC=njec,
            NIJT=0, NIJB=0, NIJE=0,  # Will be computed in __post_init__
            NLESMASK=1, NLES_TIMES=1
        )
    
    def get_physical_shape(self) -> Tuple[int, int, int]:
        """Get the shape of the physical domain (without halo/extensions)."""
        return (self.NIE - self.NIB + 1, 
                self.NJE - self.NJB + 1, 
                self.NKTE - self.NKTB + 1)
    
    def get_total_shape(self) -> Tuple[int, int, int]:
        """Get the total array shape including halo/extensions."""
        return (self.NIT, self.NJT, self.NKT)
    
    def get_horizontal_packed_shape(self) -> Tuple[int, int]:
        """Get the shape for horizontally packed arrays."""
        return (self.NIJT, self.NKT)
