"""
Turbulence scheme constants for JAX implementation.

This module provides turbulence constants from MODD_CTURB and MODE_INI_TURB.
Translated from PHYEX-IAL_CY50T1/turb/modd_cturb.F90 and mode_ini_turb.F90
"""

from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TurbulenceConstants:
    """
    Turbulence scheme constants (MODD_CTURB).

    This class contains all constants for the 1.5-order turbulence closure scheme.
    The constants are derived from theoretical considerations and tuned against
    Large-Eddy Simulation (LES) and observational data.

    Theoretical Basis
    ----------------
    The constants are based on the Redelsperger-Sommeria (1981) framework with
    updates from Schmidt-Schumann (1989) and Cheng-Canuto-Howard (2002).

    Key constant relationships (Redelsperger & Sommeria 1981, Eq. 3.1-3.10):

    - XCMFS = (2/3) / XCEP · (1 - XA0)  [momentum flux, shear]
    - XCMFB = (2/3) / XCEP · XA0        [momentum flux, buoyancy]
    - XCSHF = (2/3) / XCTP               [sensible heat flux]
    - XCTV = (2/3) / (XCTP · XCTD)      [temperature variance]

    Typical Values by Configuration
    ------------------------------
    AROME (Operational NWP):
    - XCED = 0.845  (Schmidt-Schumann 1989)
    - XCTP = 4.65   (Cheng-Canuto-Howard 2002)
    - XCMFS ≈ 0.143

    RM17 (Research):
    - XCED = 0.34   (Rodier et al. 2017, improved for stable boundary layers)
    - XCTP = 4.65
    - XRM17 = 0.5   (shear term in mixing length)

    Attributes
    ----------
    xcmfs : float
        Constant for momentum flux due to shear (RS), typically 0.066-0.143
    xcmfb : float
        Constant for momentum flux due to buoyancy
    xcshf : float
        Constant for sensible heat flux (RS), typically 0.143
    xchf : float
        Constant for humidity flux (RS)
    xctv : float
        Constant for temperature variance (RS)
    xchv : float
        Constant for humidity variance (RS)
    xcht1 : float
        First constant for temperature-humidity correlation (RS)
    xcht2 : float
        Second constant for temperature-humidity correlation (RS)
    xcpr1 : float
        First constant for turbulent Prandtl numbers
    xcpr2 : float
        Second constant for turbulent Prandtl numbers
    xcpr3 : float
        Third constant for turbulent Prandtl numbers (Schmidt number)
    xcpr4 : float
        Fourth constant for turbulent Prandtl numbers
    xcpr5 : float
        Fifth constant for turbulent Prandtl numbers
    xcet : float
        Constant in transport term of TKE equation
    xced : float
        Constant for dissipation term (0.34-0.85 depending on scheme)
    xctp : float
        Constant for temperature pressure-correlations (4.0-4.65)
    xcdp : float
        Constant for production term in dissipation equation
    xcdd : float
        Constant for destruction term in dissipation equation
    xcdt : float
        Constant for transport term in dissipation equation
    xrm17 : float
        Rodier et al. 2017 constant in shear term for mixing length
    xlinf : float
        Small value to prevent division by zero in BL algorithm
    xalpsbl : float
        Constant linking TKE and friction velocity in SBL
    xcep : float
        Constant for wind pressure-correlations
    xa0 : float
        Constant a0 for wind pressure-correlations
    xa2 : float
        Constant a2 for wind pressure-correlations
    xa3 : float
        Constant a3 for wind pressure-correlations
    xa5 : float
        Constant a5 for temperature pressure-correlations
    xctd : float
        Constant for temperature and vapor dissipation
    xphi_lim : float
        Threshold value for Phi3 and Psi3
    xsbl_o_bl : float
        SBL height / BL height ratio
    xtop_o_fsurf : float
        Fraction of surface flux used to define top of BL
    xbl89exp : float
        Exponent for BL89 mixing length (2/3 or computed)
    xusrbl89 : float
        Inverse of xbl89exp

    References
    ----------
    - Redelsperger, J.-L., and G. Sommeria, 1981: Méthode de représentation de
      la turbulence d'echelle inférieure à la maille pour un modèle
      tri-dimensionnel de convection nuageuse. Boundary-Layer Meteor., 21, 509-530.
      https://doi.org/10.1007/BF02033592
      (Foundation of closure constants, Section 3)

    - Schmidt, H., and U. Schumann, 1989: Coherent structure of the convective
      boundary layer derived from large-eddy simulations. J. Fluid Mech., 200, 511-562.
      https://doi.org/10.1017/S0022112089000753
      (XCED = 0.845 from LES, XCMFS = 0.086)

    - Cheng, Y., V. M. Canuto, and A. M. Howard, 2002: An improved model for the
      turbulent PBL. J. Atmos. Sci., 59, 1550-1565.
      https://doi.org/10.1175/1520-0469(2002)059<1550:AIMFTT>2.0.CO;2
      (XCEP = 2.11, XCTP = 4.65)

    - Rodier, Q., H. Masson, E. Couvreux, and A. Paci, 2017: Evaluation of a
      buoyancy and shear based mixing length for a turbulence scheme.
      Bound.-Layer Meteor., 165, 401-419.
      https://doi.org/10.1007/s10546-017-0272-3
      (XRM17 = 0.5, XCED = 0.34 for improved stable boundary layer)

    - Bougeault, P., and P. Lacarrere, 1989: Parameterization of orography-induced
      turbulence in a mesobeta-scale model. Mon. Wea. Rev., 117, 1872-1890.
      https://doi.org/10.1175/1520-0493(1989)117<1872:POOITI>2.0.CO;2
      (XCET = 0.4 for TKE transport)
    """

    # Wind pressure-correlations
    xcep: float = 2.11    # Cheng-Canuto-Howard (2002) value
    xa0: float = 0.6      # RS (1981)
    xa2: float = 1.0      # RS (1981)
    xa3: float = 0.0      # RS (1981)
    xa5: float = 1.0/3.0  # RS (1981)

    # Dissipation constants
    xctd: float = 1.2     # RS (1981)
    xced: float = 0.85    # AROME default (Schmidt-Schumann 1989)
    xctp: float = 4.65    # Cheng-Canuto-Howard (2002)

    # TKE equation
    xcet: float = 0.40    # Bougeault-Lacarrere (1989)

    # K-epsilon scheme (optional, for KEPS turbulence length)
    xcdp: float = 1.46    # Duynkerke (1988)
    xcdd: float = 1.83    # Duynkerke (1988)
    xcdt: float = 0.42    # Duynkerke (1988)

    # Mixing length shear term
    xrm17: float = 0.5    # Rodier et al. (2017)

    # Numerical safety
    xlinf: float = 1.0e-10  # Prevent division by zero

    # Surface boundary layer
    xalpsbl: float = 4.63   # Redelsperger et al. (2001)

    # Stability limits
    xphi_lim: float = 3.0
    xsbl_o_bl: float = 0.05
    xtop_o_fsurf: float = 0.05

    # BL89 mixing length exponent (default 2/3, can be computed)
    xbl89exp: float = 2.0/3.0
    xusrbl89: float = 1.5  # 1 / xbl89exp

    # Derived constants (computed in __post_init__)
    xcmfs: float = None   # Momentum flux due to shear
    xcmfb: float = None   # Momentum flux due to buoyancy (set in post_init)
    xcshf: float = None   # Sensible heat flux
    xchf: float = None    # Humidity flux
    xctv: float = None    # Temperature variance
    xchv: float = None    # Humidity variance
    xcht1: float = None   # Temperature-humidity correlation 1
    xcht2: float = None   # Temperature-humidity correlation 2
    xcpr1: float = None   # Prandtl number constant 1
    xcpr2: float = None   # Prandtl number constant 2
    xcpr3: float = None   # Prandtl number constant 3
    xcpr4: float = None   # Prandtl number constant 4
    xcpr5: float = None   # Prandtl number constant 5

    def __post_init__(self):
        """Compute derived constants from base constants."""
        # Momentum flux constant (RS formulation)
        xcmfs = 2.0/3.0/self.xcep * (1.0 - self.xa0)
        # RS (1981): 0.066, SS (1989): 0.086

        # Heat flux constant (RS formulation)
        xcshf = 2.0/3.0/self.xctp
        # RS (1981): 0.167, SS (1989): 0.204

        # Humidity flux (same as sensible heat)
        xchf = xcshf

        # Temperature variance constant (RS formulation)
        xctv = 2.0/3.0/self.xctp/self.xctd
        # RS (1981): 0.139, SS (1989): 0.202

        # Humidity variance (same as temperature)
        xchv = xctv

        # Temperature-humidity correlation constants
        xcht1 = xctv / 2.0
        xcht2 = xctv / 2.0

        # Prandtl/Schmidt number constants
        xcpr1 = xctv
        xcpr2 = xcht1
        xcpr3 = xcpr2
        xcpr4 = xcpr2
        xcpr5 = xcpr2

        # Momentum flux due to buoyancy
        xcmfb = 2.0 / 3.0 / self.xcep * self.xa0

        # Use object.__setattr__ to modify frozen dataclass
        object.__setattr__(self, 'xcmfs', xcmfs)
        object.__setattr__(self, 'xcmfb', xcmfb)
        object.__setattr__(self, 'xcshf', xcshf)
        object.__setattr__(self, 'xchf', xchf)
        object.__setattr__(self, 'xctv', xctv)
        object.__setattr__(self, 'xchv', xchv)
        object.__setattr__(self, 'xcht1', xcht1)
        object.__setattr__(self, 'xcht2', xcht2)
        object.__setattr__(self, 'xcpr1', xcpr1)
        object.__setattr__(self, 'xcpr2', xcpr2)
        object.__setattr__(self, 'xcpr3', xcpr3)
        object.__setattr__(self, 'xcpr4', xcpr4)
        object.__setattr__(self, 'xcpr5', xcpr5)

    @classmethod
    def arome(cls) -> 'TurbulenceConstants':
        """
        Create constants for AROME configuration.

        Returns
        -------
        TurbulenceConstants
            Constants configured for AROME operational model

        Fortran Source
        --------------
        PHYEX-IAL_CY50T1/turb/modd_cturb.F90, mode_ini_turb.F90
        AROME operational configuration (XCED=0.85, XCTP=4.65)
        """
        return cls(xced=0.85, xctp=4.65)

    @classmethod
    def rm17(cls) -> 'TurbulenceConstants':
        """
        Create constants for RM17 (Rodier et al. 2017) configuration.

        Returns
        -------
        TurbulenceConstants
            Constants configured for RM17 mixing length scheme

        Fortran Source
        --------------
        PHYEX-IAL_CY50T1/turb/modd_cturb.F90, mode_ini_turb.F90
        RM17 research configuration (XCED=0.34, XRM17=0.5)
        Based on Rodier et al. (2017) BL scheme
        """
        return cls(xced=0.34, xctp=4.65, xrm17=0.5)

    def to_dict(self) -> dict:
        """
        Convert constants to dictionary for use in JAX functions.

        Returns
        -------
        dict
            Dictionary of all constant values

        Fortran Source
        --------------
        Helper method for JAX compatibility
        Converts frozen dataclass to mutable dictionary
        """
        return {
            'xcmfs': self.xcmfs,
            'xcmfb': self.xcmfb,
            'xcshf': self.xcshf,
            'xchf': self.xchf,
            'xctv': self.xctv,
            'xchv': self.xchv,
            'xcht1': self.xcht1,
            'xcht2': self.xcht2,
            'xcpr1': self.xcpr1,
            'xcpr2': self.xcpr2,
            'xcpr3': self.xcpr3,
            'xcpr4': self.xcpr4,
            'xcpr5': self.xcpr5,
            'xcet': self.xcet,
            'xced': self.xced,
            'xctp': self.xctp,
            'xcdp': self.xcdp,
            'xcdd': self.xcdd,
            'xcdt': self.xcdt,
            'xrm17': self.xrm17,
            'xlinf': self.xlinf,
            'xalpsbl': self.xalpsbl,
            'xcep': self.xcep,
            'xa0': self.xa0,
            'xa2': self.xa2,
            'xa3': self.xa3,
            'xa5': self.xa5,
            'xctd': self.xctd,
            'xphi_lim': self.xphi_lim,
            'xsbl_o_bl': self.xsbl_o_bl,
            'xtop_o_fsurf': self.xtop_o_fsurf,
            'xbl89exp': self.xbl89exp,
            'xusrbl89': self.xusrbl89,
        }


# Global default constants instance (AROME configuration)
TURB_CONSTANTS = TurbulenceConstants.arome()


def get_turb_constants(config: str = 'AROME') -> TurbulenceConstants:
    """
    Get turbulence constants for a specific configuration.

    Parameters
    ----------
    config : str, optional
        Configuration name: 'AROME', 'RM17', or 'DEFAULT'
        Default: 'AROME'

    Returns
    -------
    TurbulenceConstants
        Turbulence constants for the specified configuration

    Examples
    --------
    >>> turb_cst = get_turb_constants('AROME')
    >>> print(f"XCED = {turb_cst.xced}")
    XCED = 0.85
    >>>
    >>> rm17_cst = get_turb_constants('RM17')
    >>> print(f"XCED = {rm17_cst.xced}, XRM17 = {rm17_cst.xrm17}")
    XCED = 0.34, XRM17 = 0.5

    Fortran Source
    --------------
    PHYEX-IAL_CY50T1/turb/mode_ini_turb.F90
    Subroutine: INI_CTURB (initialize turbulence constants)
    """
    if config.upper() == 'AROME':
        return TurbulenceConstants.arome()
    elif config.upper() == 'RM17':
        return TurbulenceConstants.rm17()
    else:
        return TurbulenceConstants()
