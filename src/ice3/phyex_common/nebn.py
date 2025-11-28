# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

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


class Condens(Enum):
    """Enumeration for condensation variance

    HCONDENS in .F90
    CB02 for AROME
    """
    CB02 = 0
    GAUS = 1


class Lambda3(Enum):
    """LAMBDA3 in AROME

    CB by default in AROME
    """
    CB = 0


# from_file="PHYEX/src/common/aux/modd_nebn.F90"
@dataclass
class Neb:
    """
    Subgrid cloud fraction and statistical cloud scheme parameters.
    
    This dataclass configures the subgrid condensation and cloud fraction
    schemes used in the atmospheric model. It controls how cloud heterogeneity
    is represented at scales smaller than the model grid, including the treatment
    of ice/liquid partitioning, supersaturation variance, and PDF-based
    condensation schemes.
    
    Attributes
    ----------
    
    **Model Configuration**
    
    HPROGRAM : {"AROME", "MESO-NH", "LMDZ"}
        Target atmospheric model. Sets appropriate defaults in __post_init__.
    
    **Mixed-Phase Temperature Range**
    
    TMINMIX : float
        Minimum temperature for mixed-phase clouds (K). Default: 273.16 (0°C).
        Below this, all condensate is liquid.
    TMAXMIX : float
        Maximum temperature for mixed-phase clouds (K). Default: 253.16 (-20°C).
        Above this, all condensate is ice.
        
    Between TMAXMIX and TMINMIX, ice fraction varies from 0 to 1.
    
    **Ice Fraction Parameterizations**
    
    FRAC_ICE_ADJUST : int
        Ice fraction method for saturation adjustments:
        - FracIceAdjust.T (0): Temperature-based (AROME default)
        - FracIceAdjust.O (1): Original scheme
        - FracIceAdjust.N (2): New scheme
        - FracIceAdjust.S (3): Statistical (Meso-NH default)
    
    FRAC_ICE_SHALLOW : int
        Ice fraction method for shallow convection mass fluxes:
        - FracIceShallow.T (0): Temperature-based (AROME default)
        - FracIceShallow.S (1): Statistical (Meso-NH default)
    
    **Subgrid Variability Parameters**
    
    VSIGQSAT : float
        Coefficient for saturation specific humidity variance contribution.
        Default: 0.02 (dimensionless).
        Controls subgrid variability of q_sat in cloud fraction calculation.
        
    LHGT_QS : bool
        Use height-dependent VSIGQSAT. Default: False.
        When True, VSIGQSAT varies with altitude to account for
        different variability at different levels.
    
    **Subgrid Condensation PDF**
    
    CONDENS : int
        Probability distribution function for subgrid condensation:
        - Condens.CB02 (0): Chaboureau and Bechtold (2002) (AROME default)
        - Condens.GAUS (1): Gaussian PDF
        
    The CB02 scheme uses a triangular PDF for supersaturation variance,
    while GAUS uses a normal distribution.
    
    LAMBDA3 : int
        Lambda3 parameter choice for subgrid cloud scheme:
        - Lambda3.CB (0): Chaboureau and Bechtold formulation (default)
        
    Defines the characteristic scale of subgrid variability.
    
    **Statistical Cloud Scheme Options**
    
    LSTATNW : bool
        Use updated full statistical cloud scheme. Default: False.
        When True, employs advanced statistical methods for cloud
        fraction and condensate distribution.
    
    LSIGMAS : bool
        Use sigma_s from turbulence scheme for cloud variability. Default: True.
        When True, subgrid standard deviation comes from prognostic
        turbulence equations. When False, diagnostic formulation used.
    
    LSUBG_COND : bool
        Enable subgrid condensation scheme. Default: False.
        When True, allows condensation at grid points with partial
        cloud cover using PDF-based approach.
        Set to True for AROME and LMDZ.
    
    Notes
    -----
    
    **Physical Basis:**
    
    Real clouds are heterogeneous at all scales. Grid-scale models cannot
    resolve subgrid variability, leading to:
    - All-or-nothing cloud decisions (0% or 100% cloud cover)
    - Inaccurate radiative transfer
    - Poor representation of precipitation onset
    
    **Statistical Approach:**
    
    The statistical cloud scheme represents subgrid variability using:
    1. PDF of total water mixing ratio q_t
    2. PDF of saturation mixing ratio q_sat
    3. Cloud fraction as probability q_t > q_sat
    
    **CB02 Scheme (Chaboureau and Bechtold, 2002):**
    - Triangular PDF for supersaturation
    - Cloud fraction: σ = max(0, 1 - a|S|/σ_s) where S is saturation deficit
    - Efficient and commonly used in NWP models
    - Performs well for stratiform and convective clouds
    
    **Ice Fraction Methods:**
    
    Temperature-based (T):
    - Simple linear interpolation between TMAXMIX and TMINMIX
    - f_ice = (T_mix - T) / (T_mix - T_max)
    - Fast and robust
    
    Statistical (S):
    - Accounts for subgrid temperature variability
    - More physical but computationally expensive
    
    **Supersaturation Variance:**
    
    VSIGQSAT controls how much q_sat variability contributes:
    - Higher values → more cloud at grid scale
    - Lower values → sharper cloud edges
    - Typical range: 0.01-0.05
    
    **Model-Specific Settings:**
    
    AROME:
    - FRAC_ICE_ADJUST = T (temperature-based)
    - FRAC_ICE_SHALLOW = T
    - VSIGQSAT = 0.02
    - LSIGMAS = True (use prognostic variance)
    - LSUBG_COND = True (enable subgrid condensation)
    
    LMDZ:
    - LSUBG_COND = True
    - Other parameters at defaults
    
    Meso-NH:
    - Uses default values (statistical methods)
    
    Source Reference
    ----------------
    PHYEX/src/common/aux/modd_nebn.F90
    
    References
    ----------
    Chaboureau, J.-P., and P. Bechtold, 2002: A simple cloud
    parameterization derived from cloud resolving model data:
    Diagnostic and prognostic applications. J. Atmos. Sci., 59, 2362-2372.
    
    Sommeria, G., and J.W. Deardorff, 1977: Subgrid-scale condensation
    in models of nonprecipitating clouds. J. Atmos. Sci., 34, 344-355.
    
    Bougeault, P., 1981: Modeling the trade-wind cumulus boundary layer.
    Part I: Testing the ensemble cloud relations against numerical data.
    J. Atmos. Sci., 38, 2414-2428.
    
    Examples
    --------
    >>> # AROME configuration
    >>> neb_arome = Neb(HPROGRAM="AROME")
    >>> print(neb_arome.LSUBG_COND)  # Subgrid condensation enabled
    True
    >>> print(neb_arome.FRAC_ICE_ADJUST)  # Temperature-based
    0
    
    >>> # Meso-NH with statistical ice fraction
    >>> neb_mnh = Neb(HPROGRAM="MESO-NH")
    >>> print(neb_mnh.FRAC_ICE_ADJUST)  # Statistical method
    3
    """

    HPROGRAM: Literal["AROME", "MESO-NH", "LMDZ"]

    TMINMIX: float = field(default=273.16)
    TMAXMIX: float = field(default=253.16)
    LHGT_QS: bool = field(default=False)
    FRAC_ICE_ADJUST: int = field(default=FracIceAdjust.S.value)
    FRAC_ICE_SHALLOW: int = field(default=FracIceShallow.S.value)
    VSIGQSAT: float = field(default=0.02)
    CONDENS: int = field(default=Condens.CB02.value)
    LAMBDA3: int = field(default=Lambda3.CB.value)
    LSTATNW: bool = field(default=False)
    LSIGMAS: bool = field(default=True)
    LSUBG_COND: bool = field(default=False)

    def __post_init__(self):
        if self.HPROGRAM == "AROME":
            self.FRAC_ICE_ADJUST = FracIceAdjust.T.value
            self.FRAC_ICE_SHALLOW = FracIceShallow.T.value
            self.VSIGQSAT = 0.02
            self.LSIGMAS = True
            self.LSUBG_COND = True

        elif self.HPROGRAM == "LMDZ":
            self.LSUBG_COND = True
