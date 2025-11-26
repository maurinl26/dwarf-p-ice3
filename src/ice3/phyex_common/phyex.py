# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass, field
from typing import Literal, Tuple
from enum import Enum

from ice3.phyex_common.constants import Constants
from ice3.phyex_common.nebn import Neb
from ice3.phyex_common.rain_ice_parameters import IceParameters
from ice3.phyex_common.rain_ice_descriptors import RainIceDescriptors
from ice3.phyex_common.rain_ice_parameters import RainIceParameters


class Boundary(Enum):
    PRESCRIBED = 0
    CYCL = 1


############# PHYEX/src/common/aux/modd_phyex.F90 #################
@dataclass
class Phyex:
    """
    Master configuration class for PHYEX atmospheric physics package.
    
    This dataclass serves as the central hub for all physics parameterizations
    in the PHYEX package. It aggregates and initializes all configuration
    dataclasses (Constants, IceParameters, RainIceDescriptors, RainIceParameters,
    Neb) and provides a unified interface for physics configuration. It also
    manages model-specific settings and provides externals for GT4Py stencils.
    
    Attributes
    ----------
    
    **Model Configuration**
    
    program : {"AROME", "MESO-NH"}
        Target atmospheric model. Required input that determines default
        configurations for all sub-components.
    
    timestep : float
        Physics time step (s). Default: 1.0.
        Note: Deprecated, use TSTEP instead.
    
    **Aggregated Configuration Objects (Auto-initialized)**
    
    cst : Constants
        Physical and thermodynamic constants. Initialized in __post_init__.
    
    param_icen : IceParameters
        ICE3/ICE4 microphysics configuration parameters.
        Model-specific defaults applied based on program.
    
    rain_ice_descrn : RainIceDescriptors
        Hydrometeor physical properties (mass-diameter, fall speeds, etc.).
        Depends on cst and param_icen.
    
    rain_ice_param : RainIceParameters
        Computed microphysical rate constants (100+ parameters).
        Depends on cst, rain_ice_descrn, and param_icen.
    
    nebn : Neb
        Subgrid cloud fraction and statistical cloud scheme parameters.
        Model-specific defaults applied based on program.
    
    **Time Stepping**
    
    TSTEP : float
        Physics time step (s). Default: 45.0.
        Main time step parameter used throughout physics calculations.
    
    INV_TSTEP : float
        Inverse time step 1/TSTEP (s⁻¹). Computed in __post_init__.
        Used for tendency calculations.
    
    ITERMAX : int
        Maximum iterations for saturation adjustment. Default: 1.
        Higher values improve accuracy but increase cost.
    
    NRR : float
        Number of rain categories. Default: 6.
        Typically 6 for ICE4 (cloud, rain, ice, snow, graupel, hail).
    
    **Condensation Scheme Options**
    
    LMFCONV : bool
        Use convective mass flux in condensation scheme. Default: True.
        When True, shallow convection mass fluxes influence condensation.
    
    COMPUTE_SRC : bool
        Compute s'r' covariance. Default: True.
        For advanced statistical cloud schemes.
    
    **Parallel Computing**
    
    KHALO : int
        Halo size for parallel domain decomposition. Default: 1.
        Number of ghost points for MPI communication in turbulence.
    
    **Turbulence Options**
    
    NOMIXLG : bool
        Turbulence for Lagrangian variables. Default: False.
        Special treatment for particle tracking.
    
    O2D : bool
        Use 2D version of turbulence scheme. Default: False.
        For idealized 2D simulations.
    
    **Ocean Coupling**
    
    OCEAN : bool
        Ocean version of turbulence scheme. Default: False.
        Use oceanic boundary layer parameterization.
    
    DEEPOC : bool
        Deep ocean mode. Default: False.
        For deep ocean applications.
    
    COUPLES : bool
        Ocean-atmosphere LES interactive coupling. Default: False.
        Enable two-way coupling in LES mode.
    
    **Boundary Conditions**
    
    LBCX : Tuple[int, int]
        Boundary conditions in x-direction (west, east).
        Default: (CYCL, CYCL) = (1, 1) - periodic boundaries.
        Options: PRESCRIBED (0), CYCL (1).
    
    LBCY : Tuple[int, int]
        Boundary conditions in y-direction (south, north).
        Default: (CYCL, CYCL) = (1, 1) - periodic boundaries.
    
    **Special Physics Options**
    
    BLOWSNOW : bool
        Enable blowing snow parameterization. Default: False.
        For surface snow transport and sublimation.
    
    RSNOW : float
        Blowing snow factor (dimensionless). Default: 1.0.
        Scaling factor for blowing snow flux.
    
    IBM : bool
        Immersed Boundary Method activated. Default: False.
        For complex terrain representation.
    
    **Diagnostics**
    
    FLYER : bool
        Meso-NH flyer diagnostic mode. Default: False.
        Special output for aircraft-like trajectory sampling.
    
    DIAG_IN_RUN : bool
        LES diagnostics during runtime. Default: False.
        Compute and output LES-specific diagnostics.
    
    PROGRAM : str
        Program name string. Default: "AROME".
        Redundant with program parameter.
    
    Methods
    -------
    to_externals()
        Convert all configuration to dictionary for GT4Py externals.
        Returns dict with all constants and parameters flattened.
    
    externals (property)
        Property accessor for to_externals().
    
    Notes
    -----
    
    **Initialization Flow:**
    
    1. User creates Phyex with program="AROME" or "MESO-NH"
    2. __post_init__ initializes in order:
       - Constants (independent)
       - IceParameters (depends on program)
       - Neb (depends on program)
       - RainIceDescriptors (depends on Constants, IceParameters)
       - RainIceParameters (depends on all above)
       - Computes INV_TSTEP
    
    **Model-Specific Defaults:**
    
    AROME:
    - Compact microphysics scheme (LRED=False, MRSTEP=0)
    - Statistical sedimentation (SEDIM=STAT)
    - PDF-based autoconversion (SUBG_AUCV_RC=PDF)
    - Temperature-based ice fraction (FRAC_ICE_ADJUST=T)
    - Subgrid condensation enabled (LSUBG_COND=True)
    - Adjustment before microphysics (LADJ_BEFORE=True, LADJ_AFTER=False)
    
    Meso-NH:
    - Time splitting for large time steps (LRED=True)
    - Mixing ratio splitting (MRSTEP=5e-5)
    - Split sedimentation (SEDIM=SPLI)
    - Statistical ice fraction (FRAC_ICE_ADJUST=S)
    - Adjustments before and after (LADJ_BEFORE=True, LADJ_AFTER=True)
    
    **GT4Py Integration:**
    
    The externals property/method flattens all nested dataclasses into a
    single dictionary suitable for passing as GT4Py stencil externals:
    
    ```python
    phyex = Phyex(program="AROME", TSTEP=60.0)
    externals = phyex.externals  # Dict with all constants
    stencil(..., **externals)
    ```
    
    **Usage Pattern:**
    
    ```python
    # Initialize for AROME
    phyex = Phyex(program="AROME", TSTEP=60.0)
    
    # Access sub-components
    print(f"Gas constant: {phyex.cst.RD} J/(kg·K)")
    print(f"Autoconversion threshold: {phyex.param_icen.CRIAUTC_NAM} kg/kg")
    print(f"Rain slope factor: {phyex.rain_ice_descrn.LBR} m^-1")
    
    # Get externals for GT4Py
    ext = phyex.externals
    ice_adjust_stencil(..., **ext)
    ```
    
    Source Reference
    ----------------
    PHYEX/src/common/aux/modd_phyex.F90
    
    See Also
    --------
    Constants : Physical constants
    IceParameters : Microphysics configuration
    RainIceDescriptors : Hydrometeor properties
    RainIceParameters : Microphysical rate constants
    Neb : Subgrid cloud scheme parameters
    
    Examples
    --------
    >>> # AROME configuration
    >>> phyex_arome = Phyex(program="AROME", TSTEP=60.0)
    >>> print(phyex_arome.param_icen.SEDIM)  # Statistical sedimentation
    1
    >>> print(phyex_arome.nebn.LSUBG_COND)  # Subgrid condensation enabled
    True
    
    >>> # Meso-NH configuration
    >>> phyex_mnh = Phyex(program="MESO-NH", TSTEP=2.0)
    >>> print(phyex_mnh.param_icen.SEDIM)  # Split sedimentation
    0
    >>> print(phyex_mnh.param_icen.MRSTEP)  # Mixing ratio splitting
    5e-05
    
    >>> # Access computed parameters
    >>> print(f"Rain evaporation coeff: {phyex_arome.rain_ice_param.O0EVAR:.2e}")
    >>> print(f"Number of externals: {len(phyex_arome.externals)}")
    """

    program: Literal["AROME", "MESO-NH"]
    timestep: float = field(default=1)

    cst: Constants = field(init=False)
    param_icen: IceParameters = field(init=False)
    rain_ice_descrn: RainIceDescriptors = field(init=False)
    rain_ice_param: RainIceParameters = field(init=False)
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
        self.param_icen = IceParameters(self.PROGRAM)
        self.nebn = Neb(self.PROGRAM)
        self.rain_ice_descrn = RainIceDescriptors(self.cst, self.param_icen)
        self.rain_ice_param = RainIceParameters(
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

    @property
    def externals(self):
        return self.to_externals()
