# -*- coding: utf-8 -*-
import sys
import numpy as np
import cython


# from_file="PHYEX/src/common/aux/modd_cst.F90"
######## phyex/common/aux/modd_cst.F90 ###########
@cython.cclass
class Constants:
    """
    Physical, thermodynamic, and astronomical constants for atmospheric modeling.
    
    This dataclass defines all fundamental constants used in the PHYEX atmospheric
    physics package, including universal physical constants, Earth parameters,
    thermodynamic properties of air and water substances, and machine precision
    limits. Values match the Meso-NH atmospheric model conventions.
    
    Attributes
    ----------
    
    **1. Fundamental Physical Constants**
    
    PI : float
        Mathematical constant π (dimensionless).
    KARMAN : float
        Von Kármán constant for turbulence (dimensionless), typically 0.4.
    LIGHTSPEED : float
        Speed of light in vacuum (m/s), 299792458.
    PLANCK : float
        Planck constant (J·s), 6.6260775×10⁻³⁴.
    BOLTZ : float
        Boltzmann constant (J/K), 1.380658×10⁻²³.
    AVOGADRO : float
        Avogadro's number (mol⁻¹), 6.0221367×10²³.
    
    **2. Astronomical Constants**
    
    DAY : float
        Duration of one day (s), 86400.
    SIYEA : float
        Sidereal year duration (s), computed from DAY.
    SIDAY : float
        Sidereal day duration (s), computed from DAY and SIYEA.
    NSDAY : int
        Number of seconds in a day, 86400.
    OMEGA : float
        Earth's angular rotation rate (rad/s), 2π/SIDAY.
    
    **3. Terrestrial Geoid Constants**
    
    RADIUS : float
        Mean Earth radius (m), 6371229.
    GRAVITY0 : float
        Standard gravitational acceleration (m/s²), 9.80665.
    
    **4. Reference States**
    
    Ocean Model (1D/CMO SURFEX):
    P00OCEAN : float
        Reference pressure for ocean (Pa), 201×10⁵.
    RHO0OCEAN : float
        Reference density for ocean (kg/m³), 1024.
    TH00OCEAN : float
        Reference potential temperature for ocean (K), 286.65.
    SA00OCEAN : float
        Reference salinity for ocean (psu), 32.6.
    
    Atmospheric Model:
    P00 : float
        Reference pressure (Pa), 10⁵ (1000 hPa).
    TH00 : float
        Reference potential temperature (K), 300.
    
    **5. Radiation Constants**
    
    STEFAN : float
        Stefan-Boltzmann constant (W/(m²·K⁴)), computed from fundamental constants.
    IO : float
        Solar constant at top of atmosphere (W/m²), 1370.
    
    **6. Thermodynamic Constants**
    
    Molecular Properties:
    MD : float
        Molar mass of dry air (kg/mol), 28.9644×10⁻³.
    MV : float
        Molar mass of water vapor (kg/mol), 18.0153×10⁻³.
    RD : float
        Gas constant for dry air (J/(kg·K)), computed as R/MD.
    RV : float
        Gas constant for water vapor (J/(kg·K)), computed as R/MV.
    EPSILO : float
        Ratio of molecular weights MV/MD (dimensionless), ≈0.622.
    
    Specific Heat Capacities:
    CPD : float
        Specific heat of dry air at constant pressure (J/(kg·K)), 3.5×RD.
    CPV : float
        Specific heat of water vapor at constant pressure (J/(kg·K)), 4×RV.
    CL : float
        Specific heat of liquid water (J/(kg·K)), 4218.
    CI : float
        Specific heat of ice (J/(kg·K)), 2106.
    
    Densities:
    RHOLW : float
        Density of liquid water (kg/m³), 1000.
    RHOLI : float
        Density of ice (kg/m³), 900.
    
    Phase Change Properties:
    TT : float
        Triple point temperature (K), 273.16.
    LVTT : float
        Latent heat of vaporization at TT (J/kg), 2.5008×10⁶.
    LSTT : float
        Latent heat of sublimation at TT (J/kg), 2.8345×10⁶.
    LMTT : float
        Latent heat of melting (J/kg), LSTT - LVTT.
    ESTT : float
        Saturation vapor pressure at TT (Pa), 611.24.
    
    Saturation Vapor Pressure Coefficients:
    ALPW, BETAW, GAMW : float
        Coefficients for e_sat over liquid water.
        Formula: e_sat = exp(ALPW - BETAW/T - GAMW×ln(T))
    ALPI, BETAI, GAMI : float
        Coefficients for e_sat over ice.
        Formula: e_sat = exp(ALPI - BETAI/T - GAMI×ln(T))
    
    Ocean/Ice Properties:
    CONDI : float
        Thermal conductivity of ice (W/(m·K)), 2.2.
    ALPHAOC : float
        Thermal expansion coefficient for ocean water (K⁻¹), 1.9×10⁻⁴.
    BETAOC : float
        Haline contraction coefficient for ocean (psu⁻¹), 7.7475.
    ROC : float
        Coefficient for shortwave penetration in ocean, 0.69 (Hoecker et al).
    D1, D2 : float
        Coefficients for shortwave penetration depth in ocean (m), 1.1 and 23.0.
    
    **7. Precomputed Convenience Constants**
    
    RD_RV : float
        Ratio RD/RV (dimensionless).
    RD_CPD : float
        Ratio RD/CPD (dimensionless).
    INVXP00 : float
        Inverse reference pressure 1/P00 (Pa⁻¹).
    
    **8. Machine Precision**
    
    MNH_TINY : float
        Machine epsilon, smallest representable positive number.
    
    Notes
    -----
    The constants are initialized with physical values or computed in
    __post_init__ from fundamental constants. This ensures consistency
    between related constants (e.g., latent heats, gas constants).
    
    The saturation vapor pressure formulas use constants fitted to
    observational data and are temperature-dependent to account for
    varying molecular properties across the atmospheric temperature range.
    
    Source Reference
    ----------------
    PHYEX/src/common/aux/modd_cst.F90
    
    References
    ----------
    Coesa, 1976: U.S. Standard Atmosphere.
    NOAA, NASA, USAF.
    
    Examples
    --------
    >>> cst = Constants()
    >>> print(f"Gas constant for dry air: {cst.RD:.2f} J/(kg·K)")
    Gas constant for dry air: 287.06 J/(kg·K)
    >>> print(f"Molecular weight ratio: {cst.EPSILO:.4f}")
    Molecular weight ratio: 0.6220
    """

    # 1. Fondamental constants
    PI: cython.double
    KARMAN: cython.double
    LIGHTSPEED: cython.double
    PLANCK: cython.double
    BOLTZ: cython.double
    AVOGADRO: cython.double

    # 2. Astronomical constants
    DAY: cython.double  # day duration
    SIYEA: cython.double  # sideral year duration
    SIDAY: cython.double  # sideral day duration
    NSDAY: cython.int  # number of seconds in a day
    OMEGA: cython.double  # earth rotation

    # 3. Terrestrial geoide constants
    RADIUS: cython.double  # earth radius
    GRAVITY0: cython.double  # gravity constant

    # 4. Reference pressure
    P00OCEAN: cython.double  # Ref pressure for ocean model
    RHO0OCEAN: cython.double  # Ref density for ocean model
    TH00OCEAN: cython.double  # Ref value for pot temp in ocean model
    SA00OCEAN: cython.double  # Ref value for salinity in ocean model

    P00: cython.double  # Reference pressure
    TH00: cython.double  # Ref value for potential temperature

    # 5. Radiation constants
    STEFAN: cython.double  # Stefan-Boltzman constant
    IO: cython.double  # Solar constant

    # 6. Thermodynamic constants
    MD: cython.double  # Molar mass of dry air
    MV: cython.double  # Molar mass of water vapour
    RD: cython.double  # Gas constant for dry air
    RV: cython.double  # Gas constant for vapour
    EPSILO: cython.double  # Mv / Md
    CPD: cython.double  # Cpd (dry air)
    CPV: cython.double  # Cpv (vapour)
    RHOLW: cython.double  # Volumic mass of liquid water
    RHOLI: cython.double  # Volumic mass of ice
    CL: cython.double  # Cl (liquid)
    CI: cython.double  # Ci (ice)
    TT: cython.double  # triple point temperature
    LVTT: cython.double  # vaporisation heat constant
    LSTT: cython.double  # sublimation heat constant
    LMTT: cython.double  # melting heat constant
    ESTT: cython.double  # Saturation vapor pressure at triple point temperature

    ALPW: cython.double  # Constants for saturation vapor pressure function
    BETAW: cython.double
    GAMW: cython.double

    ALPI: cython.double  # Constants for saturation vapor pressure function over solid ice
    BETAI: cython.double
    GAMI: cython.double

    CONDI: cython.double  # Thermal conductivity of ice (W m-1 K-1)
    ALPHAOC: cython.double  # Thermal expansion coefficient for ocean (K-1)
    BETAOC: cython.double  # Haline contraction coeff for ocean (S-1)
    ROC: cython.double  # coeff for SW penetration in ocean (Hoecker et al)
    D1: cython.double  # coeff for SW penetration in ocean (Hoecker et al)
    D2: cython.double  # coeff for SW penetration in ocean (Hoecker et al)

    # 7. Precomputed constants
    RD_RV: cython.double  # Rd / Rv
    RD_CPD: cython.double  # Rd / cpd
    INVXP00: cython.double  # 1 / p00

    # 8. Machine precision
    MNH_TINY: cython.double  # minimum real on this machine
    # MNH_TINY_12: float = field(default=sys.float_info.) # sqrt(minimum real on this machine)
    # MNH_EPSILON: float # minimum space with 1.0
    # MNH_HUGE: float    # minimum real on this machine
    # MNH_HUGE_12_LOG: float # maximum log(sqrt(real)) on this machine
    # EPS_DT: float      # default value for dt
    # RES_FLAT_CART: float   # default     flat&cart residual tolerance
    # RES_OTHER: float   # default not flat&cart residual tolerance
    # RES_PREP: float    # default     prep      residual tolerance

    def __init__(self):
        """Initialize all constants with their default values."""
        # 1. Fundamental constants
        self.PI = 2 * np.arcsin(1.0)
        self.KARMAN = 0.4
        self.LIGHTSPEED = 299792458.0
        self.PLANCK = 6.6260775e-34
        self.BOLTZ = 1.380658e-23
        self.AVOGADRO = 6.0221367e23

        # 2. Astronomical constants
        self.DAY = 86400.0
        self.NSDAY = 24 * 3600

        # 3. Terrestrial geoide constants
        self.RADIUS = 6371229.0
        self.GRAVITY0 = 9.80665

        # 4. Reference pressure
        self.P00OCEAN = 201e5
        self.RHO0OCEAN = 1024.0
        self.TH00OCEAN = 286.65
        self.SA00OCEAN = 32.6
        self.P00 = 1e5
        self.TH00 = 300.0

        # 5. Radiation constants
        self.IO = 1370.0

        # 6. Thermodynamic constants
        self.MD = 28.9644e-3
        self.MV = 18.0153e-3
        self.RHOLW = 1000.0
        self.RHOLI = 900.0
        self.CL = 4.218e3
        self.CI = 2.106e3
        self.TT = 273.16
        self.LVTT = 2.5008e6
        self.LSTT = 2.8345e6
        self.ESTT = 611.24

        self.CONDI = 2.2
        self.ALPHAOC = 1.9e-4
        self.BETAOC = 7.7475
        self.ROC = 0.69
        self.D1 = 1.1
        self.D2 = 23.0

        # 8. Machine precision
        self.MNH_TINY = sys.float_info.epsilon

        # Call post_init to compute derived values
        self.__post_init__()

    def __post_init__(self):
        # 2. Astronomical constants
        self.SIYEA = 365.25 * self.DAY / 6.283076
        self.SIDAY = self.DAY / (1 + self.DAY / self.SIYEA)
        self.OMEGA = 2 * self.PI / self.SIDAY

        # 5. Radiation constants
        self.STEFAN = (
            2
            * self.PI**5
            * self.BOLTZ**4
            / (15 * self.LIGHTSPEED**2 * self.PLANCK**3)
        )

        # 6. Thermodynamic constants
        self.RD = self.AVOGADRO * self.BOLTZ / self.MD
        self.RV = self.AVOGADRO * self.BOLTZ / self.MV
        self.EPSILO = self.MV / self.MD
        self.CPD = (7 / 2) * self.RD
        self.CPV = 4 * self.RV

        self.LMTT = self.LSTT - self.LVTT
        self.GAMW = (self.CL - self.CPV) / self.RV
        self.BETAW = (self.LVTT / self.RV) + (self.GAMW * self.TT)
        self.ALPW = (
            np.log(self.ESTT) + (self.BETAW / self.TT) + (self.GAMW * np.log(self.TT))
        )
        self.GAMI = (self.CI - self.CPV) / self.RV
        self.BETAI = (self.LSTT / self.RV) + self.GAMI * self.TT
        self.ALPI = (
            np.log(self.ESTT) + (self.BETAI / self.TT) + self.GAMI * np.log(self.TT)
        )

        self.RD_RV = self.RD / self.RV
        self.RD_CPD = self.RD / self.CPD
        self.INVXP00 = 1 / self.P00
