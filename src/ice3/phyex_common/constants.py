# -*- coding: utf-8 -*-
from dataclasses import dataclass, field

import sys
import numpy as np

@dataclass
class Constants:
    """Data class for physical constants

    1. Fondamental constants
    pi: (float)
    karman: (float)
    lightspeed: (float)
    planck: (float)
    boltz: (float)
    avogadro: (float)

    2. Astronomical constants
    day: (float) day duration
    siyea: (float) sideral year duration
    siday: (float) sidearl day duration
    nsday: (int) number of seconds in a day
    omega: (flaot) earth rotation

    3. Terrestrial geoide constants
    radius: (float): earth radius
    gravity0: (float): gravity constant

    4. Reference pressure
    Ocean model constants identical to 1D/CMO SURFEX
    p00ocean: (float)  Ref pressure for ocean model
    rho0ocean: (float) Ref density for ocean model
    th00ocean: (float) Ref value for pot temp in ocean model
    sa00ocean: (float) Ref value for salinity in ocean model

    Atmospheric model
    p00: (float) Reference pressure
    th00: (float) Ref value for potential temperature

    5. Radiation constants
    stefan: (float) Stefan-Boltzman constant
    io: (float) Solar constant

    6. Thermodynamic constants
    Md: float          # Molar mass of dry air
    Mv: float          # Molar mass of water vapour
    Rd: float          # Gas constant for dry air
    Rv: float          # Gas constant for vapour
    epsilo: float      # Mv / Md
    cpd: float         # Cpd (dry air)
    cpv: float         # Cpv (vapour)
    rholw: float       # Volumic mass of liquid water
    Cl: float          # Cl (liquid)
    Ci: float          # Ci (ice)
    tt: float          # triple point temperature
    lvtt: float        # vaporisation heat constant
    lstt: float        # sublimation heat constant
    lmtt: float        # melting heat constant
    estt: float        # Saturation vapor pressure at triple point temperature

    alpw: float        # Constants for saturation vapor pressure function over water
    betaw: float
    gamw: float

    alpi: float        # Constants for saturation vapor pressure function over solid ice
    betai: float
    gami: float

    condi: float       # Thermal conductivity of ice (W m-1 K-1)
    alphaoc: float     # Thermal expansion coefficient for ocean (K-1)
    betaoc: float      # Haline contraction coeff for ocean (S-1)
    roc: float = 0.69  # coeff for SW penetration in ocean (Hoecker et al)
    d1: float = 1.1    # coeff for SW penetration in ocean (Hoecker et al)
    d2: float = 23.0   # coeff for SW penetration in ocean (Hoecker et al)

    rholi: float       # Volumic mass of ice

    7. Precomputed constants
    Rd_Rv: float       # Rd / Rv
    Rd_cpd: float      # Rd / cpd
    invxp00: float     # 1 / p00

    8. Machine precision
    mnh_tiny: float    # minimum real on this machine
    mnh_tiny_12: float # sqrt(minimum real on this machine)
    mnh_epsilon: float # minimum space with 1.0
    mnh_huge: float    # minimum real on this machine
    mnh_huge_12_log: float # maximum log(sqrt(real)) on this machine
    eps_dt: float      # default value for dt
    res_flat_cart: float   # default     flat&cart residual tolerance
    res_other: float   # default not flat&cart residual tolerance
    res_prep: float    # default     prep      residual tolerance

    """

    # 1. Fondamental constants
    PI: float = field(default=2 * np.arcsin(1.0))
    KARMAN: float = field(default=0.4)
    LIGHTSPEED: float = field(default=299792458.0)
    PLANCK: float = field(default=6.6260775e-34)
    BOLTZ: float = field(default=1.380658e-23)
    AVOGADRO: float = field(default=6.0221367e23)

    # 2. Astronomical constants
    DAY: float = field(default=86400)  # day duration
    SIYEA: float = field(init=False)  # sideral year duration
    SIDAY: float = field(init=False)  # sideral day duration
    NSDAY: int = field(default=24 * 3600)  # number of seconds in a day
    OMEGA: float = field(init=False)  # earth rotation

    # 3. Terrestrial geoide constants
    RADIUS: float = field(default=6371229)  # earth radius
    GRAVITY0: float = field(default=9.80665)  # gravity constant

    # 4. Reference pressure
    P00OCEAN: float = field(default=201e5)  # Ref pressure for ocean model
    RHO0OCEAN: float = field(default=1024)  # Ref density for ocean model
    TH00OCEAN: float = field(default=286.65)  # Ref value for pot temp in ocean model
    SA00OCEAN: float = field(default=32.6)  # Ref value for salinity in ocean model

    P00: float = field(default=1e5)  # Reference pressure
    TH00: float = field(default=300)  # Ref value for potential temperature

    # 5. Radiation constants
    STEFAN: float = field(init=False)  # Stefan-Boltzman constant
    IO: float = field(default=1370)  # Solar constant

    # 6. Thermodynamic constants
    MD: float = field(default=28.9644e-3)  # Molar mass of dry air
    MV: float = field(default=18.0153e-3)  # Molar mass of water vapour
    RD: float = field(init=False)  # Gas constant for dry air
    RV: float = field(init=False)  # Gas constant for vapour
    EPSILO: float = field(init=False)  # Mv / Md
    CPD: float = field(init=False)  # Cpd (dry air)
    CPV: float = field(init=False)  # Cpv (vapour)
    RHOLW: float = field(default=1000)  # Volumic mass of liquid water
    RHOLI: float = field(default=900)  # Volumic mass of ice
    CL: float = field(default=4.218e3)  # Cl (liquid)
    CI: float = field(default=2.106e3)  # Ci (ice)
    TT: float = field(default=273.16)  # triple point temperature
    LVTT: float = field(default=2.5008e6)  # vaporisation heat constant
    LSTT: float = field(default=2.8345e6)  # sublimation heat constant
    LMTT: float = field(init=False)  # melting heat constant
    ESTT: float = field(
        default=611.24
    )  # Saturation vapor pressure at triple point temperature

    ALPW: float = field(init=False)  # Constants for saturation vapor pressure function
    BETAW: float = field(init=False)
    GAMW: float = field(init=False)

    ALPI: float = field(
        init=False
    )  # Constants for saturation vapor pressure function over solid ice
    BETAI: float = field(init=False)
    GAMI: float = field(init=False)

    CONDI: float = field(default=2.2)  # Thermal conductivity of ice (W m-1 K-1)
    ALPHAOC: float = field(
        default=1.9e-4
    )  # Thermal expansion coefficient for ocean (K-1)
    BETAOC: float = field(default=7.7475)  # Haline contraction coeff for ocean (S-1)
    ROC: float = 0.69  # coeff for SW penetration in ocean (Hoecker et al)
    D1: float = 1.1  # coeff for SW penetration in ocean (Hoecker et al)
    D2: float = 23.0  # coeff for SW penetration in ocean (Hoecker et al)

    # 7. Precomputed constants
    RD_RV: float = field(init=False)  # Rd / Rv
    RD_CPD: float = field(init=False)  # Rd / cpd
    INVXP00: float = field(init=False)  # 1 / p00

    # 8. Machine precision
    MNH_TINY: float = field(
        default=sys.float_info.epsilon
    )  # minimum real on this machine
    # MNH_TINY_12: float = field(default=sys.float_info.) # sqrt(minimum real on this machine)
    # MNH_EPSILON: float # minimum space with 1.0
    # MNH_HUGE: float    # minimum real on this machine
    # MNH_HUGE_12_LOG: float # maximum log(sqrt(real)) on this machine
    # EPS_DT: float      # default value for dt
    # RES_FLAT_CART: float   # default     flat&cart residual tolerance
    # RES_OTHER: float   # default not flat&cart residual tolerance
    # RES_PREP: float    # default     prep      residual tolerance

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
