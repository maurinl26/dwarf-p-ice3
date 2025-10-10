# -*- coding: utf-8 -*-
from dataclasses import dataclass, field, InitVar
import cython

import sys
import numpy as np


######## phyex/common/aux/modd_cst.F90 ###########
@dataclass
@cython.cclass
class Constants:
    """Data class for physical constants

    1. Fondamental constants
    pi: (InitVar[float])
    karman: (InitVar[float])
    lightspeed: (InitVar[float])
    planck: (InitVar[float])
    boltz: (InitVar[float])
    avogadro: (InitVar[float])

    2. Astronomical constants
    day: (InitVar[float]) day duration
    siyea: (InitVar[float]) sideral year duration
    siday: (InitVar[float]) sidearl day duration
    nsday: (int) number of seconds in a day
    omega: (flaot) earth rotation

    3. Terrestrial geoide constants
    radius: (InitVar[float]): earth radius
    gravity0: (InitVar[float]): gravity constant

    4. Reference pressure
    Ocean model constants identical to 1D/CMO SURFEX
    p00ocean: (InitVar[float])  Ref pressure for ocean model
    rho0ocean: (InitVar[float]) Ref density for ocean model
    th00ocean: (InitVar[float]) Ref value for pot temp in ocean model
    sa00ocean: (InitVar[float]) Ref value for salinity in ocean model

    Atmospheric model
    p00: (InitVar[float]) Reference pressure
    th00: (InitVar[float]) Ref value for potential temperature

    5. Radiation constants
    stefan: (InitVar[float]) Stefan-Boltzman constant
    io: (InitVar[float]) Solar constant

    6. Thermodynamic constants
    Md: InitVar[float]          # Molar mass of dry air
    Mv: InitVar[float]          # Molar mass of water vapour
    Rd: InitVar[float]          # Gas constant for dry air
    Rv: InitVar[float]          # Gas constant for vapour
    epsilo: InitVar[float]      # Mv / Md
    cpd: InitVar[float]         # Cpd (dry air)
    cpv: InitVar[float]         # Cpv (vapour)
    rholw: InitVar[float]       # Volumic mass of liquid water
    Cl: InitVar[float]          # Cl (liquid)
    Ci: InitVar[float]          # Ci (ice)
    tt: InitVar[float]          # triple point temperature
    lvtt: InitVar[float]        # vaporisation heat constant
    lstt: InitVar[float]        # sublimation heat constant
    lmtt: InitVar[float]        # melting heat constant
    estt: InitVar[float]        # Saturation vapor pressure at triple point temperature

    alpw: InitVar[float]        # Constants for saturation vapor pressure function over water
    betaw: InitVar[float]
    gamw: InitVar[float]

    alpi: InitVar[float]        # Constants for saturation vapor pressure function over solid ice
    betai: InitVar[float]
    gami: InitVar[float]

    condi: InitVar[float]       # Thermal conductivity of ice (W m-1 K-1)
    alphaoc: InitVar[float]     # Thermal expansion coefficient for ocean (K-1)
    betaoc: InitVar[float]      # Haline contraction coeff for ocean (S-1)
    roc: InitVar[float] = 0.69  # coeff for SW penetration in ocean (Hoecker et al)
    d1: InitVar[float] = 1.1    # coeff for SW penetration in ocean (Hoecker et al)
    d2: InitVar[float] = 23.0   # coeff for SW penetration in ocean (Hoecker et al)

    rholi: InitVar[float]       # Volumic mass of ice

    7. Precomputed constants
    Rd_Rv: InitVar[float]       # Rd / Rv
    Rd_cpd: InitVar[float]      # Rd / cpd
    invxp00: InitVar[float]     # 1 / p00

    8. Machine precision
    mnh_tiny: InitVar[float]    # minimum real on this machine
    mnh_tiny_12: InitVar[float] # sqrt(minimum real on this machine)
    mnh_epsilon: InitVar[float] # minimum space with 1.0
    mnh_huge: InitVar[float]    # minimum real on this machine
    mnh_huge_12_log: InitVar[float] # maximum log(sqrt(real)) on this machine
    eps_dt: InitVar[float]      # default value for dt
    res_flat_cart: InitVar[float]   # default     flat&cart residual tolerance
    res_other: InitVar[float]   # default not flat&cart residual tolerance
    res_prep: InitVar[float]    # default     prep      residual tolerance

    """

    # 1. Fondamental constants
    KARMAN: InitVar[float] = 0.4
    LIGHTSPEED: InitVar[float] = 299792458.0
    PLANCK: InitVar[float] = 6.6260775e-34
    BOLTZ: InitVar[float] = 1.380658e-23
    AVOGADRO: InitVar[float] = 6.0221367e23

    # 2. Astronomical constants
    DAY: InitVar[float] = 86400  # day duration
    NSDAY: int = 24 * 3600  # number of seconds in a day

    # 3. Terrestrial geoide constants
    RADIUS: InitVar[float] = 6371229  # earth radius
    GRAVITY0: InitVar[float] = 9.80665  # gravity constant

    # 4. Reference pressure
    P00OCEAN: InitVar[float] = 201e5  # Ref pressure for ocean model
    RHO0OCEAN: InitVar[float] = 1024  # Ref density for ocean model
    TH00OCEAN: InitVar[float] = 286.65  # Ref value for pot temp in ocean model
    SA00OCEAN: InitVar[float] = 32.6  # Ref value for salinity in ocean model

    P00: InitVar[float] = 1e5  # Reference pressure
    TH00: InitVar[float] = 300  # Ref value for potential temperature

    # 5. Radiation constants
    IO: InitVar[float] = 1370  # Solar constant

    # 6. Thermodynamic constants
    MD: InitVar[float] = 28.9644e-3  # Molar mass of dry air
    MV: InitVar[float] = 18.0153e-3  # Molar mass of water vapour
    RHOLW: InitVar[float] = 1000  # Volumic mass of liquid water
    RHOLI: InitVar[float] = 900  # Volumic mass of ice
    CL: InitVar[float] = 4.218e3  # Cl (liquid
    CI: InitVar[float] = 2.106e3  # Ci (ice
    TT: InitVar[float] = 273.16  # triple point temperature
    LVTT: InitVar[float] = 2.5008e6  # vaporisation heat constant
    LSTT: InitVar[float] = 2.8345e6  # sublimation heat constant
    ESTT: InitVar[float] = 611.24
      # Saturation vapor pressure at triple point temperature

   
    CONDI: InitVar[float] = 2.2  # Thermal conductivity of ice (W m-1 K-1
    ALPHAOC: InitVar[float] = 1.9e-4 # Thermal expansion coefficient for ocean (K-1
    BETAOC: InitVar[float] = 7.7475  # Haline contraction coeff for ocean (S-1
    ROC: InitVar[float] = 0.69  # coeff for SW penetration in ocean (Hoecker et al
    D1: InitVar[float] = 1.1  # coeff for SW penetration in ocean (Hoecker et al
    D2: InitVar[float] = 23.0  # coeff for SW penetration in ocean (Hoecker et al


    # 8. Machine precision
    # MNH_TINY: InitVar[float] = field(
        # default=sys.InitVar[float]_info.epsilon
    # )
      # minimum real on this machine
    # MNH_TINY_12: InitVar[float] = sys.InitVar[float]_info. # sqrt(minimum real on this machine)
    # MNH_EPSILON: InitVar[float] # minimum space with 1.0
    # MNH_HUGE: InitVar[float]    # minimum real on this machine
    # MNH_HUGE_12_LOG: InitVar[float] # maximum log(sqrt(real)) on this machine
    # EPS_DT: InitVar[float]      # default value for dt
    # RES_FLAT_CART: InitVar[float]   # default     flat&cart residual tolerance
    # RES_OTHER: InitVar[float]   # default not flat&cart residual tolerance
    # RES_PREP: InitVar[float]    # default     prep      residual tolerance
        
    @property
    def PI(self):
        return 2 * np.arcsin(1.0)
       
    @property 
    def SIYEA(self):
        return 365.25 * self.DAY / 6.283076
        
    @property
    def SIDAY(self):
        return self.DAY / (1 + self.DAY / self.SIYEA)
        
    @property
    def OMEGA(self):
        return 2 * self.PI / self.SIDAY

        # 5. Radiation constants
    @property
    def STEFAN(self):
        return (
            2
            * self.PI**5
            * self.BOLTZ**4
            / (15 * self.LIGHTSPEED**2 * self.PLANCK**3)
        )


   

    # 6. Thermodynamic constants
    @property
    def RD(self):
        return self.AVOGADRO * self.BOLTZ / self.MD
        
    @property
    def RV(self):
        return self.AVOGADRO * self.BOLTZ / self.MV
        
    @property
    def EPSILO(self):
        return self.MV / self.MD
        
    @property
    def CPD(self):
        return (7 / 2) * self.RD
        
    @property
    def CPV(self):
        return 4 * self.RV

    @property
    def LMTT(self):
        return self.LSTT - self.LVTT
        
    @property
    def GAMW(self):
        return (self.CL - self.CPV) / self.RV
        
    @property
    def BETAW(self):
        return (self.LVTT / self.RV) + (self.GAMW * self.TT)
        
    @property
    def ALPW(self):
        return (
            np.log(self.ESTT) + (self.BETAW / self.TT) + (self.GAMW * np.log(self.TT))
        )
        
    @property
    def GAMI(self):
        return(self.CI - self.CPV) / self.RV
        
    @property
    def BETAI(self): 
        return (self.LSTT / self.RV) + self.GAMI * self.TT
        
    @property
    def ALPI(self):
        return (
            np.log(self.ESTT) + (self.BETAI / self.TT) + self.GAMI * np.log(self.TT)
        )

    @property 
    def RD_RV(self):
        return self.RD / self.RV
        
    @property
    def RD_CPD(self):
        return self.RD / self.CPD
        
    @property
    def INVXP00(self):
        return 1 / self.P00

