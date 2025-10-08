

class PhysicalConstants:
    """Physical constants for the ICE_ADJUST computation."""
    
    def __init__(self):
        # Default values - should be set according to the specific model
        self.XALPI = 0.0
        self.XALPW = 0.0  
        self.XBETAI = 0.0
        self.XBETAW = 0.0
        self.XCI = 2106.0      # Heat capacity of ice [J/kg/K]
        self.XCL = 4218.0      # Heat capacity of liquid water [J/kg/K]
        self.XCPD = 1004.709   # Heat capacity of dry air [J/kg/K]
        self.XCPV = 1846.1     # Heat capacity of water vapor [J/kg/K]
        self.XEPSILO = 0.622   # Rd/Rv
        self.XG = 9.80665      # Gravity [m/s2]
        self.XGAMI = 0.0
        self.XGAMW = 0.0
        self.XLSTT = 2.834e6   # Latent heat of sublimation at triple point [J/kg]
        self.XLVTT = 2.501e6   # Latent heat of vaporization at triple point [J/kg]  
        self.XPI = 3.141592653589793
        self.XRD = 287.0596    # Gas constant for dry air [J/kg/K]
        self.XRV = 461.524     # Gas constant for water vapor [J/kg/K]
        self.XTT = 273.16      # Triple point temperature [K]

class IceParameters:
    """Parameters for ice microphysics."""
    
    def __init__(self):
        self.XACRIAUTI = 0.0
        self.XBCRIAUTI = 0.0  
        self.XCRIAUTC = 0.0
        self.XCRIAUTI = 0.0
        self.XFRMIN = 0.0

class NebulosityParameters:
    """Parameters for nebulosity (cloud fraction) computation."""
    
    def __init__(self):
        self.CCONDENS = 'CB02'
        self.CFRAC_ICE_ADJUST = 'S'
        self.CLAMBDA3 = 'CB'
        self.LCONDBORN = True
        self.LHGT_QS = True
        self.LSIGMAS = True
        self.LSTATNW = True
        self.LSUBG_COND = True
        self.XTMAXMIX = 0.0
        self.XTMINMIX = 0.0
