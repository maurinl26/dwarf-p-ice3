"""
Python wrapper for the ICE_ADJUST Fortran subroutine.

This module provides a Python interface to the ICE_ADJUST subroutine from PHYEX,
which computes the adjustment of water vapor in mixed-phase clouds through
saturation adjustment procedures.
"""

import numpy as np
from typing import Optional, Dict
import ctypes as ct
import os

from pyphyex.create_test_data import create_test_data
from pyphyex.physical_parameters import IceParameters, NebulosityParameters, PhysicalConstants


class IceAdjustWrapper:
    """
    Python wrapper for the ICE_ADJUST Fortran subroutine.
    
    This class provides a Python interface to the ICE_ADJUST subroutine which
    computes the adjustment of water vapor in mixed-phase clouds.
    """
    
    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize the wrapper.
        
        Parameters:
        -----------
        library_path : str, optional
            Path to the compiled Fortran library. If None, will try to find it automatically.
        """
        self.library_path = library_path or self._find_library()
        self.constants = PhysicalConstants()
        self.ice_params = IceParameters() 
        self.neb_params = NebulosityParameters()
        self._lib = None
        
    def _find_library(self) -> str:
        """Find the compiled library automatically."""
        # Look for library in common build locations
        possible_paths = [
            "build/lib/libphyex.so",
            "build/lib/libice-adjust.so",
            "../build/lib/libphyex.so",
            "/usr/local/lib/libphyex.so",
            "../PHYEX/build/lib/libphyex.so",
            "../PHYEX/build/lib/libice-adjust.so"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        raise FileNotFoundError(
            "Could not find compiled library. Please specify library_path or compile the project."
        )
    
    def _load_library(self):
        """Load the shared library."""
        if self._lib is None:
            self._lib = ct.CDLL(self.library_path)
            self._setup_function_signatures()
    
    def _setup_function_signatures(self):
        """Set up ctypes function signatures for the Fortran wrapper."""
        # This would need to be implemented based on the actual compiled interface
        pass
    
    def ice_adjust(self,
                   nijt: int,
                   nkt: int, 
                   krr: int,
                   ptstep: float,
                   psigqsat: np.ndarray,
                   prhodj: np.ndarray,
                   pexnref: np.ndarray,
                   prhodref: np.ndarray,
                   psigs: Optional[np.ndarray] = None,
                   lmfconv: bool = False,
                   pmfconv: Optional[np.ndarray] = None,
                   ppabst: np.ndarray = None,
                   pzz: np.ndarray = None,
                   pexn: np.ndarray = None,
                   pcf_mf: Optional[np.ndarray] = None,
                   prc_mf: Optional[np.ndarray] = None,
                   pri_mf: Optional[np.ndarray] = None,
                   pweight_mf_cloud: Optional[np.ndarray] = None,
                   prv: np.ndarray = None,
                   prc: np.ndarray = None,
                   prvs: np.ndarray = None,
                   prcs: np.ndarray = None,
                   pth: np.ndarray = None,
                   pths: np.ndarray = None,
                   ocompute_src: bool = False,
                   prr: Optional[np.ndarray] = None,
                   pri: Optional[np.ndarray] = None,
                   pris: Optional[np.ndarray] = None,
                   prs: Optional[np.ndarray] = None,
                   prg: Optional[np.ndarray] = None,
                   pice_cld_wgt: Optional[np.ndarray] = None,
                   prh: Optional[np.ndarray] = None,
                   hbuname: str = "ICEADJ") -> Dict[str, np.ndarray]:
        """
        Compute ice adjustment for mixed-phase clouds.
        
        Parameters:
        -----------
        nijt : int
            Horizontal dimension size
        nkt : int  
            Vertical dimension size
        krr : int
            Number of moist variables
        ptstep : float
            Time step [s]
        psigqsat : np.ndarray(nijt)
            Coefficient applied to qsat variance contribution
        prhodj : np.ndarray(nijt, nkt)
            Dry density * Jacobian [kg/m3]
        pexnref : np.ndarray(nijt, nkt)
            Reference Exner function
        prhodref : np.ndarray(nijt, nkt)
            Reference density [kg/m3]
        psigs : np.ndarray(nijt, nkt), optional
            Sigma_s at time t
        lmfconv : bool, optional
            Whether convective mass flux is present
        pmfconv : np.ndarray(nijt, nkt), optional
            Convective mass flux [kg/m2/s]
        ppabst : np.ndarray(nijt, nkt)
            Absolute pressure [Pa]
        pzz : np.ndarray(nijt, nkt)
            Height of model layer [m]
        pexn : np.ndarray(nijt, nkt)
            Exner function
        pcf_mf : np.ndarray(nijt, nkt), optional
            Convective mass flux cloud fraction
        prc_mf : np.ndarray(nijt, nkt), optional
            Convective mass flux liquid mixing ratio [kg/kg]
        pri_mf : np.ndarray(nijt, nkt), optional
            Convective mass flux ice mixing ratio [kg/kg]
        pweight_mf_cloud : np.ndarray(nijt, nkt), optional
            Weight coefficient for mass-flux cloud
        prv : np.ndarray(nijt, nkt)
            Water vapor mixing ratio to adjust [kg/kg]
        prc : np.ndarray(nijt, nkt) 
            Cloud water mixing ratio to adjust [kg/kg]
        prvs : np.ndarray(nijt, nkt)
            Water vapor mixing ratio source [kg/kg/s]
        prcs : np.ndarray(nijt, nkt)
            Cloud water mixing ratio source [kg/kg/s]
        pth : np.ndarray(nijt, nkt)
            Theta to adjust [K]
        pths : np.ndarray(nijt, nkt)
            Theta source [K/s]
        ocompute_src : bool, optional
            Whether to compute second-order flux
        prr : np.ndarray(nijt, nkt), optional
            Rain water mixing ratio [kg/kg]
        pri : np.ndarray(nijt, nkt), optional
            Cloud ice mixing ratio [kg/kg] 
        pris : np.ndarray(nijt, nkt), optional
            Cloud ice mixing ratio source [kg/kg/s]
        prs : np.ndarray(nijt, nkt), optional
            Aggregate mixing ratio [kg/kg]
        prg : np.ndarray(nijt, nkt), optional
            Graupel mixing ratio [kg/kg]
        pice_cld_wgt : np.ndarray(nijt), optional
            Ice cloud weight
        prh : np.ndarray(nijt, nkt), optional
            Hail mixing ratio [kg/kg]
        hbuname : str, optional
            Budget name (4 characters)
            
        Returns:
        --------
        dict
            Dictionary containing output arrays:
            - 'cldfr': Cloud fraction
            - 'icldfr': Ice cloud fraction  
            - 'wcldfr': Water/mixed-phase cloud fraction
            - 'ssio': Super-saturation w.r.t. ice in supersaturated fraction
            - 'ssiu': Sub-saturation w.r.t. ice in subsaturated fraction
            - 'ifr': Ratio cloud ice moist part to dry part
            - 'srcs': Second-order flux (if computed)
            - 'out_rv': Adjusted water vapor (if requested)
            - 'out_rc': Adjusted cloud water (if requested)
            - 'out_ri': Adjusted cloud ice (if requested)
            - 'out_th': Adjusted theta (if requested)
        """
        
        # Input validation
        self._validate_inputs(nijt, nkt, krr, ptstep, prhodj, pexnref, prhodref,
                             prv, prc, prvs, prcs, pth, pths)
        
        # Set default values for optional arrays
        if psigs is None:
            psigs = np.zeros((nijt, nkt))
        if pmfconv is None:
            pmfconv = np.zeros((nijt, nkt))
        if ppabst is None:
            ppabst = np.ones((nijt, nkt)) * 101325.0  # Standard pressure
        if pzz is None:
            pzz = np.zeros((nijt, nkt))
        if pexn is None:
            pexn = pexnref.copy()
        if pcf_mf is None:
            pcf_mf = np.zeros((nijt, nkt))
        if prc_mf is None:
            prc_mf = np.zeros((nijt, nkt))
        if pri_mf is None:
            pri_mf = np.zeros((nijt, nkt))
        if pweight_mf_cloud is None:
            pweight_mf_cloud = np.zeros((nijt, nkt))
        if prr is None:
            prr = np.zeros((nijt, nkt))
        if pri is None:
            pri = np.zeros((nijt, nkt))
        if pris is None:
            pris = np.zeros((nijt, nkt))
        if prs is None:
            prs = np.zeros((nijt, nkt))
        if prg is None:
            prg = np.zeros((nijt, nkt))
            
        # Initialize output arrays
        pcldfr = np.zeros((nijt, nkt), dtype=np.float64)
        picldfr = np.zeros((nijt, nkt), dtype=np.float64)
        pwcldfr = np.zeros((nijt, nkt), dtype=np.float64)
        pssio = np.zeros((nijt, nkt), dtype=np.float64)
        pssiu = np.zeros((nijt, nkt), dtype=np.float64)
        pifr = np.zeros((nijt, nkt), dtype=np.float64)
        psrcs = np.zeros((nijt, nkt), dtype=np.float64) if ocompute_src else None
        
        # For now, implement a simple saturation adjustment algorithm
        # This is a placeholder implementation that should be replaced with
        # actual calls to the Fortran library
        results = self._simple_ice_adjust(
            nijt, nkt, krr, ptstep, prhodj, pexnref, prhodref,
            prv, prc, prvs, prcs, pth, pths, ppabst, pexn
        )
        
        return results
    
    def _validate_inputs(self, nijt, nkt, krr, ptstep, prhodj, pexnref, prhodref,
                        prv, prc, prvs, prcs, pth, pths):
        """Validate input parameters."""
        if nijt <= 0 or nkt <= 0:
            raise ValueError("Dimensions must be positive")
        if krr < 2:
            raise ValueError("KRR must be at least 2 (vapor + cloud water)")
        if ptstep <= 0:
            raise ValueError("Time step must be positive")
            
        # Check array shapes
        expected_shape = (nijt, nkt)
        arrays_to_check = {
            'prhodj': prhodj, 'pexnref': pexnref, 'prhodref': prhodref,
            'prv': prv, 'prc': prc, 'prvs': prvs, 'prcs': prcs, 
            'pth': pth, 'pths': pths
        }
        
        for name, arr in arrays_to_check.items():
            if arr is not None and arr.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}")
            
    def _ice_adjust_wrapper(self):
        
        self._lib.ice_adjust()
    
    def _simple_ice_adjust(self, nijt, nkt, krr, ptstep, prhodj, pexnref, prhodref,
                          prv, prc, prvs, prcs, pth, pths, ppabst, pexn):
        """
        Simple ice adjustment implementation.
        
        This is a placeholder implementation that demonstrates the interface.
        In a real implementation, this would call the compiled Fortran code.
        """
        
        # Initialize output arrays
        pcldfr = np.zeros((nijt, nkt))
        picldfr = np.zeros((nijt, nkt))  
        pwcldfr = np.zeros((nijt, nkt))
        pssio = np.zeros((nijt, nkt))
        pssiu = np.zeros((nijt, nkt))
        pifr = np.zeros((nijt, nkt))
        
        # Simple saturation adjustment logic (placeholder)
        # In reality, this would involve complex thermodynamics calculations
        
        # Compute temperature from potential temperature and Exner function
        temp = pth * pexn
        
        # Simple cloud fraction based on relative humidity
        # (This is greatly simplified compared to the actual ICE_ADJUST)
        es = 611.2 * np.exp(17.67 * (temp - 273.15) / (temp - 29.65))  # Saturation vapor pressure
        qs = 0.622 * es / (ppabst - 0.378 * es)  # Saturation mixing ratio
        rh = prv / qs  # Relative humidity
        
        # Simple cloud fraction threshold
        pcldfr = np.where(rh > 1.0, 1.0, 0.0)
        
        # Ice vs liquid partitioning based on temperature
        ice_fraction = np.where(temp < 273.15, 1.0, 0.0)
        picldfr = pcldfr * ice_fraction
        pwcldfr = pcldfr * (1.0 - ice_fraction)
        
        return {
            'cldfr': pcldfr,
            'icldfr': picldfr,
            'wcldfr': pwcldfr,
            'ssio': pssio,
            'ssiu': pssiu,
            'ifr': pifr
        }


if __name__ == "__main__":
    # Example usage
    print("ICE_ADJUST Python Wrapper")
    print("=========================")
    
    # Create wrapper instance  
    try:
        wrapper = IceAdjustWrapper()
        print(f"Loaded library from: {wrapper.library_path}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Creating wrapper without library for demonstration...")
        wrapper = IceAdjustWrapper.__new__(IceAdjustWrapper)
        wrapper.constants = PhysicalConstants()
        wrapper.ice_params = IceParameters()
        wrapper.neb_params = NebulosityParameters()
    
    # Create test data
    print("\nGenerating test data...")
    test_data = create_test_data(nijt=50, nkt=30)
    
    # Run ice adjustment
    print("Running ice adjustment...")
    try:
        results = wrapper.ice_adjust(**test_data)
        
        print(f"\nResults:")
        print(f"Cloud fraction range: {results['cldfr'].min():.3f} - {results['cldfr'].max():.3f}")
        print(f"Ice cloud fraction range: {results['icldfr'].min():.3f} - {results['icldfr'].max():.3f}")
        print(f"Water cloud fraction range: {results['wcldfr'].min():.3f} - {results['wcldfr'].max():.3f}")
        print(f"Total cloudy points: {np.sum(results['cldfr'] > 0)}")
        
    except Exception as e:
        print(f"Error running ice adjustment: {e}")
