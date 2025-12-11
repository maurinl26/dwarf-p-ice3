"""
Utilities for converting phyex_common dataclasses to ctypes.Structure.

This module provides functions to convert Python dataclasses used in PHYEX
configuration to ctypes.Structure objects that can be passed to Fortran
via ctypes.
"""

import ctypes
from dataclasses import fields, is_dataclass
from typing import Any, Type
import numpy as np


def dataclass_to_ctypes_structure(obj: Any) -> Type[ctypes.Structure]:
    """
    Convert a dataclass instance to a ctypes.Structure class.
    
    Parameters
    ----------
    obj : dataclass instance
        The dataclass object to convert
    
    Returns
    -------
    ctypes.Structure class
        A dynamically created Structure class with fields matching
        the dataclass
    
    Example
    -------
    >>> from ice3.phyex_common.constants import Constants
    >>> cst = Constants()
    >>> CSTStructure = dataclass_to_ctypes_structure(cst)
    >>> cst_c = CSTStructure()
    >>> # Populate fields
    >>> for field in fields(cst):
    ...     setattr(cst_c, field.name.lower(), getattr(cst, field.name))
    """
    if not is_dataclass(obj):
        raise TypeError(f"{obj} is not a dataclass instance")
    
    # Get all fields from the dataclass
    dataclass_fields = fields(obj)
    
    # Map Python types to ctypes types
    type_map = {
        int: ctypes.c_int,
        float: ctypes.c_double,
        bool: ctypes.c_bool,
        str: ctypes.c_char_p,
    }
    
    # Build _fields_ list for ctypes.Structure
    structure_fields = []
    
    for field in dataclass_fields:
        field_name = field.name.lower()  # Fortran uses lowercase
        field_value = getattr(obj, field.name)
        
        # Determine ctypes type
        if field.type in type_map:
            ctype = type_map[field.type]
        elif field.type == type(None):
            # Skip None fields
            continue
        elif isinstance(field_value, (int, np.integer)):
            ctype = ctypes.c_int
        elif isinstance(field_value, (float, np.floating)):
            ctype = ctypes.c_double
        elif isinstance(field_value, bool):
            ctype = ctypes.c_bool
        elif isinstance(field_value, str):
            ctype = ctypes.c_char_p
        else:
            # Unknown type, skip
            continue
        
        structure_fields.append((field_name, ctype))
    
    # Create the Structure class dynamically
    class_name = f"{obj.__class__.__name__}_t"
    
    structure_class = type(
        class_name,
        (ctypes.Structure,),
        {'_fields_': structure_fields}
    )
    
    return structure_class


def populate_ctypes_structure(structure_instance: ctypes.Structure, 
                              dataclass_obj: Any) -> None:
    """
    Populate a ctypes.Structure instance with values from a dataclass.
    
    Parameters
    ----------
    structure_instance : ctypes.Structure
        The Structure instance to populate
    dataclass_obj : dataclass
        The source dataclass object
    
    Example
    -------
    >>> cst = Constants()
    >>> CSTStructure = dataclass_to_ctypes_structure(cst)
    >>> cst_c = CSTStructure()
    >>> populate_ctypes_structure(cst_c, cst)
    """
    for field in fields(dataclass_obj):
        field_name_lower = field.name.lower()
        if hasattr(structure_instance, field_name_lower):
            value = getattr(dataclass_obj, field.name)
            
            # Handle string encoding for ctypes
            if isinstance(value, str):
                value = value.encode('utf-8')
            
            # Handle bool to int conversion if needed
            elif isinstance(value, bool):
                value = int(value)
            
            try:
                setattr(structure_instance, field_name_lower, value)
            except (TypeError, AttributeError):
                # Skip fields that can't be set
                pass


def create_and_populate_structure(dataclass_obj: Any) -> ctypes.Structure:
    """
    Create and populate a ctypes.Structure from a dataclass in one step.
    
    Parameters
    ----------
    dataclass_obj : dataclass
        The dataclass to convert
    
    Returns
    -------
    ctypes.Structure instance
        A populated Structure instance
    
    Example
    -------
    >>> from ice3.phyex_common.constants import Constants
    >>> cst = Constants()
    >>> cst_struct = create_and_populate_structure(cst)
    >>> print(cst_struct.xtt)  # Triple point temperature
    """
    # Create the Structure class
    StructureClass = dataclass_to_ctypes_structure(dataclass_obj)
    
    # Create an instance
    struct_instance = StructureClass()
    
    # Populate it
    populate_ctypes_structure(struct_instance, dataclass_obj)
    
    return struct_instance


# Specific converters for common PHYEX types

def constants_to_ctypes(cst) -> ctypes.Structure:
    """
    Convert Constants dataclass to ctypes.Structure.
    
    Parameters
    ----------
    cst : Constants
        Constants instance
    
    Returns
    -------
    ctypes.Structure
        CST_t structure with physical constants
    """
    
    class CST_t(ctypes.Structure):
        """Fortran CST_t derived type."""
        _fields_ = [
            # Fundamental constants
            ('xtt', ctypes.c_double),      # Triple point temperature [K]
            ('xp00', ctypes.c_double),     # Reference pressure [Pa]
            ('xrd', ctypes.c_double),      # Gas constant dry air [J/(kg·K)]
            ('xrv', ctypes.c_double),      # Gas constant water vapor [J/(kg·K)]
            ('xcpd', ctypes.c_double),     # Cp dry air [J/(kg·K)]
            ('xcpv', ctypes.c_double),     # Cp water vapor [J/(kg·K)]
            ('xcl', ctypes.c_double),      # Cp liquid water [J/(kg·K)]
            ('xci', ctypes.c_double),      # Cp ice [J/(kg·K)]
            ('xlvtt', ctypes.c_double),    # Latent heat vaporization at Xtt [J/kg]
            ('xlstt', ctypes.c_double),    # Latent heat sublimation at Xtt [J/kg]
            ('xlmtt', ctypes.c_double),    # Latent heat melting at Xtt [J/kg]
            ('xg', ctypes.c_double),       # Gravity [m/s²]
            ('xmd', ctypes.c_double),      # Molecular weight dry air [kg/mol]
            ('xmv', ctypes.c_double),      # Molecular weight water [kg/mol]
            ('xepsilo', ctypes.c_double),  # Rd/Rv
            ('xalpi', ctypes.c_double),    # Ice saturation vapor pressure constant
            ('xbetai', ctypes.c_double),   # Ice saturation vapor pressure constant
            ('xgami', ctypes.c_double),    # Ice saturation vapor pressure constant
            ('xalpw', ctypes.c_double),    # Water saturation vapor pressure constant
            ('xbetaw', ctypes.c_double),   # Water saturation vapor pressure constant
            ('xgamw', ctypes.c_double),    # Water saturation vapor pressure constant
        ]
    
    cst_struct = CST_t()
    
    # Populate fields (using lowercase field names for Fortran)
    cst_struct.xtt = cst.XTT
    cst_struct.xp00 = cst.XP00
    cst_struct.xrd = cst.RD
    cst_struct.xrv = cst.RV
    cst_struct.xcpd = cst.CPD
    cst_struct.xcpv = cst.CPV
    cst_struct.xcl = cst.CL
    cst_struct.xci = cst.CI
    cst_struct.xlvtt = cst.LVTT
    cst_struct.xlstt = cst.LSTT
    cst_struct.xlmtt = cst.LMTT
    cst_struct.xg = cst.G
    cst_struct.xmd = cst.MD
    cst_struct.xmv = cst.MV
    cst_struct.xepsilo = cst.EPSILO
    cst_struct.xalpi = cst.ALPI
    cst_struct.xbetai = cst.BETAI
    cst_struct.xgami = cst.GAMI
    cst_struct.xalpw = cst.ALPW
    cst_struct.xbetaw = cst.BETAW
    cst_struct.xgamw = cst.GAMW
    
    return cst_struct


def dimphyex_to_ctypes(nijt: int, nkt: int) -> ctypes.Structure:
    """
    Create DIMPHYEX_t structure with dimensions.
    
    Parameters
    ----------
    nijt : int
        Number of horizontal points
    nkt : int
        Number of vertical levels
    
    Returns
    -------
    ctypes.Structure
        DIMPHYEX_t structure
    """
    
    class DIMPHYEX_t(ctypes.Structure):
        """Fortran DIMPHYEX_t derived type."""
        _fields_ = [
            ('nijt', ctypes.c_int),
            ('nkt', ctypes.c_int),
            ('nktb', ctypes.c_int),
            ('nkte', ctypes.c_int),
            ('nijb', ctypes.c_int),
            ('nije', ctypes.c_int),
        ]
    
    d = DIMPHYEX_t()
    d.nijt = nijt
    d.nkt = nkt
    d.nktb = 1
    d.nkte = nkt
    d.nijb = 1
    d.nije = nijt
    
    return d


def neb_to_ctypes(nebn) -> ctypes.Structure:
    """
    Convert Neb dataclass to ctypes.Structure.
    
    Parameters
    ----------
    nebn : Neb
        Neb instance
    
    Returns
    -------
    ctypes.Structure
        NEB_t structure
    """
    
    class NEB_t(ctypes.Structure):
        """Fortran NEB_t derived type (simplified)."""
        _fields_ = [
            ('lsubg_cond', ctypes.c_bool),  # Subgrid condensation
            ('lsigmas', ctypes.c_bool),     # Sigma_s scheme
            ('cfrac_ice_adjust', ctypes.c_char * 4),  # Ice fraction method
            ('ccondens', ctypes.c_char * 8),  # Condensation scheme
            ('clambda3', ctypes.c_char * 8),  # Lambda3 scheme
        ]
    
    neb_struct = NEB_t()
    neb_struct.lsubg_cond = nebn.LSUBG_COND
    neb_struct.lsigmas = nebn.LSIGMAS
    neb_struct.cfrac_ice_adjust = nebn.CFRAC_ICE_ADJUST.encode('utf-8')
    neb_struct.ccondens = nebn.CCONDENS.encode('utf-8')
    neb_struct.clambda3 = nebn.CLAMBDA3.encode('utf-8')
    
    return neb_struct
