# -*- coding: utf-8 -*-
"""Doctor norm naming conventions for Fortran interoperability.

This module implements the "doctor norm" naming convention used in French meteorological
software (ARPEGE, AROME, Méso-NH). This convention uses prefixes to indicate variable
types, making Fortran code more readable and helping avoid type-related bugs.

The doctor norm prefixes are:
- 'p' for real (float) arrays/fields
- 'l' for logical (boolean) variables
- 'x' for real (float) scalars
- 'n' for integer scalars

These functions are used when interfacing with legacy Fortran code to ensure Python
field names are converted to the expected Fortran naming convention.
"""

def field_doctor_norm(key: str, dtype: str) -> str:
    """Apply doctor norm naming convention to field (array) variables.
    
    Adds the appropriate prefix to field names based on their data type following
    the doctor norm convention used in Fortran meteorological code. Fields are
    multi-dimensional arrays (e.g., temperature, pressure, humidity fields).

    Args:
        key (str): Base field name without prefix (e.g., "rc_t", "ldcompute")
        dtype (str): Field data type, one of "float" or "bool"

    Returns:
        str: Field name with doctor norm prefix applied (e.g., "prc_t", "lldcompute")
        
    Examples:
        >>> field_doctor_norm("rc_t", "float")
        "prc_t"
        >>> field_doctor_norm("ldcompute", "bool")
        "lldcompute"
    """
    match dtype:
        case "float":
            return f"p{key}"
        case "bool":
            return f"l{key}"


def var_doctor_norm(key: str, dtype: str) -> str:
    """Apply doctor norm naming convention to scalar variables.
    
    Adds the appropriate prefix to scalar variable names based on their data type
    following the doctor norm convention. Scalars are single-value variables
    (e.g., time step, iteration count, physical constants).

    Args:
        key (str): Base variable name without prefix (e.g., "dtstep", "iter")
        dtype (str): Variable data type, one of "float", "bool", or "int"

    Returns:
        str: Variable name with doctor norm prefix applied (e.g., "xdtstep", "niter")
        
    Examples:
        >>> var_doctor_norm("dtstep", "float")
        "xdtstep"
        >>> var_doctor_norm("use_ice", "bool")
        "luse_ice"
        >>> var_doctor_norm("iter", "int")
        "niter"
    """
    match dtype:
        case "float":
            return f"x{key}"
        case "bool":
            return f"l{key}"
        case "int":
            return f"n{key}"
