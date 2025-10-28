# -*- coding: utf-8 -*-
def field_doctor_norm(key: str, dtype: str) -> str:
    """Add the doctor norm predicate for fields

    Args:
        key (str): field name
        dtype (str): field type

    Returns:
        str: field fortran name
    """
    if dtype == "float":
        fortran_key = f"p{key}"
    elif dtype == "bool":
        fortran_key = f"{key}"
    return fortran_key


def var_doctor_norm(key: str, dtype: str) -> str:
    """Add the doctor norm predicate for scalar

    Args:
        key (str): scalar name
        dtype (str): scalar type

    Returns:
        str: name with doctor norm
    """
    if dtype == "float":
        fortran_key = f"x{key}"
    elif dtype == "bool":
        fortran_key = f"l{key}"
    elif dtype == "int":
        fortran_key = f"n{key}"
    return fortran_key
