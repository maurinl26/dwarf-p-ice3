# -*- coding: utf-8 -*-
def field_doctor_norm(key: str, dtype: str) -> str:
    """Add the doctor norm predicate for fields

    Args:
        key (str): field name
        dtype (str): field type

    Returns:
        str: field fortran name
    """
    match dtype:
        case "float":
            return f"p{key}"
        case "bool":
            return f"l{key}"


def var_doctor_norm(key: str, dtype: str) -> str:
    """Add the doctor norm predicate for scalar

    Args:
        key (str): scalar name
        dtype (str): scalar type

    Returns:
        str: name with doctor norm
    """
    match dtype:
        case "float":
            return f"x{key}"
        case "bool":
            return f"l{key}"
        case "int":
            return f"n{key}"
