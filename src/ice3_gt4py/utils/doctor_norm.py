# -*- coding: utf-8 -*-
def field_doctor_norm(key, dtype):
    if dtype == "float":
        fortran_key = f"p{key}"
    elif dtype == "bool":
        fortran_key = f"{key}"
    return fortran_key


def var_doctor_norm(key, dtype):
    if dtype == "float":
        fortran_key = f"x{key}"
    elif dtype == "bool":
        fortran_key = f"l{key}"
    elif dtype == "int":
        fortran_key = f"n{key}"
    return fortran_key
