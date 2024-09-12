# -*- coding: utf-8 -*-
"""This Python code is an automatically generated wrapper
for Fortran code made by 'fmodpy'. The original documentation
for the Fortran source code follows.

!MNH_LIC Copyright 1994-2021 CNRS, Meteo-France and Universite Paul Sabatier
!MNH_LIC This is part of the Meso-NH software governed by the CeCILL-C licence
!MNH_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt
!MNH_LIC for details. version 1.
!-----------------------------------------------------------------
"""

import os
import ctypes
import platform
import numpy

# --------------------------------------------------------------------
#               CONFIGURATION
#
_verbose = True
_fort_compiler = "gfortran"
_shared_object_name = "mode_ice4_rrhong." + platform.machine() + ".so"
_this_directory = os.path.dirname(os.path.abspath(__file__))
_path_to_lib = os.path.join(_this_directory, _shared_object_name)
_compile_options = ["-fPIC", "-shared", "-O3"]
_ordered_dependencies = ["mode_ice4_rrhong.F90", "mode_ice4_rrhong_c_wrapper.f90"]
_symbol_files = []  #
# --------------------------------------------------------------------
#               AUTO-COMPILING
#
# Try to import the prerequisite symbols for the compiled code.
for _ in _symbol_files:
    _ = ctypes.CDLL(os.path.join(_this_directory, _), mode=ctypes.RTLD_GLOBAL)
# Try to import the existing object. If that fails, recompile and then try.
try:
    # Check to see if the source files have been modified and a recompilation is needed.
    if max(
        max(
            [0]
            + [
                os.path.getmtime(os.path.realpath(os.path.join(_this_directory, _)))
                for _ in _symbol_files
            ]
        ),
        max(
            [0]
            + [
                os.path.getmtime(os.path.realpath(os.path.join(_this_directory, _)))
                for _ in _ordered_dependencies
            ]
        ),
    ) > os.path.getmtime(_path_to_lib):
        print()
        print(
            "WARNING: Recompiling because the modification time of a source file is newer than the library.",
            flush=True,
        )
        print()
        if os.path.exists(_path_to_lib):
            os.remove(_path_to_lib)
        raise NotImplementedError(f"The newest library code has not been compiled.")
    # Import the library.
    clib = ctypes.CDLL(_path_to_lib)
except:
    # Remove the shared object if it exists, because it is faulty.
    if os.path.exists(_shared_object_name):
        os.remove(_shared_object_name)
    # Compile a new shared object.
    _command = (
        [_fort_compiler]
        + _ordered_dependencies
        + _compile_options
        + ["-o", _shared_object_name]
    )
    if _verbose:
        print("Running system command with arguments")
        print("  ", " ".join(_command))
    # Run the compilation command.
    import subprocess

    subprocess.check_call(_command, cwd=_this_directory)
    # Import the shared object file as a C library with ctypes.
    clib = ctypes.CDLL(_path_to_lib)
# --------------------------------------------------------------------


class mode_ice4_rrhong:
    """"""

    # ----------------------------------------------
    # Wrapper for the Fortran subroutine ICE4_RRHONG

    def ice4_rrhong(
        self,
        xtt,
        xrtmin,
        lfeedbackt,
        kproma,
        ksize,
        ldcompute,
        pexn,
        plvfact,
        plsfact,
        pt,
        prrt,
        ptht,
        prrhong_mr=None,
    ):
        """!!
        !!**  PURPOSE
        !!    -------
        !!      Computes the RRHONG process
        !!
        !!    AUTHOR
        !!    ------
        !!      S. Riette from the splitting of rain_ice source code (nov. 2014)
        !!
        !!    MODIFICATIONS
        !!    -------------
        !!
        !
        !
        !*      0. DECLARATIONS
        !          ------------
        !
        ! USE MODD_CST,            ONLY: CST_t
        ! USE MODD_PARAM_ICE_n,      ONLY: PARAM_ICE_t
        ! USE MODD_RAIN_ICE_DESCR_n, ONLY: RAIN_ICE_DESCR_t
        ! USE YOMHOOK , ONLY : LHOOK, DR_HOOK, JPHOOK"""

        # Setting up "xtt"
        if type(xtt) is not ctypes.c_float:
            xtt = ctypes.c_float(xtt)

        # Setting up "xrtmin"
        if type(xrtmin) is not ctypes.c_float:
            xrtmin = ctypes.c_float(xrtmin)

        # Setting up "lfeedbackt"
        if type(lfeedbackt) is not ctypes.c_int:
            lfeedbackt = ctypes.c_int(lfeedbackt)

        # Setting up "kproma"
        if type(kproma) is not ctypes.c_int:
            kproma = ctypes.c_int(kproma)

        # Setting up "ksize"
        if type(ksize) is not ctypes.c_int:
            ksize = ctypes.c_int(ksize)

        # Setting up "ldcompute"
        if (
            (not issubclass(type(ldcompute), numpy.ndarray))
            or (not numpy.asarray(ldcompute).flags.f_contiguous)
            or (not (ldcompute.dtype == numpy.dtype(ctypes.c_int)))
        ):
            import warnings

            warnings.warn(
                "The provided argument 'ldcompute' was not an f_contiguous NumPy array of type 'ctypes.c_int' (or equivalent). Automatically converting (probably creating a full copy)."
            )
            ldcompute = numpy.asarray(ldcompute, dtype=ctypes.c_int, order="F")
        ldcompute_dim_1 = ctypes.c_long(ldcompute.shape[0])

        # Setting up "pexn"
        if (
            (not issubclass(type(pexn), numpy.ndarray))
            or (not numpy.asarray(pexn).flags.f_contiguous)
            or (not (pexn.dtype == numpy.dtype(ctypes.c_float)))
        ):
            import warnings

            warnings.warn(
                "The provided argument 'pexn' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy)."
            )
            pexn = numpy.asarray(pexn, dtype=ctypes.c_float, order="F")
        pexn_dim_1 = ctypes.c_long(pexn.shape[0])

        # Setting up "plvfact"
        if (
            (not issubclass(type(plvfact), numpy.ndarray))
            or (not numpy.asarray(plvfact).flags.f_contiguous)
            or (not (plvfact.dtype == numpy.dtype(ctypes.c_float)))
        ):
            import warnings

            warnings.warn(
                "The provided argument 'plvfact' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy)."
            )
            plvfact = numpy.asarray(plvfact, dtype=ctypes.c_float, order="F")
        plvfact_dim_1 = ctypes.c_long(plvfact.shape[0])

        # Setting up "plsfact"
        if (
            (not issubclass(type(plsfact), numpy.ndarray))
            or (not numpy.asarray(plsfact).flags.f_contiguous)
            or (not (plsfact.dtype == numpy.dtype(ctypes.c_float)))
        ):
            import warnings

            warnings.warn(
                "The provided argument 'plsfact' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy)."
            )
            plsfact = numpy.asarray(plsfact, dtype=ctypes.c_float, order="F")
        plsfact_dim_1 = ctypes.c_long(plsfact.shape[0])

        # Setting up "pt"
        if (
            (not issubclass(type(pt), numpy.ndarray))
            or (not numpy.asarray(pt).flags.f_contiguous)
            or (not (pt.dtype == numpy.dtype(ctypes.c_float)))
        ):
            import warnings

            warnings.warn(
                "The provided argument 'pt' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy)."
            )
            pt = numpy.asarray(pt, dtype=ctypes.c_float, order="F")
        pt_dim_1 = ctypes.c_long(pt.shape[0])

        # Setting up "prrt"
        if (
            (not issubclass(type(prrt), numpy.ndarray))
            or (not numpy.asarray(prrt).flags.f_contiguous)
            or (not (prrt.dtype == numpy.dtype(ctypes.c_float)))
        ):
            import warnings

            warnings.warn(
                "The provided argument 'prrt' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy)."
            )
            prrt = numpy.asarray(prrt, dtype=ctypes.c_float, order="F")
        prrt_dim_1 = ctypes.c_long(prrt.shape[0])

        # Setting up "ptht"
        if (
            (not issubclass(type(ptht), numpy.ndarray))
            or (not numpy.asarray(ptht).flags.f_contiguous)
            or (not (ptht.dtype == numpy.dtype(ctypes.c_float)))
        ):
            import warnings

            warnings.warn(
                "The provided argument 'ptht' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy)."
            )
            ptht = numpy.asarray(ptht, dtype=ctypes.c_float, order="F")
        ptht_dim_1 = ctypes.c_long(ptht.shape[0])

        # Setting up "prrhong_mr"
        if prrhong_mr is None:
            prrhong_mr = numpy.zeros(shape=(kproma), dtype=ctypes.c_float, order="F")
        elif (
            (not issubclass(type(prrhong_mr), numpy.ndarray))
            or (not numpy.asarray(prrhong_mr).flags.f_contiguous)
            or (not (prrhong_mr.dtype == numpy.dtype(ctypes.c_float)))
        ):
            import warnings

            warnings.warn(
                "The provided argument 'prrhong_mr' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy)."
            )
            prrhong_mr = numpy.asarray(prrhong_mr, dtype=ctypes.c_float, order="F")
        prrhong_mr_dim_1 = ctypes.c_long(prrhong_mr.shape[0])

        # Call C-accessible Fortran wrapper.
        clib.c_ice4_rrhong(
            ctypes.byref(xtt),
            ctypes.byref(xrtmin),
            ctypes.byref(lfeedbackt),
            ctypes.byref(kproma),
            ctypes.byref(ksize),
            ctypes.byref(ldcompute_dim_1),
            ctypes.c_void_p(ldcompute.ctypes.data),
            ctypes.byref(pexn_dim_1),
            ctypes.c_void_p(pexn.ctypes.data),
            ctypes.byref(plvfact_dim_1),
            ctypes.c_void_p(plvfact.ctypes.data),
            ctypes.byref(plsfact_dim_1),
            ctypes.c_void_p(plsfact.ctypes.data),
            ctypes.byref(pt_dim_1),
            ctypes.c_void_p(pt.ctypes.data),
            ctypes.byref(prrt_dim_1),
            ctypes.c_void_p(prrt.ctypes.data),
            ctypes.byref(ptht_dim_1),
            ctypes.c_void_p(ptht.ctypes.data),
            ctypes.byref(prrhong_mr_dim_1),
            ctypes.c_void_p(prrhong_mr.ctypes.data),
        )

        # Return final results, 'INTENT(OUT)' arguments only.
        return xtt.value, xrtmin.value, lfeedbackt.value, prrhong_mr


mode_ice4_rrhong = mode_ice4_rrhong()
