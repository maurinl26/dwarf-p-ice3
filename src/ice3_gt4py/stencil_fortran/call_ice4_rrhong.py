# -*- coding: utf-8 -*-
import fmodpy
import numpy as np
from ifs_physics_common.framework.stencil import compile_stencil

mode_ice4_rrhong = fmodpy.fimport("mode_ice4_rrhong.F90")

XTT = 0
XRTMIN = 10e-5
LFEEDBACKT = False
KPROMA = 50
KSIZE = 50
LDCOMPUTE = np.asfortranarray(np.random.randint(2, size=KPROMA))


PEXN = np.asfortranarray(np.random.rand(KSIZE))
PLVFACT = np.asfortranarray(np.random.rand(KSIZE))
PLSFACT = np.asfortranarray(np.random.rand(KSIZE))
PT = np.asfortranarray(np.random.rand(KSIZE))
PRRT = np.asfortranarray(np.random.rand(KSIZE))
PTHT = np.asfortranarray(np.random.rand(KSIZE))
PRRHONG_MR = np.asfortranarray(np.random.rand(KSIZE))


mode_ice4_rrhong.mode_ice4_rrhong.ice4_rrhong(
    XTT,
    XRTMIN,
    LFEEDBACKT,
    KPROMA,
    KSIZE,
    LDCOMPUTE,
    PEXN,
    PLVFACT,
    PLSFACT,
    PT,
    PRRT,
    PTHT,
    PRRHONG_MR,
)

externals = {
    "XTT": XTT,
    "XRTMIN": XRTMIN,
    "LFEEDBACKT": LFEEDBACKT,
}


ice4_rrhong = compile_stencil("ice4_rrhong", gt4py_config, externals)
