# -*- coding: utf-8 -*-
from ifs_physics_common.framework.stencil import compile_stencil
from ice3_gt4py.drivers.config import default_python_config
from ifs_physics_common.framework.config import GT4PyConfig
import sys
import logging
import itertools

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


if __name__ == "__main__":

    backends = ["numpy", "gt:cpu_ifirst", "gt:gpu", "cuda", "dace:cpu", "dace:gpu"]
    stencil_collections = [
        "aro_filter",
        "ice_adjust",
        "ice4_nucleation",
        "ice4_fast_rg",
        "ice4_fast_rs",
        "ice4_fast_ri",
        "ice4_nucleation",
        "ice4_rimltc",
        "ice4_rrhong",
        "ice4_slow",
        "ice4_warm",
    ]

    externals = {
        "ice4_nucleation": {
            "tt": 0,
            "v_rtmin": 0,
            "alpi": 0,
            "betai": 0,
            "alpw": 0,
            "betaw": 0,
            "nu20": 0,
            "alpha2": 0,
            "beta2": 0,
            "nu10": 0,
            "beta1": 0,
            "alpha1": 0,
            "mnu0": 0,
            "lfeedbackt": 0,
        },
        "ice4_fast_rg": {
            "Ci": 0,
            "Cl": 0,
            "tt": 0,
            "lvtt": 0,
            "i_rtmin": 0,
            "r_rtmin": 0,
            "g_rtmin": 0,
            "s_rtmin": 0,
            "v_rtmin": 0,
            "icfrr": 0,
            "rcfri": 0,
            "exicfrr": 0,
            "exrcfri": 0,
            "cxg": 0,
            "dg": 0,
            "fcdryg": 0,
            "fidryg": 0,
            "colexig": 0,
            "colig": 0,
            "ldsoft": 0,
            "Rv": 0,
            "cpv": 0,
            "lmtt": 0,
            "o0depg": 0,
            "o1depg": 0,
            "ex0depg": 0,
            "ex1depg": 0,
            "levlimit": 0,
            "alpi": 0,
            "betai": 0,
            "gami": 0,
            "lwetgpost": 0,
            "lnullwetg": 0,
            "epsilo": 0,
            "frdryg": 0,
            "lbdryg1": 0,
            "lbdryg2": 0,
            "lbsdryg3": 0,
        },
        "ice4_fast_rs": {
            "s_rtmin": 0,
            "c_rtmin": 0,
            "epsilo": 0,
            "levlimit": 0,
            "alpi": 0,
            "betai": 0,
            "gami": 0,
            "tt": 0,
            "cpv": 0,
            "Cl": 0,
            "lvtt": 0,
            "estt": 0,
            "Rv": 0,
            "o0deps": 0,
            "o1deps": 0,
            "ex0deps": 0,
            "ex1deps": 0,
            "lmtt": 0,
            "Cl": 0,
            "Ci": 0,
            "tt": 0,
            "crimss": 0,
            "excrimss": 0,
            "cexvt": 0,
            "crimsg": 0,
            "excrimsg": 0,
            "srimcg": 0,
            "exsrimcg": 0,
            "srimcg3": 0,
            "srimcg2": 0,
            "exsrimcg2": 0,
            "fracss": 0,
            "cxs": 0,
            "lbraccs1": 0,
            "lbraccs2": 0,
            "lbraccs3": 0,
            "lbsaccr1": 0,
            "lbsaccr2": 0,
            "lbsaccr3": 0,
            "bs": 0,
            "fsaccrg": 0,
        },
        "ice4_fast_ri": {
            "c_rtmin": 0,
            "i_rtmin": 0,
            "lbi": 0,
            "lbexi": 0,
            "o0depi": 0,
            "o2depi": 0,
            "di": 0,
        },
        "ice4_rimltc": {
            "tt": 0,
            "lfeedbackt": 0,
        },
        "ice4_rrhong": {"r_rtmin": 0, "tt": 0, "lfeedbackt": 0},
        "ice4_slow": {
            "tt": 0,
            "c_rtmin": 0,
            "v_rtmin": 0,
            "s_rtmin": 0,
            "i_rtmin": 0,
            "g_rtmin": 0,
            "hon": 0,
            "alpha3": 0,
            "beta3": 0,
            "o0deps": 0,
            "ex0deps": 0,
            "ex1deps": 0,
            "o1deps": 0,
            "fiaggs": 0,
            "colexis": 0,
            "exiaggs": 0,
            "cexvt": 0,
            "xcriauti": 0,
            "acriauti": 0,
            "bcriauti": 0,
            "timauti": 0,
            "texauti": 0,
            "o0depg": 0,
            "ex0depg": 0,
            "o1depg": 0,
            "ex1depg": 0,
        },
        "ice4_warm": {
            "subg_rc_rr_accr": 0,
            "subg_rr_evap": 0,
            "c_rtmin": 0,
            "r_rtmin": 0,
            "timautc": 0,
            "criautc": 0,
            "cexvt": 0,
            "fcaccr": 0,
            "excaccr": 0,
            "alpw": 0,
            "betaw": 0,
            "gamw": 0,
            "Rv": 0,
            "Cl": 0,
            "lvtt": 0,
            "tt": 0,
            "cpv": 0,
            "o0evar": 0,
            "ex0evar": 0,
            "o1evar": 0,
            "ex1evar": 0,
            "cpd": 0,
            "epsilo": 0,
        },
    }

    for backend, stencil_collection in itertools.product(backends, stencil_collections):

        gt4py_config = GT4PyConfig(
            backend=backend, rebuild=True, validate_args=True, verbose=True
        )

        logging.info(f"Compile {stencil_collection} stencil_collection on {backend}")

        try:

            stencil = compile_stencil(
                stencil_collection, gt4py_config, externals[stencil_collection]
            )

            logging.info(f"Compilation succeeded for {stencil_collection} on {backend}")

        except:
            logging.error(f"Compilation failed for {stencil_collection} on {backend}")
