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

    stencil_collections_with_externals = {
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
    }

    for backend, stencil_collection in itertools.product(backends, stencil_collections):

        gt4py_config = GT4PyConfig(
            backend=backend, rebuild=True, validate_args=True, verbose=True
        )

        logging.info(f"Compile {stencil_collection} stencil_collection on {backend}")

        externals = {
            "lvtt": 0,
            "lstt": 0,
            "tt": 0,
            "subg_mf_pdf": 0,
            "subg_cond": 0,
            "cpd": 0,
            "cpv": 0,
            "Cl": 0,
            "Ci": 0,
            "tt": 0,
            "alpw": 0,
            "betaw": 0,
            "gamw": 0,
            "alpi": 0,
            "betai": 0,
            "gami": 0,
            "Rd": 0,
            "Rv": 0,
            "frac_ice_adjust": 0,
            "tmaxmix": 0,
            "tminmix": 0,
            "criautc": 0,
            "tstep": 0,
            "criauti": 0,
            "acriauti": 0,
            "bcriauti": 0,
            "nrr": 6,
        }

        try:

            stencil = compile_stencil(stencil_collection, gt4py_config, externals)

            logging.info(f"Compilation succeeded for {stencil_collection} on {backend}")

        except:
            logging.error(f"Compilation failed for {stencil_collection} on {backend}")
