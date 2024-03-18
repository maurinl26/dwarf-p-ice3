# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import asdict
import logging
from typing import TYPE_CHECKING, Dict, Literal
import sys
from datetime import timedelta

from gt4py.storage import ones
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from ifs_physics_common.framework.stencil import compile_stencil

from ice3_gt4py.initialisation.state import allocate_state
from ice3_gt4py.phyex_common.phyex import Phyex
from tests.utils.config import BACKEND_LIST

if TYPE_CHECKING:
    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid
    from ifs_physics_common.utils.typingx import DataArrayDict


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


def initialize_state_with_constant(
    state: DataArrayDict, C: float, gt4py_config: GT4PyConfig
) -> None:

    keys = [
        "f_sigqsat",
        "f_exnref",  # ref exner pression
        "f_exn",
        "f_tht",
        "f_rhodref",
        "f_pabs",  # absolute pressure at t
        "f_sigs",  # Sigma_s at time t
        "f_cf_mf",  # convective mass flux fraction
        "f_rc_mf",  # convective mass flux liquid mixing ratio
        "f_ri_mf",
        "f_th",
        "f_rv",
        "f_rc",
        "f_rr",
        "f_ri",
        "f_rs",
        "f_rg",
    ]

    for name in keys:
        logging.debug(f"{name}, {state[name].shape}")
        state[name][...] = C * ones(state[name].shape, backend=gt4py_config.backend)


def get_state_with_constant(
    computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig, c: float
) -> DataArrayDict:
    """All arrays are filled with a constant between 0 and 1.

    Args:
        computational_grid (ComputationalGrid): _description_
        gt4py_config (GT4PyConfig): _description_

    Returns:
        DataArrayDict: _description_
    """
    state = allocate_state(computational_grid, gt4py_config=gt4py_config)
    initialize_state_with_constant(state, c, gt4py_config)
    return state


def externals_to_stencil(
    stencil_collection_name: str,
    config_externals: Dict,
    stencil_collection_keys: Dict,
):
    tmp_externals = {}

    externals_list = stencil_collection_keys[stencil_collection_name]
    print(config_externals)
    for key in externals_list:

        try:
            tmp_externals.update(config_externals[key])
        except:
            logging.error(f"No key found for {key} in configuration externals")

    return tmp_externals


def main(
    backend: Literal[
        "numpy",
        "cuda",
        "gt:gpu",
        "gt:cpu_ifirst",
        "gt:cpu_kfirst",
        "dace:cpu",
        "dace:gpu",
    ]
):
    """Run rain_ice processes in order

    nucleation >> rrhong >> rimltc >> riming conversion >> pdf computation >> slow cold >> warm >> fast rs >> fast rg >> fast ri

    Args:
        backend (Literal): targeted backend for code generation
    """

    nx = 100
    ny = 1
    nz = 90

    cprogram = "AROME"
    phyex_config = Phyex(cprogram)
    logging.info(f"backend {backend}")
    gt4py_config = GT4PyConfig(
        backend=backend, rebuild=False, validate_args=False, verbose=True
    )

    grid = ComputationalGrid(nx, ny, nz)
    dt = timedelta(seconds=1)

    processes = [
        "ice4_nucleation",
        "ice4_fast_rg",
        "ice4_fast_rs",
        "ice4_fast_ri",
        "ice4_rimltc",
        "ice4_rrhong",
        "ice4_slow",
        "ice4_warm",
    ]

    stencil_collections_with_externals = {
        "ice4_nucleation": [
            "tt",
            "v_rtmin",
            "alpi",
            "betai",
            "gami",
            "alpw",
            "betaw",
            "gamw",
            "epsilo",
            "nu20",
            "alpha2",
            "beta2",
            "nu10",
            "beta1",
            "alpha1",
            "mnu0",
            "lfeedbackt",
        ],
        "ice4_fast_rg": [
            "Ci",
            "Cl",
            "tt",
            "lvtt",
            "i_rtmin",
            "r_rtmin",
            "g_rtmin",
            "s_rtmin",
            "icfrr",
            "rcfri",
            "exicfrr",
            "exrcfri",
            "cexvt",
            "crflimit",
            "cxg",
            "dg",
            "fcdryg",
            "fidryg",
            "colexig",
            "colig",
            "ldsoft",
            "estt",
            "Rv",
            "cpv",
            "lmtt",
            "o0depg",
            "o1depg",
            "ex0depg",
            "ex1depg",
            "levlimit",
            "alpi",
            "betai",
            "gami",
            "lwetgpost",
            "lnullwetg",
            "epsilo",
            "frdryg",
            "lbdryg1",
            "lbdryg2",
            "lbsdryg3",
        ],
        "ice4_fast_rs": [
            "s_rtmin",
            "c_rtmin",
            "epsilo",
            "levlimit",
            "alpi",
            "betai",
            "gami",
            "tt",
            "cpv",
            "lvtt",
            "estt",
            "Rv",
            "o0deps",
            "o1deps",
            "ex0deps",
            "ex1deps",
            "lmtt",
            "Cl",
            "Ci",
            "tt",
            "crimss",
            "excrimss",
            "cexvt",
            "crimsg",
            "excrimsg",
            "srimcg",
            "exsrimcg",
            "srimcg3",
            "srimcg2",
            "exsrimcg2",
            "fracss",
            "cxs",
            "lbraccs1",
            "lbraccs2",
            "lbraccs3",
            "lbsaccr1",
            "lbsaccr2",
            "lbsaccr3",
            "bs",
            "fsaccrg",
        ],
        "ice4_fast_ri": [
            "c_rtmin",
            "i_rtmin",
            "lbi",
            "lbexi",
            "o0depi",
            "o2depi",
            "di",
        ],
        "ice4_rimltc": ["tt", "lfeedbackt"],
        "ice4_rrhong": ["r_rtmin", "tt", "lfeedbackt"],
        "ice4_slow": [
            "tt",
            "c_rtmin",
            "v_rtmin",
            "s_rtmin",
            "i_rtmin",
            "g_rtmin",
            "hon",
            "alpha3",
            "beta3",
            "o0deps",
            "ex0deps",
            "ex1deps",
            "o1deps",
            "fiaggs",
            "colexis",
            "exiaggs",
            "cexvt",
            "xcriauti",
            "acriauti",
            "bcriauti",
            "timauti",
            "texauti",
            "o0depg",
            "ex0depg",
            "o1depg",
            "ex1depg",
        ],
        "ice4_warm": [
            "subg_rc_rr_accr",
            "subg_rr_evap_",
            "c_rtmin",
            "r_rtmin",
            "timautc",
            "criautc",
            "cexvt",
            "fcaccr",
            "excaccr",
            "alpw",
            "betaw",
            "gamw",
            "Rv",
            "Cl",
            "lvtt",
            "tt",
            "cpv",
            "o0evar",
            "ex0evar",
            "o1evar",
            "ex1evar",
            "cpd",
            "epsilo",
        ],
    }

    config_externals = phyex_config.to_externals()

    for process in processes:

        externals = externals_to_stencil(
            f"{process}", config_externals, stencil_collections_with_externals
        )
        print(externals)

        try:
            stencil_collection = compile_stencil(f"{process}", gt4py_config, externals)

        except:
            logging.error(
                f"Compilation failed for process {process} and backend {backend}"
            )


if __name__ == "__main__":

    for backend in BACKEND_LIST:
        main(backend)
