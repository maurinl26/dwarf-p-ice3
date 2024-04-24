# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import timedelta
from functools import cached_property
from itertools import repeat
from typing import Dict

from gt4py.storage import from_array, ones

from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from ifs_physics_common.framework.storage import managed_temporary_storage
from gt4py.storage import zeros
from ifs_physics_common.utils.typingx import NDArrayLikeDict, PropertyDict
import numpy as np

from ice3_gt4py.phyex_common.phyex import Phyex
import sys
from ice3_gt4py.phyex_common.tables import src_1d

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


class AroAdjust(ImplicitTendencyComponent):
    """Implicit Tendency Component calling sequentially
    - aro_filter : negativity filters
    - ice_adjust : saturation adjustment of temperature and mixing ratios

    aro_filter stencil is aro_adjust.F90 in PHYEX, from l210 to l366
    ice_adjust stencil is ice_adjust.F90 in PHYEX

    """

    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
        phyex: Phyex,
        *,
        enable_checks: bool = True,
    ) -> None:
        super().__init__(
            computational_grid, enable_checks=enable_checks, gt4py_config=gt4py_config
        )

        externals = phyex.to_externals()

        # aro_filter stands for the parts before 'call ice_adjust' in aro_adjust.f90
        self.aro_filter = self.compile_stencil("aro_filter", externals)

        # ice_adjust stands for ice_adjust.f90
        self.ice_adjust = self.compile_stencil("ice_adjust", externals)

    @cached_property
    def _input_properties(self) -> PropertyDict:
        # TODO : sort input properties from state
        return {}

    @cached_property
    def _tendency_properties(self) -> PropertyDict:
        # TODO : sort tendency properties from state
        return {}

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        # TODO : sort diagnostic properties from state
        return {}

    @cached_property
    def _temporaries(self) -> PropertyDict:
        # TODO : writout temporaries
        return {}

    def array_call(
        self,
        state: NDArrayLikeDict,
        timestep: timedelta,
        out_tendencies: NDArrayLikeDict,
        out_diagnostics: NDArrayLikeDict,
        overwrite_tendencies: Dict[str, bool],
    ) -> None:
        with managed_temporary_storage(
            self.computational_grid,
            *repeat(((I, J, K), "float"), 21),
            ((I, J, K), "int"),
            gt4py_config=self.gt4py_config,
        ) as (
            rt,
            pv,
            piv,
            qsl,
            qsi,
            frac_tmp,
            cond_tmp,
            a,
            sbar,
            sigma,
            q1,
            lv,
            ls,
            cph,
            criaut,
            rv_tmp,
            ri_tmp,
            rc_tmp,
            t_tmp,
            cor_tmp,
            cph_tmp,
            inq1,
        ):

            ############## AroFilter - State ####################
            state_filter = {
                **{
                    key: state[key]
                    for key in [
                        "exnref",
                        "tht",
                    ]
                },
                # Sources
                **{
                    key: state[key]
                    for key in ["ths", "rcs", "rrs", "ris", "rss", "rvs", "rgs"]
                },
            }

            temporaries_filter = {
                "t_tmp": t_tmp,
                "ls_tmp": ls,
                "lv_tmp": lv,
                "cph_tmp": cph_tmp,
                "cor_tmp": cor_tmp,
            }

            logging.info("Launching AroFilter")
            # timestep
            self.aro_filter(
                dt=timestep.total_seconds, **state_filter, **temporaries_filter
            )

            ############## IceAdjust - State ##########################
            state_ice_adjust = {
                **{
                    key: state[key]
                    for key in [
                        "sigqsat",
                        "exn",
                        "exnref",
                        "rhodref",
                        "pabs",
                        "sigs",
                        "cf_mf",
                        "rc_mf",
                        "ri_mf",
                        "th",
                        "rv",
                        "rc",
                        "rr",
                        "ri",
                        "rs",
                        "rg",
                        "cldfr",
                        "ifr",
                        "hlc_hrc",
                        "hlc_hcf",
                        "hli_hri",
                        "hli_hcf",
                        "sigrc",
                    ]
                },
                # Sources
                **{
                    key: state[key]
                    for key in [
                        "ths",
                        "rvs",
                        "rcs",
                        "ris",
                    ]
                },
            }

            temporaries_ice_adjust = {
                "rv_tmp": rv_tmp,
                "ri_tmp": ri_tmp,
                "rc_tmp": rc_tmp,
                "t_tmp": t_tmp,
                "cph": cph,
                "lv": lv,
                "ls": ls,
                "criaut": criaut,
                "rt": rt,
                "pv": pv,
                "piv": piv,
                "qsl": qsl,
                "qsi": qsi,
                "frac_tmp": frac_tmp,
                "cond_tmp": cond_tmp,
                "a": a,
                "sbar": sbar,
                "sigma": sigma,
                "q1": q1,
                "inq1": inq1,
            }

            # Global Table
            logging.info("GlobalTable")
            # src_1d = zeros(shape=(34,),  backend=self.gt4py_config.backend, dtype=np.float64),
            src_1D = from_array(src_1d, backend=self.gt4py_config.backend)

            # Timestep
            logging.info("Launching ice_adjust")
            self.ice_adjust(
                dt=timestep.total_seconds,
                src_1d=src_1D,
                **state_ice_adjust,
                **temporaries_ice_adjust,
            )


if __name__ == "__main__":

    BACKEND = "gt:cpu_kfirst"
    from ifs_physics_common.framework.stencil import compile_stencil

    ################### Grid #################
    logging.info("Initializing grid ...")
    nx = 100
    ny = 1
    nz = 90
    grid = ComputationalGrid(nx, ny, nz)
    dt = timedelta(seconds=1)

    ################## Phyex #################
    logging.info("Initializing Phyex ...")
    cprogram = "AROME"
    phyex_config = Phyex(cprogram)

    externals = phyex_config.to_externals()

    ######## Backend and gt4py config #######
    logging.info(f"With backend {BACKEND}")
    gt4py_config = GT4PyConfig(
        backend=BACKEND, rebuild=True, validate_args=False, verbose=True
    )

    ############## AroFilter - Compilation ################
    logging.info(f"Compilation for aro_filter")
    aro_filter = compile_stencil("aro_filter", gt4py_config, externals)

    ############## IceAdjust - Compilation ####################
    logging.info(f"Compilation for ice_adjust")
    ice_adjust = compile_stencil("ice_adjust", gt4py_config, externals)

    state = {
        "exnref": ones((nx, ny, nz), backend=BACKEND),
        "tht": ones((nx, ny, nz), backend=BACKEND),
        "exn": ones((nx, ny, nz), backend=BACKEND),
        "exnref": ones((nx, ny, nz), backend=BACKEND),
        "rhodref": ones((nx, ny, nz), backend=BACKEND),
        "pabs": ones((nx, ny, nz), backend=BACKEND),
        "sigs": ones((nx, ny, nz), backend=BACKEND),
        "cf_mf": ones((nx, ny, nz), backend=BACKEND),
        "rc_mf": ones((nx, ny, nz), backend=BACKEND),
        "ri_mf": ones((nx, ny, nz), backend=BACKEND),
        "th": ones((nx, ny, nz), backend=BACKEND),
        "rv": ones((nx, ny, nz), backend=BACKEND),
        "rc": ones((nx, ny, nz), backend=BACKEND),
        "ri": ones((nx, ny, nz), backend=BACKEND),
        "rr": ones((nx, ny, nz), backend=BACKEND),
        "rs": ones((nx, ny, nz), backend=BACKEND),
        "rg": ones((nx, ny, nz), backend=BACKEND),
        "sigqsat": ones((nx, ny, nz), backend=BACKEND),
        "cldfr": ones((nx, ny, nz), backend=BACKEND),
        "ifr": ones((nx, ny, nz), backend=BACKEND),
        "hlc_hrc": ones((nx, ny, nz), backend=BACKEND),
        "hlc_hcf": ones((nx, ny, nz), backend=BACKEND),
        "hli_hri": ones((nx, ny, nz), backend=BACKEND),
        "hli_hcf": ones((nx, ny, nz), backend=BACKEND),
        "sigrc": ones((nx, ny, nz), backend=BACKEND),
    }

    # sources
    diagnostics = {
        "ths": ones((nx, ny, nz), backend=BACKEND),
        "rcs": ones((nx, ny, nz), backend=BACKEND),
        "rrs": ones((nx, ny, nz), backend=BACKEND),
        "ris": ones((nx, ny, nz), backend=BACKEND),
        "rvs": ones((nx, ny, nz), backend=BACKEND),
        "rgs": ones((nx, ny, nz), backend=BACKEND),
        "rss": ones((nx, ny, nz), backend=BACKEND),
    }

    ############## AroFilter - State ####################
    state_filter = {
        "exnref": state["exnref"],
        "tht": state["tht"],
        "ths": diagnostics["ths"],
        "rcs": diagnostics["rcs"],
        "rrs": diagnostics["rrs"],
        "ris": diagnostics["ris"],
        "rvs": diagnostics["rvs"],
        "rgs": diagnostics["rgs"],
        "rss": diagnostics["rss"],
    }

    temporaries_filter = {
        "t_tmp": ones((nx, ny, nz), backend=BACKEND),
        "ls_tmp": ones((nx, ny, nz), backend=BACKEND),
        "lv_tmp": ones((nx, ny, nz), backend=BACKEND),
        "cph_tmp": ones((nx, ny, nz), backend=BACKEND),
        "cor_tmp": ones((nx, ny, nz), backend=BACKEND),
    }

    # timestep
    dt = 1.0
    aro_filter(dt=dt, **state_filter, **temporaries_filter)

    ############## IceAdjust - State ##########################
    state_ice_adjust = {
        **{
            key: state[key]
            for key in [
                "sigqsat",
                "exn",
                "exnref",
                "rhodref",
                "pabs",
                "sigs",
                "cf_mf",
                "rc_mf",
                "ri_mf",
                "th",
                "rv",
                "rc",
                "rr",
                "ri",
                "rs",
                "rg",
                "cldfr",
                "ifr",
                "hlc_hrc",
                "hlc_hcf",
                "hli_hri",
                "hli_hcf",
                "sigrc",
            ]
        },
        **{
            key: diagnostics[key]
            for key in [
                "ths",
                "rvs",
                "rcs",
                "ris",
            ]
        },
    }

    temporaries_ice_adjust = {
        "rv_tmp": ones((nx, ny, nz), backend=BACKEND),
        "ri_tmp": ones((nx, ny, nz), backend=BACKEND),
        "rc_tmp": ones((nx, ny, nz), backend=BACKEND),
        "t_tmp": ones((nx, ny, nz), backend=BACKEND),
        "cph": ones((nx, ny, nz), backend=BACKEND),
        "lv": ones((nx, ny, nz), backend=BACKEND),
        "ls": ones((nx, ny, nz), backend=BACKEND),
        "criaut": ones((nx, ny, nz), backend=BACKEND),
        "rt": ones((nx, ny, nz), backend=BACKEND),
        "pv": ones((nx, ny, nz), backend=BACKEND),
        "piv": ones((nx, ny, nz), backend=BACKEND),
        "qsl": ones((nx, ny, nz), backend=BACKEND),
        "qsi": ones((nx, ny, nz), backend=BACKEND),
        "frac_tmp": ones((nx, ny, nz), backend=BACKEND),
        "cond_tmp": ones((nx, ny, nz), backend=BACKEND),
        "a": ones((nx, ny, nz), backend=BACKEND),
        "sbar": ones((nx, ny, nz), backend=BACKEND),
        "sigma": ones((nx, ny, nz), backend=BACKEND),
        "q1": ones((nx, ny, nz), backend=BACKEND),
        "inq1": ones((nx, ny, nz), backend=BACKEND, dtype=np.int64),
    }

    # Global Table
    logging.info("GlobalTable")
    src_1D = from_array(src_1d, backend=BACKEND)

    # Timestep
    dt = 1.0
    ice_adjust(dt=dt, src_1d=src_1D, **state_ice_adjust, **temporaries_ice_adjust)
