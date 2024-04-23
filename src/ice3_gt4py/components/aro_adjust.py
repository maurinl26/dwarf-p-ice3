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
        return {
            "sigqsat": {
                "grid": (I, J, K),
                "units": "",
            },  # coeff applied to qsat variance
            "exnref": {"grid": (I, J, K), "units": ""},  # ref exner pression
            "exn": {"grid": (I, J, K), "units": ""},
            "rhodref": {"grid": (I, J, K), "units": ""},  #
            "pabs": {"grid": (I, J, K), "units": ""},  # absolute pressure at t
            "sigs": {"grid": (I, J, K), "units": ""},  # Sigma_s at time t
            "cmf": {
                "grid": (I, J, K),
                "units": "",
            },  # convective mass flux fraction
            "rc_mf": {
                "grid": (I, J, K),
                "units": "",
            },  # convective mass flux liquid mixing ratio
            "ri_mf": {"grid": (I, J, K), "units": ""},
            "tht": {"grid": (I, J, K), "units": ""},
        }

    @cached_property
    def _tendency_properties(self) -> PropertyDict:
        return {
            "ths": {"grid": (I, J, K), "units": ""},
            "rvs": {"grid": (I, J, K), "units": ""},  # PRS(1)
            "rcs": {"grid": (I, J, K), "units": ""},  # PRS(2)
            "ris": {"grid": (I, J, K), "units": ""},  # PRS(4)
            "rrs": {"grid": (I, J, K), "units": ""},
            "rss": {"grid": (I, J, K), "units": ""},
            "rgs": {"grid": (I, J, K), "units": ""},
            "th": {"grid": (I, J, K), "units": ""},  # ZRS(0)
            "rv": {"grid": (I, J, K), "units": ""},  # ZRS(1)
            "rc": {"grid": (I, J, K), "units": ""},  # ZRS(2)
            "rr": {"grid": (I, J, K), "units": ""},  # ZRS(3)
            "ri": {"grid": (I, J, K), "units": ""},  # ZRS(4)
            "rs": {"grid": (I, J, K), "units": ""},  # ZRS(5)
            "rg": {"grid": (I, J, K), "units": ""},  # ZRS(6)
            "cldfr": {"grid": (I, J, K), "units": ""},
        }

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        return {
            "ifr": {"grid": (I, J, K), "units": ""},
            "hlc_hrc": {"grid": (I, J, K), "units": ""},
            "hlc_hcf": {"grid": (I, J, K), "units": ""},
            "hli_hri": {"grid": (I, J, K), "units": ""},
            "hli_hcf": {"grid": (I, J, K), "units": ""},
        }

    @cached_property
    def _temporaries(self) -> PropertyDict:
        return {
            "rt": {
                "grid": (I, J, K),
                "units": "",
            },  # work array for total water mixing ratio
            "pv": {"grid": (I, J, K), "units": ""},  # thermodynamics
            "piv": {"grid": (I, J, K), "units": ""},  # thermodynamics
            "qsl": {"grid": (I, J, K), "units": ""},  # thermodynamics
            "qsi": {"grid": (I, J, K), "units": ""},
            "frac_tmp": {"grid": (I, J, K), "units": ""},  # ice fraction
            "cond_tmp": {"grid": (I, J, K), "units": ""},  # condensate
            "a": {"grid": (I, J, K), "units": ""},  # related to computation of Sig_s
            "sbar": {"grid": (I, J, K), "units": ""},
            "sigma": {"grid": (I, J, K), "units": ""},
            "q1": {"grid": (I, J, K), "units": ""},
            "lv": {"grid": (I, J, K), "units": ""},
            "ls": {"grid": (I, J, K), "units": ""},
            "cph": {"grid": (I, J, K), "units": ""},
            "criaut": {"grid": (I, J, K), "units": ""},
            "sigrc": {"grid": (I, J, K), "units": ""},
            "rv_tmp": {"grid": (I, J, K), "units": ""},
            "ri_tmp": {"grid": (I, J, K), "units": ""},
            "rc_tmp": {"grid": (I, J, K), "units": ""},
            "t_tmp": {"grid": (I, J, K), "units": ""},
        }

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
            *repeat(((I, J, K), "float"), 23),
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
            sigrc,
            rv_tmp,
            ri_tmp,
            rc_tmp,
            t_tmp,
            cor_tmp,
            cph_tmp,
            inq1,
        ):

            ############## AroFilter - Compilation ################
            logging.info(f"Compilation for aro_filter")
            # aro_filter = compile_stencil("aro_filter", gt4py_config, externals)

            ############## AroFilter - State ####################
            state_filter = {
                "exnref": ones(
                    self.computational_grid.grids[(I, J, K)].shape,
                    backend=self.gt4py_config.backend,
                ),
                "tht": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "ths": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "rcs": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "rrs": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "ris": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "rvs": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "rgs": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "rss": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
            }

            temporaries_filter = {
                "t_tmp": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "ls_tmp": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "lv_tmp": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "cph_tmp": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "cor_tmp": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
            }

            # timestep
            dt = 1.0
            self.aro_filter(dt=dt, **state_filter, **temporaries_filter)

            ############## IceAdjust - Compilation ####################
            # logging.info(f"Compilation for ice_adjust")
            # ice_adjust = compile_stencil("ice_adjust", gt4py_config, externals)

            ############## IceAdjust - State ##########################
            state_ice_adjust = {
                "sigqsat": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "exn": state_filter["exnref"],
                "exnref": state_filter["exnref"],
                "rhodref": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "pabs": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "sigs": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "cmf": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "rc_mf": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "ri_mf": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "th": state_filter["tht"],
                "rv": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "rc": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "ri": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "rr": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "rs": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "rg": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "ths": state_filter["ths"],
                "rvs": state_filter["rvs"],
                "rcs": state_filter["rcs"],
                "ris": state_filter["ris"],
                "cldfr": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "ifr": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "hlc_hrc": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "hlc_hcf": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "hli_hri": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "hli_hcf": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "sigrc": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "rv_tmp": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "ri_tmp": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "rc_tmp": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "t_tmp": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "cph": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "lv": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "ls": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "criaut": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "rt": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "pv": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "piv": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "qsl": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "qsi": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "frac_tmp": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "cond_tmp": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "a": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "sbar": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "sigma": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "q1": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                ),
                "inq1": ones(
                    (self.computational_grid.grids[(I, J, K)].shape),
                    backend=self.gt4py_config.backend,
                    dtype=np.int64,
                ),
            }

            # Global Table
            logging.info("GlobalTable")
            # src_1d = zeros(shape=(34,),  backend=self.gt4py_config.backend, dtype=np.float64),
            src_1D = from_array(src_1d, backend=self.gt4py_config.backend)

            # Timestep
            dt = 1.0
            self.ice_adjust(dt=dt, src_1d=src_1D, **state_ice_adjust)
