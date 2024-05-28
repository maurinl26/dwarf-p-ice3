# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from datetime import timedelta
from functools import cached_property
from itertools import repeat
from typing import Dict

from gt4py.storage import from_array, ones

from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from ifs_physics_common.framework.storage import managed_temporary_storage
from ifs_physics_common.utils.typingx import NDArrayLikeDict, PropertyDict

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
        return {
            "sigqsat": {"grid": (I, J, K), "units": ""},
            "exn": {"grid": (I, J, K), "units": ""},
            "exnref": {"grid": (I, J, K), "units": ""},
            "rhodref": {"grid": (I, J, K), "units": ""},
            "pabs": {"grid": (I, J, K), "units": ""},
            "sigs": {"grid": (I, J, K), "units": ""},
            "cf_mf": {"grid": (I, J, K), "units": ""},
            "rc_mf": {"grid": (I, J, K), "units": ""},
            "ri_mf": {"grid": (I, J, K), "units": ""},
            "th": {"grid": (I, J, K), "units": ""},
            "rv": {"grid": (I, J, K), "units": ""},
            "rc": {"grid": (I, J, K), "units": ""},
            "rr": {"grid": (I, J, K), "units": ""},
            "ri": {"grid": (I, J, K), "units": ""},
            "rs": {"grid": (I, J, K), "units": ""},
            "rg": {"grid": (I, J, K), "units": ""},
            "cldfr": {"grid": (I, J, K), "units": ""},
            "ifr": {"grid": (I, J, K), "units": ""},
            "hlc_hrc": {"grid": (I, J, K), "units": ""},
            "hlc_hcf": {"grid": (I, J, K), "units": ""},
            "hli_hri": {"grid": (I, J, K), "units": ""},
            "hli_hcf": {"grid": (I, J, K), "units": ""},
            "sigrc": {"grid": (I, J, K), "units": ""},
        }

    @cached_property
    def _tendency_properties(self) -> PropertyDict:
        # TODO : sort tendency properties from state
        return {}

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        # TODO : sort diagnostic properties from state
        return {
            "f_ths": {"grid": (I, J, K), "units": ""},
            "f_rcs": {"grid": (I, J, K), "units": ""},
            "f_rrs": {"grid": (I, J, K), "units": ""},
            "f_ris": {"grid": (I, J, K), "units": ""},
            "f_rss": {"grid": (I, J, K), "units": ""},
            "f_rvs": {"grid": (I, J, K), "units": ""},
            "f_rgs": {"grid": (I, J, K), "units": ""},
        }

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
            state_filter = {"exnref": state["exnref"], "th_t": state["th"]}

            diags_filter = {
                key.split("_")[1]: out_diagnostics[key]
                for key in [
                    "f_ths",
                    "f_rcs",
                    "f_rrs",
                    "f_ris",
                    "f_rss",
                    "f_rvs",
                    "f_rgs",
                ]
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
                **state_filter,
                **diags_filter,
                **temporaries_filter,
                dt=timestep.total_seconds(),
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )

            ############## IceAdjust - State ##########################
            state_ice_adjust = {
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
            }

            diags_ice_adjust = {
                key.split("_")[1]: out_diagnostics[key]
                for key in [
                    "f_ths",
                    "f_rvs",
                    "f_rcs",
                    "f_ris",
                ]
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
            logging.info("Loading src_1d GlobalTable")
            src_1D = from_array(src_1d, backend=self.gt4py_config.backend)

            # Timestep
            logging.info("Launching ice_adjust")
            self.ice_adjust(
                **state_ice_adjust,
                **diags_ice_adjust,
                **temporaries_ice_adjust,
                src_1d=src_1D,
                dt=timestep.total_seconds(),
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )
