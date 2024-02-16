# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import asdict
from datetime import timedelta
from functools import cached_property
from itertools import repeat
from typing import Dict
import logging

from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.utils.typingx import PropertyDict, NDArrayLikeDict
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import managed_temporary_storage
from ice3_gt4py.phyex_common.phyex import Phyex


class AroFilter(ImplicitTendencyComponent):
    """Implicit Tendency Component calling aro_filter stencil

    aro_filter isolates filter for negative values in AroAdjust

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

        externals = {}
        externals.update(asdict(phyex.nebn))
        externals.update(asdict(phyex.cst))
        externals.update(asdict(phyex.param_icen))
        externals.update(
            {
                "nrr": 6,
                "criautc": 0,
                "acriauti": 0,
                "bcriauti": 0,
                "criauti": 0,
            }
        )

        # aro_filter stands for the parts before 'call ice_adjust' in aro_adjust.f90
        self.aro_filter = self.compile_stencil("aro_filter", externals)

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "f_exnref": {"grid": (I, J, K), "units": ""},  # ref exner pression
            "f_tht": {"grid": (I, J, K), "units": ""},
        }

    @cached_property
    def _tendency_properties(self) -> PropertyDict:
        return {
            "f_ths": {"grid": (I, J, K), "units": ""},
            "f_rvs": {"grid": (I, J, K), "units": ""},  # PRS(1)
            "f_rcs": {"grid": (I, J, K), "units": ""},  # PRS(2)
            "f_ris": {"grid": (I, J, K), "units": ""},  # PRS(4)
            "f_rrs": {"grid": (I, J, K), "units": ""},
            "f_rss": {"grid": (I, J, K), "units": ""},
            "f_rgs": {"grid": (I, J, K), "units": ""},
        }

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        return {}

    @cached_property
    def _temporaries(self) -> PropertyDict:
        return {
            "lv": {"grid": (I, J, K), "units": ""},
            "ls": {"grid": (I, J, K), "units": ""},
            "t_tmp": {"grid": (I, J, K), "units": ""},
            "cor_tmp": {"grid": (I, J, K), "units": ""},
            "cph_tmp": {"grid": (I, J, K), "units": ""},
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
            *repeat(((I, J, K), "float"), 5),
            gt4py_config=self.gt4py_config,
        ) as (
            lv,
            ls,
            t_tmp,
            cor_tmp,
            cph_tmp,
        ):
            logging.debug(f"State : {state.keys()}")
            input_filter = {
                name.split("_", maxsplit=1)[1]: state[name]
                for name in self.input_properties
            }

            logging.debug(f"Out tendencies : {out_tendencies.keys()}")
            tendencies_filter = {
                name.split("_", maxsplit=1)[1]: out_tendencies[name]
                for name in self.tendency_properties
            }

            temporaries_filter = {
                "cor_tmp": cor_tmp,
                "cph_tmp": cph_tmp,
                "t_tmp": t_tmp,
                "lv_tmp": lv,
                "ls_tmp": ls,
            }

            self.aro_filter(
                **input_filter,
                **tendencies_filter,
                **temporaries_filter,
                dt=timestep.total_seconds(),
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )
