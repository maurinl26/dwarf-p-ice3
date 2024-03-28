# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import timedelta
from functools import cached_property
from itertools import repeat
from typing import Dict

from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.storage import managed_temporary_storage
from ifs_physics_common.utils.typingx import NDArrayLikeDict, PropertyDict

from ice3_gt4py.phyex_common.phyex import Phyex


class Ice4Stepping(ImplicitTendencyComponent):
    """Implicit Tendency Component calling
    ice_adjust : saturation adjustment of temperature and mixing ratios

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

        self.ice4_nucleation = self.compile_stencil("ice4_nucleation")
        self.ice4_rrhong = self.compile_stencil("ice4_rrhong")
        self.ice4_rimltc = self.compile_stencil("ice4_rimltc")
        self.ice4_slow = self.compile_stencil("ice4_slow")
        self.ice4_warm = self.compile_stencil("ice4_warm")
        self.ice4_fast_rs = self.compile_stencil("ice4_fast_rs")
        self.ice4_fast_rg = self.compile_stencil("ice4_fast_rg")
        self.ice4_fast_ri = compile_stencil("ice4_fast_ri")

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {}

    @cached_property
    def _tendency_properties(self) -> PropertyDict:
        return {}

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        return {}

    @cached_property
    def _temporaries(self) -> PropertyDict:
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
            *repeat(((I, J, K), "float"), 20),
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
        ):
            inputs = {
                name.split("_", maxsplit=1)[1]: state[name]
                for name in self.input_properties
            }
            tendencies = {
                name.split("_", maxsplit=1)[1]: out_tendencies[name]
                for name in self.tendency_properties
            }
            diagnostics = {
                name.split("_", maxsplit=1)[1]: out_diagnostics[name]
                for name in self.diagnostic_properties
            }
            temporaries = {
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
                "lv": lv,
                "ls": ls,
                "cph": cph,
                "criaut": criaut,
                "sigrc": sigrc,
                "rv_tmp": rv_tmp,
                "ri_tmp": ri_tmp,
                "rc_tmp": rc_tmp,
                "t_tmp": t_tmp,
            }

            self.ice_adjust(
                **inputs,
                **tendencies,
                **diagnostics,
                **temporaries,
                dt=timestep.total_seconds(),
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )


def stepping():
    """Stepping function for microphysical processes"""

    NotImplemented


def ice4_tendencies():
    """Run rain_ice processes in order.
    Update estimates of values given tendencies computed by each process

    nucleation >> rrhong >> rimltc >> riming conversion >> pdf computation >> slow cold >> warm >> fast rs >> fast rg >> fast ri
    """

    NotImplemented
