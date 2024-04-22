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
from ifs_physics_common.utils.f2py import ported_method


from ice3_gt4py.phyex_common.phyex import Phyex


class Ice4Tendencies(ImplicitTendencyComponent):
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

        # Tendencies
        self.ice4_nucleation = compile_stencil("ice4_nucleation", externals)
        self.nucleation_post_processing = compile_stencil(
            "nucleation_post_processing", externals
        )

        self.ice4_rrhong = compile_stencil("ice4_rrhong", externals)
        self.ice4_rrhong_post_processing = compile_stencil(
            "ice4_rrhong_post_processing", externals
        )

        self.ice4_rimltc = compile_stencil("ice4_rimltc", externals)
        self.ice4_rimltc_post_processing = compile_stencil("rimltc_post_processing")

        self.ice4_increment_update = compile_stencil("ice4_increment_update", externals)
        self.ice4_derived_fields = compile_stencil("ice4_derived_fields", externals)

        # TODO: add ice4_compute_pdf
        self.ice4_slope_parameters = compile_stencil("ice4_slope_parameters", externals)

        self.ice4_slow = compile_stencil("ice4_slow", externals)
        self.ice4_warm = compile_stencil("ice4_warm", externals)

        self.ice4_fast_rs = compile_stencil("ice4_fast_rs", externals)

        self.ice4_fast_rg_pre_processing = compile_stencil(
            "ice4_fast_rg_pre_processing", externals
        )
        self.ice4_fast_rg = compile_stencil("ice4_fast_rg", externals)

        self.ice4_fast_ri = compile_stencil("ice4_fast_ri", externals)

        self.ice4_tendencies_update = compile_stencil(
            "ice4_tendencies_update", externals
        )

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "tht": {"grid": (I, J, K), "units": ""},
            "pabs_t": {"grid": (I, J, K), "units": ""},
            "rhodref": {"grid": (I, J, K), "units": ""},
            "exn": {"grid": (I, J, K), "units": ""},
            "t": {"grid": (I, J, K), "units": ""},
            "ls_fact": {"grid": (I, J, K), "units": ""},
            "rv_t": {"grid": (I, J, K), "units": ""},
        }

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
            *repeat(((I, J, K), "bool"), 1),
            *repeat(((I, J, K), "float"), 20),
            gt4py_config=self.gt4py_config,
        ) as (ldcompute, rv_heni_mr, usw, w1, w2, ssi):
            inputs = {
                name.split("_", maxsplit=1)[1]: state[name]
                for name in self.input_properties
            }

            temporaries = {
                "ldcompute": ldcompute,
                "rv_heni_mr": rv_heni_mr,
                "usw": usw,
                "w1": w1,
                "w2": w2,
                "ssi": ssi,
            }

            self.ice4_nucleation(**inputs, **temporaries)
            # self.nucleation_post_processing()

            # self.ice4_rrhong()
            # self.ice4_rrhong_post_processing()

            # self.ice4_rimltc()
            # self.ice4_rimltc_post_processing()

            # self.ice4_increment_update()
            # self.ice4_derived_fields()

            # TODO: add ice4_compute_pdf
            # self.ice4_slope_parameters()

            # self.ice4_slow()
            # self.ice4_warm()

            # self.ice4_fast_rs()

            # self.ice4_fast_rg_pre_processing()
            # self.ice4_fast_rg()

            # self.ice4_fast_ri()

            # self.ice4_tendencies_update()


class Ice4StepLimiter(ImplicitTendencyComponent):
    """Component for step computation"""

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

        # Stencil collections
        self.tmicro_init = compile_stencil("ice4_stepping_tmicro_init", externals)
        self.tsoft_init = compile_stencil("ice4_stepping_tsoft_init", externals)
        self.ice4_stepping_heat = compile_stencil("ice4_stepping_heat", externals)
        self.ice4_step_limiter = compile_stencil("step_limiter", externals)
        self.ice4_mixing_ratio_step_limiter = compile_stencil(
            "mixing_ratio_step_limiter", externals
        )
        self.ice4_state_update = compile_stencil("state_update", externals)
        self.external_tendencies_update = compile_stencil(
            "external_tendencies_update", externals
        )

        # Component for tendency update
        self.ice4_tendencies = Ice4Tendencies(
            self.computational_grid, self.gt4py_config, phyex
        )

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

    @ported_method(
        from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
        from_line=214,
        to_line=438,
    )
    def array_call(self, state: NDArrayLikeDict, timestep: timedelta):

        # Translation note : Ice4Stepping is implemented assuming PARAMI%XTSTEP_TS = 0
        #                   l225 to l229 omitted
        #                   l334 to l341 omitted
        #                   l174 to l178 omitted

        # l214 to l221
        self.tmicro_init()

        # TODO while t < TSTEP
        # TODO while ldcompute

        # l249 to l254
        self.ice4_stepping_heat()

        # l261
        self.ice4_tendencies()

        # l290 to l332
        self.ice4_step_limiter()

        # l346 to l388
        self.ice4_mixing_ratio_step_limiter()

        # l394 to l404
        self.ice4_state_update()

        # TODO : next loop
        # l440 to l452
        self.external_tendencies_update()
