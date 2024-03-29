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
        ) as ():
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
            temporaries = {}

            # TODO fill fields
            self.ice4_nucleation()
            self.nucleation_post_processing()

            self.ice4_rrhong()
            self.ice4_rrhong_post_processing()

            self.ice4_rimltc()
            self.ice4_rimltc_post_processing()

            self.ice4_increment_update()
            self.ice4_derived_fields()

            # TODO: add ice4_compute_pdf
            self.ice4_slope_parameters()

            self.ice4_slow()
            self.ice4_warm()

            self.ice4_fast_rs()

            self.ice4_fast_rg_pre_processing()
            self.ice4_fast_rg()

            self.ice4_fast_ri()

            self.ice4_tendencies_update()


def ice4_stepping(externals):
    """Stepping function from Phyex / ice4"""

    tmicro_init = compile_stencil("ice4_stepping_tmicro_init", externals)

    # DO WHILE ZTIME < PTSTEP
    # while tmicro.any():
    tsoft_init = compile_stencil("ice4_stepping_tsoft_init", externals)

    # DO WHILE LLCOMPUTE
    # while ldcompute.any():
    ice4_stepping_heat = compile_stencil("ice4_stepping_heat", externals)
    lsoft = False

    # Tendencies stencils
    # ice4_tendencies(externals)

    # TODO : add possibility to update external tendencies
    ice4_step_limiter = compile_stencil("step_limiter", externals)
    ice4_mixing_ratio_step_limiter = compile_stencil(
        "mixing_ratio_step_limiter", externals
    )

    ice4_state_update = compile_stencil("state_update", externals)

    lsoft = True
    # end do while

    # Out of loop:
    # if lext_tend
    external_tendencies_update = compile_stencil(
        "external_tendencies_update", externals
    )
