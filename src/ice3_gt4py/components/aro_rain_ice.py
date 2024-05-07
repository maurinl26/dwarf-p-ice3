# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import timedelta
from functools import cached_property
from itertools import repeat

from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from ifs_physics_common.framework.stencil import compile_stencil
from ifs_physics_common.framework.storage import managed_temporary_storage
from ifs_physics_common.utils.typingx import NDArrayLikeDict, PropertyDict
from ifs_physics_common.utils.f2py import ported_method


from ice3_gt4py.components.ice4_stepping import Ice4Stepping
from ice3_gt4py.phyex_common.phyex import Phyex


class AroRainIce(ImplicitTendencyComponent):
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

        externals = phyex.to_externals()

        # Component for tendency update
        self.ice4_tendencies = Ice4Stepping(
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

        with managed_temporary_storage(
            self.computational_grid,
            *repeat(((I, J, K), "bool"), 1),
            *repeat(((I, J, K), "float"), 17),
            gt4py_config=self.gt4py_config,
        ) as ():
            NotImplemented

            ####### Conversions #################
            # l250 to l253 in aro_rain_ice.F90
            # TODO : create stencil or remove ops

            ####### Remove negative values ######
            # l261 to l291
            # TODO : create stencil

            ###### Adjustments ##################
            # l295 to l333
            # TODO : create stencil
            # create stencil for droplet to rain conversion + ice to snow

            # l338 to l352 no budget

            ###### RainIce ######################
            # l365 to l529
            # TODO : add stencil for ldmicro mask computation
