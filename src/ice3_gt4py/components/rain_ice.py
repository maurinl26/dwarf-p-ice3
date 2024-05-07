# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import timedelta
from functools import cached_property
from itertools import repeat

from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from ifs_physics_common.framework.storage import managed_temporary_storage
from ifs_physics_common.utils.typingx import NDArrayLikeDict, PropertyDict
from ifs_physics_common.utils.f2py import ported_method


from ice3_gt4py.components.ice4_stepping import Ice4Stepping
from ice3_gt4py.phyex_common.phyex import Phyex


class RainIce(ImplicitTendencyComponent):
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
        self.ice4_stepping = Ice4Stepping(
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
            ###### Thermo ############
            # l401 to l430 in rain_ice.F90

            ##### Sedimentation ######
            # NOT LSEDIM_AFTER
            # l438 to 449

            ###### Initial values saving #####

            ###### Nucleation #################
            # l486 to l511 (outside of ldmicro mask)

            ###### Precip fraction ############
            # l525 to l571
            # Compute pdf
            # Rain Fr vert : TODO : create stencil

            ###### Tendencies #################
            # l577 to 621
            # (Remove packing operations)

            ###### Elec tendencies ############
            # l634 to l708
            # omitted because not present in AROME

            ###### Total tendencies ###########
            # l725 to l760

            ###### Negative corrections #######
            # l774 to 804
            # (No budget)
            # ice4_correct_negativities

            ###### Sedimentation ##############
            # l811 to l841
            # ice4_sedimentation
            # rain_fr_vert : TODO : create stencil

            ###### Fog deposition ##############
            # l848 to l862
            # TODO : create fog stencil
