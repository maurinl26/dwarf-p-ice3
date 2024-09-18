# -*- coding: utf-8 -*-
import fmodpy
import numpy as np
import logging
from typing import TYPE_CHECKING
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array
import datetime
from ice3_gt4py.initialisation.utils import initialize_field
from functools import partial
import ice3_gt4py.stencils

if TYPE_CHECKING:
    from typing import Literal, Tuple

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
    from ifs_physics_common.utils.typingx import (
        DataArray,
        DataArrayDict,
        NDArrayLikeDict,
    )


##### For tests ####
def allocate_state_ice4_rrhong(
    computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig
) -> NDArrayLikeDict:
    """Allocate field to state keys following type (float, int, bool) and dimensions (2D, 3D).

    Args:
        computational_grid (ComputationalGrid): grid indexes
        gt4py_config (GT4PyConfig): gt4py configuration

    Returns:
        NDArrayLikeDict: dictionnary of field with associated keys for field name
    """

    def _allocate(
        grid_id: Tuple[DimSymbol, ...],
        units: str,
        dtype: Literal["bool", "float", "int"],
    ) -> DataArray:
        return allocate_data_array(
            computational_grid, grid_id, units, gt4py_config=gt4py_config, dtype=dtype
        )

    allocate_b_ij = partial(_allocate, grid_id=(I, J), units="", dtype="bool")
    allocate_b = partial(_allocate, grid_id=(I, J, K), units="", dtype="bool")
    allocate_f = partial(_allocate, grid_id=(I, J, K), units="", dtype="float")
    allocate_h = partial(_allocate, grid_id=(I, J, K - 1 / 2), units="", dtype="float")
    allocate_ij = partial(_allocate, grid_id=(I, J), units="", dtype="float")
    allocate_i_ij = partial(_allocate, grid_id=(I, J), units="", dtype="int")

    return {
        "ldcompute": allocate_b(),
        "exn": allocate_f(),
        "ls_fact": allocate_f(),
        "lv_fact": allocate_f(),
        "t": allocate_f(),
        "tht": allocate_f(),
        "rr_t": allocate_f(),
        "rrhong_mr": allocate_f(),
    }


class TestIce4RRHONG(ComputationalGridComponent):
    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
    ) -> None:
        super().__init__(
            computational_grid=computational_grid, gt4py_config=gt4py_config
        )

        self.externals = {
            "tt": 0,
            "r_rtmin": 10e-5,
            "lfeedbackt": True,
        }

        self.dims = {"kproma": 50, "ksize": 15}

        # aro_filter stands for the parts before 'call ice_adjust' in aro_adjust.f90
        self.ice4_rrhong_gt4py = self.compile_stencil("ice4_rrhong", self.externals)

        self.mode_ice4_rrhong = fmodpy.fimport(
            "./src/ice3_gt4py/stencils_fortran/mode_ice4_rrhong.F90"
        )

    def generate_state(self, computational_grid, gt4py_config):

        state = allocate_state_ice4_rrhong(computational_grid, gt4py_config)

        kproma, ksize = (self.dims["kproma"],)

        fields = {
            "ldcompute": np.asfortranarray(np.ones((kproma, ksize))),
            "pexn": np.asfortranarray(np.random.rand(kproma, ksize)),
            "plvfact": np.asfortranarray(np.random.rand(kproma, ksize)),
            "plsfact": np.asfortranarray(np.random.rand(kproma, ksize)),
            "pt": np.asfortranarray(np.random.rand(kproma, ksize)),
            "prrt": np.asfortranarray(np.random.rand(kproma, ksize)),
            "ptht": np.asfortranarray(np.random.rand(kproma, ksize)),
            "prrhong_mr": np.asfortranarray(np.random.rand(kproma, ksize)),
        }

        state_gt4py = {
            **{
                key: initialize_field(state[key], fields[key])
                for key in [
                    "ldcompute",
                    "exn",
                    "ls_fact",
                    "lv_fact",
                    "t",
                    "tht",
                    "rr_t",
                    "rrhong_mr",
                ]
            }
        }

        return fields, state_gt4py

    def test_feedbackt(self):
        """Call fortran stencil"""

        fortran_fields, gt4py_fields = self.generate_state(
            self.computational_grid, self.gt4py_config
        )

        self.ice4_rrhong = self.mode_ice4_rrhong.mode_ice4_rrhong.ice4_rrhong(
            **self.externals, **self.dims, **fortran_fields  # TODO : self.fields
        )

        self.ice4_rrhong_gt4py(
            **gt4py_fields,
        )


if __name__ == "__main__":

    # TODO : set in env values
    backend = "gt:cpu_ifirst"
    rebuild = True
    validate_args = True

    # TODO : set init with grid
    logging.info("Initializing grid ...")
    grid = ComputationalGrid(50, 1, 15)
    dt = datetime.timedelta(seconds=1)

    ######## Backend and gt4py config #######
    logging.info(f"With backend {backend}")
    gt4py_config = GT4PyConfig(
        backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=True
    )

    logging.info("Calling ice4_rrhong with dicts")

    ice4_rrhong = TestIce4RRHONG(
        computational_grid=grid,
        gt4py_config=gt4py_config,
    )

    # TODO : launch tests

    # ice4_rrhong.test_feedbackt()
