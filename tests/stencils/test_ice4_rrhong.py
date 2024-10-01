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
from functools import partial
from ice3_gt4py.phyex_common.phyex import Phyex
import ice3_gt4py.stencils

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
        phyex: Phyex,
    ) -> None:
        super().__init__(
            computational_grid=computational_grid, gt4py_config=gt4py_config
        )

        externals = phyex.to_externals()
        self.ice4_rrhong_gt4py = self.compile_stencil("ice4_rrhong", externals)

        self.externals = {
            "xtt": externals["TT"],
            "r_rtmin": externals["R_RTMIN"],
            "lfeedbackt": externals["LFEEDBACKT"],
        }

        self.dims = {"kproma": 50, "ksize": 15}

        self.generate_state()

        # aro_filter stands for the parts before 'call ice_adjust' in aro_adjust.f90

        self.mode_ice4_rrhong = fmodpy.fimport(
            "./src/ice3_gt4py/stencils_fortran/mode_ice4_rrhong.F90"
        )

    def generate_state(self):

        state = allocate_state_ice4_rrhong(self.computational_grid, self.gt4py_config)

        kproma, ksize = self.dims["kproma"], self.dims["ksize"]

        self.fields = {
            "ldcompute": np.asfortranarray(np.ones((kproma, ksize), dtype=np.int32)),
            "pexn": np.asfortranarray(np.random.rand(kproma, ksize)),
            "plvfact": np.asfortranarray(np.random.rand(kproma, ksize)),
            "plsfact": np.asfortranarray(np.random.rand(kproma, ksize)),
            "pt": np.asfortranarray(np.random.rand(kproma, ksize)),
            "prrt": np.asfortranarray(np.random.rand(kproma, ksize)),
            "ptht": np.asfortranarray(np.random.rand(kproma, ksize)),
            "prrhong_mr": np.asfortranarray(np.random.rand(kproma, ksize)),
        }

        self.state_gt4py = {
            "ldcompute": self.fields["ldcompute"],
            "exn": self.fields["pexn"],
            "ls_fact": self.fields["plsfact"],
            "lv_fact": self.fields["plvfact"],
            "t": self.fields["pt"],
            "tht": self.fields["ptht"],
            "rr_t": self.fields["prrt"],
            "rrhong_mr": self.fields["prrhong_mr"],
        }

    def test_feedbackt(self):
        """Call fortran stencil"""

        logging.info(f"Input field, rrhong_mr : {self.fields['prrhong_mr'].mean()}")

        self.ice4_rrhong = self.mode_ice4_rrhong.mode_ice4_rrhong.ice4_rrhong(
            **self.dims, **self.externals, **self.fields
        )

        field_fortran = self.fields["prrhong_mr"]
        logging.info(f"Mean Fortran {field_fortran.mean()}")

        self.ice4_rrhong_gt4py(
            **self.state_gt4py,
        )

        field_gt4py = self.state_gt4py["rrhong_mr"][...]
        logging.info(f"Mean GT4Py {field_gt4py.mean()}")


if __name__ == "__main__":

    # TODO : set in env values
    backend = "gt:cpu_ifirst"
    rebuild = True
    validate_args = True

    phyex = Phyex(program="AROME")

    # TODO : set init with grid
    logging.info("Initializing grid ...")
    grid = ComputationalGrid(50, 1, 15)
    dt = datetime.timedelta(seconds=1)

    ######## Backend and gt4py config #######
    logging.info(f"With backend {backend}")
    gt4py_config = GT4PyConfig(
        backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=False
    )

    logging.info("Calling ice4_rrhong with dicts")

    TestIce4RRHONG(
        computational_grid=grid, gt4py_config=gt4py_config, phyex=phyex
    ).test_feedbackt()
