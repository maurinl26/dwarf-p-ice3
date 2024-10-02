# -*- coding: utf-8 -*-
import fmodpy
import numpy as np
import logging
from functools import cached_property
from typing import TYPE_CHECKING
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array
from ice3_gt4py.initialisation.utils import initialize_field

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

        logging.info(f"fields_in tht : {self.fields_in['ptht'].mean()}")

        self.generate_gt4py_state()

        logging.info(f"fields_in tht : {self.fields_in['ptht'].mean()}")
        logging.info(f"state_gt4py tht : {self.state_gt4py['tht'][...].mean()}")

        # aro_filter stands for the parts before 'call ice_adjust' in aro_adjust.f90
        stencils_fortran_directory = "./src/ice3_gt4py/stencils_fortran/"
        self.mode_ice4_rrhong = fmodpy.fimport(
            stencils_fortran_directory + "/mode_ice4_rrhong.F90"
        )

    @cached_property
    def dims(self):
        return {"kproma": KPROMA, "ksize": KSIZE}

    @cached_property
    def fields_mapping(self):
        return {
            "ldcompute": "ldcompute",
            "pexn": "exn",
            "plsfact": "ls_fact",
            "plvfact": "lv_fact",
            "pt": "t",
            "ptht": "tht",
            "prrt": "rr_t",
            "prrhong_mr": "rrhong_mr",
        }

    @cached_property
    def fields_in(self):
        return {
            "ldcompute": np.ones((self.dims["kproma"]), dtype=np.int32),
            "pexn": np.array(np.random.rand(self.dims["kproma"]), "f", order="F"),
            "plvfact": np.array(np.random.rand(self.dims["kproma"]), "f", order="F"),
            "plsfact": np.array(np.random.rand(self.dims["kproma"]), "f", order="F"),
            "pt": np.array(np.random.rand(self.dims["kproma"]), "f", order="F"),
            "prrt": np.array(np.random.rand(self.dims["kproma"]), "f", order="F"),
            "ptht": np.array(np.random.rand(self.dims["kproma"]), "f", order="F"),
        }

    @cached_property
    def fields_out(self):
        return {"prrhong_mr": np.zeros((self.dims["kproma"]), "f", order="F")}

    def generate_gt4py_state(self):

        self.state_gt4py = allocate_state_ice4_rrhong(
            self.computational_grid, self.gt4py_config
        )
        fields = {**self.fields_in, **self.fields_out}
        for key_fortran, key_gt4py in self.fields_mapping.items():
            initialize_field(
                self.state_gt4py[key_gt4py], 2 * fields[key_fortran][:, np.newaxis]
            )

    def test(self):
        """Call fortran stencil"""

        logging.info(
            f"Input field, rrhong_mr (fortran) : {self.fields_out['prrhong_mr'].mean()}"
        )
        logging.info(
            f"Input field, rrhong_mr (gt4py) : {self.state_gt4py['rrhong_mr'][...].mean()}"
        )

        logging.info(
            f"Input field, ls_fact (fortran) : {self.fields_in['plsfact'].mean()}"
        )
        logging.info(
            f"Input field, ls_fact (gt4py) : {self.state_gt4py['ls_fact'][...].mean()}"
        )

        prrhong_mr = self.mode_ice4_rrhong.mode_ice4_rrhong.ice4_rrhong(
            **self.dims, **self.externals, **self.fields_in, **self.fields_out
        )

        self.ice4_rrhong_gt4py(
            **self.state_gt4py,
        )

        logging.info(f"Mean Fortran : {prrhong_mr.mean()}")

        field_gt4py = self.state_gt4py["rrhong_mr"][...]
        logging.info(f"Mean GT4Py {field_gt4py.mean()}")


if __name__ == "__main__":

    KPROMA, KSIZE = 50, 50

    # TODO : set in env values
    backend = "gt:cpu_ifirst"
    rebuild = True
    validate_args = True

    phyex = Phyex(program="AROME")

    logging.info("Initializing grid ...")

    # Grid has only 1 dimension since fields are packed in fortran version
    grid = ComputationalGrid(50, 1, 1)
    dt = datetime.timedelta(seconds=1)

    ######## Backend and gt4py config #######
    logging.info(f"With backend {backend}")
    gt4py_config = GT4PyConfig(
        backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=False
    )

    logging.info("Calling ice4_rrhong with dicts")

    TestIce4RRHONG(
        computational_grid=grid, gt4py_config=gt4py_config, phyex=phyex
    ).test()
