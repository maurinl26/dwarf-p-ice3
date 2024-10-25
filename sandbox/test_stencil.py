# -*- coding: utf-8 -*-
import fmodpy
import numpy as np
import logging
from functools import cached_property
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.grid import ComputationalGrid

from ice3_gt4py.initialisation.utils import initialize_field

from ice3_gt4py.phyex_common.phyex import Phyex
import ice3_gt4py.stencils

from ice3_gt4py.utils.allocate_state import allocate_b, allocate_f
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.utils.typingx import (
    NDArrayLikeDict,
)

from repro.test_config import default_epsilon, default_gt4py_config, phyex, test_grid



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


class TestStencil(ComputationalGridComponent):
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
        self.stencil_gt4py = self.compile_stencil("ice4_rrhong", externals)

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
        self.stencil_fortran = fmodpy.fimport(
            stencils_fortran_directory + "/mode_ice4_rrhong.F90"
        ).mode_ice4_rrhong.ice4_rrhong

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

    def allocate_state_gt4py(self):
        ...

    def generate_gt4py_state(self):
        """_summary_"""

        self.state_gt4py = allocate_state_ice4_rrhong(
            self.computational_grid, self.gt4py_config
        )
        fields = {**self.fields_in, **self.fields_out}
        for key_fortran, key_gt4py in self.fields_mapping.items():
            initialize_field(
                self.state_gt4py[key_gt4py], 2 * fields[key_fortran][:, np.newaxis]
            )

    def test(self, tol):
        """Call fortran stencil"""

        # Call fortran
        out_fields = self.stencil_fortran(
            **self.dims, **self.externals, **self.fields_in, **self.fields_out
        )

        # Call python
        self.stencil_gt4py(
            **self.state_gt4py,
        )

        ### Comparison
        for field_key in self.fields_out.keys():
            gt4py_key = self.fields_mapping[field_key]
            fortran_mean = out_fields[field_key].mean()
            gt4py_mean = self.state_gt4py[gt4py_key][...].mean()

            assert abs(fortran_mean - gt4py_mean) < tol




if __name__ == "__main__":
    

    logging.info("Calling ice4_rrhong with dicts")
    TestStencil(
        computational_grid=test_grid, gt4py_config=default_gt4py_config, phyex=phyex
    ).test(tol=10e-8)
