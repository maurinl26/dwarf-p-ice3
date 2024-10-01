# -*- coding: utf-8 -*-
from functools import partial
import fmodpy
import numpy as np
import logging
from ctypes import c_int32, c_double

from typing import TYPE_CHECKING

from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.grid import ComputationalGrid
import datetime

from ice3_gt4py.phyex_common.phyex import Phyex
from ice3_gt4py.phyex_common.tables import src_1d
from gt4py.storage import from_array

from typing import Literal, Tuple

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
from ifs_physics_common.utils.typingx import (
    DataArray,
    NDArrayLikeDict,
)


def allocate_state_condensation(
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
    allocate_f = partial(_allocate, grid_id=(I, J, K), units="", dtype="float")
    allocate_h = partial(_allocate, grid_id=(I, J, K - 1 / 2), units="", dtype="float")
    allocate_ij = partial(_allocate, grid_id=(I, J), units="", dtype="float")
    allocate_i_ij = partial(_allocate, grid_id=(I, J), units="", dtype="int")
    allocate_i = partial(_allocate, grid_id=(I, J, K), units="", dtype="int")

    return {
        "sigqsat": allocate_f(),
        "pabs": allocate_f(),
        "pt": allocate_f(),
        "sigs": allocate_f(),
        "rv_in": allocate_f(),
        "ri_in": allocate_f(),
        "rc_in": allocate_f(),
        "rv_out": allocate_f(),
        "rc_out": allocate_f(),
        "ri_out": allocate_f(),
        "cldfr": allocate_f(),
        "sigrc": allocate_f(),
        "cph": allocate_f(),
        "lv": allocate_f(),
        "ls": allocate_f(),
        "inq1": allocate_i(),
    }


class TestCondensation(ComputationalGridComponent):
    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
        phyex: Phyex,
    ) -> None:
        super().__init__(
            computational_grid=computational_grid, gt4py_config=gt4py_config
        )

        externals_gt4py = phyex.to_externals()

        # aro_filter stands for the parts before 'call ice_adjust' in aro_adjust.f90
        self.condensation_gt4py = self.compile_stencil("condensation", externals_gt4py)

        self.externals = {
            "xrv": c_double(externals_gt4py["RD"]),
            "xrd": c_double(externals_gt4py["RV"]),
            "xalpi": c_double(externals_gt4py["ALPI"]),
            "xbetai": c_double(externals_gt4py["BETAI"]),
            "xgami": c_double(externals_gt4py["GAMI"]),
            "xalpw": c_double(externals_gt4py["ALPW"]),
            "xbetaw": c_double(externals_gt4py["BETAW"]),
            "xgamw": c_double(externals_gt4py["GAMW"]),
            "hcondens": "cb02",
            "hlambda3": "cb",
            "ouseri": False,
            "osigmas": True,
            "ocnd2": False,
            "lstatnw": False,
        }

        self.dims = {
            "nijt": NIJT,
            "nkt": NKT,
            "nktb": NKTB,
            "nkte": NKTE,
            "nijb": NIJB,
            "nije": NIJE,
        }
        self.generate_state()

        self.fortran_directory = "./src/ice3_gt4py/stencils_fortran/"
        self.condensation = fmodpy.fimport(
            self.fortran_directory + "mode_condensation.F90",
        )

    def generate_state(self):

        nijt, nkt = self.dims["nijt"], self.dims["nkt"]

        self.fields = {
            "ppabs": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "pt": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "prv_in": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "prc_in": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "pri_in": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "psigs": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "psigrc": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "pcldfr": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "psigqsat": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "plv": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "pls": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "pcph": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
        }

        self.outs = {
            "prc_out": np.zeros((nijt, nkt), order="F"),
            "pri_out": np.zeros((nijt, nkt), order="F"),
            "prv_out": np.zeros((nijt, nkt), order="F"),
        }

        state = allocate_state_condensation(self.computational_grid, self.gt4py_config)

        self.state_gt4py = {
            "sigqsat": self.fields["psigqsat"],
            "pabs": self.fields["ppabs"],
            "t": self.fields["pt"],
            "sigs": self.fields["psigs"],
            "rv_in": self.fields["prv_in"],
            "rc_in": self.fields["prc_in"],
            "ri_in": self.fields["pri_in"],
            "rv_out": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "ri_out": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "rc_out": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "cldfr": self.fields["pcldfr"],
            "sigrc": self.fields["psigrc"],
            "cph": self.fields["pcph"],
            "lv": self.fields["plv"],
            "ls": self.fields["pls"],
            "inq1": np.asfortranarray(np.random.rand(nijt, nkt), dtype=int),
        }

    def test(self):
        """Call fortran stencil"""
        nijt, nkt = self.dims["nijt"], self.dims["nkt"]

        t = np.zeros((nijt.value, nkt.value), dtype=c_int32, order="F")
        prv_out = np.zeros((nijt.value, nkt.value), dtype=c_int32, order="F")
        pri_out = np.zeros((nijt.value, nkt.value), dtype=c_int32, order="F")
        prc_out = np.zeros((nijt.value, nkt.value), dtype=c_int32, order="F")

        t, prv_out, pri_out, prc_out = self.condensation.condensation(
            **self.dims, **self.externals, **self.fields
        )


if __name__ == "__main__":

    NIJT = 50
    NKT = 15
    NKTB = 15
    NKTE = 15
    NIJB = 0
    NIJE = 50

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

    TestCondensation(
        computational_grid=grid, gt4py_config=gt4py_config, phyex=phyex
    ).test()
