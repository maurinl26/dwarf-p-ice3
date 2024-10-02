# -*- coding: utf-8 -*-
from functools import partial
import fmodpy
import numpy as np
import logging
from ctypes import c_int32, c_double
from functools import cached_property

from typing import TYPE_CHECKING

from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.grid import ComputationalGrid
import datetime
from ice3_gt4py.initialisation.utils import initialize_field


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
        "sigqsat": allocate_ij(),
        "pabs": allocate_f(),
        "sigs": allocate_f(),
        "t": allocate_f(),
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
        self.gt4py_stencil = self.compile_stencil("condensation", externals_gt4py)

        self.externals = {
            "xrv": c_double(externals_gt4py["RD"]),
            "xrd": c_double(externals_gt4py["RV"]),
            "xalpi": c_double(externals_gt4py["ALPI"]),
            "xbetai": c_double(externals_gt4py["BETAI"]),
            "xgami": c_double(externals_gt4py["GAMI"]),
            "xalpw": c_double(externals_gt4py["ALPW"]),
            "xbetaw": c_double(externals_gt4py["BETAW"]),
            "xgamw": c_double(externals_gt4py["GAMW"]),
            "hcondens": 0,
            "hlambda3": 0,
            "lstatnw": 0,
            "ouseri": 0,
            "osigmas": 1,
            "ocnd2": 0,
        }

        # TODO : infer from gri

        self.generate_gt4py_state()

        self.fortran_directory = "./src/ice3_gt4py/stencils_fortran/"
        self.fortran_stencil = fmodpy.fimport(
            self.fortran_directory + "mode_condensation.F90",
        )

    @cached_property
    def dims(self):
        return {
            "nijt": NIJT,
            "nkt": NKT,
            "nktb": NKTB,
            "nkte": NKTE,
            "nijb": NIJB,
            "nije": NIJE,
        }

    @cached_property
    def fields_mapping(self):
        """Map Fortran field name (key) to GT4Py stencil field name (value)"""
        return {
            "psigqsat": "sigqsat",
            "ppabs": "pabs",
            "pt": "t",
            "psigs": "sigs",
            "prv_in": "rv_in",
            "prc_in": "rc_in",
            "pri_in": "ri_in",
            "prv_out": "rv_out",
            "prc_out": "rc_out",
            "pri_out": "ri_out",
            "psigrc": "sigrc",
            "pcldfr": "cldfr",
            "pcph": "cph",
            "plv": "lv",
            "pls": "ls",
        }

    @cached_property
    def fields_in(self):
        """Fields marked as intent(in) in Fortran SUBROUTINE

        Returns:
            _type_: _description_
        """
        nijt, nkt = self.dims["nijt"], self.dims["nkt"]
        return {
            "ppabs": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "pt": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "prv_in": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "prc_in": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "pri_in": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "psigs": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "psigqsat": np.array(np.random.rand(nijt), dtype=c_double, order="F"),
            "plv": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "pls": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
            "pcph": np.array(np.random.rand(nijt, nkt), dtype=c_double, order="F"),
        }

    @cached_property
    def fields_out(self):
        """Fields marked as intent(out) in Fortran SUBROUTINE

        Returns:
            Dict: _description_
        """
        nijt, nkt = self.dims["nijt"], self.dims["nkt"]
        return {
            "prc_out": np.zeros((nijt, nkt), dtype=c_double, order="F"),
            "pri_out": np.zeros((nijt, nkt), dtype=c_double, order="F"),
            "prv_out": np.zeros((nijt, nkt), dtype=c_double, order="F"),
            "pt_out": np.zeros((nijt, nkt), dtype=c_double, order="F"),
            "psigrc": np.zeros((nijt, nkt), dtype=c_double, order="F"),
            "pcldfr": np.zeros((nijt, nkt), dtype=c_double, order="F"),
        }

    def generate_gt4py_state(self):

        fields = {**self.fields_in, **self.fields_out}
        fields.pop("pt_out")
        self.state_gt4py = allocate_state_condensation(
            self.computational_grid, self.gt4py_config
        )
        for fortran_key, gt4py_key in self.fields_mapping.items():
            initialize_field(self.state_gt4py[gt4py_key], fields[fortran_key])

        # Add inq1
        nijt, nkt = self.dims["nijt"], self.dims["nkt"]
        self.state_gt4py = {
            **self.state_gt4py,
            **{
                "inq1": initialize_field(
                    self.state_gt4py["inq1"], np.zeros((nijt, nkt), dtype=int)
                ),
                "src_1d": from_array(src_1d, backend=gt4py_config.backend),
            },
        }

    def test(self):
        """Call fortran stencil"""

        # Run Fortran
        logging.info("Run fortran stencil")
        # pt_out, prv_out, pri_out, prc_out, pcldfr, psigrc
        self.fortran_stencil.mode_condensation.condensation(
            **self.dims, **self.externals, **self.fields_in, **self.fields_out
        )

        # Run gt4py
        logging.info("Run gt4py stencil")
        self.gt4py_stencil(**self.state_gt4py)


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
    grid = ComputationalGrid(NIJT, 1, NKT)
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
