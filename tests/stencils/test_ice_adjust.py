# -*- coding: utf-8 -*-
from functools import partial
import fmodpy
import numpy as np
import logging

from typing import TYPE_CHECKING

from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.grid import ComputationalGrid
import datetime

from ice3_gt4py.phyex_common.phyex import Phyex

if TYPE_CHECKING:
    from typing import Literal, Tuple

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
    from ifs_physics_common.utils.typingx import (
        DataArray,
        NDArrayLikeDict,
    )


def allocate_state_cloud_fraction(
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

    return {
        "lv": allocate_f(),
        "ls": allocate_f(),
        "cph": allocate_f(),
        "rhodref": allocate_f(),
        "exnref": allocate_f(),
        "rc": allocate_f(),
        "ri": allocate_f(),
        "rcs": allocate_f(),
        "ris": allocate_f(),
        "rc_mf": allocate_f(),
        "ri_mf": allocate_f(),
        "cf_mf": allocate_f(),
        "rc_tm": allocate_f(),
        "rc_tmp": allocate_f(),
        "ri_tmp": allocate_f(),
        "hlc_hrc": allocate_f(),
        "hlc_hcf": allocate_f(),
        "hli_hri": allocate_f(),
        "hli_hcf": allocate_f(),
    }


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
        "exn": allocate_f(),
        "pabs": allocate_f(),
        "sigs": allocate_f(),
        "th": allocate_f(),
        "rv": allocate_f(),
        "rc": allocate_f(),
        "ri": allocate_f(),
        "rr": allocate_f(),
        "rs": allocate_f(),
        "rg": allocate_f(),
        "ths": allocate_f(),
        "rvs": allocate_f(),
        "rcs": allocate_f(),
        "ris": allocate_f(),
        "rv_tmp": allocate_f(),
        "ri_tmp": allocate_f(),
        "rc_tmp": allocate_f(),
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
    ) -> None:
        super().__init__(
            computational_grid=computational_grid, gt4py_config=gt4py_config
        )

        externals = {
            "SUBG_COND": SUBG_COND,
            "CRIAUTC": CRIAUTC,
            "SUBG_MF_PDF": SUBG_MF_PDF,
            "CRIAUTI": CRIAUTI,
            "ACRIAUTI": ACRIAUTI,
            "BCRIAUTI": BCRIAUTI,
            "TT": TT,
        }

        # aro_filter stands for the parts before 'call ice_adjust' in aro_adjust.f90
        self.condensation = self.compile_stencil("condensation", externals)
        self.condensation_gt4py = fmodpy.fimport(
            "./src/ice3_gt4py/stencils_fortran/test_ice_adjust/condensation.F90"
        )

    def test_fortran(self):
        """Call fortran stencil"""

        self.condensation.condensation(
            XTT,
            XRTMIN,
            LFEEDBACKT,
            KPROMA,
            KSIZE,
            LDCOMPUTE,
            PEXN,
            PLVFACT,
            PLSFACT,
            PT,
            PRRT,
            PTHT,
            PRRHONG_MR,
        )

        state = allocate_state_condensation(self.computational_grid, self.gt4py_config)

        state_condensation = {}

        self.condensation(**state_condensation)

    def compare(self):
        """Compare Fortran and Python routines"""

        self.test_fortran()
        logging.info("Fortran called")

        self.test_gt4py()
        logging.info("GT4Py called")


class TestCloudFraction(ComputationalGridComponent):
    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
    ) -> None:
        super().__init__(
            computational_grid=computational_grid, gt4py_config=gt4py_config
        )

        externals = {
            "SUBG_COND": SUBG_COND,
            "CRIAUTC": CRIAUTC,
            "SUBG_MF_PDF": SUBG_MF_PDF,
            "CRIAUTI": CRIAUTI,
            "ACRIAUTI": ACRIAUTI,
            "BCRIAUTI": BCRIAUTI,
            "TT": TT,
        }

        # aro_filter stands for the parts before 'call ice_adjust' in aro_adjust.f90
        self.cloud_fraction = self.compile_stencil("cloud_fraction", externals)

        self.mode_cloud_fraction = fmodpy.fimport(
            "./src/ice3_gt4py/stencils_fortran/test_ice_adjust/cloud_fraction.F90"
        )

    def test_fortran(self):
        """Call fortran stencil"""

        sizes = {}

        self.mode_cloud_fraction.cloud_fraction(
            XTT,
            XRTMIN,
            LFEEDBACKT,
            KPROMA,
            KSIZE,
            LDCOMPUTE,
            PEXN,
            PLVFACT,
            PLSFACT,
            PT,
            PRRT,
            PTHT,
            PRRHONG_MR,
        )

        state_cloud_fraction = {
            "lv": allocate_f(),
            "ls": allocate_f(),
            "cph": allocate_f(),
            "rhodref": allocate_f(),
            "exnref": allocate_f(),
            "rc": allocate_f(),
            "ri": allocate_f(),
            "rcs": allocate_f(),
            "ris": allocate_f(),
            "rc_mf": allocate_f(),
            "ri_mf": allocate_f(),
            "cf_mf": allocate_f(),
            "rc_tm": allocate_f(),
            "rc_tmp": allocate_f(),
            "ri_tmp": allocate_f(),
            "hlc_hrc": allocate_f(),
            "hlc_hcf": allocate_f(),
            "hli_hri": allocate_f(),
            "hli_hcf": allocate_f(),
        }

        self.cloud_fraction(
            LFEEDBACKT,
            KPROMA,
            KSIZE,
            LDCOMPUTE,
            PEXN,
            PLVFACT,
            PLSFACT,
            PT,
            PRRT,
            PTHT,
            PRRHONG_MR,
        )

    def compare(self):
        """Compare Fortran and Python routines"""

        self.test_fortran()
        logging.info("Fortran called")

        self.test_gt4py()
        logging.info("GT4Py called")


if __name__ == "__main__":
    SUBG_COND = 0
    CRIAUTC = 0
    SUBG_MF_PDF = 0
    CRIAUTI = 0
    ACRIAUTI = 0
    BCRIAUTI = 0
    TT = 0
