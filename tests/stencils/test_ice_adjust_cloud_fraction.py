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


class TestCloudFraction(ComputationalGridComponent):
    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
        phyex: Phyex,
    ) -> None:
        super().__init__(
            computational_grid=computational_grid, gt4py_config=gt4py_config
        )

        kproma, ksize = 15, 50

        # TODO : set phyex as a fixture
        externals = phyex.to_externals()
        # aro_filter stands for the parts before 'call ice_adjust' in aro_adjust.f90
        self.cloud_fraction = self.compile_stencil("cloud_fraction", externals)

        ######
        self.externals = {
            "lsubg_cond": externals["SUBG_COND"],
            "xcriautc": externals["CRIAUTC"],
            "csubg_mf_pdf": externals["SUBG_MF_PDF"],
            "xcriauti": externals["CRIAUTI"],
            "xacriauti": externals["ACRIAUTI"],
            "xbcriauti": externals["BCRIAUTI"],
            "xtt": externals["TT"],
        }

        self.dims = {"kproma": kproma, "ksize": ksize}

        self.fields = {
            "pexnref": np.asfortranarray(np.random.rand(kproma, ksize)),
            "prhodref": np.asfortranarray(np.random.rand(kproma, ksize)),
            "pcf_mf": np.asfortranarray(np.random.rand(kproma, ksize)),
            "prc_mf": np.asfortranarray(np.random.rand(kproma, ksize)),
            "pri_mf": np.asfortranarray(np.random.rand(kproma, ksize)),
            "prc": np.asfortranarray(np.random.rand(kproma, ksize)),
            "prvs": np.asfortranarray(np.random.rand(kproma, ksize)),
            "prcs": np.asfortranarray(np.random.rand(kproma, ksize)),
            "pths": np.asfortranarray(np.random.rand(kproma, ksize)),
            "psrcs": np.asfortranarray(np.random.rand(kproma, ksize)),
            "pcldfr": np.asfortranarray(np.random.rand(kproma, ksize)),
            "pri": np.asfortranarray(np.random.rand(kproma, ksize)),
            "pris": np.asfortranarray(np.random.rand(kproma, ksize)),
            "phlc_hrc": np.asfortranarray(np.random.rand(kproma, ksize)),
            "phlc_hcf": np.asfortranarray(np.random.rand(kproma, ksize)),
            "phli_hri": np.asfortranarray(np.random.rand(kproma, ksize)),
            "phli_hcf": np.asfortranarray(np.random.rand(kproma, ksize)),
        }

        self.mode_cloud_fraction = fmodpy.fimport(
            "./src/ice3_gt4py/stencils_fortran/test_ice_adjust/cloud_fraction.F90"
        )

    def allocate_random_fields(self):
        """Allocate fields to gt4py"""

        state = allocate_state_cloud_fraction(
            self.computational_grid, self.gt4py_config
        )
        self.state = {
            "lv": self.fields["pls"],
            "ls": self.fields["plv"],
            "t": self.fields["pt"],
            "cph": self.fields["pcph"],
            "rhodref": self.fields["prhodref"],
            "exnref": self.fields["pexnref"],
            "rc": self.fields["prc"],
            "ri": self.fields["pri"],
            "ths": self.fields["pths"],
            "rvs": self.fields["prvs"],
            "rcs": self.fields["prcs"],
            "ris": self.fields["pris"],
            "rc_mf": self.fields["prc_mf"],
            "ri_mf": self.fields["pri_mf"],
            "cf_mf": self.fields["pcf_mf"],
            "rc_tmp": self.fields["prc"],
            "ri_tmp": self.fields["pri"],
            "hlc_hrc": self.fields["phlc_hrc"],
            "hlc_hcf": self.fields["phlc_hcf"],
            "hli_hri": self.fields["phli_hri"],
            "hli_hcf": self.fields["phli_hcf"],
        }

    def test(self):
        """Call fortran stencil"""

        logging.info("Test Fortran")
        self.mode_cloud_fraction.cloud_fraction(
            **self.dims, **self.externals, **self.fields
        )

        logging.info("Test GT4Py")
        self.cloud_fraction(**self.state)

        fortran_field = self.fields["phlc_hrc"]
        python_field = self.state["hlc_hrc"][...]

        logging.info(
            f"Fortran, mean {fortran_field.mean()}, GT4Py, mean {python_field.mean()}"
        )
