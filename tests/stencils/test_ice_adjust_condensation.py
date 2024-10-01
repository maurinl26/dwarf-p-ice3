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
        phyex: Phyex,
    ) -> None:
        super().__init__(
            computational_grid=computational_grid, gt4py_config=gt4py_config
        )

        externals_gt4py = phyex.to_externals()

        # aro_filter stands for the parts before 'call ice_adjust' in aro_adjust.f90
        self.condensation_gt4py = self.compile_stencil("condensation", externals_gt4py)

        self.externals = {
            "lsubg_cond": externals_gt4py["SUBG_COND"],
            "xcriautc": externals_gt4py["CRIAUTC"],
            "csubg_mf_pdf": externals_gt4py["SUBG_MF_PDF"],
            "xcriauti": externals_gt4py["CRIAUTI"],
            "xacriauti": externals_gt4py["ACRIAUTI"],
            "xbcriauti": externals_gt4py["BCRIAUTI"],
            "xtt": externals_gt4py["TT"],
            "lhgt_qs": externals_gt4py["LHGT_QS"],
            "lstatnw": externals_gt4py["LSTATNW"],
            "xfrmin": externals_gt4py["XFRMIN"],
            "xtmaxmix": externals_gt4py["XTMAXMIX"],
            "xtminmix": externals_gt4py["XTMINMIX"],
            "hfrac_ice": externals_gt4py["HFRAC_ICE"],
            "hcondens": externals_gt4py["HCONDENS"],
            "hlambda3": externals_gt4py["HLAMBDA3"],
            "lmfconv": externals_gt4py["LMFCONV"],
            "ouseri": externals_gt4py["OUSERI"],
            "osigmas": externals_gt4py["OSIGMAS"],
            "ocnd2": externals_gt4py["OCND2"],
        }
        """_summary_
        """
        self.dims = {
            "nijt": NIJT,
            "nkt": NKT,
            "nkb": NKB,
            "nke": NKE,
            "nkl": NKL,
            "nijb": NIJB,
            "nije": NIJE,
        }

        self.fortran_directory = "./src/ice3_gt4py/stencils_fortran/"
        self.condensation = fmodpy.fimport(
            self.fortran_directory + "ice_adjust.F90",
            libs=[self.fortran_directory + "condensation.F90"],
        )

    def generate_state(self):

        nijt, nkt = self.dims["nijt"], self.dims["nkt"]

        self.fields = {
            "ppabs": np.asfortranarray(np.random.rand(nijt, nkt)),
            "pzz": np.asfortranarray(np.random.rand(nijt, nkt)),
            "prhodref": np.asfortranarray(np.random.rand(nijt, nkt)),
            "pt": np.asfortranarray(np.random.rand(nijt, nkt)),
            "prv_in": np.asfortranarray(np.random.rand(nijt, nkt)),
            "prv_out": np.asfortranarray(np.random.rand(nijt, nkt)),
            "prc_in": np.asfortranarray(np.random.rand(nijt, nkt)),
            "prc_out": np.asfortranarray(np.random.rand(nijt, nkt)),
            "pri_in": np.asfortranarray(np.random.rand(nijt, nkt)),
            "pri_out": np.asfortranarray(np.random.rand(nijt, nkt)),
            "prr": np.asfortranarray(np.random.rand(nijt, nkt)),
            "prs": np.asfortranarray(np.random.rand(nijt, nkt)),
            "prg": np.asfortranarray(np.random.rand(nijt, nkt)),
            "psigs": np.asfortranarray(np.random.rand(nijt, nkt)),
            "pmfconv": np.asfortranarray(np.random.rand(nijt, nkt)),
            "pcldfr": np.asfortranarray(np.random.rand(nijt, nkt)),
            "psigrc": np.asfortranarray(np.random.rand(nijt, nkt)),
            "picldfr": np.asfortranarray(np.random.rand(nijt, nkt)),
            "pwdcldfr": np.asfortranarray(np.random.rand(nijt, nkt)),
            "pssio": np.asfortranarray(np.random.rand(nijt, nkt)),
            "pssiu": np.asfortranarray(np.random.rand(nijt, nkt)),
            "pifr": np.asfortranarray(np.random.rand(nijt, nkt)),
            "psigqsat": np.asfortranarray(np.random.rand(nijt, nkt)),
            "plv": np.asfortranarray(np.random.rand(nijt, nkt)),
            "pls": np.asfortranarray(np.random.rand(nijt, nkt)),
            "pcph": np.asfortranarray(np.random.rand(nijt, nkt)),
            "phlc_hrc": np.asfortranarray(np.random.rand(nijt, nkt)),
            "phlc_hcf": np.asfortranarray(np.random.rand(nijt, nkt)),
            "phli_hri": np.asfortranarray(np.random.rand(nijt, nkt)),
            "phli_hcf": np.asfortranarray(np.random.rand(nijt, nkt)),
            "pice_cld_wgt": np.asfortranarray(np.random.rand(nijt, nkt)),
        }

        state = allocate_state_condensation(self.computational_grid, self.gt4py_config)

        self.state_gt4py = {
            "sigqsat": self.fields["psigqsat"],
            "exn": self.fields["pexn"],
            "pabs": self.fields["ppabs"],
            "sigs": self.fields["psigs"],
            "th": self.fields["ptht"],
            "rv": self.fields["prv"],
            "rc": self.fields["prc"],
            "ri": self.fields["pri"],
            "rr": self.fields["prr"],
            "rs": self.fields["prs"],
            "rg": self.fields["prg"],
            "ths": self.fields["pths"],
            "rvs": self.fields["prvs"],
            "rcs": self.fields["prcs"],
            "ris": self.fields["pris"],
            "rv_tmp": self.fields["rv_tmp"],
            "ri_tmp": self.fields["ri_tmp"],
            "rc_tmp": self.fields["rc_tmp"],
            "cldfr": self.fields["pcldfr"],
            "sigrc": self.fields["psigrc"],
            "cph": self.fields["pcph"],
            "lv": self.fields["plv"],
            "ls": self.fields["pls"],
            "inq1": self.fields["inq1"],
        }

    def test_fortran(self):
        """Call fortran stencil"""

        self.condensation.condensation(**self.dims, **self.externals, **self.fields)

        self.condensation(**self.state_gt4py)
