# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from datetime import timedelta
import sys
from functools import cached_property
from itertools import repeat
from typing import Dict
from ifs_physics_common.framework.components import DiagnosticComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from ifs_physics_common.framework.storage import managed_temporary_storage
from ifs_physics_common.utils.typingx import NDArrayLikeDict, PropertyDict

from ice3_gt4py.phyex_common.phyex import Phyex

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


class TestComponent(DiagnosticComponent):
    """Implicit Tendency Component calling
    ice_adjust : saturation adjustment of temperature and mixing ratios

    ice_adjust stencil is ice_adjust.F90 in PHYEX
    """

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

        self.externals = phyex.to_externals()
        self.thermo = self.compile_stencil("thermodynamic_fields", self.externals)
        self.condensation = self.compile_stencil("condensation", self.externals)
        self.cloud_fraction = self.compile_stencil("cloud_fraction", self.externals)

        logging.info(f"IceAdjustSplit - Keys")
        logging.info(f"LSUBG_COND : {phyex.nebn.LSUBG_COND}")
        logging.info(f"LSIGMAS :  {phyex.nebn.LSIGMAS}")
        logging.info(f"FRAC_ICE_ADJUST : {phyex.nebn.FRAC_ICE_ADJUST}")
        logging.info(f"CONDENS : {phyex.nebn.CONDENS}")
        logging.info(f"LAMBDA3 : {phyex.nebn.LAMBDA3}")
        logging.info(f"OCOMPUTE_SRC absent")
        logging.info(f"LMFCONV : {phyex.LMFCONV}")
        logging.info(f"LOCND2 absent")
        logging.info(f"LHGT_QS : {phyex.nebn.LHGT_QS}")
        logging.info(f"LSTATNW : {phyex.nebn.LSTATNW}")
        logging.info(f"SUBG_MF_PDF : {phyex.param_icen.SUBG_MF_PDF}")

        logging.info(f"Constants for condensation")
        logging.info(f"RD : {phyex.cst.RD}")
        logging.info(f"RV : {phyex.cst.RV}")

        logging.info(f"Constants for thermodynamic fields")
        logging.info(f"CPD : {phyex.cst.CPD}")
        logging.info(f"CPV : {phyex.cst.CPV}")
        logging.info(f"CL : {phyex.cst.CL}")
        logging.info(f"CI : {phyex.cst.CI}")

        logging.info(f"Constants for cloud fraction")
        logging.info(f"CRIAUTC : {phyex.rain_ice_param.CRIAUTC}")
        logging.info(f"CRIAUTI : {phyex.rain_ice_param.CRIAUTI}")
        logging.info(f"ACRIAUTI : {phyex.rain_ice_param.ACRIAUTI}")
        logging.info(f"BCRIAUTI : {phyex.rain_ice_param.BCRIAUTI}")
        logging.info(f"TT : {phyex.cst.TT}")

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "exn": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "th": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "rv": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "rc": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "rr": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "ri": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "rs": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "rg": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
        }

    @cached_property
    def _tendency_properties(self) -> PropertyDict:
        return {}

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        return {
            "lv": {"grid": (I, J, K), "units": ""},
            "ls": {"grid": (I, J, K), "units": ""},
            "cph": {"grid": (I, J, K), "units": ""},
            "t": {"grid": (I, J, K), "units": ""},
        }

    @cached_property
    def _temporaries(self) -> PropertyDict:
        return {}

    def array_call(
        self,
        state: NDArrayLikeDict,
        out_diagnostics: NDArrayLikeDict,
    ) -> None:

        state_thermo = {
            key: state[key]
            for key in [
                "th",
                "exn",
                "rv",
                "rc",
                "rr",
                "ri",
                "rs",
                "rg",
            ]
        }

        self.thermo(
            **state_thermo,
            origin=(0, 0, 0),
            domain=self.computational_grid.grids[I, J, K].shape,
            validate_args=self.gt4py_config.validate_args,
            exec_info=self.gt4py_config.exec_info,
        )

