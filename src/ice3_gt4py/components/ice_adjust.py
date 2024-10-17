# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from datetime import timedelta
import sys
from functools import cached_property
from itertools import repeat
from typing import Dict
from gt4py.storage import from_array
from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from ifs_physics_common.framework.storage import managed_temporary_storage
from ifs_physics_common.utils.typingx import NDArrayLikeDict, PropertyDict

from ice3_gt4py.phyex_common.phyex import Phyex
from ice3_gt4py.phyex_common.tables import src_1d

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


class IceAdjust(ImplicitTendencyComponent):
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

        externals = phyex.to_externals()
        self.condensation = self.compile_stencil("condensation", externals)
        self.cloud_fraction = self.compile_stencil("cloud_fraction", externals)

        logging.info(f"Keys")
        logging.info(f"SUBG_COND : {phyex.nebn.LSUBG_COND}")
        logging.info(f"SUBG_MF_PDF : {phyex.param_icen.SUBG_MF_PDF}")
        logging.info(f"SIGMAS : {phyex.nebn.LSIGMAS}")
        logging.info(f"LMFCONV : {phyex.LMFCONV}")

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "sigqsat": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": None,
                "dtype": "float",
            },
            "exn": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "PEXNREF",
                "dtype": "float",
            },
            "exnref": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "PEXNREF",
                "dtype": "float",
            },
            "rhodref": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "PRHODREF",
                "dtype": "float",
            },
            "pabs": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "PPABSM",
                "dtype": "float",
            },
            "sigs": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "PSIGS",
                "dtype": "float",
            },
            "cf_mf": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "PCF_MF",
                "dtype": "float",
            },
            "rc_mf": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "PRC_MF",
                "dtype": "float",
            },
            "ri_mf": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "PRI_MF",
                "dtype": "float",
            },
            "th": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "ZRS",
                "irr": 0,
                "dtype": "float",
            },
            "rv": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "ZRS",
                "irr": 1,
                "dtype": "float",
            },
            "rc": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "ZRS",
                "irr": 2,
                "dtype": "float",
            },
            "rr": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "ZRS",
                "irr": 3,
                "dtype": "float",
            },
            "ri": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "ZRS",
                "irr": 4,
                "dtype": "float",
            },
            "rs": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "ZRS",
                "irr": 5,
                "dtype": "float",
            },
            "rg": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "ZRS",
                "irr": 6,
                "dtype": "float",
            },
            "cldfr": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": None,
                "dtype": "float",
            },
            "ifr": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": None,
                "dtype": "float",
            },
            "hlc_hrc": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": None,
                "dtype": "float",
            },
            "hlc_hcf": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": None,
                "dtype": "float",
            },
            "hli_hri": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": None,
                "dtype": "float",
            },
            "hli_hcf": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": None,
                "dtype": "float",
            },
            "sigrc": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": None,
                "dtype": "float",
            },
            "ths": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "PRS",
                "irr": 0,
                "dtype": "float",
            },
            "rcs": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "PRS",
                "irr": 1,
                "dtype": "float",
            },
            "rrs": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "PRS",
                "irr": 2,
                "dtype": "float",
            },
            "ris": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "PRS",
                "irr": 3,
                "dtype": "float",
            },
            "rss": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "PRS",
                "irr": 4,
                "dtype": "float",
            },
            "rvs": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "PRS",
                "irr": 5,
                "dtype": "float",
            },
            "rgs": {
                "grid": (I, J, K),
                "units": "",
                "fortran_name": "PRS",
                "irr": 6,
                "dtype": "float",
            },
        }

    @cached_property
    def _tendency_properties(self) -> PropertyDict:
        return {}

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        return {}

    @cached_property
    def _temporaries(self) -> PropertyDict:
        return {
            "lv": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "ls": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "cph": {"grid": (I, J, K), "dtype": "float", "unit": ""},
            "criaut": {"grid": (I, J, K), "dtype": "float", "unit": ""},
        }

    def array_call(
        self,
        state: NDArrayLikeDict,
        timestep: timedelta,
        out_tendencies: NDArrayLikeDict,
        out_diagnostics: NDArrayLikeDict,
        overwrite_tendencies: Dict[str, bool],
    ) -> None:
        with managed_temporary_storage(
            self.computational_grid,
            *repeat(((I, J, K), "float"), 4),
            *repeat(((I, J, K), "int"), 1),
            gt4py_config=self.gt4py_config,
        ) as (
            lv,
            ls,
            cph,
            criaut,
            inq1,
        ):

            state_condensation = {
                key: state[key]
                for key in [
                    "sigqsat",
                    "exn",
                    "pabs",
                    "sigs",
                    "th",
                    "rv",
                    "rc",
                    "ri",
                    "rr",
                    "rs",
                    "rg",
                    "ths",
                    "rvs",
                    "rcs",
                    "ris",
                    "rv_tmp",
                    "ri_tmp",
                    "rc_tmp",
                    "cldfr",
                    "sigrc",
                ]
            }

            temporaries_condensation = {
                "cph": cph,
                "lv": lv,
                "ls": ls,
                "inq1": inq1,
            }

            # Global Table
            logging.info("Loading src_1d GlobalTable")
            src_1D = from_array(src_1d, backend=self.gt4py_config.backend)

            # Timestep
            logging.info("Launching ice_adjust")
            self.condensation(
                **state_condensation,
                **temporaries_condensation,
                src_1d=src_1D,
                dt=timestep.total_seconds(),
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )

            state_cloud_fraction = {
                key: state[key]
                for key in [
                    "rhodref",
                    "exnref",
                    "rc",
                    "ri",
                    "rcs",
                    "ris",
                    "rc_mf",
                    "ri_mf",
                    "cf_mf",
                    "rc_tmp",
                    "ri_tmp",
                    "hlc_hrc",
                    "hlc_hcf",
                    "hli_hri",
                    "hli_hcf",
                ]
            }

            temporaries_cloud_fraction = {
                "lv": lv,
                "ls": ls,
                "cph": cph,
            }
