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

from ice3.phyex_common.phyex import Phyex
from ice3.phyex_common.tables import SRC_1D

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
        externals.update({
            "OCND2": True
        })
        self.ice_adjust = self.compile_stencil("ice_adjust", externals)



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
                "dtype": "float",
            },
            "exn": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "exnref": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "rhodref": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "pabs": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "sigs": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "cf_mf": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "rc_mf": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "ri_mf": {
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
            "sigrc": {
                "grid": (I, J, K),
                "dtype": "float",
            },
            "ths": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "rcs": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "rrs": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "ris": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "rss": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "rvs": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "rgs": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "sigrc": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "pv": {
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
            "cldfr": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "ifr": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "hlc_hrc": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "hlc_hcf": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "hli_hri": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
            "hli_hcf": {
                "grid": (I, J, K),
                "units":"",
                "dtype": "float",
            },
        }


    @cached_property
    def _temporaries(self) -> PropertyDict:
        return {
            "lv": {"grid": (I, J, K), "units": ""},
            "ls": {"grid": (I, J, K), "units": ""},
            "cph": {"grid": (I, J, K), "units": ""},
            "criaut": {"grid": (I, J, K), "units": ""},
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
            gt4py_config=self.gt4py_config,
        ) as (
            lv,
            ls,
            cph,
            criaut,
        ):
            state_ice_adjust = {
                key: state[key]
                for key in [
                    "sigqsat",
                    "exn",
                    "exnref",
                    "rhodref",
                    "pabs",
                    "sigs",
                    "cf_mf",
                    "rc_mf",
                    "ri_mf",
                    "th",
                    "rv",
                    "rc",
                    "rr",
                    "ri",
                    "rs",
                    "rg",
                    "cldfr",
                    "ifr",
                    "hlc_hrc",
                    "hlc_hcf",
                    "hli_hri",
                    "hli_hcf",
                    "sigrc",
                    "ths",
                    "rvs",
                    "rcs",
                    "ris",
                ]
            }

            temporaries_ice_adjust = {
                "criaut": criaut,
                "cph": cph,
                "lv": lv,
                "ls": ls,
                "inq1": inq1,
            }

            # Global Table
            logging.info("Loading src_1d GlobalTable")
            src_1D = from_array(SRC_1D, backend=self.gt4py_config.backend)

            # Timestep
            logging.info("Launching ice_adjust")
            self.ice_adjust(
                **state_ice_adjust,
                **temporaries_ice_adjust,
                src_1d=src_1D,
                dt=timestep.total_seconds(),
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )
