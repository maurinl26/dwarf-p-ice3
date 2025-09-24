# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import sys
from itertools import repeat
from gt4py.storage import from_array
from gt4py.cartesian.gtscript import stencil
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from ifs_physics_common.framework.storage import managed_temporary_storage
from ifs_physics_common.utils.typingx import NDArrayLikeDict, PropertyDict

from ice3.phyex_common.phyex import Phyex
from ice3.phyex_common.lookup_table import SRC_1D

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


class IceAdjust:
    """Implicit Tendency Component calling
    ice_adjust : saturation adjustment of temperature and mixing ratios

    ice_adjust stencil is ice_adjust.F90 in PHYEX
    """

    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
        backend: str,
        phyex: Phyex,
        *,
        enable_checks: bool = True,
    ) -> None:

        externals = phyex.to_externals()

        from ice3.stencils.ice_adjust import ice_adjust
        self.ice_adjust = stencil(ice_adjust, externals, backend=backend)

        logging.info(f"Keys")
        logging.info(f"SUBG_COND : {phyex.nebn.LSUBG_COND}")
        logging.info(f"SUBG_MF_PDF : {phyex.param_icen.SUBG_MF_PDF}")
        logging.info(f"SIGMAS : {phyex.nebn.LSIGMAS}")
        logging.info(f"LMFCONV : {phyex.LMFCONV}")

    def __call__(self, state, dt):


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
                dt=dt.total_seconds(),
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )
