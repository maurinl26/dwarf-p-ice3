# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import sys
from functools import partial
from itertools import repeat
from typing import Tuple, Dict

from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array
from ifs_physics_common.framework.storage import managed_temporary_storage

from ..phyex_common.lookup_table import SRC_1D
from ..phyex_common.phyex import Phyex
from ..utils.env import sp_dtypes

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


class IceAdjust:
    """Implicit Tendency Component calling
    ice_adjust : saturation adjustment of temperature and mixing ratios

    ice_adjust stencil is ice_adjust.F90 in PHYEX
    """

    def __init__(
        self,
        phyex: Phyex = Phyex("AROME"),
        dtypes: Dict = sp_dtypes,
        backend: str = "gt:cpu_ifirst",
    ) -> None:

        compile_stencil = partial(
            backend=backend,
            externals=phyex.externals,
            dtypes=dtypes,
        )

        from ..stencils.ice_adjust import ice_adjust

        self.ice_adjust = compile_stencil(
            name="ice_adjust",
            definition=ice_adjust
        )

        logging.info(f"Keys")
        logging.info(f"SUBG_COND : {phyex.nebn.LSUBG_COND}")
        logging.info(f"SUBG_MF_PDF : {phyex.param_icen.SUBG_MF_PDF}")
        logging.info(f"SIGMAS : {phyex.nebn.LSIGMAS}")
        logging.info(f"LMFCONV : {phyex.LMFCONV}")

    def __call__(self,
                 computational_grid,
                 state,
                 timestep,
                 domain: Tuple ,
                 exec_info: Dict,
                 validate_args: bool = False,
                 ):

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
            }

            # Global Table
            logging.info("Loading src_1d GlobalTable")
            src_1D = from_array(SRC_1D, backend=self.backend)

            # Timestep
            logging.info("Launching ice_adjust")
            self.ice_adjust(
                **state_ice_adjust,
                **temporaries_ice_adjust,
                src_1d=src_1D,
                dt=timestep.total_seconds(),
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
            )
