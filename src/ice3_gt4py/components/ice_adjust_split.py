# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from datetime import timedelta
import sys
from functools import cached_property
from itertools import repeat
from typing import Dict

import dace
from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from ifs_physics_common.framework.storage import managed_temporary_storage
from ifs_physics_common.utils.typingx import NDArrayLikeDict, PropertyDict

from ice3_gt4py.phyex_common.phyex import Phyex
from ice3_gt4py.phyex_common.tables import SRC_1D

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


class IceAdjustSplit(ImplicitTendencyComponent):
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
        self.externals.update({
            "OCND2": False
        })

        logging.info(f"LAMBDA3 : {self.externals['LAMBDA3']}")


        self.thermo = self.compile_stencil("thermodynamic_fields", self.externals)
        self.condensation = self.compile_stencil("condensation", self.externals)

        # todo : add sigrc diagnostic compilation
        # from ice3_gt4py.stencils.sigma_rc_dace import sigrc_computation
        #
        # self.nx, self.ny, self.nz = self.computational_grid.grids[(I, J, K)].shape
        # self.sigrc_diagnostic = sigrc_computation.to_sdfg().compile()

        self.cloud_fraction_1 = self.compile_stencil("cloud_fraction_1", self.externals)
        self.cloud_fraction_2 = self.compile_stencil("cloud_fraction_2", self.externals)



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

        }

    @cached_property
    def _tendency_properties(self) -> PropertyDict:
        return {
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
        }

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        return {
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
                "units": "",
                "dtype": "float",
            },
            "sigrc": {
                "grid": (I, J, K),
                "units": "",
                "dtype": "float",
            },
        }

    @cached_property
    def _temporaries(self) -> PropertyDict:
        return {
            "lv": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "ls": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "cph": {"grid": (I, J, K), "units": "", "dtype": "float"},
            "t": {"grid": (I, J, K), "units": "", "dtype":"float"}
        }

    def array_call(
        self,
        state: NDArrayLikeDict,
        timestep: timedelta,
        out_tendencies: NDArrayLikeDict,
        out_diagnostics: NDArrayLikeDict,
        overwrite_tendencies: Dict[str, bool],
    ) -> None:

        logging.info(f"Timestep : {timestep.total_seconds()}")

        with managed_temporary_storage(
            self.computational_grid,
            *repeat(((I, J, K), "float"), 9),
            ((I,J,K), "int"),
            gt4py_config=self.gt4py_config,
        ) as (lv, ls, cph, t, rc_out, ri_out, rv_out, t_out, q1, inq1):

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

            temporaries_thermo = {"cph": cph, "lv": lv, "ls": ls, "t": t}

            logging.info("Launching thermo")

            self.thermo(
                **state_thermo,
                **temporaries_thermo,
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )

            logging.info(f"Thermo output")
            logging.info(f"Mean cph {cph.mean()}")

            state_condensation = {
                key: state[key]
                for key in ["sigqsat", "pabs", "cldfr", "sigs", "ri", "rc", "rv"]
            }

            temporaries_condensation = {
                "cph": cph,
                "lv": lv,
                "ls": ls,
                "t": t,
                "rv_out": rv_out,
                "ri_out": ri_out,
                "rc_out": rc_out,
                "q1": q1
            }

            self.condensation(
                **state_condensation,
                **temporaries_condensation,
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )

            logging.info(f"Condensation output")
            logging.info(f"Mean rv_out {rv_out.mean()}")

            # todo : add dace managed sigrc diagnostic
            # self.sigrc_diagnostic(
            #     q1=q1,
            #     inq1=inq1,
            #     src_1d=SRC_1D,
            #     sigrc=out_diagnostics["sigrc"],
            #     LAMBDA3=0,
            #     I=self.nx,
            #     J=self.ny,
            #     K=self.nz,
            #     F=34
            # )

            state_cloud_fraction_1 = {
                key: state[key]
                for key in [
                    "exnref",
                    "rc",
                    "ri",
                ]
            }

            tendencies_cloud_fraction_1 = {
                name: out_tendencies[name]
                for name in [
                    "ths",
                    "rvs",
                    "rcs",
                    "ris"
                ]
            }

            # TODO: check the scope of t
            temporaries_cloud_fraction_1 = {
                "lv": lv,
                "ls": ls,
                "cph": cph,
                "rc_tmp": rc_out,
                "ri_tmp": ri_out,
            }

            logging.info("Launching cloud fraction 1")
            self.cloud_fraction_1(
                **state_cloud_fraction_1,
                **tendencies_cloud_fraction_1,
                **temporaries_cloud_fraction_1,
                dt=timestep.total_seconds(),
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )

            logging.info(f"Cloud fraction 1 output")
            logging.info(f"Mean rc_out {rc_out.mean()}")

            state_cloud_fraction_2 = {
                key: state[key]
                for key in [
                    "rhodref",
                    "exnref",
                    "rc_mf",
                    "ri_mf",
                    "cf_mf",
                    "cldfr",
                ]
            }

            diagnotics_cloud_fraction_2 = {
                name: out_diagnostics[name]
                for name in self._diagnostic_properties.keys()
            }
            diagnotics_cloud_fraction_2.pop("sigrc")

            tendencies_cloud_fraction_2 = {
                name: out_tendencies[name]
                for name in [
                    "ths",
                    "rvs",
                    "rcs",
                    "ris",
                ]
            }

            temporaries_cloud_fraction_2 = {
                "lv": lv,
                "ls": ls,
                "cph": cph,
                "t": t,
            }

            self.cloud_fraction_2(
                **state_cloud_fraction_2,
                **diagnotics_cloud_fraction_2,
                **tendencies_cloud_fraction_2,
                **temporaries_cloud_fraction_2,
                dt=timestep.total_seconds(),
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )

            logging.info(f"Cloud fraction 2 output")
            logging.info(f"Mean hlc_hrc {diagnotics_cloud_fraction_2['hlc_hrc'].mean()}")

