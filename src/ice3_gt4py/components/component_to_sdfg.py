# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import sys
import dace
from functools import cached_property
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

        logging.info(f"Constants for condensation")
        logging.info(f"RD : {phyex.cst.RD}")
        logging.info(f"RV : {phyex.cst.RV}")

        logging.info(f"Constants for thermodynamic fields")
        logging.info(f"CPD : {phyex.cst.CPD}")
        logging.info(f"CPV : {phyex.cst.CPV}")
        logging.info(f"CL : {phyex.cst.CL}")
        logging.info(f"CI : {phyex.cst.CI}")


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

    def dace_setup(self):

        from ifs_physics_common.framework.grid import I, J, K

        I = dace.symbol(I.name)
        J = dace.symbol(J.name)
        K = dace.symbol(K.name)

    @dace.method
    def orchestrated_call(
            self,
            # inputs
            th: dace.float32[I, J, K],
            exn: dace.float32[I, J, K],
            rv: dace.float32[I, J, K],
            rc: dace.float32[I, J, K],
            rr: dace.float32[I, J, K],
            ri: dace.float32[I, J, K],
            rs: dace.float32[I, J, K],
            rg: dace.float32[I, J, K],
            # outputs
            lv: dace.float32[I, J, K],
            ls:dace.float32[I, J, K],
            cph: dace.float32[I, J, K],
            t: dace.float32[I, J, K],
        ):

        self.thermo(
            th=th,
            exn=exn,
            rv=rv,
            rc=rc,
            rr=rr,
            ri=ri,
            rs=rs,
            rg=rg,
            lv=lv,
            ls=ls,
            cph=cph,
            t=t,
            origin=(0, 0, 0),
            domain=self.computational_grid.grids[I, J, K].shape,
            validate_args=self.gt4py_config.validate_args,
            exec_info=self.gt4py_config.exec_info,
        )


