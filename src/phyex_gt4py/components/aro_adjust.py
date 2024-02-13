# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import asdict
from datetime import timedelta
from functools import cached_property
from itertools import repeat
from typing import Dict
import logging

from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.utils.typingx import PropertyDict, NDArrayLikeDict
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import managed_temporary_storage
from phyex_gt4py.phyex_common.phyex import Phyex


class AroAdjust(ImplicitTendencyComponent):
    """Implicit Tendency Component calling sequentially
    - aro_filter : negativity filters
    - ice_adjust : saturation adjustment of temperature and mixing ratios

    aro_filter stencil is aro_adjust.F90 in PHYEX, from l210 to l366
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

        externals = {}
        externals.update(asdict(phyex.nebn))
        externals.update(asdict(phyex.cst))
        externals.update(asdict(phyex.param_icen))
        externals.update(
            {
                "nrr": 6,
                "criautc": 0,
                "acriauti": 0,
                "bcriauti": 0,
                "criauti": 0,
            }
        )

        # aro_filter stands for the parts before 'call ice_adjust' in aro_adjust.f90
        self.aro_filter = self.compile_stencil("aro_filter", externals)

        # ice_adjust stands for ice_adjust.f90
        self.ice_adjust = self.compile_stencil("ice_adjust", externals)

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "f_sigqsat": {
                "grid": (I, J, K),
                "units": "",
            },  # coeff applied to qsat variance
            "f_exnref": {"grid": (I, J, K), "units": ""},  # ref exner pression
            "f_exn": {"grid": (I, J, K), "units": ""},
            "f_rhodref": {"grid": (I, J, K), "units": ""},  #
            "f_pabs": {"grid": (I, J, K), "units": ""},  # absolute pressure at t
            "f_sigs": {"grid": (I, J, K), "units": ""},  # Sigma_s at time t
            "f_cf_mf": {
                "grid": (I, J, K),
                "units": "",
            },  # convective mass flux fraction
            "f_rc_mf": {
                "grid": (I, J, K),
                "units": "",
            },  # convective mass flux liquid mixing ratio
            "f_ri_mf": {"grid": (I, J, K), "units": ""},
            "f_tht": {"grid": (I, J, K), "units": ""},
        }

    @cached_property
    def _tendency_properties(self) -> PropertyDict:
        return {
            "f_ths": {"grid": (I, J, K), "units": ""},
            "f_rvs": {"grid": (I, J, K), "units": ""},  # PRS(1)
            "f_rcs": {"grid": (I, J, K), "units": ""},  # PRS(2)
            "f_ris": {"grid": (I, J, K), "units": ""},  # PRS(4)
            "f_rrs": {"grid": (I, J, K), "units": ""},
            "f_rss": {"grid": (I, J, K), "units": ""},
            "f_rgs": {"grid": (I, J, K), "units": ""},
            "f_th": {"grid": (I, J, K), "units": ""},  # ZRS(0)
            "f_rv": {"grid": (I, J, K), "units": ""},  # ZRS(1)
            "f_rc": {"grid": (I, J, K), "units": ""},  # ZRS(2)
            "f_rr": {"grid": (I, J, K), "units": ""},  # ZRS(3)
            "f_ri": {"grid": (I, J, K), "units": ""},  # ZRS(4)
            "f_rs": {"grid": (I, J, K), "units": ""},  # ZRS(5)
            "f_rg": {"grid": (I, J, K), "units": ""},  # ZRS(6)
            "f_cldfr": {"grid": (I, J, K), "units": ""},
        }

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        return {
            "f_ifr": {"grid": (I, J, K), "units": ""},
            "f_hlc_hrc": {"grid": (I, J, K), "units": ""},
            "f_hlc_hcf": {"grid": (I, J, K), "units": ""},
            "f_hli_hri": {"grid": (I, J, K), "units": ""},
            "f_hli_hcf": {"grid": (I, J, K), "units": ""},
        }

    @cached_property
    def _temporaries(self) -> PropertyDict:
        return {
            "rt": {
                "grid": (I, J, K),
                "units": "",
            },  # work array for total water mixing ratio
            "pv": {"grid": (I, J, K), "units": ""},  # thermodynamics
            "piv": {"grid": (I, J, K), "units": ""},  # thermodynamics
            "qsl": {"grid": (I, J, K), "units": ""},  # thermodynamics
            "qsi": {"grid": (I, J, K), "units": ""},
            "frac_tmp": {"grid": (I, J, K), "units": ""},  # ice fraction
            "cond_tmp": {"grid": (I, J, K), "units": ""},  # condensate
            "a": {"grid": (I, J, K), "units": ""},  # related to computation of Sig_s
            "sbar": {"grid": (I, J, K), "units": ""},
            "sigma": {"grid": (I, J, K), "units": ""},
            "q1": {"grid": (I, J, K), "units": ""},
            "lv": {"grid": (I, J, K), "units": ""},
            "ls": {"grid": (I, J, K), "units": ""},
            "cph": {"grid": (I, J, K), "units": ""},
            "criaut": {"grid": (I, J, K), "units": ""},
            "sigrc": {"grid": (I, J, K), "units": ""},
            "rv_tmp": {"grid": (I, J, K), "units": ""},
            "ri_tmp": {"grid": (I, J, K), "units": ""},
            "rc_tmp": {"grid": (I, J, K), "units": ""},
            "t_tmp": {"grid": (I, J, K), "units": ""},
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
            *repeat(((I, J, K), "float"), 22),
            gt4py_config=self.gt4py_config,
        ) as (
            rt,
            pv,
            piv,
            qsl,
            qsi,
            frac_tmp,
            cond_tmp,
            a,
            sbar,
            sigma,
            q1,
            lv,
            ls,
            cph,
            criaut,
            sigrc,
            rv_tmp,
            ri_tmp,
            rc_tmp,
            t_tmp,
            cor_tmp,
            cph_tmp,
        ):

            input_filter_keys = ["f_exnref", "f_tht"]

            tendencies_filter_keys = [
                "f_ths",
                "f_rvs",
                "f_rcs",
                "f_ris",
                "f_rrs",
                "f_rss",
                "f_rgs",
            ]

            logging.debug(f"State : {state.keys()}")
            input_filter = {
                name.split("_", maxsplit=1)[1]: state[name]
                for name in input_filter_keys
            }

            logging.debug(f"Out tendencies : {out_tendencies.keys()}")
            tendencies_filter = {
                name.split("_", maxsplit=1)[1]: out_tendencies[name]
                for name in tendencies_filter_keys
            }

            temporaries_filter = {
                "cor_tmp": cor_tmp,  # ZCOR in aro_adjust.f90
                "cph_tmp": cph_tmp,
                "t_tmp": t_tmp,
                "lv_tmp": lv,
                "ls_tmp": ls,
            }

            logging.debug("Launching aro_filter")
            self.aro_filter(
                **input_filter,
                **tendencies_filter,
                # **diagnostics_filter,
                **temporaries_filter,
                dt=timestep.total_seconds(),
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )
            logging.debug("aro_filter passed")

            inputs = {
                name.split("_", maxsplit=1)[1]: state[name]
                for name in self.input_properties
            }
            inputs.pop("tht")

            tendencies = {
                name.split("_", maxsplit=1)[1]: out_tendencies[name]
                for name in self.tendency_properties
            }
            tendencies.pop("rrs")
            tendencies.pop("rss")
            tendencies.pop("rgs")

            diagnostics = {
                name.split("_", maxsplit=1)[1]: out_diagnostics[name]
                for name in self.diagnostic_properties
            }
            temporaries = {
                "rt": rt,
                "pv": pv,
                "piv": piv,
                "qsl": qsl,
                "qsi": qsi,
                "frac_tmp": frac_tmp,
                "cond_tmp": cond_tmp,
                "a": a,
                "sbar": sbar,
                "sigma": sigma,
                "q1": q1,
                "lv": lv,
                "ls": ls,
                "cph": cph,
                "criaut": criaut,
                "sigrc": sigrc,
                "rv_tmp": rv_tmp,
                "ri_tmp": ri_tmp,
                "rc_tmp": rc_tmp,
                "t_tmp": t_tmp,
            }

            # Condensation adjustments
            self.ice_adjust(
                **inputs,
                **tendencies,
                **diagnostics,
                **temporaries,
                dt=timestep.total_seconds(),
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )
