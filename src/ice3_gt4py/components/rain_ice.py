# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import timedelta
from functools import cached_property
from itertools import repeat
from typing import Dict

from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K
from ifs_physics_common.framework.storage import managed_temporary_storage
from ifs_physics_common.utils.typingx import NDArrayLikeDict, PropertyDict
from ifs_physics_common.utils.f2py import ported_method


from ice3_gt4py.components.ice4_stepping import Ice4Stepping
from ice3_gt4py.phyex_common.param_ice import (
    Sedim,
    SubgAucvRc,
    SubgAucvRi,
    SubgPRPDF,
    SubgRREvap,
    SubgRRRCAccr,
)
from ice3_gt4py.phyex_common.phyex import Phyex


class RainIce(ImplicitTendencyComponent):
    """Component for step computation"""

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

        self.phyex = phyex
        externals = self.phyex.to_externals()

        # 1. Generalites
        self.rain_ice_init = self.compile_stencil("rain_ice_init", externals)

        # 3. Initial values saving
        self.initial_values_saving = self.compile_stencil(
            "initial_values_saving", externals
        )

        # 4.1 Slow cold processes outside of ldmicro
        self.rain_ice_nucleation_pre_processing = self.compile_stencil(
            "rain_ice_nucleation_pre_processing", externals
        )
        self.ice4_nucleation = self.compile_stencil("ice4_nucleation", externals)
        self.rain_ice_nucleation_post_processing = self.compile_stencil(
            "rain_ice_nucleation_post_processing", externals
        )

        # 4.2 Computes precipitation fraction
        self.ice4_precipitation_fraction_sigma = self.compile_stencil(
            "ice4_precipitation_fraction_sigma", externals
        )
        self.ice4_precipitation_fraction_liquid_content = self.compile_stencil(
            "ice4_precipitation_fraction_liquid_content", externals
        )
        self.ice4_compute_pdf = self.compile_stencil("ice4_compute_pdf", externals)
        self.ice4_rainfr_vert = self.compile_stencil("ice4_rainfr_vert", externals)

        # 5. Tendencies computation
        # Translation note : rain_ice.F90 calls Ice4Stepping inside Ice4Pack packing operations
        self.ice4_stepping = Ice4Stepping(
            self.computational_grid, self.gt4py_config, phyex
        )

        # 8. Total tendencies
        # 8.1 Total tendencies limited by available species
        self.total_tendencies = self.compile_stencil(
            "rain_ice_total_tendencies", externals
        )
        # 8.2 Negative corrections
        self.ice4_correct_negativities = self.compile_stencil(
            "ice4_correct_negativities", externals
        )

        # 9. Compute the sedimentation source
        self.statistical_sedimentation = self.compile_stencil(
            "statistical_sedimentation", externals
        )

        self.rain_fraction_sedimentation = self.compile_stencil(
            "rain_fraction_sedimentation", externals
        )

        # 10 Compute the fog deposition
        self.fog_deposition = self.compile_stencil("fog_deposition", externals)

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "exn": {"grid": (I, J, K), "units": ""},
            "dzz": {"grid": (I, J, K), "units": ""},
            "t": {"grid": (I, J, K), "units": ""},
            "ssi": {"grid": (I, J, K), "units": ""},
            "rhodj": {"grid": (I, J, K), "units": ""},
            "rhodref": {"grid": (I, J, K), "units": ""},
            "pabs_t": {"grid": (I, J, K), "units": ""},
            "exnref": {"grid": (I, J, K), "units": ""},
            "ci_t": {"grid": (I, J, K), "units": ""},
            "cldfr": {"grid": (I, J, K), "units": ""},
            "th_t": {"grid": (I, J, K), "units": ""},
            "rv_t": {"grid": (I, J, K), "units": ""},
            "rc_t": {"grid": (I, J, K), "units": ""},
            "rr_t": {"grid": (I, J, K), "units": ""},
            "ri_t": {"grid": (I, J, K), "units": ""},
            "rs_t": {"grid": (I, J, K), "units": ""},
            "rg_t": {"grid": (I, J, K), "units": ""},
            "ths": {"grid": (I, J, K), "units": ""},
            "rvs": {"grid": (I, J, K), "units": ""},
            "rcs": {"grid": (I, J, K), "units": ""},
            "rrs": {"grid": (I, J, K), "units": ""},
            "ris": {"grid": (I, J, K), "units": ""},
            "rss": {"grid": (I, J, K), "units": ""},
            "rgs": {"grid": (I, J, K), "units": ""},
            "fpr_c": {"grid": (I, J, K), "units": ""},
            "fpr_r": {"grid": (I, J, K), "units": ""},
            "fpr_i": {"grid": (I, J, K), "units": ""},
            "fpr_s": {"grid": (I, J, K), "units": ""},
            "fpr_g": {"grid": (I, J, K), "units": ""},
            "inprc": {"grid": (I, J), "units": ""},
            "inprr": {"grid": (I, J), "units": ""},
            "inprs": {"grid": (I, J), "units": ""},
            "inprg": {"grid": (I, J), "units": ""},
            "evap3d": {"grid": (I, J, K), "units": ""},
            "indep": {"grid": (I, J, K), "units": ""},
            "rainfr": {"grid": (I, J, K), "units": ""},
            "sigs": {"grid": (I, J, K), "units": ""},
            "pthvrefzikb": {"grid": (I, J, K), "units": ""},
            "hlc_hcf": {"grid": (I, J, K), "units": ""},
            "hlc_lcf": {"grid": (I, J, K), "units": ""},
            "hlc_hrc": {"grid": (I, J, K), "units": ""},
            "hlc_lrc": {"grid": (I, J, K), "units": ""},
            "hli_hcf": {"grid": (I, J, K), "units": ""},
            "hli_lcf": {"grid": (I, J, K), "units": ""},
            "hli_hri": {"grid": (I, J, K), "units": ""},
            "hli_lri": {"grid": (I, J, K), "units": ""},
            # Optional
            "fpr": {"grid": (I, J, K), "units": ""},
            "sea": {"grid": (I, J), "units": ""},
            "town": {"grid": (I, J), "units": ""},
        }

    @cached_property
    def _tendency_properties(self) -> PropertyDict:
        return {}

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        return {}

    @cached_property
    def _temporaries(self) -> PropertyDict:
        return {}

    @ported_method(
        from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
        from_line=214,
        to_line=438,
    )
    def array_call(
        self,
        state: NDArrayLikeDict,
        timestep: timedelta,
        out_tendencies: NDArrayLikeDict,
        out_diagnostics: NDArrayLikeDict,
        overwrite_tendencies: Dict[str, bool],
    ):

        with managed_temporary_storage(
            self.computational_grid,
            *repeat(((I, J, K), "bool"), 2),
            *repeat(((I, J, K), "float"), 16),
            *repeat(((I, J), "float"), 2),
            gt4py_config=self.gt4py_config,
        ) as (
            ldmicro,
            lw3d,
            rvheni,
            lv_fact,
            ls_fact,
            sigma_rc,
            hlc_lcf,
            hlc_lrc,
            hli_lcf,
            hli_lri,
            wr_th,
            wr_v,
            wr_c,
            wr_r,
            wr_i,
            wr_s,
            wr_g,
            w3d,
            inpri,
            remaining_time,
        ):

            # KEYS
            SUBG_RC_RR_ACCR = self.phyex.param_icen.SUBG_RC_RR_ACCR
            SUBG_RR_EVAP = self.phyex.param_icen.SUBG_RR_EVAP
            SUBG_PR_PDF = self.phyex.param_icen.SUBG_PR_PDF
            SUBG_AUCV_RC = self.phyex.param_icen.SUBG_AUCV_RC
            SUBG_AUCV_RI = self.phyex.param_icen.SUBG_AUCV_RI
            LSEDIM_AFTER = self.phyex.param_icen.LSEDIM_AFTER
            LDEPOSC = self.phyex.param_icen.LDEPOSC
            SEDIM = self.phyex.param_icen.SEDIM

            # 1. Generalites
            state_rain_ice_init = {
                key: state[key]
                for key in [
                    "exn",
                    "th_t",
                    "rv_t",
                    "rc_t",
                    "rr_t",
                    "ri_t",
                    "rs_t",
                    "rg_t",
                ]
            }
            tmps_rain_ice_init = {
                "ldmicro": ldmicro,
                "ls_fact": ls_fact,
                "lv_fact": lv_fact,
            }
            self.rain_ice_init(**state_rain_ice_init, **tmps_rain_ice_init)

            # 2. Compute the sedimentation source
            state_sed = {
                key: state[key]
                for key in [
                    "rhodref",
                    "dzz",
                    "pabs_t",
                    "th_t",
                    "rcs",
                    "rrs",
                    "ris",
                    "rss",
                    "rgs",
                    "sea",
                    "town",
                    "fpr_c",
                    "fpr_r",
                    "fpr_i",
                    "fpr_s",
                    "fpr_g",
                    "inprr",
                    "inprc",
                    "inprs",
                    "inprg",
                ]
            }
            # if not LSEDIM_AFTER:
            #     if SEDIM == Sedim.STAT.value:
            #         self.statistical_sedimentation(
            #             {
            #                 **state_sed,
            #                 "inpri": inpri,
            #             }
            #         )
            #     # TODO : add split sedimentation
            #     else:
            #         raise KeyError(f"Key not in {[option.name for option in Sedim]}")

            # 3. Initial values saving
            state_initial_values_saving = {
                key: state[key]
                for key in ["th_t", "rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t"]
            }
            tmps_initial_values_saving = {
                "wr_th": wr_th,
                "wr_v": wr_v,
                "wr_c": wr_c,
                "wr_r": wr_r,
                "wr_i": wr_i,
                "wr_s": wr_s,
                "wr_g": wr_g,
            }
            # self.initial_values_saving(
            #     **state_initial_values_saving, **tmps_initial_values_saving
            # )

            # 4.1 Slow cold processes outside of ldmicro
            state_nuc_pre = {key: state[key] for key in ["exn", "ci_t"]}
            tmps_nuc_pre = {"ldmicro": ldmicro, "w3d": w3d, "ls_fact": ls_fact}
            self.rain_ice_nucleation_pre_processing(**state_nuc_pre, **tmps_nuc_pre)

            state_nuc = {
                key: state[key]
                for key in [
                    "th_t",
                    "pabs_t",
                    "rhodref",
                    "exn",
                    "t",
                    "rv_t",
                    "ci_t",
                    "ssi",
                ]
            }
            # TODO verification with Fortran signature
            tmps_nuc = {
                "ldcompute": lw3d,
                "ls_fact": ls_fact,
                "rvheni_mr": rvheni,
            }
            self.ice4_nucleation(**state_nuc, **tmps_nuc)
            self.rain_ice_nucleation_post_processing(rvs=state["rvs"], rvheni=rvheni)

            # 4.2 Computes precipitation fraction
            if (
                SUBG_RC_RR_ACCR == SubgRRRCAccr.PRFR.value
                or SUBG_RR_EVAP == SubgRREvap.PRFR.value
            ):
                if (
                    SUBG_AUCV_RC == SubgAucvRc.PDF.value
                    and SUBG_PR_PDF == SubgPRPDF.SIGM.value
                ):
                    self.ice4_precipitation_fraction_sigma(
                        sigs=state["sigs"], sigma_rc=sigma_rc
                    )
                if (
                    SUBG_AUCV_RC == SubgAucvRc.ADJU.value
                    and SUBG_AUCV_RI == SubgAucvRi.ADJU.value
                ):

                    state_lc = {
                        **{
                            key: state[key]
                            for key in [
                                "hlc_hrc",
                                "hli_hri",
                                "hlc_hcf",
                                "hli_hcf",
                                "rc_t",
                                "ri_t",
                                "cldfr",
                            ]
                        },
                        "hlc_lrc": hlc_lrc,
                        "hli_lri": hli_lri,
                        "hlc_lcf": hlc_lcf,
                        "hli_lcf": hli_lcf,
                    }

                    self.ice4_precipitation_fraction_liquid_content(**state_lc)

                state_compute_pdf = {
                    **{
                        key: state[key]
                        for key in [
                            "rhodref",
                            "rc_t",
                            "ri_t",
                            "cf",
                            "t",
                            "hlc_hcf",
                            "hlc_hrc",
                            "hli_hcf",
                            "hli_hri",
                            "rf",
                        ]
                    },
                    "hli_lri": hli_lri,
                    "hli_lcf": hli_lcf,
                    "hlc_lrc": hlc_lrc,
                    "hlc_lcf": hlc_lcf,
                    "sigma_rc": sigma_rc,
                    "ldmicro": ldmicro,
                }

                self.ice4_compute_pdf(**state_compute_pdf)

                state_rainfr_vert = {
                    **{
                        key: state[key]
                        for key in [
                            "rrs",
                            "rss",
                            "rgs",
                        ]
                    },
                    "wr_r": wr_r,
                    "wr_s": wr_s,
                    "wr_g": wr_g,
                }
                self.ice4_rainfr_vert(**state_rainfr_vert)

            # 5. Tendencies computation
            # Translation note : rain_ice.F90 calls Ice4Stepping inside Ice4Pack packing operations
            state_stepping = {
                key: state[key]
                for key in [
                    "th_t",
                    "t",
                    "rv_t",
                    "rc_t",
                    "rr_t",
                    "ri_t",
                    "rs_t",
                    "rg_t",
                    "exn",
                ]
            }

            tmps_stepping = {
                "ldmicro": ldmicro,
                "ls_fact": ls_fact,
                "lv_fact": lv_fact,
            }

            # _, _ = self.ice4_stepping({
            #     **state_stepping,
            #     **tmps_stepping
            #     }, timestep)

            # 8. Total tendencies
            # 8.1 Total tendencies limited by available species
            state_total_tendencies = {
                key: state[key]
                for key in [
                    "exnref",
                    "ths",
                    "rvs",
                    "rcs",
                    "rrs",
                    "ris",
                    "rss",
                    "rgs",
                    "rv_t",
                    "rc_t",
                    "rr_t",
                    "ri_t",
                    "rs_t",
                    "rg_t",
                ]
            }

            # TODO: replace ls, lv by ls_fact, lv_fact in signature
            tmps_total_tendencies = {
                "rvheni": rvheni,
                "ls_fact": ls_fact,
                "lv_fact": lv_fact,
                "wr_th": wr_th,
                "wr_v": wr_v,
                "wr_c": wr_c,
                "wr_r": wr_r,
                "wr_i": wr_i,
                "wr_s": wr_s,
                "wr_g": wr_g,
            }

            self.total_tendencies(**state_total_tendencies, **tmps_total_tendencies)

            # 8.2 Negative corrections
            state_neg = {
                key: state[key]
                for key in [
                    "th_t",
                    "rv_t",
                    "rc_t",
                    "rr_t",
                    "ri_t",
                    "rs_t",
                    "rg_t",
                ]
            }
            tmps_neg = {"lv_fact": lv_fact, "ls_fact": ls_fact}
            self.ice4_correct_negativities(**state_neg, **tmps_neg)

            # 9. Compute the sedimentation source
            # if LSEDIM_AFTER:
            #     if SEDIM == Sedim.STAT.value:
            #         self.statistical_sedimentation()
            #     elif SEDIM == Sedim.SPLI.value:
            #         self.upwind_sedimentation()

            #     state_frac_sed = {
            #         **{key: state[key] for key in ["rrs", "rss", "rgs"]},
            #         **{"wr_r": wr_r, "wr_s": wr_s, "wr_g": wr_g},
            #     }
            #     self.rain_fraction_sedimentation(**state_frac_sed)

            #     state_rainfr = {**{key: state[key] for key in ["prfr", "rr_t", "rs_t"]}}
            #     self.ice4_rainfr_vert(**state_rainfr)

            # 10 Compute the fog deposition
            if LDEPOSC:
                state_fog = {
                    key: state[key]
                    for key in ["rcs", "rc_t", "rhodref", "dzz", "inprc"]
                }
                self.fog_deposition(**state_fog)
