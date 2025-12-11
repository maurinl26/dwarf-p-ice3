# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from functools import partial
from itertools import repeat
from typing import Tuple, Dict
from numpy.typing import NDArray
from gt4py.cartesian.gtscript import stencil

from ..phyex_common.phyex import Phyex
from ..utils.env import DTYPES, BACKEND
from ..utils.storage import managed_temporaries

log  = logging.getLogger(__name__)

class IceAdjust:
    """Implicit Tendency Component calling
    ice_adjust : saturation adjustment of temperature and mixing ratios

    ice_adjust stencil is ice_adjust.F90 in PHYEX
    """

    def __init__(
        self,
        phyex: Phyex = Phyex("AROME"),
        dtypes: Dict = DTYPES,
        backend: str = BACKEND,
    ) -> None:

        externals = phyex.externals
        externals.update({"OCND2": False})

        compile_stencil = partial(stencil,
            backend=backend,
            externals=externals,
            dtypes=dtypes,
        )

        from ..stencils.ice_adjust import ice_adjust

        self.ice_adjust = compile_stencil(
            name="ice_adjust",
            definition=ice_adjust
        )

        log.info(
            f"Phyex Keys"
            f"SUBG_COND : {phyex.nebn.LSUBG_COND}"
            f"SUBG_MF_PDF : {phyex.param_icen.SUBG_MF_PDF}" 
            f"SIGMAS : {phyex.nebn.LSIGMAS}"
            f"LMFCONV : {phyex.LMFCONV}"
        )

    def __call__(self,
                 sigqsat: NDArray,
                 exn: NDArray,
                 exnref: NDArray,
                 rhodref: NDArray,
                 pabs: NDArray,
                 sigs: NDArray,
                 cf_mf: NDArray,
                 rc_mf: NDArray,
                 ri_mf: NDArray,
                 th: NDArray,
                 rv: NDArray,
                 rc: NDArray,
                 rr: NDArray,
                 ri: NDArray,
                 rs: NDArray,
                 rg: NDArray,
                 cldfr: NDArray,
                 hlc_hrc: NDArray,
                 hlc_hcf: NDArray,
                 hli_hri: NDArray,
                 hli_hcf: NDArray,
                 sigrc: NDArray,
                 ths: NDArray,
                 rvs: NDArray,
                 rcs:NDArray,
                 ris: NDArray,
                 timestep: float,
                 domain: Tuple[int, ...] ,
                 exec_info: Dict,
                 validate_args: bool = False,
                 ):

        with managed_temporaries(
            *repeat((domain, "float"), 4)
        ) as (
            t,
            lv,
            ls,
            cph
        ):

            self.ice_adjust(
                sigqsat=sigqsat,
                exn=exn,
                exnref=exnref,
                rhodref=rhodref,
                pabs=pabs,
                sigs=sigs,
                cf_mf=cf_mf,
                rc_mf=rc_mf,
                ri_mf=ri_mf,
                th=th,
                rv=rv,
                rc=rc,
                rr=rr,
                ri=ri,
                rs=rs,
                rg=rg,
                cldfr=cldfr,
                hlc_hrc=hlc_hrc,
                hlc_hcf=hlc_hcf,
                hli_hri=hli_hri,
                hli_hcf=hli_hcf,
                sigrc=sigrc,
                ths=ths,
                rvs=rvs,
                rcs=rcs,
                ris=ris,
                lv=lv,
                ls=ls,
                cph=cph,
                t=t,
                dt=timestep,
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
            )
