# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from datetime import datetime
from functools import partial
from typing import Literal, Tuple

import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray
from gt4py.storage import zeros

from ..utils.allocate_state import initialize_field
from ..utils.env import DTYPES, BACKEND

KEYS = {
    "exnref": "PEXNREF",
    "dzz": "PDZZ",
    "rhodj": "PRHODJ",
    "rhodref": "PRHODREF",
    "pabs_t": "PPABSM",
    "ci_t": "PCIT",
    "cldfr": "PCLDFR",
    "hlc_hrc": "PHLC_HRC",
    "hlc_hcf": "PHLC_HCF",
    "hli_hri": "PHLI_HRI",
    "hli_hcf": "PHLI_HCF",
    "th_t": "PTHT",
    "ths": "PTHS",
    "rcs": "PRS",
    "rrs": "PRS",
    "ris": "PRS",
    "rgs": "PRS",
    "sigs": "PSIGS",
    "sea": "PSEA",
    "town": "PTOWN",
    "inprr": "PINPRR_OUT",
    "evap3d": "PEVAP_OUT",
    "inprs": "PINPRS_OUT",
    "inprg": "PINPRG_OUT",
    "fpr": "PFPR_OUT",
    "rainfr": "ZRAINFR_OUT",
    "indep": "ZINDEP_OUT",
}

KRR_MAPPING = {"v": 0, "c": 1, "r": 2, "i": 3, "s": 4, "g": 5}


def allocate_state_rain_ice(
    domain: Tuple[int, 3],
    backend: str = BACKEND,
) -> xr.Dataset:
    """Allocate field to state keys following type (float, int, bool) and dimensions (2D, 3D).

    Args:
        computational_grid (ComputationalGrid): grid indexes
        gt4py_config (GT4PyConfig): gt4py configuration

    Returns:
        NDArrayLikeDict: dictionnary of field with associated keys for field name
    """

    def _allocate(
        shape: Tuple[int, ...],
        backend: str,
        dtype: Literal["bool", "float", "int"],
    ) -> xr.DataArray:
        return zeros(
            shape,
            DTYPES[dtype],
            backend,
            aligned_index=(0, 0, 0)
        )

    allocate_b_ij = partial[DataArray](_allocate, shape=domain[0:2], dtype="bool")
    allocate_f = partial[DataArray](_allocate, shape=domain,  dtype="float")
    allocate_h = partial[DataArray](_allocate, shape=(
        domain[0],
        domain[1],
        domain[2] + 1
    ), dtype="float")
    allocate_ij = partial[DataArray](_allocate, shape=domain, dtype="float")
    allocate_i_ij = partial[DataArray](_allocate, grid_id=domain, dtype="int")

    return {
        "time": datetime(year=2024, month=1, day=1),
        "exn": allocate_f(),
        "dzz": allocate_f(),
        "ssi": allocate_f(),
        "t": allocate_f(),
        "rhodj": allocate_f(),
        "rhodref": allocate_f(),
        "pabs_t": allocate_f(),
        "exnref": allocate_f(),
        "ci_t": allocate_f(),
        "cldfr": allocate_f(),
        "th_t": allocate_f(),
        "rv_t": allocate_f(),
        "rc_t": allocate_f(),
        "rr_t": allocate_f(),
        "ri_t": allocate_f(),
        "rs_t": allocate_f(),
        "rg_t": allocate_f(),
        "ths": allocate_f(),
        "rvs": allocate_f(),
        "rcs": allocate_f(),
        "rrs": allocate_f(),
        "ris": allocate_f(),
        "rss": allocate_f(),
        "rgs": allocate_f(),
        "fpr_c": allocate_f(),
        "fpr_r": allocate_f(),
        "fpr_i": allocate_f(),
        "fpr_s": allocate_f(),
        "fpr_g": allocate_f(),
        "inprc": allocate_ij(),
        "inprr": allocate_ij(),
        "inprs": allocate_ij(),
        "inprg": allocate_ij(),
        "evap3d": allocate_f(),
        "indep": allocate_f(),
        "rainfr": allocate_f(),
        "sigs": allocate_f(),
        "pthvrefzikb": allocate_f(),
        "hlc_hcf": allocate_f(),
        "hlc_lcf": allocate_f(),
        "hlc_hrc": allocate_f(),
        "hlc_lrc": allocate_f(),
        "hli_hcf": allocate_f(),
        "hli_lcf": allocate_f(),
        "hli_hri": allocate_f(),
        "hli_lri": allocate_f(),
        # Optional
        "fpr": allocate_f(),
        "sea": allocate_ij(),
        "town": allocate_ij(),
    }


def get_state_rain_ice(
    domain: Tuple[int, 3],
    ds: xr.Dataset,
    *,
    backend: str,
) -> None:
    """Create a state with reproductibility data set.

    Args:
        computational_grid (ComputationalGrid): grid
        gt4py_config (GT4PyConfig): config for gt4py
        keys (Dict[keys]): field names

    Returns:
        DataArrayDict: dictionnary of data array containing reproductibility data
    """
    state = allocate_state_rain_ice()
    initialize_state_rain_ice(state, ds)
    return state

# todo : remove netcdf reader
def initialize_state_rain_ice(
    state: xr.Dataset ,
    dataset: xr.Dataset,
) -> None:
    """Initialize fields of state dictionnary with a constant field.

    Args:
        state (DataArrayDict): dictionnary of state
        gt4py_config (GT4PyConfig): configuration of gt4py
    """

    for name, FORTRAN_NAME in KEYS.items():

        match FORTRAN_NAME:
            case "ZRS":
                buffer = dataset[FORTRAN_NAME].values[:,:,KRR_MAPPING[name[-1]]]
            case "PRS":
                buffer = dataset[FORTRAN_NAME].values[:,:,KRR_MAPPING[name[-2]]]
            case _:
                buffer = dataset[FORTRAN_NAME].values

        logging.info(f"name = {name}, buffer.shape = {buffer.shape}")
        initialize_field(state[name], buffer)

        if FORTRAN_NAME is not None:
                if FORTRAN_NAME in ["PSEA", "PTOWN", "PINPRR_OUT", "PINPRS_OUT", "PINPRG_OUT", "ZINPRC_OUT", "ZRAINFR_OUT", "PFPR_OUT", "ZINDEP_OUT", "PEVAP_OUT", "PCIT_OUT", "LLMICRO"]:
                    buffer = dataset[FORTRAN_NAME].values
                else:
                    buffer = dataset[FORTRAN_NAME].values                           
