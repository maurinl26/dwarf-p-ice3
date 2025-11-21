# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from datetime import datetime
from functools import partial
from typing import Literal, Tuple, Dict
from xarray.core.dataarray import DataArray


import numpy as np
import xarray as xr
from gt4py.storage import zeros

from ..utils.allocate_state import initialize_field
from ..utils.env import DTYPES, BACKEND

KEYS = {
    "exn": "PEXNREF",
    "exnref": "PEXNREF",
    "rhodref": "PRHODREF",
    "pabs": "PPABSM",
    "sigs": "PSIGS",
    "cf_mf": "PCF_MF",
    "rc_mf": "PRC_MF",
    "ri_mf": "PRI_MF",
    "th": "ZRS",
    "rv": "ZRS",
    "rc": "ZRS",
    "rr": "ZRS",
    "ri": "ZRS",
    "rs": "ZRS",
    "rg": "ZRS",
    "cldfr": "PCLDFR_OUT",
    "sigqsat": None,
    "ifr": None,
    "hlc_hrc": "PHLC_HRC_OUT",
    "hlc_hcf": "PHLC_HCF_OUT",
    "hli_hri": "PHLI_HRI_OUT",
    "hli_hcf": "PHLI_HCF_OUT",
    "sigrc": None,
    "ths": "PRS",
    "rcs": "PRS",
    "rrs": "PRS",
    "ris": "PRS",
    "rss": "PRS",
    "rvs": "PRS",
    "rgs": "PRS",
    "ths": "PTHS",
}

KRR_MAPPING = {"h": 0, "v": 1, "c": 2, "r": 3, "i": 4, "s": 5, "g": 6}


######################### Ice Adjust ###########################
ice_adjust_fields_keys = [
    "sigqsat",
    "exnref",  # ref exner pression
    "exn",
    "rhodref",
    "pabs",  # absolute pressure at t
    "sigs",  # Sigma_s at time t
    "cf_mf",  # convective mass flux fraction
    "rc_mf",  # convective mass flux liquid mixing ratio
    "ri_mf",
    "th",
    "rv",
    "rc",
    "rr",
    "ri",
    "rs",
    "rg",
    "th_t",
    "sigqsat",
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
    "rrs",
    "ris",
    "rss",
    "rgs",
]


def allocate_state_ice_adjust(
        domain: Tuple[int, ...],
        backend: str,
        dtypes: Dict[str, type]
) -> xr.Dataset:
    """Allocate field to state keys following type (float, int, bool) and dimensions (2D, 3D).

    Args:
        computational_grid (ComputationalGrid): grid indexes
        gt4py_config (GT4PyConfig): gt4py configuration

    Returns:
        NDArrayLikeDict: dictionnary of field with associated keys for field name
    """

    def _allocate(
        domain: Tuple[int, ...],
        dtype: Literal["bool", "float", "int"],
    ) -> xr.DataArray:

        # todo : replace allocate data array by zeros
        return zeros(
            domain, backend=backend, dtype=DTYPES[dtype], aligned_index=(0,0,0)
        )

    allocate_b_ij = partial[DataArray](_allocate, shape=domain[0:2], dtype="bool")
    allocate_f = partial[DataArray](_allocate, shape=domain, units="", dtype="float")
    allocate_h = partial[DataArray](_allocate, shape=(
        domain[0],
        domain[1],
        domain[2] + 1),
        dtype="float")
    allocate_ij = partial[DataArray](_allocate, shape=domain[0:2], dtype="float")
    allocate_i_ij = partial[DataArray](_allocate, shape=domain[0:2], dtype="int")

    return {
        "time": datetime(year=2024, month=1, day=1),
        "sigqsat": allocate_f(),
        "exnref": allocate_f(),  # ref exner pression
        "exn": allocate_f(),
        "rhodref": allocate_f(),
        "pabs": allocate_f(),  # absolute pressure at t
        "sigs": allocate_f(),  # Sigma_s at time t
        "cf_mf": allocate_f(),  # convective mass flux fraction
        "rc_mf": allocate_f(),  # convective mass flux liquid mixing ratio
        "ri_mf": allocate_f(),
        "th": allocate_f(),
        "rv": allocate_f(),
        "rc": allocate_f(),
        "rr": allocate_f(),
        "ri": allocate_f(),
        "rs": allocate_f(),
        "rg": allocate_f(),
        "th_t": allocate_f(),
        "sigqsat": allocate_f(),
        "cldfr": allocate_f(),
        "ifr": allocate_f(),
        "hlc_hrc": allocate_f(),
        "hlc_hcf": allocate_f(),
        "hli_hri": allocate_f(),
        "hli_hcf": allocate_f(),
        "sigrc": allocate_f(),
        # tendencies
        "ths": allocate_f(),
        "rvs": allocate_f(),
        "rcs": allocate_f(),
        "rrs": allocate_f(),
        "ris": allocate_f(),
        "rss": allocate_f(),
        "rgs": allocate_f(),
    }


def get_state_ice_adjust(
    domain: Tuple[int, ...],
    *,
    backend: str,
    dataset: xr.Dataset,    
) -> xr.Dataset:
    """Create a state with reproductibility data set.

    Args:
        computational_grid (ComputationalGrid): grid
        gt4py_config (GT4PyConfig): config for gt4py
        keys (Dict[keys]): field names

    Returns:
        DataArrayDict: dictionnary of data array containing reproductibility data
    """
    state = allocate_state_ice_adjust(domain, BACKEND, DTYPES)
    initialize_state_ice_adjust(state, dataset)

    return state


def initialize_state_ice_adjust(
    state: xr.Dataset,
    dataset: xr.Dataset,
) -> None:
    """Initialize fields of state dictionnary with a constant field.

    Args:
        state (DataArrayDict): dictionnary of state
        gt4py_config (GT4PyConfig): configuration of gt4py
    """

    for name, FORTRAN_NAME in KEYS.items():
        match FORTRAN_NAME:
            case "PRS":
                buffer = dataset[FORTRAN_NAME].values[:,:,KRR_MAPPING[name[-2]]]
                buffer = np.swapaxes(buffer, axis1=1, axis2=2)
                buffer = np.swapaxes(buffer, axis1=2, axis2=3)
                buffer = np.swapaxes(buffer, axis1=1, axis2=2)
            case "ZRS":
                buffer = dataset[FORTRAN_NAME].values[:,:,KRR_MAPPING[name[-1]]]
                buffer = np.swapaxes(buffer, axis1=1, axis2=2)
                buffer = np.swapaxes(buffer, axis1=2, axis2=3)
                buffer = np.swapaxes(buffer, axis1=1, axis2=2)
            case _:
                buffer = dataset[FORTRAN_NAME].values
                buffer = np.swapaxes(buffer, axis1=1, axis2=2)
                buffer = np.swapaxes(buffer, axis1=2, axis2=3)
                buffer = np.swapaxes(buffer, axis1=1, axis2=2)
   
        logging.info(f"name = {name}, buffer.shape = {buffer.shape}")
        initialize_field(state[name], buffer)
