# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from ice3_gt4py.utils.reader import NetCDFReader

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
    "ths0": "PRS",
    "rcs0": "PRS",
    "rrs0": "PRS",
    "ris0": "PRS",
    "rss0": "PRS",
    "rvs0": "PRS",
    "rgs0": "PRS",
    "ths0": "PTHS",
}

KRR_MAPPING = {"h": 0, "v": 1, "c": 2, "r": 3, "i": 4, "s": 5, "g": 6}


######################### Ice Adjust ###########################
ICE_ADJUST_INPUTS = [
    "th",
    "exn",
    "rhodref",
    "sigqsat",
    "pabs",
    "cldfr",
    "sigs",
    "rc_mf",
    "ri_mf",
    "cf_mf",
    "rv",
    "rc",
    "rr",
    "ri",
    "rs",
    "rg",
    "ths0",
    "rvs0",
    "rcs0",
    "ris0",
]


def get_state_ice_adjust(
    netcdf_reader: NetCDFReader,
    grid_shape: Tuple[int]
):
    """Create a state with reproductibility data set.

    Args:
        computational_grid (ComputationalGrid): grid
        gt4py_config (GT4PyConfig): config for gt4py
        keys (Dict[keys]): field names

    Returns:
        DataArrayDict: dictionnary of data array containing reproductibility data
    """
    state = initialize_state(
        netcdf_reader,
        ICE_ADJUST_INPUTS
    )

    # Reshape
    for key, field in state.items():
        state.update({
            key: field.reshape((10000, 1, 15))
        })

    # Cut
    nx, ny, nz = grid_shape
    for key, field in state.items():
        state.update({
            key: field[:nx, :ny, :nz]
        })

    return state


def initialize_state(
    netcdreader: NetCDFReader,
    inputs: List[str]
) -> None:
    """Initialize fields of state dictionnary with a constant field.

    Args:
        state (DataArrayDict): dictionnary of state
        gt4py_config (GT4PyConfig): configuration of gt4py
    """
    state = dict()

    for name in inputs:

        FORTRAN_NAME = KEYS[name]
        if FORTRAN_NAME is not None:
            if FORTRAN_NAME == "ZRS":
                buffer = netcdreader.get_field(FORTRAN_NAME)[
                    :, :, KRR_MAPPING[name[-1]]
                ]

            if FORTRAN_NAME == "PRS":
                buffer = netcdreader.get_field(FORTRAN_NAME)[
                    :, :, KRR_MAPPING[name[-2]]
                ]

            elif FORTRAN_NAME not in ["ZRS", "PRS"]:
                buffer = netcdreader.get_field(FORTRAN_NAME)

        else:
            dims = netcdreader.get_dims()
            n_IJ, n_K = dims["IJ"], dims["K"]
            buffer = np.zeros((n_IJ, n_K))


        logging.info(f"name = {name}, buffer.shape = {buffer.shape}")
        state.update({
            name: buffer
        })

    return state
