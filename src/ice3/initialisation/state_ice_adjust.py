# -*- coding: utf-8 -*-
"""State initialization for ICE_ADJUST microphysics component.

This module handles the allocation and initialization of atmospheric state variables
required by the ICE_ADJUST cloud adjustment scheme. It manages the creation of GT4Py
storage fields and their initialization from NetCDF datasets containing reference data.

The ICE_ADJUST scheme adjusts temperature and vapor/cloud water content to ensure
thermodynamic consistency and saturation equilibrium.
"""
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
        backend: str = BACKEND,
        dtypes: dict = DTYPES
) -> xr.Dataset:
    """Allocate GT4Py storage for all ICE_ADJUST state variables and tendencies.
    
    Creates zero-initialized GT4Py storage fields for all atmospheric state variables
    required by the ICE_ADJUST cloud adjustment scheme. This includes thermodynamic
    variables (temperature, pressure), mixing ratios for all hydrometeor species,
    cloud fraction parameters, and tendency terms.

    Args:
        domain (Tuple[int, ...]): 3D domain shape as (ni, nj, nk) where ni, nj are
            horizontal dimensions and nk is the number of vertical levels
        backend (str): GT4Py backend name (e.g., "gt:cpu_ifirst", "gt:gpu")
        dtypes (Dict[str, type]): Dictionary mapping type names ("float", "int", "bool")
            to numpy dtypes

    Returns:
        Dict[str, DataArray]: Dictionary of allocated GT4Py storage fields with keys for:
            - Thermodynamic state: exn, exnref, rhodref, pabs, th, etc.
            - Mixing ratios: rv, rc, rr, ri, rs, rg (vapor, cloud, rain, ice, snow, graupel)
            - Cloud parameters: cldfr, cf_mf, sigqsat, sigs, etc.
            - Tendency terms: ths, rvs, rcs, rrs, ris, rss, rgs
            - Subgrid parameters: hlc_hrc, hlc_hcf, hli_hri, hli_hcf, etc.
    """

    def _allocate(
        domain: Tuple[int, ...],
        dtype: Literal["bool", "float", "int"],
    ) -> xr.DataArray:

        # todo : replace allocate data array by zeros
        return zeros(
            domain, backend=backend, dtype=dtypes[dtype], aligned_index=(0,0,0)
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
    """Create and initialize an ICE_ADJUST state from reference data.
    
    This is a convenience function that allocates all required storage fields and
    initializes them from a NetCDF dataset containing reference/reproducibility data.
    The dataset typically comes from Fortran reference simulations and is used for
    validation and testing.

    Args:
        domain (Tuple[int, ...]): 3D domain shape as (ni, nj, nk)
        backend (str): GT4Py backend name (e.g., "gt:cpu_ifirst", "gt:gpu")
        dataset (xr.Dataset): xarray Dataset containing reference data with Fortran
            naming conventions (e.g., PEXNREF, PRHODREF, ZRS, PRS, etc.)

    Returns:
        xr.Dataset: Dictionary of initialized GT4Py storage fields ready for use
            in ICE_ADJUST computations
    """
    state = allocate_state_ice_adjust(domain, BACKEND, DTYPES)
    initialize_state_ice_adjust(state, dataset)

    return state


def initialize_state_ice_adjust(
    state: xr.Dataset,
    dataset: xr.Dataset,
) -> None:
    """Initialize ICE_ADJUST state fields from a reference dataset.
    
    Populates pre-allocated GT4Py storage with data from a NetCDF dataset containing
    reference data. This function handles the mapping between Python field names and
    Fortran variable names, as well as the necessary array transpositions to convert
    from Fortran (column-major) to C (row-major) memory layout.
    
    Special handling is provided for:
    - Mixing ratio arrays (PRS, ZRS) which require indexing into the hydrometeor dimension
    - Array axis swapping to match GT4Py's expected memory layout

    Args:
        state (xr.Dataset): Pre-allocated dictionary of GT4Py storage fields to populate
        dataset (xr.Dataset): xarray Dataset containing source data with Fortran variable
            names. Must contain arrays like PEXNREF, PRHODREF, ZRS (mixing ratios),
            PRS (tendencies), etc.
    
    Side Effects:
        Modifies state dictionary in-place by copying data from dataset into storage fields
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
