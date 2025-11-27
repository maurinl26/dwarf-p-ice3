# -*- coding: utf-8 -*-
"""State initialization for RAIN_ICE microphysics component.

This module handles the allocation and initialization of atmospheric state variables
required by the RAIN_ICE mixed-phase microphysics scheme. It manages the creation of
GT4Py storage fields and their initialization from NetCDF datasets containing reference
data.

The RAIN_ICE scheme is a comprehensive bulk microphysics parameterization that includes
warm rain processes, ice crystal formation, aggregation, riming, and sedimentation for
multiple hydrometeor categories (cloud, rain, ice, snow, graupel).
"""
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
    dtypes: str = DTYPES
) -> xr.Dataset:
    """Allocate GT4Py storage for all RAIN_ICE state variables and tendencies.
    
    Creates zero-initialized GT4Py storage fields for all atmospheric state variables
    required by the RAIN_ICE mixed-phase microphysics scheme. This includes thermodynamic
    variables, mixing ratios for all hydrometeor species, precipitation fluxes, cloud
    fraction parameters, and tendency terms.

    Args:
        domain (Tuple[int, int, int]): 3D domain shape as (ni, nj, nk) where ni, nj are
            horizontal dimensions and nk is the number of vertical levels
        backend (str, optional): GT4Py backend name. Defaults to BACKEND from environment.

    Returns:
        Dict[str, DataArray]: Dictionary of allocated GT4Py storage fields with keys for:
            - Thermodynamic state: exn, exnref, rhodref, rhodj, pabs_t, th_t, t, etc.
            - Vertical grid: dzz (layer thickness)
            - Mixing ratios: rv_t, rc_t, rr_t, ri_t, rs_t, rg_t (vapor, cloud, rain, ice, snow, graupel)
            - Ice nuclei: ci_t (ice crystal number concentration)
            - Cloud parameters: cldfr, sigs, rainfr, indep
            - Tendency terms: ths, rvs, rcs, rrs, ris, rss, rgs
            - Precipitation fluxes: fpr_c, fpr_r, fpr_i, fpr_s, fpr_g
            - Integrated precipitation: inprc, inprr, inprs, inprg
            - Subgrid parameters: hlc_*, hli_* (high/low content fractions and mixing ratios)
            - Diagnostic fields: evap3d, ssi, pthvrefzikb
            - Surface types: sea, town (optional masks)
    """

    def _allocate(
        shape: Tuple[int, ...],
        backend: str,
        dtype: Literal["bool", "float", "int"],
    ) -> xr.DataArray:
        return zeros(
            shape,
            dtypes[dtype],
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
) -> xr.Dataset:
    """Create and initialize a RAIN_ICE state from reference data.
    
    This is a convenience function that allocates all required storage fields and
    initializes them from a NetCDF dataset containing reference/reproducibility data.
    The dataset typically comes from Fortran reference simulations and is used for
    validation and testing.

    Args:
        domain (Tuple[int, int, int]): 3D domain shape as (ni, nj, nk)
        ds (xr.Dataset): xarray Dataset containing reference data with Fortran
            naming conventions (e.g., PEXNREF, PRHODREF, PRS, etc.)
        backend (str): GT4Py backend name (e.g., "gt:cpu_ifirst", "gt:gpu")

    Returns:
        xr.Dataset: Dictionary of initialized GT4Py storage fields ready for use
            in RAIN_ICE computations
    """
    state = allocate_state_rain_ice(domain, backend)
    initialize_state_rain_ice(state, ds)
    return state


def initialize_state_rain_ice(
    state: xr.Dataset,
    dataset: xr.Dataset,
) -> None:
    """Initialize RAIN_ICE state fields from a reference dataset.
    
    Populates pre-allocated GT4Py storage with data from a NetCDF dataset containing
    reference data. This function handles the mapping between Python field names and
    Fortran variable names used in the reference dataset.
    
    Special handling is provided for:
    - Mixing ratio arrays (PRS, ZRS) which require indexing into the hydrometeor dimension
      using KRR_MAPPING to extract the correct species

    Args:
        state (xr.Dataset): Pre-allocated dictionary of GT4Py storage fields to populate
        dataset (xr.Dataset): xarray Dataset containing source data with Fortran variable
            names. Must contain arrays like PEXNREF, PRHODREF, PTHT, PRS (tendencies), etc.
    
    Side Effects:
        Modifies state dictionary in-place by copying data from dataset into storage fields
    
    Note:
        This function does not perform array transposition as the reference data is
        already in the expected memory layout.
    """
    for name, FORTRAN_NAME in KEYS.items():
        if FORTRAN_NAME is None:
            continue
            
        match FORTRAN_NAME:
            case "ZRS":
                buffer = dataset[FORTRAN_NAME].values[:, :, KRR_MAPPING[name[-1]]]
            case "PRS":
                buffer = dataset[FORTRAN_NAME].values[:, :, KRR_MAPPING[name[-2]]]
            case _:
                buffer = dataset[FORTRAN_NAME].values

        logging.info(f"name = {name}, buffer.shape = {buffer.shape}")
        initialize_field(state[name], buffer)
