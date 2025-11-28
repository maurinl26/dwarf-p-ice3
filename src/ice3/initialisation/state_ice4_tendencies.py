# -*- coding: utf-8 -*-
"""State initialization for ICE4_TENDENCIES microphysics component.

This module handles the allocation and initialization of atmospheric state variables
required by the ICE4_TENDENCIES component. It manages the creation of GT4Py storage
fields and their initialization from NetCDF datasets containing reference data.

The ICE4_TENDENCIES component computes microphysical tendencies for the ICE4 scheme,
including nucleation, warm processes, cold processes, and conversion between
hydrometeor species.
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

# Mapping between Python field names and Fortran variable names
KEYS = {
    "exn": "PEXN",
    "exnref": "PEXNREF",
    "rhodref": "PRHODREF",
    "pres": "PPABSM",
    "t": "PT",
    "th_t": "PTHT",
    "rv_t": "ZRS",
    "rc_t": "ZRS",
    "rr_t": "ZRS",
    "ri_t": "ZRS",
    "rs_t": "ZRS",
    "rg_t": "ZRS",
    "ci_t": "PCIT",
    "ssi": "PSSI",
    "ka": "PKA",
    "dv": "PDV",
    "ai": "PAI",
    "cj": "PCJ",
    "cf": "PCF",
    "rf": "PRF",
    "fr": "PFPR",
    "sigma_rc": "PSIGMA_RC",
    "hlc_hcf": "PHLC_HCF",
    "hlc_lcf": "PHLC_LCF",
    "hlc_hrc": "PHLC_HRC",
    "hlc_lrc": "PHLC_LRC",
    "hli_hcf": "PHLI_HCF",
    "hli_lcf": "PHLI_LCF",
    "hli_hri": "PHLI_HRI",
    "hli_lri": "PHLI_LRI",
    "theta_tnd": "PTHS",
    "rv_tnd": "PRS",
    "rc_tnd": "PRS",
    "rr_tnd": "PRS",
    "ri_tnd": "PRS",
    "rs_tnd": "PRS",
    "rg_tnd": "PRS",
    "theta_increment": "PTHS_INC",
    "rv_increment": "PRS_INC",
    "rc_increment": "PRS_INC",
    "rr_increment": "PRS_INC",
    "ri_increment": "PRS_INC",
    "rs_increment": "PRS_INC",
    "rg_increment": "PRS_INC",
    "ls_fact": "PLS_FACT",
    "lv_fact": "PLV_FACT",
    "rgsi": "PRGSI",
    "rvdepg": "PRVDEPG",
    "rsmltg": "PRSMLTG",
    "rraccsg": "PRRACCSG",
    "rsaccrg": "PRSACCRG",
    "rcrimsg": "PRCRIMSG",
    "rsrimcg": "PRSRIMCG",
}

KRR_MAPPING = {"v": 0, "c": 1, "r": 2, "i": 3, "s": 4, "g": 5}


def allocate_state_ice4_tendencies(
    domain: Tuple[int, int, int],
    backend: str = BACKEND,
    dtypes: Dict[str, type] = DTYPES,
) -> xr.Dataset:
    """Allocate GT4Py storage for all ICE4_TENDENCIES state variables.
    
    Creates zero-initialized GT4Py storage fields for all atmospheric state variables
    required by the ICE4_TENDENCIES component. This includes thermodynamic variables,
    mixing ratios for all hydrometeor species, microphysical process rates,
    cloud fraction parameters, and tendency terms.

    Args:
        domain (Tuple[int, int, int]): 3D domain shape as (ni, nj, nk) where ni, nj are
            horizontal dimensions and nk is the number of vertical levels
        backend (str, optional): GT4Py backend name (e.g., "gt:cpu_ifirst", "gt:gpu").
            Defaults to BACKEND from environment.
        dtypes (Dict[str, type], optional): Dictionary mapping type names ("float", "int", "bool")
            to numpy dtypes. Defaults to DTYPES from environment.

    Returns:
        xr.Dataset: Dictionary of allocated GT4Py storage fields with keys for:
            - Thermodynamic state: exn, exnref, rhodref, pres, t, th_t
            - Mixing ratios: rv_t, rc_t, rr_t, ri_t, rs_t, rg_t
            - Ice nuclei: ci_t (ice crystal number concentration)
            - Microphysical parameters: ssi, ka, dv, ai, cj
            - Cloud parameters: cf, rf, fr, sigma_rc
            - Subgrid parameters: hlc_hcf, hlc_lcf, hlc_hrc, hlc_lrc, hli_hcf, hli_lcf, hli_hri, hli_lri
            - Tendency terms: theta_tnd, rv_tnd, rc_tnd, rr_tnd, ri_tnd, rs_tnd, rg_tnd
            - Increment terms: theta_increment, rv_increment, rc_increment, rr_increment, ri_increment, rs_increment, rg_increment
            - Conversion factors: ls_fact, lv_fact
            - Process rates: rgsi, rvdepg, rsmltg, rraccsg, rsaccrg, rcrimsg, rsrimcg
            - Computation mask: ldcompute
            - Optional: rwetgh (wet growth rate)
    """

    def _allocate(
        shape: Tuple[int, ...],
        dtype: Literal["bool", "float", "int"],
    ) -> xr.DataArray:
        return zeros(
            shape,
            backend=backend,
            dtype=dtypes[dtype],
            aligned_index=(0, 0, 0)
        )

    allocate_b = partial[DataArray](_allocate, shape=domain, dtype="bool")
    allocate_f = partial[DataArray](_allocate, shape=domain, dtype="float")
    allocate_ij = partial[DataArray](_allocate, shape=domain[0:2], dtype="float")

    return {
        "time": datetime(year=2024, month=1, day=1),
        # Computation mask
        "ldcompute": allocate_b(),
        # Thermodynamic state
        "exn": allocate_f(),
        "exnref": allocate_f(),
        "rhodref": allocate_f(),
        "pres": allocate_f(),
        "t": allocate_f(),
        "th_t": allocate_f(),
        # Mixing ratios
        "rv_t": allocate_f(),
        "rc_t": allocate_f(),
        "rr_t": allocate_f(),
        "ri_t": allocate_f(),
        "rs_t": allocate_f(),
        "rg_t": allocate_f(),
        # Ice nuclei concentration
        "ci_t": allocate_f(),
        # Microphysical parameters
        "ssi": allocate_f(),
        "ka": allocate_f(),
        "dv": allocate_f(),
        "ai": allocate_f(),
        "cj": allocate_f(),
        # Cloud parameters
        "cf": allocate_f(),
        "rf": allocate_f(),
        "fr": allocate_f(),
        "sigma_rc": allocate_f(),
        # Subgrid cloud fraction parameters
        "hlc_hcf": allocate_f(),
        "hlc_lcf": allocate_f(),
        "hlc_hrc": allocate_f(),
        "hlc_lrc": allocate_f(),
        "hli_hcf": allocate_f(),
        "hli_lcf": allocate_f(),
        "hli_hri": allocate_f(),
        "hli_lri": allocate_f(),
        # Tendencies
        "theta_tnd": allocate_f(),
        "rv_tnd": allocate_f(),
        "rc_tnd": allocate_f(),
        "rr_tnd": allocate_f(),
        "ri_tnd": allocate_f(),
        "rs_tnd": allocate_f(),
        "rg_tnd": allocate_f(),
        # Increments
        "theta_increment": allocate_f(),
        "rv_increment": allocate_f(),
        "rc_increment": allocate_f(),
        "rr_increment": allocate_f(),
        "ri_increment": allocate_f(),
        "rs_increment": allocate_f(),
        "rg_increment": allocate_f(),
        # Conversion factors
        "ls_fact": allocate_f(),
        "lv_fact": allocate_f(),
        # Process rates
        "rgsi": allocate_f(),
        "rvdepg": allocate_f(),
        "rsmltg": allocate_f(),
        "rraccsg": allocate_f(),
        "rsaccrg": allocate_f(),
        "rcrimsg": allocate_f(),
        "rsrimcg": allocate_f(),
        # Optional fields
        "rwetgh": 0.0,  # scalar parameter
    }


def get_state_ice4_tendencies(
    domain: Tuple[int, int, int],
    dataset: xr.Dataset,
    *,
    backend: str = BACKEND,
    dtypes: Dict[str, type] = DTYPES,
) -> xr.Dataset:
    """Create and initialize an ICE4_TENDENCIES state from reference data.
    
    This is a convenience function that allocates all required storage fields and
    initializes them from a NetCDF dataset containing reference/reproducibility data.
    The dataset typically comes from Fortran reference simulations and is used for
    validation and testing.

    Args:
        domain (Tuple[int, int, int]): 3D domain shape as (ni, nj, nk)
        dataset (xr.Dataset): xarray Dataset containing reference data with Fortran
            naming conventions (e.g., PEXN, PRHODREF, ZRS, PRS, etc.)
        backend (str, optional): GT4Py backend name. Defaults to BACKEND.
        dtypes (Dict[str, type], optional): Dictionary mapping type names to numpy dtypes.
            Defaults to DTYPES.

    Returns:
        xr.Dataset: Dictionary of initialized GT4Py storage fields ready for use
            in ICE4_TENDENCIES computations
    """
    state = allocate_state_ice4_tendencies(domain, backend, dtypes)
    initialize_state_ice4_tendencies(state, dataset)
    return state


def initialize_state_ice4_tendencies(
    state: xr.Dataset,
    dataset: xr.Dataset,
) -> None:
    """Initialize ICE4_TENDENCIES state fields from a reference dataset.
    
    Populates pre-allocated GT4Py storage with data from a NetCDF dataset containing
    reference data. This function handles the mapping between Python field names and
    Fortran variable names, as well as the necessary array transpositions to convert
    from Fortran (column-major) to C (row-major) memory layout.
    
    Special handling is provided for:
    - Mixing ratio arrays (PRS, ZRS) which require indexing into the hydrometeor dimension
    - Increment arrays (PRS_INC) which require similar indexing
    - Array axis swapping to match GT4Py's expected memory layout

    Args:
        state (xr.Dataset): Pre-allocated dictionary of GT4Py storage fields to populate
        dataset (xr.Dataset): xarray Dataset containing source data with Fortran variable
            names. Must contain arrays like PEXN, PRHODREF, ZRS (mixing ratios),
            PRS (tendencies), etc.
    
    Side Effects:
        Modifies state dictionary in-place by copying data from dataset into storage fields
    """
    for name, FORTRAN_NAME in KEYS.items():
        if FORTRAN_NAME is None or FORTRAN_NAME not in dataset:
            logging.warning(f"Skipping {name}: Fortran variable {FORTRAN_NAME} not found in dataset")
            continue

        match FORTRAN_NAME:
            case "ZRS":
                # Mixing ratios at time t
                buffer = dataset[FORTRAN_NAME].values[:, :, :, KRR_MAPPING[name[-1]]]
                buffer = np.swapaxes(buffer, axis1=1, axis2=2)
                buffer = np.swapaxes(buffer, axis1=2, axis2=3)
                buffer = np.swapaxes(buffer, axis1=1, axis2=2)
            case "PRS":
                # Tendencies
                buffer = dataset[FORTRAN_NAME].values[:, :, :, KRR_MAPPING[name[-2]]]
                buffer = np.swapaxes(buffer, axis1=1, axis2=2)
                buffer = np.swapaxes(buffer, axis1=2, axis2=3)
                buffer = np.swapaxes(buffer, axis1=1, axis2=2)
            case "PRS_INC":
                # Increments
                buffer = dataset[FORTRAN_NAME].values[:, :, :, KRR_MAPPING[name[-10]]]
                buffer = np.swapaxes(buffer, axis1=1, axis2=2)
                buffer = np.swapaxes(buffer, axis1=2, axis2=3)
                buffer = np.swapaxes(buffer, axis1=1, axis2=2)
            case _:
                # Standard 3D fields
                buffer = dataset[FORTRAN_NAME].values
                if buffer.ndim == 3:
                    buffer = np.swapaxes(buffer, axis1=1, axis2=2)
                    buffer = np.swapaxes(buffer, axis1=2, axis2=3)
                    buffer = np.swapaxes(buffer, axis1=1, axis2=2)

        logging.info(f"Initializing {name} from {FORTRAN_NAME}, buffer.shape = {buffer.shape}")
        initialize_field(state[name], buffer)
