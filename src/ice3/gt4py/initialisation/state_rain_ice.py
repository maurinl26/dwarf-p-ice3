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
from typing import Any, Literal, Tuple

import numpy as np
import xarray as xr
import logging
from xarray.core.dataarray import DataArray

from ...utils.allocate_state import initialize_field, get_allocator
from ...utils.env import DTYPES, BACKEND

KEYS = {
    "exnref": "PEXNREF",
    "dzz": "PDZZ",
    "rhodj": "PRHODJ",
    "rhodref": "PRHODREF",
    "pres": "PPABSM",
    "ci_t": "PCIT",
    "cldfr": "PCLDFR",
    "hlc_hrc": "PHLC_HRC",
    "hlc_hcf": "PHLC_HCF",
    "hli_hri": "PHLI_HRI",
    "hli_hcf": "PHLI_HCF",
    "th_t": "PTHT",
    "rv_t": "PRT",
    "rc_t": "PRT",
    "rr_t": "PRT",
    "ri_t": "PRT",
    "rs_t": "PRT",
    "rg_t": "PRT",
    "ths": "PTHS",
    "rvs": "PRS",
    "rcs": "PRS",
    "rrs": "PRS",
    "ris": "PRS",
    "rss": "PRS",
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
    "rainfr": "ZRAINFR_OUT",
    "indep": "ZINDEP_OUT",
    "fpr_c": "PFPR_OUT",
    "fpr_r": "PFPR_OUT",
    "fpr_i": "PFPR_OUT",
    "fpr_s": "PFPR_OUT",
    "fpr_g": "PFPR_OUT",
}

KRR_MAPPING = {"v": 0, "c": 1, "r": 2, "i": 3, "s": 4, "g": 5}


def allocate_state_rain_ice(
    domain: Tuple[int, 3],
    backend: str = BACKEND,
    dtypes: dict = DTYPES
) -> dict[str, Any]:
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

    allocator = get_allocator(backend)

    def _allocate(
        domain: Tuple[int, ...] = None,
        dtype: Literal["bool", "float", "int"] = "float",
        **kwargs
    ) -> Any:
        shape = domain or kwargs.get("shape") or kwargs.get("grid_id")
        return allocator.zeros(shape, dtype=dtypes[dtype])

    allocate_b_ij = partial[DataArray](_allocate, domain=domain[0:2], dtype="bool")
    allocate_f = partial[DataArray](_allocate, domain=domain,  dtype="float")
    allocate_h = partial[DataArray](_allocate, domain=(
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
        "pres": allocate_f(),
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
    domain: Tuple[int, int, int],
    dataset: xr.Dataset = None,
    backend: str = BACKEND,
    dtypes: dict = DTYPES,
) -> Dict[str, Any]:
    """Create and initialize a RAIN_ICE state from reference data.
    
    This is a convenience function that allocates all required storage fields and
    initializes them from a NetCDF dataset containing reference/reproducibility data.
    The dataset typically comes from Fortran reference simulations and is used for
    validation and testing.

    Args:
        domain (Tuple[int, int, int]): 3D domain shape as (ni, nj, nk)
        dataset (xr.Dataset): xarray Dataset containing reference data with Fortran
            naming conventions (e.g., PEXNREF, PRHODREF, PRS, etc.)
        backend (str): GT4Py backend name (e.g., "gt:cpu_ifirst", "gt:gpu")
        dtypes (Dict[str, type]): Dictionary mapping type names to numpy dtypes

    Returns:
        Dict[str, Any]: Dictionary of initialized GT4Py storage fields ready for use
            in RAIN_ICE computations
    """
    state = allocate_state_rain_ice(domain, backend, dtypes)
    if dataset is not None:
        initialize_state_rain_ice(state, dataset)
    return state


def initialize_state_rain_ice(
    state: Dict[str, Any],
    dataset: xr.Dataset,
) -> None:
    """Initialize RAIN_ICE state fields from a reference dataset.
    
    Populates pre-allocated GT4Py storage with data from a NetCDF dataset containing
    reference data. This function handles the mapping between Python field names and
    Fortran variable names used in the reference dataset.
    
    Special handling is provided for:
    - Mixing ratio arrays (PRS, ZRS) which require indexing into the hydrometeor dimension
      using KRR_MAPPING to extract the correct species
    - Array axis swapping to match GT4Py's expected memory layout: (ngpblks, nproma, nflevg)

    Args:
        state (Dict[str, Any]): Pre-allocated dictionary of GT4Py storage fields to populate
        dataset (xr.Dataset): xarray Dataset containing source data with Fortran variable
            names.
    
    Side Effects:
        Modifies state dictionary in-place by copying data from dataset into storage fields
    """
    for name, FORTRAN_NAME in KEYS.items():
        if FORTRAN_NAME not in dataset:
            logging.warning(f"Field {FORTRAN_NAME} for {name} missing from dataset")
            continue
            
        buffer = dataset[FORTRAN_NAME].values
        
        # Determine how to handle the dataset buffer based on its dimensionality
        if buffer.ndim == 4:
            # 4D fields (ngpblks, krr, nflevg, nproma)
            # Map name to species index
            idx = 0
            if name.endswith("_t") or name.endswith("_mf") or name.startswith("fpr_"):
                char = name[-3] if name.endswith("_t") else name[-1]
                idx = KRR_MAPPING.get(char, 0)
            elif name.endswith("s"): # Tendencies like rvs, rcs
                char = name[-2]
                idx = KRR_MAPPING.get(char, 0)
            
            # Slice and transpose: (ng, krr, nz, np) -> (ng, nz, np) -> (ng, np, nz)
            buffer = buffer[:, idx, :, :]
            buffer = np.swapaxes(buffer, axis1=1, axis2=2)
            
        elif buffer.ndim == 3:
            # 3D fields (ngpblks, nflevg, nproma)
            # Swap to (ngpblks, nproma, nflevg)
            buffer = np.swapaxes(buffer, axis1=1, axis2=2)
            
        elif buffer.ndim == 2:
            # 2D fields (ngpblks, nproma)
            # Expand to (ngpblks, nproma, 1) for vertical broadcasting
            buffer = buffer[:, :, np.newaxis]
            
        logging.info(f"Initialized {name} from {FORTRAN_NAME} (shape: {buffer.shape})")
        
        try:
            initialize_field(state[name], buffer)
        except Exception as e:
            logging.error(f"Failed to initialize {name} form {FORTRAN_NAME}: {e}")
            logging.error(f"  Destination shape: {state[name].shape}, Source shape: {buffer.shape}")
            raise
