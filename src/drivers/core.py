# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

class NetCDFReader:
    """Simple wrapper around xarray to read NetCDF fields."""
    def __init__(self, path: Path):
        self.path = path
        self.ds = xr.open_dataset(path)
    
    def get_field(self, name: str) -> np.ndarray:
        """Get values of a variable from the dataset with consistent dimensions."""
        if name in self.ds:
            var = self.ds[name]
            # Standard output layout: (ngpblks, nproma, nflevg)
            # Reference layout: (ngpblks, nflevg, nproma) or (ngpblks, krr, nflevg, nproma)
            
            target_dims_3d = ("ngpblks", "nproma", "nflevg")
            target_dims_4d = ("ngpblks", "krr", "nproma", "nflevg")
            target_dims_4d_alt = ("ngpblks", "krr1", "nproma", "nflevg")
            
            if var.ndim == 3:
                if "nproma" in var.dims and "nflevg" in var.dims:
                    return var.transpose("ngpblks", "nproma", "nflevg").values
            elif var.ndim == 4:
                if "krr" in var.dims:
                    return var.transpose("ngpblks", "krr", "nproma", "nflevg").values
                elif "krr1" in var.dims:
                    return var.transpose("ngpblks", "krr1", "nproma", "nflevg").values
                    
            return var.values
        raise KeyError(f"Variable {name} not found in {self.path}")

def write_performance_tracking(
    exec_info: Dict[str, float], metrics: Dict[str, float], tracking_file: Path
):
    """Write performance tracking in a log file

    Args:
        exec_info (str) : gt4py exec infos
        tracking_file (str): tracking file to write in
    """

    log.info(f"Extracting exec tracking to {tracking_file}")
    
    def convert_to_float(obj):
        if isinstance(obj, dict):
            return {k: convert_to_float(v) for k, v in obj.items()}
        elif hasattr(obj, "item"): # handles numpy scalars
            return obj.item()
        elif isinstance(obj, list):
            return [convert_to_float(v) for v in obj]
        return obj

    with open(tracking_file, "w") as file:
        json.dump({
            "performances": convert_to_float(exec_info), 
            "metrics": convert_to_float(metrics)
        }, file, indent=2)


def write_dataset(state: xr.Dataset, output_path: Path):
    """Write output state to netCDF

    Args:
        state (_type_): xr.Dataset
        keys (_type_): keys to write in netCDF
        output_path (_type_): path to write field
    """
    log.info(f"Extracting state data to {output_path}")
    output_fields = xr.Dataset(state)
    output_fields.to_netcdf(output_path)
    log.info(f"Data Array written to {output_path}")


def compare_fields(
    ref_path: Path, run_path: Path, component: Literal["ice_adjust", "rain_ice"]
) -> Dict[str, float]:
    """Read and compare fields in reference and run datasets and write results in output.

    Args:
        ref_reader (str): path of reference dataset
        run_reader (str): path of run dataset
        output (str): output file to write comparison results
    """
    run_reader = NetCDFReader(run_path)
    ref_reader = NetCDFReader(ref_path)

    inf_error = lambda ref, run: np.max(np.abs(ref - run))
    l2_error = lambda ref, run: np.sum((ref - run) ** 2) / run.size

    KEYS_ICE_ADJUST = [
        ("hli_hcf", "PHLI_HCF_OUT"),
        ("hli_hri", "PHLI_HRI_OUT"),
        ("hlc_hcf", "PHLC_HCF_OUT"),
        ("hlc_hrc", "PHLC_HRC_OUT"),
        ("cldfr", "PCLDFR_OUT"),
        ("ths", "PTHS"),  # PTHS is usually separate for ice_adjust
        ("rvs", "PRS_OUT"),
        ("rcs", "PRS_OUT"),
        ("ris", "PRS_OUT"),
    ]

    KEYS_RAIN_ICE = [
        ("rainfr", "ZRAINFR_OUT"),
        ("fpr", "PFPR_OUT"),
        ("indep", "ZINDEP_OUT"),
        ("inprg", "PINPRG_OUT"),
        ("inprs", "PINPRS_OUT"),
        ("evap3d", "PEVAP_OUT"),
        ("inprr", "PINPRR_OUT"),
        ("inprc", "ZINPRC_OUT"),
        ("rvs", "PRS_OUT"),
        ("rcs", "PRS_OUT"),
        ("rrs", "PRS_OUT"),
        ("ris", "PRS_OUT"),
        ("rss", "PRS_OUT"),
        ("rgs", "PRS_OUT"),
        ("ci_t", "PCIT_OUT"),
    ]

    KEYS = KEYS_ICE_ADJUST if component == "ice_adjust" else KEYS_RAIN_ICE
    tendencies = ["ths", "rvs", "rcs", "rrs", "ris", "rss", "rgs"]

    metrics = dict()
    for run_name, ref_name in KEYS:
        if run_name in tendencies:
            run_field = run_reader.get_field(run_name)

            if component == "ice_adjust":
                if run_name == "ths":
                    ref_field = ref_reader.get_field(ref_name)
                else:
                    # species index for PRS_OUT in ice_adjust (size 6): v=0, c=1, r=2, i=3, s=4, g=5
                    # tendencies list: ["ths", "rvs", "rcs", "rrs", "ris", "rss", "rgs"]
                    # so rvs (index 1) -> PRS_OUT index 0
                    ref_field = ref_reader.get_field(ref_name)[
                        :, tendencies.index(run_name) - 1, :, :
                    ]
            elif component == "rain_ice":
                # rain_ice might have ths in PRS_OUT? check mapping.
                # If ths is not in PRS_OUT, then index - 1 is correct for v, c, etc.
                ref_field = ref_reader.get_field(ref_name)[
                    :, tendencies.index(run_name) - 1, :, :
                ]
        else:
            run_field = run_reader.get_field(run_name)
            ref_field = ref_reader.get_field(ref_name)

        log.info(
            f"Field {run_name}, ref : {ref_field.shape}, run : {run_field.shape}"
        )
        e_inf = inf_error(ref_field, run_field)
        e_l2 = l2_error(ref_field, run_field)
        relative_e_inf = e_inf / np.max(ref_field)
        relative_e_l2 = ref_field.size * e_l2 / np.sum(ref_field**2)

        metrics.update(
            {
                f"{run_name}": {
                    "mean_ref": np.mean(ref_field),
                    "mean_run": np.mean(run_field),
                    "e_inf": e_inf,
                    "e_2": e_l2,
                    "relative_e_inf": relative_e_inf,
                    "relative_e_2": relative_e_l2,
                }
            }
        )

    return metrics
