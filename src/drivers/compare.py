from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from ice3_gt4py.utils.reader import NetCDFReader
from ifs_physics_common.utils.typingx import (
    DataArray,
    DataArrayDict,
    NDArrayLikeDict,
)
from ice3_gt4py.initialisation.state_ice_adjust import KRR_MAPPING


def compare_fields(ref_path: str, run_path: str) -> Dict[str, float]:
    """Read and compare fields in reference and run datasets and write results in output.

    Args:
        ref_reader (str): path of reference dataset
        run_reader (str): path of run dataset
        output (str): output file to write comparison results
    """
    run_reader = NetCDFReader(Path(run_path))
    ref_reader = NetCDFReader(Path(ref_path))
    
    inf_error = lambda ref, run: np.max(np.abs(ref - run))
    l2_error = lambda ref, run: np.sum((ref - run)**2) / run.size    

    KEYS = [
        ("hli_hcf", "PHLI_HCF_OUT"),
        ("hli_hri", "PHLI_HRI_OUT"),
        ("hlc_hcf", "PHLC_HCF_OUT"),
        ("hlc_hrc", "PHLC_HRC_OUT"),
        ("cldfr", "PCLDFR_OUT"),
        ("ths", "PRS_OUT"),
        ("rvs", "PRS_OUT"),
        ("rcs", "PRS_OUT"),
        ("rrs", "PRS_OUT"),
        ("ris", "PRS_OUT"),
        ("rss", "PRS_OUT"),
        ("rgs", "PRS_OUT")
    ]
    
    tendencies = ["ths", "rvs", "rcs", "rrs", "ris", "rss", "rgs"]
    
    metrics = dict()

    for run_name, ref_name in KEYS:
    
        if run_name in tendencies:
            run_field = run_reader.get_field(run_name)
            ref_field = ref_reader.get_field(ref_name)[:, :, tendencies.index(run_name)]
        
        else:
            run_field = run_reader.get_field(run_name)
            ref_field = ref_reader.get_field(ref_name)

        e_inf = inf_error(ref_field, run_field)
        e_l2 = l2_error(ref_field, run_field)  
        relative_e_inf = e_inf / np.max(ref_field)
        relative_e_l2 = ref_field.size * e_l2 / np.sum(ref_field ** 2)
        
        metrics.update({
            f"{run_name}": {
                "mean_ref": np.mean(ref_field),
                "mean_run": np.mean(run_field),
                "e_inf": e_inf,
                "e_2": e_l2,
                "relative_e_inf": relative_e_inf,
                "relative_e_2": relative_e_l2
            }
       })
            
    return metrics    