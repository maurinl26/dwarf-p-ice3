# -*- coding: utf-8 -*-
import datetime
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import typer
import xarray as xr

import jax.numpy as jnp
from ice3.jax.ice_adjust import IceAdjustJAX
from ice3.gt4py.initialisation.state_ice_adjust import get_state_ice_adjust
from ice3.phyex_common.phyex import Phyex
from ice3.utils.env import ROOT_PATH
from ice3.utils.env import ALL_BACKENDS

from drivers.core import write_performance_tracking, compare_fields


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
log = logging.getLogger(__name__)

app = typer.Typer()

######################## GT4Py drivers #######################
@app.command(name="ice-adjust")
def ice_adjust(
    domain: Tuple[int, int, int] = (10000, 1, 50),
    dataset: Path = Path(ROOT_PATH.parent, "data", "ice_adjust.nc"),
    output_path: Path = Path(ROOT_PATH.parent, "data", "ice_adjust_run.nc"),
    tracking_file: Path = Path(ROOT_PATH.parent, "ice_adjust_run.json"),
    backend: str = "jax",
    rebuild: bool = True,
    validate_args: bool = False,
):
    """Run ice_adjust component"""

    ################## Domain ################
    log.info("Initializing grid ...")
    dt = datetime.timedelta(seconds=1)

    ################## Phyex #################
    log.info("Initializing Phyex ...")
    phyex = Phyex("AROME")

    ######## Instanciation + compilation #####
    ice_adjust = None
    if backend in ALL_BACKENDS and backend != "jax":
        from ice3.gt4py.ice_adjust import IceAdjust

        log.info(f"Compilation for IceAdjust stencils")
        start_time = time.time()
        ice_adjust = IceAdjust()
        elapsed_time = time.time() - start_time
        log.info(f"Compilation duration for IceAdjust : {elapsed_time} s")

    ####### Create state for IceAdjust #######
    log.info("Getting state for IceAdjust")
    ds = xr.load_dataset(dataset)
    
    # Infer domain from dataset if possible
    if "ngpblks" in ds.sizes and "nproma" in ds.sizes and "nflevg" in ds.sizes:
        inferred_domain = (ds.sizes["ngpblks"], ds.sizes["nproma"], ds.sizes["nflevg"])
        if inferred_domain != domain:
            log.info(f"Inferred domain {inferred_domain} from dataset, overriding user domain {domain}")
            domain = inferred_domain

    # Use numpy backend for state initialization if JAX to avoid compatibility issues
    state_backend = "numpy" if backend == "jax" else backend
    state = get_state_ice_adjust(domain, state_backend, ds)

    # TODO: decorator for tracking
    match backend:
        case "jax":
            log.info("Running with JAX backend")
            ice_adjust_jax = IceAdjustJAX(phyex=phyex, jit=True)
            
            # Unpack state to JAX arguments
                # Note: using .values or trusting implicit conversion
            args = {
                "sigqsat": state["sigqsat"],
                "pabs": state["pabs"],
                "sigs": state["sigs"],
                "th": state["th"],
                "exn": state["exn"],
                "exn_ref": state["exnref"],
                "rho_dry_ref": state["rhodref"],
                "rv": state["rv"],
                "rc": state["rc"],
                "ri": state["ri"],
                "rr": state["rr"],
                "rs": state["rs"],
                "rg": state["rg"],
                "cf_mf": state["cf_mf"],
                "rc_mf": state["rc_mf"],
                "ri_mf": state["ri_mf"],
                "rvs": state["rvs"],
                "rcs": state["rcs"],
                "ris": state["ris"],
                "ths": state["ths"],
                "timestep": dt.total_seconds(),
            }

            start = time.time()
            results = ice_adjust_jax(**args)
            # Block until ready for accurate timing
            results[0].block_until_ready()
            stop = time.time()
            elapsed_time = stop - start
            log.info(f"Execution duration for IceAdjust : {elapsed_time} s")
            # Map results back to state
            (
            t_out, rv_out, rc_out, ri_out, cldfr_out, 
            hlc_hrc_out, hlc_hcf_out, hli_hri_out, hli_hcf_out, 
            cph_out, lv_out, ls_out, 
            rvs_out, rcs_out, ris_out, ths_out
            ) = results

            # Update state variables
            state["th"] = t_out # Using updated T/Th
            state["rv"] = rv_out
            state["rc"] = rc_out
            state["ri"] = ri_out
            state["cldfr"] = cldfr_out
            state["hlc_hrc"] = hlc_hrc_out
            state["hlc_hcf"] = hlc_hcf_out
            state["hli_hri"] = hli_hri_out
            state["hli_hcf"] = hli_hcf_out
            state["rvs"] = rvs_out
            state["rcs"] = rcs_out
            state["ris"] = ris_out
            state["ths"] = ths_out
        
            elapsed_time = stop - start

        case "fortran":
            from _phyex_wrapper import ice_adjust

            ice_adjust(
                timestep=dt.total_seconds(),
                krr=krr,
                sigqsat=state["sigqsat"],
                pabs=state["pabs"],
                sigs=state["sigs"],
                th=state["th"],
                exn=state["exn"],
                exn_ref=state["exnref"],
                rho_dry_ref=state["rhodref"],
                rv=state["rv"],
                rc=state["rc"],
                ri=state["ri"],
                rr=state["rr"],
                rs=state["rs"],
                rg=state["rg"],
                cf_mf=state["cf_mf"],
                rc_mf=state["rc_mf"],
                ri_mf=state["ri_mf"],
                rvs=state["rvs"],
                rcs=state["rcs"],
                ris=state["ris"],
                ths=state["ths"],
                cldfr=state["cldfr"],
                icldfr=state["icldfr"],
                wcldfr=state["wcldfr"]
            )

        case _:
            log.info("Running with gt4py backend")
            start = time.time()
            ice_adjust(state, dt)
            stop = time.time()
            elapsed_time = stop - start

    log.info(f"Execution duration for IceAdjust : {elapsed_time} s")

    #################### Write dataset ######################
    log.info(f"Extracting state data to {output_path}")
    data_vars = {}
    for k, v in state.items():
        if k == "time":
            continue
        if hasattr(v, "shape"):
            if len(v.shape) == 3:
                data_vars[k] = (["ngpblks", "nproma", "nflevg"], np.asarray(v))
            elif len(v.shape) == 2:
                data_vars[k] = (["ngpblks", "nproma"], np.asarray(v))
            else:
                data_vars[k] = (["dim_" + str(i) for i in range(len(v.shape))], np.asarray(v))
        else:
            data_vars[k] = v
    xr.Dataset(data_vars).to_netcdf(output_path)

    ############### Compute differences per field ###########
    metrics = compare_fields(dataset, output_path, "ice_adjust")

    ####################### Tracking ########################
    write_performance_tracking({"execution_time": elapsed_time}, metrics, tracking_file)


@app.command(name="rain-ice")
def rain_ice(
    domain: Tuple[int, int, int] = (5000, 1, 15),
    dataset: Path = Path(ROOT_PATH.parent, "data", "rain_ice.nc"),
    output_path: Path = Path(ROOT_PATH.parent, "data", "rain_ice_run.nc"),
    tracking_file: Path = Path(ROOT_PATH.parent, "rain_ice_run.json"),
    backend: str = "jax",
    rebuild: bool = True,
    validate_args: bool = False,
):
    """Run rain_ice component"""

    ################## Grid ##################
    log.info("Initializing grid ...")
    dt = datetime.timedelta(seconds=1)

    ################## Phyex #################
    log.info("Initializing Phyex ...")
    phyex = Phyex("AROME")

    ######## Backend and gt4py config #######
    log.info(f"With backend {backend}")

    ######## Instanciation + compilation #####
    rain_ice = None
    if backend != "jax":
        log.info(f"Compilation for RainIce stencils")
        start = time.time()
        from ice3.gt4py.rain_ice import RainIce
        rain_ice = RainIce()
        stop = time.time()
        elapsed_time = stop - start
        log.info(f"Compilation duration for RainIce : {elapsed_time} s")

    ####### Create state for AroAdjust #######
    log.info("Getting state for RainIce")
    ds = xr.load_dataset(dataset)
    
    # Infer domain from dataset if possible
    if "ngpblks" in ds.sizes and "nproma" in ds.sizes and "nflevg" in ds.sizes:
        inferred_domain = (ds.sizes["ngpblks"], ds.sizes["nproma"], ds.sizes["nflevg"])
        if inferred_domain != domain:
            log.info(f"Inferred domain {inferred_domain} from dataset, overriding user domain {domain}")
            domain = inferred_domain

    from ice3.gt4py.initialisation.state_rain_ice import get_state_rain_ice
    state_backend = "numpy" if backend == "jax" else backend
    state = get_state_rain_ice(domain, ds, backend=state_backend)

    ###### Launching RainIce #################
    log.info(f"Launching RainIce with backend {backend}")
    start = time.time()
    if backend == "jax":
        from ice3.jax.rain_ice import RainIceJAX
        rain_ice_jax = RainIceJAX(constants=phyex.to_externals())
        
        # Prepare state for JAX
        # state is from get_state_rain_ice, which returns storage
        # We need to extract values, skipping non-numeric fields like 'time'
        jax_state = {k: jnp.asarray(v) for k, v in state.items() if k != "time"}
        
        updated_state, diags = rain_ice_jax(jax_state, dt.total_seconds())
        
        # Sync back
        for k, v in updated_state.items():
            if k in state:
                try:
                    val_np = np.asarray(v)
                    if val_np.shape == state[k].shape:
                        state[k][...] = val_np
                    else:
                        # Try to squeeze if 1 is present
                        squeezed = np.squeeze(val_np)
                        if squeezed.shape == state[k].shape:
                             state[k][...] = squeezed
                        else:
                             log.debug(f"Skipping sync back for {k}: incompatible shapes {val_np.shape} vs {state[k].shape}")
                except Exception as e:
                    log.debug(f"Error syncing back {k}: {e}")
        stop = time.time()
    else:
        tends, diags = rain_ice(state, dt)
        stop = time.time()
    
    elapsed_time = stop - start
    log.info(f"Execution duration for RainIce : {elapsed_time} s")

    log.info(f"Extracting state data to {output_path}")
    data_vars = {}
    for k, v in state.items():
        if k == "time":
            continue
        if hasattr(v, "shape"):
            if len(v.shape) == 3:
                data_vars[k] = (["ngpblks", "nproma", "nflevg"], np.asarray(v))
            elif len(v.shape) == 2:
                data_vars[k] = (["ngpblks", "nproma"], np.asarray(v))
            else:
                data_vars[k] = (["dim_" + str(i) for i in range(len(v.shape))], np.asarray(v))
        else:
            data_vars[k] = v
    xr.Dataset(data_vars).to_netcdf(output_path)

    ################# Metrics and Performance tracking ############
    metrics = compare_fields(dataset, output_path, "rain_ice")
    write_performance_tracking({"execution_time": elapsed_time}, metrics, tracking_file)


if __name__ == "__main__":
    app()
