# -*- coding: utf-8 -*-
from ice3_gt4py.utils.reader import NetCDFReader
from ice3_gt4py.initialisation.state_ice_adjust import KRR_MAPPING

import typer
import logging
import sys

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()

app = typer.Typer()


OUTPUT_KEYS = {
    # "exn": "PEXNREF",
    # "exnref": "PEXNREF",
    # "rhodref": "PRHODREF",
    # "pabs": "PPABSM",
    # "sigs": "PSIGS",
    # "cf_mf": "PCF_MF",
    # "rc_mf": "PRC_MF",
    # "ri_mf": "PRI_MF",
    # "th": "ZRS",
    # "rv": "ZRS",
    # "rc": "ZRS",
    # "rr": "ZRS",
    # "ri": "ZRS",
    # "rs": "ZRS",
    # "rg": "ZRS",
    "cldfr": "PCLDFR_OUT",
    # "sigqsat": None,
    # "ifr": None,
    "hlc_hrc": "PHLC_HRC_OUT",
    "hlc_hcf": "PHLC_HCF_OUT",
    "hli_hri": "PHLI_HRI_OUT",
    "hli_hcf": "PHLI_HCF_OUT",
    # "sigrc": None,
    "ths": "PRS_OUT",
    "rcs": "PRS_OUT",
    # "rrs": "PRS",
    "ris": "PRS_OUT",
    # "rss": "PRS",
    "rvs": "PRS_OUT",
    # "rgs": "PRS",
}


@app.command()
def compare(ref_path: str, run_path: str, output_path: str):

    ref = NetCDFReader(ref_path)
    run = NetCDFReader(run_path)

    with open(output_path, "w") as f:
        f.write("Set, var, lev, min, mean, max \n")

        for name, fortran_name in OUTPUT_KEYS.items():

            if fortran_name is not None:
                if fortran_name == "ZRS":
                    ref_field = ref.get_field(fortran_name)[:, :, KRR_MAPPING[name[-1]]]

                if fortran_name == "PRS":
                    ref_field = ref.get_field(fortran_name)[:, :, KRR_MAPPING[name[-2]]]

                if fortran_name == "PRS_OUT":
                    ref_field = ref.get_field(fortran_name)[:, :, KRR_MAPPING[name[-2]]]

                elif fortran_name not in ["ZRS", "PRS"]:
                    ref_field = ref.get_field(fortran_name)

            logging.info(f"Ref field, name : {name}, shape : {ref_field.shape}")

            run_field = run.get_field(name)
            logging.info(f"Run field, name : {name}, shape : {run_field.shape}")

            for lev in range(14):
                f.write(
                    f"Ref, {name}, {lev}, {ref_field[:, lev].min()}, {ref_field[:, lev].mean()}, {ref_field[:, lev].max()} \n"
                )
                f.write(
                    f"Run, {name}, {lev}, {run_field[:,:,lev].min()}, {run_field[:,:,lev + 1].mean()}, {run_field[:,:,lev + 1].max()} \n"
                )


if __name__ == "__main__":
    app()
