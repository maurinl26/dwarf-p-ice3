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
    "cldfr": "PCLDFR_OUT",
    "hlc_hrc": "PHLC_HRC_OUT",
    "hlc_hcf": "PHLC_HCF_OUT",
    "hli_hri": "PHLI_HRI_OUT",
    "hli_hcf": "PHLI_HCF_OUT",
    "ths": "PRS_OUT",
    "rcs": "PRS_OUT",
    "ris": "PRS_OUT",
    "rvs": "PRS_OUT",
}


@app.command()
def compare(ref_path: str, run_path: str, output_path: str):

    ref = NetCDFReader(ref_path)
    run = NetCDFReader(run_path)

    with open(output_path, "w") as f:
        f.write("Set, var, lev, min, mean, max\n")

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
                ref_max = (
                    ref_field[:, lev].max() if ref_field[:, lev].max() > 1e-15 else 0
                )
                ref_mean = (
                    ref_field[:, lev].min() if ref_field[:, lev].min() > 1e-15 else 0
                )
                ref_min = (
                    ref_field[:, lev].mean() if ref_field[:, lev].mean() > 1e-15 else 0
                )

                run_max = (
                    run_field[:, :, lev + 1].max()
                    if run_field[:, :, lev + 1].max() > 1e-15
                    else 0
                )
                run_min = (
                    run_field[:, :, lev + 1].min()
                    if run_field[:, :, lev + 1].min() > 1e-15
                    else 0
                )
                run_mean = (
                    run_field[:, :, lev + 1].mean()
                    if run_field[:, :, lev + 1].mean() > 1e-15
                    else 0
                )

                rel_diff_max = abs(ref_max - run_max) / ref_max if ref_max != 0 else 1
                rel_diff_mean = (
                    abs(ref_mean - run_mean) / ref_mean if ref_mean != 0 else 1
                )
                rel_diff_min = abs(ref_min - run_min) / ref_min if ref_min != 0 else 1

                f.write(
                    f"Ref, {name}, {lev}, {ref_field[:, lev].min()}, {ref_field[:, lev].mean()}, {ref_field[:, lev].max()}\n"
                )
                f.write(
                    f"Run, {name}, {lev}, {run_field[:,:,lev].min()}, {run_field[:,:,lev + 1].mean()}, {run_field[:,:,lev + 1].max()}\n"
                )
                f.write(
                    f"Diff, {name}, {lev}, {rel_diff_min}, {rel_diff_mean}, {rel_diff_max}\n"
                )


if __name__ == "__main__":
    app()
