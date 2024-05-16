# -*- coding: utf-8 -*-
from ice3_gt4py.utils.reader import NetCDFReader
from ice3_gt4py.initialisation.state_ice_adjust import KEYS, KRR_MAPPING

import typer

app = typer.Typer()


@app.command()
def compare(ref_path: str, run_path: str, output_path: str):

    ref = NetCDFReader(ref_path)
    run = NetCDFReader(run_path)

    with open(output_path, "w") as f:
        f.write("Set, var, lev, min, mean, max \n")

        for name, fortran_name in KEYS.items():

            if fortran_name is not None:
                if fortran_name == "ZRS":
                    ref_field = ref.get_field(fortran_name)[:, :, KRR_MAPPING[name[-1]]]

                if fortran_name == "PRS":
                    ref_field = ref.get_field(fortran_name)[:, :, KRR_MAPPING[name[-2]]]

                elif fortran_name not in ["ZRS", "PRS"]:
                    ref_field = ref.get_field(fortran_name)

            run_field = run.get_field(name)

            for lev in range(14):
                f.write(
                    f"Ref, {name}, {lev}, {ref_field[:, lev].min()}, {ref_field[:, lev].mean()}, {ref_field[:, lev].max()} \n"
                )
                f.write(
                    f"Run, {name}, {lev}, {run_field[:,:,lev].min()}, {run_field[:,:,lev].mean()}, {run_field[:,:,lev].max()} \n"
                )


if __name__ == "__main__":
    app()

    ref_path = "/home/maurinl/install/dwarf-p-ice3/data/ice_adjust/reference.nc"
    run_path = "/home/maurinl/install/dwarf-p-ice3/output/run.nc"
