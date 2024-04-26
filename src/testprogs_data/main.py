# -*- coding: utf-8 -*-
import numpy as np
import logging
import sys
import os
import xarray as xr
import typer
from pathlib import Path

from testprogs_data.generate_data import get_array, get_dims

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()

################## APP ####################################
app = typer.Typer()


@app.command()
def extract_testsprogs_data(
    dir: str,
    output_file: str,
    nproma: int = 23,
    ngpblks: int = 296,
):

    KRR = 6

    # in getdata_ice_adjust.F90
    ibl = 0  # File number
    file_path = Path(dir, f"{ibl:08}.dat")
    output_path = Path(output_file)

    FIELD_KEYS_LIST = [
        "PRHODJ",
        "PEXNREF",
        "PRHODREF",
        "PSIGS",
        "PMFCONV",
        "PPABSM",
        "ZZZ",
        "PCF_MF",
        "PRC_MF",
        "PRI_MF",
        "ZRS",
        "PRS",
        "PTHS",
        "PRS_OUT",
        "PSRCS_OUT",
        "PCLDFR_OUT",
        "PHLC_HRC_OUT",
        "PHLC_HCF_OUT",
        "PHLI_HRI_OUT",
        "PHLI_HCF_OUT",
    ]

    output_dataset = xr.Dataset()

    ###### Loop over files ########
    # Slicing
    IOFF = 0
    while file_path.is_file():

        logging.info(f"IBL : {ibl}")
        logging.info(f"Decoding : {file_path}")
        ##### Processing file.dat #####
        with open(file_path, "r") as f:

            #  READ (IFILE) KLON, KDUM, KLEV
            KLON, KDUM, KLEV = get_dims(f)
            logging.info(f"KLON={KLON}, KLEV={KLEV}, KDUM={KDUM}")

            file_dataset = xr.Dataset()

            for key in FIELD_KEYS_LIST:
                logging.info(f"Decoding : {key}")

                if key in ["PRS", "PRS_OUT"]:
                    data_array = xr.DataArray(
                        data=get_array(f, KLON * KLEV * KRR).reshape(
                            (KLON, KLEV, KRR), order="F"
                        ),
                        dims=["IJ", "K", "Specy"],
                        coords={
                            "IJ": range(IOFF + 1, IOFF + KLON + 1),
                            "K": range(0, KLEV),
                            "Specy": ["v", "c", "r", "i", "s", "g"],
                        },
                        name=f"{key}",
                    )

                elif key in ["ZRS"]:
                    data_array = xr.DataArray(
                        data=get_array(f, KLON * KLEV * (KRR + 1)).reshape(
                            (KLON, KLEV, KRR + 1), order="F"
                        ),
                        dims=["IJ", "K", "Specy"],
                        coords={
                            "IJ": range(IOFF + 1, IOFF + KLON + 1),
                            "K": range(0, KLEV),
                            "Specy": ["th", "v", "c", "r", "i", "s", "g"],
                        },
                        name=f"{key}",
                    )

                elif key not in ["PRS", "PRS_OUT", "ZRS"]:
                    data_array = xr.DataArray(
                        data=get_array(f, KLON * KLEV).reshape((KLON, KLEV), order="F"),
                        dims=["IJ", "K"],
                        coords={
                            "IJ": range(IOFF + 1, IOFF + KLON + 1),
                            "K": range(0, KLEV),
                        },
                        name=f"{key}",
                    )

                file_dataset[key] = data_array

                if ibl == 0:
                    output_dataset = file_dataset
                else:
                    output_dataset = xr.merge([output_dataset, file_dataset])

            ibl += 1
            file_path = Path(dir, f"{ibl:08}.dat")

            IOFF += KLON

    # Output
    output_dataset.to_netcdf(output_path, format="NETCDF4", engine="netcdf4")


if __name__ == "__main__":
    app()
