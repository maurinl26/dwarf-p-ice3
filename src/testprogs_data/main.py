# -*- coding: utf-8 -*-
from typing import Literal, Union
import numpy as np
import logging
import sys
import os
import xarray as xr
import typer
from pathlib import Path
import yaml

from testprogs_data.generate_data import get_array, get_dims

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()

################## APP ####################################
app = typer.Typer()


@app.command()
def extract_testsprogs_data(
    dir: str, output_file: str, conf: str, dataset: str, krr: int = 6
):
    """Extract unformatted fortran datasets to netcdf via xarray

    Args:
        dir (str): directory with raw data (.dat files)
        output_file (str): name of output file
        conf (str): configuration yaml with fields names
        dataset (str): dataset to extract, whether ice_adjust or rain_ice
        krr (int, optional): number of microphysical species. Defaults to 6.
    """

    KRR = krr

    output_path = Path(dir, output_file)

    with open(Path(conf), "r") as f:
        conf = yaml.safe_load(f)
        FIELD_KEYS_LIST = conf[dataset]

    logging.info(f"{FIELD_KEYS_LIST}")

    ###### Init loop ##############
    # in getdata_ice_adjust.F90
    ibl = 0  # File number
    file_path = Path(dir, f"{ibl:08}.dat")

    logging.info(
        f"Init decoding : indice={ibl}, file_path={file_path}, is_file={file_path.is_file()}"
    )

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

            if dataset == "ice_adjust":
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
                            data=get_array(f, KLON * KLEV).reshape(
                                (KLON, KLEV), order="F"
                            ),
                            dims=["IJ", "K"],
                            coords={
                                "IJ": range(IOFF + 1, IOFF + KLON + 1),
                                "K": range(0, KLEV),
                            },
                            name=f"{key}",
                        )

                    file_dataset[key] = data_array

            if dataset == "rain_ice":

                for key in FIELD_KEYS_LIST:
                    logging.info(f"Decoding : {key}")

                    if key in [
                        "PSEA",
                        "PTOWN",
                        "ZINPRC_OUT",
                        "PINPRR_OUT",
                        "PINPRS_OUT",
                        "PINPRG_OUT",
                    ]:
                        data_array = xr.DataArray(
                            data=get_array(f, KLON),
                            dims=["IJ", "K", "Specy"],
                            coords={
                                "IJ": range(IOFF + 1, IOFF + KLON + 1),
                            },
                            name=f"{key}",
                        )

                    elif key in ["PRT", "PRS", "PRS_OUT", "PFPR_OUT"]:
                        data_array = xr.DataArray(
                            data=get_array(f, KLON * KLEV * KRR).reshape(
                                (KLON, KLEV, KRR + 1), order="F"
                            ),
                            dims=["IJ", "K", "Specy"],
                            coords={
                                "IJ": range(IOFF + 1, IOFF + KLON + 1),
                                "K": range(0, KLEV),
                                "Specy": ["v", "c", "r", "i", "s", "g"],
                            },
                            name=f"{key}",
                        )

                    elif key not in [
                        "PRT",
                        "PRS",
                        "PRS_OUT",
                        "PFPR_OUT",
                        "PSEA",
                        "PTOWN",
                        "ZINPRC_OUT",
                        "PINPRR_OUT",
                        "PINPRS_OUT",
                        "PINPRG_OUT",
                    ]:
                        data_array = xr.DataArray(
                            data=get_array(f, KLON * KLEV).reshape(
                                (KLON, KLEV), order="F"
                            ),
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
