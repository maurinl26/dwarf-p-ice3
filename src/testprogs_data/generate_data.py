# -*- coding: utf-8 -*-
import numpy as np
import logging
import sys
import os
import xarray as xr
import typer

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()

################### READ FORTRAN FILE #####################
def get_array(count):
    n_memory = np.fromfile(f, dtype=">i4", count=1)
    logging.info(f"Memory {n_memory}")
    array = np.fromfile(f, dtype=">f8", count=count)
    _ = np.fromfile(f, dtype=">i4", count=1)

    return array


def get_dims():
    dims = np.fromfile(f, dtype=">i4", count=1)
    logging.info(f"Dims={dims}")
    KLON, KDUM, KLEV = np.fromfile(f, dtype=">i4", count=3)
    _ = np.fromfile(f, dtype=">i4", count=1)

    return KLON, KDUM, KLEV


if __name__ == "__main__":

    # in main_ice_adjust.F90
    NPROMA = 32
    NGPBLKS = 296
    NGPTOT = NPROMA * NGPBLKS

    KRR = 6  # Number of species

    # in getdata_ice_adjust.F90
    IBL = 1  # File number
    FILE_PATH = f"/home/maurinl/PHYEX/tools/testprogs_data/ice_adjust/{IBL:08}.dat"
    OUTPUT_PATH = "/home/maurinl/PHYEX/tools/testprogs_data/ice_adjust/reference.nc"

    FIELD_KEYS = [
        "PRHODJ",
        "PEXNREF",
        "PRHODREF",
        "PPABSM",
        "PTHT",
        "PSIGS",
        "PMFCONV",
        "PRC_MF",
        "PRI_MF",
        "PCF_MF",
        "PTHS",
        "PRS",  # decode_array(KLON*KLEV*KRR)
        # "PRS_OUT",      # decode_array(KLON*KLEV*KRR)
        # "PSRCS_OUT",
        # "PCLDFR_OUT",
        # "ZRS",          # decode_array(KLON*KLEV*(KRR+1))
        # "ZZZ",
        # "PHLC_HRC_OUT",
        # "PHLC_HCF_OUT",
        # "PHLI_HRI_OUT",
        # "PHLI_HCF_OUT",
    ]

    output_dataset = xr.Dataset()

    ###### Loop over files ########
    # Slicing
    IOFF = 0
    while os.path.isfile(FILE_PATH):

        logging.info(f"IBL : {IBL}")
        logging.info(f"Decoding : {FILE_PATH}")
        ##### Processing file.dat #####
        with open(FILE_PATH, "r") as f:

            #  READ (IFILE) KLON, KDUM, KLEV
            KLON, KDUM, KLEV = get_dims()
            logging.info(f"KLON={KLON}, KLEV={KLEV}, KDUM={KDUM}")

            for key in FIELD_KEYS:
                logging.info(f"Decoding : {key}")

                if key in ["PRS", "PRS_OUT"]:
                    data_array = xr.DataArray(
                        data=get_array(KLON * KLEV * KRR).reshape(
                            (KLON, KLEV, KRR), order="F"
                        ),
                        dims=["IJ", "K", "Specy"],
                        coords={
                            "IJ": range(IOFF + 1, IOFF + KLON + 1),
                            "K": KLEV,
                            "Specy": ["v", "c", "r", "i", "s", "g"],
                        },
                        name=key,
                    )

                if key in ["ZRS"]:
                    data_array = xr.DataArray(
                        data=get_array(KLON * KLEV * (KRR + 1)).reshape(
                            (KLON, KLEV, KRR + 1), order="F"
                        ),
                        dims=["IJ", "K", "Specy"],
                        coords={
                            "IJ": range(IOFF + 1, IOFF + KLON + 1),
                            "K": KLEV,
                            "Specy": ["th", "v", "c", "r", "i", "s", "g"],
                        },
                        name=key,
                    )

                else:
                    data_array = xr.DataArray(
                        data=get_array(KLON * KLEV).reshape((KLON, KLEV), order="F"),
                        dims=["IJ", "K"],
                        coords={
                            "IJ": range(IOFF + 1, IOFF + KLON + 1),
                            "K": range(0, KLEV),
                        },
                        name=key,
                    )

                output_dataset.update({f"{key}": data_array})

            IBL += 1
            FILE_PATH = (
                f"/home/maurinl/PHYEX/tools/testprogs_data/ice_adjust/{IBL:08}.dat"
            )

            IOFF += KLON

    # Output
    output_dataset.to_netcdf(OUTPUT_PATH, format="NETCDF4", engine="netcdf4")
