# -*- coding: utf-8 -*-
import numpy as np
import logging
import sys
import os
import xarray as xr
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()

################### READ FORTRAN FILE #####################
def get_array(f, count):
    n_memory = np.fromfile(f, dtype=">i4", count=1)
    if n_memory != 0:
        logging.info(f"Memory {n_memory}")
        array = np.fromfile(f, dtype=">f8", count=count)
        _ = np.fromfile(f, dtype=">i4", count=1)
    else:
        array = np.empty()

    return array


def get_dims(f):
    dims = np.fromfile(f, dtype=">i4", count=1)
    logging.info(f"Dims={dims}")
    KLON, KDUM, KLEV = np.fromfile(f, dtype=">i4", count=3)
    _ = np.fromfile(f, dtype=">i4", count=1)

    return KLON, KDUM, KLEV
