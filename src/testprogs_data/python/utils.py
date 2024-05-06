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
def get_array_double(f, count):
    """Read an array of float64

    Args:
        f (file): file to read
        count (int): count of entities to decode

    Returns:
        np.ndarray: unformatted numpy array
    """
    n_memory = np.fromfile(f, dtype=">i4", count=1)
    if n_memory != 0:
        logging.info(f"Memory {n_memory}")
        array = np.fromfile(f, dtype=">f8", count=count)
        _ = np.fromfile(f, dtype=">i4", count=1)
    else:
        array = np.empty()

    return array


def get_array_simple(f, count):
    """Read an array of float32

    Args:
        f (file): file to read
        count (int): count of entities to decode

    Returns:
        np.ndarray: unformatted numpy array
    """
    n_memory = np.fromfile(f, dtype=">i4", count=1)
    if n_memory != 0:
        logging.info(f"Memory {n_memory}")
        array = np.fromfile(f, dtype=">f4", count=count)
        _ = np.fromfile(f, dtype=">i4", count=1)
    else:
        array = np.empty()

    return array


################## For ICE ADJUST ########################
def get_dims(f):
    """Gets dimensions from header of file.
    Works only for ice_adjust.

    Args:
        f (file): file corresponding to ice_adjust unformatted data.

    Returns:
        Tuple[int]: KLON, KDUM, KLEV
    """
    dims = np.fromfile(f, dtype=">i4", count=1)
    logging.info(f"Dims={dims}")
    KLON, KDUM, KLEV = np.fromfile(f, dtype=">i4", count=3)
    _ = np.fromfile(f, dtype=">i4", count=1)

    return KLON, KDUM, KLEV


################### RAIN ICE ##############################
def get_size_info(f):
    """Get dimensions in file header. Works only for rain_ice.

    Args:
        f (file): file corresponding to rain_ice data

    Returns:
        Tuple[int]: IPROMA, ISIZE, KLON, KDUM, KLEV, KRR
    """
    # IPROMA, ISIZE
    dims = np.fromfile(f, dtype=">i4", count=1)
    logging.info(f"Dims={dims}")
    IPROMA, ISIZE = np.fromfile(f, dtype=">i4", count=2)
    _ = np.fromfile(f, dtype=">i4", count=1)

    # KLON, KDUM, KLEV, KRR
    dims = np.fromfile(f, dtype=">i4", count=1)
    logging.info(f"Dims={dims}")
    KLON, KDUM, KLEV, KRR = np.fromfile(f, dtype=">i4", count=4)
    _ = np.fromfile(f, dtype=">i4", count=1)

    return IPROMA, ISIZE, KLON, KDUM, KLEV, KRR
