"""
SIGRC Computation - Cupy Implementation

This module implements the computation of subgrid standard deviation of rc (cloud water)
using a lookup table, translated from the Fortran reference in mode_sigrc_computation.F90.

The SIGRC (sigma_rc) computation is used in the Chaboureau-Bechtold (CB) subgrid
condensation scheme to represent subgrid-scale variability of cloud water.

Process implemented:
- Compute sigma_rc from lookup table based on normalized saturation deficit

Reference:
    mode_sigrc_computation.F90
"""

import numpy as np
import numpy.typing as npt

# Global lookup table for SIGRC computation (CB scheme)
# This table represents empirical relationships for subgrid cloud water variability
SRC_1D = np.array(
    [
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0001,
        0.0002,
        0.0005,
        0.0010,
        0.0020,
        0.0039,
        0.0072,
        0.0124,
        0.0199,
        0.0301,
        0.0435,
        0.0601,
        0.0796,
        0.1014,
        0.1245,
        0.1476,
        0.1695,
        0.1888,
        0.2046,
        0.2165,
        0.2240,
        0.2274,
        0.2274,
        0.2260,
        0.2247,
        0.2239,
    ],
    dtype=np.float32,
)


def sigrc_computation(zq1: npt.ArrayLike, src_table: npt.NDArray[(34), npt.Float64]):
    """
    Compute subgrid standard deviation of rc using lookup table.

    This function implements the Chaboureau-Bechtold subgrid condensation scheme
    for cloud water variability. It uses a lookup table to compute sigma_rc based
    on the normalized saturation deficit.

    Args:
        zq1: Normalized saturation deficit field [I, J, K]
        psigrc: Output subgrid standard deviation of rc [I, J, K]
        inq1: Output index field (floor of 2*zq1) [I, J, K]
        src_table: Lookup table with 34 values
        nktb: Starting k index for computation
        nkte: Ending k index for computation
        nijb: Starting horizontal index
        nije: Ending horizontal index

    The computation:
    1. Computes index INQ1 = floor(min(100, max(-100, 2*ZQ1)))
    2. Clamps INQ2 to range [-22, 10]
    3. Performs linear interpolation in lookup table
    4. Clamps result to maximum of 1.0
    """

    # Compute initial index (floor of 2*zq1, clamped to [-100, 100])
    zq1_clamped = np.clip(2.0 * zq1, -100.0, 100.0)

    # Conversion to int
    inq1 = np.floor(zq1_clamped).astype(np.int32)

    # Clamp index to valid range for lookup table
    inq1_clamped = np.clip(inq1, -22, 10)

    # Compute interpolation weight
    zinc = 2.0 * zq1 - inq1_clamped

    # Linear interpolation in lookup table (offset by 23 to handle negative indices)
    table_idx1 = inq1_clamped + 22
    table_idx2 = inq1_clamped + 23

    # Perform linear interpolation
    sigrc_interp = (1.0 - zinc) * src_table.take(table_idx1) + zinc * src_table.take(
        table_idx2
    )

    # Clamp result to [0, 1]
    psigrc = np.minimum(1.0, sigrc_interp)

    return psigrc


def get_src_table():
    """
    Return the global lookup table for SIGRC computation.

    Returns:
        numpy array of shape (34,) containing the lookup table values
    """
    return SRC_1D.copy()
