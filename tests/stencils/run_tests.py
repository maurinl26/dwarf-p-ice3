# -*- coding: utf-8 -*-
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
import datetime
import sys
import logging

import numpy as np

from ice3_gt4py.phyex_common.phyex import Phyex

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid

from stencils.test_cloud_fraction import CloudFraction
from stencils.test_condensation import Condensation
from stencils.test_latent_heat import LatentHeat

### Fortran dims
NIT = 50
NJT = 1
NKT = 15
NIJT = NIT * NJT

# Fortran indexing from 1 to end index
NKB, NKE = 1, NKT

# Fortran indexing from 1 to end index
NIJB, NIJE = 1, NIJT

backend = "gt:cpu_ifirst"
rebuild = True
validate_args = True

phyex = Phyex(program="AROME")

test_grid = ComputationalGrid(50, 1, 15)
dt = datetime.timedelta(seconds=1)

default_gt4py_config = GT4PyConfig(
    backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=False
)

if __name__ == "__main__":

    logging.info("Test Latent Heat")
    component = LatentHeat(
        computational_grid=test_grid,
        phyex=phyex,
        gt4py_config=default_gt4py_config,
        fortran_script="mode_thermo.F90",
        fortran_module="mode_thermo",
        fortran_subroutine="latent_heat",
        gt4py_stencil="thermodynamic_fields",
    )

    fields = {
        **{
            key: np.array(
                np.random.rand(component.array_shape[0], component.array_shape[1]),
                "f",
                order="F",
            )
            for key in component.fields_in.keys()
        },
        **{
            key: np.array(
                np.random.rand(component.array_shape[0], component.array_shape[1]),
                "f",
                order="F",
            )
            for key in component.fields_inout.keys()
        },
        **{
            key: np.array(
                np.random.rand(component.array_shape[0], component.array_shape[1]),
                "f",
                order="F",
            )
            for key in component.fields_out.keys()
        },
    }

    logging.info("Calling fortran field")
    fortran_fields = component.call_fortran_stencil(fields)
    print(fortran_fields)

    logging.info("Calling gt4py field")
    gt4py_fields = component.call_gt4py_stencil(fields)

    logging.info("Test CloudFraction")

    component = CloudFraction(
        computational_grid=test_grid,
        phyex=phyex,
        gt4py_config=default_gt4py_config,
        fortran_script="mode_cloud_fraction.F90",
        fortran_module="mode_cloud_fraction",
        fortran_subroutine="cloud_fraction",
        gt4py_stencil="cloud_fraction",
    )

    logging.info(f"Component array shape {component.array_shape}")
    logging.info(f"dtype : {type(component.array_shape[0])}")

    fields = {
        **{
            key: np.array(
                np.random.rand(component.array_shape[0], component.array_shape[1]),
                "f",
                order="F",
            )
            for key in component.fields_in.keys()
        },
        **{
            key: np.array(
                np.random.rand(component.array_shape[0], component.array_shape[1]),
                "f",
                order="F",
            )
            for key in component.fields_inout.keys()
        },
        **{
            key: np.array(
                np.random.rand(component.array_shape[0], component.array_shape[1]),
                "f",
                order="F",
            )
            for key in component.fields_out.keys()
        },
    }

    logging.info("Calling fortran field")
    fortran_fields = component.call_fortran_stencil(fields)
    print(fortran_fields)

    logging.info("Calling gt4py field")
    gt4py_fields = component.call_gt4py_stencil(fields)

    # Compare

    ####
    component = Condensation(
        computational_grid=test_grid,
        phyex=phyex,
        gt4py_config=default_gt4py_config,
        fortran_script="mode_condensation.F90",
        fortran_module="mode_condensation",
        fortran_subroutine="condensation",
        gt4py_stencil="condensation",
    )
