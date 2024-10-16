# -*- coding: utf-8 -*-
import datetime
import logging

import numpy as np
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.utils.typingx import (
    NDArrayLikeDict,
)

from stencils.test_cloud_fraction import CloudFraction
from stencils.test_condensation import Condensation
from stencils.test_latent_heat import LatentHeat
from utils.allocate_state import allocate_state


from ice3_gt4py.initialisation.utils import initialize_field
from ice3_gt4py.phyex_common.phyex import Phyex


###### Default config for tests #######
backend = "gt:cpu_ifirst"
rebuild = True
validate_args = True

phyex = Phyex(program="AROME")

test_grid = ComputationalGrid(50, 1, 15)
dt = datetime.timedelta(seconds=1)

default_gt4py_config = GT4PyConfig(
    backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=False
)

####### Field allocation functions #######
def allocate_gt4py_fields(component: ComputationalGridComponent, fields: dict) -> NDArrayLikeDict:
    """Allocate storage for gt4py fields and
    initialize fields with given np arrays

    Args:
        component (ComputationalGridComponent): a ComputationalGridComponent with well described fields
        fields (dict): _description_

    Returns:
        NDArrayLikeDict: _description_
    """
    
    fields_metadata = {
        **component.fields_in,
        **component.fields_out,
        **component.fields_inout
    }
    state_gt4py = allocate_state(test_grid, default_gt4py_config, fields_metadata)
    for key, field_array in fields.items():
        initialize_field(state_gt4py[key], field_array)
        
    return state_gt4py


def draw_fields(component: ComputationalGridComponent) -> NDArrayLikeDict:
    """Draw random fields according to component description

    Args:
        component (ComputationalGridComponent): a ComputationalGridComponent, with
        well described fields

    Returns:
        _type_: _description_
    """
    
    return {
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

##### Test components ######
import unittest

class TestLatentHeat(unittest.TestCase):
    
    def setUp(self):
        ...
        
    def tearDown(self):
        ...
        
    def test_latent_heat(self):
        ...

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

    fields = draw_fields(component)

    logging.info("Calling fortran field")
    fortran_fields = component.call_fortran_stencil(fields)
    
    state_gt4py = allocate_gt4py_fields(component, fields)

    logging.info("Calling gt4py field")
    gt4py_fields = component.call_gt4py_stencil(state_gt4py)

    ########### Condensation #############
    logging.info("Test Condensation")
    component = Condensation(
        computational_grid=test_grid,
        phyex=phyex,
        gt4py_config=default_gt4py_config,
        fortran_script="mode_condensation.F90",
        fortran_module="mode_condensation",
        fortran_subroutine="condensation",
        gt4py_stencil="condensation"
    )
    
    fields = draw_fields(component) 

    logging.info("Calling fortran field")
    fortran_fields = component.call_fortran_stencil(fields)
    
    fields = allocate_gt4py_fields(component, fields)

    logging.info("Calling gt4py field")
    gt4py_fields = component.call_gt4py_stencil(state_gt4py)

    

    ########### CloudFraction ############
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
    
    fields = draw_fields(component)

    logging.info("Calling fortran field")
    fortran_fields = component.call_fortran_stencil(fields)
    print(fortran_fields)

    logging.info("Calling gt4py field")
    gt4py_fields = component.call_gt4py_stencil(fields)

    # Compare
