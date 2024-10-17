# -*- coding: utf-8 -*-
import datetime
import logging

import numpy as np
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.utils.typingx import NDArrayLikeDict
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
def allocate_gt4py_fields(
    component: ComputationalGridComponent, fields: dict
) -> NDArrayLikeDict:
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
        **component.fields_inout,
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
        NDArrayLikeDict: dictionnary of random arrays associated with their field name
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


def compare(fortran_fields: dict, gt4py_state: dict):
    """Compare fortran and gt4py field mean

    Args:
        fortran_fields (dict): output fields from fortran
        gt4py_state (dict): output fields from gt4py
    """
    absolute_differences = dict()
    for field_name, field_array in fortran_fields.items():
        logging.info(f"{field_name}")
        fortran_mean = field_array.mean()
        gt4py_mean = gt4py_state[field_name].mean()
        absolute_diff = abs(gt4py_mean - fortran_mean)[0]
        logging.info(f"{field_name}, absolute mean difference {absolute_diff}")
        absolute_differences.update({
            field_name: absolute_diff
        })
        
    return absolute_differences


def run_test(component: ComputationalGridComponent):
    """Draw random arrays and call gt4py and fortran stencils side-by-side

    Args:
        component (ComputationalGridComponent): component to test
    """

    logging.info(f"Start test {component.__class__.__name__}")
    fields = draw_fields(component)
    state_gt4py = allocate_gt4py_fields(component, fields)

    logging.info("Calling fortran field")
    fortran_output_fields = component.call_fortran_stencil(fields)

    logging.info("Calling gt4py field")
    gt4py_output_fields = component.call_gt4py_stencil(state_gt4py)

    logging.info("Compare output fields")
    absolute_differences = compare(fortran_fields=fortran_output_fields, gt4py_state=gt4py_output_fields)
    logging.info(f"End test {component.__class__.__name__}")
    
    return absolute_differences
    
    