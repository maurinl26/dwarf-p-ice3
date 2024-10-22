# -*- coding: utf-8 -*-
import datetime
import logging

import numpy as np
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.utils.typingx import NDArrayLikeDict
from repro.generic_test_component import TestComponent
from utils.initialize_fields import initialize_field
from utils.allocate_state import allocate_state

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
    
    np.random.seed(23)

    return {
        **{
            key: np.array(
                np.random.rand(*component.array_shape),
                dtype=float,
                order="F",
            )
            for key in component.fields_in.keys()
        },
        **{
            key: np.array(
                np.random.rand(*component.array_shape),
                float,
                order="F",
            )
            for key in component.fields_inout.keys()
        },
        **{
            key: np.array(
                np.random.rand(*component.array_shape),
                float,
                order="F",
            )
            for key in component.fields_out.keys()
        },
    }


def compare_output(component, fortran_fields: dict, gt4py_state: dict):
    """Compare fortran and gt4py field mean on inout and out fields for a TestComponent

    Args:
        fortran_fields (dict): output fields from fortran
        gt4py_state (dict): output fields from gt4py
    """
    absolute_differences = dict()
    fields_to_compare = {**component.fields_inout, **component.fields_out}
    for field_name, field_attributes in fields_to_compare.items():
        logging.info(f"{field_name}")
        fortran_name = field_attributes["fortran_name"]
        fortran_field = fortran_fields[fortran_name][:, np.newaxis,:]
        logging.info(f"fortran field shape {fortran_field.shape}")
        logging.info(f"fortran field mean : {fortran_field.mean()}")
        
        # 2D fields + removing shadow level
        gt4py_field = gt4py_state[field_name]
        logging.info(f"gt4py field shape {gt4py_field.shape}")
        logging.info(f"gt4py field mean : {gt4py_field.values.mean()}")
        absolute_diff = abs(gt4py_field - fortran_field).values.mean()
        logging.info(f"{field_name}, absolute mean difference {absolute_diff}\n")
        absolute_differences.update({
            field_name: absolute_diff
        })
        
    return absolute_differences


def compare_input(component, fortran_fields: dict, gt4py_state: dict):
    """Compare fortran and gt4py field mean on inout and out fields for a TestComponent

    Args:
        fortran_fields (dict): output fields from fortran
        gt4py_state (dict): output fields from gt4py
    """
    absolute_differences = dict()
    fields_to_compare = {**component.fields_in, **component.fields_inout}
    logging.info(f"Input fields to compare {fields_to_compare.keys()}")
    logging.info(f"Fortran field keys {fortran_fields.keys()}")
    for field_name, field_attributes in fields_to_compare.items():
        logging.info(f"Input field name : {field_name}")
        fortran_name = field_attributes["fortran_name"]
        logging.info(f"(Input) fortran name {fortran_name}")
        fortran_field = fortran_fields[field_name][:, np.newaxis,:]
        logging.info(f"(Input) fortran field shape {fortran_field.shape}")
        logging.info(f"(Input) fortran field mean : {fortran_field.mean()}")
        
        # 2D fields + removing shadow level
        gt4py_field = gt4py_state[field_name]
        logging.info(f"gt4py field shape {gt4py_field.shape}")
        logging.info(f"gt4py field mean : {gt4py_field.values.mean()}")
        absolute_diff = abs(gt4py_field - fortran_field).values.mean()
        logging.info(f"{field_name}, absolute mean difference {absolute_diff}\n")
        absolute_differences.update({
            field_name: absolute_diff
        })
    
    for field, diff in absolute_differences.items():
            logging.info(f"Field name : {field}, error on input field : {diff}")
    

def run_test(component: ComputationalGridComponent):
    """Draw random arrays and call gt4py and fortran stencils side-by-side

    Args:
        component (ComputationalGridComponent): component to test
    """

    logging.info(f"\n Start test {component.__class__.__name__}")
    fields = draw_fields(component)
    state_gt4py = allocate_gt4py_fields(component, fields)
    
    logging.info(f"Compare input  fields")
    compare_input(component, fields, state_gt4py)
    
    logging.info("Calling fortran field")
    fortran_output_fields = component.call_fortran_stencil(fields)

    logging.info("Calling gt4py field")
    gt4py_output_fields = component.call_gt4py_stencil(state_gt4py)
    
    # TODO: remove shadow first level

    logging.info("Compare output fields")
    absolute_differences = compare_output(component=component, fortran_fields=fortran_output_fields, gt4py_state=gt4py_output_fields)
    logging.info(f"End test {component.__class__.__name__}\n")
    
    return absolute_differences
    
    