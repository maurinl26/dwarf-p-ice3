# -*- coding: utf-8 -*-
import fmodpy
import numpy as np
import logging
from functools import cached_property
from typing import TYPE_CHECKING
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array
from ice3_gt4py.initialisation.utils import initialize_field
from ice3_gt4py.utils.allocate import allocate
from pathlib import Path

import datetime
from functools import partial
from ice3_gt4py.phyex_common.phyex import Phyex
import ice3_gt4py.stencils

from typing import Literal, Tuple
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
from ifs_physics_common.utils.typingx import (
    DataArray,
    DataArrayDict,
    NDArrayLikeDict,
)


##### For tests ####
def allocate_state_ice4_rrhong(
    computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig
) -> NDArrayLikeDict:
    """Allocate field to state keys following type (float, int, bool) and dimensions (2D, 3D).

    Args:
        computational_grid (ComputationalGrid): grid indexes
        gt4py_config (GT4PyConfig): gt4py configuration

    Returns:
        NDArrayLikeDict: dictionnary of field with associated keys for field name
    """

    def _allocate(
        grid_id: Tuple[DimSymbol, ...],
        units: str,
        dtype: Literal["bool", "float", "int"],
    ) -> DataArray:
        return allocate_data_array(
            computational_grid, grid_id, units, gt4py_config=gt4py_config, dtype=dtype
        )

    allocate_b_ij = partial(_allocate, grid_id=(I, J), units="", dtype="bool")
    allocate_b = partial(_allocate, grid_id=(I, J, K), units="", dtype="bool")
    allocate_f = partial(_allocate, grid_id=(I, J, K), units="", dtype="float")
    allocate_h = partial(_allocate, grid_id=(I, J, K - 1 / 2), units="", dtype="float")
    allocate_ij = partial(_allocate, grid_id=(I, J), units="", dtype="float")
    allocate_i_ij = partial(_allocate, grid_id=(I, J), units="", dtype="int")

    return {
        "ldcompute": allocate_b(),
        "exn": allocate_f(),
        "ls_fact": allocate_f(),
        "lv_fact": allocate_f(),
        "t": allocate_f(),
        "tht": allocate_f(),
        "rr_t": allocate_f(),
        "rrhong_mr": allocate_f(),
    }


class TestIce4RRHONG(ComputationalGridComponent):
    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
        phyex: Phyex,
        fortran_module: str,
        fortran_subroutine: str,
        fortran_script: str,
        gt4py_stencil: str
    ) -> None:
        super().__init__(
            computational_grid=computational_grid, gt4py_config=gt4py_config
        )

        externals = phyex.to_externals()
        self.ice4_rrhong_gt4py = self.compile_stencil(gt4py_stencil, externals)

        self.externals = {
            "xtt": externals["TT"],
            "r_rtmin": externals["R_RTMIN"],
            "lfeedbackt": externals["LFEEDBACKT"],
        }

        self.generate_gt4py_state()

        # aro_filter stands for the parts before 'call ice_adjust' in aro_adjust.f90
        #stencils_fortran_directory = "./src/ice3_gt4py/stencils_fortran"
        #fortran_script = "mode_ice4_rrhong.F90"
    
        project_dir = Path.cwd()
        stencils_fortran_dir = Path(project_dir, "src", "ice3_gt4py", "stencils_fortran")
        
        fortran_script_path = Path(stencils_fortran_dir, fortran_script)
        self.fortran_script = fmodpy.fimport(
            fortran_script_path
        )
        self.fortran_module = fortran_module
        self.fortran_subroutine = fortran_subroutine
        fortran_module = getattr(self.fortran_script, self.fortran_module)
        self.fortran_stencil = getattr(fortran_module, self.fortran_subroutine)
        
    @cached_property
    def dims(self):
        return {"kproma": KPROMA, "ksize": KSIZE}

    @cached_property
    def fields_mapping(self):
        return {
            "ldcompute": "ldcompute",
            "pexn": "exn",
            "plsfact": "ls_fact",
            "plvfact": "lv_fact",
            "pt": "t",
            "ptht": "tht",
            "prrt": "rr_t",
            "prrhong_mr": "rrhong_mr",
        }

    @cached_property
    def fields_in(self):
        return {
            "ldcompute": np.ones((self.dims["kproma"]), dtype=np.int32),
            "pexn": np.array(np.random.rand(self.dims["kproma"]), "f", order="F"),
            "plvfact": np.array(np.random.rand(self.dims["kproma"]), "f", order="F"),
            "plsfact": np.array(np.random.rand(self.dims["kproma"]), "f", order="F"),
            "pt": np.array(np.random.rand(self.dims["kproma"]), "f", order="F"),
            "prrt": np.array(np.random.rand(self.dims["kproma"]), "f", order="F"),
            "ptht": np.array(np.random.rand(self.dims["kproma"]), "f", order="F"),
        }

    @cached_property
    def fields_out(self):
        return {"prrhong_mr": np.zeros((self.dims["kproma"]), "f", order="F")}

    def generate_gt4py_state(self):

        self.state_gt4py = allocate_state_ice4_rrhong(
            self.computational_grid, self.gt4py_config
        )
        fields = {**self.fields_in, **self.fields_out}
        for key_fortran, key_gt4py in self.fields_mapping.items():
            initialize_field(
                self.state_gt4py[key_gt4py], fields[key_fortran][:, np.newaxis]
            )

    def test(self):
        """Call fortran stencil"""

        logging.info(
            f"Input field, rrhong_mr (fortran) : {self.fields_out['prrhong_mr'].mean()}"
        )
        logging.info(
            f"Input field, rrhong_mr (gt4py) : {self.state_gt4py['rrhong_mr'][...].mean()}"
        )

        logging.info(
            f"Input field, ls_fact (fortran) : {self.fields_in['plsfact'].mean()}"
        )
        logging.info(
            f"Input field, ls_fact (gt4py) : {self.state_gt4py['ls_fact'][...].mean()}"
        )
        
        
        self.fortran_stencil(**self.dims, **self.externals, **self.fields_in, **self.fields_out)

        self.ice4_rrhong_gt4py(
            **self.state_gt4py,
        )

        # logging.info(f"Mean Fortran : {prrhong_mr.mean()}")

        field_gt4py = self.state_gt4py["rrhong_mr"][...]
        logging.info(f"Mean GT4Py {field_gt4py.mean()}")
        
class Ice4Rrhong(ComputationalGridComponent):
    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
        phyex: Phyex,
        fortran_subroutine: str,
        fortran_script: str,
        fortran_module: str,
        gt4py_stencil_name: str,
    ) -> None:
        super().__init__(
            computational_grid=computational_grid, gt4py_config=gt4py_config
        )
        
        self.phyex_externals = phyex.to_externals()
        self.compile_fortran_stencil(
            fortran_module=fortran_module,
            fortran_script=fortran_script,
            fortran_subroutine=fortran_subroutine
        )
        self.compile_gt4py_stencil(gt4py_stencil_name, self.phyex_externals)
    
    @cached_property
    def externals(self):
        """Filter phyex externals"""
        return {
            "xtt": self.phyex_externals["TT"],
            "r_rtmin": self.phyex_externals["R_RTMIN"],
            "lfeedbackt": self.phyex_externals["LFEEDBACKT"],
        }

    @cached_property
    def dims(self) -> dict:
        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
        return {"kproma": nit * njt, "ksize": nit * njt}

    @cached_property
    def array_shape(self) -> dict:
        return self.dims["kproma"]

    @cached_property
    def fields_in(self):
        return {
            "ldcompute": {"grid": (I, J, K), "dtype": "int", "fortran_name": "ldcompute"},
            "exn": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pexn"},
            "lvfact": {"grid": (I, J, K), "dtype": "float", "fortran_name": "plvfact"},
            "lsfact": {"grid": (I, J, K), "dtype": "float", "fortran_name": "plsfact"},
            "t": {"grid": (I, J, K), "dtype": "float", "fortran_name": "pt"},
            "rrt":{"grid": (I, J, K), "dtype": "float", "fortran_name": "prrt"},
            "tht": {"grid": (I, J, K), "dtype": "float", "fortran_name": "ptht"},
        }

    @cached_property
    def fields_out(self):
        return {
            "rrhong_mr": {"grid": (I, J, K), "dtype": "int", "fortran_name": "prrhong_mr"}
        }

    @cached_property
    def fields_inout(self):
        return {}

    #### Compilations ####
    def compile_gt4py_stencil(self, gt4py_stencil_name: str, externals: dict):
        """Compile GT4Py script given

        Args:
            gt4py_stencil (str): _description_
            externals (dict): _description_
        """
        self.gt4py_stencil = self.compile_stencil(gt4py_stencil_name, externals)

    def compile_fortran_stencil(self, fortran_script, fortran_module, fortran_subroutine):
        current_directory = Path.cwd()
        logging.info(f"Root directory {current_directory}")
        root_directory = current_directory
        
        stencils_directory = Path(root_directory, "src", "ice3_gt4py", "stencils_fortran")
        script_path = Path(stencils_directory, fortran_script)
        
        logging.info(f"Fortran script path {script_path}")
        self.fortran_script = fmodpy.fimport(script_path)
        fortran_module = getattr(self.fortran_script, fortran_module)
        self.fortran_stencil = getattr(fortran_module, fortran_subroutine)

    ###### Allocations ######
    def set_allocators(self):
        """Allocate zero array in storage given grid and type characteristics"""
        _allocate_on_computational_grid = partial(
            allocate,
            computational_grid=self.computational_grid,
            gt4py_config=self.gt4py_config,
        )
        self.allocate_b_ij = partial(
            _allocate_on_computational_grid, grid_id=(I, J), units="", dtype="bool"
        )
        self.allocate_b_ijk = partial(
            _allocate_on_computational_grid, grid_id=(I, J, K), units="", dtype="bool"
        )

        self.allocate_f_ij = partial(
            _allocate_on_computational_grid, grid_id=(I, J), units="", dtype="float"
        )
        self.allocate_f_ijk = partial(
            _allocate_on_computational_grid, grid_id=(I, J, K), units="", dtype="float"
        )
        self.allocate_f_ijk_h = partial(
            _allocate_on_computational_grid,
            grid_id=(I, J, K - 1 / 2),
            units="",
            dtype="float",
        )

        self.allocate_i_ij = partial(
            _allocate_on_computational_grid, grid_id=(I, J), units="", dtype="int"
        )
        self.allocate_i_ijk = partial(
            _allocate_on_computational_grid, grid_id=(I, J, K), units="", dtype="int"
        )

    def allocate_state(self):
        """Allocate GT4Py state"""
        self.set_allocators()
        
        state = dict()
        fields = {**self.fields_in, **self.fields_inout, **self.fields_out}
        for field_key, field_attributes in fields.items():

            # 3D fields
            if field_attributes["dtype"] == "float" and field_attributes["grid"] == (
                I,
                J,
                K,
            ):
                state.update({field_key: self.allocate_f_ijk()})
            elif field_attributes["dtype"] == "bool" and field_attributes["grid"] == (
                I,
                J,
                K,
            ):
                state.update({field_key: self.allocate_b_ijk()})
            elif field_attributes["dtype"] == "int" and field_attributes["grid"] == (
                I,
                J,
                K,
            ):
                state.update({field_key: self.allocate_i_ijk()})

            # 2D fields
            if field_attributes["dtype"] == "float" and field_attributes["grid"] == (
                I,
                J,
            ):
                state.update({field_key: self.allocate_f_ij()})
            elif field_attributes["dtype"] == "bool" and field_attributes["grid"] == (
                I,
                J,
            ):
                state.update({field_key: self.allocate_b_ij()})
            elif field_attributes["dtype"] == "int" and field_attributes["grid"] == (
                I,
                J,
            ):
                state.update({field_key: self.allocate_i_ij()})

    def allocate_fields(self, fields: dict):
        """Allocate fields (as gt4py storage)

        Interface with fortran like fields
        """
        state = self.allocate_state()
        fields_properties = {**self.fields_in, **self.fields_inout, **self.fields_out}
        for field_key, field_attributes in fields.items():
            initialize_field(state[field_key], fields[field_key][:, np.newaxis])
        return state

    ##### Calls #####
    def call_fortran_stencil(self, fields: dict):
        """Call fortran stencil on a given field dict

        externals and dims are handled by component attributes itself

        Args:
            fields (dict): dictionnary of numpy arrays
        """
        
        fortran_state = {}
        component_fields = {**self.fields_in, **self.fields_out, **self.fields_inout}
        for field_name, field_array in fields.items():
            fortran_name = component_fields[field_name]["fortran_name"] if field_name in list(component_fields.keys()) else None
            fortran_state.update({fortran_name: field_array})
        
        output_fields = self.fortran_stencil(**fortran_state, **self.dims, **self.externals)
        return output_fields

    def call_gt4py_stencil(self, fields):
        """Call gt4py stencil on a given field dict

        Args:
            fields (dict): dictionnary of numpy arrays
        """
        state = self.allocate_fields(fields)
        self.gt4py_stencil(**state)
        return state


if __name__ == "__main__":

    KPROMA, KSIZE = 50, 50

    # TODO : set in env values
    backend = "gt:cpu_ifirst"
    rebuild = True
    validate_args = True

    phyex = Phyex(program="AROME")

    logging.info("Initializing grid ...")

    # Grid has only 1 dimension since fields are packed in fortran version
    grid = ComputationalGrid(50, 1, 1)
    dt = datetime.timedelta(seconds=1)

    ######## Backend and gt4py config #######
    logging.info(f"With backend {backend}")
    gt4py_config = GT4PyConfig(
        backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=False
    )

    logging.info("Calling ice4_rrhong with dicts")

    test_ice4_rrhong = TestIce4RRHONG(
        computational_grid=grid, gt4py_config=gt4py_config, phyex=phyex, 
        fortran_module="mode_ice4_rrhong", fortran_subroutine="ice4_rrhong",
        fortran_script="mode_ice4_rrhong.F90",
        gt4py_stencil="ice4_rrhong"
    ).test()
    
    # New component
    logging.info("Ice4Rrhong component")
    component = Ice4Rrhong(
        computational_grid=grid, 
        gt4py_config=gt4py_config, 
        phyex=phyex, 
        fortran_script="mode_ice4_rrhong.F90",
        fortran_module="mode_ice4_rrhong", 
        fortran_subroutine="ice4_rrhong",
        gt4py_stencil_name="ice4_rrhong"
    )
    
    fields = {
        **{
            key: np.array(np.random.rand(component.array_shape), "f", order="F")
            for key in component.fields_in.keys()
        },
        **{
            key: np.array(np.random.rand(component.array_shape), "f", order="F")
            for key in component.fields_inout.keys()
        },
        **{
            key: np.zeros((component.array_shape), "f", order="F")
            for key in component.fields_out.keys()
        },
    }
    
    
    output = component.call_fortran_stencil(fields)
    logging.info(f"{output}")
    component.call_gt4py_stencil(fields)
    
