# -*- coding: utf-8 -*-
from functools import cached_property, partial
import fmodpy
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.framework.grid import ComputationalGrid
import numpy as np

from ice3_gt4py.initialisation.utils import initialize_field
from ice3_gt4py.phyex_common.phyex import Phyex

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.utils.typingx import (
    DataArray,
)

from ice3_gt4py.utils.allocate import allocate
from ice3_gt4py.utils.doctor_norm import field_doctor_norm


class TestComponent(ComputationalGridComponent):
    def __init__(
        self,
        computational_grid: ComputationalGrid,
        gt4py_config: GT4PyConfig,
        phyex: Phyex,
        fortran_subroutine: str,
        gt4py_stencil: str,
    ) -> None:
        super().__init__(
            computational_grid=computational_grid, gt4py_config=gt4py_config
        )
        externals = phyex.to_externals()

        self.compile_fortran_stencil(fortran_subroutine)
        self.compile_gt4py_stencil(gt4py_stencil)

    def compile_gt4py_stencil(self, gt4py_stencil, externals):
        self.gt4py_stencil = self.compile_stencil(gt4py_stencil, externals)

    def compile_fortran_stencil(self, fortran_subroutine):
        stencils_directory = "./src/ice3_gt4py/stencils_fortran/"
        fortran_script = fmodpy.fimport(stencils_directory + "cloud_fraction.F90")
        fortran_module = getattr(self.fortran_script, fortran_module)
        self.fortran_stencil = getattr(fortran_module, fortran_subroutine)

    # Attributes
    @cached_property
    def externals(self, phyex_externals: dict):
        """Filter phyex externals"""
        return {
            "lsubg_cond": phyex_externals["SUBG_COND"],
            "xcriautc": phyex_externals["CRIAUTC"],
            "csubg_mf_pdf": phyex_externals["SUBG_MF_PDF"],
            "xcriauti": phyex_externals["CRIAUTI"],
            "xacriauti": phyex_externals["ACRIAUTI"],
            "xbcriauti": phyex_externals["BCRIAUTI"],
            "xtt": phyex_externals["TT"],
        }

    @cached_property
    def dims(self):
        nit, njt, nkt = self.computational_grid.grids[(I, J, K)].shape
        return {"nijt": nit * njt, "nkt": nkt}

    @cached_property
    def fields_in(self):
        return {
            "lv": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "ls": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "t": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "cph": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "rhodref": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "exnref": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "rc": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "ri": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "rc_mf": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "ri_mf": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "cf_mf": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "rc_tmp": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "ri_tmp": {"grid": (I, J, K), "dtype": "float", "units": ""},
        }

    @cached_property
    def fields_out(self):
        return {
            "hlc_hrc": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "hlc_hcf": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "hli_hri": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "hli_hcf": {"grid": (I, J, K), "dtype": "float", "units": ""},
        }

    @cached_property
    def fields_inout(self):
        return {
            "ths": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "rvs": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "rcs": {"grid": (I, J, K), "dtype": "float", "units": ""},
            "ris": {"grid": (I, J, K), "dtype": "float", "units": ""},
        }

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
        """Allocate GT4Py state (zero arrays)"""
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
        state = self.allocate_state(self.computational_grid, self.gt4py_config)
        fields = {**self.fields_in, **self.fields_inout, **self.fields_out}
        for field_key, field_attributes in fields.items():
            key_fortran = field_doctor_norm(field_key, field_attributes.dtype)
            initialize_field(state[field_key], fields[key_fortran][:, np.newaxis])

        return state

    ##### Calls #####
    def call_fortran_stencil(self, fields):
        self.fortran_stencil(**fields, **self.dims, **self.externals)

    def call_gt4py_stencil(self, fields):
        state = self.allocate(fields)
        self.gt4py_stencil(**state)
