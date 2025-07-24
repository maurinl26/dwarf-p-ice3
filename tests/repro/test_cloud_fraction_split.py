from ifs_physics_common.framework.stencil import compile_stencil
from gt4py.storage import from_array
import numpy as np
from numpy.testing import assert_allclose
import pytest
from ctypes import c_float, c_double

from tests.conftest import compile_fortran_stencil, get_backends


def allocate_random_fields(names, gt4py_config, grid, dtype=None):
    dtype = dtype or (c_float if gt4py_config.dtypes.float == np.float32 else c_double)
    fields = {name: np.array(np.random.rand(*grid.shape), dtype=dtype, order="F") for name in names}
    gt4py_buffers = {name: from_array(fields[name], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend) for name in names}
    return fields, gt4py_buffers


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_thermo(gt4py_config, externals, fortran_dims, precision, backend, grid, origin):
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)
    dtype = c_float if gt4py_config.dtypes.float == np.float32 else c_double
    f2py_mapping = {"prv": "rv", "prc": "rc", "pri": "ri", "prr": "rr", "prs": "rs", "prg": "rg", "pth": "th", "pexn": "exn", "zt": "t", "zls": "ls", "zlv": "lv", "zcph": "cph"}
    py2f_mapping = dict(map(reversed, f2py_mapping.items()))
    externals_mapping = {"xlvtt": "LVTT", "xlstt": "LSTT", "xcpv": "CPV", "xci": "CI", "xcl": "CL", "xtt": "TT", "xcpd": "CPD"}
    fortran_externals = {fname: externals[pyname] for fname, pyname in externals_mapping.items()}
    fortran_stencil = compile_fortran_stencil("mode_thermo.F90", "mode_thermo", "latent_heat")
    thermo_fields = compile_stencil("thermodynamic_fields", gt4py_config, externals)
    field_names = ["th", "exn", "rv", "rc", "rr", "ri", "rs", "rg", "lv", "ls", "cph", "t"]
    fields, gt4py_buffers = allocate_random_fields(field_names, gt4py_config, grid, dtype)
    fortran_fields = {py2f_mapping[name]: field.reshape(grid.shape[0] * grid.shape[1], grid.shape[2]) for name, field in fields.items() if name in py2f_mapping}
    thermo_fields(**gt4py_buffers, domain=grid.shape, origin=origin)
    result = fortran_stencil(krr=6, **fortran_fields, **fortran_externals, **fortran_dims)
    output_names = ['zt', 'zlv', 'zls', 'zcph']
    fields_out = {name: result[i] for i, name in enumerate(output_names)}
    result_mapping = {'zt': 't', 'zlv': 'lv', 'zls': 'ls', 'zcph': 'cph'}
    for fname, pyname in result_mapping.items():
        assert_allclose(fields_out[fname], gt4py_buffers[pyname].reshape(grid.shape[0] * grid.shape[1], grid.shape[2]), rtol=1e-6)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_cloud_fraction_1(gt4py_config, externals, fortran_dims, precision, backend, grid, origin):
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)
    dtype = c_float if gt4py_config.dtypes.float == np.float32 else c_double
    externals["LSUBG_COND"] = True
    cloud_fraction_1 = compile_stencil("cloud_fraction_1", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil("mode_cloud_fraction_split.F90", "mode_cloud_fraction_split", "cloud_fraction_1")
    dt = gt4py_config.dtypes.float(50.0)
    f2py_mapping = {"zrc": "rc_tmp", "zri": "ri_tmp", "pexnref": "exnref", "zcph": "cph", "zlv": "lv", "zls": "ls", "prc": "rc", "pri": "ri", "prvs": "rvs", "prcs": "rcs", "pths": "ths", "pris": "ris"}
    py2f_mapping = dict(map(reversed, f2py_mapping.items()))
    field_names = ["lv", "ls", "cph", "exnref", "rc", "ri", "ths", "rvs", "rcs", "ris", "rc_tmp", "ri_tmp"]
    fields, gt4py_buffers = allocate_random_fields(field_names, gt4py_config, grid, dtype)
    fortran_fields = {py2f_mapping[name]: field.reshape(grid.shape[0] * grid.shape[1], grid.shape[2]) for name, field in fields.items() if name in py2f_mapping}
    cloud_fraction_1(**gt4py_buffers, dt=dt, domain=grid.shape, origin=origin)
    result = fortran_stencil(ptstep=dt, **fortran_fields, **fortran_dims)
    output_names = ["pths", "prvs", "prcs", "pris"]
    fields_out = {name: result[i] for i, name in enumerate(output_names)}
    result_mapping = {"pths": "ths", "prvs": "rvs", "prcs": "rcs", "pris": "ris"}
    for fname, pyname in result_mapping.items():
        assert_allclose(fields_out[fname], gt4py_buffers[pyname].reshape(grid.shape[0] * grid.shape[1], grid.shape[2]), rtol=1e-6)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_cloud_fraction_2(gt4py_config, externals, fortran_dims, precision, backend, grid, origin):
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)
    dtype = c_float if gt4py_config.dtypes.float == np.float32 else c_double
    externals["LSUBG_COND"] = True
    externals.update({"SUBG_MF_PDF": 0})
    cloud_fraction_2 = compile_stencil("cloud_fraction_2", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil("mode_cloud_fraction_split.F90", "mode_cloud_fraction_split", "cloud_fraction_2")
    dt = gt4py_config.dtypes.float(50.0)
    f2py_mapping = {"pexnref": "exnref", "prhodref": "rhodref", "zcph": "cph", "zlv": "lv", "zls": "ls", "zt": "t", "pcf_mf": "cf_mf", "prc_mf": "rc_mf", "pri_mf": "ri_mf", "pths": "ths", "prvs": "rvs", "prcs": "rcs", "pris": "ris", "pcldfr": "cldfr", "phlc_hrc": "hlc_hrc", "phlc_hcf": "hlc_hcf", "phli_hri": "hli_hri", "phli_hcf": "hli_hcf"}
    py2f_mapping = dict(map(reversed, f2py_mapping.items()))
    field_names = ["rhodref", "exnref", "t", "cph", "lv", "ls", "ths", "rvs", "rcs", "ris", "rc_mf", "ri_mf", "cf_mf", "cldfr", "hlc_hrc", "hlc_hcf", "hli_hri", "hli_hcf"]
    fields, gt4py_buffers = allocate_random_fields(field_names, gt4py_config, grid, dtype)
    fortran_fields = {py2f_mapping[name]: field.reshape(grid.shape[0] * grid.shape[1], grid.shape[2]) for name, field in fields.items() if name in py2f_mapping}
    cloud_fraction_2(**gt4py_buffers, dt=dt, domain=grid.shape, origin=origin)
    result = fortran_stencil(ptstep=dt, **fortran_fields, **fortran_dims)
    output_fields = ["pths", "prvs", "prcs", "pris", "pcldfr", "phlc_hrc", "phlc_hcf", "phli_hri", "phli_hcf"]
    results = {name: result[i] for i, name in enumerate(output_fields)}
    result_mapping = {"pcldfr": "cldfr", "phlc_hcf": "hlc_hcf", "phlc_hrc": "hlc_hrc", "phli_hri": "hli_hri", "phli_hcf": "hli_hcf"}
    for fname, pyname in result_mapping.items():
        assert_allclose(results[fname], gt4py_buffers[pyname].reshape(grid.shape[0] * grid.shape[1], grid.shape[2]), rtol=1e-6)
