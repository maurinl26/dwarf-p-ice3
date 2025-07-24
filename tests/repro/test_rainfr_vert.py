import pytest
from tests.conftest import get_backends
from ctypes import  c_float

import numpy as np
import pytest
from gt4py.storage import from_array
from ifs_physics_common.framework.stencil import compile_stencil
from numpy.testing import assert_allclose


from tests.conftest import compile_fortran_stencil, get_backends

@pytest.mark.skip("Not Implemented")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("backend", get_backends())
def test_rainfr_vert(
    gt4py_config, externals, fortran_packed_dims, precision, backend, grid, origin
):
    ldsoft = False
    gt4py_config.backend = backend
    gt4py_config.dtypes = gt4py_config.dtypes.with_precision(precision)
    ice4_fast_rs = compile_stencil("ice4_rainfr_vert", gt4py_config, externals)
    fortran_stencil = compile_fortran_stencil("mode_ice4_rainfr_vert.F90", "mode_ice4_rainfr_vert", "ice4_rainfr_vert")
    field_names = [
        "prfr", "rr", "rs", "rg"
    ]
    fields, gt4py_buffers = allocate_random_fields(field_names, gt4py_config, grid, c_float)
    ldcompute = np.ones(grid.shape, dtype=bool, order="F")
    ldcompute_gt4py = from_array(ldcompute, dtype=bool, backend=gt4py_config.backend)
    ice4_fast_rs(ldsoft=ldsoft, ldcompute=ldcompute_gt4py, **gt4py_buffers, domain=grid.shape, origin=origin)
    # Fortran externals and mapping
    externals_mapping = {"ngaminc": "NGAMINC", "nacclbdas": "NACCLBDAS", "nacclbdar": "NACCLBDAR", "levlimit": "LEVLIMIT", "lpack_interp": "LPACK_INTERP", "csnowriming": "CSNOWRIMING", "xcrimss": "CRIMSS", "xexcrimss": "EXCRIMSS", "xcrimsg": "CRIMSG", "xexcrimsg": "EXCRIMSG", "xexsrimcg2": "EXSRIMCG2", "xfraccss": "FRACCSS", "s_rtmin": "S_RTMIN", "c_rtmin": "C_RTMIN", "r_rtmin": "R_RTMIN", "xepsilo": "EPSILO", "xalpi": "ALPI", "xbetai": "BETAI", "xgami": "GAMI", "xtt": "TT", "xlvtt": "LVTT", "xcpv": "CPV", "xci": "CI", "xcl": "CL", "xlmtt": "LMTT", "xestt": "ESTT", "xrv": "RV", "x0deps": "O0DEPS", "x1deps": "O1DEPS", "xex0deps": "EX0DEPS", "xex1deps": "EX1DEPS", "xlbraccs1": "LBRACCS1", "xlbraccs2": "LBRACCS2", "xlbraccs3": "LBRACCS3", "xcxs": "CXS", "xsrimcg2": "SRIMCG2", "xsrimcg3": "SRIMCG3", "xbs": "BS", "xlbsaccr1": "LBSACCR1", "xlbsaccr2": "LBSACCR2", "xlbsaccr3": "LBSACCR3", "xfsaccrg": "FSACCRG", "xsrimcg": "SRIMCG", "xexsrimcg": "EXSRIMCG", "xcexvt": "CVEXT", "xalpw": "ALPW", "xbetaw": "BETAW", "xgamw": "GAMW", "xfscvmg": "FSCVMG"}
    fortran_externals = {fkey: externals[pykey] for fkey, pykey in externals_mapping.items()}
    fortran_lookup_tables = {"xker_raccss": KER_RACCSS, "xker_raccs": KER_RACCS, "xker_saccrg": KER_SACCRG}
    f2py_mapping = {"prhodref": "rhodref", "ppres": "pres", "pdv": "dv", "pka": "ka", "pcj": "cj", "plbdar": "lbdar", "plbdas": "lbdas", "pt": "t", "prvt": "rvt", "prct": "rct", "prrt": "rrt", "prst": "rst", "priaggs": "riaggs", "prcrimss": "rcrimss", "prcrimsg": "rcrimsg", "prsrimcg": "rsrimcg", "prraccss": "rraccss", "prraccsg": "rraccsg", "prsaccrg": "rsaccrg", "prsmltg": "rs_mltg_tnd", "prcmltsr": "rc_mltsr_tnd", "rs_rcrims_tend": "rs_rcrims_tnd", "rs_rcrimss_tend": "rs_rcrimss_tnd", "rs_rsrimcg_tend": "rs_rsrimcg_tnd", "rs_rraccs_tend": "rs_rraccs_tnd", "rs_rraccss_tend": "rs_rraccss_tnd", "rs_rsaccrg_tend": "rs_rsaccrg_tnd", "rs_freez1_tend": "rs_freez1_tnd", "rs_freez2_tend": "rs_freez2_tnd"}
    fortran_FloatFieldsIJK = {fname: fields[pyname].ravel() for fname, pyname in f2py_mapping.items() if pyname in fields}
    result = fortran_stencil(ldsoft=ldsoft, ldcompute=ldcompute, **fortran_FloatFieldsIJK, **fortran_packed_dims, **fortran_externals, **fortran_lookup_tables)
    # Output assertions (example for a few fields)
    assert_allclose(result[0], gt4py_buffers["prfr"].ravel(), rtol=1e-6)
    