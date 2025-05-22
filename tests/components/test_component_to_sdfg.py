from ice3_gt4py.components.component_to_sdfg import TestComponent
from ifs_physics_common.framework.grid import I, J, K
import numpy as np
import logging

def test_orchestrated_call(computational_grid, gt4py_config, phyex):

    gt4py_config.backend = "dace:cpu"

    nx, ny, nz = computational_grid.grids[(I, J, K)].shape

    component = TestComponent(
        gt4py_config=gt4py_config,
        computational_grid=computational_grid,
        phyex=phyex
    )

    component.dace_setup()

    th = np.ones((nx,ny, nz), dtype=np.float32)
    exn = np.ones((nx, ny, nz), dtype=np.float32)
    rv = np.ones((nx, ny, nz), dtype=np.float32)
    rc = np.ones((nx, ny, nz), dtype=np.float32)
    rr = np.ones((nx, ny, nz), dtype=np.float32)
    ri = np.ones((nx, ny, nz), dtype=np.float32)
    rs = np.ones((nx, ny, nz), dtype=np.float32)
    rg = np.ones((nx, ny, nz), dtype=np.float32)

    lv = np.zeros((nx, ny, nz), dtype=np.float32)
    ls = np.zeros((nx, ny, nz), dtype=np.float32)
    t = np.zeros((nx, ny, nz), dtype=np.float32)
    cph = np.zeros((nx, ny, nz), dtype=np.float32)


    component.orchestrated_call(
        th=th,
        exn=exn,
        rv=rv,
        rc=rc,
        rr=rr,
        ri=ri,
        rs=rs,
        rg=rg,
        lv=lv,
        ls=ls,
        cph=cph,
        t=t,
        I=nx,
        J=ny,
        K=nz
    )

    logging.info(f"lv, mean {lv.mean()}")
    logging.info(f"ls, mean {ls.mean()}")
    logging.info(f"cph, mean {cph.mean()}")
    logging.info(f"t, mean {t.mean()}")


