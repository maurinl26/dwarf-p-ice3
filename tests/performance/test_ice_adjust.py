import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array, zeros
from numpy.testing import assert_allclose

from ice3.utils.env import sp_dtypes, dp_dtypes


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("debug", marks=pytest.mark.debug),
        pytest.param("numpy", marks=pytest.mark.numpy),
        pytest.param("gt:cpu_ifirst", marks=pytest.mark.cpu),
        pytest.param("gt:gpu", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
def test_ice_adjust(benchmark, backend, externals, dtypes, ice_adjust_repro_ds):

    from ice3.stencils.ice_adjust import ice_adjust

    externals.update(
        {"OCND2": False}
    )

    ice_adjust_stencil = stencil(
        backend,
        definition=ice_adjust, 
        name="ice_adjust",
        externals=externals,
        dtypes=dtypes,
        validate_args=False
        )

    shape = (
        ice_adjust_repro_ds.sizes["ngpblks"],
        ice_adjust_repro_ds.sizes["nproma"],
        ice_adjust_repro_ds.sizes["nflevg"]
    )

    print("Reshaping inputs")
    sigqsat = ice_adjust_repro_ds["ZSIGQSAT"].data[:,:,np.newaxis]
    sigqsat = np.broadcast_to(sigqsat, shape)

    zrs = ice_adjust_repro_ds["ZRS"].data
    prs = ice_adjust_repro_ds["PRS"].data

    prs = np.swapaxes(prs, axis1=1, axis2=2)
    prs = np.swapaxes(prs, axis1=2, axis2=3)
    prs = np.swapaxes(prs, axis1=1, axis2=2)
    print(f"PRS shape {prs.shape}")

    zrs = np.swapaxes(zrs, axis1=1, axis2=2)
    zrs = np.swapaxes(zrs, axis1=2, axis2=3)
    zrs = np.swapaxes(zrs, axis1=1, axis2=2)
    print(f"ZRS shape {zrs.shape}")

    ppabsm = np.swapaxes(ice_adjust_repro_ds["PPABSM"].data, axis1=1, axis2=2)
    psigs = np.swapaxes(ice_adjust_repro_ds["PSIGS"].data, axis1=1, axis2=2)
    pexnref = np.swapaxes(ice_adjust_repro_ds["PEXNREF"].data, axis1=1, axis2=2)
    prhodref = np.swapaxes(ice_adjust_repro_ds["PRHODREF"].data, axis1=1, axis2=2)
    pcf_mf = np.swapaxes(ice_adjust_repro_ds["PCF_MF"].data, axis1=1, axis2=2)
    pri_mf = np.swapaxes(ice_adjust_repro_ds["PRI_MF"].data, axis1=1, axis2=2)
    prc_mf = np.swapaxes(ice_adjust_repro_ds["PRC_MF"].data, axis1=1, axis2=2)
    pths = np.swapaxes(ice_adjust_repro_ds["PTHS"].data, axis1=1, axis2=2)


    sigqsat = from_array(sigqsat, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    pabs = from_array(ppabsm, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    sigs = from_array(psigs, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    th = from_array(zrs[:,:,:,0], backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    exn_ref = from_array(pexnref, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    rhodref = from_array(prhodref, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))


    t = from_array(zrs[:,:,:,0], backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    rv = from_array(zrs[:,:,:,1], backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    ri = from_array(zrs[:,:,:,2], backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    rc = from_array(zrs[:,:,:,3], backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    rr = from_array(zrs[:,:,:,4], backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    rs = from_array(zrs[:,:,:,5], backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0)) 
    rg = from_array(zrs[:,:,:,6], backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    cf_mf = from_array(pcf_mf, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    rc_mf = from_array(prc_mf, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    ri_mf = from_array(pri_mf, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))

    ths = from_array(pths, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    rvs = from_array(prs[:,:,:,0], backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    rcs = from_array(prs[:,:,:,1], backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    ris = from_array(prs[:,:,:,3], backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))

    hlc_hcf = zeros(shape, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    hlc_hrc = zeros(shape, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    hli_hcf = zeros(shape, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    hli_hri = zeros(shape, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))

    cldfr = zeros(shape, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))

    rv_out = zeros(shape, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    rc_out = zeros(shape, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    ri_out = zeros(shape, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))

    cph = zeros(shape, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    lv  = zeros(shape, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    ls  = zeros(shape, backend=backend, dtype=dtypes["float"], aligned_index=(0,0,0))
    dt = dtypes["float"](50.0)

    def run_ice_adjust():
        ice_adjust_stencil(
            sigqsat=sigqsat,
            pabs=pabs,
            sigs=sigs,
            th=th,
            exn=exn_ref,
            exn_ref=exn_ref,
            rho_dry_ref=rhodref,
            t=t,
            rv=rv,
            ri=ri,
            rc=rc,
            rr=rr,
            rs=rs,
            rg=rg,
            cf_mf=cf_mf,
            rc_mf=rc_mf,
            ri_mf=ri_mf,
            rv_out=rv_out,
            rc_out=rc_out,
            ri_out=ri_out,
            hli_hri=hli_hri,
            hli_hcf=hli_hcf,
            hlc_hrc=hlc_hrc,
            hlc_hcf=hlc_hcf,
            ths=ths,
            rvs=rvs,
            rcs=rcs,
            ris=ris,
            cldfr=cldfr,
            cph=cph,
            lv=lv,
            ls=ls,
            dt=dt,
            origin=(0,0,0),
            domain=shape
        )

        return (
            ths,
            rvs,
            rcs,
            ris,
            hlc_hcf,
            hlc_hrc,
            hli_hcf,
            hli_hri
        )
    
    benchmark(run_ice_adjust)

    print("Reshaping output")
    prs_out = ice_adjust_repro_ds["PRS_OUT"].data
    prs_out = np.swapaxes(prs_out, axis1=2, axis2=3)
    rvs_out = prs_out[:,0,:,:]
    rcs_out = prs_out[:,1,:,:]
    ris_out = prs_out[:,3,:,:]

    phlc_hrc_out = ice_adjust_repro_ds["PHLC_HRC_OUT"].data
    phlc_hcf_out = ice_adjust_repro_ds["PHLC_HCF_OUT"].data
    phli_hri_out = ice_adjust_repro_ds["PHLI_HRI_OUT"].data
    phli_hcf_out = ice_adjust_repro_ds["PHLI_HCF_OUT"].data


