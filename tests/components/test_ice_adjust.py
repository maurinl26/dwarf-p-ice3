from gt4py.storage import from_array, zeros
from gt4py.cartesian.gtscript import stencil
import numpy as np
import xarray as xr
from numpy.testing import assert_allclose

from ice3.phyex_common.phyex import Phyex

def test_ice_adjust(benchmark):

    from ice3.stencils.ice_adjust import ice_adjust

    externals = Phyex("AROME").to_externals()
    backend = "gt:cpu_ifirst"

    ice_adjust_stencil = stencil(
        backend,
        definition=ice_adjust, 
        name="ice_adjust",
        externals=externals,
        dtypes={
            "float": np.float32,
            "int": np.int32,
            "bool": np.bool_
        }
        )
    
    ds = xr.open_dataset("./data/ice_adjust.nc", engine="netcdf4")

    shape = (
        ds.sizes["ngpblks"],
        ds.sizes["nproma"],
        ds.sizes["nflevg"]
    )   
    origin = (0, 0, 0)

    print("Reshaping inputs")
    sigqsat = ds["ZSIGQSAT"].data[:,:,np.newaxis]
    sigqsat = np.broadcast_to(sigqsat, shape)

    zrs = ds["ZRS"].data
    prs = ds["PRS"].data

    prs = np.swapaxes(prs, axis1=1, axis2=2)
    prs = np.swapaxes(prs, axis1=2, axis2=3)
    prs = np.swapaxes(prs, axis1=1, axis2=2)
    print(f"PRS shape {prs.shape}")

    zrs = np.swapaxes(zrs, axis1=1, axis2=2)
    zrs = np.swapaxes(zrs, axis1=2, axis2=3)
    zrs = np.swapaxes(zrs, axis1=1, axis2=2)
    print(f"ZRS shape {zrs.shape}")

    ppabsm = np.swapaxes(ds["PPABSM"].data, axis1=1, axis2=2)
    psigs = np.swapaxes(ds["PSIGS"].data, axis1=1, axis2=2)
    pexnref = np.swapaxes(ds["PEXNREF"].data, axis1=1, axis2=2)
    prhodref = np.swapaxes(ds["PRHODREF"].data, axis1=1, axis2=2)
    pcf_mf = np.swapaxes(ds["PCF_MF"].data, axis1=1, axis2=2)
    pri_mf = np.swapaxes(ds["PRI_MF"].data, axis1=1, axis2=2)
    prc_mf = np.swapaxes(ds["PRC_MF"].data, axis1=1, axis2=2)
    pths = np.swapaxes(ds["PTHS"].data, axis1=1, axis2=2)


    sigqsat = from_array(sigqsat, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    pabs = from_array(ppabsm, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    sigs = from_array(psigs, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    th = from_array(zrs[:,:,:,0], backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    exn_ref = from_array(pexnref, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    rhodref = from_array(prhodref, backend=backend, dtype=np.float32, aligned_index=(0,0,0))


    t = from_array(zrs[:,:,:,0], backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    rv = from_array(zrs[:,:,:,1], backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    ri = from_array(zrs[:,:,:,2], backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    rc = from_array(zrs[:,:,:,3], backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    rr = from_array(zrs[:,:,:,4], backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    rs = from_array(zrs[:,:,:,5], backend=backend, dtype=np.float32, aligned_index=(0,0,0)) 
    rg = from_array(zrs[:,:,:,6], backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    cf_mf = from_array(pcf_mf, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    rc_mf = from_array(prc_mf, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    ri_mf = from_array(pri_mf, backend=backend, dtype=np.float32, aligned_index=(0,0,0))

    ths = from_array(pths, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    rvs = from_array(prs[:,:,:,0], backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    rcs = from_array(prs[:,:,:,1], backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    ris = from_array(prs[:,:,:,3], backend=backend, dtype=np.float32, aligned_index=(0,0,0))

    hlc_hcf = zeros(shape, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    hlc_hrc = zeros(shape, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    hli_hcf = zeros(shape, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    hli_hri = zeros(shape, backend=backend, dtype=np.float32, aligned_index=(0,0,0))

    cldfr = zeros(shape, backend=backend, dtype=np.float32, aligned_index=(0,0,0))

    rv_out = zeros(shape, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    rc_out = zeros(shape, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    ri_out = zeros(shape, backend=backend, dtype=np.float32, aligned_index=(0,0,0))

    cph = zeros(shape, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    lv  = zeros(shape, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    ls  = zeros(shape, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    pv  = zeros(shape, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
    piv = zeros(shape, backend=backend, dtype=np.float32, aligned_index=(0,0,0))
 
    dt = np.float32(50.0)

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
            pv=pv,
            piv=piv,
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
    prs_out = ds["PRS_OUT"].data
    prs_out = np.swapaxes(prs_out, axis1=2, axis2=3)
    rvs_out = prs_out[:,0,:,:]
    rcs_out = prs_out[:,1,:,:]
    ris_out = prs_out[:,3,:,:]
    

    assert_allclose(rvs_out, rvs, atol=1e-4, rtol=1e-4)
    assert_allclose(rcs_out, rcs, atol=1e-4, rtol=1e-4)
    assert_allclose(ris_out, ris, atol=1e-4, rtol=1e-4)

