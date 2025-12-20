import numpy as np
import pytest
import sys
import os
from pathlib import Path

# Add build directory to path to find the compiled extension
build_dir = Path(__file__).parent.parent.parent / 'build'
if build_dir.exists():
    for sub in build_dir.iterdir():
        if sub.is_dir() and sub.name.startswith('cp'):
            sys.path.insert(0, str(sub))
            break

try:
    from ice3._phyex_wrapper import rain_ice
except ImportError:
    # Fallback to local import if building in-place
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "PHYEX-IAL_CY50T1/bridge"))
    try:
        from _phyex_wrapper import rain_ice
    except ImportError:
        rain_ice = None


def test_rain_ice_cython_with_repro_data(rain_ice_repro_ds):
    """
    Test Cython wrapper with reproduction dataset from rain_ice.nc.
    """
    print("\n" + "="*70)
    print("TEST: Cython RAIN_ICE with Reproduction Data")
    print("="*70)
    
    if rain_ice is None:
        pytest.skip("Cython rain_ice wrapper not available")

    try:
        from numpy.testing import assert_allclose
        
        # Get dataset dimensions (ngpblks, nflevg, nproma)
        shape = (
            rain_ice_repro_ds.sizes["ngpblks"],
            rain_ice_repro_ds.sizes["nproma"],
            rain_ice_repro_ds.sizes["nflevg"]
        )
        nijt = shape[0] * shape[1]
        nkt = shape[2]
        
        print(f"\nDataset shape: {shape}")
        print(f"Effective domain: nijt={nijt}, nkt={nkt}")
        
        # Load input data from dataset
        print("\n1. Loading input data...")
        
        def reshape_input(var, ndim=2):
            """Reshape from (ngpblks, nflevg, nproma) to (nijt, nkt) Fortran order."""
            if ndim == 2:
                v = np.swapaxes(var, 1, 2)  # (ngpblks, nproma, nflevg)
                v = v.reshape(nijt, nkt)  # (nijt, nkt)
                return np.asfortranarray(v, dtype=np.float32)
            elif ndim == 1:
                return np.asfortranarray(var.reshape(nijt), dtype=np.float32)
            return var
        
        # Load all required fields for RAIN_ICE
        exn = reshape_input(rain_ice_repro_ds["PEXNREF"].values)
        dzz = reshape_input(rain_ice_repro_ds["PDZZ"].values)
        rhodj = reshape_input(rain_ice_repro_ds["PRHODJ"].values)
        rhodref = reshape_input(rain_ice_repro_ds["PRHODREF"].values)
        exnref = reshape_input(rain_ice_repro_ds["PEXNREF"].values)
        pabs = reshape_input(rain_ice_repro_ds["PPABSM"].values)
        cldfr = reshape_input(rain_ice_repro_ds["PCLDFR"].values)
        
        # Atmospheric state from PRT (ngpblks, krr, nflevg, nproma)
        prt = rain_ice_repro_ds["PRT"].values
        prt = np.swapaxes(prt, 2, 3)  # → (ngpblks, krr, nproma, nflevg)
        
        def extract_prt(idx):
             return prt[:, idx, :, :].reshape(nijt, nkt).copy(order='F').astype(np.float32)

        tht = reshape_input(rain_ice_repro_ds["PTHT"].values)
        rvt = extract_prt(0)
        rct = extract_prt(1)
        rrt = extract_prt(2)
        rit = extract_prt(3)
        rst = extract_prt(4)
        rgt = extract_prt(5)
        
        # HLCLOUDS arrays
        hlc_hrc = reshape_input(rain_ice_repro_ds["PHLC_HRC"].values)
        hlc_hcf = reshape_input(rain_ice_repro_ds["PHLC_HCF"].values)
        hli_hri = reshape_input(rain_ice_repro_ds["PHLI_HRI"].values)
        hli_hcf = reshape_input(rain_ice_repro_ds["PHLI_HCF"].values)
        
        # Cloud ice number concentration
        cit = reshape_input(rain_ice_repro_ds["PCIT"].values)
        
        # Sigma_s
        sigs = reshape_input(rain_ice_repro_ds["PSIGS"].values)
        
        # OCND2 arrays
        icldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        ssio = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        ssiu = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        ifr = np.zeros((nijt, nkt), dtype=np.float32, order='F')

        # Initialize tendencies from PRS (ngpblks, krr, nflevg, nproma)
        prs = rain_ice_repro_ds["PRS"].values
        prs = np.swapaxes(prs, 2, 3)  # → (ngpblks, krr, nproma, nflevg)

        def extract_prs(idx):
             return prs[:, idx, :, :].reshape(nijt, nkt).copy(order='F').astype(np.float32)

        ths = reshape_input(rain_ice_repro_ds["PTHS"].values)
        rvs = extract_prs(0)
        rcs = extract_prs(1)
        rrs = extract_prs(2)
        ris = extract_prs(3)
        rss = extract_prs(4)
        rgs = extract_prs(5)
        
        # Output arrays
        evap3d = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        rainfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        inprc = np.zeros(nijt, dtype=np.float32, order='F')
        inprr = np.zeros(nijt, dtype=np.float32, order='F')
        inprs = np.zeros(nijt, dtype=np.float32, order='F')
        inprg = np.zeros(nijt, dtype=np.float32, order='F')
        indep = np.zeros(nijt, dtype=np.float32, order='F')
        
        print("✓ Input data loaded")
        
        # Call RAIN_ICE via Cython
        print("\n2. Calling RAIN_ICE via Cython...")
        timestep = float(rain_ice_repro_ds.attrs.get("PTSTEP", 50.0))
        print(f"   Timestep: {timestep} s")
        
        rain_ice(
            timestep=np.float32(timestep),
            krr=6,
            exn=exn,
            dzz=dzz,
            rhodj=rhodj,
            rhodref=rhodref,
            exnref=exnref,
            pabs=pabs,
            cldfr=cldfr,
            icldfr=icldfr,
            ssio=ssio,
            ssiu=ssiu,
            ifr=ifr,
            tht=tht,
            rvt=rvt,
            rct=rct,
            rrt=rrt,
            rit=rit,
            rst=rst,
            rgt=rgt,
            sigs=sigs,
            # Input/output arrays
            cit=cit,
            hlc_hrc=hlc_hrc,
            hlc_hcf=hlc_hcf,
            hli_hri=hli_hri,
            hli_hcf=hli_hcf,
            ths=ths,
            rvs=rvs,
            rcs=rcs,
            rrs=rrs,
            ris=ris,
            rss=rss,
            rgs=rgs,
            # Output arrays
            evap3d=evap3d,
            rainfr=rainfr,
            inprc=inprc,
            inprr=inprr,
            inprs=inprs,
            inprg=inprg,
            indep=indep
        )
        
        print("✓ RAIN_ICE completed")
        
        # Compare with reference data
        print("\n3. Comparing with reference data...")
        
        # PRS_OUT: (ngpblks, krr+1, nflevg, nproma)
        # 0=ths, 1=rvs, 2=rcs, 3=rrs, 4=ris, 5=rss, 6=rgs
        prs_out = rain_ice_repro_ds["PRS_OUT"].values
        prs_out = np.swapaxes(prs_out, 2, 3)  # (ngpblks, krr+1, nproma, nflevg)
        
        def extract_ref(idx):
             return prs_out[:, idx, :, :].reshape(nijt, nkt).copy()

        # Tendencies
        ths_ref = extract_ref(0)
        rvs_ref = extract_ref(1)
        rcs_ref = extract_ref(2)
        rrs_ref = extract_ref(3)
        ris_ref = extract_ref(4)
        rss_ref = extract_ref(5)
        rgs_ref = extract_ref(6)
        
        # Compare tendencies with appropriate tolerances
        tolerances = {
            'ths': (1e-5, 1e-4),
            'rvs': (1e-5, 1e-4),
            'rcs': (1e-5, 1e-4),
            'rrs': (1e-5, 1e-4),
            'ris': (1e-5, 1e-4),
            'rss': (1e-5, 1e-4),
            'rgs': (1e-5, 1e-4),
        }
        
        results_map = {
            'ths': ths,
            'rvs': rvs,
            'rcs': rcs,
            'rrs': rrs,
            'ris': ris,
            'rss': rss,
            'rgs': rgs,
        }
        
        references = {
            'ths': ths_ref,
            'rvs': rvs_ref,
            'rcs': rcs_ref,
            'rrs': rrs_ref,
            'ris': ris_ref,
            'rss': rss_ref,
            'rgs': rgs_ref,
        }
        
        comparison_results = {}
        
        for var_name, ref_data in references.items():
            atol, rtol = tolerances[var_name]
            try:
                assert_allclose(results_map[var_name], ref_data, atol=atol, rtol=rtol)
                print(f"✓ {var_name} (tolerance: atol={atol}, rtol={rtol})")
                comparison_results[var_name] = True
            except AssertionError as e:
                max_diff = np.max(np.abs(results_map[var_name] - ref_data))
                print(f"✗ {var_name} mismatch: max_diff={max_diff:.6e}")
                comparison_results[var_name] = False
        
        # Precipitation outputs
        if "PINPRC_OUT" in rain_ice_repro_ds:
            pinprc_ref = reshape_input(rain_ice_repro_ds["PINPRC_OUT"].values, ndim=1)
            assert_allclose(inprc, pinprc_ref, atol=1e-5, rtol=1e-4)
            print("✓ inprc (cloud precipitation)")
        
        if "PINPRR_OUT" in rain_ice_repro_ds:
            pinprr_ref = reshape_input(rain_ice_repro_ds["PINPRR_OUT"].values, ndim=1)
            assert_allclose(inprr, pinprr_ref, atol=1e-5, rtol=1e-4)
            print("✓ inprr (rain precipitation)")
        
        if "PINPRS_OUT" in rain_ice_repro_ds:
            pinprs_ref = reshape_input(rain_ice_repro_ds["PINPRS_OUT"].values, ndim=1)
            assert_allclose(inprs, pinprs_ref, atol=1e-5, rtol=1e-4)
            print("✓ inprs (snow precipitation)")
        
        if "PINPRG_OUT" in rain_ice_repro_ds:
            pinprg_ref = reshape_input(rain_ice_repro_ds["PINPRG_OUT"].values, ndim=1)
            assert_allclose(inprg, pinprg_ref, atol=1e-5, rtol=1e-4)
            print("✓ inprg (graupel precipitation)")
        
        # Precipitation fraction
        if "PRAINFR_OUT" in rain_ice_repro_ds:
            prainfr_ref = reshape_input(rain_ice_repro_ds["PRAINFR_OUT"].values)
            assert_allclose(rainfr, prainfr_ref, atol=1e-4, rtol=1e-4)
            print("✓ rainfr (precipitation fraction)")
        
        print("\n" + "="*70)
        print("COMPARISON COMPLETE")
        print("="*70)
        
        # Summary
        passed = sum(comparison_results.values())
        total = len(comparison_results)
        print(f"\nTendency comparisons: {passed}/{total} passed")
        
        if passed != total:
             pytest.fail(f"{total-passed} tendency mismatches found")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Test failed: {e}")

