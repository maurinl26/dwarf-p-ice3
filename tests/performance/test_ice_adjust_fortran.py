# -*- coding: utf-8 -*-
"""Performance tests for Fortran implementation of ICE_ADJUST component."""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add build directory to path to find the compiled extension
build_dir = Path(__file__).parent.parent.parent / 'build'
if build_dir.exists():
    for sub in build_dir.iterdir():
        if sub.is_dir() and sub.name.startswith('cp'):
            sys.path.insert(0, str(sub))
            break

try:
    from ice3._phyex_wrapper import ice_adjust
    FORTRAN_AVAILABLE = True
except ImportError:
    # Fallback to local import
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "PHYEX-IAL_CY50T1/bridge"))
    try:
        from _phyex_wrapper import ice_adjust
        FORTRAN_AVAILABLE = True
    except ImportError:
        FORTRAN_AVAILABLE = False
        ice_adjust = None


def create_fortran_test_data(nijt=2500, nkt=60):
    """
    Create realistic atmospheric test data for Fortran ICE_ADJUST.

    Parameters
    ----------
    nijt : int
        Number of horizontal points (flattened)
    nkt : int
        Number of vertical levels

    Returns
    -------
    dict
        Dictionary with all required fields (Fortran-contiguous, float32)
    """
    # Use float32 for all calculations to match PHYEX expectations
    z = np.linspace(0, 10000, nkt, dtype=np.float32)

    # Standard atmosphere
    p0 = 101325.0
    T0 = 288.15
    gamma = 0.0065

    # Physical constants
    Rd = 287.0
    cp = 1004.0
    p00 = 100000.0

    data = {}

    # Pressure profile
    pressure = p0 * (1 - gamma * z / T0) ** 5.26
    data['ppabst'] = np.tile(pressure, (nijt, 1)).T.astype(np.float32).copy(order='F')

    # Temperature profile
    temperature = T0 - gamma * z
    data['temperature'] = np.tile(temperature, (nijt, 1)).T.astype(np.float32).copy(order='F')

    # Add variability
    np.random.seed(42)
    data['temperature'] += (np.random.randn(nkt, nijt) * 0.5).astype(np.float32)
    data['ppabst'] += (np.random.randn(nkt, nijt) * 100).astype(np.float32)

    # Exner function
    data['pexn'] = np.asfortranarray((data['ppabst'] / p00) ** (Rd / cp), dtype=np.float32)
    data['pth'] = np.asfortranarray(data['temperature'] / data['pexn'], dtype=np.float32)

    # Reference values
    data['pexnref'] = np.asfortranarray(data['pexn'].copy(), dtype=np.float32)
    data['prhodref'] = np.asfortranarray(data['ppabst'] / (Rd * data['temperature']), dtype=np.float32)

    # Height
    data['pzz'] = np.tile(z, (nijt, 1)).T.astype(np.float32).copy(order='F')

    # Water vapor (decreasing with height)
    rv_surf = 0.015
    data['prv'] = (rv_surf * np.exp(-z / 2000)).astype(np.float32)
    data['prv'] = np.tile(data['prv'], (nijt, 1)).T.astype(np.float32).copy(order='F')
    data['prv'] += (np.abs(np.random.randn(nkt, nijt)) * 0.001).astype(np.float32)

    # Cloud fields
    data['prc'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    cloud_levels = (z > 2000) & (z < 6000)
    for i in range(nijt):
        data['prc'][cloud_levels, i] = np.abs(np.random.rand(cloud_levels.sum())).astype(np.float32) * 0.002

    data['pri'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    ice_levels = z > 5000
    for i in range(nijt):
        data['pri'][ice_levels, i] = np.abs(np.random.rand(ice_levels.sum())).astype(np.float32) * 0.001

    data['prr'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['prs'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['prg'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')

    # Tendencies
    data['prvs'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['prcs'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['pris'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['pths'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')

    # Mass flux
    data['pcf_mf'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['prc_mf'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')
    data['pri_mf'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')

    # 1D sigqsat
    data['sigqsat'] = np.ones(nijt, dtype=np.float32, order='F') * 0.01

    # Subgrid turbulence
    data['psigs'] = np.zeros((nkt, nijt), dtype=np.float32, order='F')

    return data


@pytest.fixture
def small_fortran_data():
    """Small domain (10x10x20 = 100 horizontal points)."""
    return create_fortran_test_data(nijt=100, nkt=20)


@pytest.fixture
def medium_fortran_data():
    """Medium domain (50x50x40 = 2500 horizontal points)."""
    return create_fortran_test_data(nijt=2500, nkt=40)


@pytest.fixture
def large_fortran_data():
    """Large domain (100x100x60 = 10000 horizontal points)."""
    return create_fortran_test_data(nijt=10000, nkt=60)


def swap_to_cython(arr, nkt, nijt):
    """
    Convert from (nkt, nijt) to (nijt, nkt) for Cython wrapper.

    Parameters
    ----------
    arr : np.ndarray
        Array in (nkt, nijt) format
    nkt : int
        Number of vertical levels
    nijt : int
        Number of horizontal points

    Returns
    -------
    np.ndarray
        Array in (nijt, nkt) Fortran order
    """
    if arr.ndim == 2 and arr.shape[0] == nkt:
        return np.asfortranarray(arr.T, dtype=np.float32)
    return np.asfortranarray(arr, dtype=np.float32)


@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrapper not available")
class TestIceAdjustFortranPerformanceSmall:
    """Performance tests on small domain (100 horizontal points, 20 levels)."""

    def test_small_domain(self, benchmark, small_fortran_data):
        """Benchmark small domain."""
        print("\n" + "="*75)
        print("Fortran ICE_ADJUST Performance: Small Domain (100x20)")
        print("="*75)

        nijt = 100
        nkt = 20
        total_points = nijt * nkt

        print(f"Horizontal points: {nijt:,}")
        print(f"Vertical levels: {nkt}")
        print(f"Total grid points: {total_points:,}")

        # Prepare output arrays
        cldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        icldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        wcldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')

        # Benchmark function
        def run_ice_adjust():
            ice_adjust(
                timestep=np.float32(50.0),
                krr=6,
                sigqsat=small_fortran_data['sigqsat'],
                pabs=swap_to_cython(small_fortran_data['ppabst'], nkt, nijt),
                sigs=swap_to_cython(small_fortran_data['psigs'], nkt, nijt),
                th=swap_to_cython(small_fortran_data['pth'], nkt, nijt),
                exn=swap_to_cython(small_fortran_data['pexn'], nkt, nijt),
                exn_ref=swap_to_cython(small_fortran_data['pexnref'], nkt, nijt),
                rho_dry_ref=swap_to_cython(small_fortran_data['prhodref'], nkt, nijt),
                rv=swap_to_cython(small_fortran_data['prv'], nkt, nijt),
                rc=swap_to_cython(small_fortran_data['prc'], nkt, nijt),
                ri=swap_to_cython(small_fortran_data['pri'], nkt, nijt),
                rr=swap_to_cython(small_fortran_data['prr'], nkt, nijt),
                rs=swap_to_cython(small_fortran_data['prs'], nkt, nijt),
                rg=swap_to_cython(small_fortran_data['prg'], nkt, nijt),
                cf_mf=swap_to_cython(small_fortran_data['pcf_mf'], nkt, nijt),
                rc_mf=swap_to_cython(small_fortran_data['prc_mf'], nkt, nijt),
                ri_mf=swap_to_cython(small_fortran_data['pri_mf'], nkt, nijt),
                rvs=swap_to_cython(small_fortran_data['prvs'], nkt, nijt),
                rcs=swap_to_cython(small_fortran_data['prcs'], nkt, nijt),
                ris=swap_to_cython(small_fortran_data['pris'], nkt, nijt),
                ths=swap_to_cython(small_fortran_data['pths'], nkt, nijt),
                cldfr=cldfr,
                icldfr=icldfr,
                wcldfr=wcldfr
            )

        # Run benchmark
        result = benchmark(run_ice_adjust)

        # Statistics
        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")
        print(f"  Performance: {throughput*50.0/1e6:.2f} M point-steps/s")


@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrapper not available")
class TestIceAdjustFortranPerformanceMedium:
    """Performance tests on medium domain (2500 horizontal points, 40 levels)."""

    def test_medium_domain(self, benchmark, medium_fortran_data):
        """Benchmark medium domain."""
        print("\n" + "="*75)
        print("Fortran ICE_ADJUST Performance: Medium Domain (2500x40)")
        print("="*75)

        nijt = 2500
        nkt = 40
        total_points = nijt * nkt

        print(f"Horizontal points: {nijt:,}")
        print(f"Vertical levels: {nkt}")
        print(f"Total grid points: {total_points:,}")

        # Prepare output arrays
        cldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        icldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        wcldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')

        def run_ice_adjust():
            ice_adjust(
                timestep=np.float32(50.0),
                krr=6,
                sigqsat=medium_fortran_data['sigqsat'],
                pabs=swap_to_cython(medium_fortran_data['ppabst'], nkt, nijt),
                sigs=swap_to_cython(medium_fortran_data['psigs'], nkt, nijt),
                th=swap_to_cython(medium_fortran_data['pth'], nkt, nijt),
                exn=swap_to_cython(medium_fortran_data['pexn'], nkt, nijt),
                exn_ref=swap_to_cython(medium_fortran_data['pexnref'], nkt, nijt),
                rho_dry_ref=swap_to_cython(medium_fortran_data['prhodref'], nkt, nijt),
                rv=swap_to_cython(medium_fortran_data['prv'], nkt, nijt),
                rc=swap_to_cython(medium_fortran_data['prc'], nkt, nijt),
                ri=swap_to_cython(medium_fortran_data['pri'], nkt, nijt),
                rr=swap_to_cython(medium_fortran_data['prr'], nkt, nijt),
                rs=swap_to_cython(medium_fortran_data['prs'], nkt, nijt),
                rg=swap_to_cython(medium_fortran_data['prg'], nkt, nijt),
                cf_mf=swap_to_cython(medium_fortran_data['pcf_mf'], nkt, nijt),
                rc_mf=swap_to_cython(medium_fortran_data['prc_mf'], nkt, nijt),
                ri_mf=swap_to_cython(medium_fortran_data['pri_mf'], nkt, nijt),
                rvs=swap_to_cython(medium_fortran_data['prvs'], nkt, nijt),
                rcs=swap_to_cython(medium_fortran_data['prcs'], nkt, nijt),
                ris=swap_to_cython(medium_fortran_data['pris'], nkt, nijt),
                ths=swap_to_cython(medium_fortran_data['pths'], nkt, nijt),
                cldfr=cldfr,
                icldfr=icldfr,
                wcldfr=wcldfr
            )

        result = benchmark(run_ice_adjust)

        # Statistics
        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")
        print(f"  Performance: {throughput*50.0/1e6:.2f} M point-steps/s")


@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrapper not available")
class TestIceAdjustFortranPerformanceLarge:
    """Performance tests on large domain (10000 horizontal points, 60 levels)."""

    def test_large_domain(self, benchmark, large_fortran_data):
        """Benchmark large domain."""
        print("\n" + "="*75)
        print("Fortran ICE_ADJUST Performance: Large Domain (10000x60)")
        print("="*75)

        nijt = 10000
        nkt = 60
        total_points = nijt * nkt

        print(f"Horizontal points: {nijt:,}")
        print(f"Vertical levels: {nkt}")
        print(f"Total grid points: {total_points:,}")

        # Prepare output arrays
        cldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        icldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
        wcldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')

        def run_ice_adjust():
            ice_adjust(
                timestep=np.float32(50.0),
                krr=6,
                sigqsat=large_fortran_data['sigqsat'],
                pabs=swap_to_cython(large_fortran_data['ppabst'], nkt, nijt),
                sigs=swap_to_cython(large_fortran_data['psigs'], nkt, nijt),
                th=swap_to_cython(large_fortran_data['pth'], nkt, nijt),
                exn=swap_to_cython(large_fortran_data['pexn'], nkt, nijt),
                exn_ref=swap_to_cython(large_fortran_data['pexnref'], nkt, nijt),
                rho_dry_ref=swap_to_cython(large_fortran_data['prhodref'], nkt, nijt),
                rv=swap_to_cython(large_fortran_data['prv'], nkt, nijt),
                rc=swap_to_cython(large_fortran_data['prc'], nkt, nijt),
                ri=swap_to_cython(large_fortran_data['pri'], nkt, nijt),
                rr=swap_to_cython(large_fortran_data['prr'], nkt, nijt),
                rs=swap_to_cython(large_fortran_data['prs'], nkt, nijt),
                rg=swap_to_cython(large_fortran_data['prg'], nkt, nijt),
                cf_mf=swap_to_cython(large_fortran_data['pcf_mf'], nkt, nijt),
                rc_mf=swap_to_cython(large_fortran_data['prc_mf'], nkt, nijt),
                ri_mf=swap_to_cython(large_fortran_data['pri_mf'], nkt, nijt),
                rvs=swap_to_cython(large_fortran_data['prvs'], nkt, nijt),
                rcs=swap_to_cython(large_fortran_data['prcs'], nkt, nijt),
                ris=swap_to_cython(large_fortran_data['pris'], nkt, nijt),
                ths=swap_to_cython(large_fortran_data['pths'], nkt, nijt),
                cldfr=cldfr,
                icldfr=icldfr,
                wcldfr=wcldfr
            )

        result = benchmark(run_ice_adjust)

        # Statistics
        mean_time = result.stats['mean']
        throughput = total_points / mean_time

        print(f"\nResults:")
        print(f"  Mean time: {mean_time*1000:.3f} ms")
        print(f"  Throughput: {throughput/1e6:.2f} M points/s")
        print(f"  Performance: {throughput*50.0/1e6:.2f} M point-steps/s")

        if hasattr(result.stats, 'stddev'):
            print(f"  Std dev: {result.stats['stddev']*1000:.3f} ms")
        if hasattr(result.stats, 'min'):
            print(f"  Min time: {result.stats['min']*1000:.3f} ms")
        if hasattr(result.stats, 'max'):
            print(f"  Max time: {result.stats['max']*1000:.3f} ms")


@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrapper not available")
class TestIceAdjustFortranScaling:
    """Test scaling characteristics."""

    def test_scaling_with_domain_size(self):
        """Test how performance scales with domain size."""
        print("\n" + "="*75)
        print("Fortran ICE_ADJUST: Scaling Analysis")
        print("="*75)

        sizes = [
            (100, 10),    # Small
            (400, 20),    # Medium-small
            (1000, 30),   # Medium
        ]

        results = []
        for nijt, nkt in sizes:
            data = create_fortran_test_data(nijt=nijt, nkt=nkt)
            total_points = nijt * nkt

            # Prepare output arrays
            cldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
            icldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
            wcldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')

            # Time single execution
            import time
            start = time.time()

            ice_adjust(
                timestep=np.float32(50.0),
                krr=6,
                sigqsat=data['sigqsat'],
                pabs=swap_to_cython(data['ppabst'], nkt, nijt),
                sigs=swap_to_cython(data['psigs'], nkt, nijt),
                th=swap_to_cython(data['pth'], nkt, nijt),
                exn=swap_to_cython(data['pexn'], nkt, nijt),
                exn_ref=swap_to_cython(data['pexnref'], nkt, nijt),
                rho_dry_ref=swap_to_cython(data['prhodref'], nkt, nijt),
                rv=swap_to_cython(data['prv'], nkt, nijt),
                rc=swap_to_cython(data['prc'], nkt, nijt),
                ri=swap_to_cython(data['pri'], nkt, nijt),
                rr=swap_to_cython(data['prr'], nkt, nijt),
                rs=swap_to_cython(data['prs'], nkt, nijt),
                rg=swap_to_cython(data['prg'], nkt, nijt),
                cf_mf=swap_to_cython(data['pcf_mf'], nkt, nijt),
                rc_mf=swap_to_cython(data['prc_mf'], nkt, nijt),
                ri_mf=swap_to_cython(data['pri_mf'], nkt, nijt),
                rvs=swap_to_cython(data['prvs'], nkt, nijt),
                rcs=swap_to_cython(data['prcs'], nkt, nijt),
                ris=swap_to_cython(data['pris'], nkt, nijt),
                ths=swap_to_cython(data['pths'], nkt, nijt),
                cldfr=cldfr,
                icldfr=icldfr,
                wcldfr=wcldfr
            )

            elapsed = time.time() - start
            throughput = total_points / elapsed
            results.append((total_points, elapsed, throughput))

            print(f"\nDomain {nijt}x{nkt}: {total_points:,} points")
            print(f"  Time: {elapsed*1000:.3f} ms")
            print(f"  Throughput: {throughput/1e6:.2f} M points/s")

        # Check scaling
        print("\n" + "-"*75)
        print("Scaling Summary:")
        print("-"*75)
        for i, (points, time, tput) in enumerate(results):
            if i > 0:
                size_ratio = points / results[0][0]
                time_ratio = time / results[0][1]
                print(f"  {points:>10,} points: {time_ratio:.2f}x time for {size_ratio:.2f}x points")


@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrapper not available")
def test_ice_adjust_fortran_with_repro_data(benchmark, ice_adjust_repro_ds):
    """
    Performance test with reproduction dataset from ice_adjust.nc.

    Parameters
    ----------
    benchmark : pytest_benchmark fixture
        Benchmark fixture for timing
    ice_adjust_repro_ds : xr.Dataset
        Reference dataset from ice_adjust.nc fixture
    """
    print("\n" + "="*75)
    print("Fortran ICE_ADJUST Performance: Reproduction Dataset")
    print("="*75)

    # Get dataset dimensions
    shape = (
        ice_adjust_repro_ds.sizes["ngpblks"],
        ice_adjust_repro_ds.sizes["nproma"],
        ice_adjust_repro_ds.sizes["nflevg"]
    )
    nijt = shape[0] * shape[1]
    nkt = shape[2]

    print(f"Dataset shape: {shape}")
    print(f"Effective domain: nijt={nijt}, nkt={nkt}")
    print(f"Total grid points: {nijt * nkt:,}")

    # Load and reshape data
    def reshape_input(var):
        """Reshape from (ngpblks, nflevg, nproma) to (nijt, nkt) Fortran order."""
        v = np.swapaxes(var, 1, 2)
        v = v.reshape(nijt, nkt)
        return np.asfortranarray(v, dtype=np.float32)

    pabs = reshape_input(ice_adjust_repro_ds["PPABSM"].values)
    exn = reshape_input(ice_adjust_repro_ds["PEXNREF"].values)
    exnref = reshape_input(ice_adjust_repro_ds["PEXNREF"].values)
    rhodref = reshape_input(ice_adjust_repro_ds["PRHODREF"].values)
    sigs = reshape_input(ice_adjust_repro_ds["PSIGS"].values)

    # ZRS: (ngpblks, krr, nflevg, nproma)
    zrs = ice_adjust_repro_ds["ZRS"].values
    zrs = np.swapaxes(zrs, 2, 3)

    def extract_zrs(idx):
        return zrs[:, idx, :, :].reshape(nijt, nkt).copy(order='F').astype(np.float32)

    th = extract_zrs(0)
    rv = extract_zrs(1)
    rc = extract_zrs(2)
    rr = extract_zrs(3)
    ri = extract_zrs(4)
    rs = extract_zrs(5)
    rg = extract_zrs(6)

    cf_mf = reshape_input(ice_adjust_repro_ds["PCF_MF"].values)
    rc_mf = reshape_input(ice_adjust_repro_ds["PRC_MF"].values)
    ri_mf = reshape_input(ice_adjust_repro_ds["PRI_MF"].values)

    sigqsat = reshape_input(ice_adjust_repro_ds["ZSIGQSAT"].values)[:, 0].copy(order='F')

    # Prepare tendencies
    rvs = np.zeros((nijt, nkt), dtype=np.float32, order='F')
    rcs = np.zeros((nijt, nkt), dtype=np.float32, order='F')
    ris = np.zeros((nijt, nkt), dtype=np.float32, order='F')
    ths = np.zeros((nijt, nkt), dtype=np.float32, order='F')

    # Output arrays
    cldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
    icldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')
    wcldfr = np.zeros((nijt, nkt), dtype=np.float32, order='F')

    # Benchmark function
    def run_ice_adjust():
        ice_adjust(
            timestep=np.float32(50.0),
            krr=6,
            sigqsat=sigqsat,
            pabs=pabs,
            sigs=sigs,
            th=th,
            exn=exn,
            exn_ref=exnref,
            rho_dry_ref=rhodref,
            rv=rv, rc=rc, ri=ri, rr=rr, rs=rs, rg=rg,
            cf_mf=cf_mf, rc_mf=rc_mf, ri_mf=ri_mf,
            rvs=rvs, rcs=rcs, ris=ris, ths=ths,
            cldfr=cldfr, icldfr=icldfr, wcldfr=wcldfr
        )

    # Run benchmark
    result = benchmark(run_ice_adjust)

    # Statistics
    print("\n" + "-"*75)
    print("PERFORMANCE STATISTICS")
    print("-"*75)

    total_points = nijt * nkt
    mean_time = result.stats['mean']
    throughput = total_points / mean_time

    print(f"Mean time: {mean_time*1000:.3f} ms")
    print(f"Throughput: {throughput/1e6:.2f} M points/s")
    print(f"Performance: {throughput*50.0/1e6:.2f} M point-steps/s")

    if hasattr(result.stats, 'stddev'):
        print(f"Std dev: {result.stats['stddev']*1000:.3f} ms")
    if hasattr(result.stats, 'min'):
        print(f"Min time: {result.stats['min']*1000:.3f} ms")
    if hasattr(result.stats, 'max'):
        print(f"Max time: {result.stats['max']*1000:.3f} ms")

    print("-"*75)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
