#!/usr/bin/env python3
"""
Test script for the ICE_ADJUST Python wrapper.

This script demonstrates how to use the Python wrapper for the ICE_ADJUST
Fortran subroutine and validates the results.
"""

import numpy as np
from pyphyex.ice_adjust_wrapper import IceAdjustWrapper, create_test_data

def test_basic_functionality():
    """Test basic functionality of the ice_adjust wrapper."""
    print("Testing basic functionality...")
    
    # Create wrapper
    try:
        wrapper = IceAdjustWrapper()
        print(f"✓ Loaded library from: {wrapper.library_path}")
    except FileNotFoundError as e:
        print(f"⚠ Warning: {e}")
        print("  Using placeholder implementation for testing...")
        wrapper = IceAdjustWrapper.__new__(IceAdjustWrapper)
        from phyex.pyphyex.ice_adjust_wrapper import PhysicalConstants, IceParameters, NebulosityParameters
        wrapper.constants = PhysicalConstants()
        wrapper.ice_params = IceParameters()
        wrapper.neb_params = NebulosityParameters()
        
    # Create small test case
    nijt, nkt = 10, 5
    test_data = create_test_data(nijt=nijt, nkt=nkt)
    
    # Run ice adjustment
    results = wrapper.ice_adjust(**test_data)
    
    # Basic validation
    assert 'cldfr' in results, "Missing cloud fraction output"
    assert 'icldfr' in results, "Missing ice cloud fraction output"  
    assert 'wcldfr' in results, "Missing water cloud fraction output"
    
    # Check output shapes
    expected_shape = (nijt, nkt)
    for key, arr in results.items():
        if arr is not None:
            assert arr.shape == expected_shape, f"{key} has wrong shape: {arr.shape} vs {expected_shape}"
    
    # Check value ranges
    for key in ['cldfr', 'icldfr', 'wcldfr']:
        arr = results[key]
        assert np.all(arr >= 0), f"{key} contains negative values"
        assert np.all(arr <= 1), f"{key} contains values > 1"
    
    print("✓ Basic functionality test passed")
    return results

def test_input_validation():
    """Test input validation."""
    print("Testing input validation...")
    
    wrapper = IceAdjustWrapper.__new__(IceAdjustWrapper)
    
    # Test dimension validation
    try:
        wrapper._validate_inputs(-1, 10, 6, 300.0, None, None, None, None, None, None, None, None, None)
        assert False, "Should have raised ValueError for negative dimensions"
    except ValueError:
        print("✓ Negative dimension validation passed")
    
    # Test KRR validation  
    try:
        wrapper._validate_inputs(10, 10, 1, 300.0, None, None, None, None, None, None, None, None, None)
        assert False, "Should have raised ValueError for KRR < 2"
    except ValueError:
        print("✓ KRR validation passed")
        
    # Test timestep validation
    try:
        wrapper._validate_inputs(10, 10, 6, -300.0, None, None, None, None, None, None, None, None, None)
        assert False, "Should have raised ValueError for negative timestep"
    except ValueError:
        print("✓ Timestep validation passed")
    
    print("✓ Input validation tests passed")

def test_realistic_case():
    """Test with realistic atmospheric data."""
    print("Testing with realistic atmospheric case...")
    
    # Create wrapper
    wrapper = IceAdjustWrapper.__new__(IceAdjustWrapper)
    
    # Create realistic test case
    nijt, nkt = 100, 50
    test_data = create_test_data(nijt=nijt, nkt=nkt)
    
    print(f"Grid size: {nijt} x {nkt}")
    print(f"Temperature range: {(test_data['pth'] * test_data['pexn']).min():.1f} - {(test_data['pth'] * test_data['pexn']).max():.1f} K")
    print(f"Pressure range: {test_data['ppabst'].min():.0f} - {test_data['ppabst'].max():.0f} Pa")
    print(f"Water vapor mixing ratio range: {test_data['prv'].min():.6f} - {test_data['prv'].max():.6f} kg/kg")
    
    # Run ice adjustment
    results = wrapper.ice_adjust(**test_data)
    
    # Analyze results
    total_points = nijt * nkt
    cloudy_points = np.sum(results['cldfr'] > 0)
    ice_cloudy_points = np.sum(results['icldfr'] > 0)  
    water_cloudy_points = np.sum(results['wcldfr'] > 0)
    
    print(f"Results:")
    print(f"  Total grid points: {total_points}")
    print(f"  Cloudy points: {cloudy_points} ({100*cloudy_points/total_points:.1f}%)")
    print(f"  Ice cloudy points: {ice_cloudy_points} ({100*ice_cloudy_points/total_points:.1f}%)")
    print(f"  Water cloudy points: {water_cloudy_points} ({100*water_cloudy_points/total_points:.1f}%)")
    print(f"  Cloud fraction range: {results['cldfr'].min():.3f} - {results['cldfr'].max():.3f}")
    
    print("✓ Realistic case test passed")
    return test_data, results

def visualize_results(test_data, results, save_plots=True):
    """Visualize the results."""
    print("Creating visualization...")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ Matplotlib not available, skipping visualization")
        return
    
    # Extract data
    nijt, nkt = test_data['nijt'], test_data['nkt']
    temp = test_data['pth'] * test_data['pexn']
    height = test_data['pzz'] / 1000  # Convert to km
    prv = test_data['prv'] * 1000  # Convert to g/kg
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('ICE_ADJUST Results', fontsize=16)
    
    # Temperature profile
    ax = axes[0, 0]
    im = ax.contourf(np.arange(nijt), height[0, :], temp.T, levels=20, cmap='RdYlBu_r')
    ax.set_xlabel('Horizontal index')
    ax.set_ylabel('Height [km]')
    ax.set_title('Temperature [K]')
    plt.colorbar(im, ax=ax)
    
    # Water vapor mixing ratio
    ax = axes[0, 1] 
    im = ax.contourf(np.arange(nijt), height[0, :], prv.T, levels=20, cmap='Blues')
    ax.set_xlabel('Horizontal index')
    ax.set_ylabel('Height [km]')
    ax.set_title('Water Vapor [g/kg]')
    plt.colorbar(im, ax=ax)
    
    # Total cloud fraction
    ax = axes[0, 2]
    im = ax.contourf(np.arange(nijt), height[0, :], results['cldfr'].T, levels=np.linspace(0, 1, 11), cmap='gray_r')
    ax.set_xlabel('Horizontal index')
    ax.set_ylabel('Height [km]')
    ax.set_title('Cloud Fraction')
    plt.colorbar(im, ax=ax)
    
    # Ice cloud fraction
    ax = axes[1, 0]
    im = ax.contourf(np.arange(nijt), height[0, :], results['icldfr'].T, levels=np.linspace(0, 1, 11), cmap='Blues')
    ax.set_xlabel('Horizontal index') 
    ax.set_ylabel('Height [km]')
    ax.set_title('Ice Cloud Fraction')
    plt.colorbar(im, ax=ax)
    
    # Water cloud fraction
    ax = axes[1, 1]
    im = ax.contourf(np.arange(nijt), height[0, :], results['wcldfr'].T, levels=np.linspace(0, 1, 11), cmap='Reds')
    ax.set_xlabel('Horizontal index')
    ax.set_ylabel('Height [km]')
    ax.set_title('Water Cloud Fraction')
    plt.colorbar(im, ax=ax)
    
    # Vertical profiles (column averages)
    ax = axes[1, 2]
    ax.plot(np.mean(temp, axis=0), height[0, :], 'r-', label='Temperature [K]', linewidth=2)
    ax.plot(np.mean(prv, axis=0) * 50, height[0, :], 'b-', label='Water Vapor [g/kg] × 50', linewidth=2)
    ax.plot(np.mean(results['cldfr'], axis=0) * 300, height[0, :], 'k-', label='Cloud Fraction × 300', linewidth=2)
    ax.plot(np.mean(results['icldfr'], axis=0) * 300, height[0, :], 'c--', label='Ice Cloud Frac. × 300', linewidth=2)
    ax.set_xlabel('Value')
    ax.set_ylabel('Height [km]')
    ax.set_title('Vertical Profiles (Mean)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('ice_adjust_results.png', dpi=150, bbox_inches='tight')
        print("✓ Plot saved as ice_adjust_results.png")
    
    plt.show()

def benchmark_performance():
    """Benchmark the performance of the wrapper."""
    print("Benchmarking performance...")
    
    import time
    
    wrapper = IceAdjustWrapper.__new__(IceAdjustWrapper)
    
    # Test different grid sizes
    sizes = [(50, 30), (100, 50), (200, 100)]
    
    print(f"{'Grid Size':<10} {'Points':<8} {'Time (s)':<10} {'Points/s':<12}")
    print("-" * 45)
    
    for nijt, nkt in sizes:
        test_data = create_test_data(nijt=nijt, nkt=nkt)
        
        start_time = time.time()
        results = wrapper.ice_adjust(**test_data)
        end_time = time.time()
        
        elapsed = end_time - start_time
        points = nijt * nkt
        rate = points / elapsed
        
        print(f"{nijt}x{nkt:<6} {points:<8} {elapsed:<10.4f} {rate:<12.0f}")
    
    print("✓ Performance benchmark completed")

def main():
    """Main test function."""
    print("ICE_ADJUST Python Wrapper Test Suite")
    print("=" * 40)
    
    try:
        # Run tests
        test_input_validation()
        print()
        
        test_basic_functionality()  
        print()
        
        test_data, results = test_realistic_case()
        print()
        
        benchmark_performance()
        print()
        
        # Create visualization
        visualize_results(test_data, results)
        
        print("=" * 40)
        print("✓ All tests passed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
