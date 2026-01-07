# Ice Adjust Data Documentation

## Overview

The `ice_adjust.nc` file contains data for the ICE_ADJUST (ice adjustment) microphysics scheme from PHYEX. This NetCDF file is used for testing and validation of the ice adjustment parameterization.

**Source:** Based on PHYEX `getdata_ice_adjust_mod.F90`
**Reference:** https://github.com/UMR-CNRM/PHYEX/blob/master/src/offline/progs/getdata_ice_adjust_mod.F90

## File Structure

### Dimensions

- **nproma** (32): Number of points in a block (horizontal blocking dimension)
- **nflevg** (15): Number of full vertical levels
- **ngpblks** (296): Number of grid point blocks
- **krr** (6): Number of rain/ice species
- **krr1** (7): Number of rain/ice species + 1 (includes index 0)

**Total grid points:** ngpblks × nproma = 296 × 32 = 9,472 horizontal points

## Variables

### Diagnostic Variables (2D: ngpblks × nproma)

| Variable | Long Name | Units | Description |
|----------|-----------|-------|-------------|
| `ZSIGQSAT` | Sigma at saturation | 1 | Saturation sigma parameter |
| `ZICE_CLD_WGT` | Ice cloud weight | 1 | Weight for ice cloud fraction |

### Meteorological State Variables (3D: ngpblks × nflevg × nproma)

| Variable | Long Name | Units | Description |
|----------|-----------|-------|-------------|
| `PRHODJ` | Density × Jacobian | kg/m³ | Product of density and grid Jacobian |
| `PEXNREF` | Reference Exner function | 1 | Exner function reference state |
| `PRHODREF` | Reference density | kg/m³ | Reference density profile |
| `PSIGS` | Sigma_s at t | 1 | Sigma parameter at current time |
| `PMFCONV` | Mass flux from convection | kg/m²/s | Convective mass flux |
| `PPABSM` | Absolute pressure at mass point | Pa | Absolute pressure |
| `ZZZ` | Height/altitude | m | Geometric height |

### Cloud Fraction Variables (3D: ngpblks × nflevg × nproma)

| Variable | Long Name | Units | Description |
|----------|-----------|-------|-------------|
| `PCF_MF` | Cloud fraction from mass flux | 1 | Cloud fraction diagnosed from mass flux scheme |
| `PRC_MF` | Cloud water from mass flux | kg/kg | Liquid water mixing ratio from mass flux |
| `PRI_MF` | Ice from mass flux | kg/kg | Ice mixing ratio from mass flux |

### Thermodynamic Variables (3D: ngpblks × nflevg × nproma)

| Variable | Long Name | Units | Description |
|----------|-----------|-------|-------------|
| `PTHT` | Potential temperature at t | K | Potential temperature at current time (input) |
| `PTHS` | Potential temperature | K | Potential temperature (used in scheme) |

### Mixing Ratio Variables

**4D Variables (ngpblks × krr × nflevg × nproma or similar ordering):**

| Variable | Long Name | Units | Description | Species Index |
|----------|-----------|-------|-------------|---------------|
| `ZRS` | Precipitation rates | kg/kg | Mixing ratios (0:KRR) | 0-6 (krr1 dimension) |
| `PRS` | Mixing ratios | kg/kg | Input mixing ratios for 6 species | 1-6 (krr dimension) |
| `PRS_OUT` | Output mixing ratios | kg/kg | Output mixing ratios after ice adjustment | 1-6 (krr dimension) |

**Species (krr=6):**
1. Vapor (RV)
2. Cloud water (RC)
3. Rain (RR)
4. Ice (RI)
5. Snow (RS)
6. Graupel (RG)

### Output Diagnostic Variables (3D: ngpblks × nflevg × nproma)

| Variable | Long Name | Units | Description |
|----------|-----------|-------|-------------|
| `PSRCS_OUT` | Output source/sink | kg/kg/s | Total source/sink term for mixing ratios |
| `PCLDFR_OUT` | Output cloud fraction | 1 | Cloud fraction after ice adjustment |
| `PHLC_HRC_OUT` | Liquid cloud hydrometeor output | kg/kg | Liquid cloud water for radiation |
| `PHLC_HCF_OUT` | Liquid cloud fraction output | 1 | Liquid cloud fraction for radiation |
| `PHLI_HRI_OUT` | Ice cloud hydrometeor output | kg/kg | Ice cloud water for radiation |
| `PHLI_HCF_OUT` | Ice cloud fraction output | 1 | Ice cloud fraction for radiation |

## Usage Example

```python
import xarray as xr
import numpy as np

# Load the dataset
ds = xr.open_dataset('data/ice_adjust.nc')

print(f"Grid dimensions: {ds.dims}")
print(f"Total horizontal points: {ds.dims['ngpblks'] * ds.dims['nproma']}")
print(f"Vertical levels: {ds.dims['nflevg']}")

# Access potential temperature
ptht = ds['PTHT']
print(f"Temperature range: {ptht.min().values:.2f} - {ptht.max().values:.2f} K")

# Access mixing ratios (4D: ngpblks × krr × nflevg × nproma)
prs = ds['PRS']
print(f"Mixing ratios shape: {prs.shape}")

# Extract a specific species (e.g., cloud water, index 1)
cloud_water = prs[:, 1, :, :]  # ngpblks × nflevg × nproma
print(f"Cloud water range: {cloud_water.min().values:.2e} - {cloud_water.max().values:.2e} kg/kg")

# Compare input and output mixing ratios
prs_out = ds['PRS_OUT']
delta_prs = prs_out - prs
print(f"Max change in mixing ratios: {np.abs(delta_prs).max().values:.2e} kg/kg")

# Flatten spatial dimensions for easier analysis
# Reshape from (ngpblks, nflevg, nproma) to (ngpblks*nproma, nflevg)
n_horiz = ds.dims['ngpblks'] * ds.dims['nproma']
n_vert = ds.dims['nflevg']

ptht_flat = ptht.values.reshape(ds.dims['ngpblks'], ds.dims['nflevg'], ds.dims['nproma'])
ptht_2d = ptht_flat.transpose(0, 2, 1).reshape(n_horiz, n_vert)
print(f"Flattened temperature shape: {ptht_2d.shape}")
```

## Physical Interpretation

The ICE_ADJUST scheme adjusts cloud ice and liquid water to maintain saturation equilibrium:

1. **Input State:**
   - Atmospheric state: `PRHODJ`, `PEXNREF`, `PRHODREF`, `PPABSM`
   - Thermodynamic state: `PTHT` (potential temperature)
   - Mixing ratios: `PRS` (6 species)
   - Cloud fraction from convection: `PCF_MF`, `PRC_MF`, `PRI_MF`

2. **Adjustment Process:**
   - Diagnoses saturation state using `ZSIGQSAT`
   - Adjusts ice/liquid partitioning based on `ZICE_CLD_WGT`
   - Updates mixing ratios and cloud fraction

3. **Output:**
   - Adjusted mixing ratios: `PRS_OUT`
   - Cloud fraction: `PCLDFR_OUT`
   - Cloud hydrometeors for radiation: `PHLC_HRC_OUT`, `PHLI_HRI_OUT`
   - Tendencies: `PSRCS_OUT`

## Grid Structure

The data uses a blocked grid structure for computational efficiency:

```
Total points = ngpblks × nproma = 296 × 32 = 9,472 horizontal points
Vertical levels = nflevg = 15

Block structure:
  for iblock in 1:ngpblks
    for ipoint in 1:nproma
      for ilevel in 1:nflevg
        # Process grid point (iblock, ilevel, ipoint)
```

To convert to standard (horizontal, vertical) layout:
```python
# From (ngpblks, nflevg, nproma) to (ngpblks*nproma, nflevg)
data_blocked = ds['PTHT'].values  # shape: (296, 15, 32)
data_2d = data_blocked.transpose(0, 2, 1).reshape(-1, 15)  # shape: (9472, 15)
```

## Scheme Parameters

- **KRR = 6**: Number of hydrometeor species (vapor, cloud, rain, ice, snow, graupel)
- **Saturation adjustment**: Performed using subgrid-scale variability (`PSIGS`)
- **Ice/liquid partitioning**: Temperature-dependent using `ZICE_CLD_WGT`

## References

1. PHYEX Documentation: https://github.com/UMR-CNRM/PHYEX
2. `getdata_ice_adjust_mod.F90`: Variable definitions and I/O structure
3. ICE_ADJUST scheme: Adjusts cloud condensate to maintain saturation equilibrium
