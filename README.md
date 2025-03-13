# dwarf-ice3-gt4py

ICE3 microphysics on gt4py.

dwarf-ice3-gt4py is a porting of PHYEX microphysics on gt4py dsl. Original source code can be retrieved on [PHYEX](https://github.com/UMR-CNRM/PHYEX) repository or updated as a submodule in this project -via _install.sh_ script.

## Installation and build

- installation :
    ```
    source install.sh
    ```

- tests :
    ```
    source tests.sh
    ```

- doc :
    ```
    source build_docs.sh
    ```

## Data generation

Data generation script is made to transform _.dat_ files from PHYEX to netcdf with named fields. _.dat_ files are retrieved from PHYEX reproductibility sets (testprogs_data).

```
python testprogs_data/main.py extract-data-ice-adjust ../../PHYEX/tools/testprogs_data/ice_adjust reference.nc ./testprogs_data/ice_adjust.yaml
```

## Microphysical Adjustments (Ice Adjust)

There are three components available for microphysical adjustments, under _/src/ice3_gt4py/components_ directory:
- IceAdjust (ice_adjust.py) : performs condensation and adjustements following supersaturation, and is the mirror of PHYEX's ice_adjust.F90,
- AroAdjust (aro_adjust.py) : combines both stencil collections to reproduce aro_adjust.F90.


- To launch ice_adjust (with cli):
```
python src/drivers/cli.py run-ice-adjust gt:cpu_ifirst ./data/ice_adjust/reference.nc ./data/ice_adjust/run.nc track_ice_adjust.json
```


## Rain Ice

There are three components available for rain_ice (one-moment microphysical processes computation), under _/src/ice3_gt4py/components_ directory:
- RainIce (rain_ice.py) : calls stencils involved in RainIce computation,
- AroRainIce (aro_rain_ice.py) : calls RainIce common computation plus non-negative filters for model coupling,
- Ice4Tendencies (ice4_tendencies.py) : responsible for processes computation,
- Ice4Stepping (ice4_stepping.py) : responsible for orchestration of processes computations (handling soft and heavy cycles plus accumulating tendencies).

- To launch rain_ice (with cli):
```
python src/drivers/cli.py run-rain-ice gt:cpu_ifirst ./data/rain_ice/reference.nc ./data/rain_ice/run.nc track_rain_ice.json
```

## Unit tests with fmodpy

Fortran and GT4Py stencils can be tested side-by-side with test components (_stencil_fortran_ directory).

Fortran routines are issued from CY49T0 version of the code and reworked to eliminate
derivate types from routines. Then both stencils are ran with random numpy arrays
as an input.

## Extraction of .dat files
