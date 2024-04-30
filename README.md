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

```
python testprogs_data/main.py extract-data-ice-adjust ../../PHYEX/tools/testprogs_data/ice_adjust reference.nc ./testprogs_data/ice_adjust.yaml
```

## Microphysical Adjustments (Ice Adjust)

There are three components available for microphysical adjustments, under _/src/ice3_gt4py/components_ directory:
- IceAdjust (ice_adjust.py) : performs condensation and adjustements following supersaturation, and is the mirror of PHYEX's ice_adjust.F90,
- AroAdjust (aro_adjust.py) : combines both stencil collections to reproduce aro_adjust.F90.


- To launch ice_adjust (with cli):
```
python src/ice3_gt4py/drivers/cli.py run-ice-adjust gt:cpu_ifirst /data/ice_adjust/reference.nc /data/ice_adjust/run.nc track_ice_adjust.json
```


## Rain Ice
