# dwarf-ice3-gt4py

<<<<<<< HEAD
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

## Microphysical Adjustments (Ice Adjust)

There are three components available for microphysical adjustments, under _/src/ice3_gt4py/components_ directory:
- IceAdjust (ice_adjust.py) : performs condensation and adjustements following supersaturation, and is the mirror of PHYEX's ice_adjust.F90,
- AroFilter (aro_filter.py) : performs non-negative filtering on specific contents
- AroAdjust (aro_adjust.py) : combines both stencil collections to reproduce aro_adjust.F90.
## Rain Ice
=======
ICE3 microphysics on gt4py. 

## Microphysical Adjustments (Ice Adjust)

## Sources and Sedimentation
>>>>>>> 3c00207 (project name change)
