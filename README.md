![ice3-logo](night-cloud-snow.png)

![coverage-badge](coverage.svg)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# ICE3 microphysics on gt4py.

dwarf-p-ice3 is a porting of ice3 microphysics with Python and GT4Py dsl. 
Original source code can be retrieved on [PHYEX](https://github.com/UMR-CNRM/PHYEX)
repository.

The official version for reproducibility is CY50T1.

## Installation and build

### LUMI

Data must be setup on the scratch.

Run debug :

Run GPU :

- Module load + environment variables
```bash
    source ./config/lumi/lumi_env
```

- Interactive session
```bash
    srun --nodes=1  \
    --ntasks-per-node=1 \
    --cpus-per-task=56 \
    --gpus-per-node=1 \
    --account=project_465000527 \
    --partition=dev-g \
    --time=03:00:00  \
    --mem=0 \
    --pty bash
```

- Launch
```bash
    uv run standalone-model ice-adjust-split \
    gt:gpu \
    $SCRATCH_PATH/data/ice_adjust/reference.nc \
    $SCRATCH_PATH/data/ice_adjust/run.nc \
    track_ice_adjust.json
```

#### Warning

It works well with cupy 14.0 and the last versions of gt4py.cartesian (see config [pyproject.toml](pyproject.toml)).

### Atos ECMWF

## Working with Containers

[container](./container) is defined to run dwarf-p-ice3 inside a container with nvidia runtime and python dependencies.

- Build :

```bash
    sudo docker build -t ice3 ./dwarf-p-ice3/container
```

- Retrieve from ghcr.io :

```bash
    docker pull ghcr.io/maurinl26/dwarf-p-ice3
```

### Nice-to-have
- Setup .devcontainer to develop ice3 inside an isolated virtual env with access to gpus
- Setup singularity image to run on an HPC cluster
- Install dependencies with spack

## Data generation for reproductibility

Data generation script is made to transform _.dat_ files from PHYEX to netcdf with named fields. _.dat_ files are retrieved from PHYEX reproductibility sets (testprogs_data).

Load PHYEX testprogs dataset :

- ice_adjust
```bash
  cd ./data/
  wget --no-check-certificate https://github.com/UMR-CNRM/PHYEX/files/12783926/ice_adjust.tar.gz \
   -O ice_adjust.tar.gz
  tar xf ice_adjust.tar.gz
  rm -f ice_adjust.tar.gz
  cd ..
```

- rain_ice :
```bash
  cd ./data/
  wget --no-check-certificate https://github.com/UMR-CNRM/PHYEX/files/12783935/rain_ice.tar.gz \
  -O rain_ice.tar.gz
  tar xf ice_adjust.tar.gz
  rm -f ice_adjust.tar.gz
  cd ..
```

Decode files to netcdf :

```bash
   uv run testprogs-data extract-data-ice-adjust \
   data/ice_adjust/ \
   reference.nc \
   ./src/testprogs_data/ice_adjust.yaml 
```

## Fortran reference source code 

Fortran reference source code is [PHYEX-IAL_CY50T1](https://github.com/UMR-CNRM/PHYEX/releases/tag/IAL_CY50T1) release.

It can be downloaded :

```bash
    wget --no-check-certificate https://github.com/UMR-CNRM/PHYEX/archive/refs/tags/IAL_CY50T1.tar.gz \
      -O IAL_CY50T1.tar.gz
    tar xf IAL_CY50T1.tar.gz
    rm -f IAL_CY50T1.tar.gz
```

## Microphysical Adjustments (Ice Adjust)

There are 2 components available for microphysical adjustments, under [/src/ice3/components](./src/ice3/components) directory:

- IceAdjust (ice_adjust.py) : performs condensation and adjustements following supersaturation, and is the mirror of PHYEX's ice_adjust.F90,
- IceAdjustModular (ice_adjust_modular.py) : ice_adjust written with 4 stencils.
  
To launch ice_adjust (with cli):

```bash
  uv run standalone-model ice-adjust-split \
  gt:cpu_kfirst \
  ./data/ice_adjust/reference.nc \
  ./data/ice_adjust/run.nc \
  track_ice_adjust.json --no-rebuild 
```

## Microphysical Processes (Rain Ice)

There are 2 components available for rain_ice (one-moment microphysical processes computation), under [/src/ice3/components](./src/ice3/components) directory:

- RainIce (rain_ice.py) : rain_ice component,
- Ice4Tendencies (ice4_tendencies.py) : the microphysical species computation of RainIce (equivalent to ice4_tendencies.F90).

To launch rain_ice (with cli):

```bash
    uv run standalone-model ice-adjust-split \
    gt:cpu_kfirst \
    ./data/ice_adjust/reference.nc \
    ./data/ice_adjust/run.nc \
    track_ice_adjust.json --no-rebuild 
```

## Unit tests for compilation and numerical reproducibility

Unit tests for reproductibility are using pytest. Numpy, CPU and GPU backends can be activated :

- Numpy or debug :
  ```bash
  uv run pytest tests/repro -k "debug or numpy"
  ```
- CPU :
  ```bash
  uv run pytest tests/repro -m cpu
  ```
- GPU :
  ```bash
  uv run pytest tests/repro -m gpu
  ```

Fortran and GT4Py stencils can be tested side-by-side with test components ([stencil_fortran](stencil_fortran) directory).

Fortran routines are issued from CY49T0 version of the code and reworked to eliminate
derivate types from routines. Then both stencils are ran with random numpy arrays
as an input.

- conftest.py rassemble toutes les fixtures (utilitaires) pour :
    - les tests : grille, domain, origine de test et config gt4py
    - compile_fortran_stencil(fichier, module, subroutine)
 
## Component tests

Component tests [./tests/components](./tests/components) assess reproducibility using pytest, and checking differences with netcdf references [./data](./data).

## Additional tests

- _gtscript.function_ tests under [./tests/functions](./tests/functions),
- _gtscript.stencil_ tests under [./tests/stencils](./tests/stencils),
- utilities tests under [./tests/utils](./tests/utils).


## Continuous benchmarking

Components under (components)[tests/components] are monitored with continuous benchmarking.

```bash
  bencher run --adapter json \
  --file result.json \
  --token your_bencher_token \
  "uv run pytest tests/components/test_ice_adjust.py -m debug"
```

## Structure du projet 

- [src](./src) :
  - [drivers](./src/drivers) : Command Line Interface'
  - [ice3](./src/ice3/) :
    - [stencils](./src/ice3/stencils) : stencils gt4py et dace,
    - [functions](./src/ice3/functions) : fonctions gt4py,
    - [initialisation](./src/ice3/initialisation) : initialisation des champs (arrays),
    - [phyex_common](./src/ice3/phyex_common) : équivalents des types dérivés fortran, recodés commme dataclasses,
    - [stencils_fortran](./src/ice3/stencils_fortran) : équivalent fortran des stencisl gt4py (modules + 1 subroutine = 1 stencil gt4py),
    - [utils](./src/ice3/utils) : utilitaires pour la config et l'allocation des champs.
- [tests](./tests) : tests de reproductibilité et de performance.

## Work in Progress

### (WIP) Integration with PHYEX
  
- directives Serialbox / Netcdf pour les standalone rain_ice / ice_adjust,
- intégration des composants DaCe (C++) -> voir [DaCe-Fortran-utilities](https://github.com/maurinl26/DaCe-Fortran-utilities)

### (WIP) Integration with PMAP-L

- intégration ice_adjust,
- intégration rain_ice

### (WIP) Lookup Tables

Les lookup tables ne sont pas implémentées en gt4py (indexation dynamique), et restent à implémenter.

- diagnostic sigma rc : [mode_sigrc_computation.F90](./src/ice3/stencils_fortran/mode_sigrc_computation.F90) :

Detailed issue : 

GT4Py (cartesian) does not manage the interpolation on relative index "SRC_1D(INQ1 + 24)".

```fortran

SUBROUTINE SIGRC_COMPUTATION(NIJT, NKT, NKTE, NKTB,     NIJE, NIJB, &
HLAMBDA3, ZQ1, PSIGRC, INQ1)
!!
IMPLICIT NONE

INTEGER, INTENT(IN) :: NIJT, NKT, NKTE, NKTB, NIJE, NIJB
INTEGER, INTENT(IN) :: HLAMBDA3
REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: ZQ1
REAL, DIMENSION(NIJT,NKT), INTENT(OUT) :: PSIGRC
INTEGER, DIMENSION(NIJT,NKT), INTENT(OUT) :: INQ1

INTEGER :: JIJ, JK, INQ2
REAL :: ZINC

DO JK = NKTB, NKTE
  DO JIJ = NIJB, NIJE
    
    INQ1(JIJ,JK) = FLOOR(MIN(100.0, MAX(-100.0, 2.0 * ZQ1(JIJ,JK))))
    INQ2 = MIN(MAX(-22, INQ1(JIJ,JK)), 10)
    
    ZINC = 2.0 * ZQ1(JIJ,JK) - REAL(INQ2)
    PSIGRC(JIJ,JK) = MIN(1.0, (1.0 - ZINC) * SRC_1D(INQ2 + 23) + &
                                     ZINC * SRC_1D(INQ2 + 24))
    
  END DO
END DO

END SUBROUTINE SIGRC_COMPUTATION

```

- kernels [ice4_fast_rs.py](src/ice3/stencils/ice4_fast_rs.py) et [ice4_fast_rg.py](src/ice3/stencils/ice4_fast_rg.py),





