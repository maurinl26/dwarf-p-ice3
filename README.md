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

Working with containers :

Tutorial on working with containers and virtual environments is found [here](https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop/blob/ai-20251009/07_Extending_containers_with_virtual_environments_for_faster_testing/examples/extending_containers_with_venv.md)

The base singularity container for Lumi is : 

```bash
    /appl/local/containers/sif-images/lumi-mpi4py-rocm-6.2.0-python-3.12-mpi4py-3.1.6.sif
```


#### Warning

It works well with cupy 14.0 and the last versions of gt4py.cartesian (see config [pyproject.toml](pyproject.toml)).

### Atos ECMWF

Tutorial on working with GPUs on Atos is found [here](https://confluence.ecmwf.int/display/UDOC/HPC2020%3A+GPU+usage+for+AI+and+Machine+Learning)

## Working with Containers

[container](./container) is defined to run dwarf-p-ice3 inside a container with nvidia runtime and python dependencies.

- Build :

```bash
    sudo docker build -t ice3 ./dwarf-p-ice3/container
```

- Retrieve from ghcr.io :
  1. With docker :
    ```bash
        docker pull ghcr.io/maurinl26/dwarf-p-ice3
    ```
  2. With singularity :
     ```
         singularity pull docker://ghcr.io/maurinl26/dwarf-p-ice3
     ```

### Development Container (Devcontainer)

For local development with a consistent, isolated environment, we provide VS Code devcontainer configurations:

- **CPU-only development**: Lightweight Python environment for general development
- **GPU-enabled development**: Full CUDA support for GPU testing (requires NVIDIA GPU + nvidia-docker)

To get started:
1. Install [VS Code](https://code.visualstudio.com/) and [Docker](https://docs.docker.com/get-docker/)
2. Install the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. Open the project and press `F1` → **Remote-Containers: Reopen in Container**

See [.devcontainer/README.md](.devcontainer/README.md) for detailed setup instructions.

### Nice-to-have
- Setup singularity image to run on an HPC cluster

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

## Available Implementations

### JAX Backend (CPU/GPU)

JAX implementations provide automatic differentiation and accelerated computing on CPU and GPU:

- **ice_adjust** ([src/ice3/jax/ice_adjust.py](src/ice3/jax/ice_adjust.py)) - Microphysical adjustments with JAX backend
- **rain_ice** ([src/ice3/jax/rain_ice.py](src/ice3/jax/rain_ice.py)) - One-moment microphysical processes with JAX backend
- **shallow_convection** ([src/ice3/jax/convection/shallow_convection.py](src/ice3/jax/convection/shallow_convection.py)) - Shallow convection scheme with JAX backend

JAX implementations support both CPU and GPU execution with automatic device selection.

### Cython/Fortran Backend

Python wrappers for Fortran implementations with Cython bindings for improved performance:

- **ice_adjust** ([src/ice3/fortran/ice_adjust_cython.py](src/ice3/fortran/ice_adjust_cython.py)) - Ice adjust with Cython/Fortran backend
- **rain_ice** ([src/ice3/fortran/rain_ice_cython.py](src/ice3/fortran/rain_ice_cython.py)) - Rain ice with Cython/Fortran backend

These implementations leverage the original PHYEX Fortran code for maximum performance and reproducibility.

## Microphysical Adjustments (Ice Adjust)

Ice adjust performs condensation and adjustments following supersaturation, mirroring PHYEX's ice_adjust.F90.

**Available backends:**
- GT4Py (legacy)
- JAX (CPU/GPU) - [src/ice3/jax/ice_adjust.py](src/ice3/jax/ice_adjust.py)
- Cython/Fortran - [src/ice3/fortran/ice_adjust_cython.py](src/ice3/fortran/ice_adjust_cython.py)

To launch ice_adjust (with cli):

```bash
  uv run standalone-model ice-adjust-split \
  gt:cpu_kfirst \
  ./data/ice_adjust/reference.nc \
  ./data/ice_adjust/run.nc \
  track_ice_adjust.json --no-rebuild
```

## Microphysical Processes (Rain Ice)

Rain ice performs one-moment microphysical processes computation, including ice4_tendencies (equivalent to ice4_tendencies.F90).

**Available backends:**
- GT4Py (legacy)
- JAX (CPU/GPU) - [src/ice3/jax/rain_ice.py](src/ice3/jax/rain_ice.py)
- Cython/Fortran - [src/ice3/fortran/rain_ice_cython.py](src/ice3/fortran/rain_ice_cython.py)

To launch rain_ice (with cli):

```bash
    uv run standalone-model rain-ice \
    gt:cpu_kfirst \
    ./data/rain_ice/reference.nc \
    ./data/rain_ice/run.nc \
    track_rain_ice.json --no-rebuild
```

## Shallow Convection

Shallow convection scheme implementing the shallow convection parameterization from PHYEX.

**Available backends:**
- JAX (CPU/GPU) - [src/ice3/jax/convection/shallow_convection.py](src/ice3/jax/convection/shallow_convection.py)
- Fortran (reference) - PHYEX-IAL_CY50T1

The JAX implementation is split into modular components:
- [shallow_convection_part1.py](src/ice3/jax/convection/shallow_convection_part1.py)
- [shallow_convection_part2.py](src/ice3/jax/convection/shallow_convection_part2.py)
- [shallow_convection_part2_select.py](src/ice3/jax/convection/shallow_convection_part2_select.py)

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

