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
  - JAX GPU :

```bash
  docker build --target jax-gpu --platform linux/amd64 -f container/Dockerfile -t ghcr.io/maurinl26/dwarf-p-ice3:jax-gpu .
```

```bash
  docker push ghcr.io/maurinl26/dwarf-p-ice3:jax-gpu
```

  - Cython/Fortran :

```bash
  docker build --target cython-fortran --platform linux/amd64 -f container/Dockerfile -t ghcr.io/maurinl26/dwarf-p-ice3:cython-fortran .
```

```bash
  docker push ghcr.io/maurinl26/dwarf-p-ice3:cython-fortran
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

### SURFEX Surface Physics Backend

Column-parallel surface scheme (Open SURFEX V9.1) with three execution tiers:

| Backend | Compiler | Trigger |
|---|---|---|
| `SurfexJAXGPU` | nvfortran + OpenACC + CUDA graph | `_surfex_wrapper_acc` importable |
| `SurfexJAX` | gfortran (CPU, `jax.pure_callback`) | `libsurfex_offline` available |
| `_NullSurfex` | — (bulk aerodynamic fallback) | always available |

Tiles handled: **ISBA** (nature/vegetation), **SEAFLUX/WATFLUX** (sea/inland water), **FLAKE** (freshwater lakes).

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

## SURFEX Surface Physics

SURFEX (Open SURFEX V9.1, CeCILL-C) provides land/ocean/lake surface fluxes to the
AROME atmospheric physics. It is integrated as a tight coupling inside
[src/ice3/jax/surfex_jax.py](src/ice3/jax/surfex_jax.py).

### Physics

Each atmospheric column is classified by tile type:

| Constant | Value | Scheme | Variables computed |
|---|---|---|---|
| `TILE_NATURE` | 1 | ISBA (Noilhan & Mahfouf 1996) | H, LE, τ, albedo, emissivity |
| `TILE_SEA`    | 2 | SEAFLUX — Charnock (1955) drag | H, LE, τ |
| `TILE_LAKE`   | 3 | FLAKE (Mironov 2010) / WATFLUX | H, LE, τ |

All schemes use the neutral Monin–Obukhov framework: `C_D = (κ / ln(z/z₀))²`.
Sea roughness iterates one Charnock step from a first-guess neutral `C_D`.

### Build

**CPU build (default, gfortran):**
```bash
pip install -e ".[test]"
# → SurfexJAX available if libsurfex_offline.{so,dylib} is compiled separately
# → _NullSurfex (bulk-aerodynamic fallback) always available
```

**GPU build (nvfortran + OpenACC + CUDA graph):**
```bash
export SKBUILD_CMAKE_ARGS="-DCMAKE_Fortran_COMPILER=nvfortran;-DENABLE_OPENACC=ON;-DENABLE_SURFEX=ON"
pip install -e ".[gpu]"
# Produces: libsurfex_acc.so + _surfex_wrapper_acc.{so,pyd}
```

Or directly with CMake:
```bash
mkdir build-gpu && cd build-gpu
cmake .. \
  -DCMAKE_Fortran_COMPILER=nvfortran \
  -DENABLE_OPENACC=ON \
  -DENABLE_SURFEX=ON
make surfex_acc _surfex_wrapper_acc -j$(nproc)
```

### CUDA Graph Capture

The GPU backend (`SurfexJAXGPU`) captures a **CUDA graph** on the first call and
replays it on all subsequent calls, eliminating Python and driver overhead:

```
Call 1  →  acc_set_cuda_stream(1, stream.ptr)  →  cudaStreamBeginCapture
            c_surfex_step_acc(...)  [kernels recorded, not executed]
            cudaStreamEndCapture   →  graph instantiated

Call 2+ →  graph.launch(stream)    [μs replay, zero Python overhead]
```

Key constraint: CuPy input/output buffers are allocated **once** at `__init__` —
their device pointers remain stable across calls, satisfying the CUDA graph requirement.

### Usage

**Factory (automatic tier selection):**
```python
import numpy as np
from ice3.jax.surfex_jax import make_surfex, TILE_NATURE, TILE_SEA, TILE_LAKE

n_cols   = 1024
tiles    = np.ones(n_cols, dtype=np.int32) * TILE_NATURE
tiles[512:] = TILE_SEA

surfex = make_surfex(n_cols, tile_type=tiles)
# → SurfexJAXGPU  if _surfex_wrapper_acc available (nvfortran build)
# → SurfexJAX     if libsurfex_offline available (CPU)
# → _NullSurfex   otherwise (analytical bulk fallback)
```

**Inside AromePhysicsOrchestrator:**
```python
from ice3.jax.surfex_jax import make_surfex, SurfexState

orchestrator = AromePhysicsOrchestrator(
    surfex=make_surfex(n_cols, tile_type=tiles),
    ...
)
# SURFEX is called at each timestep inside the JIT-compiled step()
fluxes = orchestrator.step(state, dt=60.0)
```

**Direct GPU call (Cython level):**
```python
import cupy as cp
from _surfex_wrapper_acc import SurfexGPUWrapper, TILE_NATURE

tiles   = np.ones(1024, dtype=np.int32) * TILE_NATURE
wrapper = SurfexGPUWrapper(n_cols=1024, tile_type=tiles)

# Inputs: DLPack capsules (JAX GPU arrays or CuPy.toDlpack())
out = wrapper(
    t_a=t_a_dlpack, q_a=q_a_dlpack,
    u_a=u_a_dlpack, v_a=v_a_dlpack,
    p_a=p_a_dlpack, rhodref=rho_dlpack,
    sw_down=sw_dlpack, lw_down=lw_dlpack,
    rain_rate=rain_dlpack, snow_rate=snow_dlpack,
    dt=60.0,
)
# out: dict of DLPack capsules
# keys: surf_flux_th, surf_flux_rv, surf_flux_u, surf_flux_v, albedo, emissivity
surf_flux_th = cp.from_dlpack(out['surf_flux_th'])   # (n_cols,) float32 on GPU
```

### Source Files

| File | Role |
|---|---|
| [src/ice3/jax/surfex_jax.py](src/ice3/jax/surfex_jax.py) | Python API: `SurfexJAXGPU`, `SurfexJAX`, `_NullSurfex`, `make_surfex()` |
| [external/open-SURFEX-V9-1-0/bridge/surfex_c_api_acc.F90](external/open-SURFEX-V9-1-0/bridge/surfex_c_api_acc.F90) | Fortran C-binding + OpenACC column loop (ISBA/SEA/LAKE) |
| [external/open-SURFEX-V9-1-0/bridge/_surfex_wrapper_acc.pyx](external/open-SURFEX-V9-1-0/bridge/_surfex_wrapper_acc.pyx) | Cython bridge: CUDA graph capture, CuPy/DLPack exchange |
| [tests/components/test_surfex_gpu.py](tests/components/test_surfex_gpu.py) | Unit + integration tests (skipped without GPU) |

### Tests

```bash
# All SURFEX tests (skipped automatically if GPU/CuPy unavailable)
uv run pytest tests/components/test_surfex_gpu.py -v

# With GPU available
uv run pytest tests/components/test_surfex_gpu.py -v -m gpu
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

