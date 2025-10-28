# ICE3 microphysics on gt4py.

dwarf-ice3-gt4py is a porting of PHYEX microphysics on gt4py dsl. Original source code can be retrieved on [PHYEX](https://github.com/UMR-CNRM/PHYEX)
repository or updated as a submodule in this project -via _install.sh_ script.

## Installation and build

[uv](https://docs.astral.sh/uv/#highlights) is required to manage project and virtual environment
    
uv can be download through :

```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
```

Virtual environnement setup :

```bash
    uv init
    uv venv --python 3.10
    source .venv/bin/activate
    uv add --editable .
```

## Rain Ice

There are three components available for rain_ice (one-moment microphysical processes computation), under _/src/ice3_gt4py/components_ directory:

- RainIce (rain_ice.py) : calls stencils involved in RainIce computation,
- AroRainIce (aro_rain_ice.py) : calls RainIce common computation plus non-negative filters for model coupling,
- Ice4Tendencies (ice4_tendencies.py) : responsible for processes computation,
- Ice4Stepping (ice4_stepping.py) : responsible for orchestration of processes computations (handling soft and heavy cycles plus accumulating tendencies).
- To launch rain_ice (with cli):

```bash
  uv run standalone-model \
  gt:cpu_ifirst \
  ./data/rain_ice/reference.nc \
  ./data/rain_ice/run.nc \
  track_rain_ice.json
```

## Unit tests

Unit tests for reproductibility are using pytest. 

Fortran and GT4Py stencils can be tested side-by-side with test components (_stencil_fortran_ directory).

Fortran routines are issued from CY49T0 version of the code and reworked to eliminate
derivate types from routines. Then both stencils are ran with random numpy arrays
as an input.

