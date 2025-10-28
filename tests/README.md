# Tests 

Launch tests :

```bash
    uv run pytest tests/repro/ -m debug
```

On DaCe CPU backend :

```bash
    uv run pytest tests/repro/ -m cpu
```

On DaCe GPU backend :

```bash
    uv run pytest tests/repro/ -m gpu
```

## Components

Test the reproducibility of components based on netcdf reference datasets.
See [data](../data) directory.

```bash
    uv run pytest tests/components/ -m debug
```

## Unit tests

Unit tests for reproducibility are using pytest. They test both single and 
double precision codes.

```bash
    uv run pytest tests/repro/ -m debug
```

Fortran and GT4Py stencils can be tested side-by-side with test components (_stencil_fortran_ directory).

Fortran routines are issued from CY49T0 version of the code and reworked to eliminate
derivate types from routines. Then both stencils are ran with random numpy arrays
as an input.


