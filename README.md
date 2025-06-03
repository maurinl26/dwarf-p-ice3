# ICE3 microphysics on gt4py.

dwarf-ice3-gt4py is a porting of PHYEX microphysics on gt4py dsl. Original source code can be retrieved on [PHYEX](https://github.com/UMR-CNRM/PHYEX)
repository or updated as a submodule in this project -via _install.sh_ script.

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

It works well with cupy 14.0 and the last versions of gt4py.cartesian (see config _pyproject.toml_) 

### Atos ECMWF

### Leonardo

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

## Microphysical Adjustments (Ice Adjust)

There are three components available for microphysical adjustments, under _/src/ice3_gt4py/components_ directory:

- IceAdjust (ice_adjust.py) : performs condensation and adjustements following supersaturation, and is the mirror of PHYEX's ice_adjust.F90,
- AroAdjust (aro_adjust.py) : combines both stencil collections to reproduce aro_adjust.F90.
- To launch ice_adjust (with cli):

```bash
  uv run standalone-model ice-adjust-split \
  gt:cpu_kfirst \
  ./data/ice_adjust/reference.nc \
  ./data/ice_adjust/run.nc \
  track_ice_adjust.json --no-rebuild 
```

## (WIP) Integration with PHYEX
  
- Option 1:
  - intégration par Serialbox dans IAL_de330 sous ecbuild
  - édition de lien vers Serialbox à faire

- Option 2:
  - intégration des fonctions DaCe (C++)

## (WIP) Integration with PMAP-L

- Option 1:
  - Intégration Python (nécessite peut-être de la réécriture de composants).

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

## Unit tests

Unit tests for reproductibility are using pytest. 

They test the components for every backend.

Fortran and GT4Py stencils can be tested side-by-side with test components (_stencil_fortran_ directory).

Fortran routines are issued from CY49T0 version of the code and reworked to eliminate
derivate types from routines. Then both stencils are ran with random numpy arrays
as an input.

- conftest.py : 
  - tous les utilitaires pour les tests : grille, domain, origine de test et config gt4py
  - compile_fortran_stencil(fichier, module, subroutine)


## Structure du projet 

- src 
  - drivers : Command Line Interface
  - ice3_gt4py :
    - stencils : stencils gt4py et dace
    - functions : fonctions gt4py
    - initialisation : initialisation des champs (arrays)
    - phyex_common : équivalents des modd en python : les modd ont été recodés en dataclasses
    - stencils_fortran : équivalent fortran des stencisl gt4py (modules + 1 subroutine = 1 stencil gt4py)
    - utils : utilitaires pour la config et l'allocation des champs
  - testprogs_data :
    - main : Command Line Interface pour le décodage des testprogs phyex
    - .yaml : config de décodage des fichiers

## Work in Progress

- branche ice_adjust_review :
  - tous les stencils ont été testés unitairement (y compris l'interpolation en DaCe). 
  - l'interpolation (sigrc_computation_dace) doit, être intégrée au composant ice_adjust_split
    -> Couplage stencil dace + stencil gt4py à investiguer
  - il y a un bug, les stencils cloud_fraction_1 et cloud_fraction_2 renvoient des valeurs nulles
    -> Les variables inout doivent être découpées en in / out

  
- branche rain_ice_review :
  - tous les stencils des tendances (Ice4Tendencies) ont été testés
  - les stencils dace ice4_fast_rs (2 stencils gt4py + 2 stencils DaCe) et ice4_fast_rg doivent être intégrés au composant
  - les appels des stencils de Ice4Tendencies doivent être mis à jour (débuggage)
  - les tests unitaires sur le stepping ne sont pas nécessaire -> il est préférable de tester le composant dans un cas 
simple (1 boucle ldsoft)
  - les tests unitaires des rain_fr et sedimentation sont à réaliser

- branches expérimentales :
  - dace-interpolation : intégration des interpolations DaCe dans le stencil
  - dace-orchestration : orchestration DaCe des composants pour livraison en standalone et l'intégration à pmapl :
un composant DaCe fournit sa librairie partagée à la compilation,
  - fortran-plugin : branche pour évaluer les branchements des composants DaCe (code complet)
dans fortran



