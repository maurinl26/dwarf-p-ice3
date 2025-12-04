# Test de reproductibilité - condensation.py vs PHYEX-IAL_CY50T1

## Objectif

Ce test valide que l'implémentation Python GT4Py du schéma de condensation (`src/ice3/stencils/condensation.py`) reproduit fidèlement l'implémentation Fortran de référence issue de PHYEX-IAL_CY50T1.

## État actuel

⚠️ **Le test nécessite le code source Fortran de référence qui n'est pas présent dans le dépôt.**

### Fichier requis manquant

```
src/ice3/stencils_fortran/mode_condensation.F90
```

Ce fichier devrait provenir de:
```
PHYEX-IAL_CY50T1/micro/condensation.F90
```

### Wrapper pré-compilé disponible

Un wrapper Fortran pré-compilé existe dans le répertoire `mode_condensation/`:
- `mode_condensation.arm64.so` : Bibliothèque partagée compilée
- `mode_condensation_python_wrapper.py` : Interface Python
- `mode_condensation_c_wrapper.f90` : Wrapper C pour l'interface

Cependant, ce wrapper fait référence à un fichier source manquant via un lien symbolique.

## Structure du test

### Test: `test_condensation`

Le test compare bit-à-bit (à la tolérance numérique près) les résultats de:

1. **Version Python GT4Py** (`src/ice3/stencils/condensation.py`)
2. **Version Fortran PHYEX** (mode_condensation.F90 - manquant)

### Configuration testée (AROME par défaut)

- `OCND2 = False` : Schéma CB02 (Chaboureau & Bechtold 2002)
- `OUSERI = True` : Utilisation de la phase glace
- `LSIGMAS = True` : Schéma sous-maille activé
- `FRAC_ICE_ADJUST = 0` : Mode AROME pour la fraction de glace
- `CONDENS = 0` : Option CB02 pour la condensation

### Champs validés

#### Principaux
- `pt_out` : Température après condensation [K]
- `prv_out` : Rapport de mélange vapeur d'eau [kg/kg]
- `prc_out` : Rapport de mélange condensat liquide [kg/kg]
- `pri_out` : Rapport de mélange condensat solide [kg/kg]
- `pcldfr` : Fraction nuageuse [0-1]
- `zq1` : Paramètre distribution sous-maille

#### Intermédiaires (diagnostic)
- `zpv`, `zpiv` : Pressions de saturation (eau/glace) [Pa]
- `zfrac` : Fraction de glace [0-1]
- `zqsl`, `zqsi` : Rapports de mélange à saturation [kg/kg]
- `zsigma` : Écart-type sous-maille
- `zcond` : Quantité de condensat
- `za`, `zb` : Coefficients thermodynamiques
- `zsbar` : Sursaturation moyenne sous-maille

### Tolérances

- `rtol=1e-6, atol=1e-6` : Champs principaux
- `rtol=1e-6, atol=1e-8` : Champs intermédiaires

## Installation du code Fortran de référence

Pour exécuter ce test, vous devez:

1. **Obtenir le code source PHYEX-IAL_CY50T1**
   ```bash
   # Cloner ou copier le dépôt PHYEX
   # Le chemin attendu est: ../PHYEX-IAL_CY50T1/micro/condensation.F90
   ```

2. **Créer le lien vers le code Fortran**
   ```bash
   cd src/ice3/stencils_fortran/
   ln -s /chemin/vers/PHYEX-IAL_CY50T1/micro/condensation.F90 mode_condensation.F90
   ```

   Ou copier directement le fichier:
   ```bash
   cp /chemin/vers/PHYEX-IAL_CY50T1/micro/condensation.F90 \
      src/ice3/stencils_fortran/mode_condensation.F90
   ```

3. **Compiler le wrapper Fortran** (optionnel, sera fait automatiquement)
   ```bash
   cd mode_condensation/
   rm mode_condensation.arm64.so  # Forcer la recompilation
   ```

## Exécution du test

Une fois le code Fortran en place:

```bash
# Test avec backend numpy, simple précision
pytest tests/repro/test_condensation.py::test_condensation -v -s -m numpy -k dtypes0

# Test avec backend numpy, double précision  
pytest tests/repro/test_condensation.py::test_condensation -v -s -m numpy -k dtypes1

# Tous les backends
pytest tests/repro/test_condensation.py::test_condensation -v -s
```

## Interprétation des résultats

### Succès attendu

```
================================================================================
TEST DE REPRODUCTIBILITÉ: condensation.py vs PHYEX-IAL_CY50T1
================================================================================
✓ pt_out (température) : OK
✓ prv_out (vapeur d'eau) : OK
✓ prc_out (condensat liquide) : OK
✓ pri_out (condensat solide/glace) : OK
✓ pcldfr (fraction nuageuse) : OK
✓ zq1 (paramètre distribution sous-maille) : OK

Validation des champs intermédiaires:
  ✓ zpv (pression saturation eau)
  ✓ zpiv (pression saturation glace)
  ✓ zfrac (fraction de glace)
  ✓ zqsl (rapport mélange saturation eau)
  ✓ zqsi (rapport mélange saturation glace)
  ✓ zsigma (écart-type sous-maille)
  ✓ zcond (quantité de condensat)
  ✓ za (coefficient thermodynamique a)
  ✓ zb (coefficient thermodynamique b)
  ✓ zsbar (sursaturation moyenne sous-maille)

================================================================================
SUCCÈS: Reproductibilité validée pour tous les champs!
Le stencil Python GT4Py reproduit fidèlement PHYEX-IAL_CY50T1
================================================================================
```

### En cas d'échec

Si des divergences apparaissent, le test affichera:
- Le champ concerné
- Un message d'erreur explicite
- Les différences numériques (via assert_allclose)

Cela peut indiquer:
- Une erreur dans la traduction Python
- Une différence d'arrondi numérique
- Un bug dans l'une des implémentations

## Références

- **PHYEX** : PHYsique EXternalisée - Bibliothèque de paramétrisations physiques
- **IAL_CY50T1** : Version du cycle 50T1 d'ARPEGE/IFS/AROME
- **CB02** : Chaboureau & Bechtold (2002) - Schéma de condensation
- **GT4Py** : GridTools for Python - Framework de génération de code

## Contact

Pour toute question sur ce test ou problème d'installation du code Fortran de référence, contacter l'équipe de développement.
