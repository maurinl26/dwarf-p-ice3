# Test de reproductibilité - cloud_fraction.py vs PHYEX-IAL_CY50T1

## Objectif

Ces tests valident que les implémentations Python GT4Py des schémas de fraction nuageuse reproduisent fidèlement les implémentations Fortran de référence PHYEX-IAL_CY50T1.

Les stencils testés font partie du schéma d'ajustement microphysique **ICE_ADJUST** qui calcule:
- Les transferts entre phases (vapeur ↔ liquide ↔ glace)
- La fraction nuageuse avec variabilité sous-maille
- Les champs thermodynamiques (température, chaleurs latentes, chaleur spécifique)

## Référence PHYEX

**Fichier source**: `PHYEX-IAL_CY50T1/micro/ice_adjust.F90`

Les stencils Python sont des traductions directes de sections spécifiques de ce fichier, avec conservation de la numérotation des lignes dans les commentaires.

## Structure des tests

### 1. test_thermodynamic_fields_repro

**Référence**: `ice_adjust.F90`, lignes 450-473

**Objectif**: Valider le calcul des champs thermodynamiques de base

**Champs calculés**:
- `T` : Température absolue [K] (T = θ × Π)
- `Lv` : Chaleur latente de vaporisation [J/kg]
- `Ls` : Chaleur latente de sublimation [J/kg]
- `Cph` : Chaleur spécifique de l'air humide [J/kg/K]

**Configuration**:
- Support 2 à 6 espèces d'hydrométéores (NRR)
- Formules de Clausius-Clapeyron pour Lv et Ls
- Dépendance en température explicite

**Entrées**:
```
th   : Température potentielle [K]
exn  : Fonction d'Exner (sans dimension)
rv   : Rapport de mélange vapeur [kg/kg]
rc   : Rapport de mélange liquide nuageux [kg/kg]
rr   : Rapport de mélange pluie [kg/kg]
ri   : Rapport de mélange glace [kg/kg]
rs   : Rapport de mélange neige [kg/kg]
rg   : Rapport de mélange graupel [kg/kg]
```

**Sorties validées**:
```
T   : rtol=1e-6
Lv  : rtol=1e-6
Ls  : rtol=1e-6
Cph : rtol=1e-6
```

### 2. test_cloud_fraction_1_repro

**Référence**: `ice_adjust.F90`, lignes 278-312

**Objectif**: Valider le calcul des sources microphysiques après la boucle de condensation

**Processus physiques**:
1. Calcul de la variation temporelle des condensats (drc/dt, dri/dt)
2. Limitation des sources par conservation de l'eau totale
3. Mise à jour des rapports de mélange (rv, rc, ri)
4. Ajustement thermique (température potentielle)

**Configuration**:
- `LSUBG_COND = True` : Schéma sous-maille activé (AROME)
- `dt = 50s` : Pas de temps typique

**Entrées principales**:
```
lv, ls    : Chaleurs latentes [J/kg]
cph       : Chaleur spécifique [J/kg/K]
exnref    : Fonction d'Exner de référence
rc, ri    : Condensats au début du pas de temps [kg/kg]
rc_tmp    : Condensat liquide après condensation [kg/kg]
ri_tmp    : Condensat solide après condensation [kg/kg]
ths, rvs  : Température potentielle et vapeur sources [K], [kg/kg]
rcs, ris  : Condensats sources [kg/kg]
```

**Sorties validées**:
```
ths : rtol=1e-6, atol=1e-8
rvs : rtol=1e-6, atol=1e-8
rcs : rtol=1e-6, atol=1e-8
ris : rtol=1e-6, atol=1e-8
```

**Conservation vérifiée**:
- Eau totale : rv + rc + ri = constante
- Énergie : Cph×T + Lv×rc + Ls×ri = constante

### 3. test_cloud_fraction_2 (dans test_cloud_fraction.py original)

**Référence**: `ice_adjust.F90`, lignes 313-419

**Objectif**: Calcul de la fraction nuageuse sous-maille et autoconversion

**Processus physiques**:
1. Calcul de la fraction nuageuse (cldfr) avec schéma sous-maille
2. Transfert des flux de masse convectifs (rc_mf, ri_mf)
3. Autoconversion des gouttelettes en pluie
4. Autoconversion des cristaux en neige
5. Paramétrisations PDF (SUBG_MF_PDF: None ou Triangle)

**Champs de sortie**:
- `cldfr` : Fraction nuageuse [0-1]
- `hlc_hrc`, `hlc_hcf` : Contenu et fraction liquide haute résolution
- `hli_hri`, `hli_hcf` : Contenu et fraction glace haute résolution

## Exécution des tests

### Prérequis

Les tests nécessitent les wrappers Fortran compilés:
```bash
ls mode_cloud_fraction_split/
# Doit contenir: mode_cloud_fraction_split.arm64.so (ou .so selon l'architecture)

ls mode_thermo/
# Doit contenir: mode_thermo.arm64.so
```

Ces wrappers sont généralement pré-compilés et inclus dans le dépôt.

### Commandes d'exécution

```bash
# Test complet (tous les stencils, tous les backends)
pytest tests/repro/test_cloud_fraction_repro.py -v

# Test spécifique avec backend numpy
pytest tests/repro/test_cloud_fraction_repro.py::test_thermodynamic_fields_repro -v -s -m numpy

pytest tests/repro/test_cloud_fraction_repro.py::test_cloud_fraction_1_repro -v -s -m numpy

# Simple vs double précision
pytest tests/repro/test_cloud_fraction_repro.py -v -k "dtypes0"  # Simple précision
pytest tests/repro/test_cloud_fraction_repro.py -v -k "dtypes1"  # Double précision

# Tous les backends
pytest tests/repro/test_cloud_fraction_repro.py -v -m "numpy or cpu or debug"
```

### Tests de performance (benchmark)

Les tests originaux dans `test_cloud_fraction.py` incluent des benchmarks:
```bash
pytest tests/repro/test_cloud_fraction.py::test_thermo -v --benchmark-only
```

## Interprétation des résultats

### Succès attendu

```
================================================================================
TEST REPRODUCTIBILITÉ: thermodynamic_fields vs PHYEX-IAL_CY50T1
================================================================================
✓ T (température)
✓ Lv (chaleur latente vaporisation)
✓ Ls (chaleur latente sublimation)
✓ Cph (chaleur spécifique air humide)

================================================================================
SUCCÈS: Champs thermodynamiques validés!
================================================================================

================================================================================
TEST REPRODUCTIBILITÉ: cloud_fraction_1 vs PHYEX-IAL_CY50T1
================================================================================
✓ ths (température potentielle)
✓ rvs (rapport mélange vapeur)
✓ rcs (rapport mélange liquide)
✓ ris (rapport mélange glace)

================================================================================
SUCCÈS: Fraction nuageuse #1 validée!
================================================================================
```

### En cas d'échec

Les messages d'erreur indiqueront:
- Le champ concerné
- Les différences numériques (via numpy.testing.assert_allclose)
- La tolérance violée (rtol ou atol)

Causes possibles:
1. **Erreur de traduction Python** : Revérifier le code source PHYEX
2. **Différence d'arrondi** : Vérifier la précision (float32 vs float64)
3. **Bug dans l'implémentation** : Comparer ligne par ligne avec Fortran
4. **Problème de compilation GT4Py** : Essayer un backend différent

## Schéma physique - Contexte

### ICE_ADJUST dans AROME

Le schéma ICE_ADJUST est appelé à chaque pas de temps du modèle AROME pour:

1. **Ajuster les phases de l'eau** en fonction de la thermodynamique
2. **Calculer la fraction nuageuse** avec variabilité sous-maille
3. **Gérer l'autoconversion** (cloud → rain, ice → snow)

### Variabilité sous-maille

Le schéma utilise une approche statistique pour représenter la variabilité spatiale non résolue:
- Distribution de l'humidité relative sous-maille
- Fonction de densité de probabilité (PDF)
- Calcul de la fraction couverte par les nuages

### Séquence de calcul

```
1. thermodynamic_fields
   ├─> Calcul T, Lv, Ls, Cph
   └─> Préparation pour la condensation

2. CONDENSATION (autre stencil, voir test_condensation.py)
   ├─> Schéma CB02 (Chaboureau & Bechtold 2002)
   └─> Production rc_tmp, ri_tmp

3. cloud_fraction_1
   ├─> Calcul des sources (drc/dt, dri/dt)
   ├─> Conservation de l'eau
   └─> Ajustement thermique

4. cloud_fraction_2
   ├─> Fraction nuageuse finale
   ├─> Flux de masse convectifs
   └─> Autoconversion
```

## Références

### Publications scientifiques

- **Chaboureau & Bechtold (2002)**: Statistical representation of clouds in a regional model
- **Bechtold et al. (1995)**: Modeling of convective precipitation

### Documentation PHYEX

- Code source: `PHYEX-IAL_CY50T1/micro/ice_adjust.F90`
- Documentation: Manuel AROME (Météo-France)

### Liens utiles

- [GT4Py Documentation](https://gridtools.github.io/gt4py/)
- [PHYEX on GitHub](https://github.com/UMR-CNRM/PHYEX)

## Notes techniques

### Différences Python vs Fortran

1. **Indexation**: Python (0-based) vs Fortran (1-based)
2. **Ordre mémoire**: C-order vs Fortran-order (F-contiguous)
3. **Boucles implicites**: GT4Py stencils vs DO loops Fortran
4. **Constantes**: Externals GT4Py vs MODULE Fortran

### Optimisations

Les stencils GT4Py peuvent être compilés pour différents backends:
- `numpy`: CPU, non optimisé, facile à débugger
- `gt:cpu_ifirst`: CPU optimisé, vectorisation
- `gt:gpu`: GPU (CUDA/HIP), parallélisation massive

### Maintenance

Lors de mises à jour de PHYEX:
1. Vérifier les numéros de lignes dans les commentaires
2. Comparer avec la nouvelle version de ice_adjust.F90
3. Adapter les tests si les interfaces changent
4. Revalider la reproductibilité

## Contact

Pour toute question sur ces tests ou problème de reproductibilité, contacter l'équipe de développement dwarf-p-ice3.
