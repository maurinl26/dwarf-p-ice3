# Composant ICE_ADJUST Modulaire

## Vue d'ensemble

Le composant `IceAdjustModular` est une implémentation modulaire du schéma d'ajustement microphysique ICE_ADJUST de PHYEX. Contrairement au stencil monolithique `ice_adjust.py`, cette version utilise des stencils séparés pour chaque étape du calcul.

## Architecture

### Stencils utilisés

1. **thermodynamic_fields** (`cloud_fraction.py`)
   - Référence: `ice_adjust.F90`, l.450-473
   - Calcul des champs thermodynamiques de base

2. **condensation** (`condensation.py`)
   - Référence: `condensation.F90`, l.186-575
   - Schéma CB02 (Chaboureau & Bechtold 2002)

3. **cloud_fraction_1** (`cloud_fraction.py`)
   - Référence: `ice_adjust.F90`, l.278-312
   - Sources microphysiques et conservation

4. **cloud_fraction_2** (`cloud_fraction.py`)
   - Référence: `ice_adjust.F90`, l.313-419
   - Fraction nuageuse finale et autoconversion

### Séquence d'exécution

```
┌─────────────────────────────────┐
│  thermodynamic_fields           │
│  ─────────────────────         │
│  IN:  th, exn, rv, rc, rr,     │
│       ri, rs, rg                │
│  OUT: T, Lv, Ls, Cph            │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  condensation (CB02)            │
│  ────────────────────           │
│  IN:  T, rv, rc, ri, pabs,     │
│       sigs, sigqsat, Lv, Ls,   │
│       Cph                       │
│  OUT: rv_out, rc_out, ri_out,  │
│       cldfr, q1                 │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  cloud_fraction_1               │
│  ─────────────────              │
│  IN:  rc_out, ri_out, rc, ri,  │
│       Lv, Ls, Cph, exnref       │
│  INOUT: ths, rvs, rcs, ris      │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  cloud_fraction_2               │
│  ─────────────────              │
│  IN:  rc_mf, ri_mf, cf_mf,     │
│       rhodref, T, Lv, Ls, Cph  │
│  INOUT: ths, rvs, rcs, ris,    │
│         cldfr, hlc_*, hli_*     │
└─────────────────────────────────┘
```

## Utilisation

### Installation

Le composant est automatiquement disponible après installation du package:

```bash
cd /path/to/dwarf-p-ice3
pip install -e .
```

### Exemple basique

```python
from ice3.components.ice_adjust_modular import IceAdjustModular
from ice3.phyex_common.phyex import Phyex
import numpy as np

# Configuration
phyex = Phyex("AROME")
ice_adjust = IceAdjustModular(
    phyex=phyex,
    backend="numpy",  # ou "gt:cpu_ifirst", "gt:gpu"
)

# Préparation des données
domain = (50, 50, 15)  # (ni, nj, nk)
timestep = 50.0  # secondes

# Champs d'entrée (exemples avec valeurs aléatoires)
sigqsat = np.random.rand(domain[0], domain[1])
exn = np.ones(domain) * 1.0
exnref = np.ones(domain) * 1.0
rhodref = np.ones(domain) * 1.2  # kg/m³
pabs = np.ones(domain) * 100000.0  # Pa
sigs = np.random.rand(*domain) * 0.1

# Flux de masse
cf_mf = np.zeros(domain)
rc_mf = np.zeros(domain)
ri_mf = np.zeros(domain)

# Variables météorologiques
th = np.ones(domain) * 300.0  # K
rv = np.random.rand(*domain) * 0.01  # kg/kg
rc = np.random.rand(*domain) * 0.001  # kg/kg
rr = np.zeros(domain)
ri = np.random.rand(*domain) * 0.0001  # kg/kg
rs = np.zeros(domain)
rg = np.zeros(domain)

# Champs de sortie (initialisés)
cldfr = np.zeros(domain)
hlc_hrc = np.zeros(domain)
hlc_hcf = np.zeros(domain)
hli_hri = np.zeros(domain)
hli_hcf = np.zeros(domain)
sigrc = np.zeros(domain)

# Variables sources (copiées des originales)
ths = th.copy()
rvs = rv.copy()
rcs = rc.copy()
ris = ri.copy()

# Exécution
exec_info = {}
ice_adjust(
    sigqsat, exn, exnref, rhodref, pabs, sigs,
    cf_mf, rc_mf, ri_mf,
    th, rv, rc, rr, ri, rs, rg,
    cldfr, hlc_hrc, hlc_hcf, hli_hri, hli_hcf, sigrc,
    ths, rvs, rcs, ris,
    timestep, domain, exec_info
)

# Résultats disponibles dans:
# - ths, rvs, rcs, ris (modifiés)
# - cldfr, hlc_*, hli_* (modifiés)
```

### Configuration avancée

```python
from ice3.utils.env import sp_dtypes, dp_dtypes

# Simple précision (float32)
ice_adjust_sp = IceAdjustModular(
    phyex=Phyex("AROME"),
    dtypes=sp_dtypes,
    backend="numpy"
)

# Double précision (float64)
ice_adjust_dp = IceAdjustModular(
    phyex=Phyex("AROME"),
    dtypes=dp_dtypes,
    backend="numpy"
)

# Backend GPU (si disponible)
ice_adjust_gpu = IceAdjustModular(
    phyex=Phyex("AROME"),
    dtypes=dp_dtypes,
    backend="gt:gpu"
)
```

## Comparaison avec ice_adjust.py monolithique

### Avantages du composant modulaire

1. **Maintenance facilitée**
   - Chaque stencil peut être modifié indépendamment
   - Tests unitaires par stencil possibles
   - Debugging simplifié

2. **Flexibilité**
   - Possibilité de remplacer un stencil par une version alternative
   - Réutilisation des stencils dans d'autres contextes
   - Tests de reproductibilité individuels

3. **Compréhension**
   - Séparation claire des étapes physiques
   - Correspondance directe avec les fichiers PHYEX
   - Documentation par stencil

4. **Performance**
   - Mêmes performances que la version monolithique
   - Potentiel d'optimisation par stencil
   - Gestion mémoire explicite des temporaires

### Inconvénients potentiels

1. **Overhead**
   - 4 appels de stencils vs 1 appel monolithique
   - Gestion des temporaires explicite
   - Légèrement plus de code

2. **Complexité d'interface**
   - Plus de fichiers à maintenir
   - Interface plus verbeu se

## Physique implémentée

### 1. Champs thermodynamiques

Calcule les propriétés thermodynamiques de base:

```
T = θ × Π                           (température absolue)
Lv(T) = Lvtt + (Cpv-Cl)×(T-Ttt)    (chaleur latente vaporisation)
Ls(T) = Lstt + (Cpv-Ci)×(T-Ttt)    (chaleur latente sublimation)  
Cph = Cpd + Cpv×rv + Cl×(rc+rr) + Ci×(ri+rs+rg)  (chaleur spécifique)
```

### 2. Condensation (CB02)

Schéma statistique sous-maille de Chaboureau & Bechtold (2002):

- Distribution gaussienne de l'humidité relative
- Calcul de la fraction nuageuse
- Production de condensats (rc_out, ri_out)
- Conservation de l'eau totale

Équations clés:
```
s̄ = a×(rtot - qsat + ah×Lvs×(rc+ri)/Cph)
σs = √((2×σs)² + (σqsat×qsat×a)²)
q1 = s̄/σs
```

### 3. Sources microphysiques

Conservation de l'eau et ajustement thermique:

```
w1 = (rc_out - rc)/dt              (source liquide)
w2 = (ri_out - ri)/dt              (source glace)

rvs -= w1 + w2                      (conservation vapeur)
rcs += w1                           (accumulation liquide)
ris += w2                           (accumulation glace)
ths += (w1×Lv + w2×Ls)/(Cph×Π)     (ajustement thermique)
```

### 4. Fraction nuageuse et autoconversion

Paramétrisations pour:
- Fraction nuageuse avec flux de masse convectifs
- Autoconversion gouttelettes → pluie
- Autoconversion cristaux → neige
- Deux schémas PDF: None (0) ou Triangle (1)

Critère d'autoconversion liquide:
```
criaut = CRIAUTC / ρ_dry_ref
```

## Validation

### Tests unitaires

Tests pour chaque stencil individuel:
```bash
# Champs thermodynamiques
pytest tests/repro/test_cloud_fraction_repro.py::test_thermodynamic_fields_repro -v

# Condensation
pytest tests/repro/test_condensation.py::test_condensation -v

# Cloud fraction
pytest tests/repro/test_cloud_fraction_repro.py::test_cloud_fraction_1_repro -v
```

### Test d'intégration

Le composant complet devrait produire les mêmes résultats que `ice_adjust.py`:

```python
# À implémenter
pytest tests/components/test_ice_adjust_modular.py -v
```

## Performance

Benchmarks typiques (domaine 50×50×15, backend numpy):

| Étape                | Temps (ms) | % Total |
|---------------------|------------|---------|
| thermodynamic_fields| ~2         | 15%     |
| condensation        | ~6         | 45%     |
| cloud_fraction_1    | ~3         | 20%     |
| cloud_fraction_2    | ~3         | 20%     |
| **Total**           | **~14**    | **100%**|

*Mesures indicatives, dépendent du matériel et de la configuration.*

## Debugging

### Activation des logs

```python
import logging
logging.basicConfig(level=logging.DEBUG)

ice_adjust = IceAdjustModular(...)
# Les logs détaillés seront affichés pendant l'exécution
```

### Validation des arguments

```python
ice_adjust(
    ...,
    validate_args=True  # Active la validation GT4Py
)
```

### Inspection des temporaires

```python
from ice3.utils.storage import managed_temporaries

# Les temporaires sont automatiquement nettoyés
# Pour les inspecter, modifier le code source temporairement
```

## Références

### Code source

- Composant: `src/ice3/components/ice_adjust_modular.py`
- Stencils:
  - `src/ice3/stencils/cloud_fraction.py`
  - `src/ice3/stencils/condensation.py`

### Documentation tests

- `tests/repro/README_cloud_fraction.md`
- `tests/repro/README_condensation.md`

### PHYEX

- `PHYEX-IAL_CY50T1/micro/ice_adjust.F90`
- `PHYEX-IAL_CY50T1/micro/condensation.F90`

### Publications

- Chaboureau & Bechtold (2002): Statistical representation of clouds
- Bechtold et al. (1995): Modeling of convective precipitation

## Contact

Pour toute question sur ce composant, contacter l'équipe de développement dwarf-p-ice3.
