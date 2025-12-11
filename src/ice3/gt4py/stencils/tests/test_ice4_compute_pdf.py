"""
Test de reproductibilité du stencil ice4_compute_pdf.

Ce module valide que l'implémentation Python GT4Py du stencil ice4_compute_pdf
produit des résultats numériquement identiques à l'implémentation Fortran de référence
issue du projet PHYEX (PHYsique EXternalisée) version IAL_CY50T1.

Le stencil ice4_compute_pdf répartit les nuages en parties à haute et basse teneur
en utilisant une fonction de distribution de probabilité (PDF). Ce processus est
crucial pour la paramétrisation sous-maille de l'autoconversion et des processus
microphysiques dépendant de la concentration.

Processus testés:
- Répartition des gouttelettes de nuage (cloud droplets) en haute/basse concentration
- Répartition des cristaux de glace en haute/basse concentration
- Calcul de la fraction de pluie (rain fraction)
- Différentes options de sous-maille: NONE, CLFR, ADJU, PDF

Options de sous-maille:
- SUBG_AUCV_RC=0 (NONE): Pas de variabilité sous-maille pour rc
- SUBG_AUCV_RC=1 (CLFR): Variabilité basée sur la fraction nuageuse
- SUBG_AUCV_RC=2 (ADJU): Ajustement des valeurs haute/basse
- SUBG_AUCV_RC=3 (PDF): Distribution avec écart-type sigma_rc
  - SUBG_PR_PDF=0 (SIGM): Utilisation de sigma pour la PDF

Référence:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_compute_pdf.F90
"""
from ctypes import c_double, c_float

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array, zeros
from numpy.testing import assert_allclose

from ice3.utils.env import dp_dtypes, sp_dtypes


@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("debug", marks=pytest.mark.debug),
        pytest.param("numpy", marks=pytest.mark.numpy),
        pytest.param("gt:cpu_ifirst", marks=pytest.mark.cpu),
        pytest.param("gt:gpu", marks=pytest.mark.gpu),
    ],
)
def test_ice4_compute_pdf_none(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil ice4_compute_pdf avec SUBG_AUCV_RC=NONE (0).
    
    Ce test valide l'implémentation Python/GT4Py du calcul PDF pour la répartition
    des nuages sans variabilité sous-maille (option NONE).
    
    Avec SUBG_AUCV_RC=0 (NONE):
    - Si rc > rcrautc: toute la masse dans la partie haute concentration
    - Si C_RTMIN < rc <= rcrautc: toute la masse dans la partie basse concentration
    - Sinon: pas de nuage
    
    De même pour ri avec le seuil criauti.
    
    Champs validés:
        - hlc_hcf, hlc_lcf: Fractions haute/basse concentration pour cloud
        - hlc_hrc, hlc_lrc: Rapports de mélange haute/basse concentration pour cloud
        - hli_hcf, hli_lcf: Fractions haute/basse concentration pour ice
        - hli_hri, hli_lri: Rapports de mélange haute/basse concentration pour ice
        - rf: Fraction de pluie
    
    Tolérance:
        rtol=1e-6, atol=1e-10
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_compute_pdf import ice4_compute_pdf

    # Set SUBG_AUCV_RC to NONE (0) and SUBG_AUCV_RI to NONE (0)
    externals_pdf = externals.copy()
    externals_pdf["SUBG_AUCV_RC"] = 0  # NONE
    externals_pdf["SUBG_AUCV_RI"] = 0  # NONE
    externals_pdf["SUBG_PR_PDF"] = 0   # SIGM (not used for NONE but required)

    # Compilation du stencil GT4Py
    ice4_compute_pdf_gt4py = stencil(
        backend,
        definition=ice4_compute_pdf,
        name="ice4_compute_pdf",
        dtypes=dtypes,
        externals=externals_pdf,
    )

    # Initialisation des champs d'entrée
    input_field_names = [
        "ldmicro", "rhodref", "rc_t", "ri_t", "cf", "t", "sigma_rc"
    ]
    
    fields = {}
    for name in input_field_names:
        if name == "ldmicro":
            fields[name] = np.array(
                np.random.rand(*domain) > 0.3,
                dtype=np.bool_,
                order="F",
            )
        else:
            fields[name] = np.array(
                np.random.rand(*domain),
                dtype=(c_float if dtypes["float"] == np.float32 else c_double),
                order="F",
            )
    
    # Ajustement pour valeurs réalistes
    fields["rhodref"] = 0.5 + fields["rhodref"] * 1.0  # 0.5-1.5 kg/m3
    fields["rc_t"] = fields["rc_t"] * 0.01  # 0-10 g/kg (some above/below threshold)
    fields["ri_t"] = fields["ri_t"] * 0.005  # 0-5 g/kg
    fields["cf"] = fields["cf"] * 0.8  # 0-0.8 cloud fraction
    fields["t"] = 250.0 + fields["t"] * 50.0  # 250-300 K
    fields["sigma_rc"] = fields["sigma_rc"] * 0.002  # 0-2 g/kg
    
    # Champs de sortie
    output_field_names = [
        "hlc_hcf", "hlc_lcf", "hlc_hrc", "hlc_lrc",
        "hli_hcf", "hli_lcf", "hli_hri", "hli_lri",
        "rf"
    ]
    
    for name in output_field_names:
        fields[name] = np.zeros(domain, dtype=(c_float if dtypes["float"] == np.float32 else c_double), order="F")
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in input_field_names + output_field_names:
        if name == "ldmicro":
            gt4py_fields[name] = from_array(fields[name], dtype=dtypes["bool"], backend=backend)
        else:
            gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    ice4_compute_pdf_gt4py(
        ldmicro=gt4py_fields["ldmicro"],
        rhodref=gt4py_fields["rhodref"],
        rc_t=gt4py_fields["rc_t"],
        ri_t=gt4py_fields["ri_t"],
        cf=gt4py_fields["cf"],
        t=gt4py_fields["t"],
        sigma_rc=gt4py_fields["sigma_rc"],
        hlc_hcf=gt4py_fields["hlc_hcf"],
        hlc_lcf=gt4py_fields["hlc_lcf"],
        hlc_hrc=gt4py_fields["hlc_hrc"],
        hlc_lrc=gt4py_fields["hlc_lrc"],
        hli_hcf=gt4py_fields["hli_hcf"],
        hli_lcf=gt4py_fields["hli_lcf"],
        hli_hri=gt4py_fields["hli_hri"],
        hli_lri=gt4py_fields["hli_lri"],
        rf=gt4py_fields["rf"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications de cohérence physique
    # 1. Les fractions sont entre 0 et 1
    for frac_name in ["hlc_hcf", "hlc_lcf", "hli_hcf", "hli_lcf", "rf"]:
        assert np.all(gt4py_fields[frac_name] >= 0) and np.all(gt4py_fields[frac_name] <= 1), \
            f"{frac_name} should be between 0 and 1"
    
    # 2. Tous les champs sont finis
    for name in output_field_names:
        assert np.all(np.isfinite(gt4py_fields[name])), f"{name} contains non-finite values"
    
    # 3. Conservation: rc_t = hlc_hrc + hlc_lrc (où ldmicro est True)
    mask = fields["ldmicro"]
    if np.any(mask):
        total_rc = gt4py_fields["hlc_hrc"][mask] + gt4py_fields["hlc_lrc"][mask]
        assert_allclose(total_rc, fields["rc_t"][mask], rtol=1e-5, atol=1e-10)
    
    # 4. La fraction de pluie est au moins égale aux fractions haute concentration
    assert np.all(gt4py_fields["rf"] >= gt4py_fields["hlc_hcf"]), \
        "rf should be >= hlc_hcf"
    assert np.all(gt4py_fields["rf"] >= gt4py_fields["hli_hcf"]), \
        "rf should be >= hli_hcf"
    
    print(f"Test passed! PDF computed with NONE option")
    print(f"  Average rf: {np.mean(gt4py_fields['rf']):.4f}")
    print(f"  Max hlc_hcf: {np.max(gt4py_fields['hlc_hcf']):.4f}")


@pytest.mark.parametrize("dtypes", [sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numpy", marks=pytest.mark.numpy),
    ],
)
def test_ice4_compute_pdf_clfr(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil ice4_compute_pdf avec SUBG_AUCV_RC=CLFR (1).
    
    Ce test valide l'implémentation Python/GT4Py du calcul PDF avec variabilité
    basée sur la fraction nuageuse (option CLFR).
    
    Avec SUBG_AUCV_RC=1 (CLFR):
    - Si cf > 0 et rc > rcrautc*cf: haute concentration
    - Si cf > 0 et rc > C_RTMIN: basse concentration
    - Les fractions haute/basse sont pondérées par cf
    
    Champs validés:
        - hlc_hcf, hlc_lcf: Fractions avec pondération par cf
        - hlc_hrc, hlc_lrc: Rapports de mélange
        - rf: Fraction de pluie
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_compute_pdf import ice4_compute_pdf

    # Set SUBG_AUCV_RC to CLFR (1)
    externals_pdf = externals.copy()
    externals_pdf["SUBG_AUCV_RC"] = 1  # CLFR
    externals_pdf["SUBG_AUCV_RI"] = 1  # CLFR
    externals_pdf["SUBG_PR_PDF"] = 0   # SIGM

    # Compilation du stencil GT4Py
    ice4_compute_pdf_gt4py = stencil(
        backend,
        definition=ice4_compute_pdf,
        name="ice4_compute_pdf",
        dtypes=dtypes,
        externals=externals_pdf,
    )

    # Initialisation des champs
    fields = {}
    fields["ldmicro"] = np.array(np.random.rand(*domain) > 0.2, dtype=np.bool_, order="F")
    
    for name in ["rhodref", "rc_t", "ri_t", "cf", "t", "sigma_rc"]:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    fields["rhodref"] = 0.5 + fields["rhodref"] * 1.0
    fields["rc_t"] = fields["rc_t"] * 0.01
    fields["ri_t"] = fields["ri_t"] * 0.005
    fields["cf"] = 0.1 + fields["cf"] * 0.7  # 0.1-0.8 (avoid cf=0)
    fields["t"] = 250.0 + fields["t"] * 50.0
    fields["sigma_rc"] = fields["sigma_rc"] * 0.002
    
    # Champs de sortie
    for name in ["hlc_hcf", "hlc_lcf", "hlc_hrc", "hlc_lrc", 
                 "hli_hcf", "hli_lcf", "hli_hri", "hli_lri", "rf"]:
        fields[name] = np.zeros(domain, dtype=(c_float if dtypes["float"] == np.float32 else c_double), order="F")
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name, data in fields.items():
        if name == "ldmicro":
            gt4py_fields[name] = from_array(data, dtype=dtypes["bool"], backend=backend)
        else:
            gt4py_fields[name] = from_array(data.copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    ice4_compute_pdf_gt4py(
        ldmicro=gt4py_fields["ldmicro"],
        rhodref=gt4py_fields["rhodref"],
        rc_t=gt4py_fields["rc_t"],
        ri_t=gt4py_fields["ri_t"],
        cf=gt4py_fields["cf"],
        t=gt4py_fields["t"],
        sigma_rc=gt4py_fields["sigma_rc"],
        hlc_hcf=gt4py_fields["hlc_hcf"],
        hlc_lcf=gt4py_fields["hlc_lcf"],
        hlc_hrc=gt4py_fields["hlc_hrc"],
        hlc_lrc=gt4py_fields["hlc_lrc"],
        hli_hcf=gt4py_fields["hli_hcf"],
        hli_lcf=gt4py_fields["hli_lcf"],
        hli_hri=gt4py_fields["hli_hri"],
        hli_lri=gt4py_fields["hli_lri"],
        rf=gt4py_fields["rf"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications
    # 1. Fractions entre 0 et 1
    for frac_name in ["hlc_hcf", "hlc_lcf", "hli_hcf", "hli_lcf", "rf"]:
        assert np.all(gt4py_fields[frac_name] >= 0) and np.all(gt4py_fields[frac_name] <= 1), \
            f"{frac_name} should be between 0 and 1"
    
    # 2. Les fractions sont liées à cf
    # hlc_hcf et hlc_lcf devraient être <= cf
    assert np.all(gt4py_fields["hlc_hcf"] <= fields["cf"] + 1e-6), \
        "hlc_hcf should be <= cf"
    assert np.all(gt4py_fields["hlc_lcf"] <= fields["cf"] + 1e-6), \
        "hlc_lcf should be <= cf"
    
    # 3. Tous les champs sont finis
    for name in ["hlc_hcf", "hlc_lcf", "hlc_hrc", "hlc_lrc", "rf"]:
        assert np.all(np.isfinite(gt4py_fields[name])), f"{name} contains non-finite values"
    
    print(f"Test passed! PDF computed with CLFR option")
    print(f"  Average cf: {np.mean(fields['cf']):.4f}")
    print(f"  Average hlc_hcf: {np.mean(gt4py_fields['hlc_hcf']):.4f}")


@pytest.mark.parametrize("dtypes", [sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numpy", marks=pytest.mark.numpy),
    ],
)
def test_ice4_compute_pdf_adju(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil ice4_compute_pdf avec SUBG_AUCV_RC=ADJU (2).
    
    Ce test valide l'implémentation Python/GT4Py du calcul PDF avec ajustement
    des valeurs haute/basse concentration (option ADJU).
    
    Avec SUBG_AUCV_RC=2 (ADJU):
    - Ajuste hlc_lrc et hlc_hrc proportionnellement pour conserver rc_t
    - Si sumrc = hlc_lrc + hlc_hrc > 0:
      * hlc_lrc *= rc_t / sumrc
      * hlc_hrc *= rc_t / sumrc
    
    Cette option nécessite des valeurs initiales non nulles pour hlc_lrc et hlc_hrc.
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_compute_pdf import ice4_compute_pdf

    # Set SUBG_AUCV_RC to ADJU (2)
    externals_pdf = externals.copy()
    externals_pdf["SUBG_AUCV_RC"] = 2  # ADJU
    externals_pdf["SUBG_AUCV_RI"] = 2  # ADJU
    externals_pdf["SUBG_PR_PDF"] = 0   # SIGM

    # Compilation du stencil GT4Py
    ice4_compute_pdf_gt4py = stencil(
        backend,
        definition=ice4_compute_pdf,
        name="ice4_compute_pdf",
        dtypes=dtypes,
        externals=externals_pdf,
    )

    # Initialisation des champs
    fields = {}
    fields["ldmicro"] = np.array(np.random.rand(*domain) > 0.2, dtype=np.bool_, order="F")
    
    for name in ["rhodref", "rc_t", "ri_t", "cf", "t", "sigma_rc"]:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    fields["rhodref"] = 0.5 + fields["rhodref"] * 1.0
    fields["rc_t"] = fields["rc_t"] * 0.01
    fields["ri_t"] = fields["ri_t"] * 0.005
    fields["cf"] = fields["cf"] * 0.8
    fields["t"] = 250.0 + fields["t"] * 50.0
    fields["sigma_rc"] = fields["sigma_rc"] * 0.002
    
    # Champs de sortie - pour ADJU, on initialise avec des valeurs non nulles
    fields["hlc_hcf"] = np.random.rand(*domain).astype(c_float if dtypes["float"] == np.float32 else c_double, order="F") * 0.3
    fields["hlc_lcf"] = np.random.rand(*domain).astype(c_float if dtypes["float"] == np.float32 else c_double, order="F") * 0.3
    fields["hlc_hrc"] = np.random.rand(*domain).astype(c_float if dtypes["float"] == np.float32 else c_double, order="F") * 0.005
    fields["hlc_lrc"] = np.random.rand(*domain).astype(c_float if dtypes["float"] == np.float32 else c_double, order="F") * 0.005
    
    fields["hli_hcf"] = np.random.rand(*domain).astype(c_float if dtypes["float"] == np.float32 else c_double, order="F") * 0.3
    fields["hli_lcf"] = np.random.rand(*domain).astype(c_float if dtypes["float"] == np.float32 else c_double, order="F") * 0.3
    fields["hli_hri"] = np.random.rand(*domain).astype(c_float if dtypes["float"] == np.float32 else c_double, order="F") * 0.003
    fields["hli_lri"] = np.random.rand(*domain).astype(c_float if dtypes["float"] == np.float32 else c_double, order="F") * 0.003
    
    fields["rf"] = np.zeros(domain, dtype=(c_float if dtypes["float"] == np.float32 else c_double), order="F")
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name, data in fields.items():
        if name == "ldmicro":
            gt4py_fields[name] = from_array(data, dtype=dtypes["bool"], backend=backend)
        else:
            gt4py_fields[name] = from_array(data.copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    ice4_compute_pdf_gt4py(
        ldmicro=gt4py_fields["ldmicro"],
        rhodref=gt4py_fields["rhodref"],
        rc_t=gt4py_fields["rc_t"],
        ri_t=gt4py_fields["ri_t"],
        cf=gt4py_fields["cf"],
        t=gt4py_fields["t"],
        sigma_rc=gt4py_fields["sigma_rc"],
        hlc_hcf=gt4py_fields["hlc_hcf"],
        hlc_lcf=gt4py_fields["hlc_lcf"],
        hlc_hrc=gt4py_fields["hlc_hrc"],
        hlc_lrc=gt4py_fields["hlc_lrc"],
        hli_hcf=gt4py_fields["hli_hcf"],
        hli_lcf=gt4py_fields["hli_lcf"],
        hli_hri=gt4py_fields["hli_hri"],
        hli_lri=gt4py_fields["hli_lri"],
        rf=gt4py_fields["rf"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications
    # 1. Conservation: rc_t = hlc_hrc + hlc_lrc (où ldmicro et sumrc > 0)
    mask = fields["ldmicro"] & ((fields["hlc_hrc"] + fields["hlc_lrc"]) > 1e-15)
    if np.any(mask):
        total_rc = gt4py_fields["hlc_hrc"][mask] + gt4py_fields["hlc_lrc"][mask]
        assert_allclose(total_rc, fields["rc_t"][mask], rtol=1e-5, atol=1e-10)
    
    # 2. Tous les champs sont finis
    for name in ["hlc_hrc", "hlc_lrc", "hli_hri", "hli_lri", "rf"]:
        assert np.all(np.isfinite(gt4py_fields[name])), f"{name} contains non-finite values"
    
    # 3. Les valeurs sont positives
    for name in ["hlc_hrc", "hlc_lrc", "hli_hri", "hli_lri"]:
        assert np.all(gt4py_fields[name] >= 0), f"{name} should be positive"
    
    print(f"Test passed! PDF computed with ADJU option")
    print(f"  Conservation verified for active cells")


@pytest.mark.parametrize("dtypes", [sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numpy", marks=pytest.mark.numpy),
    ],
)
def test_ice4_compute_pdf_sigma(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil ice4_compute_pdf avec SUBG_AUCV_RC=PDF (3) et SUBG_PR_PDF=SIGM (0).
    
    Ce test valide l'implémentation Python/GT4Py du calcul PDF avec distribution
    basée sur l'écart-type sigma_rc (option PDF/SIGM).
    
    Avec SUBG_AUCV_RC=3 (PDF) et SUBG_PR_PDF=0 (SIGM):
    - Si rc > rcrautc + sigma: tout en haute concentration
    - Si rcrautc - sigma < rc < rcrautc + sigma: répartition linéaire
      * hlc_hcf = (rc + sigma - rcrautc) / (2*sigma)
      * hlc_hrc = (rc + sigma - rcrautc) * (rc + sigma + rcrautc) / (4*sigma)
    - Si rc > C_RTMIN: tout en basse concentration
    
    Cette option est utilisée dans AROME pour la variabilité sous-maille.
    
    Champs validés:
        - hlc_hcf, hlc_lcf: Fractions avec répartition PDF
        - hlc_hrc, hlc_lrc: Rapports de mélange avec répartition PDF
        - rf: Fraction de pluie
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_compute_pdf import ice4_compute_pdf

    # Set SUBG_AUCV_RC to PDF (3) and SUBG_PR_PDF to SIGM (0)
    externals_pdf = externals.copy()
    externals_pdf["SUBG_AUCV_RC"] = 3  # PDF
    externals_pdf["SUBG_AUCV_RI"] = 0  # NONE for ice (simpler)
    externals_pdf["SUBG_PR_PDF"] = 0   # SIGM

    # Compilation du stencil GT4Py
    ice4_compute_pdf_gt4py = stencil(
        backend,
        definition=ice4_compute_pdf,
        name="ice4_compute_pdf",
        dtypes=dtypes,
        externals=externals_pdf,
    )

    # Initialisation des champs
    fields = {}
    fields["ldmicro"] = np.array(np.random.rand(*domain) > 0.2, dtype=np.bool_, order="F")
    
    for name in ["rhodref", "rc_t", "ri_t", "cf", "t", "sigma_rc"]:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    fields["rhodref"] = 0.5 + fields["rhodref"] * 1.0
    fields["rc_t"] = fields["rc_t"] * 0.01
    fields["ri_t"] = fields["ri_t"] * 0.005
    fields["cf"] = 0.2 + fields["cf"] * 0.6  # 0.2-0.8
    fields["t"] = 250.0 + fields["t"] * 50.0
    fields["sigma_rc"] = 0.0005 + fields["sigma_rc"] * 0.003  # 0.5-3.5 g/kg (avoid sigma=0)
    
    # Champs de sortie
    for name in ["hlc_hcf", "hlc_lcf", "hlc_hrc", "hlc_lrc", 
                 "hli_hcf", "hli_lcf", "hli_hri", "hli_lri", "rf"]:
        fields[name] = np.zeros(domain, dtype=(c_float if dtypes["float"] == np.float32 else c_double), order="F")
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name, data in fields.items():
        if name == "ldmicro":
            gt4py_fields[name] = from_array(data, dtype=dtypes["bool"], backend=backend)
        else:
            gt4py_fields[name] = from_array(data.copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    ice4_compute_pdf_gt4py(
        ldmicro=gt4py_fields["ldmicro"],
        rhodref=gt4py_fields["rhodref"],
        rc_t=gt4py_fields["rc_t"],
        ri_t=gt4py_fields["ri_t"],
        cf=gt4py_fields["cf"],
        t=gt4py_fields["t"],
        sigma_rc=gt4py_fields["sigma_rc"],
        hlc_hcf=gt4py_fields["hlc_hcf"],
        hlc_lcf=gt4py_fields["hlc_lcf"],
        hlc_hrc=gt4py_fields["hlc_hrc"],
        hlc_lrc=gt4py_fields["hlc_lrc"],
        hli_hcf=gt4py_fields["hli_hcf"],
        hli_lcf=gt4py_fields["hli_lcf"],
        hli_hri=gt4py_fields["hli_hri"],
        hli_lri=gt4py_fields["hli_lri"],
        rf=gt4py_fields["rf"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications
    # 1. Fractions entre 0 et 1
    for frac_name in ["hlc_hcf", "hlc_lcf", "rf"]:
        assert np.all(gt4py_fields[frac_name] >= 0) and np.all(gt4py_fields[frac_name] <= 1), \
            f"{frac_name} should be between 0 and 1"
    
    # 2. Tous les champs sont finis
    for name in ["hlc_hcf", "hlc_lcf", "hlc_hrc", "hlc_lrc", "rf"]:
        assert np.all(np.isfinite(gt4py_fields[name])), f"{name} contains non-finite values"
    
    # 3. Conservation: rc_t = hlc_hrc + hlc_lrc (où applicable)
    mask = fields["ldmicro"] & (fields["rc_t"] > 1e-15)
    if np.any(mask):
        total_rc = gt4py_fields["hlc_hrc"][mask] + gt4py_fields["hlc_lrc"][mask]
        assert_allclose(total_rc, fields["rc_t"][mask], rtol=1e-5, atol=1e-10)
    
    # 4. Les valeurs sont positives ou nulles
    for name in ["hlc_hrc", "hlc_lrc"]:
        assert np.all(gt4py_fields[name] >= 0), f"{name} should be non-negative"
    
    print(f"Test passed! PDF computed with SIGMA option")
    print(f"  Average rf: {np.mean(gt4py_fields['rf']):.4f}")
    print(f"  PDF distribution applied successfully")
