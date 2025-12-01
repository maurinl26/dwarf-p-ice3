"""
Test de reproductibilité du stencil ice4_fast_rs (DaCe) par rapport au Fortran.

Ce module valide que l'implémentation Python DaCe des processus rapides de la neige/agrégats
de la microphysique ICE4 produit des résultats numériquement identiques à l'implémentation 
Fortran de référence issue du projet PHYEX.

Les processus rapides de la neige représentent:
- Le givrage des agrégats par les gouttelettes nuageuses (RCRIMSS, RCRIMSG, RSRIMCG)
- L'accrétion de pluie sur les agrégats (RRACCSS, RRACCSG, RSACCRG)
- La conversion-fonte des agrégats (RSMLTG, RCMLTSR)

Référence:
    mode_ice4_fast_rs.F90
"""

def test_compute_freezing_rate(dtypes, backend, externals, domain, origin):
    ...

def test_cloud_droplet_riming_snow(dtypes, backend, externals, domain, origin):
    ...

def test_rain_accretion_snow(dtypes, backend, externals, domain, origin):
    ...

def test_conversion_melting_snow(dtypes, backend, externals, domain, origin):
    ...