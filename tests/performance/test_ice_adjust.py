# -*- coding: utf-8 -*-
"""Performance test for IceAdjust component"""
import numpy as np
import pytest

from ice3.components.ice_adjust import IceAdjust
from ice3.phyex_common.phyex import Phyex
from ice3.utils.env import sp_dtypes, dp_dtypes
from ice3.initialisation.state_ice_adjust import get_state_ice_adjust


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
def test_ice_adjust_performance(benchmark, backend, domain, dtypes, ice_adjust_repro_ds):
    """
    Test de performance du composant IceAdjust avec le jeu de données ice_adjust.nc.
    
    Ce test mesure les performances d'exécution du composant IceAdjust
    (ajustement à saturation) sur différents backends GT4Py.
    
    Le benchmark mesure le temps d'exécution moyen sur plusieurs itérations
    pour évaluer les performances du schéma ICE_ADJUST qui effectue:
    - Calcul des variables thermodynamiques (T, Cp, Lv/Cp, Ls/Cp)
    - Ajustement de saturation (condensation/évaporation)
    - Calcul des fractions nuageuses
    - Diagnostics haute résolution
    
    Args:
        benchmark: Fixture pytest-benchmark pour mesurer les performances
        backend: Backend GT4Py (numpy, cpu, gpu, debug)
        domain: Taille du domaine de calcul
        dtypes: Types de données (float32/float64)
        ice_adjust_repro_ds: Dataset xarray ice_adjust.nc
    """
    print("\n" + "="*75)
    print("TEST PERFORMANCE: IceAdjust")
    print("="*75)
    print(f"Backend: {backend}")
    print(f"Precision: {dtypes['float']}")
    
    # ========================================================================
    # Configuration PHYEX et création du composant
    # ========================================================================
    phyex = Phyex("AROME")
    
    # Add OCND2 external that's needed by ice_adjust stencil
    original_to_externals = phyex.to_externals
    def custom_to_externals():
        externals = original_to_externals()
        externals["OCND2"] = False
        return externals
    phyex.to_externals = custom_to_externals
    
    ice_adjust = IceAdjust(
        phyex=phyex,
        dtypes=dtypes,
        backend=backend,
    )
    
    print(f"Composant créé: {ice_adjust}")
    
    # ========================================================================
    # Chargement et préparation des données
    # ========================================================================
    shape = (
        ice_adjust_repro_ds.sizes["ngpblks"],
        ice_adjust_repro_ds.sizes["nproma"],
        ice_adjust_repro_ds.sizes["nflevg"]
    )
    print(f"Domaine: {shape}")
    print(f"Points de grille: {np.prod(shape):,}")

    # Initialize state from dataset
    state = get_state_ice_adjust(domain, backend=backend, dataset=ice_adjust_repro_ds)
    
    # Timestep
    timestep = 50.0  # seconds

    # ========================================================================
    # Fonction d'exécution du composant (pour benchmark)
    # ========================================================================
    def run_ice_adjust():
        """Exécute le composant IceAdjust pour le benchmark."""
        ice_adjust(
            sigqsat=state["sigqsat"],
            exn=state["exn"],
            exnref=state["exnref"],
            rhodref=state["rhodref"],
            pabs=state["pabs"],
            sigs=state["sigs"],
            cf_mf=state["cf_mf"],
            rc_mf=state["rc_mf"],
            ri_mf=state["ri_mf"],
            th=state["th"],
            rv=state["rv"],
            rc=state["rc"],
            rr=state["rr"],
            ri=state["ri"],
            rs=state["rs"],
            rg=state["rg"],
            cldfr=state["cldfr"],
            hlc_hrc=state["hlc_hrc"],
            hlc_hcf=state["hlc_hcf"],
            hli_hri=state["hli_hri"],
            hli_hcf=state["hli_hcf"],
            sigrc=state["sigrc"],
            ths=state["ths"],
            rvs=state["rvs"],
            rcs=state["rcs"],
            ris=state["ris"],
            timestep=timestep,
            domain=shape,
            exec_info={},
            validate_args=False,
        )
    
    # ========================================================================
    # Exécution du benchmark
    # ========================================================================
    print("\nExécution du benchmark...")
    result = benchmark(run_ice_adjust)
    
    # ========================================================================
    # Affichage des statistiques de performance
    # ========================================================================
    print("\n" + "-"*75)
    print("STATISTIQUES DE PERFORMANCE")
    print("-"*75)
    
    # Calculate throughput (grid points per second)
    total_points = np.prod(shape)
    mean_time = result.stats['mean']  # seconds
    throughput = total_points / mean_time
    
    print(f"Temps moyen: {mean_time*1000:.3f} ms")
    print(f"Débit: {throughput/1e6:.2f} M points/s")
    print(f"Performance: {throughput*timestep/1e6:.2f} M point-steps/s")
    
    if hasattr(result.stats, 'stddev'):
        print(f"Écart-type: {result.stats['stddev']*1000:.3f} ms")
    if hasattr(result.stats, 'min'):
        print(f"Temps min: {result.stats['min']*1000:.3f} ms")
    if hasattr(result.stats, 'max'):
        print(f"Temps max: {result.stats['max']*1000:.3f} ms")
    
    print("-"*75)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
