# -*- coding: utf-8 -*-
"""
Composant ICE_ADJUST modulaire utilisant condensation.py et cloud_fraction.py.

Ce composant reproduit le schéma d'ajustement microphysique ICE_ADJUST de PHYEX
en utilisant les stencils séparés plutôt qu'un stencil monolithique.

Référence:
    PHYEX-IAL_CY50T1/micro/ice_adjust.F90
    PHYEX-IAL_CY50T1/micro/condensation.F90

Architecture:
    1. thermodynamic_fields : Calcul T, Lv, Ls, Cph
    2. condensation         : Schéma CB02, production rc_out, ri_out
    3. cloud_fraction_1     : Sources microphysiques, conservation
    4. cloud_fraction_2     : Fraction nuageuse finale, autoconversion
"""
from __future__ import annotations

import logging
from functools import partial
from itertools import repeat
from typing import Tuple, Dict
from numpy.typing import NDArray
from gt4py.cartesian.gtscript import stencil

from ..phyex_common.phyex import Phyex
from ..utils.env import DTYPES, BACKEND
from ..utils.storage import managed_temporaries

log = logging.getLogger(__name__)


class IceAdjustModular:
    """
    Composant d'ajustement microphysique modulaire.
    
    Implémentation modulaire d'ICE_ADJUST reproduisant la référence PHYEX
    en utilisant les stencils séparés condensation.py et cloud_fraction.py.
    
    Séquence d'exécution:
        1. thermodynamic_fields : T = θ×Π, Lv, Ls, Cph
        2. condensation         : Schéma CB02 → rc_tmp, ri_tmp
        3. cloud_fraction_1     : Calcul sources, conservation eau/énergie
        4. cloud_fraction_2     : Fraction nuageuse, autoconversion
    
    Args:
        phyex: Configuration PHYEX (AROME par défaut)
        dtypes: Types de données (float32/float64)
        backend: Backend GT4Py (numpy, cpu, gpu)
    
    Attributes:
        thermodynamic_fields_stencil: Stencil champs thermodynamiques
        condensation_stencil: Stencil condensation CB02
        cloud_fraction_1_stencil: Stencil sources microphysiques
        cloud_fraction_2_stencil: Stencil fraction nuageuse
    
    Example:
        >>> phyex = Phyex("AROME")
        >>> ice_adjust = IceAdjustModular(phyex)
        >>> ice_adjust(
        ...     sigqsat, exn, exnref, rhodref, pabs, sigs,
        ...     cf_mf, rc_mf, ri_mf, th, rv, rc, rr, ri, rs, rg,
        ...     cldfr, hlc_hrc, hlc_hcf, hli_hri, hli_hcf, sigrc,
        ...     ths, rvs, rcs, ris, timestep, domain, exec_info
        ... )
    """

    def __init__(
        self,
        phyex: Phyex = Phyex("AROME"),
        dtypes: Dict = DTYPES,
        backend: str = BACKEND,
    ) -> None:
        """
        Initialise le composant ICE_ADJUST modulaire.
        
        Compile tous les stencils nécessaires avec la configuration PHYEX.
        """
        self.phyex = phyex
        self.dtypes = dtypes
        self.backend = backend

        # Configuration de la compilation des stencils
        compile_stencil = partial(
            stencil,
            backend=backend,
            externals=phyex.externals,
            dtypes=dtypes,
        )

        # Import et compilation des stencils
        from ..stencils.cloud_fraction import (
            thermodynamic_fields,
            cloud_fraction_1,
            cloud_fraction_2,
        )
        from ..stencils.condensation import condensation

        # Compilation des stencils
        self.thermodynamic_fields_stencil = compile_stencil(
            name="thermodynamic_fields",
            definition=thermodynamic_fields,
        )

        self.condensation_stencil = compile_stencil(
            name="condensation",
            definition=condensation,
        )

        self.cloud_fraction_1_stencil = compile_stencil(
            name="cloud_fraction_1",
            definition=cloud_fraction_1,
        )

        self.cloud_fraction_2_stencil = compile_stencil(
            name="cloud_fraction_2",
            definition=cloud_fraction_2,
        )

        # Log de la configuration
        log.info("="*70)
        log.info("IceAdjustModular - Configuration PHYEX")
        log.info("="*70)
        log.info(f"Backend          : {backend}")
        log.info(f"Precision        : {dtypes['float']}")
        log.info(f"SUBG_COND        : {phyex.nebn.LSUBG_COND}")
        log.info(f"SUBG_MF_PDF      : {phyex.param_icen.SUBG_MF_PDF}")
        log.info(f"SIGMAS           : {phyex.nebn.LSIGMAS}")
        log.info(f"LMFCONV          : {phyex.LMFCONV}")
        log.info(f"CONDENS (scheme) : {phyex.nebn.CONDENS}")
        log.info(f"FRAC_ICE_ADJUST  : {phyex.nebn.FRAC_ICE_ADJUST}")
        log.info("="*70)

    def __call__(
        self,
        sigqsat: NDArray,
        exn: NDArray,
        exnref: NDArray,
        rhodref: NDArray,
        pabs: NDArray,
        sigs: NDArray,
        cf_mf: NDArray,
        rc_mf: NDArray,
        ri_mf: NDArray,
        th: NDArray,
        rv: NDArray,
        rc: NDArray,
        rr: NDArray,
        ri: NDArray,
        rs: NDArray,
        rg: NDArray,
        cldfr: NDArray,
        hlc_hrc: NDArray,
        hlc_hcf: NDArray,
        hli_hri: NDArray,
        hli_hcf: NDArray,
        sigrc: NDArray,
        ths: NDArray,
        rvs: NDArray,
        rcs: NDArray,
        ris: NDArray,
        timestep: float,
        domain: Tuple[int, ...],
        exec_info: Dict,
        validate_args: bool = False,
    ):
        """
        Exécute la séquence complète d'ajustement microphysique ICE_ADJUST.
        
        Séquence d'exécution correspondant à ice_adjust.F90:
        
        1. THERMODYNAMIC_FIELDS (ice_adjust.F90, l.450-473)
           - Calcul température T = θ × Π
           - Chaleurs latentes Lv, Ls (dépendantes de T)
           - Chaleur spécifique Cph (fonction des hydrométéores)
        
        2. CONDENSATION (condensation.F90, l.186-575)
           - Schéma CB02 (Chaboureau & Bechtold 2002)
           - Calcul variabilité sous-maille
           - Production condensats rc_out, ri_out
           - Fraction nuageuse initiale
        
        3. CLOUD_FRACTION_1 (ice_adjust.F90, l.278-312)
           - Calcul sources microphysiques drc/dt, dri/dt
           - Conservation eau totale
           - Ajustement thermique température potentielle
        
        4. CLOUD_FRACTION_2 (ice_adjust.F90, l.313-419)
           - Fraction nuageuse finale avec flux de masse
           - Autoconversion liquide (cloud → rain)
           - Autoconversion solide (ice → snow)
        
        Args:
            sigqsat: Coefficient σ_qsat pour variabilité sous-maille
            exn: Fonction d'Exner (Π = (P/P0)^(R/Cp))
            exnref: Fonction d'Exner de référence
            rhodref: Densité de l'air sec de référence [kg/m³]
            pabs: Pression absolue [Pa]
            sigs: Écart-type sous-maille de s
            cf_mf: Fraction nuageuse des flux de masse
            rc_mf: Contenu en eau liquide des flux de masse [kg/kg]
            ri_mf: Contenu en glace des flux de masse [kg/kg]
            th: Température potentielle [K]
            rv: Rapport mélange vapeur [kg/kg]
            rc: Rapport mélange liquide nuageux [kg/kg]
            rr: Rapport mélange pluie [kg/kg]
            ri: Rapport mélange glace [kg/kg]
            rs: Rapport mélange neige [kg/kg]
            rg: Rapport mélange graupel [kg/kg]
            cldfr: Fraction nuageuse [0-1] (modifié)
            hlc_hrc: Contenu liquide haute résolution [kg/kg] (modifié)
            hlc_hcf: Fraction liquide haute résolution [0-1] (modifié)
            hli_hri: Contenu glace haute résolution [kg/kg] (modifié)
            hli_hcf: Fraction glace haute résolution [0-1] (modifié)
            sigrc: σ_rc pour autoconversion [kg/kg]
            ths: Température potentielle source [K] (modifié)
            rvs: Rapport mélange vapeur source [kg/kg] (modifié)
            rcs: Rapport mélange liquide source [kg/kg] (modifié)
            ris: Rapport mélange glace source [kg/kg] (modifié)
            timestep: Pas de temps [s]
            domain: Taille du domaine (ni, nj, nk)
            exec_info: Informations d'exécution GT4Py
            validate_args: Validation des arguments GT4Py
        
        Returns:
            None (modification in-place des champs de sortie)
        
        Raises:
            ValueError: Si les dimensions des champs ne correspondent pas
        """
        # Création des champs temporaires nécessaires
        with managed_temporaries(
            *repeat((domain, "float"), 8)
        ) as (
            t,          # Température [K]
            lv,         # Chaleur latente vaporisation [J/kg]
            ls,         # Chaleur latente sublimation [J/kg]
            cph,        # Chaleur spécifique air humide [J/kg/K]
            rv_out,     # Rapport mélange vapeur après condensation [kg/kg]
            rc_out,     # Rapport mélange liquide après condensation [kg/kg]
            ri_out,     # Rapport mélange glace après condensation [kg/kg]
            q1,         # Paramètre distribution sous-maille
        ):
            
            log.debug("Étape 1/4: Calcul champs thermodynamiques")
            # ================================================================
            # ÉTAPE 1: CHAMPS THERMODYNAMIQUES
            # Référence: ice_adjust.F90, lignes 450-473
            # ================================================================
            self.thermodynamic_fields_stencil(
                th=th,              # IN
                exn=exn,            # IN
                rv=rv,              # IN
                rc=rc,              # IN
                rr=rr,              # IN
                ri=ri,              # IN
                rs=rs,              # IN
                rg=rg,              # IN
                lv=lv,              # OUT
                ls=ls,              # OUT
                cph=cph,            # OUT
                t=t,                # OUT
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
            )

            log.debug("Étape 2/4: Condensation (schéma CB02)")
            # ================================================================
            # ÉTAPE 2: CONDENSATION
            # Référence: condensation.F90, lignes 186-575
            # Schéma CB02 de Chaboureau & Bechtold (2002)
            # ================================================================
            self.condensation_stencil(
                sigqsat=sigqsat,    # IN
                pabs=pabs,          # IN
                sigs=sigs,          # IN
                t=t,                # IN
                rv=rv,              # IN
                ri=ri,              # IN
                rc=rc,              # IN
                rv_out=rv_out,      # OUT
                rc_out=rc_out,      # OUT
                ri_out=ri_out,      # OUT
                cldfr=cldfr,        # OUT
                cph=cph,            # IN
                lv=lv,              # IN
                ls=ls,              # IN
                q1=q1,              # OUT
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
            )

            log.debug("Étape 3/4: Calcul sources microphysiques")
            # ================================================================
            # ÉTAPE 3: SOURCES MICROPHYSIQUES
            # Référence: ice_adjust.F90, lignes 278-312
            # Conservation de l'eau et ajustement thermique
            # ================================================================
            self.cloud_fraction_1_stencil(
                lv=lv,              # IN
                ls=ls,              # IN
                cph=cph,            # IN
                exnref=exnref,      # IN
                rc=rc,              # IN
                ri=ri,              # IN
                ths=ths,            # INOUT
                rvs=rvs,            # INOUT
                rcs=rcs,            # INOUT
                ris=ris,            # INOUT
                rc_tmp=rc_out,      # IN (condensation output)
                ri_tmp=ri_out,      # IN (condensation output)
                dt=timestep,        # IN
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
            )

            log.debug("Étape 4/4: Fraction nuageuse et autoconversion")
            # ================================================================
            # ÉTAPE 4: FRACTION NUAGEUSE ET AUTOCONVERSION
            # Référence: ice_adjust.F90, lignes 313-419
            # Calcul fraction nuageuse finale, autoconversion
            # ================================================================
            self.cloud_fraction_2_stencil(
                rhodref=rhodref,    # IN
                exnref=exnref,      # IN
                t=t,                # IN
                cph=cph,            # IN
                lv=lv,              # IN
                ls=ls,              # IN
                ths=ths,            # INOUT
                rvs=rvs,            # INOUT
                rcs=rcs,            # INOUT
                ris=ris,            # INOUT
                rc_mf=rc_mf,        # IN
                ri_mf=ri_mf,        # IN
                cf_mf=cf_mf,        # IN
                cldfr=cldfr,        # INOUT
                hlc_hrc=hlc_hrc,    # INOUT
                hlc_hcf=hlc_hcf,    # INOUT
                hli_hri=hli_hri,    # INOUT
                hli_hcf=hli_hcf,    # INOUT
                dt=timestep,        # IN
                origin=(0, 0, 0),
                domain=domain,
                validate_args=validate_args,
                exec_info=exec_info,
            )

        log.debug("ICE_ADJUST modulaire: séquence complète terminée")

    def __repr__(self) -> str:
        """Représentation textuelle du composant."""
        return (
            f"IceAdjustModular(backend={self.backend}, "
            f"dtype={self.dtypes['float']}, "
            f"SUBG_COND={self.phyex.nebn.LSUBG_COND})"
        )
