## Translation options

### ice_adjust

_ice_adjust.F90_ and _condensation.F90_ loops have been reunited in a single stencil_collection under ice_adjust.py

The options below have been retained concerning keys, in order to reproduce AROME's behavior of microphysics :

- OCND2 = .FALSE.
    OCND2 is dedicated to split between ice and liquid water clouds
    the option is not used in Arome
    therefore, we do not implement following fields :
        - PWCLDFR
        - PICLDFR
        - PSSIO
        - PSSIU
        - PIFR
        - PICE_CLD_WGT

    l506 to l514 kept
    l515 to l575 removed in condensation.F90
    l316 to l331 removed

    icecloud routine is not implemented from mode_iceloud.F90

- OSIGMAS = .TRUE. or LSIGMAS=.TRUE.
    l276 to l310 removed in condensation.F90
    ! Preliminary calculations needed for computing the "turbulent part" of Sigma_s - omitted

    l391 to l405 omitted in condensation.F90

- OUSERI = .FALSE.
    l341 to l350 omitted in condensation.F90

- LHGT_QS = .FALSE.
    l373 to l377 omitted in condensation.F90

- LSTATNW = .FALSE.
    l378 to l379 omitted in condensation.F90
    l384 to l385 omitted in condensation.F90

- LSUBG_COND = .TRUE.
    LSUBG_COND = .FALSE. corresponds to the case where sigsqat and sigs are null and don't come from the turbulence scheme

    Because LSIGMAS is true for Arome, the case where lsubg_cond is false is tolerant null fields for sigs and sigqsat at the instanciation of ice_adjust

- HCONDENS = "CB02"
    l413 to l469 omitted in condensation.F90
    corresponding to HCONDENS = 'GAUS'

- HLAMBDA3 = "CB"


- hlc_hcf, hlc_hrc, hli_hcf, hli_hri are assumed to be present: the assumptions _IF PRESENT(HLI_HCF)_ is removed (respectively for the 4 fields)

    fields are initialized at the start of the stencil
    therefore l496 to l504 are removed

- lv, ls, cph are assumed to be present : _IF PRESENT(PLV)_ resp. _PLS_, _PCPH_ are removed

- POUT_RV, POUT_RC, POUT_RI, POUT_TH have been removed :
    l402 to l427 removed

- JITER = 1 is retained since option to run many iterations cannot be passed as a configuration parameter in Fortran sources.
    - ITERATION subroutine have been removed and merged with ice_adjust main stencil
    - JITER evolve between 1 and ITERMAX in _ice_adjust.F90_, and ITERMAX=1 is hard coded in _ice_adjust.F90_
    - Following Langlois (1973), a unique iteration is needed to find zeros with satisfying precision

    - l269 to l270 is skipped, 1 iteration call is kept and merged with the code in ice_adjust stencil

- l175 to l283 in ice_adjust stencil (gt4py) corresponds to condensation.F90 routine

- option with hail (nrr=7) is not implemented
    therefore, PRH is not present


### Rain ice

- OELEC = .False. meaning electric charges are not sedimented
    PQXT, PQXS, PPQXS are omitted in mode_ice4_sedimentation_split.F90

    l465 to l475 removed in mode_ice4_sedimentation_split.F90
    l486 to l489 removed in mode_ice4_sedimentation_split.F90
    l504 to l512 removed in mode_ice4_sedimentation_split.F90
    l517 to l526 removed in mode_ice4_sedimentation_split.F90
    l151 is omitted in mode_ice4_slow.F90

    PLATHAM_IAGGS field is omitted (contribution of electrical field to aggregation)

- OSEDIM_BEARD = .False. meaning electric forces have no effect on sedimentation
    PEFIELDW

- #ifdef REPRO48 set to True
    REPRO48 is a flag to keep consistent results with cy48
    It has been chosen to keep RERPO48 to True because it enforces a more compatible version of RainIce with operational settings

    To date (19-02-2024) the operational cycles runs with REPRO48 set to True

    mode_ice4_fast_rs.F90
    mode_ice4_slow.F90      l118 to l120 // l138 to l142 kept
                            l126 to l26  // l143 to l150 omitted
