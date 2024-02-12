## Translation options 

### ice_adjust

_ice_adjust.F90_ and _condensation.F90_ loops have been reunited in a single stencil_collection under ice_adjust.py

The options below have been retained concerning keys, in order to reproduce AROME's beahvior of microphysics :

- OCND2 = .FALSE.
    l506 to l514 kept
    l515 to l575 removed
    l316 to l331 removed

- OSIGMAS = .TRUE.
    l276 to l310 removed

- HCONDENS = "CB02"

- hlc_hcf, hlc_hrc, hli_hcf, hli_hri are assumed to be present: the assumptions _IF PRESENT(HLI_HCF)_ is removed (respectively for the 4 fields)

- lv, ls, cph are assumed to be present : _IF PRESENT(PLV)_ resp. _PLS_, _PCPH_ are removed

- POUT_RV, POUT_RC, POUT_RI, POUT_TH have been removed :
    l402 to l427 removed

- JITER = 1 is retained since option to run many iterations cannot be passed as a configuration parameter in Fortran sources.
    - ITERATION subroutine have been removed and merged with ice_adjust main stencil
    - JITER evolve between 1 and ITERMAX in _ice_adjust.F90_, and ITERMAX=1 is hard coded in _ice_adjust.F90_
    - Following Langlois (1973), a unique iteration is needed to find zeros with satisfying precision



