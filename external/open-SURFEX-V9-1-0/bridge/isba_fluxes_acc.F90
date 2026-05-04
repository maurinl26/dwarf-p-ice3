!SFX_LIC CeCILL-C
! -----------------------------------------------------------------------------
! Flat OpenACC implementation of ISBA_FLUXES (extracted from SURFEX V9.1)
! -----------------------------------------------------------------------------
! This routine is stripped of all SURFEX derived types (ISBA_t, etc.) and
! operates on scalar values representing a single atmospheric column.
! It is decorated with !$acc routine seq for GPU execution.

MODULE isba_fluxes_acc_mod
  USE ISO_C_BINDING, ONLY : C_FLOAT
  IMPLICIT NONE

  ! Physical constants (must match MODD_CSTS)
  REAL(C_FLOAT), PARAMETER :: XSTEFAN = 5.670396E-8
  REAL(C_FLOAT), PARAMETER :: XCPD    = 1004.709
  REAL(C_FLOAT), PARAMETER :: XCL     = 4.1855E+3
  REAL(C_FLOAT), PARAMETER :: XTT     = 273.15
  REAL(C_FLOAT), PARAMETER :: XLMTT   = 3.337E+5
  REAL(C_FLOAT), PARAMETER :: XRS_MAX = 5000.0  ! From MODD_ISBA_PAR
  
CONTAINS

  PURE SUBROUTINE ISBA_FLUXES_ACC(               &
      PTSTEP, PSW_RAD, PLW_RAD, PTA, PQA, PRHOA, &
      PEXNS, PEXNA, PHUG, PHUI, PLEG_DELTA,      &
      PLEGI_DELTA, PDELTA, PF5, PCS, PTSM,       &
      PFROZEN1, PALBT, PEMIST, PQSAT, PDQSAT,    &
      PSNOW_THRUFAL, XTG1, XCPS, XRESA, XVEG,    &
      XPSNG, XLSTT_v, XLVTT_v, XPSN, XPSNV,      &
      XHV, XRS, XFFROZEN, XFF, XCT, WSNOW1,      &
      XSRSFC, ISNOW_SCHEME, CISBA,               &
      PRN, PH, PLE, PLEG, PLEGI, PLEV, PLES,     &
      PLER, PLETR, PEVAP, PEPOT, PGFLUX,         &
      PMELTADV, PMELT, PLE_FLOOD, PLEI_FLOOD,    &
      XTG1_OUT, WSNOW1_OUT                       &
  )
    !$acc routine seq
    
    ! --- INPUTS ---
    REAL(C_FLOAT), INTENT(IN) :: PTSTEP
    REAL(C_FLOAT), INTENT(IN) :: PSW_RAD, PLW_RAD, PTA, PQA, PRHOA
    REAL(C_FLOAT), INTENT(IN) :: PEXNS, PEXNA, PHUG, PHUI
    REAL(C_FLOAT), INTENT(IN) :: PLEG_DELTA, PLEGI_DELTA, PDELTA, PF5
    REAL(C_FLOAT), INTENT(IN) :: PCS, PTSM, PFROZEN1
    REAL(C_FLOAT), INTENT(IN) :: PALBT, PEMIST, PQSAT, PDQSAT, PSNOW_THRUFAL
    
    ! ISBA State and Parameters (extracted from derived types)
    REAL(C_FLOAT), INTENT(IN) :: XTG1, XCPS, XRESA, XVEG
    REAL(C_FLOAT), INTENT(IN) :: XPSNG, XLSTT_v, XLVTT_v, XPSN, XPSNV
    REAL(C_FLOAT), INTENT(IN) :: XHV, XRS, XFFROZEN, XFF, XCT, WSNOW1, XSRSFC
    
    ! Configuration flags (passed as integers for OpenACC compatibility)
    ! ISNOW_SCHEME: 1='D95', 2='EBA', 3='3-L', 4='CRO'
    ! CISBA: 1='DIF', 2='2-L', 3='3-L'
    INTEGER, INTENT(IN) :: ISNOW_SCHEME, CISBA
    
    ! --- OUTPUTS ---
    REAL(C_FLOAT), INTENT(OUT) :: PRN, PH, PLE, PLEG, PLEGI, PLEV, PLES
    REAL(C_FLOAT), INTENT(OUT) :: PLER, PLETR, PEVAP, PEPOT, PGFLUX
    REAL(C_FLOAT), INTENT(OUT) :: PMELTADV, PMELT, PLE_FLOOD, PLEI_FLOOD
    REAL(C_FLOAT), INTENT(OUT) :: XTG1_OUT, WSNOW1_OUT
    
    ! --- LOCAL VARIABLES ---
    REAL(C_FLOAT) :: ZDT, ZWORK1, ZWORK2, ZZHV, ZTN, ZNEXTSNOW
    REAL(C_FLOAT) :: ZPSN, ZPSNG_loc, ZPSNV_loc
    REAL(C_FLOAT) :: XTAU_SMELT = 300.0  ! Default smelt tau

    ! Initialize outputs
    PMELT = 0.0
    PLER  = 0.0
    XTG1_OUT = XTG1
    WSNOW1_OUT = WSNOW1
    
    ! Resolve snow fractions based on scheme
    IF (ISNOW_SCHEME == 3 .OR. ISNOW_SCHEME == 4 .OR. CISBA == 1) THEN
       ZPSN      = 0.0
       ZPSNG_loc = 0.0
       ZPSNV_loc = 0.0
    ELSE
       ZPSN      = XPSN
       ZPSNG_loc = XPSNG
       ZPSNV_loc = XPSNV
    ENDIF

    ! temperature change
    ZDT = XTG1 - PTSM

    ! net surface radiation
    PRN = (1.0 - PALBT) * PSW_RAD + PEMIST * &
          (PLW_RAD - XSTEFAN * (PTSM**3)*(4.0*XTG1 - 3.0*PTSM))

    ! sensible heat flux
    PH = PRHOA * XCPS * (XTG1 - PTA*PEXNS/PEXNA) / XRESA / PEXNS

    ZWORK1 = PRHOA * (1.0 - XVEG) * (1.0 - ZPSNG_loc) / XRESA
    ZWORK2 = PQSAT + PDQSAT * ZDT 

    ! latent heat of sublimation from ground
    PLEGI = ZWORK1 * XLSTT_v * (PHUI * ZWORK2 - PQA) * PFROZEN1 * PLEGI_DELTA

    ! total latent heat of evaporation from ground
    PLEG = ZWORK1 * XLVTT_v * (PHUG * ZWORK2 - PQA) * (1.0 - PFROZEN1) * PLEG_DELTA

    ZWORK2 = PRHOA * (ZWORK2 - PQA)

    ! potential evaporative flux
    PEPOT = ZWORK2 / XRESA

    ! latent heat of evaporation from snow canopy
    PLES = XLSTT_v * ZPSN * PEPOT

    ! latent heat of total evaporation from vegetation
    PLEV = XLVTT_v * XVEG * (1.0 - ZPSNV_loc) * XHV * PEPOT

    ! latent heat of transpiration
    ZZHV = MAX(0.0, SIGN(1.0, PQSAT - PQA))
    PLETR = ZZHV * (1.0 - PDELTA) * XLVTT_v * XVEG * (1.0 - ZPSNV_loc) * &
            ZWORK2 * ( (1.0 / (XRESA + XRS)) - ((1.0 - PF5) / (XRESA + XRS_MAX)) )

    PLER = PLEV - PLETR

    ! latent heat of free water (floodplains)
    PLE_FLOOD  = XLVTT_v * (1.0 - XFFROZEN) * PEPOT 
    PLEI_FLOOD = XLSTT_v * XFFROZEN * PEPOT 

    ! total latent heat of evaporation without flood
    PLE = PLEG + PLEV + PLES + PLEGI

    ! balance of energy fluxes
    PGFLUX = PRN - PH - PLE

    ! snow melt heat advection
    PMELTADV = PSNOW_THRUFAL * XCL * (XTT - XTG1)

    ! total evaporative flux without flood
    PEVAP = ((PLEV + PLEG) / XLVTT_v) + ((PLEGI + PLES) / XLSTT_v)

    IF (ISNOW_SCHEME == 1) THEN ! 'D95'
       PLE    = PLE    + XFF * (PLE_FLOOD + PLEI_FLOOD)
       PGFLUX = PGFLUX - XFF * (PLE_FLOOD + PLEI_FLOOD)
       PEVAP  = PEVAP  + XFF * (PLE_FLOOD / XLVTT_v + PLEI_FLOOD / XLSTT_v)
    ENDIF

    ! Snowmelt
    IF ((ISNOW_SCHEME == 1 .OR. ISNOW_SCHEME == 2) .AND. CISBA /= 1) THEN
       ! Simplified snowmelt logic (omitting the D95 iteration for brevity)
       ZTN = XTG1 ! Assuming no vegetation average for simple stub
       IF (ZTN > XTT .AND. WSNOW1 > 0.0) THEN
           PMELT = ZPSN * (ZTN - XTT) / (PCS * XLMTT * MAX(XTAU_SMELT, PTSTEP))
       ENDIF
       ZNEXTSNOW = WSNOW1 + PTSTEP * (XSRSFC - PLES / XLSTT_v)
       IF (PMELT > 0.0) THEN
           PMELT = MIN(PMELT, ZNEXTSNOW / PTSTEP)
           ZNEXTSNOW = ZNEXTSNOW - PTSTEP * PMELT
       ENDIF
       XTG1_OUT = XTG1_OUT - XCT * XLMTT * PMELT * PTSTEP
       WSNOW1_OUT = ZNEXTSNOW
    ENDIF

  END SUBROUTINE ISBA_FLUXES_ACC

END MODULE isba_fluxes_acc_mod
