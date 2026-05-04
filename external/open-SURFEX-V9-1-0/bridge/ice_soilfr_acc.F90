!SFX_LIC CeCILL-C
! -----------------------------------------------------------------------------
! Flat OpenACC implementation of ICE_SOILFR (extracted from SURFEX V9.1)
! -----------------------------------------------------------------------------
! This routine is stripped of all SURFEX derived types (ISBA_t, etc.) and
! operates on scalar values representing a single atmospheric column.
! It calculates the evolution of the surface and deep-soil temperature due to 
! soil water phase changes in the Force-Restore scheme.
! It is decorated with !$acc routine seq for GPU execution.

MODULE ice_soilfr_acc_mod
  USE ISO_C_BINDING, ONLY : C_FLOAT
  IMPLICIT NONE

  ! Physical constants (must match MODD_CSTS & MODD_ISBA_PAR)
  REAL(C_FLOAT), PARAMETER :: XCL      = 4.1855E+3
  REAL(C_FLOAT), PARAMETER :: XTT      = 273.15
  REAL(C_FLOAT), PARAMETER :: XPI      = 3.14159265358979323846
  REAL(C_FLOAT), PARAMETER :: XDAY     = 86400.0
  REAL(C_FLOAT), PARAMETER :: XCI      = 2.106E+3
  REAL(C_FLOAT), PARAMETER :: XRHOLI   = 917.0
  REAL(C_FLOAT), PARAMETER :: XLMTT    = 3.337E+5
  REAL(C_FLOAT), PARAMETER :: XRHOLW   = 1000.0
  REAL(C_FLOAT), PARAMETER :: XG       = 9.80665
  REAL(C_FLOAT), PARAMETER :: XCONDI   = 2.22
  REAL(C_FLOAT), PARAMETER :: XWGMIN   = 0.001
  REAL(C_FLOAT), PARAMETER :: XSPHSOIL = 7.33E+2
  REAL(C_FLOAT), PARAMETER :: XDRYWGHT = 1400.0
  REAL(C_FLOAT), PARAMETER :: ZEFFIC_MIN = 0.01

CONTAINS

  PURE SUBROUTINE ICE_SOILFR_ACC(               &
      PTSTEP, PKSFC_IVEG,                       &
      XWG1, XWG2, XWGI1, XWGI2, XTG1, XTG2,     &
      XCT, XCG, XWSAT_AVGZ, XDG1, XDG2,         &
      XTAUICE_IN, XMPOTSAT, XBCOEF,             &
      XPSNG, XFFG,                              &
      ISNOW_SCHEME, CSOILFRZ,                   &
      PDWGI1, PDWGI2                            &
  )
    !$acc routine seq
    
    ! --- INPUTS ---
    REAL(C_FLOAT), INTENT(IN) :: PTSTEP
    REAL(C_FLOAT), INTENT(IN) :: PKSFC_IVEG
    
    ! --- STATE VARIABLES (Extracted from PEK%, PK%, DMK%, KK%) ---
    REAL(C_FLOAT), INTENT(INOUT) :: XWG1, XWG2, XWGI1, XWGI2, XTG1, XTG2
    REAL(C_FLOAT), INTENT(IN)    :: XCT, XCG, XWSAT_AVGZ
    REAL(C_FLOAT), INTENT(IN)    :: XDG1, XDG2, XTAUICE_IN
    REAL(C_FLOAT), INTENT(IN)    :: XMPOTSAT, XBCOEF, XPSNG, XFFG
    
    ! Configuration flags
    ! ISNOW_SCHEME: 1='D95', 2='EBA', 3='3-L', 4='CRO'
    ! CSOILFRZ: 1='LWT', 0='DEF'
    INTEGER, INTENT(IN) :: ISNOW_SCHEME, CSOILFRZ
    
    ! --- OUTPUTS ---
    REAL(C_FLOAT), INTENT(OUT) :: PDWGI1, PDWGI2
    
    ! --- LOCAL VARIABLES ---
    REAL(C_FLOAT) :: ZKSOIL
    REAL(C_FLOAT) :: ZKSFC_FRZ, ZFREEZING, ZICE_MELT, ZWIM, ZWIT
    REAL(C_FLOAT) :: ZWGI1_L, ZWGI2_L, ZWM, ZSOILHEATCAP, ZICEEFF
    REAL(C_FLOAT) :: ZEFFIC, ZTAUICE, ZWGMIN, ZTGMAX, ZMATPOT, ZDELTAT
    REAL(C_FLOAT) :: ZPSNG_L, ZWORK1, ZWORK2, ZTDIURN

    ! Initialization
    ZFREEZING = 0.0
    ZKSFC_FRZ = 0.0
    ZEFFIC    = 0.0
    ZICE_MELT = 0.0
    ZWGI1_L   = 0.0
    ZWIM      = 0.0
    ZSOILHEATCAP = 0.0
    ZWIT      = 0.0
    ZWGI2_L   = 0.0
    ZTGMAX    = 0.0
    ZWGMIN    = 0.0
    ZMATPOT   = 0.0
    ZDELTAT   = 0.0
    ZTDIURN   = 0.0

    ! Snow fraction
    IF (ISNOW_SCHEME == 3 .OR. ISNOW_SCHEME == 4) THEN
      ZPSNG_L = 0.0
    ELSE
      ZPSNG_L = XPSNG + XFFG
    ENDIF

    ! Melting/freezing normalized coefficient
    ZKSOIL  = (0.5 * SQRT(XCONDI*XCI*XRHOLI*XDAY/XPI)) / XLMTT
    ZTAUICE = MAX(PTSTEP, XTAUICE_IN)

    ! EFFECT OF THE MELTING/FREEZING ON THE SURFACE-SOIL HEAT AND ICE CONTENTS
    ZKSFC_FRZ = ZKSOIL * PKSFC_IVEG

    ! Water freezing
    IF (CSOILFRZ == 1) THEN
      ! 'LWT' option
      ZMATPOT = MIN(XMPOTSAT, XLMTT*(XTG1 - XTT)/(XG*XTG1))
      ZWGMIN  = XWSAT_AVGZ * ((ZMATPOT/XMPOTSAT)**(-1.0/XBCOEF))

      ZMATPOT = XMPOTSAT * ((XWG1/XWSAT_AVGZ)**(-XBCOEF))
      ZTGMAX  = XLMTT*XTT / (XLMTT - XG*ZMATPOT)
    ELSE
      ZWGMIN  = XWGMIN
      ZTGMAX  = XTT
    ENDIF

    ZDELTAT = XTG1 - ZTGMAX

    ZWORK2 = XRHOLW * XDG1
    ZEFFIC = MAX(ZEFFIC_MIN, (XWG1 - XWGMIN)/XWSAT_AVGZ)
    ZFREEZING = MIN(MAX(0.0, XWG1 - ZWGMIN)*ZWORK2, &
                    ZKSFC_FRZ * ZEFFIC * MAX(-ZDELTAT, 0.0))

    ! Ground Ice melt
    ZEFFIC = MAX(ZEFFIC_MIN, XWGI1/(XWSAT_AVGZ - XWGMIN))
    ZICE_MELT = MIN(XWGI1*ZWORK2, ZKSFC_FRZ * ZEFFIC * MAX(ZDELTAT, 0.0))

    ! Ice reservoir evolution
    ZWGI1_L = XWGI1 + (PTSTEP/ZTAUICE)*(1.0 - ZPSNG_L) * &
              (ZFREEZING - ZICE_MELT)/ZWORK2

    ZWGI1_L = MAX(ZWGI1_L, 0.0)
    ZWGI1_L = MIN(ZWGI1_L, XWSAT_AVGZ - XWGMIN)

    PDWGI1 = ZWGI1_L - XWGI1

    ! Effect on temperature
    XTG1 = XTG1 + PDWGI1 * XLMTT * XCT * ZWORK2

    ! EFFECT OF THE MELTING/FREEZING ON THE DEEP-SOIL HEAT AND ICE CONTENTS
    ZWORK1 = XDG1 / XDG2

    ! Available Deep ice content
    ZWIM = (XWGI2 - ZWORK1 * XWGI1) / (1.0 - ZWORK1)
    ZWIM = MAX(0.0, ZWIM)

    ! Deep liquid water content
    ZWM = (XWG2 - ZWORK1 * XWG1) / (1.0 - ZWORK1)

    ! Water freezing
    ZSOILHEATCAP = XCL*XRHOLW*XWG2 + XCI*XRHOLI*XWGI2 + &
                   XSPHSOIL*XDRYWGHT*(1.0-XWSAT_AVGZ)*(1.0-XWSAT_AVGZ)

    ZTDIURN = MIN(XDG2, 4.0/(ZSOILHEATCAP*XCG))
    ZICEEFF = (XWGI2/(XWGI2 + XWG2)) * XDG2

    IF (CSOILFRZ == 1) THEN
      ZMATPOT = MIN(XMPOTSAT, XLMTT*(XTG2 - XTT)/(XG*XTG2))
      ZWGMIN  = XWSAT_AVGZ * ((ZMATPOT/XMPOTSAT)**(-1.0/XBCOEF))

      ZMATPOT = XMPOTSAT * ((XWG2/XWSAT_AVGZ)**(-XBCOEF))
      ZTGMAX  = XLMTT*XTT / (XLMTT - XG*ZMATPOT)
    ELSE
      ZWGMIN  = XWGMIN
      ZTGMAX  = XTT
    ENDIF

    ZDELTAT = XTG2 - ZTGMAX

    ZWORK2 = XRHOLW * (XDG2 - XDG1)

    ZFREEZING = 0.0
    IF (ZICEEFF <= ZTDIURN) THEN
      ZEFFIC = MAX(ZEFFIC_MIN, MAX(0.0, ZWM - XWGMIN)/XWSAT_AVGZ)
      ZFREEZING = MIN(MAX(0.0, ZWM - ZWGMIN)*ZWORK2, &
                      ZKSOIL * ZEFFIC * MAX(-ZDELTAT, 0.0))
    ENDIF

    ! Ground Ice melt
    ZEFFIC = MAX(ZEFFIC_MIN, ZWIM/(XWSAT_AVGZ - XWGMIN))
    ZICE_MELT = MIN(ZWIM*ZWORK2, ZKSOIL * ZEFFIC * MAX(ZDELTAT, 0.0))

    ! Deep-part of deep-soil Ice reservoir evolution
    ZWIT = ZWIM + (PTSTEP/ZTAUICE)*(1.0 - ZPSNG_L) * &
           ((ZFREEZING - ZICE_MELT) / ZWORK2)

    ZWIT = MAX(ZWIT, 0.0)
    ZWIT = MIN(ZWIT, XWSAT_AVGZ - XWGMIN)

    ! Add reservoir evolution from surface freezing
    ZWGI2_L = (1.0 - ZWORK1)*ZWIT + ZWORK1*ZWGI1_L
    PDWGI2 = ZWGI2_L - XWGI2

    ! Effect on temperature
    XTG2 = XTG2 + PDWGI2 * XLMTT * XCG * XRHOLW * XDG2

  END SUBROUTINE ICE_SOILFR_ACC

END MODULE ice_soilfr_acc_mod
