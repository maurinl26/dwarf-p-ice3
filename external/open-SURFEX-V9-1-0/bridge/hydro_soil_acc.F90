!SFX_LIC CeCILL-C
! -----------------------------------------------------------------------------
! Flat OpenACC implementation of HYDRO_SOIL (extracted from SURFEX V9.1)
! -----------------------------------------------------------------------------
! This routine is stripped of all SURFEX derived types (ISBA_t, etc.) and
! operates on scalar values representing a single atmospheric column.
! It implements the 3-Layer Force-Restore soil moisture scheme (ISBA-3L).
! It is decorated with !$acc routine seq for GPU execution.

MODULE hydro_soil_acc_mod
  USE ISO_C_BINDING, ONLY : C_FLOAT
  IMPLICIT NONE

  ! Physical constants (must match MODD_CSTS & MODD_ISBA_PAR)
  REAL(C_FLOAT), PARAMETER :: XLVTT   = 2.5008E+6
  REAL(C_FLOAT), PARAMETER :: XRHOLW  = 1000.0
  REAL(C_FLOAT), PARAMETER :: XLMTT   = 3.337E+5
  REAL(C_FLOAT), PARAMETER :: XLSTT   = 2.8345E+6
  REAL(C_FLOAT), PARAMETER :: XDAY    = 86400.0
  REAL(C_FLOAT), PARAMETER :: XWGMIN  = 0.001

CONTAINS

  PURE SUBROUTINE HYDRO_SOIL_ACC(                   &
      PTSTEP, PLETR, PLEG, PPG, PEVAPCOR,           &
      PD_G3, PWSAT, PWFC, PDWGI1, PDWGI2, PLEGI,    &
      PWG3, PRUNOFF, PDRAIN, PWWILT,                &
      XWG1, XWG2, XWGI1, XWGI2, XTG1, XTG2,         &
      XC1, XC2, XWGEQ, XCT, XCG, XWDRAIN, XC4B,     &
      XDG1, XDG2, XC3_1, XC3_2, XC4REF,             &
      CISBA, CKSAT                                  &
  )
    !$acc routine seq
    
    ! --- INPUTS ---
    REAL(C_FLOAT), INTENT(IN) :: PTSTEP
    REAL(C_FLOAT), INTENT(IN) :: PLETR, PLEG, PPG, PEVAPCOR
    REAL(C_FLOAT), INTENT(IN) :: PD_G3, PWSAT, PWFC, PWWILT
    REAL(C_FLOAT), INTENT(IN) :: PDWGI1, PDWGI2, PLEGI
    
    ! --- STATE VARIABLES (Extracted from PEK%, PK%, DMK%, KK%) ---
    REAL(C_FLOAT), INTENT(INOUT) :: PWG3, XWG1, XWG2
    REAL(C_FLOAT), INTENT(INOUT) :: XWGI1, XWGI2
    REAL(C_FLOAT), INTENT(INOUT) :: XTG1, XTG2
    REAL(C_FLOAT), INTENT(OUT)   :: PRUNOFF, PDRAIN

    REAL(C_FLOAT), INTENT(IN) :: XC1, XC2, XWGEQ
    REAL(C_FLOAT), INTENT(IN) :: XCT, XCG, XWDRAIN, XC4B
    REAL(C_FLOAT), INTENT(IN) :: XDG1, XDG2, XC3_1, XC3_2, XC4REF
    
    ! Configuration flags (passed as integers for OpenACC)
    ! CISBA: 1='2-L', 2='3-L'
    ! CKSAT: 1='DEF', 2='SGH', 3='EXP'
    INTEGER, INTENT(IN) :: CISBA, CKSAT

    ! --- LOCAL VARIABLES ---
    REAL(C_FLOAT) :: ZWG2M, ZWG3M, ZWGI1M, ZWGI2M
    REAL(C_FLOAT) :: ZETR, ZEG, ZWSAT, ZWFC_L, ZWWILT
    REAL(C_FLOAT) :: ZC4, ZWAVG, ZSINK2, ZFACTOR, ZDRAINCF2, ZDRAINCF3, ZDRAIN2
    REAL(C_FLOAT) :: ZDELTA2, ZDELTA3, ZDELTA22, ZDELTA33, ZWDRAIN2, ZWDRAIN3
    REAL(C_FLOAT) :: ZEXCESSF, ZA2, ZB2, ZC2, ZA3, ZB3, ZC3, ZWDRAIN, ZEXCESSFC
    REAL(C_FLOAT) :: ZWLIM2, ZWLIM3

    ! Initialization
    ZWSAT  = 0.0
    ZWFC_L = 0.0
    ZWWILT = 0.0
    PDRAIN = 0.0
    PRUNOFF= 0.0
    
    ZDRAIN2   = 0.0
    ZDRAINCF2 = 0.0
    ZDRAINCF3 = 0.0
    ZDELTA2   = 0.0
    ZDELTA3   = 0.0
    ZDELTA22  = 0.0
    ZDELTA33  = 0.0
    ZSINK2    = 0.0
    ZWDRAIN   = 0.0
    ZWDRAIN2  = 0.0
    ZWDRAIN3  = 0.0
    ZA2 = 0.0; ZB2 = 0.0; ZC2 = 0.0
    ZA3 = 0.0; ZB3 = 0.0; ZC3 = 0.0
    ZEXCESSF  = 0.0

    ! Fields at time t-dt
    ZWG2M  = XWG2
    ZWG3M  = PWG3
    ZWGI1M = XWGI1
    ZWGI2M = XWGI2

    ! New Wsat
    ZWSAT  = PWSAT - ZWGI2M
    ZWFC_L = PWFC * ZWSAT / PWSAT
    ZWWILT = PWWILT * ZWSAT / PWSAT

    ! evaporation rates
    ZETR = PLETR / XLVTT
    ZEG  = PLEG / XLVTT + PEVAPCOR

    ! EVOLUTION OF THE SUPERFICIAL WATER CONTENT WG1
    XWG1 = (XWG1 - PTSTEP * (XC1*(ZEG - PPG)/XRHOLW - XC2*XWGEQ/XDAY)) &
           / (1.0 + PTSTEP * XC2 / XDAY)

    IF (CKSAT == 2 .OR. CKSAT == 3) THEN
      ZWLIM2 = ZWWILT
      ZWLIM3 = PWWILT
    ELSE
      ZWLIM2 = XWGMIN
      ZWLIM3 = XWGMIN
    ENDIF

    ! EVOLUTION OF THE DEEP WATER CONTENT WG2 and WG3
    IF (CISBA == 1) THEN
      ! 2-L ISBA version
      XWG2 = ZWG2M - PTSTEP*(ZEG + ZETR - PPG) / (XDG2 * XRHOLW)
      
      ZWDRAIN = XWDRAIN * MAX(0.0, MIN(ZWFC_L, XWG2)-ZWLIM2)/(ZWFC_L-ZWLIM2)
      ZDRAIN2 = MAX( MIN(ZWDRAIN, XWG2), XWG2-ZWFC_L ) * XC3_1 / (XDG2*XDAY) * PTSTEP
      
      XWG2   = XWG2 - ZDRAIN2
      PDRAIN = ZDRAIN2 * XDG2 * XRHOLW / PTSTEP
      
    ELSE
      ! 3-L ISBA version
      IF (XDG2 >= PD_G3) THEN
        ! With only 2 active layers
        XWG2 = ZWG2M - PTSTEP*(ZEG + ZETR - PPG) / (XDG2 * XRHOLW)
        ZWDRAIN = XWDRAIN * MAX(0.0, MIN(ZWFC_L, XWG2)-ZWLIM2)/(ZWFC_L-ZWLIM2)
        ZDRAIN2 = MAX( MIN(ZWDRAIN, XWG2), XWG2-ZWFC_L ) * XC3_1 / (XDG2*XDAY) * PTSTEP
        XWG2    = XWG2 - ZDRAIN2
        PWG3    = XWG2
        PDRAIN  = ZDRAIN2 * XDG2 * XRHOLW / PTSTEP
      ELSE
        ! With 3 active layers
        ZWDRAIN2 = XWDRAIN * MAX(0.0, MIN(ZWFC_L, ZWG2M)-ZWLIM2)/(ZWFC_L-ZWLIM2)
        ZWDRAIN3 = XWDRAIN * MAX(0.0, MIN(PWFC, ZWG3M)-ZWLIM3)/(PWFC-ZWLIM3)
        
        ZDELTA2 = 0.0
        IF (ZWG2M - ZWFC_L > ZWDRAIN2) ZDELTA2 = 1.0
        
        ZDELTA3 = 0.0
        IF (ZWG3M - PWFC > ZWDRAIN3) ZDELTA3 = 1.0
        
        ZWAVG   = (((ZWG2M**6)*XDG2 + (ZWG3M**6)*(PD_G3-XDG2))/PD_G3)**(1.0/6.0)
        ZFACTOR = XDG2 / (PD_G3 - XDG2)
        ZC4     = XC4REF * (ZWAVG**XC4B) * (10.0**(-XC4B*XWGI2/(PWSAT-XWGMIN)))
        
        ZSINK2  = -(ZEG + ZETR - PPG) / (XDG2 * XRHOLW)
        
        ZDRAINCF2 = XC3_1 / (XDG2 * XDAY)
        ZDELTA22  = ZDELTA2*ZWFC_L - (1.0-ZDELTA2)*ZWDRAIN2
        ZC2       = 1.0 + PTSTEP*(ZDELTA2*ZDRAINCF2 + (ZC4/XDAY))
        ZB2       = PTSTEP*ZC4 / (XDAY*ZC2)
        ZA2       = (ZWG2M + PTSTEP*(ZSINK2 + ZDRAINCF2*ZDELTA22)) / ZC2
        
        ZDRAINCF3 = XC3_2 / ((PD_G3 - XDG2) * XDAY)
        ZDELTA33  = ZDELTA3*PWFC - (1.0-ZDELTA3)*ZWDRAIN3
        ZC3       = 1.0 + PTSTEP*(ZDELTA3*ZDRAINCF3 + ZFACTOR*(ZC4/XDAY))
        ZB3       = PTSTEP*ZFACTOR*(ZDELTA2*ZDRAINCF2 + (ZC4/XDAY)) / ZC3
        ZA3       = (ZWG3M + PTSTEP*(-ZFACTOR*ZDRAINCF2*ZDELTA22 + ZDRAINCF3*ZDELTA33)) / ZC3
        
        XWG2 = (ZA2 + ZB2*ZA3) / (1.0 - ZB2*ZB3)
        PWG3  = ZA3 + ZB3*XWG2
        
        ZWDRAIN = (XRHOLW*XC3_2/XDAY) * (ZDELTA3*(PWG3-PWFC) + (1.0-ZDELTA3)*ZWDRAIN3)
        PDRAIN  = MAX(0.0, ZWDRAIN)
        PWG3    = PWG3 + (PDRAIN - ZWDRAIN)*PTSTEP/((PD_G3 - XDG2)*XRHOLW)
      ENDIF
    ENDIF

    ! EFFECT OF THE MELTING/FREEZING ON THE SOIL WATER CONTENT
    ! 7.1 Surface water liquid and ice reservoirs
    XWGI1 = ZWGI1M + PDWGI1 - PLEGI*PTSTEP/(XLSTT*XDG1*XRHOLW)
    XWG1  = XWG1 - PDWGI1
    
    ZEXCESSFC = 0.0
    
    ZEXCESSF = MAX(0.0, -XWGI1)
    XWG1  = XWG1 - ZEXCESSF
    XWGI1 = XWGI1 + ZEXCESSF
    ZEXCESSFC = ZEXCESSFC - ZEXCESSF
    
    ZEXCESSF = MIN(0.0, PWSAT - XWGMIN - XWGI1)
    XWG1  = XWG1 - ZEXCESSF
    XWGI1 = XWGI1 + ZEXCESSF
    ZEXCESSFC = ZEXCESSFC - ZEXCESSF
    
    ZEXCESSF = MAX(0.0, XWGMIN - XWG1)
    XWGI1 = XWGI1 - ZEXCESSF
    XWG1  = XWG1 + ZEXCESSF
    ZEXCESSFC = ZEXCESSFC + ZEXCESSF
    
    IF (XWGI1 < 1.0E-10) THEN
      ZEXCESSF = XWGI1
      XWG1  = XWG1 + ZEXCESSF
      XWGI1 = 0.0
      ZEXCESSFC = ZEXCESSFC + ZEXCESSF
    ENDIF
    
    XTG1 = XTG1 - ZEXCESSFC*XLMTT*XCT*XRHOLW*XDG1

    ! 7.2 Deep-soil liquid and ice reservoirs
    XWGI2 = ZWGI2M + PDWGI2 - PLEGI*PTSTEP/(XLSTT*XDG2*XRHOLW)
    XWG2  = XWG2 - PDWGI2
    
    ZEXCESSFC = 0.0
    
    ZEXCESSF = MAX(0.0, -XWGI2)
    XWG2  = XWG2 - ZEXCESSF
    XWGI2 = XWGI2 + ZEXCESSF
    ZEXCESSFC = ZEXCESSFC - ZEXCESSF
    
    ZEXCESSF = MAX(0.0, XWGMIN - XWG2)
    XWGI2 = XWGI2 - ZEXCESSF
    XWG2  = XWG2 + ZEXCESSF
    ZEXCESSFC = ZEXCESSFC + ZEXCESSF
    
    IF (XWGI2 < 1.0E-10 * PTSTEP) THEN
      ZEXCESSF = XWGI2
      XWG2  = XWG2 + ZEXCESSF
      XWGI2 = 0.0
      ZEXCESSFC = ZEXCESSFC + ZEXCESSF
    ENDIF
    
    XTG2 = XTG2 - ZEXCESSFC*XLMTT*XCG*XRHOLW*XDG2

    ! PHYSICAL LIMITS AND RUNOFF
    PRUNOFF = MAX(0.0, XWG2 + XWGI2 - PWSAT) * XDG2 * XRHOLW / PTSTEP
    
    XWG1 = MIN(XWG1, PWSAT - XWGI1)
    XWG1 = MAX(XWG1, XWGMIN)
    
    XWG2 = MIN(XWG2, PWSAT - XWGI2)
    XWG2 = MAX(XWG2, XWGMIN)
    
    IF (CISBA == 2) THEN
      PDRAIN = PDRAIN + MAX(0.0, PWG3 - PWSAT) * (PD_G3 - XDG2) * XRHOLW / PTSTEP
      PWG3   = MIN(PWG3, PWSAT)
      PWG3   = MAX(PWG3, XWGMIN)
    ENDIF

  END SUBROUTINE HYDRO_SOIL_ACC

END MODULE hydro_soil_acc_mod
