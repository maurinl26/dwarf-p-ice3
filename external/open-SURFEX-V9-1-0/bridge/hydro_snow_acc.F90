!SFX_LIC CeCILL-C
! -----------------------------------------------------------------------------
! Flat OpenACC implementation of HYDRO_SNOW (extracted from SURFEX V9.1)
! -----------------------------------------------------------------------------
! This routine is stripped of all SURFEX derived types (SURF_SNOW, etc.) and
! operates on scalar values representing a single atmospheric column.
! It implements the Douville et al. (1995) 'DEF' or 'D95' snow scheme.
! It is decorated with !$acc routine seq for GPU execution.

MODULE hydro_snow_acc_mod
  USE ISO_C_BINDING, ONLY : C_FLOAT
  IMPLICIT NONE

  ! Physical constants (must match MODD_CSTS & MODD_SNOW_PAR)
  REAL(C_FLOAT), PARAMETER :: XLSTT      = 2.8345E+6
  REAL(C_FLOAT), PARAMETER :: XDAY       = 86400.0
  REAL(C_FLOAT), PARAMETER :: XANS_T     = 10.0      ! days
  REAL(C_FLOAT), PARAMETER :: XANS_TODRY = 0.008     ! 1/days
  REAL(C_FLOAT), PARAMETER :: XANSMIN    = 0.50
  REAL(C_FLOAT), PARAMETER :: XANSMAX    = 0.85
  REAL(C_FLOAT), PARAMETER :: XRHOSMAX   = 300.0     ! kg/m3
  REAL(C_FLOAT), PARAMETER :: XRHOSMIN   = 100.0     ! kg/m3
  REAL(C_FLOAT), PARAMETER :: XWCRN      = 10.0      ! kg/m2
  REAL(C_FLOAT), PARAMETER :: XAGLAMIN   = 0.80
  REAL(C_FLOAT), PARAMETER :: XAGLAMAX   = 0.85
  REAL(C_FLOAT), PARAMETER :: XUNDEF     = 1.0E+20

CONTAINS

  PURE SUBROUTINE HYDRO_SNOW_ACC(                   &
      PTSTEP, OGLACIER, XVEG_SNOW,                  &
      PSR, PLES, PMELT, PPG_MELT,                   &
      WSNOW1, ALB, RHO1                             &
  )
    !$acc routine seq
    
    ! --- INPUTS ---
    REAL(C_FLOAT), INTENT(IN) :: PTSTEP
    REAL(C_FLOAT), INTENT(IN) :: XVEG_SNOW
    REAL(C_FLOAT), INTENT(IN) :: PSR, PLES, PMELT
    
    INTEGER, INTENT(IN) :: OGLACIER  ! 1=True, 0=False
    
    ! --- STATE VARIABLES (Extracted from TPSNOW%) ---
    REAL(C_FLOAT), INTENT(INOUT) :: WSNOW1, ALB, RHO1
    REAL(C_FLOAT), INTENT(INOUT) :: PPG_MELT

    ! --- LOCAL VARIABLES ---
    REAL(C_FLOAT) :: ZSNOWSWEM, ZWSX, ZANSMIN_L, ZANSMAX_L

    ! Initialization
    ZWSX      = 0.0
    ZANSMIN_L = XANSMIN
    ZANSMAX_L = XANSMAX

    ! Fields at time t-dt
    ZSNOWSWEM = WSNOW1

    ! EVOLUTION OF THE EQUIVALENT WATER CONTENT snowSWE ('DEF' option)
    ! evolution of Ws (without melting)
    WSNOW1 = ZSNOWSWEM + PTSTEP * (PSR - PLES/XLSTT - PMELT)

    ! melting of snow: more liquid water reaches the surface
    PPG_MELT = PPG_MELT + PMELT
    
    ! removes very small values due to computation precision
    IF (WSNOW1 < 1.0E-10) WSNOW1 = 0.0

    ! EVOLUTION OF SNOW ALBEDO 
    IF (OGLACIER == 1) THEN
      ZANSMIN_L = XAGLAMIN * XVEG_SNOW + XANSMIN * (1.0 - XVEG_SNOW)
      ZANSMAX_L = XAGLAMAX * XVEG_SNOW + XANSMAX * (1.0 - XVEG_SNOW)
    ELSE
      ZANSMIN_L = XANSMIN
      ZANSMAX_L = XANSMAX
    ENDIF

    IF (WSNOW1 > 0.0) THEN
      IF (ZSNOWSWEM > 0.0) THEN
        ! when there is melting 
        IF (PMELT > 0.0) THEN
          ALB = (ALB - ZANSMIN_L)*EXP(-XANS_T*PTSTEP/XDAY) + ZANSMIN_L &
                + PSR*PTSTEP/XWCRN*(ZANSMAX_L - ZANSMIN_L)
        ! when there is no melting
        ELSE
          ALB = ALB - XANS_TODRY*PTSTEP/XDAY &
                + PSR*PTSTEP/XWCRN*(ZANSMAX_L - ZANSMIN_L)
        ENDIF
      ELSE
        ! new snow covered surface
        ALB = ZANSMAX_L
      ENDIF
      
      ! limits of the albedo
      ALB = MIN(ZANSMAX_L, ALB)
      ALB = MAX(ZANSMIN_L, ALB)
    ENDIF

    ! EVOLUTION OF SNOW DENSITY 
    IF (WSNOW1 > 0.0) THEN
      IF (ZSNOWSWEM > 0.0) THEN
        ZWSX = MAX(WSNOW1, PSR*PTSTEP)
        RHO1 = (RHO1 - XRHOSMAX)*EXP(-XANS_T*PTSTEP/XDAY) + XRHOSMAX
        RHO1 = ( (ZWSX - PSR*PTSTEP) * RHO1 + (PSR*PTSTEP) * XRHOSMIN ) / ZWSX
      ELSE
        RHO1 = XRHOSMIN
      ENDIF
    ENDIF

    ! No SNOW
    IF (WSNOW1 == 0.0) THEN
      RHO1 = XUNDEF
      ALB  = XUNDEF
    ENDIF

  END SUBROUTINE HYDRO_SNOW_ACC

END MODULE hydro_snow_acc_mod
