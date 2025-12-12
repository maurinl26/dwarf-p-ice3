!!      ######spl
       MODULE MODE_ICE4_FAST_RS_KERNELS
!!      ######spl
!!
!!    PURPOSE
!!    -------
!!      Individual kernels for fast rs (snow/aggregate) processes (split for testing)
!!
!!**  METHOD
!!    ------
!!      Each subroutine implements one microphysical process from ice4_fast_rs
!!      to enable individual testing against Python/CuPy implementations
!!
!!    REFERENCE
!!    ---------
!!      PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_rs.F90
!!
!
IMPLICIT NONE
!
CONTAINS
!
!-------------------------------------------------------------------------------
!
!*       1. MAXIMUM FREEZING RATE COMPUTATION
!
SUBROUTINE ICE4_COMPUTE_FREEZING_RATE(KPROMA, KSIZE, &
                       &LDSOFT, LEVLIMIT, LDCOMPUTE, &
                       &S_RTMIN, XEPSILO, XALPI, XBETAI, XGAMI, &
                       &XTT, XLVTT, XCPV, XCL, XCI, XLMTT, XESTT, XRV, &
                       &X0DEPS, X1DEPS, XEX0DEPS, XEX1DEPS, &
                       &PRHODREF, PPRES, PDV, PKA, PCJ, PLBDAS, PT, PRVT, PRST, &
                       &PRIAGGS, &
                       &PFREEZ_RATE, PFREEZ1_TEND, PFREEZ2_TEND)
!!
!!**  PURPOSE
!!    -------
!!      Compute maximum freezing rate for snow processes
!!
!*       0.1   Declarations of dummy arguments :
!
INTEGER,                      INTENT(IN)    :: KPROMA, KSIZE
LOGICAL,                      INTENT(IN)    :: LDSOFT, LEVLIMIT
LOGICAL, DIMENSION(KPROMA),   INTENT(IN)    :: LDCOMPUTE
REAL,                         INTENT(IN)    :: S_RTMIN, XEPSILO, XALPI, XBETAI, XGAMI
REAL,                         INTENT(IN)    :: XTT, XLVTT, XCPV, XCL, XCI, XLMTT, XESTT, XRV
REAL,                         INTENT(IN)    :: X0DEPS, X1DEPS, XEX0DEPS, XEX1DEPS

REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRHODREF, PPRES, PDV, PKA, PCJ
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PLBDAS, PT, PRVT, PRST, PRIAGGS
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PFREEZ_RATE, PFREEZ1_TEND, PFREEZ2_TEND
!
!*       0.2  Declaration of local variables
!
INTEGER :: JL
!
DO JL = 1, KSIZE
  IF (PRST(JL) > S_RTMIN .AND. LDCOMPUTE(JL)) THEN
    IF (.NOT. LDSOFT) THEN
      PFREEZ1_TEND(JL) = PRVT(JL)*PPRES(JL)/(XEPSILO + PRVT(JL))
      IF (LEVLIMIT) THEN
        PFREEZ1_TEND(JL) = MIN(PFREEZ1_TEND(JL), EXP(XALPI - XBETAI/PT(JL) - XGAMI*ALOG(PT(JL))))
      END IF
      PFREEZ1_TEND(JL) = PKA(JL)*(XTT - PT(JL)) + &
                         (PDV(JL)*(XLVTT + (XCPV - XCL)*(PT(JL) - XTT)) &
                         *(XESTT - PFREEZ1_TEND(JL))/(XRV*PT(JL)))
      PFREEZ1_TEND(JL) = PFREEZ1_TEND(JL)*(X0DEPS*PLBDAS(JL)**XEX0DEPS + &
                                           X1DEPS*PCJ(JL)*PLBDAS(JL)**XEX1DEPS)/ &
                         (PRHODREF(JL)*(XLMTT - XCL*(XTT - PT(JL))))
      PFREEZ2_TEND(JL) = (PRHODREF(JL)*(XLMTT + (XCI - XCL)*(XTT - PT(JL))))/ &
                         (PRHODREF(JL)*(XLMTT - XCL*(XTT - PT(JL))))
    END IF
    PFREEZ_RATE(JL) = MAX(0., MAX(0., PFREEZ1_TEND(JL) + &
                          PFREEZ2_TEND(JL)*PRIAGGS(JL)) - PRIAGGS(JL))
  ELSE
    PFREEZ1_TEND(JL) = 0.
    PFREEZ2_TEND(JL) = 0.
    PFREEZ_RATE(JL) = 0.
  END IF
END DO
!
END SUBROUTINE ICE4_COMPUTE_FREEZING_RATE
!
!-------------------------------------------------------------------------------
!
!*       2. CONVERSION-MELTING OF AGGREGATES
!
SUBROUTINE ICE4_CONVERSION_MELTING_SNOW(KPROMA, KSIZE, &
                       &LDSOFT, LEVLIMIT, LDCOMPUTE, &
                       &S_RTMIN, XEPSILO, XALPW, XBETAW, XGAMW, &
                       &XTT, XLVTT, XCPV, XCL, XLMTT, XESTT, XRV, &
                       &X0DEPS, X1DEPS, XEX0DEPS, XEX1DEPS, XFSCVMG, &
                       &PRHODREF, PPRES, PDV, PKA, PCJ, PLBDAS, PT, PRVT, PRST, &
                       &PRCRIMS_TEND, PRRACCS_TEND, &
                       &PRSMLTG, PRCMLTSR)
!!
!!**  PURPOSE
!!    -------
!!      Compute conversion-melting of aggregates
!!
!*       0.1   Declarations of dummy arguments :
!
INTEGER,                      INTENT(IN)    :: KPROMA, KSIZE
LOGICAL,                      INTENT(IN)    :: LDSOFT, LEVLIMIT
LOGICAL, DIMENSION(KPROMA),   INTENT(IN)    :: LDCOMPUTE
REAL,                         INTENT(IN)    :: S_RTMIN, XEPSILO
REAL,                         INTENT(IN)    :: XALPW, XBETAW, XGAMW
REAL,                         INTENT(IN)    :: XTT, XLVTT, XCPV, XCL, XLMTT, XESTT, XRV
REAL,                         INTENT(IN)    :: X0DEPS, X1DEPS, XEX0DEPS, XEX1DEPS, XFSCVMG

REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRHODREF, PPRES, PDV, PKA, PCJ
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PLBDAS, PT, PRVT, PRST
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRCRIMS_TEND, PRRACCS_TEND
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRSMLTG, PRCMLTSR
!
!*       0.2  Declaration of local variables
!
INTEGER :: JL
!
DO JL = 1, KSIZE
  IF (PRST(JL) > S_RTMIN .AND. PT(JL) > XTT .AND. LDCOMPUTE(JL)) THEN
    IF (.NOT. LDSOFT) THEN
      PRSMLTG(JL) = PRVT(JL)*PPRES(JL)/(XEPSILO + PRVT(JL))
      IF (LEVLIMIT) THEN
        PRSMLTG(JL) = MIN(PRSMLTG(JL), EXP(XALPW - XBETAW/PT(JL) - XGAMW*ALOG(PT(JL))))
      END IF
      PRSMLTG(JL) = PKA(JL)*(XTT - PT(JL)) + &
                    (PDV(JL)*(XLVTT + (XCPV - XCL)*(PT(JL) - XTT)) &
                    *(XESTT - PRSMLTG(JL))/(XRV*PT(JL)))
      PRSMLTG(JL) = XFSCVMG*MAX(0., (-PRSMLTG(JL)* &
                                     (X0DEPS*PLBDAS(JL)**XEX0DEPS + &
                                      X1DEPS*PCJ(JL)*PLBDAS(JL)**XEX1DEPS) &
                                     - (PRCRIMS_TEND(JL) + PRRACCS_TEND(JL))* &
                                     (PRHODREF(JL)*XCL*(XTT - PT(JL))) &
                                     )/(PRHODREF(JL)*XLMTT))
      PRCMLTSR(JL) = PRCRIMS_TEND(JL)
    END IF
  ELSE
    PRSMLTG(JL) = 0.
    PRCMLTSR(JL) = 0.
  END IF
END DO
!
END SUBROUTINE ICE4_CONVERSION_MELTING_SNOW
!
!-------------------------------------------------------------------------------
!
END MODULE MODE_ICE4_FAST_RS_KERNELS
