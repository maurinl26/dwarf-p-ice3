!!      ######spl
       MODULE MODE_ICE4_FAST_RG_KERNELS
!!      ######spl
!!
!!    PURPOSE
!!    -------
!!      Individual kernels for fast rg processes (split for testing)
!!
!!**  METHOD
!!    ------
!!      Each subroutine implements one microphysical process from ice4_fast_rg
!!      to enable individual testing against Python/CuPy implementations
!!
!!    REFERENCE
!!    ---------
!!      PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_rg.F90
!!
!
IMPLICIT NONE
!
CONTAINS
!
!-------------------------------------------------------------------------------
!
!*       1. RAIN CONTACT FREEZING
!
SUBROUTINE ICE4_RAIN_CONTACT_FREEZING(KPROMA, KSIZE, &
                       &LDSOFT, LCRFLIMIT, LDCOMPUTE, &
                       &I_RTMIN, R_RTMIN, &
                       &XICFRR, XEXICFRR, XCEXVT, XRCFRI, XEXRCFRI, &
                       &XTT, XCI, XCL, XLVTT, &
                       &PRHODREF, PLBDAR, PT, PRIT, PRRT, PCIT, &
                       &PRICFRRG, PRRCFRIG, PRICFRR)
!!
!!**  PURPOSE
!!    -------
!!      Compute rain contact freezing
!!
!*       0.1   Declarations of dummy arguments :
!
INTEGER,                      INTENT(IN)    :: KPROMA, KSIZE
LOGICAL,                      INTENT(IN)    :: LDSOFT, LCRFLIMIT
LOGICAL, DIMENSION(KPROMA),   INTENT(IN)    :: LDCOMPUTE
REAL,                         INTENT(IN)    :: I_RTMIN, R_RTMIN
REAL,                         INTENT(IN)    :: XICFRR, XEXICFRR, XCEXVT, XRCFRI, XEXRCFRI
REAL,                         INTENT(IN)    :: XTT, XCI, XCL, XLVTT

REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRHODREF, PLBDAR, PT, PRIT, PRRT, PCIT
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRICFRRG, PRRCFRIG, PRICFRR
!
!*       0.2  Declaration of local variables
!
INTEGER :: JL
REAL :: ZZW0D
!
DO JL = 1, KSIZE
  IF (PRIT(JL) > I_RTMIN .AND. PRRT(JL) > R_RTMIN .AND. LDCOMPUTE(JL)) THEN
    IF (.NOT. LDSOFT) THEN
      PRICFRRG(JL) = XICFRR*PRIT(JL) & ! RICFRRG
                     *PLBDAR(JL)**XEXICFRR &
                     *PRHODREF(JL)**(-XCEXVT)
      PRRCFRIG(JL) = XRCFRI*PCIT(JL) & ! RRCFRIG
                     *PLBDAR(JL)**XEXRCFRI &
                     *PRHODREF(JL)**(-XCEXVT - 1.)
      IF (LCRFLIMIT) THEN
        ZZW0D = MAX(0., MIN(1., (PRICFRRG(JL)*XCI + PRRCFRIG(JL)*XCL)*(XTT - PT(JL))/ &
                            MAX(1.E-20, XLVTT*PRRCFRIG(JL))))
        PRRCFRIG(JL) = ZZW0D*PRRCFRIG(JL)
        PRICFRR(JL) = (1.-ZZW0D)*PRICFRRG(JL)
        PRICFRRG(JL) = ZZW0D*PRICFRRG(JL)
      ELSE
        PRICFRR(JL) = 0.
      END IF
    END IF
  ELSE
    PRICFRRG(JL) = 0.
    PRRCFRIG(JL) = 0.
    PRICFRR(JL) = 0.
  END IF
END DO
!
END SUBROUTINE ICE4_RAIN_CONTACT_FREEZING
!
!-------------------------------------------------------------------------------
!
!*       2. CLOUD AND PRISTINE ICE COLLECTION ON GRAUPEL
!
SUBROUTINE ICE4_CLOUD_PRISTINE_COLLECTION(KPROMA, KSIZE, &
                       &LDSOFT, LDCOMPUTE, &
                       &C_RTMIN, I_RTMIN, G_RTMIN, &
                       &XTT, XFCDRYG, XFIDRYG, XCOLIG, XCOLEXIG, &
                       &XCXG, XDG, XCEXVT, &
                       &PRHODREF, PLBDAG, PT, PRCT, PRIT, PRGT, &
                       &PRCDRYG_TEND, PRIDRYG_TEND, PRIWETG_TEND)
!!
!!**  PURPOSE
!!    -------
!!      Compute wet and dry collection of cloud and pristine ice on graupel
!!
!*       0.1   Declarations of dummy arguments :
!
INTEGER,                      INTENT(IN)    :: KPROMA, KSIZE
LOGICAL,                      INTENT(IN)    :: LDSOFT
LOGICAL, DIMENSION(KPROMA),   INTENT(IN)    :: LDCOMPUTE
REAL,                         INTENT(IN)    :: C_RTMIN, I_RTMIN, G_RTMIN
REAL,                         INTENT(IN)    :: XTT, XFCDRYG, XFIDRYG, XCOLIG, XCOLEXIG
REAL,                         INTENT(IN)    :: XCXG, XDG, XCEXVT

REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRHODREF, PLBDAG, PT, PRCT, PRIT, PRGT
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRCDRYG_TEND, PRIDRYG_TEND, PRIWETG_TEND
!
!*       0.2  Declaration of local variables
!
INTEGER :: JL
REAL :: ZBASE_TEND
!
DO JL = 1, KSIZE
  ! Cloud droplet collection
  IF (PRGT(JL) > G_RTMIN .AND. PRCT(JL) > C_RTMIN .AND. LDCOMPUTE(JL)) THEN
    IF (.NOT. LDSOFT) THEN
      PRCDRYG_TEND(JL) = PLBDAG(JL)**(XCXG - XDG - 2.)*PRHODREF(JL)**(-XCEXVT)
      PRCDRYG_TEND(JL) = XFCDRYG*PRCT(JL)*PRCDRYG_TEND(JL)
    END IF
  ELSE
    PRCDRYG_TEND(JL) = 0.
  END IF

  ! Pristine ice collection
  IF (PRGT(JL) > G_RTMIN .AND. PRIT(JL) > I_RTMIN .AND. LDCOMPUTE(JL)) THEN
    IF (.NOT. LDSOFT) THEN
      ZBASE_TEND = PLBDAG(JL)**(XCXG - XDG - 2.)*PRHODREF(JL)**(-XCEXVT)
      PRIDRYG_TEND(JL) = XFIDRYG*EXP(XCOLEXIG*(PT(JL) - XTT))*PRIT(JL)*ZBASE_TEND
      PRIWETG_TEND(JL) = PRIDRYG_TEND(JL)/(XCOLIG*EXP(XCOLEXIG*(PT(JL) - XTT)))
    END IF
  ELSE
    PRIDRYG_TEND(JL) = 0.
    PRIWETG_TEND(JL) = 0.
  END IF
END DO
!
END SUBROUTINE ICE4_CLOUD_PRISTINE_COLLECTION
!
!-------------------------------------------------------------------------------
!
!*       3. GRAUPEL MELTING
!
SUBROUTINE ICE4_GRAUPEL_MELTING(KPROMA, KSIZE, &
                       &LDSOFT, LEVLIMIT, LDCOMPUTE, &
                       &G_RTMIN, XTT, XEPSILO, XALPW, XBETAW, XGAMW, &
                       &XLVTT, XCPV, XCL, XESTT, XRV, XLMTT, &
                       &X0DEPG, X1DEPG, XEX0DEPG, XEX1DEPG, &
                       &PRHODREF, PPRES, PDV, PKA, PCJ, PLBDAG, PT, PRVT, PRGT, &
                       &PRCDRYG_TEND, PRRDRYG_TEND, &
                       &PRGMLTR)
!!
!!**  PURPOSE
!!    -------
!!      Compute melting of graupel
!!
!*       0.1   Declarations of dummy arguments :
!
INTEGER,                      INTENT(IN)    :: KPROMA, KSIZE
LOGICAL,                      INTENT(IN)    :: LDSOFT, LEVLIMIT
LOGICAL, DIMENSION(KPROMA),   INTENT(IN)    :: LDCOMPUTE
REAL,                         INTENT(IN)    :: G_RTMIN, XTT, XEPSILO
REAL,                         INTENT(IN)    :: XALPW, XBETAW, XGAMW
REAL,                         INTENT(IN)    :: XLVTT, XCPV, XCL, XESTT, XRV, XLMTT
REAL,                         INTENT(IN)    :: X0DEPG, X1DEPG, XEX0DEPG, XEX1DEPG

REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRHODREF, PPRES, PDV, PKA, PCJ
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PLBDAG, PT, PRVT, PRGT
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRCDRYG_TEND, PRRDRYG_TEND
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRGMLTR
!
!*       0.2  Declaration of local variables
!
INTEGER :: JL
!
DO JL = 1, KSIZE
  IF (PRGT(JL) > G_RTMIN .AND. PT(JL) > XTT .AND. LDCOMPUTE(JL)) THEN
    IF (.NOT. LDSOFT) THEN
      PRGMLTR(JL) = PRVT(JL)*PPRES(JL)/(XEPSILO + PRVT(JL))
      IF (LEVLIMIT) THEN
        PRGMLTR(JL) = MIN(PRGMLTR(JL), EXP(XALPW - XBETAW/PT(JL) - XGAMW*ALOG(PT(JL))))
      END IF
      PRGMLTR(JL) = PKA(JL)*(XTT - PT(JL)) + &
                    PDV(JL)*(XLVTT + (XCPV - XCL)*(PT(JL) - XTT)) &
                    *(XESTT - PRGMLTR(JL))/(XRV*PT(JL))
      PRGMLTR(JL) = MAX(0., (-PRGMLTR(JL)* &
                             (X0DEPG*PLBDAG(JL)**XEX0DEPG + &
                              X1DEPG*PCJ(JL)*PLBDAG(JL)**XEX1DEPG) - &
                             (PRCDRYG_TEND(JL) + PRRDRYG_TEND(JL))* &
                             (PRHODREF(JL)*XCL*(XTT - PT(JL))))/ &
                        (PRHODREF(JL)*XLMTT))
    END IF
  ELSE
    PRGMLTR(JL) = 0.
  END IF
END DO
!
END SUBROUTINE ICE4_GRAUPEL_MELTING
!
!-------------------------------------------------------------------------------
!
END MODULE MODE_ICE4_FAST_RG_KERNELS
