!!      ######spl
       MODULE MODE_ICE4_FAST_RG
!!      ######spl
!!
!!    PURPOSE
!!    -------
!!      Computes the fast rg process
!!
!!
!!**  METHOD
!!    ------
!!      The fast growth processes of graupel are treated in this routine.
!!      Processes include:
!!        - Rain contact freezing (RICFRRG, RRCFRIG)
!!        - Wet and dry collection of cloud droplets and pristine ice on graupel
!!        - Collection of snow on graupel
!!        - Collection of rain on graupel
!!        - Graupel melting
!!
!!    REFERENCE
!!    ---------
!!
!!      PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_rg.F90
!!
!!    AUTHOR
!!    ------
!!      S. Riette from the splitting of rain_ice source code (nov. 2014)
!!
!!    MODIFICATIONS
!!    -------------
!!
!
!*       0. DECLARATIONS
!
USE PARKIND1, ONLY : JPRB
!
IMPLICIT NONE
!
CONTAINS
!
!-------------------------------------------------------------------------------
!
SUBROUTINE ICE4_FAST_RG(KPROMA, KSIZE, KRR, &
                       &LDSOFT, LCRFLIMIT, LEVLIMIT, LNULLWETG, LWETGPOST, LPACK_INTERP, LDCOMPUTE, &
                       &NDRYLBDAG, NDRYLBDAS, NDRYLBDAR, &
                       &C_RTMIN, I_RTMIN, R_RTMIN, G_RTMIN, S_RTMIN, &
                       &XALPW, XBETAW, XGAMW, XEXRCFRI, XDG, XEPSILO, &
                       &XICFRR, XEXICFRR, XCEXVT, XRCFRI, XTT, XCI, XCL, XLVTT, &
                       &XCPV, XESTT, X0DEPG, X1DEPG, XRV, XLMTT, &
                       &XCXG, XFCDRYG, XFIDRYG, XCOLIG, XCOLEXIG, &
                       &XLBSDRYG1, XLBSDRYG2, XLBSDRYG3, XCOLEXSG, &
                       &XFSDRYG, XCOLSG, XCXS, XBS, &
                       &XFRDRYG, XLBRDRYG1, XLBRDRYG2, XLBRDRYG3, &
                       &XEX0DEPG, XEX1DEPG, XALPI, XBETAI, XGAMI, &
                       &XDRYINTP1G, XDRYINTP2G, XDRYINTP1S, XDRYINTP2S, &
                       &XDRYINTP1R, XDRYINTP2R, &
                       &XKER_SDRYG, XKER_RDRYG, &
                       &PRHODREF, PPRES, &
                       &PDV, PKA, PCJ, PCIT, &
                       &PLBDAR, PLBDAS, PLBDAG, &
                       &PT, PRVT, PRCT, PRRT, PRIT, PRST, PRGT, &
                       &PRGSI, PRGSI_MR, &
                       &LDWETG, &
                       &PRICFRRG, PRRCFRIG, PRICFRR, &
                       &PRCWETG, PRIWETG, PRRWETG, PRSWETG, &
                       &PRCDRYG, PRIDRYG, PRRDRYG, PRSDRYG, &
                       &PRWETGH, PRWETGH_MR, PRGMLTR, &
                       &PRG_TEND)
!!
!!**  PURPOSE
!!    -------
!!      Computes the fast graupel processes
!!
!!    AUTHOR
!!    ------
!!      S. Riette from code in rain_ice
!!
!!    MODIFICATIONS
!!    -------------
!!      Original 01/2016
!!
!-------------------------------------------------------------------------------
!
!*       0.1   Declarations of dummy arguments :
!
INTEGER,                      INTENT(IN)    :: KPROMA, KSIZE
LOGICAL,                      INTENT(IN)    :: LDSOFT
LOGICAL,                      INTENT(IN)    :: LCRFLIMIT
LOGICAL,                      INTENT(IN)    :: LEVLIMIT, LNULLWETG, LWETGPOST
LOGICAL, DIMENSION(KPROMA),   INTENT(IN)    :: LDCOMPUTE
INTEGER,                      INTENT(IN)    :: KRR      ! Number of moist variable
REAL,                         INTENT(IN)    :: C_RTMIN, I_RTMIN, R_RTMIN, G_RTMIN, S_RTMIN
REAL,                         INTENT(IN)    :: XFIDRYG, XCOLIG, XCOLEXIG, XEPSILO
REAL,                         INTENT(IN)    :: XALPW, XBETAW, XGAMW
REAL,                         INTENT(IN)    :: XICFRR, XEXICFRR, XCEXVT, XRCFRI, XTT, XCI, XCL
REAL,                         INTENT(IN)    :: XEXRCFRI, XLVTT, XDG, XCXG, XFCDRYG
REAL,                         INTENT(IN)    :: XFSDRYG, XCOLSG, XCXS, XBS
REAL,                         INTENT(IN)    :: XLBSDRYG1, XLBSDRYG2, XLBSDRYG3, XCOLEXSG
REAL,                         INTENT(IN)    :: XCPV, XESTT, X0DEPG, X1DEPG, XRV, XLMTT
REAL,                         INTENT(IN)    :: XFRDRYG, XLBRDRYG1, XLBRDRYG2, XLBRDRYG3
REAL,                         INTENT(IN)    :: XEX0DEPG, XEX1DEPG, XALPI, XBETAI, XGAMI

INTEGER,                      INTENT(IN)    :: NDRYLBDAG, NDRYLBDAS, NDRYLBDAR
REAL,                         INTENT(IN)    :: XDRYINTP1G, XDRYINTP2G, XDRYINTP1S, XDRYINTP2S
REAL,                         INTENT(IN)    :: XDRYINTP1R, XDRYINTP2R
LOGICAL,                      INTENT(IN)    :: LPACK_INTERP

REAL, DIMENSION(:,:),         INTENT(IN)    :: XKER_SDRYG, XKER_RDRYG

REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRHODREF ! Reference density
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PPRES    ! Absolute pressure at t
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PDV      ! Diffusivity of water vapor in the air
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PKA      ! Thermal conductivity of the air
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PCJ      ! Function to compute the ventilation coefficient
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PCIT     ! Pristine ice conc. at t
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PLBDAR   ! Slope parameter of the raindrop  distribution
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PLBDAS   ! Slope parameter of the aggregate distribution
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PLBDAG   ! Slope parameter of the graupel   distribution
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PT       ! Temperature
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRVT     ! Water vapor m.r. at t
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRCT     ! Cloud water m.r. at t
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRRT     ! Rain water m.r. at t
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRIT     ! Pristine ice m.r. at t
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRST     ! Snow/aggregate m.r. at t
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRGT     ! Graupel m.r. at t
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRGSI    ! Graupel tendency by other processes
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRGSI_MR ! Graupel mr change by other processes
LOGICAL, DIMENSION(KPROMA),   INTENT(OUT)   :: LDWETG   ! .TRUE. where graupel grows in wet mode
REAL, DIMENSION(KPROMA),      INTENT(INOUT) :: PRICFRRG ! Rain contact freezing
REAL, DIMENSION(KPROMA),      INTENT(INOUT) :: PRRCFRIG ! Rain contact freezing
REAL, DIMENSION(KPROMA),      INTENT(INOUT) :: PRICFRR  ! Rain contact freezing
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRCWETG  ! Graupel wet growth
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRIWETG  ! Graupel wet growth
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRRWETG  ! Graupel wet growth
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRSWETG  ! Graupel wet growth
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRCDRYG  ! Graupel dry growth
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRIDRYG  ! Graupel dry growth
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRRDRYG  ! Graupel dry growth
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRSDRYG  ! Graupel dry growth
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRWETGH  ! Conversion of graupel into hail
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRWETGH_MR ! Conversion of graupel into hail, mr change
REAL, DIMENSION(KPROMA),      INTENT(INOUT) :: PRGMLTR  ! Melting of the graupel
REAL, DIMENSION(KPROMA, 8),   INTENT(INOUT) :: PRG_TEND ! Individual tendencies
!
!*       0.2  Declaration of local variables
!
INTEGER, PARAMETER :: IRCDRYG = 1, IRIDRYG = 2, IRIWETG = 3, IRSDRYG = 4, IRSWETG = 5, IRRDRYG = 6, &
 & IFREEZ1 = 7, IFREEZ2 = 8
LOGICAL, DIMENSION(KPROMA) :: GDRY, LLDRYG
INTEGER :: IGDRY
REAL, DIMENSION(KPROMA) :: ZBUF1, ZBUF2, ZBUF3
INTEGER, DIMENSION(KPROMA) :: IBUF1, IBUF2, IBUF3
REAL, DIMENSION(KPROMA) :: ZZW, &
                           ZRDRYG_INIT, & !Initial dry growth rate of the graupeln
                           ZRWETG_INIT !Initial wet growth rate of the graupeln
REAL :: ZZW0D
INTEGER :: JL
!-------------------------------------------------------------------------------
!
!
!*       6.1    Rain contact freezing
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
        !Comparison between heat to be released (to freeze rain) and heat sink (rain and ice temperature change)
        !ZZW0D is the proportion of process that can take place
        ZZW0D = MAX(0., MIN(1., (PRICFRRG(JL)*XCI + PRRCFRIG(JL)*XCL)*(XTT - PT(JL))/ &
                            MAX(1.E-20, XLVTT*PRRCFRIG(JL))))
        PRRCFRIG(JL) = ZZW0D*PRRCFRIG(JL) !Part of rain that can be freezed
        PRICFRR(JL) = (1.-ZZW0D)*PRICFRRG(JL) !Part of collected pristine ice converted to rain
        PRICFRRG(JL) = ZZW0D*PRICFRRG(JL) !Part of collected pristine ice that lead to graupel
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
!
!*       6.3    Compute the graupel growth
!
! Wet and dry collection of rc and ri on graupel
DO JL = 1, KSIZE
  IF (PRGT(JL) > G_RTMIN .AND. PRCT(JL) > C_RTMIN .AND. LDCOMPUTE(JL)) THEN
    IF (.NOT. LDSOFT) THEN
      PRG_TEND(JL, IRCDRYG) = PLBDAG(JL)**(XCXG - XDG - 2.)*PRHODREF(JL)**(-XCEXVT)
      PRG_TEND(JL, IRCDRYG) = XFCDRYG*PRCT(JL)*PRG_TEND(JL, IRCDRYG)
    END IF
  ELSE
    PRG_TEND(JL, IRCDRYG) = 0.
  END IF

  IF (PRGT(JL) > G_RTMIN .AND. PRIT(JL) > I_RTMIN .AND. LDCOMPUTE(JL)) THEN
    IF (.NOT. LDSOFT) THEN
      PRG_TEND(JL, IRIDRYG) = PLBDAG(JL)**(XCXG - XDG - 2.)*PRHODREF(JL)**(-XCEXVT)
      PRG_TEND(JL, IRIDRYG) = XFIDRYG*EXP(XCOLEXIG*(PT(JL) - XTT))*PRIT(JL)*PRG_TEND(JL, IRIDRYG)
      PRG_TEND(JL, IRIWETG) = PRG_TEND(JL, IRIDRYG)/(XCOLIG*EXP(XCOLEXIG*(PT(JL) - XTT)))
    END IF
  ELSE
    PRG_TEND(JL, IRIDRYG) = 0.
    PRG_TEND(JL, IRIWETG) = 0.
  END IF
END DO

! Wet and dry collection of rs on graupel (6.2.1)
DO JL = 1, KSIZE
  IF (PRST(JL) > S_RTMIN .AND. PRGT(JL) > G_RTMIN .AND. LDCOMPUTE(JL)) THEN
    GDRY(JL) = .TRUE.
  ELSE
    GDRY(JL) = .FALSE.
    PRG_TEND(JL, IRSDRYG) = 0.
    PRG_TEND(JL, IRSWETG) = 0.
  END IF
END DO

IF (.NOT. LDSOFT) THEN
  CALL INTERP_MICRO_2D(KPROMA, KSIZE, PLBDAG(:), PLBDAS(:), NDRYLBDAG, NDRYLBDAS, &
      &XDRYINTP1G, XDRYINTP2G, XDRYINTP1S, XDRYINTP2S, &
      &LPACK_INTERP, GDRY(:), IBUF1(:), IBUF2(:), IBUF3(:), ZBUF1(:), ZBUF2(:), ZBUF3(:), &
      &IGDRY, &
      &XKER_SDRYG(:, :), ZZW(:))
  IF (IGDRY > 0) THEN
    WHERE (GDRY(1:KSIZE))
      PRG_TEND(1:KSIZE, IRSWETG) = XFSDRYG*ZZW(1:KSIZE) & ! RSDRYG
                                   /XCOLSG &
                                   *(PLBDAS(1:KSIZE)**(XCXS - XBS))*(PLBDAG(1:KSIZE)**XCXG) &
                                   *(PRHODREF(1:KSIZE)**(-XCEXVT - 1.)) &
                                   *(XLBSDRYG1/(PLBDAG(1:KSIZE)**2) + &
                                     XLBSDRYG2/(PLBDAG(1:KSIZE)*PLBDAS(1:KSIZE)) + &
                                     XLBSDRYG3/(PLBDAS(1:KSIZE)**2))

      PRG_TEND(1:KSIZE, IRSDRYG) = PRG_TEND(1:KSIZE, IRSWETG)*XCOLSG*EXP(XCOLEXSG*(PT(1:KSIZE) - XTT))
    END WHERE
  END IF
END IF
!
!*       6.2.6  Accretion of raindrops on the graupeln
!
DO JL = 1, KSIZE
  IF (PRRT(JL) > R_RTMIN .AND. PRGT(JL) > G_RTMIN .AND. LDCOMPUTE(JL)) THEN
    GDRY(JL) = .TRUE.
  ELSE
    GDRY(JL) = .FALSE.
    PRG_TEND(JL, IRRDRYG) = 0.
  END IF
END DO
IF (.NOT. LDSOFT) THEN
!
  CALL INTERP_MICRO_2D(KPROMA, KSIZE, PLBDAG(:), PLBDAR(:), NDRYLBDAG, NDRYLBDAR, &
      &XDRYINTP1G, XDRYINTP2G, XDRYINTP1R, XDRYINTP2R, &
      &LPACK_INTERP, GDRY(:), IBUF1(:), IBUF2(:), IBUF3(:), ZBUF1(:), ZBUF2(:), ZBUF3(:), &
      &IGDRY, &
      &XKER_RDRYG(:, :), ZZW(:))
  IF (IGDRY > 0) THEN
    WHERE (GDRY(1:KSIZE))
      PRG_TEND(1:KSIZE, IRRDRYG) = XFRDRYG*ZZW(1:KSIZE) & ! RRDRYG
                                   *(PLBDAR(1:KSIZE)**(-4))*(PLBDAG(1:KSIZE)**XCXG) &
                                   *(PRHODREF(1:KSIZE)**(-XCEXVT - 1.)) &
                                   *(XLBRDRYG1/(PLBDAG(1:KSIZE)**2) + &
                                     XLBRDRYG2/(PLBDAG(1:KSIZE)*PLBDAR(1:KSIZE)) + &
                                     XLBRDRYG3/(PLBDAR(1:KSIZE)**2))
    END WHERE
  END IF
END IF

DO JL = 1, KSIZE
  ZRDRYG_INIT(JL) = PRG_TEND(JL, IRCDRYG) + PRG_TEND(JL, IRIDRYG) + &
  &PRG_TEND(JL, IRSDRYG) + PRG_TEND(JL, IRRDRYG)
END DO

DO JL = 1, KSIZE
  IF (PRGT(JL) > G_RTMIN .AND. LDCOMPUTE(JL)) THEN
    IF (.NOT. LDSOFT) THEN
      PRG_TEND(JL, IFREEZ1) = PRVT(JL)*PPRES(JL)/(XEPSILO + PRVT(JL)) ! Vapor pressure
      IF (LEVLIMIT) THEN
        PRG_TEND(JL, IFREEZ1) = MIN(PRG_TEND(JL, IFREEZ1), EXP(XALPI - XBETAI/PT(JL) - XGAMI*ALOG(PT(JL)))) ! min(ev, es_i(t))
      END IF
      PRG_TEND(JL, IFREEZ1) = PKA(JL)*(XTT - PT(JL)) + &
                              (PDV(JL)*(XLVTT + (XCPV - XCL)*(PT(JL) - XTT)) &
                               *(XESTT - PRG_TEND(JL, IFREEZ1))/(XRV*PT(JL)))
      PRG_TEND(JL, IFREEZ1) = PRG_TEND(JL, IFREEZ1)*(X0DEPG*PLBDAG(JL)**XEX0DEPG + &
                                                     X1DEPG*PCJ(JL)*PLBDAG(JL)**XEX1DEPG)/ &
                              (PRHODREF(JL)*(XLMTT - XCL*(XTT - PT(JL))))
      PRG_TEND(JL, IFREEZ2) = (PRHODREF(JL)*(XLMTT + (XCI - XCL)*(XTT - PT(JL))))/ &
                              (PRHODREF(JL)*(XLMTT - XCL*(XTT - PT(JL))))
    END IF
    ZRWETG_INIT(JL) = MAX(PRG_TEND(JL, IRIWETG) + PRG_TEND(JL, IRSWETG), &
        &MAX(0., PRG_TEND(JL, IFREEZ1) + &
        &        PRG_TEND(JL, IFREEZ2)*(PRG_TEND(JL, IRIWETG) + PRG_TEND(JL, IRSWETG))))

    LDWETG(JL) = MAX(0., ZRWETG_INIT(JL) - PRG_TEND(JL, IRIWETG) - PRG_TEND(JL, IRSWETG)) <= &
    &MAX(0., ZRDRYG_INIT(JL) - PRG_TEND(JL, IRIDRYG) - PRG_TEND(JL, IRSDRYG))

    IF (LNULLWETG) THEN
      LDWETG(JL) = LDWETG(JL) .AND. ZRDRYG_INIT(JL) > 0.
    ELSE
      LDWETG(JL) = LDWETG(JL) .AND. ZRWETG_INIT(JL) > 0.
    END IF
    IF (.NOT. LWETGPOST) THEN
      LDWETG(JL) = LDWETG(JL) .AND. PT(JL) < XTT
    END IF

    LLDRYG(JL) = PT(JL) < XTT .AND. ZRDRYG_INIT(JL) > 1.E-20 .AND. &
    &MAX(0., ZRWETG_INIT(JL) - PRG_TEND(JL, IRIWETG) - PRG_TEND(JL, IRSWETG)) > &
    &MAX(0., ZRDRYG_INIT(JL) - PRG_TEND(JL, IRIDRYG) - PRG_TEND(JL, IRSDRYG))
  ELSE
    PRG_TEND(JL, IFREEZ1) = 0.
    PRG_TEND(JL, IFREEZ2) = 0.
    ZRWETG_INIT(JL) = 0.
    LDWETG(JL) = .FALSE.
    LLDRYG(JL) = .FALSE.
  END IF
END DO

IF (KRR == 7) THEN
  WHERE (LDWETG(1:KSIZE))
    PRWETGH(1:KSIZE) = (MAX(0., PRGSI(1:KSIZE) + PRICFRRG(1:KSIZE) + PRRCFRIG(1:KSIZE)) + ZRWETG_INIT(1:KSIZE))* &
     &ZRDRYG_INIT(1:KSIZE)/(ZRWETG_INIT(1:KSIZE) + ZRDRYG_INIT(1:KSIZE))
    PRWETGH_MR(1:KSIZE) = MAX(0., PRGSI_MR(1:KSIZE))*ZRDRYG_INIT(1:KSIZE)/(ZRWETG_INIT(1:KSIZE) + ZRDRYG_INIT(1:KSIZE))
  ELSEWHERE
    PRWETGH(1:KSIZE) = 0.
    PRWETGH_MR(1:KSIZE) = 0.
  END WHERE
ELSE
  PRWETGH(:) = 0.
  PRWETGH_MR(:) = 0.
END IF

DO JL = 1, KSIZE
!Aggregated minus collected
  IF (LDWETG(JL)) THEN
    PRRWETG(JL) = -(PRG_TEND(JL, IRIWETG) + PRG_TEND(JL, IRSWETG) + &
    &PRG_TEND(JL, IRCDRYG) - ZRWETG_INIT(JL))
    PRCWETG(JL) = PRG_TEND(JL, IRCDRYG)
    PRIWETG(JL) = PRG_TEND(JL, IRIWETG)
    PRSWETG(JL) = PRG_TEND(JL, IRSWETG)
  ELSE
    PRRWETG(JL) = 0.
    PRCWETG(JL) = 0.
    PRIWETG(JL) = 0.
    PRSWETG(JL) = 0.
  END IF

  IF (LLDRYG(JL)) THEN
    PRCDRYG(JL) = PRG_TEND(JL, IRCDRYG)
    PRRDRYG(JL) = PRG_TEND(JL, IRRDRYG)
    PRIDRYG(JL) = PRG_TEND(JL, IRIDRYG)
    PRSDRYG(JL) = PRG_TEND(JL, IRSDRYG)
  ELSE
    PRCDRYG(JL) = 0.
    PRRDRYG(JL) = 0.
    PRIDRYG(JL) = 0.
    PRSDRYG(JL) = 0.
  END IF
END DO
!
!*       6.5    Melting of the graupeln
!
DO JL = 1, KSIZE
  IF (PRGT(JL) > G_RTMIN .AND. PT(JL) > XTT .AND. LDCOMPUTE(JL)) THEN
    IF (.NOT. LDSOFT) THEN
      PRGMLTR(JL) = PRVT(JL)*PPRES(JL)/(XEPSILO + PRVT(JL)) ! Vapor pressure
      IF (LEVLIMIT) THEN
        PRGMLTR(JL) = MIN(PRGMLTR(JL), EXP(XALPW - XBETAW/PT(JL) - XGAMW*ALOG(PT(JL)))) ! min(ev, es_w(t))
      END IF
      PRGMLTR(JL) = PKA(JL)*(XTT - PT(JL)) + &
                    PDV(JL)*(XLVTT + (XCPV - XCL)*(PT(JL) - XTT)) &
                    *(XESTT - PRGMLTR(JL))/(XRV*PT(JL))
      PRGMLTR(JL) = MAX(0., (-PRGMLTR(JL)* &
                             (X0DEPG*PLBDAG(JL)**XEX0DEPG + &
                              X1DEPG*PCJ(JL)*PLBDAG(JL)**XEX1DEPG) - &
                             (PRG_TEND(JL, IRCDRYG) + PRG_TEND(JL, IRRDRYG))* &
                             (PRHODREF(JL)*XCL*(XTT - PT(JL))))/ &
                        (PRHODREF(JL)*XLMTT))
    END IF
  ELSE
    PRGMLTR(JL) = 0.
  END IF
END DO
!
!
END SUBROUTINE ICE4_FAST_RG
!
!-------------------------------------------------------------------------------

!
END MODULE MODE_ICE4_FAST_RG

MODULE PARKIND1
!
!     *** Define usual kinds for strong typing ***
!
IMPLICIT NONE
SAVE
!
!     Integer Kinds
!     -------------
!
INTEGER, PARAMETER :: JPIT = SELECTED_INT_KIND(2)
INTEGER, PARAMETER :: JPIS = SELECTED_INT_KIND(4)
INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
INTEGER, PARAMETER :: JPIB = SELECTED_INT_KIND(12)

!Special integer type to be used for sensative adress calculations
!should be *8 for a machine with 8byte adressing for optimum performance
#ifdef ADDRESS64
INTEGER, PARAMETER :: JPIA = JPIB
#else
INTEGER, PARAMETER :: JPIA = JPIM
#endif

!
!     Real Kinds
!     ----------
!
INTEGER, PARAMETER :: JPRT = SELECTED_REAL_KIND(2,1)
INTEGER, PARAMETER :: JPRS = SELECTED_REAL_KIND(4,2)
INTEGER, PARAMETER :: JPRM = SELECTED_REAL_KIND(6,37)
#ifdef PARKIND1_SINGLE
INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(6,37)
#else
INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
#endif

! Double real for C code and special places requiring 
!    higher precision. 
INTEGER, PARAMETER :: JPRD = SELECTED_REAL_KIND(13,300)


! Logical Kinds for RTTOV....

INTEGER, PARAMETER :: JPLM = JPIM   !Standard logical type

END MODULE PARKIND1

