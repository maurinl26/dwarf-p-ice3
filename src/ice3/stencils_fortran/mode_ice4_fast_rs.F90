!!      ######spl
       MODULE MODE_ICE4_FAST_RS
!!      ######spl
!!
!!    PURPOSE
!!    -------
!!      Computes the fast rs process
!!
!!
!!**  METHOD
!!    ------
!!      The fast growth processes of snow/aggregates are treated in this routine.
!!      Processes include:
!!        - Cloud droplet riming of the aggregates (RCRIMSS, RCRIMSG, RSRIMCG)
!!        - Rain accretion onto the aggregates (RRACCSS, RRACCSG, RSACCRG)
!!        - Conversion-melting of the aggregates (RSMLTG, RCMLTSR)
!!
!!    REFERENCE
!!    ---------
!!
!!      PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_processes.F90
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
!USE PARKIND1, ONLY : JPRB
!
IMPLICIT NONE
!
CONTAINS
!
!-------------------------------------------------------------------------------
!
SUBROUTINE ICE4_FAST_RS(KPROMA, KSIZE, LDSOFT, LDCOMPUTE, &
                       &NGAMINC, NACCLBDAS, NACCLBDAR, &
                       &LEVLIMIT, LPACK_INTERP, CSNOWRIMING, &
                       &XCRIMSS, XEXCRIMSS, XCRIMSG, XEXCRIMSG, XEXSRIMCG2,&
                       &XFRACCSS, &
                       &S_RTMIN, C_RTMIN, R_RTMIN, XEPSILO, XALPI, XBETAI, &
                       &XGAMI, XTT, XLVTT, XCPV, XCI, XCL, XLMTT, &
                       &XESTT, XRV, X0DEPS, X1DEPS, XEX0DEPS, XEX1DEPS, &
                       &XLBRACCS1, XLBRACCS2, XLBRACCS3, &
                       &XCXS, XSRIMCG2, XSRIMCG3, XBS, &
                       &XLBSACCR1, XLBSACCR2, XLBSACCR3, XFSACCRG, &
                       &XSRIMCG, XEXSRIMCG, XCEXVT, &
                       &XALPW, XBETAW, XGAMW, XFSCVMG, &
                       &XKER_RACCSS, XKER_RACCS, XKER_SACCRG, &
                       &XGAMINC_RIM1, XGAMINC_RIM2, XGAMINC_RIM4, &
                       &XRIMINTP1, XRIMINTP2, XACCINTP1S, XACCINTP2S, XACCINTP1R, XACCINTP2R, &
                       &PRHODREF, PPRES, &
                       &PDV, PKA, PCJ, &
                       &PLBDAR, PLBDAS, &
                       &PT, PRVT, PRCT, PRRT, PRST, &
                       &PRIAGGS, &
                       &PRCRIMSS, PRCRIMSG, PRSRIMCG, &
                       &PRRACCSS, PRRACCSG, PRSACCRG, PRSMLTG, &
                       &PRCMLTSR, &
                       &PRS_TEND)
!!
!!**  PURPOSE
!!    -------
!!      Computes the fast snow/aggregate processes
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
LOGICAL,                      INTENT(IN)    :: LEVLIMIT, LPACK_INTERP
LOGICAL, DIMENSION(KPROMA),   INTENT(IN)    :: LDCOMPUTE
INTEGER,                      INTENT(IN)    :: NGAMINC
CHARACTER(LEN=4),             INTENT(IN)    :: CSNOWRIMING
REAL,                         INTENT(IN)    :: S_RTMIN, C_RTMIN, R_RTMIN, XEPSILO, XALPI, XBETAI
REAL,                         INTENT(IN)    :: XGAMI, XTT, XLVTT, XCPV, XCI, XCL, XLMTT
REAL,                         INTENT(IN)    :: XESTT, XRV, X0DEPS, X1DEPS, XEX0DEPS, XEX1DEPS
REAL,                         INTENT(IN)    :: XCRIMSS, XEXCRIMSS, XCRIMSG, XEXCRIMSG
REAL,                         INTENT(IN)    :: XSRIMCG, XEXSRIMCG, XCEXVT, XEXSRIMCG2
REAL,                         INTENT(IN)    :: XFRACCSS, XLBRACCS1, XLBRACCS2, XLBRACCS3
REAL,                         INTENT(IN)    :: XCXS, XSRIMCG2, XSRIMCG3, XBS
REAL,                         INTENT(IN)    :: XLBSACCR1, XLBSACCR2, XLBSACCR3, XFSACCRG
REAL,                         INTENT(IN)    :: XALPW, XBETAW, XGAMW, XFSCVMG
REAL, DIMENSION(:,:),         INTENT(IN)    :: XKER_RACCSS, XKER_RACCS, XKER_SACCRG
REAL, DIMENSION(:),           INTENT(IN)    :: XGAMINC_RIM1, XGAMINC_RIM2, XGAMINC_RIM4
REAL,                         INTENT(IN)    :: XRIMINTP1, XRIMINTP2
INTEGER,                      INTENT(IN)    :: NACCLBDAS, NACCLBDAR
REAL,                         INTENT(IN)    :: XACCINTP1S, XACCINTP2S, XACCINTP1R, XACCINTP2R

REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRHODREF ! Reference density
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PPRES    ! Absolute pressure at t
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PDV      ! Diffusivity of water vapor in the air
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PKA      ! Thermal conductivity of the air
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PCJ      ! Function to compute the ventilation coefficient
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PLBDAR   ! Slope parameter of the raindrop  distribution
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PLBDAS   ! Slope parameter of the aggregate distribution
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PT       ! Temperature
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRVT     ! Water vapor m.r. at t
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRCT     ! Cloud water m.r. at t
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRRT     ! Rain water m.r. at t
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRST     ! Snow/aggregate m.r. at t
REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRIAGGS  ! R_i aggregation on r_s
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRCRIMSS ! Cloud droplet riming of the aggregates
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRCRIMSG ! Cloud droplet riming of the aggregates
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRSRIMCG ! Cloud droplet riming of the aggregates
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRRACCSS ! Rain accretion onto the aggregates
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRRACCSG ! Rain accretion onto the aggregates
REAL, DIMENSION(KPROMA),      INTENT(OUT)   :: PRSACCRG ! Rain accretion onto the aggregates
REAL, DIMENSION(KPROMA),      INTENT(INOUT) :: PRSMLTG  ! Conversion-melting of the aggregates
REAL, DIMENSION(KPROMA),      INTENT(INOUT) :: PRCMLTSR ! Cloud droplet collection onto aggregates by positive temperature
REAL, DIMENSION(KPROMA, 8),   INTENT(INOUT) :: PRS_TEND ! Individual tendencies
!
!*       0.2  Declaration of local variables
!
INTEGER, PARAMETER :: IRCRIMS = 1, IRCRIMSS = 2, IRSRIMCG = 3, IRRACCS = 4, IRRACCSS = 5, IRSACCRG = 6, &
 & IFREEZ1 = 7, IFREEZ2 = 8
LOGICAL, DIMENSION(KPROMA) :: GRIM, GACC
INTEGER :: IGRIM, IGACC
INTEGER, DIMENSION(KPROMA) :: IBUF1, IBUF2, IBUF3
REAL, DIMENSION(KPROMA) :: ZBUF1, ZBUF2, ZBUF3
REAL, DIMENSION(KPROMA) :: ZZW, ZZW1, ZZW2, ZZW3, ZFREEZ_RATE
INTEGER :: JL
REAL :: ZZW0D
!-------------------------------------------------------------------------------
!
!
!*       5.0    Maximum freezing rate
!
DO JL = 1, KSIZE
  IF (PRST(JL) > S_RTMIN .AND. LDCOMPUTE(JL)) THEN
    IF (.NOT. LDSOFT) THEN
      PRS_TEND(JL, IFREEZ1) = PRVT(JL)*PPRES(JL)/(XEPSILO + PRVT(JL)) ! Vapor pressure
      IF (LEVLIMIT) THEN
        PRS_TEND(JL, IFREEZ1) = MIN(PRS_TEND(JL, IFREEZ1), EXP(XALPI - XBETAI/PT(JL) - XGAMI*ALOG(PT(JL)))) ! min(ev, es_i(t))
      END IF
      PRS_TEND(JL, IFREEZ1) = PKA(JL)*(XTT - PT(JL)) +                              &
                              (PDV(JL)*(XLVTT + (XCPV - XCL)*(PT(JL) - XTT)) &
                              *(XESTT - PRS_TEND(JL, IFREEZ1))/(XRV*PT(JL)))
      PRS_TEND(JL, IFREEZ1) = PRS_TEND(JL, IFREEZ1)*(X0DEPS*PLBDAS(JL)**XEX0DEPS +     &
                                                     X1DEPS*PCJ(JL)*PLBDAS(JL)**XEX1DEPS)/ &
                              (PRHODREF(JL)*(XLMTT - XCL*(XTT - PT(JL))))

      PRS_TEND(JL, IFREEZ2) = (PRHODREF(JL)*(XLMTT + (XCI - XCL)*(XTT - PT(JL))))/ &
                              (PRHODREF(JL)*(XLMTT - XCL*(XTT - PT(JL))))
    END IF

    ZFREEZ_RATE(JL) = MAX(0., MAX(0., PRS_TEND(JL, IFREEZ1) + &
                             PRS_TEND(JL, IFREEZ2)*PRIAGGS(JL)) - &
                     PRIAGGS(JL))
  ELSE
    PRS_TEND(JL, IFREEZ1) = 0.
    PRS_TEND(JL, IFREEZ2) = 0.
    ZFREEZ_RATE(JL) = 0.
  END IF
END DO
!
!
!*       5.1    Cloud droplet riming of the aggregates
!
DO JL = 1, KSIZE
  IF (PRCT(JL) > C_RTMIN .AND. PRST(JL) > S_RTMIN .AND. LDCOMPUTE(JL)) THEN
    ZZW(JL) = PLBDAS(JL)

    GRIM(JL) = .TRUE.
  ELSE
    GRIM(JL) = .FALSE.
    PRS_TEND(JL, IRCRIMS) = 0.
    PRS_TEND(JL, IRCRIMSS) = 0.
    PRS_TEND(JL, IRSRIMCG) = 0.
  END IF
END DO
!
! Collection of cloud droplets by snow: this rate is used for riming (t<0) and for conversion/melting (t>0)
IF (.NOT. LDSOFT) THEN
  CALL INTERP_MICRO_1D(KPROMA, KSIZE, ZZW, NGAMINC, XRIMINTP1, XRIMINTP2, &
                       LPACK_INTERP, GRIM(:), IBUF1, IBUF2, ZBUF1, ZBUF2, &
                       IGRIM, &
                       XGAMINC_RIM1(:), ZZW1(:), XGAMINC_RIM2(:), ZZW2(:), XGAMINC_RIM4(:), ZZW3(:))
  IF (IGRIM > 0) THEN
!
!        5.1.4  Riming of the small sized aggregates
!
    WHERE (GRIM(1:KSIZE))
      PRS_TEND(1:KSIZE, IRCRIMSS) = XCRIMSS*ZZW1(1:KSIZE)*PRCT(1:KSIZE) & ! RCRIMSS
                                    *PLBDAS(1:KSIZE)**XEXCRIMSS &
                                    *PRHODREF(1:KSIZE)**(-XCEXVT)

    END WHERE
!
!        5.1.6  Riming-conversion of the large sized aggregates into graupeln
!
    WHERE (GRIM(1:KSIZE))
      PRS_TEND(1:KSIZE, IRCRIMS) = XCRIMSG*PRCT(1:KSIZE) & ! RCRIMS
                                   *PLBDAS(1:KSIZE)**XEXCRIMSG &
                                   *PRHODREF(1:KSIZE)**(-XCEXVT)

    END WHERE

    IF (CSNOWRIMING == 'M90 ') THEN
      ! Murakami 1990
      WHERE (GRIM(1:KSIZE))
        ZZW(1:KSIZE) = PRS_TEND(1:KSIZE, IRCRIMS) - PRS_TEND(1:KSIZE, IRCRIMSS) ! RCRIMSG
        PRS_TEND(1:KSIZE, IRSRIMCG) = XSRIMCG*PLBDAS(1:KSIZE)**XEXSRIMCG*(1.0 - ZZW2(1:KSIZE))

        PRS_TEND(1:KSIZE, IRSRIMCG) = ZZW(1:KSIZE)*PRS_TEND(1:KSIZE, IRSRIMCG)/ &
                                      MAX(1.E-20, &
                                          XSRIMCG3*XSRIMCG2*PLBDAS(1:KSIZE)**XEXSRIMCG2*(1.-ZZW3(1:KSIZE)) - &
                                          XSRIMCG3*PRS_TEND(1:KSIZE, IRSRIMCG))

      END WHERE
    ELSE
      PRS_TEND(:, IRSRIMCG) = 0.
    END IF
  END IF
END IF
!
DO JL = 1, KSIZE
  ! More restrictive rim mask to be used for riming by negative temperature only
  IF (GRIM(JL) .AND. PT(JL) < XTT) THEN
    PRCRIMSS(JL) = MIN(ZFREEZ_RATE(JL), PRS_TEND(JL, IRCRIMSS))
    ZFREEZ_RATE(JL) = MAX(0., ZFREEZ_RATE(JL) - PRCRIMSS(JL))
    ZZW0D = MIN(1., ZFREEZ_RATE(JL)/MAX(1.E-20, PRS_TEND(JL, IRCRIMS) - PRCRIMSS(JL))) ! Proportion we are able to freeze
    PRCRIMSG(JL) = ZZW0D*MAX(0., PRS_TEND(JL, IRCRIMS) - PRCRIMSS(JL)) ! RCRIMSG
    ZFREEZ_RATE(JL) = MAX(0., ZFREEZ_RATE(JL) - PRCRIMSG(JL))
    PRSRIMCG(JL) = ZZW0D*PRS_TEND(JL, IRSRIMCG)

    PRSRIMCG(JL) = PRSRIMCG(JL)*MAX(0., -SIGN(1., -PRCRIMSG(JL)))
    PRCRIMSG(JL) = MAX(0., PRCRIMSG(JL))
  ELSE
    PRCRIMSS(JL) = 0.
    PRCRIMSG(JL) = 0.
    PRSRIMCG(JL) = 0.
  END IF
END DO
!
!
!*       5.2    Rain accretion onto the aggregates
!
DO JL = 1, KSIZE
  IF (PRRT(JL) > R_RTMIN .AND. PRST(JL) > S_RTMIN .AND. LDCOMPUTE(JL)) THEN
    GACC(JL) = .TRUE.
  ELSE
    GACC(JL) = .FALSE.
    PRS_TEND(JL, IRRACCS) = 0.
    PRS_TEND(JL, IRRACCSS) = 0.
    PRS_TEND(JL, IRSACCRG) = 0.
  END IF
END DO
IF (.NOT. LDSOFT) THEN
  PRS_TEND(:, IRRACCS) = 0.
  PRS_TEND(:, IRRACCSS) = 0.
  PRS_TEND(:, IRSACCRG) = 0.
  CALL INTERP_MICRO_2D(KPROMA, KSIZE, PLBDAS, PLBDAR, NACCLBDAS, NACCLBDAR, &
                       XACCINTP1S, XACCINTP2S, XACCINTP1R, XACCINTP2R,&
                       LPACK_INTERP, GACC(:), IBUF1(:), IBUF2(:), IBUF3(:), ZBUF1(:), ZBUF2(:), ZBUF3(:), &
                       IGACC, &
                       XKER_RACCSS(:, :), ZZW1(:), XKER_RACCS(:, :), ZZW2(:), XKER_SACCRG(:, :), ZZW3(:))
  IF (IGACC > 0) THEN
    !        5.2.4  Raindrop accretion on the small sized aggregates
    WHERE (GACC(1:KSIZE))
      ZZW(1:KSIZE) = & !! Coef of RRACCS
         XFRACCSS*(PLBDAS(1:KSIZE)**XCXS)*(PRHODREF(1:KSIZE)**(-XCEXVT - 1.)) &
         *(XLBRACCS1/((PLBDAS(1:KSIZE)**2)) + &
           XLBRACCS2/(PLBDAS(1:KSIZE)*PLBDAR(1:KSIZE)) + &
           XLBRACCS3/((PLBDAR(1:KSIZE)**2)))/PLBDAR(1:KSIZE)**4

      PRS_TEND(1:KSIZE, IRRACCSS) = ZZW1(1:KSIZE)*ZZW(1:KSIZE)
    END WHERE

    WHERE (GACC(1:KSIZE))
      PRS_TEND(1:KSIZE, IRRACCS) = ZZW2(1:KSIZE)*ZZW(1:KSIZE)
    END WHERE
!
!        5.2.6  Raindrop accretion-conversion of the large sized aggregates
!               into graupeln
!
    WHERE (GACC(1:KSIZE))
      PRS_TEND(1:KSIZE, IRSACCRG) = XFSACCRG*ZZW3(1:KSIZE)* & ! RSACCRG
                                    (PLBDAS(1:KSIZE)**(XCXS - XBS))*(PRHODREF(1:KSIZE)**(-XCEXVT - 1.)) &
                                    *(XLBSACCR1/((PLBDAR(1:KSIZE)**2)) + &
                                      XLBSACCR2/(PLBDAR(1:KSIZE)*PLBDAS(1:KSIZE)) + &
                                      XLBSACCR3/((PLBDAS(1:KSIZE)**2)))/PLBDAR(1:KSIZE)

    END WHERE
  END IF
END IF
!
DO JL = 1, KSIZE
  ! More restrictive acc mask to be used for accretion by negative temperature only
  IF (GACC(JL) .AND. PT(JL) < XTT) THEN
    PRRACCSS(JL) = MIN(ZFREEZ_RATE(JL), PRS_TEND(JL, IRRACCSS))
    ZFREEZ_RATE(JL) = MAX(0., ZFREEZ_RATE(JL) - PRRACCSS(JL))
    ZZW(JL) = MIN(1., ZFREEZ_RATE(JL)/MAX(1.E-20, PRS_TEND(JL, IRRACCS) - PRRACCSS(JL))) ! Proportion we are able to freeze
    PRRACCSG(JL) = ZZW(JL)*MAX(0., PRS_TEND(JL, IRRACCS) - PRRACCSS(JL))
    ZFREEZ_RATE(JL) = MAX(0., ZFREEZ_RATE(JL) - PRRACCSG(JL))
    PRSACCRG(JL) = ZZW(JL)*PRS_TEND(JL, IRSACCRG)

    PRSACCRG(JL) = PRSACCRG(JL)*MAX(0., -SIGN(1., -PRRACCSG(JL)))
    PRRACCSG(JL) = MAX(0., PRRACCSG(JL))
  ELSE
    PRRACCSS(JL) = 0.
    PRRACCSG(JL) = 0.
    PRSACCRG(JL) = 0.
  END IF
END DO
!
!
!*       5.3    Conversion-melting of the aggregates
!
DO JL = 1, KSIZE
  IF (PRST(JL) > S_RTMIN .AND. PT(JL) > XTT .AND. LDCOMPUTE(JL)) THEN
    IF (.NOT. LDSOFT) THEN
      PRSMLTG(JL) = PRVT(JL)*PPRES(JL)/(XEPSILO + PRVT(JL)) ! Vapor pressure
      IF (LEVLIMIT) THEN
        PRSMLTG(JL) = MIN(PRSMLTG(JL), EXP(XALPW - XBETAW/PT(JL) - XGAMW*ALOG(PT(JL)))) ! min(ev, es_w(t))
      END IF
      PRSMLTG(JL) = PKA(JL)*(XTT - PT(JL)) +                                 &
                    (PDV(JL)*(XLVTT + (XCPV - XCL)*(PT(JL) - XTT)) &
                    *(XESTT - PRSMLTG(JL))/(XRV*PT(JL)))
!
! Compute RSMLT
!
      PRSMLTG(JL) = XFSCVMG*MAX(0., (-PRSMLTG(JL)* &
                                     (X0DEPS*PLBDAS(JL)**XEX0DEPS + &
                                      X1DEPS*PCJ(JL)*PLBDAS(JL)**XEX1DEPS) &
                                     - (PRS_TEND(JL, IRCRIMS) + PRS_TEND(JL, IRRACCS))* &
                                     (PRHODREF(JL)*XCL*(XTT - PT(JL))) &
                                     )/(PRHODREF(JL)*XLMTT))
!

      PRCMLTSR(JL) = PRS_TEND(JL, IRCRIMS) ! Both species are liquid, no heat is exchanged
    END IF
  ELSE
    PRSMLTG(JL) = 0.
    PRCMLTSR(JL) = 0.
  END IF
END DO
!
!
END SUBROUTINE ICE4_FAST_RS
!
!-------------------------------------------------------------------------------
!
SUBROUTINE INTERP_MICRO_1D(KPROMA, KSIZE, PIN, KNUM, P1, P2, &
                           LDPACK, LDMASK, KBUF1, KBUF2, PBUF1, PBUF2, &
                           KLEN, &
                           PLT1, POUT1, PLT2, POUT2, PLT3, POUT3)

IMPLICIT NONE

INTEGER,                    INTENT(IN)  :: KPROMA       ! Array size
INTEGER,                    INTENT(IN)  :: KSIZE        ! Last usefull array index
REAL,    DIMENSION(KPROMA), INTENT(IN)  :: PIN          ! Input array
INTEGER,                    INTENT(IN)  :: KNUM         ! Number of points in the look-up table
REAL,                       INTENT(IN)  :: P1           ! Scaling factor
REAL,                       INTENT(IN)  :: P2           ! Scaling factor
LOGICAL,                    INTENT(IN)  :: LDPACK       ! .TRUE. to perform packing
LOGICAL, DIMENSION(KPROMA), INTENT(IN)  :: LDMASK       ! Computation mask
INTEGER, DIMENSION(KPROMA), INTENT(OUT) :: KBUF1, KBUF2 ! Buffer arrays
REAL,    DIMENSION(KPROMA), INTENT(OUT) :: PBUF1, PBUF2 ! Buffer arrays
INTEGER,                    INTENT(OUT) :: KLEN         ! Number of active points
REAL,    DIMENSION(KNUM),   INTENT(IN)            :: PLT1  ! Look-up table
REAL,    DIMENSION(KPROMA), INTENT(OUT)           :: POUT1 ! Interpolated values
REAL,    DIMENSION(KNUM),   INTENT(IN) , OPTIONAL :: PLT2
REAL,    DIMENSION(KPROMA), INTENT(OUT), OPTIONAL :: POUT2
REAL,    DIMENSION(KNUM),   INTENT(IN) , OPTIONAL :: PLT3
REAL,    DIMENSION(KPROMA), INTENT(OUT), OPTIONAL :: POUT3

INTEGER :: JL
INTEGER :: IINDEX
REAL :: ZINDEX

IF (LDPACK) THEN

  ! Pack input array
  KLEN=0
  DO JL=1, KSIZE
    IF (LDMASK(JL)) THEN
      KLEN=KLEN+1
      PBUF1(KLEN)=PIN(JL)
      KBUF1(KLEN)=JL
    ENDIF
  ENDDO

  IF (KLEN>0) THEN
    ! Index computation
    PBUF1(1:KLEN) = MAX(1.00001, MIN(REAL(KNUM)-0.00001, P1 * LOG(PBUF1(1:KLEN)) + P2))
    KBUF2(1:KLEN) = INT(PBUF1(1:KLEN))
    PBUF1(1:KLEN) = PBUF1(1:KLEN) - REAL(KBUF2(1:KLEN))

    ! Interpolation and unpack
    PBUF2(1:KLEN) = PLT1(KBUF2(1:KLEN)+1) *  PBUF1(1:KLEN)       &
                   -PLT1(KBUF2(1:KLEN)  ) * (PBUF1(1:KLEN) - 1.0)
    POUT1(:)=0.
    DO JL=1, KLEN
      POUT1(KBUF1(JL))=PBUF2(JL)
    ENDDO

    ! Interpolation and unpack 2
    IF(PRESENT(PLT2)) THEN
      PBUF2(1:KLEN) = PLT2(KBUF2(1:KLEN)+1) *  PBUF1(1:KLEN)       &
                     -PLT2(KBUF2(1:KLEN)  ) * (PBUF1(1:KLEN) - 1.0)
      POUT2(:)=0.
      DO JL=1, KLEN
        POUT2(KBUF1(JL))=PBUF2(JL)
      ENDDO
    ENDIF

    ! Interpolation and unpack 3
    IF(PRESENT(PLT3)) THEN
      PBUF2(1:KLEN) = PLT3(KBUF2(1:KLEN)+1) *  PBUF1(1:KLEN)       &
                     -PLT3(KBUF2(1:KLEN)  ) * (PBUF1(1:KLEN) - 1.0)
      POUT3(:)=0.
      DO JL=1, KLEN
        POUT3(KBUF1(JL))=PBUF2(JL)
      ENDDO
    ENDIF

  ENDIF

ELSE

  KLEN=0
  DO JL=1, KSIZE
    IF (LDMASK(JL)) THEN
      KLEN=KLEN+1

      ! Index computation
      ZINDEX = MAX(1.00001, MIN(REAL(KNUM)-0.00001, P1 * LOG(PIN(JL)) + P2))
      IINDEX = INT(ZINDEX)
      ZINDEX = ZINDEX - REAL(IINDEX)

      ! Interpolations
      POUT1(JL) = PLT1(IINDEX+1) *  ZINDEX       &
                 -PLT1(IINDEX  ) * (ZINDEX - 1.0)

      IF(PRESENT(PLT2)) THEN
        POUT2(JL) = PLT2(IINDEX+1) *  ZINDEX       &
                   -PLT2(IINDEX  ) * (ZINDEX - 1.0)
      ENDIF

      IF(PRESENT(PLT3)) THEN
        POUT3(JL) = PLT3(IINDEX+1) *  ZINDEX       &
                   -PLT3(IINDEX  ) * (ZINDEX - 1.0)
      ENDIF

    ELSE
      POUT1(JL) = 0.
      IF(PRESENT(PLT2)) POUT2(JL) = 0.
      IF(PRESENT(PLT3)) POUT3(JL) = 0.
    ENDIF
  ENDDO

ENDIF
END SUBROUTINE INTERP_MICRO_1D

SUBROUTINE INTERP_MICRO_2D(KPROMA, KSIZE, PIN1, PIN2, KNUM1, KNUM2, P11, P12, P21, P22,&
                           LDPACK, LDMASK, KBUF1, KBUF2, KBUF3, PBUF1, PBUF2, PBUF3, &
                           KLEN, &
                           PLT1, POUT1, PLT2, POUT2, PLT3, POUT3)

IMPLICIT NONE

INTEGER,                    INTENT(IN)  :: KPROMA       ! Array size
INTEGER,                    INTENT(IN)  :: KSIZE        ! Last usefull array index
REAL,    DIMENSION(KPROMA), INTENT(IN)  :: PIN1         ! Input array
REAL,    DIMENSION(KPROMA), INTENT(IN)  :: PIN2         ! Input array
INTEGER,                    INTENT(IN)  :: KNUM1        ! First dimension of the look-up table
INTEGER,                    INTENT(IN)  :: KNUM2        ! Second dimension of the look-up table
REAL,                       INTENT(IN)  :: P11          ! Scaling factor
REAL,                       INTENT(IN)  :: P12          ! Scaling factor
REAL,                       INTENT(IN)  :: P21          ! Scaling factor
REAL,                       INTENT(IN)  :: P22          ! Scaling factor
LOGICAL,                    INTENT(IN)  :: LDPACK       ! .TRUE. to perform packing
LOGICAL, DIMENSION(KPROMA), INTENT(IN)  :: LDMASK       ! Computation mask
INTEGER, DIMENSION(KPROMA), INTENT(OUT) :: KBUF1, KBUF2, KBUF3 ! Buffer arrays
REAL,    DIMENSION(KPROMA), INTENT(OUT) :: PBUF1, PBUF2, PBUF3 ! Buffer arrays
INTEGER,                    INTENT(OUT) :: KLEN         ! Number of active points
REAL,    DIMENSION(KNUM1, KNUM2),   INTENT(IN)            :: PLT1  ! Look-up table
REAL,    DIMENSION(KPROMA),         INTENT(OUT)           :: POUT1 ! Interpolated values from the first look-up table
REAL,    DIMENSION(KNUM1, KNUM2),   INTENT(IN) , OPTIONAL :: PLT2  ! Other look-up table
REAL,    DIMENSION(KPROMA),         INTENT(OUT), OPTIONAL :: POUT2 ! Interpolated values from the second look-up table
REAL,    DIMENSION(KNUM2, KNUM1),   INTENT(IN) , OPTIONAL :: PLT3  ! Another look-up table **caution, table is reversed**
REAL,    DIMENSION(KPROMA),         INTENT(OUT), OPTIONAL :: POUT3 ! Interpolated values from the third look-up table

INTEGER :: JL
INTEGER :: IINDEX1, IINDEX2
REAL :: ZINDEX1, ZINDEX2

IF (LDPACK) THEN

  ! Pack input array
  KLEN=0
  DO JL=1, KSIZE
    IF (LDMASK(JL)) THEN
      KLEN=KLEN+1
      PBUF1(KLEN)=PIN1(JL)
      PBUF2(KLEN)=PIN2(JL)
      KBUF3(KLEN)=JL
    ENDIF
  ENDDO

  IF (KLEN>0) THEN
    ! Index computation
    PBUF1(1:KLEN) = MAX(1.00001, MIN(REAL(KNUM1)-0.00001, P11 * LOG(PBUF1(1:KLEN)) + P12))
    KBUF1(1:KLEN) = INT(PBUF1(1:KLEN))
    PBUF1(1:KLEN) = PBUF1(1:KLEN) - REAL(KBUF1(1:KLEN))

    PBUF2(1:KLEN) = MAX(1.00001, MIN(REAL(KNUM2)-0.00001, P21 * LOG(PBUF2(1:KLEN)) + P22))
    KBUF2(1:KLEN) = INT(PBUF2(1:KLEN))
    PBUF2(1:KLEN) = PBUF2(1:KLEN) - REAL(KBUF2(1:KLEN))

    ! Interpolation and unpack 1
    DO JL=1, KLEN
      PBUF3(JL) = ( PLT1(KBUF1(JL)+1,KBUF2(JL)+1)* PBUF2(JL)         &
                   -PLT1(KBUF1(JL)+1,KBUF2(JL)  )*(PBUF2(JL) - 1.0)) *  PBUF1(JL) &
                 -( PLT1(KBUF1(JL)  ,KBUF2(JL)+1)* PBUF2(JL)         &
                   -PLT1(KBUF1(JL)  ,KBUF2(JL)  )*(PBUF2(JL) - 1.0)) * (PBUF1(JL) - 1.0)
    ENDDO
    POUT1(:)=0.
    DO JL=1, KLEN
      POUT1(KBUF3(JL))=PBUF3(JL)
    ENDDO

    ! Interpolation and unpack 2
    IF(PRESENT(PLT2)) THEN
      DO JL=1, KLEN
        PBUF3(JL) = ( PLT2(KBUF1(JL)+1,KBUF2(JL)+1)* PBUF2(JL)         &
                     -PLT2(KBUF1(JL)+1,KBUF2(JL)  )*(PBUF2(JL) - 1.0)) *  PBUF1(JL) &
                   -( PLT2(KBUF1(JL)  ,KBUF2(JL)+1)* PBUF2(JL)         &
                     -PLT2(KBUF1(JL)  ,KBUF2(JL)  )*(PBUF2(JL) - 1.0)) * (PBUF1(JL) - 1.0)
      ENDDO
      POUT2(:)=0.
      DO JL=1, KLEN
        POUT2(KBUF3(JL))=PBUF3(JL)
      ENDDO
    ENDIF

    ! Interpolation and unpack 3
    IF(PRESENT(PLT3)) THEN
      DO JL=1, KLEN
        PBUF3(JL) = ( PLT3(KBUF2(JL)+1,KBUF1(JL)+1)* PBUF1(JL)         &
                     -PLT3(KBUF2(JL)+1,KBUF1(JL)  )*(PBUF1(JL) - 1.0)) *  PBUF2(JL) &
                   -( PLT3(KBUF2(JL)  ,KBUF1(JL)+1)* PBUF1(JL)         &
                     -PLT3(KBUF2(JL)  ,KBUF1(JL)  )*(PBUF1(JL) - 1.0)) * (PBUF2(JL) - 1.0)
      ENDDO
      POUT3(:)=0.
      DO JL=1, KLEN
        POUT3(KBUF3(JL))=PBUF3(JL)
      ENDDO
    ENDIF
  ENDIF

ELSE

  KLEN=0
  DO JL=1, KSIZE
    IF (LDMASK(JL)) THEN
      KLEN=KLEN+1

      ! Indexes computation
      ZINDEX1 = MAX(1.00001, MIN(REAL(KNUM1)-0.00001, P11 * LOG(PIN1(JL)) + P12))
      IINDEX1 = INT(ZINDEX1)
      ZINDEX1 = ZINDEX1 - REAL(IINDEX1)
  
      ZINDEX2 = MAX(1.00001, MIN(REAL(KNUM1)-0.00001, P21 * LOG(PIN2(JL)) + P22))
      IINDEX2 = INT(ZINDEX2)
      ZINDEX2 = ZINDEX2 - REAL(IINDEX2)
  
      ! Interpolations
      POUT1(JL) = ( PLT1(IINDEX1+1,IINDEX2+1)* ZINDEX2         &
                   -PLT1(IINDEX1+1,IINDEX2  )*(ZINDEX2 - 1.0)) *  ZINDEX1 &
                 -( PLT1(IINDEX1  ,IINDEX2+1)* ZINDEX2         &
                   -PLT1(IINDEX1  ,IINDEX2  )*(ZINDEX2 - 1.0)) * (ZINDEX1 - 1.0)

      IF(PRESENT(PLT2)) THEN
        POUT2(JL) = ( PLT2(IINDEX1+1,IINDEX2+1)* ZINDEX2         &
                     -PLT2(IINDEX1+1,IINDEX2  )*(ZINDEX2 - 1.0)) *  ZINDEX1 &
                   -( PLT2(IINDEX1  ,IINDEX2+1)* ZINDEX2         &
                     -PLT2(IINDEX1  ,IINDEX2  )*(ZINDEX2 - 1.0)) * (ZINDEX1 - 1.0)
      ENDIF

      IF(PRESENT(PLT3)) THEN
        POUT3(JL) = ( PLT3(IINDEX2+1,IINDEX1+1)* ZINDEX1         &
                     -PLT3(IINDEX2+1,IINDEX1  )*(ZINDEX1 - 1.0)) *  ZINDEX2 &
                   -( PLT3(IINDEX2  ,IINDEX1+1)* ZINDEX1         &
                     -PLT3(IINDEX2  ,IINDEX1  )*(ZINDEX1 - 1.0)) * (ZINDEX2 - 1.0)
      ENDIF

    ELSE
      POUT1(JL)=0.
      IF(PRESENT(PLT2)) POUT2(JL)=0.
      IF(PRESENT(PLT3)) POUT3(JL)=0.
    ENDIF
  ENDDO

ENDIF
END SUBROUTINE INTERP_MICRO_2D

!
END MODULE MODE_ICE4_FAST_RS

