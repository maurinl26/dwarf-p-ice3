MODULE MODE_ICE4_SLOW
    IMPLICIT NONE
    CONTAINS
    SUBROUTINE ICE4_SLOW(XTT, V_RTMIN, C_RTMIN, I_RTMIN, S_RTMIN, G_RTMIN,&
                        &XEXIAGGS, XFIAGGS, XCOLEXIS, XTIMAUTI, XCRIAUTI, XACRIAUTI, XBCRIAUTI,&
                        &XCEXVT,XTEXAUTI,X0DEPG, X1DEPG, XEX0DEPG, XEX1DEPG,&
                        &XHON, XALPHA3, XBETA3, X0DEPS, X1DEPS, XEX1DEPS, XEX0DEPS, &
                        &KPROMA, KSIZE, LDSOFT, LDCOMPUTE, PRHODREF, PT, &
                        &PSSI, &
                        &PRVT, PRCT, PRIT, PRST, PRGT, &
                        &PLBDAS, PLBDAG, &
                        &PAI, PCJ, PHLI_HCF, PHLI_HRI,&
                        &PRCHONI, PRVDEPS, PRIAGGS, PRIAUTS, PRVDEPG)
    !!
    !!**  PURPOSE
    !!    -------
    !!      Computes the slow process
    !!
    !!    AUTHOR
    !!    ------
    !!      S. Riette from the splitting of rain_ice source code (nov. 2014)
    !!
    !!    MODIFICATIONS
    !!    -------------
    !!
    !!     R. El Khatib 24-Aug-2021 Optimizations
    !  J. Wurtz       03/2022: New snow characteristics with LSNOW_T
    !
    !
    !*      0. DECLARATIONS
    !          ------------
    !
    !
    IMPLICIT NONE
    !
    !*       0.1   Declarations of dummy arguments :
    !
    real, intent(in) :: XTT
    real, intent(in) :: V_RTMIN, C_RTMIN, I_RTMIN, S_RTMIN, G_RTMIN
    real, intent(in) :: XHON, XALPHA3, XBETA3, X0DEPS, X1DEPS, XEX1DEPS, XEX0DEPS
    real, intent(in) :: XEXIAGGS, XFIAGGS, XCOLEXIS, XTIMAUTI, XCRIAUTI, XACRIAUTI, XBCRIAUTI
    real, intent(in) :: XCEXVT, XTEXAUTI, X0DEPG, X1DEPG, XEX0DEPG, XEX1DEPG
    INTEGER,                      INTENT(IN)    :: KPROMA, KSIZE
    LOGICAL,                      INTENT(IN)    :: LDSOFT
    LOGICAL, DIMENSION(KPROMA),   INTENT(IN)    :: LDCOMPUTE
    REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRHODREF ! Reference density
    REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PT       ! Temperature
    REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PSSI     ! Supersaturation over ice
    REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRVT
    REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRCT     ! Cloud water m.r. at t
    REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRIT     ! Pristine ice m.r. at t
    REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRST     ! Snow/aggregate m.r. at t
    REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PRGT     ! Graupel/hail m.r. at t
    REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PLBDAS   ! Slope parameter of the aggregate distribution
    REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PLBDAG   ! Slope parameter of the graupel   distribution
    REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PAI      ! Thermodynamical function
    REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PCJ      ! Function to compute the ventilation coefficient
    REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PHLI_HCF !
    REAL, DIMENSION(KPROMA),      INTENT(IN)    :: PHLI_HRI !
    REAL, DIMENSION(SIZE(prhodref)),      INTENT(INOUT) :: PRCHONI  ! Homogeneous nucleation
    REAL, DIMENSION(SIZE(prhodref)),      INTENT(INOUT) :: PRVDEPS  ! Deposition on r_s
    REAL, DIMENSION(SIZE(prhodref)),      INTENT(INOUT) :: PRIAGGS  ! Aggregation on r_s
    REAL, DIMENSION(SIZE(prhodref)),      INTENT(INOUT) :: PRIAUTS  ! Autoconversion of r_i for r_s production
    REAL, DIMENSION(SIZE(prhodref)),      INTENT(INOUT) :: PRVDEPG  ! Deposition on r_g
    !
    !*       0.2  declaration of local variables
    !
    REAL, DIMENSION(KPROMA) :: ZCRIAUTI
    INTEGER                 :: JL
    !-------------------------------------------------------------------------------
    !
    !
    !*       3.2     compute the homogeneous nucleation source: RCHONI
    !
    DO JL=1, KSIZE
      IF(PT(JL)<XTT-35.0 .AND. PRCT(JL)>C_RTMIN .AND. LDCOMPUTE(JL)) THEN
        IF(.NOT. LDSOFT) THEN
          PRCHONI(JL) = MIN(1000.,XHON*PRHODREF(JL)*PRCT(JL)       &
                                     *EXP( XALPHA3*(PT(JL)-XTT)-XBETA3 ))
        ENDIF
      ELSE
        PRCHONI(JL) = 0.
      ENDIF
    ENDDO
    !
    !*       3.4    compute the deposition, aggregation and autoconversion sources
    !
    !*       3.4.3  compute the deposition on r_s: RVDEPS
    !
    DO JL=1, KSIZE
      IF(PRVT(JL)> V_RTMIN .AND. PRST(JL)> S_RTMIN .AND. LDCOMPUTE(JL)) THEN
        IF(.NOT. LDSOFT) THEN
    !!#ifdef REPRO48
          PRVDEPS(JL) = ( PSSI(JL)/(PRHODREF(JL)*PAI(JL)) ) *                               &
                     ( X0DEPS*PLBDAS(JL)**XEX0DEPS + X1DEPS*PCJ(JL)*PLBDAS(JL)**XEX1DEPS )
        ENDIF
      ELSE
        PRVDEPS(JL) = 0.
      ENDIF
    ENDDO
    !
    !*       3.4.4  compute the aggregation on r_s: RIAGGS
    !
    DO JL=1, KSIZE
      IF(PRIT(JL)>I_RTMIN .AND. PRST(JL)>S_RTMIN .AND. LDCOMPUTE(JL)) THEN
        IF(.NOT. LDSOFT) THEN
            !!#ifdef REPRO48
          PRIAGGS(JL) = XFIAGGS * EXP(XCOLEXIS*(PT(JL)- XTT) ) &
                             * PRIT(JL)                      &
                             * PLBDAS(JL)**XEXIAGGS          &
                             * PRHODREF(JL)**(-XCEXVT)
        ENDIF
      ELSE
        PRIAGGS(JL) = 0.
      ENDIF
    ENDDO
    !
    !*       3.4.5  compute the autoconversion of r_i for r_s production: RIAUTS
    !
    DO JL=1, KSIZE
      IF(PHLI_HRI(JL)>I_RTMIN .AND. LDCOMPUTE(JL)) THEN
        IF(.NOT. LDSOFT) THEN
          ZCRIAUTI(JL)=MIN(XCRIAUTI,10**(XACRIAUTI*(PT(JL)-XTT)+XBCRIAUTI))
          PRIAUTS(JL) = XTIMAUTI * EXP( XTEXAUTI*(PT(JL)-XTT) ) &
                                      * MAX(PHLI_HRI(JL)-ZCRIAUTI(JL)*PHLI_HCF(JL), 0.)
        ENDIF
      ELSE
        PRIAUTS(JL) = 0.
      ENDIF
    ENDDO
    !
    !*       3.4.6  compute the deposition on r_g: RVDEPG
    !
    !
    DO JL=1, KSIZE
      IF(PRVT(JL)>V_RTMIN .AND. PRGT(JL)>G_RTMIN .AND. LDCOMPUTE(JL)) THEN
        IF(.NOT. LDSOFT) THEN
          PRVDEPG(JL) = ( PSSI(JL)/(PRHODREF(JL)*PAI(JL)) ) *                               &
                     ( X0DEPG*PLBDAG(JL)**XEX0DEPG + X1DEPG*PCJ(JL)*PLBDAG(JL)**XEX1DEPG )
        ENDIF
      ELSE
        PRVDEPG(JL) = 0.
      ENDIF
    ENDDO
    !
    !
    END SUBROUTINE ICE4_SLOW
    END MODULE MODE_ICE4_SLOW

