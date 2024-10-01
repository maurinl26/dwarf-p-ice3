SUBROUTINE CONDENSATION(NIJT, NKT, NKTB, NKTE, NKB, NKE, NKL, NIJB, NIJE, &
    &XTT, XLVTT, XLSTT, XCPV, XCPD, XCI, XCL, XRD, XRV, XALPW, XBETAW, XGAMW, XALPI, XPI,  XBETAI, XGAMI, &
    &LSTATNW, LHGT_QS, &
    &XCRIAUTI, XACRIAUTI, XBCRIAUTI, XCRIAUTC, XFRMIN, &
    &HFRAC_ICE, HCONDENS, HLAMBDA3,                                                  &
    &PPABS, PZZ, PRHODREF, PT, PRV_IN, PRV_OUT, PRC_IN, PRC_OUT, PRI_IN, PRI_OUT,    &
    &PRR, PRS, PRG, PSIGS, PCLDFR, PSIGRC,                  &
    &OSIGMAS,                                                                &
    &PIFR, PSIGQSAT,                                 &
    &PLV, PLS, PCPH,                                                                 &
    &PHLC_HRC, PHLC_HCF, PHLI_HRI, PHLI_HCF,                                         &
    &PICE_CLD_WGT)
!   ################################################################################
!
!!
!!    PURPOSE
!!    -------
!!**  Routine to diagnose cloud fraction, liquid and ice condensate mixing ratios
!!    and s'rl'/sigs^2
!!
!!
!!**  METHOD
!!    ------
!!    Based on the large-scale fields of temperature, water vapor, and possibly
!!    liquid and solid condensate, the conserved quantities r_t and h_l are constructed
!!    and then fractional cloudiness, liquid and solid condensate is diagnosed.
!!
!!    The total variance is parameterized as the sum of  stratiform/turbulent variance
!!    and a convective variance.
!!    The turbulent variance is parameterized as a function of first-order moments, and
!!    the convective variance is modelled as a function of the convective mass flux
!!    (units kg/s m^2) as provided by the  mass flux convection scheme.
!!
!!    Nota: if the host model does not use prognostic values for liquid and solid condensate
!!    or does not provide a convective mass flux, put all these values to zero.
!!
!!
!!    EXTERNAL
!!    --------
!!      INI_CST
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!      Module MODD_CST       : contains physical constants
!!
!!    REFERENCE
!!    ---------
!!      Chaboureau J.P. and P. Bechtold (J. Atmos. Sci. 2002)
!!
!!    AUTHOR
!!    ------
!!      P. BECHTOLD       * Laboratoire d'Aerologie *
!!
!!    MODIFICATIONS
!!    -------------
!!      Original: 31.1.2002
!!     modified : 21.3.2002
!!     S.Malardel : 05.2006 : Correction sur le calcul de la fonction de
!!                                         Bougeault F2
!!     W. de Rooy: 06-06-2010: Modification in the statistical cloud scheme
!!                             more specifically adding a variance term
!!                             following ideas of Lenderink & Siebesma 2002
!!                             and adding a height dependence
!!     S. Riette, 18 May 2010 : PSIGQSAT is added
!!     S. Riette, 11 Oct 2011 : MIN function in PDF for continuity
!!                              modification of minimum value for Rc+Ri to create cloud and minimum value for sigma
!!                              Use of guess point as a starting point instead of liquid point
!!                              Better computation of ZCPH and dRsat/dT
!!                              Set ZCOND to zero if PCLDFR==0
!!                              Safety limitation to .99*Pressure for saturation vapour pressure
!!      2012-02 Y. Seity,  add possibility to run with reversed vertical levels
!!      2014-11 K.I Ivarsson add possibility to run with OCND2 option
!!      2016   S.Riette Change INQ1
!!      2016-11 S. Riette: use HFRAC_ICE, output adjusted state
!!      2018-02 K.I Ivarsson: Some modificatons of OCND2 option, mainly for optimation - new outputs
!!      2019-06 W.C. de Rooy: Mods for new set up statistical cloud scheme
!!      2019-07 K.I.Ivarsson: Switch for height dependent VQSIGSAT: LHGT_QS
!!      2020-12 U. Andrae : Introduce SPP for HARMONIE-AROME
!!     R. El Khatib 24-Aug-2021 Optimizations
!!      2021-01: SPP computations moved in aro_adjust (AROME/HARMONIE)
!-------------------------------------------------------------------------------
!
!*       0.    DECLARATIONS
!              ------------
!
! USE YOMHOOK , ONLY : LHOOK, DR_HOOK, JPHOOK
! USE MODD_DIMPHYEX,       ONLY: DIMPHYEX_t
! USE MODD_CST,            ONLY: CST_t
! USE MODD_RAIN_ICE_PARAM_n, ONLY: RAIN_ICE_PARAM_t
! USE MODD_NEB_n,          ONLY: NEB_t
! USE MODD_TURB_n,     ONLY: TURB_t
! USE MODE_TIWMX,          ONLY : ESATW, ESATI
! USE MODE_ICECLOUD,       ONLY : ICECLOUD

! USE COMPUTE_FRAC_ICE, ONLY : COMPUTE_FRAC_ICE
!
IMPLICIT NONE
!
!*       0.1   Declarations of dummy arguments :
!
!
! TYPE(DIMPHYEX_t),             INTENT(IN)    :: D
! TYPE(CST_t),                  INTENT(IN)    :: CST
! TYPE(RAIN_ICE_PARAM_t),       INTENT(IN)    :: ICEP
! TYPE(NEB_t),                  INTENT(IN)    :: NEBN
! TYPE(TURB_t),                 INTENT(IN)    :: TURBN
integer, intent(in) :: NIJT, NKT, NKTB, NKTE, NIJE, NIJB, NKL, NKB, NKE
real, intent(in) :: XTT, XLVTT, XLSTT, XCPV, XCPD, XCI, XCL, XRD, XRV, XALPW, XBETAW, XGAMW, XALPI, XPI, XBETAI, XGAMI
real, intent(in) :: XCRIAUTI, XACRIAUTI, XBCRIAUTI, XCRIAUTC
real, dimension(50), intent(in) :: XFRMIN
logical, intent(in) :: LSTATNW, LHGT_QS
CHARACTER(LEN=1),             INTENT(IN)    :: HFRAC_ICE
CHARACTER(LEN=4),             INTENT(IN)    :: HCONDENS
CHARACTER(LEN=*),             INTENT(IN)    :: HLAMBDA3 ! formulation for lambda3 coeff
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PPABS  ! pressure (Pa)
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PZZ    ! height of model levels (m)
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PRHODREF
REAL, DIMENSION(NIJT,NKT), INTENT(INOUT) :: PT     ! grid scale T  (K)
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PRV_IN ! grid scale water vapor mixing ratio (kg/kg) in input
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PRV_OUT! grid scale water vapor mixing ratio (kg/kg) in output
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PRC_IN ! grid scale r_c mixing ratio (kg/kg) in input
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PRC_OUT! grid scale r_c mixing ratio (kg/kg) in output
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PRI_IN ! grid scale r_i (kg/kg) in input
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PRI_OUT! grid scale r_i (kg/kg) in output
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PRR    ! grid scale mixing ration of rain (kg/kg)
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PRS    ! grid scale mixing ration of snow (kg/kg)
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PRG    ! grid scale mixing ration of graupel (kg/kg)
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PSIGS  ! Sigma_s from turbulence scheme
! LOGICAL,                                                       INTENT(IN)    ::  LMFCONV ! =SIZE(PMFCONV)!=0
! REAL, DIMENSION(MERGE(NIJT,0,LMFCONV),&
! MERGE(NKT,0,LMFCONV)),              INTENT(IN)    :: PMFCONV! convective mass flux (kg /s m^2)
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PCLDFR ! cloud fraction
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PSIGRC ! s r_c / sig_s^2

! LOGICAL, INTENT(IN)                         :: OUSERI ! logical switch to compute both
                               ! liquid and solid condensate (OUSERI=.TRUE.)
                               ! or only solid condensate (OUSERI=.FALSE.)
LOGICAL, INTENT(IN)                         :: OSIGMAS! use present global Sigma_s values
                               ! or that from turbulence scheme
! LOGICAL, INTENT(IN)                         :: OCND2  ! logical switch to sparate liquid and ice
                               ! more rigid (DEFALT value : .FALSE.)
! REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PICLDFR  ! ice cloud fraction
! REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PWCLDFR  ! water or mixed-phase cloud fraction
! REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PSSIO    ! Super-saturation with respect to ice in the
                                       ! supersaturated fraction
! REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PSSIU    ! Sub-saturation with respect to ice in the
                                       ! subsaturated fraction
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PIFR     ! Ratio cloud ice moist part
REAL, DIMENSION(NIJT),       INTENT(IN)    :: PSIGQSAT ! use an extra "qsat" variance contribution (OSIGMAS case)
                                       ! multiplied by PSIGQSAT

REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(IN)    :: PLV    ! Latent heat L_v
REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(IN)    :: PLS    ! Latent heat L_s
REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(IN)    :: PCPH   ! Specific heat C_ph
REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(OUT)   :: PHLC_HRC
REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(OUT)   :: PHLC_HCF ! cloud fraction
REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(OUT)   :: PHLI_HRI
REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(OUT)   :: PHLI_HCF
REAL, DIMENSION(NIJT),       OPTIONAL, INTENT(IN)    :: PICE_CLD_WGT
!
!
!*       0.2   Declarations of local variables :
!
INTEGER :: JIJ, JK, JKP, JKM                    ! loop index
INTEGER :: IKTB, IKTE, IKB, IKE, IKL, IIJB, IIJE
! REAL, DIMENSION(NIJT,NKT) :: ZTLK, 
real, dimension(nijt, nkt) ::ZRT     ! work arrays for T_l and total water mixing ratio
! REAL, DIMENSION(NIJT,NKT) :: ZL            ! length scale
! INTEGER, DIMENSION(NIJT)  :: ITPL            ! top levels of troposphere
! REAL,    DIMENSION(NIJT)  :: ZTMIN           ! minimum Temp. related to ITPL
!
REAL, DIMENSION(NIJT,NKT) :: ZLV, ZLS, ZCPD
REAL :: ZGCOND, ZAUTC, ZAUTI, ZGAUV, ZGAUC, ZGAUI, ZGAUTC, ZGAUTI, ZCRIAUTI   ! Used for Gaussian PDF integration
REAL :: ZLVS                                      ! thermodynamics
REAL, DIMENSION(NIJT) :: ZPV, ZPIV, ZQSL, ZQSI ! thermodynamics
! REAL :: ZLL, DZZ, ZZZ                           ! used for length scales
REAL :: ZAH
! real :: ZDRW, ZDTL, ZSIG_CONV                     ! related to computation of Sig_s
REAL, DIMENSION(NIJT) :: ZA, ZB, ZSBAR, ZSIGMA, ZQ1 ! related to computation of Sig_s
REAL, DIMENSION(NIJT) :: ZCOND
REAL, DIMENSION(NIJT) :: ZFRAC           ! Ice fraction
INTEGER  :: INQ1
REAL :: ZINC
! related to OCND2 noise check :
! REAL :: ZRSP,  ZRSW, ZRFRAC, ZRSDIF, ZRCOLD
! related to OCND2  ice cloud calulation :
! REAL, DIMENSION(NIJT) :: ESATW_T
! REAL :: ZDUM1,ZDUM2,ZDUM3,ZDUM4,ZLWINC,
real :: ZPRIFACT
! REAL, DIMENSION(NIJT) :: ZDZ, ZARDUM, 
real, dimension(nijt) :: ZARDUM2, ZCLDINI
! end OCND2

! LHGT_QS:
REAL :: ZDZFACT,ZDZREF
! LHGT_QS END

! REAL(KIND=JPHOOK) :: ZHOOK_HANDLE
! INTEGER :: IERR
!
!
!*       0.3  Definition of constants :
!
!-------------------------------------------------------------------------------
!
REAL,PARAMETER :: ZL0     = 600.        ! tropospheric length scale
REAL,PARAMETER :: ZCSIGMA = 0.2         ! constant in sigma_s parameterization
REAL,PARAMETER :: ZCSIG_CONV = 0.30E-2  ! scaling factor for ZSIG_CONV as function of mass flux
!

REAL, DIMENSION(-22:11),PARAMETER :: ZSRC_1D =(/                         &
0.           ,  0.           ,  2.0094444E-04,   0.316670E-03,    &
4.9965648E-04,  0.785956E-03 ,  1.2341294E-03,   0.193327E-02,    &
3.0190963E-03,  0.470144E-02 ,  7.2950651E-03,   0.112759E-01,    &
1.7350994E-02,  0.265640E-01 ,  4.0427860E-02,   0.610997E-01,    &
9.1578111E-02,  0.135888E+00 ,  0.1991484    ,   0.230756E+00,    &
0.2850565    ,  0.375050E+00 ,  0.5000000    ,   0.691489E+00,    &
0.8413813    ,  0.933222E+00 ,  0.9772662    ,   0.993797E+00,    &
0.9986521    ,  0.999768E+00 ,  0.9999684    ,   0.999997E+00,    &
1.0000000    ,  1.000000     /)
!
!-------------------------------------------------------------------------------
!
!
! IF (LHOOK) CALL DR_HOOK('CONDENSATION',0,ZHOOK_HANDLE)
!
IKTB=NKTB
IKTE=NKTE
IKB=NKB
IKE=NKE
IKL=NKL
IIJB=NIJB
IIJE=NIJE
!
PCLDFR(:,:) = 0. ! Initialize values
PSIGRC(:,:) = 0. ! Initialize values
PRV_OUT(:,:)= 0. ! Initialize values
PRC_OUT(:,:)= 0. ! Initialize values
PRI_OUT(:,:)= 0. ! Initialize values
ZPRIFACT = 1.    ! Initialize value
ZARDUM2 = 0.  ! Initialize values
ZCLDINI = -1. ! Dummy Initialized cloud input to icecloud routine
PIFR = 10. ! ratio of cloud ice water mixing ratio wet to dry
! part of a gridbox
ZDZREF = XFRMIN(25) ! Thickness for unchanged vqsigsat (only used for LHGT_QS)
!
! IF(OCND2)ZPRIFACT = 0. ! AROME OCND2 = FALSE
!
!
!-------------------------------------------------------------------------------
! store total water mixing ratio
DO JK=IKTB,IKTE
DO JIJ=IIJB,IIJE
ZRT(JIJ,JK)  = PRV_IN(JIJ,JK) + PRC_IN(JIJ,JK) + PRI_IN(JIJ,JK)*ZPRIFACT
END DO
END DO
!-------------------------------------------------------------------------------
! Preliminary calculations
! latent heat of vaporisation/sublimation
IF(PRESENT(PLV) .AND. PRESENT(PLS)) THEN
    ZLV(:,:)=PLV(:,:)
    ZLS(:,:)=PLS(:,:)
ELSE
    DO JK=IKTB,IKTE
        DO JIJ=IIJB,IIJE
        ! latent heat of vaporisation/sublimation
            ZLV(JIJ,JK) = XLVTT + ( XCPV - XCL ) * ( PT(JIJ,JK) - XTT )
            ZLS(JIJ,JK) = XLSTT + ( XCPV - XCI ) * ( PT(JIJ,JK) - XTT )
        ENDDO
    ENDDO
ENDIF
IF(PRESENT(PCPH)) THEN
    ZCPD(:,:)=PCPH(:,:)
ELSE
    DO JK=IKTB,IKTE
        DO JIJ=IIJB,IIJE
            ZCPD(JIJ,JK) = XCPD + XCPV*PRV_IN(JIJ,JK) + XCL*PRC_IN(JIJ,JK) + XCI*PRI_IN(JIJ,JK) + &
           XCL*PRR(JIJ,JK) +  &
           XCI*(PRS(JIJ,JK) + PRG(JIJ,JK) )
        ENDDO
    ENDDO
ENDIF
! Preliminary calculations needed for computing the "turbulent part" of Sigma_s
! IF ( .NOT. OSIGMAS ) THEN
!     DO JK=IKTB,IKTE
!         DO JIJ=IIJB,IIJE
!             ! store temperature at saturation
!             ZTLK(JIJ,JK) = PT(JIJ,JK) - ZLV(JIJ,JK)*PRC_IN(JIJ,JK)/ZCPD(JIJ,JK) &
!              - ZLS(JIJ,JK)*PRI_IN(JIJ,JK)/ZCPD(JIJ,JK)*ZPRIFACT
!         END DO
!     END DO
!     ! Determine tropopause/inversion  height from minimum temperature
!     ITPL(:)  = IKB+IKL
!     ZTMIN(:) = 400.
!     DO JK = IKTB+1,IKTE-1
!         DO JIJ=IIJB,IIJE
!             IF ( PT(JIJ,JK) < ZTMIN(JIJ) ) THEN
!                 ZTMIN(JIJ) = PT(JIJ,JK)
!                 ITPL(JIJ) = JK
!             ENDIF
!         END DO
!     END DO
!     ! Set the mixing length scale
!     ZL(:,IKB) = 20.
!     DO JK = IKB+IKL,IKE,IKL
!         DO JIJ=IIJB,IIJE
!         ! free troposphere
!             ZL(JIJ,JK) = ZL0
!             ZZZ = PZZ(JIJ,JK) -  PZZ(JIJ,IKB)
!             JKP = ITPL(JIJ)
!         ! approximate length for boundary-layer
!         IF ( ZL0 > ZZZ ) ZL(JIJ,JK) = ZZZ
!         ! gradual decrease of length-scale near and above tropopause
!         IF ( ZZZ > 0.9*(PZZ(JIJ,JKP)-PZZ(JIJ,IKB)) ) &
!             ZL(JIJ,JK) = .6 * ZL(JIJ,JK-IKL)
!         END DO
!     END DO
! END IF
!-------------------------------------------------------------------------------
!
DO JK=IKTB,IKTE
    JKP=MAX(MIN(JK+IKL,IKTE),IKTB)
    JKM=MAX(MIN(JK-IKL,IKTE),IKTB)
    ! IF (OCND2) THEN
    !     DO JIJ = IIJB, IIJE
    !         ZDZ(JIJ) = PZZ(JIJ,JKP) - PZZ(JIJ,JKP-IKL)
    !     ENDDO
    !     CALL ICECLOUD(D,PPABS(:,JK),PZZ(:,JK),ZDZ(:), &
    !         & PT(:,JK),PRV_IN(:,JK),1.,-1., &
    !         & ZCLDINI(:),PIFR(IIJB,JK),PICLDFR(:,JK), &
    !         & PSSIO(:,JK),PSSIU(:,JK),ZARDUM2(:),ZARDUM(:))
    !         ! latent heats
    !         ! saturated water vapor mixing ratio over liquid water and ice
    !     DO JIJ=IIJB,IIJE
    !         ESATW_T(JIJ)=ESATW(PT(JIJ,JK))
    !         ZPV(JIJ)  = MIN(ESATW_T(JIJ), .99*PPABS(JIJ,JK))
    !         ZPIV(JIJ) = MIN(ESATI(PT(JIJ,JK)), .99*PPABS(JIJ,JK))
    !     END DO
    ! ELSE
    ! latent heats
    ! saturated water vapor mixing ratio over liquid water and ice
        DO JIJ=IIJB,IIJE
            ZPV(JIJ)  = MIN(EXP( XALPW - XBETAW / PT(JIJ,JK) - XGAMW * LOG( PT(JIJ,JK) ) ), .99*PPABS(JIJ,JK))
            ZPIV(JIJ) = MIN(EXP( XALPI - XBETAI / PT(JIJ,JK) - XGAMI * LOG( PT(JIJ,JK) ) ), .99*PPABS(JIJ,JK))
        END DO
    ! ENDIF
!Ice fraction
ZFRAC(:) = 0.
! IF (OUSERI .AND. .NOT.OCND2) THEN
!     DO JIJ=IIJB,IIJE
!         IF (PRC_IN(JIJ,JK)+PRI_IN(JIJ,JK) > 1.E-20) THEN
!             ZFRAC(JIJ) = PRI_IN(JIJ,JK) / (PRC_IN(JIJ,JK)+PRI_IN(JIJ,JK))
!         ENDIF
!     END DO
!     DO JIJ=IIJB,IIJE
!         CALL COMPUTE_FRAC_ICE(HFRAC_ICE, XTT, XTAMXMIX, XTMINMIX, ZFRAC(JIJ), PT(JIJ,JK), IERR) !error code IERR cannot be checked here to not break vectorization
!     ENDDO
! ENDIF
DO JIJ=IIJB,IIJE
    ZQSL(JIJ)   = XRD / XRV * ZPV(JIJ) / ( PPABS(JIJ,JK) - ZPV(JIJ) )
    ZQSI(JIJ)   = XRD / XRV * ZPIV(JIJ) / ( PPABS(JIJ,JK) - ZPIV(JIJ) )

    ! interpolate between liquid and solid as function of temperature
    ZQSL(JIJ) = (1. - ZFRAC(JIJ)) * ZQSL(JIJ) + ZFRAC(JIJ) * ZQSI(JIJ)
    ZLVS = (1. - ZFRAC(JIJ)) * ZLV(JIJ,JK) + &
    & ZFRAC(JIJ)      * ZLS(JIJ,JK)

    ! coefficients a and b
    ZAH  = ZLVS * ZQSL(JIJ) / ( XRV * PT(JIJ,JK)**2 ) * (XRV * ZQSL(JIJ) / XRD + 1.)
    ZA(JIJ)   = 1. / ( 1. + ZLVS/ZCPD(JIJ,JK) * ZAH )
    ZB(JIJ)   = ZAH * ZA(JIJ)
    ZSBAR(JIJ) = ZA(JIJ) * ( ZRT(JIJ,JK) - ZQSL(JIJ) + &
        & ZAH * ZLVS * (PRC_IN(JIJ,JK)+PRI_IN(JIJ,JK)*ZPRIFACT) / ZCPD(JIJ,JK))
END DO
! switch to take either present computed value of SIGMAS
! or that of Meso-NH turbulence scheme
IF ( OSIGMAS ) THEN
    DO JIJ=IIJB,IIJE
        IF (PSIGQSAT(JIJ)/=0.) THEN
            ZDZFACT = 1.
            ! IF(LHGT_QS .AND. JK+1 <= IKTE)THEN
            !     ZDZFACT= MAX(XFRMIN(23),MIN(XFRMIN(24),(PZZ(JIJ,JK) - PZZ(JIJ,JK+1))/ZDZREF))
            ! ELSEIF(LHGT_QS)THEN
                ZDZFACT= MAX(XFRMIN(23),MIN(XFRMIN(24),((PZZ(JIJ,JK-1) - PZZ(JIJ,JK)))*0.8/ZDZREF))
            ! ENDIF
            ! IF (LSTATNW) THEN
            !     ZSIGMA(JIJ) = SQRT((PSIGS(JIJ,JK))**2 + (PSIGQSAT(JIJ)*ZDZFACT*ZQSL(JIJ)*ZA(JIJ))**2)
            ! ELSE
                ZSIGMA(JIJ) = SQRT((2*PSIGS(JIJ,JK))**2 + (PSIGQSAT(JIJ)*ZQSL(JIJ)*ZA(JIJ))**2)         
            ! ENDIF
        ELSE
            ! IF (LSTATNW) THEN
            !     ZSIGMA(JIJ) = PSIGS(JIJ,JK)
            ! ELSE
                ZSIGMA(JIJ) = 2*PSIGS(JIJ,JK)
            ! ENDIF
        END IF
    END DO
! ELSE
!     DO JIJ=IIJB,IIJE
!         ! parameterize Sigma_s with first_order closure
!         DZZ    =  PZZ(JIJ,JKP) - PZZ(JIJ,JKM)
!         ZDRW   =  ZRT(JIJ,JKP) - ZRT(JIJ,JKM)
!         ZDTL   =  ZTLK(JIJ,JKP) - ZTLK(JIJ,JKM) + XG/ZCPD(JIJ,JK) * DZZ
!         ZLL = ZL(JIJ,JK)
!         ! standard deviation due to convection
!         ZSIG_CONV =0.
!         IF(LMFCONV) ZSIG_CONV = ZCSIG_CONV * PMFCONV(JIJ,JK) / ZA(JIJ)
!         ! zsigma should be of order 4.e-4 in lowest 5 km of atmosphere
!         ZSIGMA(JIJ) =  SQRT( MAX( 1.E-25, ZCSIGMA * ZCSIGMA * ZLL*ZLL/(DZZ*DZZ)*(&
!         ZA(JIJ)*ZA(JIJ)*ZDRW*ZDRW - 2.*ZA(JIJ)*ZB(JIJ)*ZDRW*ZDTL + ZB(JIJ)*ZB(JIJ)*ZDTL*ZDTL) + &
!         ZSIG_CONV * ZSIG_CONV ) )
!     END DO
END IF
DO JIJ=IIJB,IIJE
    ZSIGMA(JIJ)= MAX( 1.E-10, ZSIGMA(JIJ) )
    ! normalized saturation deficit
    ZQ1(JIJ)   = ZSBAR(JIJ)/ZSIGMA(JIJ)
END DO
IF(HCONDENS == 'GAUS') THEN
    DO JIJ=IIJB,IIJE
        ! Gaussian Probability Density Function around ZQ1
        ! Computation of ZG and ZGAM(=erf(ZG))
        ZGCOND = -ZQ1(JIJ)/SQRT(2.)

        !Approximation of erf function for Gaussian distribution
        ZGAUV = 1 - SIGN(1., ZGCOND) * SQRT(1-EXP(-4*ZGCOND**2/XPI))

        !Computation Cloud Fraction
        PCLDFR(JIJ,JK) = MAX( 0., MIN(1.,0.5*ZGAUV))

        !Computation of condensate
        ZCOND(JIJ) = (EXP(-ZGCOND**2)-ZGCOND*SQRT(XPI)*ZGAUV)*ZSIGMA(JIJ)/SQRT(2.*XPI)
        ZCOND(JIJ) = MAX(ZCOND(JIJ), 0.)

        PSIGRC(JIJ,JK) = PCLDFR(JIJ,JK)
    END DO
!Computation warm/cold Cloud Fraction and content in high water content part
IF(PRESENT(PHLC_HCF) .AND. PRESENT(PHLC_HRC))THEN
    DO JIJ=IIJB,IIJE
        IF(1-ZFRAC(JIJ) > 1.E-20)THEN
            ZAUTC = (ZSBAR(JIJ) - XCRIAUTC/(PRHODREF(JIJ,JK)*(1-ZFRAC(JIJ))))/ZSIGMA(JIJ)
            ZGAUTC = -ZAUTC/SQRT(2.)
            !Approximation of erf function for Gaussian distribution
            ZGAUC = 1 - SIGN(1., ZGAUTC) * SQRT(1-EXP(-4*ZGAUTC**2/XPI))
            PHLC_HCF(JIJ,JK) = MAX( 0., MIN(1.,0.5*ZGAUC))
            PHLC_HRC(JIJ,JK) = (1-ZFRAC(JIJ))*(EXP(-ZGAUTC**2)-ZGAUTC*SQRT(XPI)*ZGAUC)*ZSIGMA(JIJ)/SQRT(2.*XPI)
            PHLC_HRC(JIJ,JK) = PHLC_HRC(JIJ,JK) + XCRIAUTC/PRHODREF(JIJ,JK) * PHLC_HCF(JIJ,JK)
            PHLC_HRC(JIJ,JK) = MAX(PHLC_HRC(JIJ,JK), 0.)
        ELSE
            PHLC_HCF(JIJ,JK)=0.
            PHLC_HRC(JIJ,JK)=0.
        ENDIF
    END DO
ENDIF

IF(PRESENT(PHLI_HCF) .AND. PRESENT(PHLI_HRI))THEN
    DO JIJ=IIJB,IIJE
        IF(ZFRAC(JIJ) > 1.E-20)THEN
            ZCRIAUTI=MIN(XCRIAUTI,10**(XACRIAUTI*(PT(JIJ,JK)-XTT)+XBCRIAUTI))
            ZAUTI = (ZSBAR(JIJ) - ZCRIAUTI/ZFRAC(JIJ))/ZSIGMA(JIJ)
            ZGAUTI = -ZAUTI/SQRT(2.)
            !Approximation of erf function for Gaussian distribution
            ZGAUI = 1 - SIGN(1., ZGAUTI) * SQRT(1-EXP(-4*ZGAUTI**2/XPI))
            PHLI_HCF(JIJ,JK) = MAX( 0., MIN(1.,0.5*ZGAUI))
            PHLI_HRI(JIJ,JK) = ZFRAC(JIJ)*(EXP(-ZGAUTI**2)-ZGAUTI*SQRT(XPI)*ZGAUI)*ZSIGMA(JIJ)/SQRT(2.*XPI)
            PHLI_HRI(JIJ,JK) = PHLI_HRI(JIJ,JK) + ZCRIAUTI*PHLI_HCF(JIJ,JK)
            PHLI_HRI(JIJ,JK) = MAX(PHLI_HRI(JIJ,JK), 0.)
        ELSE
            PHLI_HCF(JIJ,JK)=0.
            PHLI_HRI(JIJ,JK)=0.
        ENDIF
    END DO
ENDIF

ELSEIF(HCONDENS == 'CB02')THEN ! AROME option
    DO JIJ=IIJB,IIJE
    !Total condensate
        IF (ZQ1(JIJ) > 0. .AND. ZQ1(JIJ) <= 2) THEN
            ZCOND(JIJ) = MIN(EXP(-1.)+.66*ZQ1(JIJ)+.086*ZQ1(JIJ)**2, 2.) ! We use the MIN function for continuity
        ELSE IF (ZQ1(JIJ) > 2.) THEN
            ZCOND(JIJ) = ZQ1(JIJ)
        ELSE
            ZCOND(JIJ) = EXP( 1.2*ZQ1(JIJ)-1. )
        ENDIF
            ZCOND(JIJ) = ZCOND(JIJ) * ZSIGMA(JIJ)

        !Cloud fraction
        IF (ZCOND(JIJ) < 1.E-12) THEN
            PCLDFR(JIJ,JK) = 0.
        ELSE
            PCLDFR(JIJ,JK) = MAX( 0., MIN(1.,0.5+0.36*ATAN(1.55*ZQ1(JIJ))) )
        ENDIF
        IF (PCLDFR(JIJ,JK)==0.) THEN
            ZCOND(JIJ)=0.
        ENDIF

        INQ1 = MIN( MAX(-22,FLOOR(MIN(100., MAX(-100., 2*ZQ1(JIJ)))) ), 10)  !inner min/max prevents sigfpe when 2*zq1 does not fit into an int
        ZINC = 2.*ZQ1(JIJ) - INQ1

        PSIGRC(JIJ,JK) =  MIN(1.,(1.-ZINC)*ZSRC_1D(INQ1)+ZINC*ZSRC_1D(INQ1+1))
    END DO
    IF(PRESENT(PHLC_HCF) .AND. PRESENT(PHLC_HRC))THEN
        PHLC_HCF(:,JK)=0.
        PHLC_HRC(:,JK)=0.
    ENDIF
    IF(PRESENT(PHLI_HCF) .AND. PRESENT(PHLI_HRI))THEN
        PHLI_HCF(:,JK)=0.
        PHLI_HRI(:,JK)=0.
    ENDIF
END IF !HCONDENS

IF(HLAMBDA3=='CB')THEN
    DO JIJ=IIJB,IIJE
        PSIGRC(JIJ,JK) = PSIGRC(JIJ,JK)* MIN( 3. , MAX(1.,1.-ZQ1(JIJ)) )
    END DO
END IF
END DO
!
!
! CONTAINS
! INCLUDE "compute_frac_ice.func.h"
!
END SUBROUTINE CONDENSATION
