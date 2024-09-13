!MNH_LIC Copyright 1996-2021 CNRS, Meteo-France and Universite Paul Sabatier
!MNH_LIC This is part of the Meso-NH software governed by the CeCILL-C licence
!MNH_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt
!MNH_LIC for details. version 1.
!-----------------------------------------------------------------
!     ##########################################################################
      SUBROUTINE ICE_ADJUST (KRR,   &
                            &LSUBG_COND, LOCND2, &
                            &NIJT, NKT, NKTB, NKTE, NIJB, NIJE, &
                            &HBUNAME,                                          &
                            &PTSTEP, PSIGQSAT,                                 &
                            &PRHODJ, PEXNREF, PRHODREF, PSIGS, LMFCONV, PMFCONV,&
                            &PPABST, PZZ,                                      &
                            &PEXN, PCF_MF, PRC_MF, PRI_MF,                     &
                            &PICLDFR, PWCLDFR, PSSIO, PSSIU, PIFR,             &
                            &PRV, PRC, PRVS, PRCS, PTH, PTHS,                  &
                            &OCOMPUTE_SRC, PSRCS, PCLDFR,                      &
                            &PRR, PRI, PRIS, PRS, PRG,    &
                            &PICE_CLD_WGT,                                     &
                            &PRH,                                              &
                            &POUT_RV, POUT_RC, POUT_RI, POUT_TH,               &
                            &PHLC_HRC, PHLC_HCF, PHLI_HRI, PHLI_HCF)

!
!
!!****  *ICE_ADJUST* -  compute the ajustment of water vapor in mixed-phase
!!                      clouds
!!
!!    PURPOSE
!!    -------
!!    The purpose of this routine is to compute the fast microphysical sources
!!    through a saturation ajustement procedure in case of mixed-phase clouds.
!!
!!
!!**  METHOD
!!    ------
!!    Langlois, Tellus, 1973 for the cloudless version.
!!    When cloud water is taken into account, refer to book 1 of the
!!    documentation.
!!
!!
!!
!!    EXTERNAL
!!    --------
!!      None
!!
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!      Module MODD_CST
!!         XP00               ! Reference pressure
!!         XMD,XMV            ! Molar mass of dry air and molar mass of vapor
!!         XRD,XRV            ! Gaz constant for dry air, gaz constant for vapor
!!         XCPD,XCPV          ! Cpd (dry air), Cpv (vapor)
!!         XCL                ! Cl (liquid)
!!         XCI                ! Ci (ice)
!!         XTT                ! Triple point temperature
!!         XLVTT              ! Vaporization heat constant
!!         XLSTT              ! Sublimation  heat constant
!!         XALPW,XBETAW,XGAMW ! Constants for saturation vapor over liquid
!!                            !  pressure  function
!!         XALPI,XBETAI,XGAMI ! Constants for saturation vapor over ice
!!                            !  pressure  function
!!      Module  MODD_CONF
!!         CCONF
!!      Module MODD_BUDGET:
!!         NBUMOD
!!         CBUTYPE
!!         LBU_RTH
!!         LBU_RRV
!!         LBU_RRC
!!         LBU_RRI
!!
!!
!!    REFERENCE
!!    ---------
!!      Book 1 and Book2 of documentation ( routine ICE_ADJUST )
!!      Langlois, Tellus, 1973
!!
!!    AUTHOR
!!    ------
!!      J.-P. Pinty    * Laboratoire d'Aerologie*
!!
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    06/12/96
!!      M. Tomasini 27/11/00 Change CND and DEP fct of the T instead of rc and ri
!!                           Avoid the sub- and super-saturation before the ajustment
!!                           Avoid rc>0 if T<T00 before the ajustment
!!      P Bechtold 12/02/02  change subgrid condensation
!!      JP Pinty   29/11/02  add ICE2 and IC4 cases
!!      (P. Jabouille) 27/05/04 safety test for case where esw/i(T)> pabs (~Z>40km)
!!      J.Pergaud and S.Malardel Add EDKF case
!!      S. Riette ice for EDKF
!!      2012-02 Y. Seity,  add possibility to run with reversed vertical levels
!!      J.Escobar : 15/09/2015 : WENO5 & JPHEXT <> 1
!!      2016-07 S. Riette: adjustement is now realized on state variables (PRV, PRC, PRI, PTH)
!!                         whereas tendencies are still applied on S variables.
!!                         This modification allows to call ice_adjust on T variable
!!                         or to call it on S variables
!!      2016-11 S. Riette: all-or-nothing adjustment now uses condensation
!  P. Wautelet 05/2016-04/2018: new data structures and calls for I/O
!!      2018-02 K.I.Ivarsson : More outputs for OCND2 option
!  P. Wautelet    02/2020: use the new data structures and subroutines for budgets
!!      2020-12 U. Andrae : Introduce SPP for HARMONIE-AROME
!!     R. El Khatib 24-Aug-2021 Optimizations
!!     R. El Khatib 24-Oct-2023 Re-vectorize ;-)
!!
!-------------------------------------------------------------------------------
!
!*       0.    DECLARATIONS
!              ------------
!
IMPLICIT NONE
!
!
!*       0.1   Declarations of dummy arguments :
!
integer, intent(in) :: NIJT, NKT, NKTB, NKTE, NIJB, NIJE
logical, intent(in) :: LSUBG_COND, LOCND2



INTEGER,                  INTENT(IN)    :: KRR      ! Number of moist variables
CHARACTER(LEN=4),         INTENT(IN)    :: HBUNAME  ! Name of the budget
REAL,                     INTENT(IN)   :: PTSTEP    ! Double Time step
                                                    ! (single if cold start)
REAL, DIMENSION(NIJT),       INTENT(IN)    :: PSIGQSAT  ! coeff applied to qsat variance contribution
!
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    ::  PRHODJ  ! Dry density * Jacobian
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    ::  PEXNREF ! Reference Exner function
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    ::  PRHODREF
!
REAL, DIMENSION(MERGE(NIJT,0, LSUBG_COND),&
                MERGE(NKT,0, LSUBG_COND)),           INTENT(IN)    ::  PSIGS   ! Sigma_s at time t
LOGICAL,                                              INTENT(IN)    ::  LMFCONV ! =SIZE(PMFCONV)!=0
REAL, DIMENSION(MERGE(NIJT,0,LMFCONV),&
                MERGE(NKT,0,LMFCONV)),              INTENT(IN)   ::  PMFCONV ! convective mass flux
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    ::  PPABST  ! Absolute Pressure at t
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    ::  PZZ     ! height of model layer
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    ::  PEXN    ! Exner function
!
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PCF_MF   ! Convective Mass Flux Cloud fraction
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PRC_MF   ! Convective Mass Flux liquid mixing ratio
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PRI_MF   ! Convective Mass Flux ice mixing ratio
!
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PRV     ! Water vapor m.r. to adjust
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PRC     ! Cloud water m.r. to adjust
REAL, DIMENSION(NIJT,NKT), INTENT(INOUT) :: PRVS    ! Water vapor m.r. source
REAL, DIMENSION(NIJT,NKT), INTENT(INOUT) :: PRCS    ! Cloud water m.r. source
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PTH     ! Theta to adjust
REAL, DIMENSION(NIJT,NKT), INTENT(INOUT) :: PTHS    ! Theta source
LOGICAL,                            INTENT(IN)    :: OCOMPUTE_SRC
REAL, DIMENSION(MERGE(NIJT,0,OCOMPUTE_SRC),&
                MERGE(NKT,0,OCOMPUTE_SRC)), INTENT(OUT)   :: PSRCS   ! Second-order flux
                                                                       ! s'rc'/2Sigma_s2 at time t+1
                                                                       ! multiplied by Lambda_3
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)  ::  PCLDFR  ! Cloud fraction
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)  ::  PICLDFR ! ice cloud fraction
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)  ::  PWCLDFR ! water or mixed-phase cloud fraction
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)  ::  PSSIO   ! Super-saturation with respect to ice in the
                                                        ! supersaturated fraction
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)  ::  PSSIU   ! Sub-saturation with respect to ice in the
                                                        ! subsaturated fraction
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)  ::  PIFR    ! Ratio cloud ice moist part to dry part
!
REAL, DIMENSION(NIJT,NKT), INTENT(INOUT)::  PRIS ! Cloud ice  m.r. at t+1
REAL, DIMENSION(NIJT,NKT), INTENT(IN)   ::  PRR  ! Rain water m.r. to adjust
REAL, DIMENSION(NIJT,NKT), INTENT(IN)   ::  PRI  ! Cloud ice  m.r. to adjust
REAL, DIMENSION(NIJT,NKT), INTENT(IN)   ::  PRS  ! Aggregate  m.r. to adjust
REAL, DIMENSION(NIJT,NKT), INTENT(IN)   ::  PRG  ! Graupel    m.r. to adjust
INTEGER,                                      INTENT(IN)   ::  KBUDGETS
REAL, DIMENSION(NIJT),       OPTIONAL, INTENT(IN)   ::  PICE_CLD_WGT
REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(IN)   ::  PRH  ! Hail       m.r. to adjust
REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(OUT)  ::  POUT_RV ! Adjusted value
REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(OUT)  ::  POUT_RC ! Adjusted value
REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(OUT)  ::  POUT_RI ! Adjusted value
REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(OUT)  ::  POUT_TH ! Adjusted value
REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(OUT)  ::  PHLC_HRC
REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(OUT)  ::  PHLC_HCF
REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(OUT)  ::  PHLI_HRI
REAL, DIMENSION(NIJT,NKT), OPTIONAL, INTENT(OUT)  ::  PHLI_HCF
!
!
!*       0.2   Declarations of local variables :
!
!
REAL  :: ZW1,ZW2    ! intermediate fields
REAL, DIMENSION(NIJT,NKT) &
                         :: ZT,   &  ! adjusted temperature
                   ZRV, ZRC, ZRI, &  ! adjusted state
                            ZCPH, &  ! guess of the CPh for the mixing
                            ZLV,  &  ! guess of the Lv at t+1
                            ZLS      ! guess of the Ls at t+1
REAL :: ZCRIAUT, & ! Autoconversion thresholds
        ZHCF, ZHR
!
INTEGER             :: JITER,ITERMAX ! iterative loop for first order adjustment
INTEGER             :: JIJ, JK
INTEGER :: IKTB, IKTE, IIJB, IIJE
!
REAL, DIMENSION(NIJT,NKT) :: ZSIGS, ZSRCS
REAL, DIMENSION(NIJT) :: ZSIGQSAT
LOGICAL :: LLNONE, LLTRIANGLE, LLHLC_H, LLHLI_H

!
!-------------------------------------------------------------------------------
!
!*       1.     PRELIMINARIES
!               -------------
!
!
IKTB=NKTB
IKTE=NKTE
IIJB=NIJB
IIJE=NIJE
!
ITERMAX=1
!
!-------------------------------------------------------------------------------
!
!*       2.     COMPUTE QUANTITIES WITH THE GUESS OF THE FUTURE INSTANT
!               -------------------------------------------------------
!
!
!    beginning of the iterative loop (to compute the adjusted state)
!
DO JITER =1,ITERMAX
  !
  !*       2.3    compute the latent heat of vaporization Lv(T*) at t+1
  !                   and the latent heat of sublimation  Ls(T*) at t+1
  !
  DO JK=IKTB,IKTE
    DO JIJ=IIJB,IIJE
      IF (JITER==1) ZT(JIJ,JK) = PTH(JIJ,JK) * PEXN(JIJ,JK)
      ZLV(JIJ,JK) = XLVTT + ( XCPV - XCL ) * ( ZT(JIJ,JK) -XTT )
      ZLS(JIJ,JK) = XLSTT + ( XCPV - XCI ) * ( ZT(JIJ,JK) -XTT )
    ENDDO
  ENDDO
  !
  !*       2.4   Iterate
  !
  IF (JITER==1) THEN
    ! compute with input values
    CALL ITERATION(PRV,PRC,PRI,ZRV,ZRC,ZRI)
  ELSE
    ! compute with updated values
    CALL ITERATION(ZRV,ZRC,ZRI,ZRV,ZRC,ZRI)
  ENDIF
ENDDO         ! end of the iterative loop
!
!
CONTAINS
SUBROUTINE ITERATION(PRV_IN,PRC_IN,PRI_IN,PRV_OUT,PRC_OUT,PRI_OUT)

REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: PRV_IN ! Water vapor m.r. to adjust in input
REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: PRC_IN ! Cloud water m.r. to adjust in input
REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: PRI_IN ! Cloud ice   m.r. to adjust in input
REAL, DIMENSION(NIJT,NKT), INTENT(OUT) :: PRV_OUT ! Water vapor m.r. to adjust in output
REAL, DIMENSION(NIJT,NKT), INTENT(OUT) :: PRC_OUT ! Cloud water m.r. to adjust in output
REAL, DIMENSION(NIJT,NKT), INTENT(OUT) :: PRI_OUT ! Cloud ice   m.r. to adjust in output
!
!*       2.4    compute the specific heat for moist air (Cph) at t+1
DO JK=IKTB,IKTE
  DO JIJ=IIJB,IIJE
    SELECT CASE(KRR)
      CASE(7)
        ZCPH(JIJ,JK) = XCPD + XCPV * PRV_IN(JIJ,JK)                             &
                                + XCL  * (PRC_IN(JIJ,JK) + PRR(JIJ,JK))             &
                                + XCI  * (PRI_IN(JIJ,JK) + PRS(JIJ,JK) + PRG(JIJ,JK) + PRH(JIJ,JK))
      CASE(6)
        ZCPH(JIJ,JK) = XCPD + XCPV * PRV_IN(JIJ,JK)                             &
                                + XCL  * (PRC_IN(JIJ,JK) + PRR(JIJ,JK))             &
                                + XCI  * (PRI_IN(JIJ,JK) + PRS(JIJ,JK) + PRG(JIJ,JK))
      CASE(5)
        ZCPH(JIJ,JK) = XCPD + XCPV * PRV_IN(JIJ,JK)                             &
                                + XCL  * (PRC_IN(JIJ,JK) + PRR(JIJ,JK))             &
                                + XCI  * (PRI_IN(JIJ,JK) + PRS(JIJ,JK))
      CASE(3)
        ZCPH(JIJ,JK) = XCPD + XCPV * PRV_IN(JIJ,JK)               &
                                + XCL  * (PRC_IN(JIJ,JK) + PRR(JIJ,JK))
      CASE(2)
        ZCPH(JIJ,JK) = XCPD + XCPV * PRV_IN(JIJ,JK) &
                                + XCL  * PRC_IN(JIJ,JK)
    END SELECT
  ENDDO
ENDDO
!
IF (LSUBG_COND ) THEN
  !
  !*       3.     SUBGRID CONDENSATION SCHEME
  !               ---------------------------
  !
  !   PSRC= s'rci'/Sigma_s^2
  !   ZT is INOUT
  CALL CONDENSATION(LVTT, XLSTT, XCPV, XCL, XCI, XTT, XCPD, XALPW, XALPI, XRD, XRV, &
        & XCRIAUTI, XCRIAUTC, XACRIAUTI, XBCRIAUTI, &
        & LHGT_QS, LSTATNW, &
        & NKB,NKE,NKL,NIJB,NIJE, &
        & XFRMIN, &                             &
       PPABST, PZZ, PRHODREF, ZT, PRV_IN, PRV_OUT, PRC_IN, PRC_OUT, PRI_IN, PRI_OUT, &
       PRR, PRS, PRG, PSIGS, LMFCONV, PMFCONV, PCLDFR, &
       PSRCS, .TRUE., NEBN%LSIGMAS, LOCND2,                       &
       PICLDFR, PWCLDFR, PSSIO, PSSIU, PIFR, PSIGQSAT,                   &
       PLV=ZLV, PLS=ZLS, PCPH=ZCPH,                                      &
       PHLC_HRC=PHLC_HRC, PHLC_HCF=PHLC_HCF, PHLI_HRI=PHLI_HRI, PHLI_HCF=PHLI_HCF,&
       PICE_CLD_WGT=PICE_CLD_WGT)
ELSE
  !
  !*       4.     ALL OR NOTHING CONDENSATION SCHEME
  !                            FOR MIXED-PHASE CLOUD
  !               -----------------------------------------------
  !
  ZSIGS(:,:)=0.
  ZSIGQSAT(:)=0.
  !We use ZSRCS because in Méso-NH, PSRCS can be a zero-length array in this case
  !ZT is INOUT
  CALL CONDENSATION(LVTT, XLSTT, XCPV, XCL, XCI, XTT, XCPD, XALPW, XALPI, XRD, XRV, &
        & XCRIAUTI, XCRIAUTC, XACRIAUTI, XBCRIAUTI, &
        & LHGT_QS, LSTATNW, &
        & NKB,NKE,NKL,NIJB,NIJE, &
        & XFRMIN, &
        &HFRAC_ICE, HCONDENS, HLAMBDA3,                             &
       PPABST, PZZ, PRHODREF, ZT, PRV_IN, PRV_OUT, PRC_IN, PRC_OUT, PRI_IN, PRI_OUT, &
       PRR, PRS, PRG, ZSIGS, LMFCONV, PMFCONV, PCLDFR, &
       ZSRCS, .TRUE., .TRUE., LOCND2,                             &
       PICLDFR, PWCLDFR, PSSIO, PSSIU, PIFR, ZSIGQSAT,                   &
       PLV=ZLV, PLS=ZLS, PCPH=ZCPH,                                      &
       PHLC_HRC=PHLC_HRC, PHLC_HCF=PHLC_HCF, PHLI_HRI=PHLI_HRI, PHLI_HCF=PHLI_HCF,&
       PICE_CLD_WGT=PICE_CLD_WGT)
ENDIF

END SUBROUTINE ITERATION

SUBROUTINE CONDENSATION(XLVTT, XLSTT, XCPV, XCL, XCI, XTT, XCPD, XALPW, XALPI, XRD, XRV, &
    & XCRIAUTI, XCRIAUTC, XACRIAUTI, XBCRIAUTI, &
    & LHGT_QS, LSTATNW, &
    & NKB,NKE,NKL,NIJB,NIJE, &
    & XFRMIN, &
    & XTMAXMIX, XTMINMIX, &
    &HFRAC_ICE, HCONDENS, HLAMBDA3,                                                  &
    &PPABS, PZZ, PRHODREF, PT, PRV_IN, PRV_OUT, PRC_IN, PRC_OUT, PRI_IN, PRI_OUT,    &
    &PRR, PRS, PRG, PSIGS, LMFCONV, PMFCONV, PCLDFR, PSIGRC, OUSERI,                 &
    &OSIGMAS, OCND2,                                                                 &
    &PICLDFR, PWCLDFR, PSSIO, PSSIU, PIFR, PSIGQSAT,                                 &
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

real, dimension(50), intent(in) :: XFRMIN
real, intent(in) :: XLVTT, XLSTT, XCPV, XCL, XCI, XTT, XCPD, XALPW, XALPI, XRD, XRV ! cst
real, intent(in) :: XCRIAUTI, XCRIAUTC, XACRIAUTI, XBCRIAUTI ! ICEP
logical, intent(in) :: LHGT_QS, LSTATNW
real, intent(in) :: NKB,NKE,NKL,NIJB,NIJE
real, intent(in) :: XTMAXMIX, XTMINMIX



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
LOGICAL,                                                       INTENT(IN)    ::  LMFCONV ! =SIZE(PMFCONV)!=0
REAL, DIMENSION(MERGE(NIJT,0,LMFCONV),&
MERGE(NKT,0,LMFCONV)),              INTENT(IN)    :: PMFCONV! convective mass flux (kg /s m^2)
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PCLDFR ! cloud fraction
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PSIGRC ! s r_c / sig_s^2

LOGICAL, INTENT(IN)                         :: OUSERI ! logical switch to compute both
                               ! liquid and solid condensate (OUSERI=.TRUE.)
                               ! or only solid condensate (OUSERI=.FALSE.)
LOGICAL, INTENT(IN)                         :: OSIGMAS! use present global Sigma_s values
                               ! or that from turbulence scheme
LOGICAL, INTENT(IN)                         :: OCND2  ! logical switch to sparate liquid and ice
                               ! more rigid (DEFALT value : .FALSE.)
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PICLDFR  ! ice cloud fraction
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PWCLDFR  ! water or mixed-phase cloud fraction
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PSSIO    ! Super-saturation with respect to ice in the
                                       ! supersaturated fraction
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)   :: PSSIU    ! Sub-saturation with respect to ice in the
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
REAL, DIMENSION(NIJT,NKT) :: ZTLK, ZRT     ! work arrays for T_l and total water mixing ratio
REAL, DIMENSION(NIJT,NKT) :: ZL            ! length scale
INTEGER, DIMENSION(NIJT)  :: ITPL            ! top levels of troposphere
REAL,    DIMENSION(NIJT)  :: ZTMIN           ! minimum Temp. related to ITPL
!
REAL, DIMENSION(NIJT,NKT) :: ZLV, ZLS, ZCPD
REAL :: ZGCOND, ZAUTC, ZAUTI, ZGAUV, ZGAUC, ZGAUI, ZGAUTC, ZGAUTI, ZCRIAUTI   ! Used for Gaussian PDF integration
REAL :: ZLVS                                      ! thermodynamics
REAL, DIMENSION(NIJT) :: ZPV, ZPIV, ZQSL, ZQSI ! thermodynamics
REAL :: ZLL, DZZ, ZZZ                           ! used for length scales
REAL :: ZAH, ZDRW, ZDTL, ZSIG_CONV                     ! related to computation of Sig_s
REAL, DIMENSION(NIJT) :: ZA, ZB, ZSBAR, ZSIGMA, ZQ1 ! related to computation of Sig_s
REAL, DIMENSION(NIJT) :: ZCOND
REAL, DIMENSION(NIJT) :: ZFRAC           ! Ice fraction
INTEGER  :: INQ1
REAL :: ZINC
! related to OCND2 noise check :
REAL :: ZRSP,  ZRSW, ZRFRAC, ZRSDIF, ZRCOLD
! related to OCND2  ice cloud calulation :
REAL, DIMENSION(NIJT) :: ESATW_T
REAL :: ZDUM1,ZDUM2,ZDUM3,ZDUM4,ZPRIFACT,ZLWINC
REAL, DIMENSION(NIJT) :: ZDZ, ZARDUM, ZARDUM2, ZCLDINI
! end OCND2

! LHGT_QS:
REAL :: ZDZFACT,ZDZREF
! LHGT_QS END

INTEGER :: IERR
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
IF(OCND2)ZPRIFACT = 0.
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
IF ( .NOT. OSIGMAS ) THEN
DO JK=IKTB,IKTE
DO JIJ=IIJB,IIJE
! store temperature at saturation
ZTLK(JIJ,JK) = PT(JIJ,JK) - ZLV(JIJ,JK)*PRC_IN(JIJ,JK)/ZCPD(JIJ,JK) &
             - ZLS(JIJ,JK)*PRI_IN(JIJ,JK)/ZCPD(JIJ,JK)*ZPRIFACT
END DO
END DO
! Determine tropopause/inversion  height from minimum temperature
ITPL(:)  = IKB+IKL
ZTMIN(:) = 400.
DO JK = IKTB+1,IKTE-1
DO JIJ=IIJB,IIJE
IF ( PT(JIJ,JK) < ZTMIN(JIJ) ) THEN
ZTMIN(JIJ) = PT(JIJ,JK)
ITPL(JIJ) = JK
ENDIF
END DO
END DO
! Set the mixing length scale
ZL(:,IKB) = 20.
DO JK = IKB+IKL,IKE,IKL
DO JIJ=IIJB,IIJE
! free troposphere
ZL(JIJ,JK) = ZL0
ZZZ =  PZZ(JIJ,JK) -  PZZ(JIJ,IKB)
JKP = ITPL(JIJ)
! approximate length for boundary-layer
IF ( ZL0 > ZZZ ) ZL(JIJ,JK) = ZZZ
! gradual decrease of length-scale near and above tropopause
IF ( ZZZ > 0.9*(PZZ(JIJ,JKP)-PZZ(JIJ,IKB)) ) &
ZL(JIJ,JK) = .6 * ZL(JIJ,JK-IKL)
END DO
END DO
END IF
!-------------------------------------------------------------------------------
!
DO JK=IKTB,IKTE
JKP=MAX(MIN(JK+IKL,IKTE),IKTB)
JKM=MAX(MIN(JK-IKL,IKTE),IKTB)
IF (OCND2) THEN
DO JIJ = IIJB, IIJE
ZDZ(JIJ) = PZZ(JIJ,JKP) - PZZ(JIJ,JKP-IKL)
ENDDO
CALL ICECLOUD(D,PPABS(:,JK),PZZ(:,JK),ZDZ(:), &
& PT(:,JK),PRV_IN(:,JK),1.,-1., &
& ZCLDINI(:),PIFR(IIJB,JK),PICLDFR(:,JK), &
& PSSIO(:,JK),PSSIU(:,JK),ZARDUM2(:),ZARDUM(:))
! latent heats
! saturated water vapor mixing ratio over liquid water and ice
DO JIJ=IIJB,IIJE
ESATW_T(JIJ)=ESATW(PT(JIJ,JK))
ZPV(JIJ)  = MIN(ESATW_T(JIJ), .99*PPABS(JIJ,JK))
ZPIV(JIJ) = MIN(ESATI(PT(JIJ,JK)), .99*PPABS(JIJ,JK))
END DO
ELSE
! latent heats
! saturated water vapor mixing ratio over liquid water and ice
DO JIJ=IIJB,IIJE
ZPV(JIJ)  = MIN(EXP( XALPW - XBETAW / PT(JIJ,JK) - XGAMW * LOG( PT(JIJ,JK) ) ), .99*PPABS(JIJ,JK))
ZPIV(JIJ) = MIN(EXP( XALPI - XBETAI / PT(JIJ,JK) - XGAMI * LOG( PT(JIJ,JK) ) ), .99*PPABS(JIJ,JK))
END DO
ENDIF
!Ice fraction
ZFRAC(:) = 0.
IF (OUSERI .AND. .NOT.OCND2) THEN
DO JIJ=IIJB,IIJE
IF (PRC_IN(JIJ,JK)+PRI_IN(JIJ,JK) > 1.E-20) THEN
ZFRAC(JIJ) = PRI_IN(JIJ,JK) / (PRC_IN(JIJ,JK)+PRI_IN(JIJ,JK))
ENDIF
END DO
DO JIJ=IIJB,IIJE
CALL COMPUTE_FRAC_ICE(XTT, XTMAXMIX, XTMINMIX, HFRAC_ICE, ZFRAC(JIJ), PT(JIJ,JK), IERR) !error code IERR cannot be checked here to not break vectorization
ENDDO
ENDIF
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
IF(LHGT_QS .AND. JK+1 <= IKTE)THEN
ZDZFACT= MAX(XFRMIN(23),MIN(XFRMIN(24),(PZZ(JIJ,JK) - PZZ(JIJ,JK+1))/ZDZREF))
ELSEIF(LHGT_QS)THEN
ZDZFACT= MAX(XFRMIN(23),MIN(XFRMIN(24),((PZZ(JIJ,JK-1) - PZZ(JIJ,JK)))*0.8/ZDZREF))
ENDIF
IF (LSTATNW) THEN
ZSIGMA(JIJ) = SQRT((PSIGS(JIJ,JK))**2 + (PSIGQSAT(JIJ)*ZDZFACT*ZQSL(JIJ)*ZA(JIJ))**2)
ELSE
ZSIGMA(JIJ) = SQRT((2*PSIGS(JIJ,JK))**2 + (PSIGQSAT(JIJ)*ZQSL(JIJ)*ZA(JIJ))**2)
ENDIF
ELSE
IF (LSTATNW) THEN
ZSIGMA(JIJ) = PSIGS(JIJ,JK)
ELSE
ZSIGMA(JIJ) = 2*PSIGS(JIJ,JK)
ENDIF
END IF
END DO
ELSE
DO JIJ=IIJB,IIJE
! parameterize Sigma_s with first_order closure
DZZ    =  PZZ(JIJ,JKP) - PZZ(JIJ,JKM)
ZDRW   =  ZRT(JIJ,JKP) - ZRT(JIJ,JKM)
ZDTL   =  ZTLK(JIJ,JKP) - ZTLK(JIJ,JKM) + XG/ZCPD(JIJ,JK) * DZZ
ZLL = ZL(JIJ,JK)
! standard deviation due to convection
ZSIG_CONV =0.
IF(LMFCONV) ZSIG_CONV = ZCSIG_CONV * PMFCONV(JIJ,JK) / ZA(JIJ)
! zsigma should be of order 4.e-4 in lowest 5 km of atmosphere
ZSIGMA(JIJ) =  SQRT( MAX( 1.E-25, ZCSIGMA * ZCSIGMA * ZLL*ZLL/(DZZ*DZZ)*(&
ZA(JIJ)*ZA(JIJ)*ZDRW*ZDRW - 2.*ZA(JIJ)*ZB(JIJ)*ZDRW*ZDTL + ZB(JIJ)*ZB(JIJ)*ZDTL*ZDTL) + &
ZSIG_CONV * ZSIG_CONV ) )
END DO
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

ELSEIF(HCONDENS == 'CB02')THEN
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

IF(.NOT. OCND2) THEN
DO JIJ=IIJB,IIJE
PRC_OUT(JIJ,JK) = (1.-ZFRAC(JIJ)) * ZCOND(JIJ) ! liquid condensate
PRI_OUT(JIJ,JK) = ZFRAC(JIJ) * ZCOND(JIJ)   ! solid condensate
PT(JIJ,JK) = PT(JIJ,JK) + ((PRC_OUT(JIJ,JK)-PRC_IN(JIJ,JK))*ZLV(JIJ,JK) + &
             &(PRI_OUT(JIJ,JK)-PRI_IN(JIJ,JK))*ZLS(JIJ,JK)   ) &
           & /ZCPD(JIJ,JK)
PRV_OUT(JIJ,JK) = ZRT(JIJ,JK) - PRC_OUT(JIJ,JK) - PRI_OUT(JIJ,JK)*ZPRIFACT
END DO
ELSE
DO JIJ=IIJB,IIJE
PRC_OUT(JIJ,JK) = (1.-ZFRAC(JIJ)) * ZCOND(JIJ) ! liquid condensate
ZLWINC = PRC_OUT(JIJ,JK) - PRC_IN(JIJ,JK)
!
!     This check is mainly for noise reduction :
!     -------------------------
IF(ABS(ZLWINC)>1.0E-12  .AND.  ESATW(PT(JIJ,JK)) < PPABS(JIJ,JK)*0.5 )THEN
ZRCOLD = PRC_OUT(JIJ,JK)
ZRFRAC = PRV_IN(JIJ,JK) - ZLWINC
IF( PRV_IN(JIJ,JK) < ZRSW )THEN ! sub - saturation over water:
! Avoid drying of cloudwater leading to supersaturation with
! respect to water
ZRSDIF= MIN(0.,ZRSP-ZRFRAC)
ELSE  ! super - saturation over water:
! Avoid deposition of water leading to sub-saturation with
! respect to water
!            ZRSDIF= MAX(0.,ZRSP-ZRFRAC)
ZRSDIF= 0. ! t7
ENDIF
PRC_OUT(JIJ,JK) = ZCOND(JIJ)  - ZRSDIF
ELSE
ZRCOLD = PRC_IN(JIJ,JK)
ENDIF
!    end check

!    compute separate ice cloud:
PWCLDFR(JIJ,JK) = PCLDFR(JIJ,JK)
ZDUM1 = MIN(1.0,20.* PRC_OUT(JIJ,JK)*SQRT(ZDZ(JIJ))/ZQSL(JIJ)) ! cloud liquid water factor
ZDUM3 = MAX(0.,PICLDFR(JIJ,JK)-PWCLDFR(JIJ,JK)) ! pure ice cloud part
IF (JK==IKTB) THEN
ZDUM4 = PRI_IN(JIJ,JK)
ELSE
ZDUM4 = PRI_IN(JIJ,JK) + PRS(JIJ,JK)*0.5 + PRG(JIJ,JK)*0.25
ENDIF

ZDUM4 = MAX(0.,MIN(1.,PICE_CLD_WGT(JIJ)*ZDUM4*SQRT(ZDZ(JIJ))/ZQSI(JIJ))) ! clould ice+solid
                                  ! precip. water factor

ZDUM2 = (0.8*PCLDFR(JIJ,JK)+0.2)*MIN(1.,ZDUM1 + ZDUM4*PCLDFR(JIJ,JK))
! water cloud, use 'statistical' cloud, but reduce it in case of low liquid content

PCLDFR(JIJ,JK) = MIN(1., ZDUM2 + (0.5*ZDUM3+0.5)*ZDUM4) ! Rad cloud
! Reduce ice cloud part in case of low ice water content
PRI_OUT(JIJ,JK) = PRI_IN(JIJ,JK)
PT(JIJ,JK) = PT(JIJ,JK) + ((PRC_OUT(JIJ,JK)-ZRCOLD)*ZLV(JIJ,JK) + &
             &(PRI_OUT(JIJ,JK)-PRI_IN(JIJ,JK))*ZLS(JIJ,JK)   ) &
           & /ZCPD(JIJ,JK)
PRV_OUT(JIJ,JK) = ZRT(JIJ,JK) - PRC_OUT(JIJ,JK) - PRI_OUT(JIJ,JK)*ZPRIFACT
END DO
END IF ! End OCND2
IF(HLAMBDA3=='CB')THEN
DO JIJ=IIJB,IIJE
! s r_c/ sig_s^2
!    PSIGRC(JIJ,JK) = PCLDFR(JIJ,JK)  ! use simple Gaussian relation
!
!    multiply PSRCS by the lambda3 coefficient
!
!      PSIGRC(JIJ,JK) = 2.*PCLDFR(JIJ,JK) * MIN( 3. , MAX(1.,1.-ZQ1(JIJ)) )
! in the 3D case lambda_3 = 1.

PSIGRC(JIJ,JK) = PSIGRC(JIJ,JK)* MIN( 3. , MAX(1.,1.-ZQ1(JIJ)) )
END DO
END IF
END DO
!
!
!
END SUBROUTINE CONDENSATION


SUBROUTINE COMPUTE_FRAC_ICE(XTT,XTMAXMIX, XTMINMIX, HFRAC_ICE, PFRAC_ICE,PT,KERR)

    ! ******* TO BE INCLUDED IN THE *CONTAINS* OF A SUBROUTINE, IN ORDER TO EASE AUTOMATIC INLINING ******
    ! => Don't use drHook !!!
    !
    !!****  *COMPUTE_FRAC_ICE* - computes ice fraction
    !
    !!    AUTHOR
    !!    ------
    !!      Julien PERGAUD      * Meteo-France *
    !!
    !!    MODIFICATIONS
    !!    -------------
    !!      Original         13/03/06
    !!      S. Riette        April 2011 optimisation
    !!      S. Riette        08/2016 add option O
    !!      R. El Khatib     12-Aug-2021 written as a include file
    !
    !! --------------------------------------------------------------------------

    !
    IMPLICIT NONE
    !
    real, intent(in) :: XTT,XTMAXMIX, XTMINMIX

    CHARACTER(LEN=1), INTENT(IN)    :: HFRAC_ICE       ! scheme to use
    REAL,             INTENT(IN)    :: PT              ! temperature
    REAL,             INTENT(INOUT) :: PFRAC_ICE       ! Ice fraction (1 for ice only, 0 for liquid only)
    INTEGER, OPTIONAL,        INTENT(OUT)   :: KERR            ! Error code in return
    !
    !------------------------------------------------------------------------

    !                1. Compute FRAC_ICE
    !
    IF (PRESENT(KERR)) KERR=0
    SELECT CASE(HFRAC_ICE)
      CASE ('T') !using Temperature
        PFRAC_ICE = MAX( 0., MIN(1., (( XTMAXMIX - PT ) / (XTMAXMIX - XTMINMIX )) ) ) ! freezing interval
      CASE ('O') !using Temperature with old formulae
        PFRAC_ICE = MAX( 0., MIN(1., (( XTT - PT ) / 40.) ) ) ! freezing interval
      CASE ('N') !No ice
        PFRAC_ICE = 0.
      CASE ('S') !Same as previous
        ! (almost) nothing to do
        PFRAC_ICE = MAX( 0., MIN(1., PFRAC_ICE ) )
      CASE DEFAULT
        IF (PRESENT(KERR)) KERR=1
    END SELECT

END SUBROUTINE COMPUTE_FRAC_ICE


END SUBROUTINE ICE_ADJUST
