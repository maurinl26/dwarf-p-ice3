!MNH_LIC Copyright 1996-2021 CNRS, Meteo-France and Universite Paul Sabatier
!MNH_LIC This is part of the Meso-NH software governed by the CeCILL-C licence
!MNH_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt
!MNH_LIC for details. version 1.
!-----------------------------------------------------------------
!     ##########################################################################
      SUBROUTINE CLOUD_FRACTION(NIJT, NKT, &
                            &XCRIAUTC, XCRIAUTI, XACRIAUTI, XBCRIAUTI, XTT, &
                            &NKTE, NKTB, &
                            &NIJB, NIJE, &
                            &CSUBG_MF_PDF, &
                            &LSUBG_COND, &
                            &ZRI, ZRC, &
                            &PTSTEP,                                  &
                            &PEXNREF, PRHODREF, &
                            &PCF_MF, PRC_MF, PRI_MF,                     &
                            &PRC, PRVS, PRCS, PTHS,                  &
                            &OCOMPUTE_SRC, PSRCS, PCLDFR,                      &
                            &PRI, PRIS, &
                            &PHLC_HRC, PHLC_HCF, PHLI_HRI, PHLI_HCF)

!     #########################################################################
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
!

integer, intent(in) :: NIJT, NKT
integer, intent(in) :: NKTE, NKTB
integer, intent(in) :: NIJB, NIJE
logical, intent(in) :: LSUBG_COND
real, intent(in) :: XCRIAUTC, XCRIAUTI, XACRIAUTI, XBCRIAUTI, XTT
character(len=80), intent(in) :: CSUBG_MF_PDF

real, dimension(NIJT, NKT), intent(in) :: ZRC, ZRI


REAL,                     INTENT(IN)   :: PTSTEP    ! Double Time step
                                                    ! (single if cold start)
!
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    ::  PEXNREF ! Reference Exner function
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    ::  PRHODREF
!
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PCF_MF   ! Convective Mass Flux Cloud fraction
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PRC_MF   ! Convective Mass Flux liquid mixing ratio
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PRI_MF   ! Convective Mass Flux ice mixing ratio
!
REAL, DIMENSION(NIJT,NKT), INTENT(IN)    :: PRC     ! Cloud water m.r. to adjust
REAL, DIMENSION(NIJT,NKT), INTENT(INOUT) :: PRVS    ! Water vapor m.r. source
REAL, DIMENSION(NIJT,NKT), INTENT(INOUT) :: PRCS    ! Cloud water m.r. source
REAL, DIMENSION(NIJT,NKT), INTENT(INOUT) :: PTHS    ! Theta source
LOGICAL,                            INTENT(IN)    :: OCOMPUTE_SRC
REAL, DIMENSION(MERGE(NIJT,0,OCOMPUTE_SRC),&
                MERGE(NKT,0,OCOMPUTE_SRC)), INTENT(OUT)   :: PSRCS   ! Second-order flux
                                                                       ! s'rc'/2Sigma_s2 at time t+1
                                                                       ! multiplied by Lambda_3
REAL, DIMENSION(NIJT,NKT), INTENT(OUT)  ::  PCLDFR  ! Cloud fraction

REAL, DIMENSION(NIJT,NKT), INTENT(INOUT)::  PRIS ! Cloud ice  m.r. at t+1
REAL, DIMENSION(NIJT,NKT), INTENT(IN)   ::  PRI  ! Cloud ice  m.r. to adjust
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
                            ZCPH, &  ! guess of the CPh for the mixing
                            ZLV,  &  ! guess of the Lv at t+1
                            ZLS      ! guess of the Ls at t+1
REAL :: ZCRIAUT, & ! Autoconversion thresholds
        ZHCF, ZHR
!
INTEGER             :: JIJ, JK
INTEGER :: IKTB, IKTE, IIJB, IIJE
!
LOGICAL :: LLNONE, LLTRIANGLE, LLHLC_H, LLHLI_H

! REAL(KIND=JPHOOK) :: ZHOOK_HANDLE
!
!-------------------------------------------------------------------------------
!
!*       1.     PRELIMINARIES
!               -------------
!
! IF (LHOOK) CALL DR_HOOK('ICE_ADJUST',0,ZHOOK_HANDLE)
!
IKTB=NKTB
IKTE=NKTE
IIJB=NIJB
IIJE=NIJE
!
!*       5.     COMPUTE THE SOURCES AND STORES THE CLOUD FRACTION
!               -------------------------------------------------
!
!
DO JK=IKTB,IKTE
  DO JIJ=IIJB,IIJE
    !
    !*       5.0    compute the variation of mixing ratio
    !
                                                         !         Rc - Rc*
    ZW1 = (ZRC(JIJ,JK) - PRC(JIJ,JK)) / PTSTEP       ! Pcon = ----------
                                                         !         2 Delta t
    ZW2 = (ZRI(JIJ,JK) - PRI(JIJ,JK)) / PTSTEP       ! idem ZW1 but for Ri
    !
    !*       5.1    compute the sources
    !
    IF( ZW1 < 0.0 ) THEN
      ZW1 = MAX ( ZW1, -PRCS(JIJ,JK) )
    ELSE
      ZW1 = MIN ( ZW1,  PRVS(JIJ,JK) )
    ENDIF
    PRVS(JIJ,JK) = PRVS(JIJ,JK) - ZW1
    PRCS(JIJ,JK) = PRCS(JIJ,JK) + ZW1
    PTHS(JIJ,JK) = PTHS(JIJ,JK) +        &
                    ZW1 * ZLV(JIJ,JK) / (ZCPH(JIJ,JK) * PEXNREF(JIJ,JK))
    !
    IF( ZW2 < 0.0 ) THEN
      ZW2 = MAX ( ZW2, -PRIS(JIJ,JK) )
    ELSE
      ZW2 = MIN ( ZW2,  PRVS(JIJ,JK) )
    ENDIF
    PRVS(JIJ,JK) = PRVS(JIJ,JK) - ZW2
    PRIS(JIJ,JK) = PRIS(JIJ,JK) + ZW2
    PTHS(JIJ,JK) = PTHS(JIJ,JK) +        &
                  ZW2 * ZLS(JIJ,JK) / (ZCPH(JIJ,JK) * PEXNREF(JIJ,JK))
  ENDDO
  !
  !*       5.2    compute the cloud fraction PCLDFR
  !
  IF ( .NOT. LSUBG_COND ) THEN
    DO JIJ=IIJB,IIJE
      IF (PRCS(JIJ,JK) + PRIS(JIJ,JK) > 1.E-12 / PTSTEP) THEN
        PCLDFR(JIJ,JK)  = 1.
      ELSE
        PCLDFR(JIJ,JK)  = 0.
      ENDIF
      IF (OCOMPUTE_SRC) THEN
        PSRCS(JIJ,JK) = PCLDFR(JIJ,JK)
      END IF
    ENDDO
  ELSE !NEBN%LSUBG_COND case
    ! Tests on characters strings can break the vectorization, or at least they would
    ! slow down considerably the performance of a vector loop. One should use tests on
    ! reals, integers or booleans only. REK.
    LLNONE=CSUBG_MF_PDF=='NONE'
    LLTRIANGLE=CSUBG_MF_PDF=='TRIANGLE'
    LLHLC_H=PRESENT(PHLC_HRC).AND.PRESENT(PHLC_HCF)
    LLHLI_H=PRESENT(PHLI_HRI).AND.PRESENT(PHLI_HCF)
    DO JIJ=IIJB,IIJE
      !We limit PRC_MF+PRI_MF to PRVS*PTSTEP to avoid negative humidity
      ZW1=PRC_MF(JIJ,JK)/PTSTEP
      ZW2=PRI_MF(JIJ,JK)/PTSTEP
      IF(ZW1+ZW2>PRVS(JIJ,JK)) THEN
        ZW1=ZW1*PRVS(JIJ,JK)/(ZW1+ZW2)
        ZW2=PRVS(JIJ,JK)-ZW1
      ENDIF
      PCLDFR(JIJ,JK)=MIN(1.,PCLDFR(JIJ,JK)+PCF_MF(JIJ,JK))
      PRCS(JIJ,JK)=PRCS(JIJ,JK)+ZW1
      PRIS(JIJ,JK)=PRIS(JIJ,JK)+ZW2
      PRVS(JIJ,JK)=PRVS(JIJ,JK)-(ZW1+ZW2)
      PTHS(JIJ,JK) = PTHS(JIJ,JK) + &
                    (ZW1 * ZLV(JIJ,JK) + ZW2 * ZLS(JIJ,JK)) / ZCPH(JIJ,JK) / PEXNREF(JIJ,JK)
      !
      IF(LLHLC_H) THEN
        ZCRIAUT=XCRIAUTC/PRHODREF(JIJ,JK)
        IF(LLNONE)THEN
          IF(ZW1*PTSTEP>PCF_MF(JIJ,JK) * ZCRIAUT) THEN
            PHLC_HRC(JIJ,JK)=PHLC_HRC(JIJ,JK)+ZW1*PTSTEP
            PHLC_HCF(JIJ,JK)=MIN(1.,PHLC_HCF(JIJ,JK)+PCF_MF(JIJ,JK))
          ENDIF
        ELSEIF(LLTRIANGLE)THEN
          !ZHCF is the precipitating part of the *cloud* and not of the grid cell
          IF(ZW1*PTSTEP>PCF_MF(JIJ,JK)*ZCRIAUT) THEN
            ZHCF=1.-.5*(ZCRIAUT*PCF_MF(JIJ,JK) / MAX(1.E-20, ZW1*PTSTEP))**2
            ZHR=ZW1*PTSTEP-(ZCRIAUT*PCF_MF(JIJ,JK))**3 / &
                                        &(3*MAX(1.E-20, ZW1*PTSTEP)**2)
          ELSEIF(2.*ZW1*PTSTEP<=PCF_MF(JIJ,JK) * ZCRIAUT) THEN
            ZHCF=0.
            ZHR=0.
          ELSE
            ZHCF=(2.*ZW1*PTSTEP-ZCRIAUT*PCF_MF(JIJ,JK))**2 / &
                       &(2.*MAX(1.E-20, ZW1*PTSTEP)**2)
            ZHR=(4.*(ZW1*PTSTEP)**3-3.*ZW1*PTSTEP*(ZCRIAUT*PCF_MF(JIJ,JK))**2+&
                        (ZCRIAUT*PCF_MF(JIJ,JK))**3) / &
                      &(3*MAX(1.E-20, ZW1*PTSTEP)**2)
          ENDIF
          ZHCF=ZHCF*PCF_MF(JIJ,JK) !to retrieve the part of the grid cell
          PHLC_HCF(JIJ,JK)=MIN(1.,PHLC_HCF(JIJ,JK)+ZHCF) !total part of the grid cell that is precipitating
          PHLC_HRC(JIJ,JK)=PHLC_HRC(JIJ,JK)+ZHR
        ENDIF
      ENDIF
      IF(LLHLI_H) THEN
        ZCRIAUT=MIN(XCRIAUTI,10**(XACRIAUTI*(ZT(JIJ,JK)-XTT)+XBCRIAUTI))
        IF(LLNONE)THEN
          IF(ZW2*PTSTEP>PCF_MF(JIJ,JK) * ZCRIAUT) THEN
            PHLI_HRI(JIJ,JK)=PHLI_HRI(JIJ,JK)+ZW2*PTSTEP
            PHLI_HCF(JIJ,JK)=MIN(1.,PHLI_HCF(JIJ,JK)+PCF_MF(JIJ,JK))
          ENDIF
        ELSEIF(LLTRIANGLE)THEN
          !ZHCF is the precipitating part of the *cloud* and not of the grid cell
          IF(ZW2*PTSTEP>PCF_MF(JIJ,JK)*ZCRIAUT) THEN
            ZHCF=1.-.5*(ZCRIAUT*PCF_MF(JIJ,JK) / (ZW2*PTSTEP))**2
            ZHR=ZW2*PTSTEP-(ZCRIAUT*PCF_MF(JIJ,JK))**3/(3*(ZW2*PTSTEP)**2)
          ELSEIF(2.*ZW2*PTSTEP<=PCF_MF(JIJ,JK) * ZCRIAUT) THEN
            ZHCF=0.
            ZHR=0.
          ELSE
            ZHCF=(2.*ZW2*PTSTEP-ZCRIAUT*PCF_MF(JIJ,JK))**2 / (2.*(ZW2*PTSTEP)**2)
            ZHR=(4.*(ZW2*PTSTEP)**3-3.*ZW2*PTSTEP*(ZCRIAUT*PCF_MF(JIJ,JK))**2+&
                        (ZCRIAUT*PCF_MF(JIJ,JK))**3)/(3*(ZW2*PTSTEP)**2)
          ENDIF
          ZHCF=ZHCF*PCF_MF(JIJ,JK) !to retrieve the part of the grid cell
          PHLI_HCF(JIJ,JK)=MIN(1.,PHLI_HCF(JIJ,JK)+ZHCF) !total part of the grid cell that is precipitating
          PHLI_HRI(JIJ,JK)=PHLI_HRI(JIJ,JK)+ZHR
        ENDIF
      ENDIF
    ENDDO
    !

  ENDIF !NEBN%LSUBG_COND
ENDDO

END SUBROUTINE CLOUD_FRACTION
