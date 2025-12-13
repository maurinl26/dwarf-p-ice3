!MNH_LIC Copyright 1996-2021 CNRS, Meteo-France and Universite Paul Sabatier
!MNH_LIC This is part of the Meso-NH software governed by the CeCILL-C licence
!MNH_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt
!MNH_LIC for details. version 1.
!-----------------------------------------------------------------
!     ##########################################################################
      SUBROUTINE ICE_ADJUST (D, CST, ICEP, NEBN, TURBN, PARAMI, BUCONF, KRR,   &
                            &HBUNAME,                                          &
                            &PTSTEP, PSIGQSAT,                                 &
                            &PRHODJ, PEXNREF, PRHODREF, PSIGS, LMFCONV, PMFCONV,&
                            &PPABST, PZZ,                                      &
                            &PEXN, PCF_MF, PRC_MF, PRI_MF, PWEIGHT_MF_CLOUD,   &
                            &PICLDFR, PWCLDFR, PSSIO, PSSIU, PIFR,             &
                            &PRV, PRC, PRVS, PRCS, PTH, PTHS,                  &
                            &OCOMPUTE_SRC, PSRCS, PCLDFR,                      &
                            &PRR, PRI, PRIS, PRS, PRG, TBUDGETS, KBUDGETS,     &
                            &PICE_CLD_WGT,                                     &
                            &PRH,                                              &
                            &POUT_RV, POUT_RC, POUT_RI, POUT_TH,               &
                            &PHLC_HRC, PHLC_HCF, PHLI_HRI, PHLI_HCF,           &
                            &PHLC_HRC_MF, PHLC_HCF_MF, PHLI_HRI_MF, PHLI_HCF_MF)

!     #########################################################################
!
!!****  *ICE_ADJUST* -  compute the ajustment of water vapor in mixed-phase
!!                      clouds (OpenACC GPU-enabled version)
!!
!!    MODIFICATIONS
!!    -------------
!!      2025-12 OpenACC directives added for GPU acceleration
!!
!-------------------------------------------------------------------------------
!
!*       0.    DECLARATIONS
!              ------------
!
USE YOMHOOK , ONLY : LHOOK, DR_HOOK, JPHOOK
USE MODD_DIMPHYEX,   ONLY: DIMPHYEX_t
USE MODD_CST,        ONLY: CST_t
USE MODD_NEB_n,      ONLY: NEB_t
USE MODD_TURB_n,         ONLY: TURB_t
USE MODD_PARAM_ICE_n,    ONLY: PARAM_ICE_t
USE MODD_BUDGET,     ONLY: TBUDGETDATA_PTR, TBUDGETCONF_t, NBUDGET_TH, NBUDGET_RV, NBUDGET_RC, NBUDGET_RI
USE MODD_RAIN_ICE_PARAM_n, ONLY : RAIN_ICE_PARAM_t
!
!
USE MODI_CONDENSATION
!
IMPLICIT NONE
!
!
!*       0.1   Declarations of dummy arguments :
!
!
TYPE(DIMPHYEX_t),         INTENT(IN)    :: D
TYPE(CST_t),              INTENT(IN)    :: CST
TYPE(RAIN_ICE_PARAM_t),   INTENT(IN)    :: ICEP
TYPE(NEB_t),              INTENT(IN)    :: NEBN
TYPE(TURB_t),             INTENT(IN)    :: TURBN
TYPE(PARAM_ICE_t),        INTENT(IN)    :: PARAMI
TYPE(TBUDGETCONF_t),      INTENT(IN)    :: BUCONF
INTEGER,                  INTENT(IN)    :: KRR      ! Number of moist variables
CHARACTER(LEN=4),         INTENT(IN)    :: HBUNAME  ! Name of the budget
REAL,                     INTENT(IN)   :: PTSTEP    ! Double Time step
!
REAL, DIMENSION(D%NIJT),       INTENT(IN)    :: PSIGQSAT
!
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    ::  PRHODJ
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    ::  PEXNREF
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    ::  PRHODREF
!
REAL, DIMENSION(MERGE(D%NIJT,0,NEBN%LSUBG_COND),&
                MERGE(D%NKT,0,NEBN%LSUBG_COND)),           INTENT(IN)    ::  PSIGS
LOGICAL,                                              INTENT(IN)    ::  LMFCONV
REAL, DIMENSION(MERGE(D%NIJT,0,LMFCONV),&
                MERGE(D%NKT,0,LMFCONV)),              INTENT(IN)   ::  PMFCONV
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    ::  PPABST
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    ::  PZZ
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    ::  PEXN
!
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    :: PCF_MF
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    :: PRC_MF
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    :: PRI_MF
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    :: PWEIGHT_MF_CLOUD
!
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    :: PRV
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    :: PRC
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(INOUT) :: PRVS
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(INOUT) :: PRCS
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    :: PTH
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(INOUT) :: PTHS
LOGICAL,                            INTENT(IN)    :: OCOMPUTE_SRC
REAL, DIMENSION(MERGE(D%NIJT,0,OCOMPUTE_SRC),&
                MERGE(D%NKT,0,OCOMPUTE_SRC)), INTENT(OUT)   :: PSRCS
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(OUT)  ::  PCLDFR
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(OUT)  ::  PICLDFR
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(OUT)  ::  PWCLDFR
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(OUT)  ::  PSSIO
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(OUT)  ::  PSSIU
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(OUT)  ::  PIFR
!
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(INOUT)::  PRIS
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)   ::  PRR
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)   ::  PRI
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)   ::  PRS
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)   ::  PRG
TYPE(TBUDGETDATA_PTR), DIMENSION(KBUDGETS),       INTENT(INOUT)::  TBUDGETS
INTEGER,                                      INTENT(IN)   ::  KBUDGETS
REAL, DIMENSION(D%NIJT),       OPTIONAL, INTENT(IN)   ::  PICE_CLD_WGT
REAL, DIMENSION(D%NIJT,D%NKT), OPTIONAL, INTENT(IN)   ::  PRH
REAL, DIMENSION(D%NIJT,D%NKT), OPTIONAL, INTENT(OUT)  ::  POUT_RV
REAL, DIMENSION(D%NIJT,D%NKT), OPTIONAL, INTENT(OUT)  ::  POUT_RC
REAL, DIMENSION(D%NIJT,D%NKT), OPTIONAL, INTENT(OUT)  ::  POUT_RI
REAL, DIMENSION(D%NIJT,D%NKT), OPTIONAL, INTENT(OUT)  ::  POUT_TH
REAL, DIMENSION(D%NIJT,D%NKT), OPTIONAL, INTENT(OUT)  ::  PHLC_HRC
REAL, DIMENSION(D%NIJT,D%NKT), OPTIONAL, INTENT(OUT)  ::  PHLC_HCF
REAL, DIMENSION(D%NIJT,D%NKT), OPTIONAL, INTENT(OUT)  ::  PHLI_HRI
REAL, DIMENSION(D%NIJT,D%NKT), OPTIONAL, INTENT(OUT)  ::  PHLI_HCF
REAL, DIMENSION(D%NIJT,D%NKT), OPTIONAL, INTENT(IN)   ::  PHLC_HRC_MF
REAL, DIMENSION(D%NIJT,D%NKT), OPTIONAL, INTENT(IN)   ::  PHLC_HCF_MF
REAL, DIMENSION(D%NIJT,D%NKT), OPTIONAL, INTENT(IN)   ::  PHLI_HRI_MF
REAL, DIMENSION(D%NIJT,D%NKT), OPTIONAL, INTENT(IN)   ::  PHLI_HCF_MF
!
!
!*       0.2   Declarations of local variables :
!
!
REAL  :: ZW1,ZW2
REAL, DIMENSION(D%NIJT,D%NKT) &
                         :: ZT,   &
                   ZRV, ZRC, ZRI, &
                            ZCPH, &
                            ZLV,  &
                            ZLS
REAL :: ZCRIAUT, &
        ZHCF, ZHR
!
INTEGER             :: JITER,ITERMAX
INTEGER             :: JIJ, JK
INTEGER :: IKTB, IKTE, IIJB, IIJE
!
REAL, DIMENSION(D%NIJT,D%NKT) :: ZSIGS, ZSRCS
REAL, DIMENSION(D%NIJT) :: ZSIGQSAT
LOGICAL :: LLNONE, LLTRIANGLE, LLBIGA, LLHLC_H, LLHLI_H

REAL(KIND=JPHOOK) :: ZHOOK_HANDLE
!
!-------------------------------------------------------------------------------
!
!*       1.     PRELIMINARIES
!               -------------
!
IF (LHOOK) CALL DR_HOOK('ICE_ADJUST',0,ZHOOK_HANDLE)
!
IKTB=D%NKTB
IKTE=D%NKTE
IIJB=D%NIJB
IIJE=D%NIJE
!
ITERMAX=1
!
IF(BUCONF%LBUDGET_TH) CALL TBUDGETS(NBUDGET_TH)%PTR%INIT_PHY(D, TRIM(HBUNAME), PTHS(:, :)*PRHODJ(:, :))
IF(BUCONF%LBUDGET_RV) CALL TBUDGETS(NBUDGET_RV)%PTR%INIT_PHY(D, TRIM(HBUNAME), PRVS(:, :)*PRHODJ(:, :))
IF(BUCONF%LBUDGET_RC) CALL TBUDGETS(NBUDGET_RC)%PTR%INIT_PHY(D, TRIM(HBUNAME), PRCS(:, :)*PRHODJ(:, :))
IF(BUCONF%LBUDGET_RI) CALL TBUDGETS(NBUDGET_RI)%PTR%INIT_PHY(D, TRIM(HBUNAME), PRIS(:, :)*PRHODJ(:, :))
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
  !$acc data &
  !$acc&  present(PTH, PEXN, ZT, ZLV, ZLS, CST, D) &
  !$acc&  present(PRV, PRC, PRI, PRVS, PRCS, PRIS, PTHS) &
  !$acc&  present(PRHODJ, PEXNREF, PRHODREF, PPABST, PZZ) &
  !$acc&  present(PCF_MF, PRC_MF, PRI_MF, PWEIGHT_MF_CLOUD) &
  !$acc&  present(PRR, PRS, PRG, PEXN) &
  !$acc&  present(PCLDFR, PICLDFR, PWCLDFR, PSSIO, PSSIU, PIFR) &
  !$acc&  present(ZRV, ZRC, ZRI, ZCPH, ZSRCS) &
  !$acc&  copyin(PSIGQSAT, PSIGS)

  !$acc parallel loop gang vector collapse(2) private(JIJ, JK)
  DO JK=IKTB,IKTE
    DO JIJ=IIJB,IIJE
      IF (JITER==1) ZT(JIJ,JK) = PTH(JIJ,JK) * PEXN(JIJ,JK)
      ZLV(JIJ,JK) = CST%XLVTT + ( CST%XCPV - CST%XCL ) * ( ZT(JIJ,JK) -CST%XTT )
      ZLS(JIJ,JK) = CST%XLSTT + ( CST%XCPV - CST%XCI ) * ( ZT(JIJ,JK) -CST%XTT )
    ENDDO
  ENDDO
  !$acc end parallel loop

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
  !$acc end data
ENDDO         ! end of the iterative loop
!
!*       5.     COMPUTE THE SOURCES AND STORES THE CLOUD FRACTION
!               -------------------------------------------------
!
!
! Apply a ponderation between condensation and mas flux cloud
LLHLC_H=PRESENT(PHLC_HRC).AND.PRESENT(PHLC_HCF)
LLHLI_H=PRESENT(PHLI_HRI).AND.PRESENT(PHLI_HCF)

!$acc data &
!$acc&  present(ZRC, ZRI, PCLDFR, ZSRCS, PWEIGHT_MF_CLOUD) &
!$acc&  present(PHLC_HRC, PHLC_HCF, PHLI_HRI, PHLI_HCF)

!$acc parallel loop gang vector collapse(2) private(JIJ, JK)
DO JK=IKTB,IKTE
  DO JIJ=IIJB,IIJE
    ZRC(JIJ,JK)=ZRC(JIJ,JK)*(1.-PWEIGHT_MF_CLOUD(JIJ,JK))
    ZRI(JIJ,JK)=ZRI(JIJ,JK)*(1.-PWEIGHT_MF_CLOUD(JIJ,JK))
    PCLDFR(JIJ,JK)=PCLDFR(JIJ,JK)*(1.-PWEIGHT_MF_CLOUD(JIJ,JK))
    ZSRCS(JIJ,JK)=ZSRCS(JIJ,JK)*(1.-PWEIGHT_MF_CLOUD(JIJ,JK))
    IF(LLHLC_H) THEN
      PHLC_HRC(JIJ,JK)=PHLC_HRC(JIJ,JK)*(1.-PWEIGHT_MF_CLOUD(JIJ,JK))
      PHLC_HCF(JIJ,JK)=PHLC_HCF(JIJ,JK)*(1.-PWEIGHT_MF_CLOUD(JIJ,JK))
    ENDIF
    IF(LLHLI_H) THEN
      PHLI_HRI(JIJ,JK)=PHLI_HRI(JIJ,JK)*(1.-PWEIGHT_MF_CLOUD(JIJ,JK))
      PHLI_HCF(JIJ,JK)=PHLI_HCF(JIJ,JK)*(1.-PWEIGHT_MF_CLOUD(JIJ,JK))
    ENDIF
  ENDDO
ENDDO
!$acc end parallel loop

!
!$acc parallel loop gang vector collapse(2) private(JIJ, JK, ZW1, ZW2)
DO JK=IKTB,IKTE
  DO JIJ=IIJB,IIJE
    !
    !*       5.0    compute the variation of mixing ratio
    !
    ZW1 = (ZRC(JIJ,JK) - PRC(JIJ,JK)) / PTSTEP
    ZW2 = (ZRI(JIJ,JK) - PRI(JIJ,JK)) / PTSTEP
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
ENDDO
!$acc end parallel loop

  !
  !*       5.2    compute the cloud fraction PCLDFR
  !
IF ( .NOT. NEBN%LSUBG_COND ) THEN

  !$acc parallel loop gang vector collapse(2) private(JIJ, JK)
  DO JK=IKTB,IKTE
    DO JIJ=IIJB,IIJE
      IF (PRCS(JIJ,JK) + PRIS(JIJ,JK) > 1.E-12 / PTSTEP) THEN
        PCLDFR(JIJ,JK)  = 1.
      ELSE
        PCLDFR(JIJ,JK)  = 0.
      ENDIF
      ZSRCS(JIJ,JK) = PCLDFR(JIJ,JK)
    ENDDO
  ENDDO
  !$acc end parallel loop

ELSE !NEBN%LSUBG_COND case

    LLNONE=PARAMI%CSUBG_MF_PDF=='NONE'
    LLTRIANGLE=PARAMI%CSUBG_MF_PDF=='TRIANGLE'
    LLBIGA=PARAMI%CSUBG_MF_PDF=='BIGA'

  !$acc parallel loop gang vector collapse(2) &
  !$acc&  private(JIJ, JK, ZW1, ZW2, ZCRIAUT, ZHCF, ZHR)
  DO JK=IKTB,IKTE
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
        ZCRIAUT=ICEP%XCRIAUTC/PRHODREF(JIJ,JK)
        IF(LLNONE)THEN
          IF(ZW1*PTSTEP>PCF_MF(JIJ,JK) * ZCRIAUT) THEN
            PHLC_HRC(JIJ,JK)=PHLC_HRC(JIJ,JK)+ZW1*PTSTEP
            PHLC_HCF(JIJ,JK)=MIN(1.,PHLC_HCF(JIJ,JK)+PCF_MF(JIJ,JK))
          ENDIF
        ELSEIF(LLTRIANGLE)THEN
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
          ZHCF=ZHCF*PCF_MF(JIJ,JK)
          PHLC_HCF(JIJ,JK)=MIN(1.,PHLC_HCF(JIJ,JK)+ZHCF)
          PHLC_HRC(JIJ,JK)=PHLC_HRC(JIJ,JK)+ZHR
        ELSEIF(LLBIGA)THEN
          PHLC_HCF(JIJ,JK)=MIN(1., PHLC_HCF(JIJ,JK)+PHLC_HCF_MF(JIJ,JK))
          PHLC_HRC(JIJ,JK)=PHLC_HRC(JIJ,JK)+PHLC_HRC_MF(JIJ,JK)
        ENDIF
      ENDIF
      IF(LLHLI_H) THEN
        ZCRIAUT=MIN(ICEP%XCRIAUTI,10**(ICEP%XACRIAUTI*(ZT(JIJ,JK)-CST%XTT)+ICEP%XBCRIAUTI))
        IF(LLNONE)THEN
          IF(ZW2*PTSTEP>PCF_MF(JIJ,JK) * ZCRIAUT) THEN
            PHLI_HRI(JIJ,JK)=PHLI_HRI(JIJ,JK)+ZW2*PTSTEP
            PHLI_HCF(JIJ,JK)=MIN(1.,PHLI_HCF(JIJ,JK)+PCF_MF(JIJ,JK))
          ENDIF
        ELSEIF(LLTRIANGLE)THEN
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
          ZHCF=ZHCF*PCF_MF(JIJ,JK)
          PHLI_HCF(JIJ,JK)=MIN(1.,PHLI_HCF(JIJ,JK)+ZHCF)
          PHLI_HRI(JIJ,JK)=PHLI_HRI(JIJ,JK)+ZHR
        ELSEIF(LLBIGA)THEN
          PHLI_HCF(JIJ,JK)=MIN(1., PHLI_HCF(JIJ,JK)+PHLI_HCF_MF(JIJ,JK))
          PHLI_HRI(JIJ,JK)=PHLI_HRI(JIJ,JK)+PHLI_HRI_MF(JIJ,JK)
        ENDIF
      ENDIF
    !
    IF(PRESENT(POUT_RV) .OR. PRESENT(POUT_RC) .OR. &
      &PRESENT(POUT_RI) .OR. PRESENT(POUT_TH)) THEN
        ZW1=PRC_MF(JIJ,JK)
        ZW2=PRI_MF(JIJ,JK)
        IF(ZW1+ZW2>ZRV(JIJ,JK)) THEN
          ZW1=ZW1*ZRV(JIJ,JK)/(ZW1+ZW2)
          ZW2=ZRV(JIJ,JK)-ZW1
        ENDIF
        ZRC(JIJ,JK)=ZRC(JIJ,JK)+ZW1
        ZRI(JIJ,JK)=ZRI(JIJ,JK)+ZW2
        ZRV(JIJ,JK)=ZRV(JIJ,JK)-(ZW1+ZW2)
        ZT(JIJ,JK) = ZT(JIJ,JK) + &
                    (ZW1 * ZLV(JIJ,JK) + ZW2 * ZLS(JIJ,JK)) / ZCPH(JIJ,JK)
      ENDIF
    END DO
  ENDDO
  !$acc end parallel loop

ENDIF !NEBN%LSUBG_COND
!$acc end data
!
IF (OCOMPUTE_SRC) PSRCS=ZSRCS
IF(PRESENT(POUT_RV)) POUT_RV=ZRV
IF(PRESENT(POUT_RC)) POUT_RC=ZRC
IF(PRESENT(POUT_RI)) POUT_RI=ZRI
IF(PRESENT(POUT_TH)) POUT_TH=ZT / PEXN(:,:)
!
!
!*       6.  STORE THE BUDGET TERMS
!            ----------------------
!
IF(BUCONF%LBUDGET_TH) CALL TBUDGETS(NBUDGET_TH)%PTR%END_PHY(D, TRIM(HBUNAME), PTHS(:, :)*PRHODJ(:, :))
IF(BUCONF%LBUDGET_RV) CALL TBUDGETS(NBUDGET_RV)%PTR%END_PHY(D, TRIM(HBUNAME), PRVS(:, :)*PRHODJ(:, :))
IF(BUCONF%LBUDGET_RC) CALL TBUDGETS(NBUDGET_RC)%PTR%END_PHY(D, TRIM(HBUNAME), PRCS(:, :)*PRHODJ(:, :))
IF(BUCONF%LBUDGET_RI) CALL TBUDGETS(NBUDGET_RI)%PTR%END_PHY(D, TRIM(HBUNAME), PRIS(:, :)*PRHODJ(:, :))
!------------------------------------------------------------------------------
!
!
IF (LHOOK) CALL DR_HOOK('ICE_ADJUST',1,ZHOOK_HANDLE)
!
CONTAINS
SUBROUTINE ITERATION(PRV_IN,PRC_IN,PRI_IN,PRV_OUT,PRC_OUT,PRI_OUT)

REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN) :: PRV_IN
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN) :: PRC_IN
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN) :: PRI_IN
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(OUT) :: PRV_OUT
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(OUT) :: PRC_OUT
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(OUT) :: PRI_OUT
!
!*       2.4    compute the specific heat for moist air (Cph) at t+1
!

!$acc parallel loop gang vector collapse(2) private(JIJ, JK)
DO JK=IKTB,IKTE
  DO JIJ=IIJB,IIJE
    SELECT CASE(KRR)
      CASE(7)
        ZCPH(JIJ,JK) = CST%XCPD + CST%XCPV * PRV_IN(JIJ,JK)                             &
                                + CST%XCL  * (PRC_IN(JIJ,JK) + PRR(JIJ,JK))             &
                                + CST%XCI  * (PRI_IN(JIJ,JK) + PRS(JIJ,JK) + PRG(JIJ,JK) + PRH(JIJ,JK))
      CASE(6)
        ZCPH(JIJ,JK) = CST%XCPD + CST%XCPV * PRV_IN(JIJ,JK)                             &
                                + CST%XCL  * (PRC_IN(JIJ,JK) + PRR(JIJ,JK))             &
                                + CST%XCI  * (PRI_IN(JIJ,JK) + PRS(JIJ,JK) + PRG(JIJ,JK))
      CASE(5)
        ZCPH(JIJ,JK) = CST%XCPD + CST%XCPV * PRV_IN(JIJ,JK)                             &
                                + CST%XCL  * (PRC_IN(JIJ,JK) + PRR(JIJ,JK))             &
                                + CST%XCI  * (PRI_IN(JIJ,JK) + PRS(JIJ,JK))
      CASE(3)
        ZCPH(JIJ,JK) = CST%XCPD + CST%XCPV * PRV_IN(JIJ,JK)               &
                                + CST%XCL  * (PRC_IN(JIJ,JK) + PRR(JIJ,JK))
      CASE(2)
        ZCPH(JIJ,JK) = CST%XCPD + CST%XCPV * PRV_IN(JIJ,JK) &
                                + CST%XCL  * PRC_IN(JIJ,JK)
    END SELECT
  ENDDO
ENDDO
!$acc end parallel loop

!
IF ( NEBN%LSUBG_COND ) THEN
  !
  !*       3.     SUBGRID CONDENSATION SCHEME
  !               ---------------------------
  !
  !   ZSRC= s'rci'/Sigma_s^2
  !   ZT is INOUT
  CALL CONDENSATION(D, CST, ICEP, NEBN, TURBN, &
       NEBN%CFRAC_ICE_ADJUST,NEBN%CCONDENS, NEBN%CLAMBDA3,                             &
       PPABST, PZZ, PRHODREF, ZT, PRV_IN, PRV_OUT, PRC_IN, PRC_OUT, PRI_IN, PRI_OUT, &
       PRR, PRS, PRG, PSIGS, LMFCONV, PMFCONV, PCLDFR, &
       ZSRCS, .TRUE., NEBN%LSIGMAS, PARAMI%LOCND2,                       &
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

  !ZT is INOUT
  CALL CONDENSATION(D, CST, ICEP, NEBN, TURBN, &
       NEBN%CFRAC_ICE_ADJUST,NEBN%CCONDENS, NEBN%CLAMBDA3,                             &
       PPABST, PZZ, PRHODREF, ZT, PRV_IN, PRV_OUT, PRC_IN, PRC_OUT, PRI_IN, PRI_OUT, &
       PRR, PRS, PRG, ZSIGS, LMFCONV, PMFCONV, PCLDFR, &
       ZSRCS, .TRUE., .TRUE., PARAMI%LOCND2,                             &
       PICLDFR, PWCLDFR, PSSIO, PSSIU, PIFR, ZSIGQSAT,                   &
       PLV=ZLV, PLS=ZLS, PCPH=ZCPH,                                      &
       PHLC_HRC=PHLC_HRC, PHLC_HCF=PHLC_HCF, PHLI_HRI=PHLI_HRI, PHLI_HCF=PHLI_HCF,&
       PICE_CLD_WGT=PICE_CLD_WGT)
ENDIF

END SUBROUTINE ITERATION

END SUBROUTINE ICE_ADJUST
