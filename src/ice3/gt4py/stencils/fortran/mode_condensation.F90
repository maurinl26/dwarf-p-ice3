!MNH_LIC Copyright 1994-2023 CNRS, Meteo-France and Universite Paul Sabatier
!MNH_LIC This is part of the Meso-NH software governed by the CeCILL-C licence
!MNH_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt
!MNH_LIC for details. version 1.
!-----------------------------------------------------------------
!     #######################
      MODULE MODE_CONDENSATION
!     #######################
!
!!****  *MODE_CONDENSATION* - Contains condensation routines from PHYEX
!!
!!    PURPOSE
!!    -------
!!      Compute microphysical adjustments due to condensation/evaporation
!!      using subgrid cloud scheme (CB02 - Chaboureau & Bechtold 2002)
!!
!!    REFERENCE
!!    ---------
!!      Chaboureau & Bechtold (2002), Bechtold et al. (1995)
!!      PHYEX-IAL_CY50T1/micro/condensation.F90
!!
!!    AUTHOR
!!    ------
!!      Generated from PHYEX reference for dwarf-p-ice3 project
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    01/11/2024
!-------------------------------------------------------------------------------
!
IMPLICIT NONE



CONTAINS

!-------------------------------------------------------------------------------
!
!     ##############################################################################
      SUBROUTINE CONDENSATION(NIJB, NIJE, NKTB, NKTE, NIJT, NKT, &
                            XRV, XRD, XALPI, XBETAI, XGAMI, &
                            XALPW, XBETAW, XGAMW, XTMAXMIX, XTMINMIX, &
                            OSIGMAS, OCND2, OUSERI, HFRAC_ICE, HCONDENS, LSTATNW, &
                            PPABS, PT, PRV_IN, PRC_IN, PRI_IN, &
                            PSIGS, PSIGQSAT, PLV, PLS, PCPH, &
                            PT_OUT, PRV_OUT, PRC_OUT, PRI_OUT, PCLDFR, &
                            ZQ1, ZPV, ZPIV, ZFRAC, ZQSL, ZQSI, &
                            ZSIGMA, ZCOND, ZA, ZB, ZSBAR)
!     ##############################################################################
!
!!****  *CONDENSATION* - Compute microphysical adjustments
!!
!!    PURPOSE
!!    -------
!!      Compute condensation/evaporation with subgrid cloud scheme
!!
!!    METHOD
!!    ------
!!      CB02 scheme (Chaboureau & Bechtold 2002)
!!      
!!    EXTERNAL
!!    --------
!!      None
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!      None
!!
!!    REFERENCE
!!    ---------
!!      PHYEX-IAL_CY50T1/micro/condensation.F90
!!
!!    AUTHOR
!!    ------
!!      Generated from PHYEX
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    01/11/2024
!-------------------------------------------------------------------------------
!
IMPLICIT NONE
!
!*       0.1   Declarations of dummy arguments
!
INTEGER, INTENT(IN) :: NIJB, NIJE, NKTB, NKTE, NIJT, NKT
REAL, INTENT(IN) :: XRV, XRD
REAL, INTENT(IN) :: XALPI, XBETAI, XGAMI
REAL, INTENT(IN) :: XALPW, XBETAW, XGAMW
REAL, INTENT(IN) :: XTMAXMIX, XTMINMIX
LOGICAL, INTENT(IN) :: OSIGMAS, OCND2, OUSERI, LSTATNW
INTEGER, INTENT(IN) :: HFRAC_ICE, HCONDENS

REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: PPABS, PT
REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: PRV_IN, PRC_IN, PRI_IN
REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: PSIGS
REAL, DIMENSION(NIJT), INTENT(IN) :: PSIGQSAT
REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: PLV, PLS, PCPH

REAL, DIMENSION(size(PPABS, 1), size(PPABS, 2)), INTENT(OUT) :: PT_OUT, PRV_OUT, PRC_OUT, PRI_OUT
REAL, DIMENSION(size(PPABS, 1), size(PPABS, 2)), INTENT(OUT) :: PCLDFR, ZQ1
REAL, DIMENSION(size(PPABS, 1), size(PPABS, 2)), INTENT(OUT) :: ZPV, ZPIV, ZFRAC
REAL, DIMENSION(size(PPABS, 1), size(PPABS, 2)), INTENT(OUT) :: ZQSL, ZQSI, ZSIGMA, ZCOND
REAL, DIMENSION(size(PPABS, 1), size(PPABS, 2)), INTENT(OUT) :: ZA, ZB, ZSBAR

!*       0.2   Declarations of local variables
!
INTEGER :: JIJ, JK
REAL :: ZRT, ZPRIFACT, ZAH, ZLVS, ZFRAC_TMP
REAL :: ZCOND_TMP, ZQSAT
REAL, PARAMETER :: ZEPS = 1.E-20

!-------------------------------------------------------------------------------
!
!*       1.    INITIALIZATIONS
!              ---------------
!
ZPRIFACT = 1.0  ! OCND2=False for AROME

! Initialize output fields
PCLDFR(:,:) = 0.0
PRV_OUT(:,:) = 0.0
PRC_OUT(:,:) = 0.0
PRI_OUT(:,:) = 0.0
PT_OUT(:,:) = PT(:,:)

!-------------------------------------------------------------------------------
!
!*       2.    SUBGRID CONDENSATION SCHEME
!              ---------------------------
!
DO JK = NKTB, NKTE
  DO JIJ = NIJB, NIJE
    
    ! Store total water mixing ratio
    ZRT = PRV_IN(JIJ,JK) + PRC_IN(JIJ,JK) + PRI_IN(JIJ,JK) * ZPRIFACT
    
    ! Saturation vapor pressures (over water and ice)
    IF (.NOT. OCND2) THEN
      ! Tetens formula for saturation vapor pressure
      ZPV(JIJ,JK) = MIN(611.2 * EXP(XALPW - XBETAW / PT(JIJ,JK) - &
                        XGAMW * LOG(PT(JIJ,JK))), 0.99 * PPABS(JIJ,JK))
      ZPIV(JIJ,JK) = MIN(611.2 * EXP(XALPI - XBETAI / PT(JIJ,JK) - &
                         XGAMI * LOG(PT(JIJ,JK))), 0.99 * PPABS(JIJ,JK))
    END IF
    
    ! Compute ice fraction
    IF (.NOT. OCND2 .AND. OUSERI) THEN
      IF (PRC_IN(JIJ,JK) + PRI_IN(JIJ,JK) > ZEPS) THEN
        ZFRAC_TMP = PRC_IN(JIJ,JK) / (PRC_IN(JIJ,JK) + PRI_IN(JIJ,JK))
      ELSE
        ZFRAC_TMP = 0.0
      END IF
      
      ! Apply ice fraction adjustment scheme
      SELECT CASE(HFRAC_ICE)
        CASE(0)  ! AROME mode (T)
          ZFRAC_TMP = MAX(0.0, MIN(1.0, (XTMAXMIX - PT(JIJ,JK)) / &
                                        (XTMAXMIX - XTMINMIX)))
        CASE(3)  ! Default mode (S)
          ZFRAC_TMP = MAX(0.0, MIN(1.0, ZFRAC_TMP))
      END SELECT
      
      ZFRAC(JIJ,JK) = ZFRAC_TMP
    ELSE
      ZFRAC(JIJ,JK) = 0.0
      ZFRAC_TMP = 0.0
    END IF
    
    ! Saturation mixing ratios
    ZQSL(JIJ,JK) = XRD / XRV * ZPV(JIJ,JK) / (PPABS(JIJ,JK) - ZPV(JIJ,JK))
    ZQSI(JIJ,JK) = XRD / XRV * ZPIV(JIJ,JK) / (PPABS(JIJ,JK) - ZPIV(JIJ,JK))
    
    ! Interpolate between liquid and solid
    ZQSAT = (1.0 - ZFRAC_TMP) * ZQSL(JIJ,JK) + ZFRAC_TMP * ZQSI(JIJ,JK)
    ZLVS = (1.0 - ZFRAC_TMP) * PLV(JIJ,JK) + ZFRAC_TMP * PLS(JIJ,JK)
    
    ! Thermodynamic coefficients a and b
    ZAH = ZLVS * ZQSAT / (XRV * PT(JIJ,JK)**2) * (1.0 + XRV * ZQSAT / XRD)
    ZA(JIJ,JK) = 1.0 / (1.0 + ZLVS / PCPH(JIJ,JK) * ZAH)
    ZB(JIJ,JK) = ZAH * ZA(JIJ,JK)
    ZSBAR(JIJ,JK) = ZA(JIJ,JK) * (ZRT - ZQSAT + &
                     ZAH * ZLVS * (PRC_IN(JIJ,JK) + PRI_IN(JIJ,JK) * ZPRIFACT) / PCPH(JIJ,JK))
    
    ! Subgrid standard deviation (sigma_s formulation)
    IF (OSIGMAS .AND. .NOT. LSTATNW) THEN
      IF (PSIGQSAT(JIJ) /= 0.0) THEN
        ZSIGMA(JIJ,JK) = SQRT((2.0 * PSIGS(JIJ,JK))**2 + &
                              (PSIGQSAT(JIJ) * ZQSAT * ZA(JIJ,JK))**2)
      ELSE
        ZSIGMA(JIJ,JK) = 2.0 * PSIGS(JIJ,JK)
      END IF
    ELSE
      ZSIGMA(JIJ,JK) = MAX(1.E-10, ZSIGMA(JIJ,JK))
    END IF
    
    ZSIGMA(JIJ,JK) = MAX(1.E-10, ZSIGMA(JIJ,JK))
    ZQ1(JIJ,JK) = ZSBAR(JIJ,JK) / ZSIGMA(JIJ,JK)
    
    ! CB02 fractional cloudiness and cloud condensate
    IF (HCONDENS == 0) THEN  ! CB02 option
      
      ! Compute condensate amount
      IF (ZQ1(JIJ,JK) > 0.0) THEN
        IF (ZQ1(JIJ,JK) <= 2.0) THEN
          ZCOND_TMP = MIN(EXP(-1.0) + 0.66 * ZQ1(JIJ,JK) + &
                         0.086 * ZQ1(JIJ,JK)**2, 2.0)
        ELSE
          ZCOND_TMP = ZQ1(JIJ,JK)
        END IF
      ELSE
        ZCOND_TMP = EXP(1.2 * ZQ1(JIJ,JK) - 1.0)
      END IF
      ZCOND_TMP = ZCOND_TMP * ZSIGMA(JIJ,JK)
      
      ! Cloud fraction
      IF (ZCOND_TMP >= 1.E-12) THEN
        PCLDFR(JIJ,JK) = MAX(0.0, MIN(1.0, 0.5 + 0.36 * ATAN(1.55 * ZQ1(JIJ,JK))))
      ELSE
        PCLDFR(JIJ,JK) = 0.0
      END IF
      
      IF (PCLDFR(JIJ,JK) == 0.0) THEN
        ZCOND_TMP = 0.0
      END IF
      
      ZCOND(JIJ,JK) = ZCOND_TMP
      
      ! Separate liquid and solid condensates
      IF (.NOT. OCND2) THEN
        PRC_OUT(JIJ,JK) = (1.0 - ZFRAC_TMP) * ZCOND_TMP
        PRI_OUT(JIJ,JK) = ZFRAC_TMP * ZCOND_TMP
        
        ! Update temperature
        PT_OUT(JIJ,JK) = PT(JIJ,JK) + &
                        ((PRC_OUT(JIJ,JK) - PRC_IN(JIJ,JK)) * PLV(JIJ,JK) + &
                         (PRI_OUT(JIJ,JK) - PRI_IN(JIJ,JK)) * PLS(JIJ,JK)) / PCPH(JIJ,JK)
        
        ! Update vapor
        PRV_OUT(JIJ,JK) = ZRT - PRC_OUT(JIJ,JK) - PRI_OUT(JIJ,JK) * ZPRIFACT
      END IF
      
    END IF  ! HCONDENS == 0
    
  END DO  ! JIJ
END DO  ! JK

END SUBROUTINE CONDENSATION

END MODULE MODE_CONDENSATION
