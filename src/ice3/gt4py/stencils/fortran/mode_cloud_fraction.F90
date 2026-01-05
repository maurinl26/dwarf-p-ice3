!MNH_LIC Copyright 1994-2023 CNRS, Meteo-France and Universite Paul Sabatier
!MNH_LIC This is part of the Meso-NH software governed by the CeCILL-C licence
!MNH_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt
!MNH_LIC for details. version 1.
!-----------------------------------------------------------------
!     ###########################
      MODULE MODE_CLOUD_FRACTION
!     ###########################
!
!!****  *MODE_CLOUD_FRACTION* - Contains cloud fraction routines from PHYEX
!!
!!    PURPOSE
!!    -------
!!      Compute cloud fraction and related microphysical adjustments
!!      including subgrid autoconversion for liquid and ice
!!
!!    REFERENCE
!!    ---------
!!      PHYEX-IAL_CY50T1/src/common/micro/ice_adjust.F90
!!      Lines 278-419, 450-473
!!
!!    AUTHOR
!!    ------
!!      Generated from PHYEX reference for dwarf-p-ice3 project
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    23/11/2024
!-------------------------------------------------------------------------------
!
IMPLICIT NONE

CONTAINS

!-------------------------------------------------------------------------------
!
!     ##############################################################################
      SUBROUTINE THERMODYNAMIC_FIELDS(NIJB, NIJE, NKTB, NKTE, NIJT, NKT, &
                                     NRR, &
                                     CPD, &
                                     PTH, PEXN, PRV, PRC, PRR, PRI, PRS, PRG, &
                                     PLV, PLS, PCPH, PT)
!     ##############################################################################
!
!!****  *THERMODYNAMIC_FIELDS* - Compute thermodynamic fields
!!
!!    PURPOSE
!!    -------
!!      Compute temperature, latent heats, and specific heat for moist air
!!
!!    METHOD
!!    ------
!!      From PHYEX ice_adjust.F90, lines 450-473
!!
!!    REFERENCE
!!    ---------
!!      PHYEX-IAL_CY50T1/src/common/micro/ice_adjust.F90 (l450-l473)
!!
!!    AUTHOR
!!    ------
!!      Generated from PHYEX
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    23/11/2024
!-------------------------------------------------------------------------------
!
IMPLICIT NONE
!
!*       0.1   Declarations of dummy arguments
!
INTEGER, INTENT(IN) :: NIJB, NIJE, NKTB, NKTE, NIJT, NKT
INTEGER, INTENT(IN) :: NRR
REAL, INTENT(IN) :: CPD

REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: PTH, PEXN
REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: PRV, PRC, PRR, PRI, PRS, PRG

REAL, DIMENSION(SIZE(PTH, 1), SIZE(PTH, 2)), INTENT(OUT) :: PLV, PLS, PCPH, PT

!*       0.2   Declarations of local variables
!
INTEGER :: JIJ, JK
REAL :: ZLVOCPEXN, ZLSOCPEXN

! Constants for latent heat computation
REAL, PARAMETER :: XLVTT = 2.5008E6  ! Vaporization latent heat at TT
REAL, PARAMETER :: XLSTT = 2.8345E6  ! Sublimation latent heat at TT
REAL, PARAMETER :: XTT = 273.16      ! Triple point temperature
REAL, PARAMETER :: XCPV = 1846.1     ! Specific heat of water vapor
REAL, PARAMETER :: XCL = 4218.0      ! Specific heat of liquid water
REAL, PARAMETER :: XCI = 2106.0      ! Specific heat of ice

!-------------------------------------------------------------------------------
!
!*       1.    COMPUTE TEMPERATURE AND LATENT HEATS
!              -------------------------------------
!
DO JK = NKTB, NKTE
  DO JIJ = NIJB, NIJE
    
    ! 2.3 Compute temperature
    PT(JIJ,JK) = PTH(JIJ,JK) * PEXN(JIJ,JK)
    
    ! Compute latent heat of vaporization
    PLV(JIJ,JK) = XLVTT + (XCPV - XCL) * (PT(JIJ,JK) - XTT)
    
    ! Compute latent heat of sublimation
    PLS(JIJ,JK) = XLSTT + (XCPV - XCI) * (PT(JIJ,JK) - XTT)
    
  END DO
END DO

!-------------------------------------------------------------------------------
!
!*       2.    COMPUTE SPECIFIC HEAT FOR MOIST AIR
!              ------------------------------------
!
DO JK = NKTB, NKTE
  DO JIJ = NIJB, NIJE
    
    ! Number of moist variables fixed to NRR parameter
    IF (NRR == 6) THEN
      PCPH(JIJ,JK) = CPD + XCPV * PRV(JIJ,JK) + &
                     XCL * (PRC(JIJ,JK) + PRR(JIJ,JK)) + &
                     XCI * (PRI(JIJ,JK) + PRS(JIJ,JK) + PRG(JIJ,JK))
    ELSE IF (NRR == 5) THEN
      PCPH(JIJ,JK) = CPD + XCPV * PRV(JIJ,JK) + &
                     XCL * (PRC(JIJ,JK) + PRR(JIJ,JK)) + &
                     XCI * (PRI(JIJ,JK) + PRS(JIJ,JK))
    ELSE IF (NRR == 4) THEN
      PCPH(JIJ,JK) = CPD + XCPV * PRV(JIJ,JK) + &
                     XCL * (PRC(JIJ,JK) + PRR(JIJ,JK))
    ELSE IF (NRR == 2) THEN
      PCPH(JIJ,JK) = CPD + XCPV * PRV(JIJ,JK) + &
                     XCL * PRC(JIJ,JK) + XCI * PRI(JIJ,JK)
    END IF
    
  END DO
END DO

END SUBROUTINE THERMODYNAMIC_FIELDS

!-------------------------------------------------------------------------------
!
!     ##############################################################################
      SUBROUTINE CLOUD_FRACTION_1(NIJB, NIJE, NKTB, NKTE, NIJT, NKT, &
                                 PLV, PLS, PCPH, PEXNREF, &
                                 PRC, PRI, &
                                 PTHS, PRVS, PRCS, PRIS, &
                                 PRC_TMP, PRI_TMP, &
                                 PDT)
!     ##############################################################################
!
!!****  *CLOUD_FRACTION_1* - First part of cloud fraction computation
!!
!!    PURPOSE
!!    -------
!!      Compute sources after condensation loop and prepare for cloud fraction
!!
!!    METHOD
!!    ------
!!      From PHYEX ice_adjust.F90, lines 278-312
!!      Compute variation of mixing ratios and update sources
!!
!!    REFERENCE
!!    ---------
!!      PHYEX-IAL_CY50T1/src/common/micro/ice_adjust.F90 (l278-l312)
!!
!!    AUTHOR
!!    ------
!!      Generated from PHYEX
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    23/11/2024
!-------------------------------------------------------------------------------
!
IMPLICIT NONE
!
!*       0.1   Declarations of dummy arguments
!
INTEGER, INTENT(IN) :: NIJB, NIJE, NKTB, NKTE, NIJT, NKT
REAL, INTENT(IN) :: PDT

REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: PLV, PLS, PCPH, PEXNREF
REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: PRC, PRI
REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: PRC_TMP, PRI_TMP

REAL, DIMENSION(NIJT,NKT), INTENT(INOUT) :: PTHS, PRVS, PRCS, PRIS

!*       0.2   Declarations of local variables
!
INTEGER :: JIJ, JK
REAL :: ZW1, ZW2

!-------------------------------------------------------------------------------
!
!*       5.0   COMPUTE THE VARIATION OF MIXING RATIO
!              --------------------------------------
!
DO JK = NKTB, NKTE
  DO JIJ = NIJB, NIJE
    
    ! Compute variation rates for liquid and ice
    ZW1 = (PRC_TMP(JIJ,JK) - PRC(JIJ,JK)) / PDT
    ZW2 = (PRI_TMP(JIJ,JK) - PRI(JIJ,JK)) / PDT
    
    !---------------------------------------------------------------------------
    !
    !*       5.1   COMPUTE THE SOURCES
    !              -------------------
    !
    ! Liquid water source
    IF (ZW1 < 0.0) THEN
      ZW1 = MAX(ZW1, -PRCS(JIJ,JK))
    ELSE
      ZW1 = MIN(ZW1, PRVS(JIJ,JK))
    END IF
    
    PRVS(JIJ,JK) = PRVS(JIJ,JK) - ZW1
    PRCS(JIJ,JK) = PRCS(JIJ,JK) + ZW1
    PTHS(JIJ,JK) = PTHS(JIJ,JK) + ZW1 * PLV(JIJ,JK) / (PCPH(JIJ,JK) * PEXNREF(JIJ,JK))
    
    ! Ice source
    IF (ZW2 < 0.0) THEN
      ZW2 = MAX(ZW2, -PRIS(JIJ,JK))
    ELSE
      ZW2 = MIN(ZW2, PRVS(JIJ,JK))
    END IF
    
    PRVS(JIJ,JK) = PRVS(JIJ,JK) - ZW2
    PRIS(JIJ,JK) = PRIS(JIJ,JK) + ZW2
    PTHS(JIJ,JK) = PTHS(JIJ,JK) + ZW2 * PLS(JIJ,JK) / (PCPH(JIJ,JK) * PEXNREF(JIJ,JK))
    
  END DO
END DO

END SUBROUTINE CLOUD_FRACTION_1

!-------------------------------------------------------------------------------
!
!     ##############################################################################
      SUBROUTINE CLOUD_FRACTION_2(NIJB, NIJE, NKTB, NKTE, NIJT, NKT, &
                                 LSUBG_COND, SUBG_MF_PDF, &
                                 ACRIAUTI, BCRIAUTI, CRIAUTC, CRIAUTI, TT, &
                                 PRHODREF, PEXNREF, PT, PCPH, PLV, PLS, &
                                 PTHS, PRVS, PRCS, PRIS, &
                                 PRC_MF, PRI_MF, PCF_MF, &
                                 PCLDFR, PHLC_HRC, PHLC_HCF, PHLI_HRI, PHLI_HCF, &
                                 PDT)
!     ##############################################################################
!
!!****  *CLOUD_FRACTION_2* - Second part of cloud fraction computation
!!
!!    PURPOSE
!!    -------
!!      Compute cloud fraction with subgrid autoconversion
!!
!!    METHOD
!!    ------
!!      From PHYEX ice_adjust.F90, lines 313-419
!!      Apply CB02 scheme with subgrid mean-flux PDF
!!      Compute droplet and ice subgrid autoconversion
!!
!!    REFERENCE
!!    ---------
!!      PHYEX-IAL_CY50T1/src/common/micro/ice_adjust.F90 (l313-l419)
!!
!!    AUTHOR
!!    ------
!!      Generated from PHYEX
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    23/11/2024
!-------------------------------------------------------------------------------
!
IMPLICIT NONE
!
!*       0.1   Declarations of dummy arguments
!
INTEGER, INTENT(IN) :: NIJB, NIJE, NKTB, NKTE, NIJT, NKT
LOGICAL, INTENT(IN) :: LSUBG_COND
INTEGER, INTENT(IN) :: SUBG_MF_PDF
REAL, INTENT(IN) :: ACRIAUTI, BCRIAUTI, CRIAUTC, CRIAUTI, TT
REAL, INTENT(IN) :: PDT

REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: PRHODREF, PEXNREF, PT, PCPH, PLV, PLS
REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: PRC_MF, PRI_MF, PCF_MF

REAL, DIMENSION(NIJT,NKT), INTENT(INOUT) :: PTHS, PRVS, PRCS, PRIS
REAL, DIMENSION(NIJT,NKT), INTENT(INOUT) :: PCLDFR
REAL, DIMENSION(NIJT,NKT), INTENT(INOUT) :: PHLC_HRC, PHLC_HCF
REAL, DIMENSION(NIJT,NKT), INTENT(INOUT) :: PHLI_HRI, PHLI_HCF

!*       0.2   Declarations of local variables
!
INTEGER :: JIJ, JK
REAL :: ZW1, ZW2, ZCRIAUT
REAL :: ZHCF, ZHR, ZHRI
REAL, PARAMETER :: ZEPS = 1.E-20

!-------------------------------------------------------------------------------
!
!*       5.2   COMPUTE THE CLOUD FRACTION CLDFR
!              ---------------------------------
!
DO JK = NKTB, NKTE
  DO JIJ = NIJB, NIJE
    
    IF (.NOT. LSUBG_COND) THEN
      ! Simple cloud fraction scheme
      IF ((PRCS(JIJ,JK) + PRIS(JIJ,JK)) * PDT > 1.E-12) THEN
        PCLDFR(JIJ,JK) = 1.0
      ELSE
        PCLDFR(JIJ,JK) = 0.0
      END IF
      
    ELSE
      ! Subgrid condensation scheme (AROME)
      ZW1 = PRC_MF(JIJ,JK) / PDT
      ZW2 = PRI_MF(JIJ,JK) / PDT
      
      ! Adjust if sum exceeds vapor content
      IF (ZW1 + ZW2 > PRVS(JIJ,JK)) THEN
        ZW1 = ZW1 * PRVS(JIJ,JK) / (ZW1 + ZW2)
        ZW2 = PRVS(JIJ,JK) - ZW1
      END IF
      
      ! Update cloud fraction
      PCLDFR(JIJ,JK) = MIN(1.0, PCLDFR(JIJ,JK) + PCF_MF(JIJ,JK))
      
      ! Update mixing ratios and temperature
      PRCS(JIJ,JK) = PRCS(JIJ,JK) + ZW1
      PRIS(JIJ,JK) = PRIS(JIJ,JK) + ZW2
      PRVS(JIJ,JK) = PRVS(JIJ,JK) - ZW1 - ZW2
      PTHS(JIJ,JK) = PTHS(JIJ,JK) + &
                     (ZW1 * PLV(JIJ,JK) + ZW2 * PLS(JIJ,JK)) / &
                     (PCPH(JIJ,JK) * PEXNREF(JIJ,JK))
      
      !-------------------------------------------------------------------------
      !
      !*       5.3   DROPLETS SUBGRID AUTOCONVERSION
      !              --------------------------------
      !
      ZCRIAUT = CRIAUTC / PRHODREF(JIJ,JK)
      
      ! SUBG_MF_PDF == 0 (None scheme)
      IF (SUBG_MF_PDF == 0) THEN
        IF (ZW1 * PDT > PCF_MF(JIJ,JK) * ZCRIAUT) THEN
          PHLC_HRC(JIJ,JK) = PHLC_HRC(JIJ,JK) + ZW1 * PDT
          PHLC_HCF(JIJ,JK) = MIN(1.0, PHLC_HCF(JIJ,JK) + PCF_MF(JIJ,JK))
        END IF
      END IF
      
      ! SUBG_MF_PDF == 1 (Triangle scheme)
      IF (SUBG_MF_PDF == 1) THEN
        IF (ZW1 * PDT > PCF_MF(JIJ,JK) * ZCRIAUT) THEN
          ZHCF = 1.0 - 0.5 * (ZCRIAUT * PCF_MF(JIJ,JK) / MAX(ZEPS, ZW1 * PDT))**2
          ZHR = ZW1 * PDT - (ZCRIAUT * PCF_MF(JIJ,JK))**3 / &
                (3.0 * MAX(ZEPS, (ZW1 * PDT)**2))
          
        ELSE IF (2.0 * ZW1 * PDT <= PCF_MF(JIJ,JK) * ZCRIAUT) THEN
          ZHCF = 0.0
          ZHR = 0.0
          
        ELSE
          ZHCF = (2.0 * ZW1 * PDT - ZCRIAUT * PCF_MF(JIJ,JK))**2 / &
                 (2.0 * MAX(ZEPS, (ZW1 * PDT)**2))
          ZHR = (4.0 * (ZW1 * PDT)**3 - &
                 3.0 * ZW1 * PDT * (ZCRIAUT * PCF_MF(JIJ,JK))**2 + &
                 (ZCRIAUT * PCF_MF(JIJ,JK))**3) / &
                (3.0 * MAX(ZEPS, (ZW1 * PDT)**2))
        END IF
        
        ZHCF = ZHCF * PCF_MF(JIJ,JK)
        PHLC_HCF(JIJ,JK) = MIN(1.0, PHLC_HCF(JIJ,JK) + ZHCF)
        PHLC_HRC(JIJ,JK) = PHLC_HRC(JIJ,JK) + ZHR
      END IF
      
      !-------------------------------------------------------------------------
      !
      !*       5.4   ICE SUBGRID AUTOCONVERSION
      !              ---------------------------
      !
      ZCRIAUT = MIN(CRIAUTI, 10.0**(ACRIAUTI * (PT(JIJ,JK) - TT) + BCRIAUTI))
      
      ! SUBG_MF_PDF == 0 (None scheme)
      IF (SUBG_MF_PDF == 0) THEN
        IF (ZW2 * PDT > PCF_MF(JIJ,JK) * ZCRIAUT) THEN
          PHLI_HRI(JIJ,JK) = PHLI_HRI(JIJ,JK) + ZW2 * PDT
          PHLI_HCF(JIJ,JK) = MIN(1.0, PHLI_HCF(JIJ,JK) + PCF_MF(JIJ,JK))
        END IF
      END IF
      
      ! SUBG_MF_PDF == 1 (Triangle scheme)
      IF (SUBG_MF_PDF == 1) THEN
        IF (ZW2 * PDT > PCF_MF(JIJ,JK) * ZCRIAUT) THEN
          ZHCF = 1.0 - 0.5 * ((ZCRIAUT * PCF_MF(JIJ,JK)) / (ZW2 * PDT))**2
          ZHRI = ZW2 * PDT - (ZCRIAUT * PCF_MF(JIJ,JK))**3 / &
                 (3.0 * (ZW2 * PDT)**2)
          
        ELSE IF (2.0 * ZW2 * PDT <= PCF_MF(JIJ,JK) * ZCRIAUT) THEN
          ZHCF = 0.0
          ZHRI = 0.0
          
        ELSE
          ZHCF = (2.0 * ZW2 * PDT - ZCRIAUT * PCF_MF(JIJ,JK))**2 / &
                 (2.0 * (ZW2 * PDT)**2)
          ZHRI = (4.0 * (ZW2 * PDT)**3 - &
                  3.0 * ZW2 * PDT * (ZCRIAUT * PCF_MF(JIJ,JK))**2 + &
                  (ZCRIAUT * PCF_MF(JIJ,JK))**3) / &
                 (3.0 * (ZW2 * PDT)**2)
        END IF
        
        ZHCF = ZHCF * PCF_MF(JIJ,JK)
        PHLI_HCF(JIJ,JK) = MIN(1.0, PHLI_HCF(JIJ,JK) + ZHCF)
        PHLI_HRI(JIJ,JK) = PHLI_HRI(JIJ,JK) + ZHRI
      END IF
      
    END IF  ! LSUBG_COND
    
  END DO  ! JIJ
END DO  ! JK

END SUBROUTINE CLOUD_FRACTION_2

!-------------------------------------------------------------------------------

END MODULE MODE_CLOUD_FRACTION
