!MNH_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!MNH_LIC This is part of the Meso-NH software governed by the CeCILL-C licence
!MNH_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt
!MNH_LIC for details. version 1.
!     ######spl
      MODULE MODD_NEB_n
!     #############################
!> @file
!!****  *MODD_NEB_n* - OpenACC GPU-compatible version
!!      Declaration of nebulosity constants
!!
!!    PURPOSE
!!    -------
!!      The purpose of this declarative module is to declare some
!!      constants for nebulosity calculation
!!
!!    GPU-COMPATIBLE: Added !$acc declare copyin for NEB_MODEL structure
!
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!      None
!!
!!    REFERENCE
!!    ---------
!!
!!
!!    AUTHOR
!!    ------
!!       S. Riette (Meteo France)
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    24 Aug 2011
!!      OpenACC     20/12/2025 added !$acc declare for GPU
!-------------------------------------------------------------------------------
!
!*       0.   DECLARATIONS
!             ------------
!
USE MODD_PARAMETERS, ONLY: JPMODELMAX
IMPLICIT NONE
!
TYPE NEB_t
  REAL          :: XTMINMIX   !< minimum temperature of mixed phase
  REAL          :: XTMAXMIX   !< maximum temperature of mixed phase
  LOGICAL       :: LHGT_QS    !< Switch for height dependent VQSIGSAT
  CHARACTER(LEN=1) :: CFRAC_ICE_ADJUST     !< ice fraction for adjustments
  CHARACTER(LEN=1) :: CFRAC_ICE_SHALLOW_MF !< ice fraction for shallow_mf
  REAL               :: VSIGQSAT      !< coeff applied to qsat variance contribution
  CHARACTER(LEN=80)  :: CCONDENS      !< subrgrid condensation PDF
  CHARACTER(LEN=4)   :: CLAMBDA3      !< lambda3 choice for subgrid cloud scheme
  LOGICAL            :: LSTATNW       !< updated full statistical cloud scheme
  LOGICAL            :: LSIGMAS       !< Switch for using Sigma_s from turbulence scheme
  LOGICAL            :: LSUBG_COND    !< Switch for subgrid condensation
  LOGICAL            :: LCONDBORN     !< Switch to limit condensation
END TYPE NEB_t

TYPE(NEB_t), DIMENSION(JPMODELMAX), SAVE, TARGET :: NEB_MODEL
TYPE(NEB_t), POINTER, SAVE :: NEBN => NULL()

! OpenACC directive to copy nebulosity parameters to GPU device memory
!$acc declare copyin(NEB_MODEL)

REAL, POINTER :: XTMINMIX=>NULL(), &
                 XTMAXMIX=>NULL()
LOGICAL, POINTER :: LHGT_QS=>NULL()
CHARACTER(LEN=1), POINTER :: CFRAC_ICE_ADJUST => NULL()
CHARACTER(LEN=1), POINTER :: CFRAC_ICE_SHALLOW_MF => NULL()
REAL, POINTER :: VSIGQSAT=>NULL()
CHARACTER(LEN=80),POINTER :: CCONDENS=>NULL()
CHARACTER(LEN=4),POINTER :: CLAMBDA3=>NULL()
LOGICAL, POINTER :: LSTATNW=>NULL()
LOGICAL, POINTER :: LSIGMAS=>NULL()
LOGICAL, POINTER :: LSUBG_COND=>NULL()
LOGICAL, POINTER :: LCONDBORN=>NULL()
!
NAMELIST/NAM_NEBn/XTMINMIX, XTMAXMIX, LHGT_QS, CFRAC_ICE_ADJUST, CFRAC_ICE_SHALLOW_MF, &
                 &VSIGQSAT, CCONDENS, CLAMBDA3, LSTATNW, LSIGMAS, LSUBG_COND, LCONDBORN
!
!===============================================================================
! USAGE NOTES FOR GPU:
!
! The NEB_MODEL structure contains nebulosity scheme configuration parameters.
! These are small data (< 1 KB) but accessed frequently in GPU kernels.
!
! INITIALIZATION:
! 1. Initialize NEB_MODEL on CPU using NEBN_INIT()
! 2. Call NEB_GOTO_MODEL() to set active model
! 3. Update GPU copy: !$acc update device(NEB_MODEL)
!
! ACCESSED IN GPU KERNELS:
! - COMPUTE_FRAC_ICE uses NEBN%XTMINMIX and NEBN%XTMAXMIX
! - CONDENSATION uses NEBN%CFRAC_ICE_ADJUST, NEBN%CCONDENS, NEBN%CLAMBDA3
!
! Example usage in host code:
!   CALL NEBN_INIT(...)         ! Initialize on CPU
!   CALL NEB_GOTO_MODEL(1, 1)   ! Set active model
!   !$acc update device(NEB_MODEL)  ! Transfer to GPU
!   CALL CONDENSATION(...)      ! Can now access NEB_MODEL on GPU
!===============================================================================
!
!-------------------------------------------------------------------------------
!
CONTAINS
SUBROUTINE NEB_GOTO_MODEL(KFROM, KTO)
!! This subroutine associate all the pointers to the right component of
!! the right strucuture. A value can be accessed through the structure NEBN
!! or through the strucuture NEB_MODEL(KTO) or directly through these pointers.
IMPLICIT NONE
INTEGER, INTENT(IN) :: KFROM, KTO
!
IF(.NOT. ASSOCIATED(NEBN, NEB_MODEL(KTO))) THEN
  !
  NEBN => NEB_MODEL(KTO)
  !
  XTMINMIX => NEBN%XTMINMIX
  XTMAXMIX => NEBN%XTMAXMIX
  LHGT_QS => NEBN%LHGT_QS
  CFRAC_ICE_ADJUST => NEBN%CFRAC_ICE_ADJUST
  CFRAC_ICE_SHALLOW_MF => NEBN%CFRAC_ICE_SHALLOW_MF
  VSIGQSAT => NEBN%VSIGQSAT
  CCONDENS => NEBN%CCONDENS
  CLAMBDA3 => NEBN%CLAMBDA3
  LSTATNW => NEBN%LSTATNW
  LSIGMAS => NEBN%LSIGMAS
  LSUBG_COND => NEBN%LSUBG_COND
  LCONDBORN => NEBN%LCONDBORN
  !
  ! Update GPU copy after changing active model
  !$acc update device(NEB_MODEL)
  !
ENDIF
END SUBROUTINE NEB_GOTO_MODEL
!
SUBROUTINE NEBN_INIT(HPROGRAM, TFILENAM, LDNEEDNAM, KLUOUT, &
                    &LDDEFAULTVAL, LDREADNAM, LDCHECK, KPRINT)
!!*** *NEBN_INIT* - Code needed to initialize the MODD_NEB_n module
!!
!!*   PURPOSE
!!    -------
!!    Sets the default values, reads the namelist, performs the checks and prints
!!
!!*   METHOD
!!    ------
!!    0. Declarations
!!       1. Declaration of arguments
!!       2. Declaration of local variables
!!    1. Default values
!!    2. Namelist
!!    3. Checks
!!    4. Prints
!!
!!    AUTHOR
!!    ------
!!    S. Riette
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    Mar 2023
!!      OpenACC     Dec 2025 - Added GPU update after initialization
!-------------------------------------------------------------------------------
!
!*      0. DECLARATIONS
!       ---------------
!
USE MODE_POSNAM_PHY, ONLY: POSNAM_PHY
USE MODE_CHECK_NAM_VAL, ONLY: CHECK_NAM_VAL_CHAR
USE MODD_IO,  ONLY: TFILEDATA
!
IMPLICIT NONE
!
!* 0.1. Declaration of arguments
!       ------------------------
!
CHARACTER(LEN=6),  INTENT(IN) :: HPROGRAM     !< Name of the calling program
TYPE(TFILEDATA),   INTENT(IN) :: TFILENAM     !< Namelist file
LOGICAL,           INTENT(IN) :: LDNEEDNAM    !< True to abort if namelist is absent
INTEGER,           INTENT(IN) :: KLUOUT       !< Logical unit for outputs
LOGICAL, OPTIONAL, INTENT(IN) :: LDDEFAULTVAL !< Must we initialize variables with default values (defaults to .TRUE.)
LOGICAL, OPTIONAL, INTENT(IN) :: LDREADNAM    !< Must we read the namelist (defaults to .TRUE.)
LOGICAL, OPTIONAL, INTENT(IN) :: LDCHECK      !< Must we perform some checks on values (defaults to .TRUE.)
INTEGER, OPTIONAL, INTENT(IN) :: KPRINT       !< Print level (defaults to 0): 0 for no print, 1 to safely print namelist,
                                              !! 2 to print informative messages
!
!* 0.2 Declaration of local variables
!      ------------------------------
!
LOGICAL :: LLDEFAULTVAL, LLREADNAM, LLCHECK, LLFOUND
INTEGER :: IPRINT

LLDEFAULTVAL=.TRUE.
LLREADNAM=.TRUE.
LLCHECK=.TRUE.
IPRINT=0
IF(PRESENT(LDDEFAULTVAL)) LLDEFAULTVAL=LDDEFAULTVAL
IF(PRESENT(LDREADNAM   )) LLREADNAM   =LDREADNAM
IF(PRESENT(LDCHECK     )) LLCHECK     =LDCHECK
IF(PRESENT(KPRINT      )) IPRINT      =KPRINT
!
!*      1. DEFAULT VALUES
!       -----------------
!
IF(LLDEFAULTVAL) THEN
  !NOTES ON GENERAL DEFAULTS AND MODEL-SPECIFIC DEFAULTS :
  !- General default values *MUST* remain unchanged.
  !- To change the default value for a given application,
  !  an "IF(HPROGRAM=='...')" condition must be used.

  !Freezing between 0 and -20. Other possibilities are 0/-40 or -5/-25
  XTMAXMIX    = 273.16
  XTMINMIX    = 253.16
  LHGT_QS     = .FALSE.
  CFRAC_ICE_ADJUST='S'
  CFRAC_ICE_SHALLOW_MF='S'
  VSIGQSAT  = 0.02
  CCONDENS='CB02'
  CLAMBDA3='CB'
  LSUBG_COND=.FALSE.
  LCONDBORN=.FALSE.
  LSIGMAS   =.TRUE.
  LSTATNW=.FALSE.

  IF(HPROGRAM=='AROME') THEN
    CFRAC_ICE_ADJUST='T'
    CFRAC_ICE_SHALLOW_MF='T'
    VSIGQSAT=0.
    LSIGMAS=.FALSE.
  ELSEIF(HPROGRAM=='LMDZ') THEN
    LSUBG_COND=.TRUE.
  ENDIF
ENDIF
!
!*      2. NAMELIST
!       -----------
!
IF(LLREADNAM) THEN
  CALL POSNAM_PHY(TFILENAM, 'NAM_NEBN', LDNEEDNAM, LLFOUND)
  IF(LLFOUND) READ(UNIT=TFILENAM%NLU, NML=NAM_NEBn)
ENDIF
!
!*      3. CHECKS
!       ---------
!
IF(LLCHECK) THEN
  CALL CHECK_NAM_VAL_CHAR(KLUOUT, 'CFRAC_ICE_ADJUST', CFRAC_ICE_ADJUST, 'T', 'O', 'N', 'S')
  CALL CHECK_NAM_VAL_CHAR(KLUOUT, 'CFRAC_ICE_SHALLOW_MF', CFRAC_ICE_SHALLOW_MF, 'T', 'O', 'N', 'S')
  CALL CHECK_NAM_VAL_CHAR(KLUOUT, 'CCONDENS', CCONDENS, 'CB02', 'GAUS')
  CALL CHECK_NAM_VAL_CHAR(KLUOUT, 'CLAMBDA3', CLAMBDA3, 'CB', 'NONE')
ENDIF
!
!*      3. PRINTS
!       ---------
!
IF(IPRINT>=1) THEN
  WRITE(UNIT=KLUOUT,NML=NAM_NEBn)
ENDIF
!
! Update GPU copy after initialization
!$acc update device(NEB_MODEL)
!
END SUBROUTINE NEBN_INIT
!
END MODULE MODD_NEB_n
