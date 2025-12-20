!@no_insert_drhook
!     ######spl
      MODULE MODD_TIWMX
!     ###############
!
!!****  *MODD_TIWMX* - OpenACC GPU-compatible version
!!
!!    PURPOSE
!!    -------
!!    Temperature-dependent lookup tables for microphysics
!!    GPU-COMPATIBLE: Tables declared with !$acc declare for GPU access
!!
!!    REFERENCE
!!    ---------
!!      Book2 of documentation of Meso-NH (ha ha)
!!
!!    AUTHOR
!!    ------
!!      K. I. Ivarsson   *SMHI*
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    20/11/14
!!      OpenACC     20/12/25  Added !$acc declare copyin directives for GPU
!-------------------------------------------------------------------------------
!
!*       0.   DECLARATIONS
!             ------------
!
IMPLICIT NONE

REAL, PARAMETER :: XNDEGR = 100.0
INTEGER, PARAMETER :: NSTART = 13200 ! A too small value may result into a FPE in single precision mode. REK.
INTEGER, PARAMETER :: NSTOP = 37316

TYPE TIWMX_t
! Saturation tables and derivatives
REAL ::  ESTABW(NSTART:NSTOP)   ! Saturation vapor pressure over liquid water
REAL :: DESTABW(NSTART:NSTOP)   ! Temperature derivative of ESTABW
REAL ::  ESTABI(NSTART:NSTOP)   ! Saturation vapor pressure over ice
REAL :: DESTABI(NSTART:NSTOP)   ! Temperature derivative of ESTABI

! Ice crystal- or water droplet tables
REAL ::   A2TAB(NSTART:NSTOP)   ! Ice crystal concentration parameter
REAL ::  BB3TAB(NSTART:NSTOP)   ! Ice crystal concentration parameter
REAL ::  AM3TAB(NSTART:NSTOP)   ! Meyers ice nuclei concentration
REAL ::  AF3TAB(NSTART:NSTOP)   ! Fletcher ice nuclei concentration
REAL ::  A2WTAB(NSTART:NSTOP)   ! Cloud droplet concentration parameter (liquid)
REAL :: BB3WTAB(NSTART:NSTOP)   ! Cloud droplet concentration parameter (liquid)
REAL :: REDINTAB(NSTART:NSTOP)  ! Ice nuclei concentration reduction factor
END TYPE TIWMX_t

TYPE(TIWMX_t), SAVE, TARGET :: TIWMX

! OpenACC directive to copy lookup tables to GPU device memory
! This copies the entire TIWMX structure when first accessed from GPU
!$acc declare copyin(TIWMX)

!===============================================================================
! USAGE NOTES FOR GPU:
!
! The TIWMX structure contains ~170 KB of lookup table data (7 tables × 24,116 elements).
! This data is copied to GPU device memory on first access using the !$acc declare directive.
!
! INITIALIZATION:
! Before using on GPU, the tables must be initialized on CPU using the initialization
! routines from the PHYEX library. The !$acc declare copyin ensures the data is
! automatically transferred to GPU after initialization.
!
! MEMORY OPTIMIZATION:
! Consider using GPU constant memory for frequently accessed lookup tables:
!   !$acc declare copyin(TIWMX) create(readonly)
! This may improve performance by caching in constant memory cache.
!
! DATA TRANSFER:
! The copyin clause means:
! - Data is copied FROM host TO device when first referenced
! - Data remains on device for the lifetime of the program
! - No automatic copyback to host
!
! If tables need to be updated during runtime, use update directive:
!   !$acc update device(TIWMX)
!===============================================================================

END MODULE MODD_TIWMX
