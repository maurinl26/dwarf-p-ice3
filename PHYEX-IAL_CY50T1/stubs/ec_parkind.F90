!     ##################
      MODULE EC_PARKIND
!     ##################
!
!!****  *EC_PARKIND* - Alias module for PARKIND1 for IFS compatibility
!!
!!    PURPOSE
!!    -------
!     Stub module to provide EC_PARKIND as an alias for PARKIND1
!
!!    AUTHOR
!!    ------
!!      Stub implementation for standalone compilation
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    01/2025
!-------------------------------------------------------------------------------
!
USE PARKIND1
!
! Re-export integer parameter with IFS naming
INTEGER, PARAMETER :: JPIM = JPIT  ! Standard integer for IFS compatibility
!
END MODULE EC_PARKIND
