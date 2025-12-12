!     ##################
      MODULE YOMLUN
!     ##################
!
!!****  *YOMLUN* - Stub module for IFS logical unit numbers
!!
!!    PURPOSE
!!    -------
!     Stub module to provide logical unit numbers for output
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
IMPLICIT NONE
!
! Logical unit for standard output
INTEGER :: NULOUT = 6  ! Standard output (stdout)
INTEGER :: NULERR = 0  ! Error output
INTEGER :: NULNAM = 0  ! Namelist input
!
END MODULE YOMLUN
