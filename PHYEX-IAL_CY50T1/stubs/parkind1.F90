!     ##################
      MODULE PARKIND1
!     ##################
!
!!****  *PARKIND1* - Defines kind parameters for real and integer types
!!
!!    PURPOSE
!!    -------
!     Stub module to provide kind parameters when IFS libraries are not available
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
! Integer kinds
INTEGER, PARAMETER :: JPIT = SELECTED_INT_KIND(9)   ! Integer working precision
INTEGER, PARAMETER :: JPIS = SELECTED_INT_KIND(4)   ! Short integer
INTEGER, PARAMETER :: JPIB = SELECTED_INT_KIND(2)   ! Byte integer
!
! Real kinds  
INTEGER, PARAMETER :: JPRM = SELECTED_REAL_KIND(13,300) ! Double precision real
INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300) ! Real working precision (alias)
INTEGER, PARAMETER :: JPRS = SELECTED_REAL_KIND(6,37)   ! Single precision real
INTEGER, PARAMETER :: JPRD = SELECTED_REAL_KIND(13,300) ! Double precision (alias)
!
! Logical kinds
INTEGER, PARAMETER :: JPLM = KIND(.TRUE.) ! Logical working precision
!
END MODULE PARKIND1
