!     ##################
      MODULE YOMHOOK
!     ##################
!
!!****  *YOMHOOK* - Dr Hook profiling interface
!!
!!    PURPOSE
!!    -------
!     Stub module to provide Dr Hook profiling hooks when IFS libraries are not available
!     All profiling calls are disabled in this stub version
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
USE PARKIND1, ONLY : JPRB
!
IMPLICIT NONE
!
! Logical flag to enable/disable profiling (always false in stub)
LOGICAL, PARAMETER :: LHOOK = .FALSE.
!
! Hook handle type
INTEGER, PARAMETER :: JPHOOK = KIND(0.0_JPRB)
!
CONTAINS
!
!     ################################################################
      SUBROUTINE DR_HOOK(CDNAME, KSWITCH, PKEY)
!     ################################################################
!
!!****  *DR_HOOK* - Dummy profiling routine
!!
!!    PURPOSE
!!    -------
!     Provides a no-op implementation of DR_HOOK for profiling
!
!!**  METHOD
!!    ------
!     Does nothing - profiling is disabled in standalone mode
!
      IMPLICIT NONE
!
      CHARACTER(LEN=*), INTENT(IN) :: CDNAME   ! Routine name
      INTEGER, INTENT(IN)          :: KSWITCH  ! 0=enter, 1=exit
      REAL(KIND=JPHOOK), INTENT(INOUT) :: PKEY ! Key for this routine
!
      ! Do nothing - profiling disabled
      PKEY = 0.0_JPRB
!
      END SUBROUTINE DR_HOOK
!
END MODULE YOMHOOK
