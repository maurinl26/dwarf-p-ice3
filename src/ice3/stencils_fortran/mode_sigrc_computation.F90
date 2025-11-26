module mode_sigrc_computation

IMPLICIT None

CONTAINS

! Global lookup table for SIGRC computation (CB scheme)
REAL, DIMENSION(34), SAVE :: SRC_1D = (/&
  & 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, &
  & 0.0000, 0.0000, 0.0001, 0.0002, 0.0005, 0.0010, &
  & 0.0020, 0.0039, 0.0072, 0.0124, 0.0199, 0.0301, &
  & 0.0435, 0.0601, 0.0796, 0.1014, 0.1245, 0.1476, &
  & 0.1695, 0.1888, 0.2046, 0.2165, 0.2240, 0.2274, &
  & 0.2274, 0.2260, 0.2247, 0.2239/)


!-------------------------------------------------------------------------------
!
!     ##############################################################################
      SUBROUTINE SIGRC_COMPUTATION(NIJT, NKT, NKTE, NKTB, NIJE, NIJB, &
                                   HLAMBDA3, ZQ1, PSIGRC, INQ1)
!     ##############################################################################
!
!!****  *SIGRC_COMPUTATION* - Compute sigma_rc from lookup table
!!
!!    PURPOSE
!!    -------
!!      Compute subgrid standard deviation of rc using lookup table
!!
IMPLICIT NONE

INTEGER, INTENT(IN) :: NIJT, NKT, NKTE, NKTB, NIJE, NIJB
INTEGER, INTENT(IN) :: HLAMBDA3
REAL, DIMENSION(NIJT,NKT), INTENT(IN) :: ZQ1
REAL, DIMENSION(NIJT,NKT), INTENT(OUT) :: PSIGRC
INTEGER, DIMENSION(NIJT,NKT), INTENT(OUT) :: INQ1

INTEGER :: JIJ, JK, INQ2
REAL :: ZINC

DO JK = NKTB, NKTE
  DO JIJ = NIJB, NIJE
    
    INQ1(JIJ,JK) = FLOOR(MIN(100.0, MAX(-100.0, 2.0 * ZQ1(JIJ,JK))))
    INQ2 = MIN(MAX(-22, INQ1(JIJ,JK)), 10)
    
    ZINC = 2.0 * ZQ1(JIJ,JK) - REAL(INQ2)
    PSIGRC(JIJ,JK) = MIN(1.0, (1.0 - ZINC) * SRC_1D(INQ2 + 23) + &
                                     ZINC * SRC_1D(INQ2 + 24))
    
  END DO
END DO

END SUBROUTINE SIGRC_COMPUTATION

!-------------------------------------------------------------------------------
!
!     ##############################################################################
      SUBROUTINE GLOBAL_TABLE(OUT_TABLE)
!     ##############################################################################
!
!!****  *GLOBAL_TABLE* - Return the global lookup table
!!
!!    PURPOSE
!!    -------
!!      Provide access to the SRC_1D lookup table
!!
IMPLICIT NONE

REAL, DIMENSION(34), INTENT(OUT) :: OUT_TABLE

OUT_TABLE = SRC_1D

END SUBROUTINE GLOBAL_TABLE

!-------------------------------------------------------------------------------

end module mode_sigrc_computation