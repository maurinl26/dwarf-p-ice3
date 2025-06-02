! Created by  on 02/06/2025.

module mode_ice4_rainfr_vert

    IMPLICIT NONE
CONTAINS
SUBROUTINE ICE4_RAINFR_VERT(NKB, NKE, NKL, NIJB, NIJE, &
        &R_RTMIN, S_RTMIN, G_RTMIN, &
        &PPRFR, PRR, PRS, PRG, PRH)
!!
!!**  PURPOSE
!!    -------
!!      Computes the rain fraction
!!
!!    AUTHOR
!!    ------
!!      S. Riette from the plitting of rain_ice source code (nov. 2014)
!!
!!    MODIFICATIONS
!!    -------------
!!
!  P. Wautelet 13/02/2019: bugfix: intent of PPRFR OUT->INOUT
!  S. Riette 21/9/23: collapse JI/JJ
!
!
!*      0. DECLARATIONS
!          ------------
!
IMPLICIT NONE
!
!*       0.1   Declarations of dummy arguments :
!
INTEGER, intent(in) :: NKB, NKE, NKL, NIJB, NIJE
real, intent(in) :: R_RTMIN, S_RTMIN, G_RTMIN
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(INOUT) :: PPRFR !Precipitation fraction
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    :: PRR !Rain field
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    :: PRS !Snow field
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)    :: PRG !Graupel field
REAL, DIMENSION(D%NIJT,D%NKT), OPTIONAL, INTENT(IN)    :: PRH !Hail field
!
INTEGER :: NKB, NKE, NKL, NIJB, NIJE
!*       0.2  declaration of local variables
!
INTEGER :: JIJ, JK
LOGICAL :: MASK
!
!$acc kernels
PPRFR(NIJB:NIJE,NKE)=0.
!$acc end kernels
DO JK=NKE-NKL, NKB, -NKL
  IF(PRESENT(PRH)) THEN
!$acc kernels
!$acc loop independent
    DO JIJ = NIJB, NIJE
      MASK=PRR(JIJ,JK) > R_RTMIN .OR. PRS(JIJ,JK) > S_RTMIN &
      .OR. PRG(JIJ,JK) > G_RTMIN
      IF (MASK) THEN
        PPRFR(JIJ,JK)=MAX(PPRFR(JIJ,JK),PPRFR(JIJ,JK+NKL))
        IF (PPRFR(JIJ,JK)==0) THEN
          PPRFR(JIJ,JK)=1.
        END IF
      ELSE
        PPRFR(JIJ,JK)=0.
      END IF
    END DO
!$acc end kernels
  ELSE
!$acc kernels
!$acc loop independent
    DO JIJ = NIJB, NIJE
      MASK=PRR(JIJ,JK) > R_RTMIN .OR. PRS(JIJ,JK) > S_RTMIN &
      .OR. PRG(JIJ,JK) > G_RTMIN
      IF (MASK) THEN
        PPRFR(JIJ,JK)=MAX(PPRFR(JIJ,JK),PPRFR(JIJ,JK+NKL))
        IF (PPRFR(JIJ,JK)==0) THEN
          PPRFR(JIJ,JK)=1.
        END IF
      ELSE
        PPRFR(JIJ,JK)=0.
      END IF
    END DO
!$acc end kernels
  END IF
END DO
!
END SUBROUTINE ICE4_RAINFR_VERT

end module mode_ice4_rainfr_vert