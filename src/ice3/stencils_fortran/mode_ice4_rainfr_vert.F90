! Created by  on 02/06/2025.

module mode_ice4_rainfr_vert

    IMPLICIT NONE
CONTAINS
SUBROUTINE ICE4_RAINFR_VERT(KIB, KIE, KIT, KJB, &
&KJE, KJT, KKB, KKE, KKT, KKL, &
&R_RTMIN, S_RTMIN, G_RTMIN, &
&PPRFR, PRR)
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
!
INTEGER,                      INTENT(IN) :: KIB, KIE, KIT, KJB, KJE, KJT, KKB, KKE, KKT, KKL
real, dimension(kit, kjt, kkt), intent(in) :: R_RTMIN
REAL, DIMENSION(KIT,KJT,KKT), INTENT(OUT) :: PPRFR !Precipitation fraction
REAL, DIMENSION(KIT,KJT,KKT), INTENT(IN)   :: PRR !Rain field
!
INTEGER :: NKB, NKE, NKL, NIJB, NIJE
!*       0.2  declaration of local variables
!
INTEGER :: JIJ, JK
LOGICAL :: MASK
!
DO JI = KIB,KIE
   DO JJ = KJB, KJE
      PPRFR(JI,JJ,KKE)=0.
      DO JK=KKE-KKL, KKB, -KKL
         IF (PRR(JI,JJ,JK) .GT. R_RTMIN) THEN
            PPRFR(JI,JJ,JK)=MAX(PPRFR(JI,JJ,JK),PPRFR(JI,JJ,JK+KKL))
            IF (PPRFR(JI,JJ,JK)==0) THEN
               PPRFR(JI,JJ,JK)=1.
            END IF
         ELSE
            PPRFR(JI,JJ,JK)=0.
         END IF
      END DO
   END DO
END DO

END SUBROUTINE ICE4_RAINFR_VERT

end module mode_ice4_rainfr_vert