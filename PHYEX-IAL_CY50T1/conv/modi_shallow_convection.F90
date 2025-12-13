!     #################################
      MODULE MODI_SHALLOW_CONVECTION
!     #################################
!
IMPLICIT NONE
INTERFACE
!
    SUBROUTINE SHALLOW_CONVECTION(CVP_SHAL, CST, D, NSV, CONVPAR, KBDIA, KTDIA, &
                                  KICE, OSETTADJ, PTADJS, PPABST, PZZ, &
                                  PTKECLS, PTT, PRVT, PRCT, PRIT, PWT, &
                                  PTTEN, PRVTEN, PRCTEN, PRITEN,       &
                                  KCLTOP, KCLBAS, PUMF, OCH1CONV, KCH1,&
                                  PCH1, PCH1TEN)
USE MODD_CST, ONLY : CST_T
USE MODD_CONVPAR_SHAL, ONLY: CONVPAR_SHAL
USE MODD_CONVPAR, ONLY: CONVPAR_T
USE MODD_DIMPHYEX, ONLY: DIMPHYEX_T
USE MODD_NSV, ONLY: NSV_T
IMPLICIT NONE
!
TYPE(CONVPAR_SHAL) ,INTENT(IN)     :: CVP_SHAL
TYPE(CST_T)        ,INTENT(IN)     :: CST
TYPE(DIMPHYEX_T)   ,INTENT(IN)     :: D
TYPE(NSV_T)        ,INTENT(IN)     :: NSV
TYPE(CONVPAR_T)    ,INTENT(IN)     :: CONVPAR
INTEGER            ,INTENT(IN)     :: KBDIA    ! vertical  computations start at
INTEGER            ,INTENT(IN)     :: KTDIA    ! vertical computations can be limited
INTEGER            ,INTENT(IN)     :: KICE     ! flag for ice ( 1 = yes, 0 = no ice )
LOGICAL            ,INTENT(IN)     :: OSETTADJ ! logical to set convective adjustment time
REAL               ,INTENT(IN)     :: PTADJS   ! user defined adjustment time
REAL               ,INTENT(IN)     :: PPABST(D%NIT,D%NKT)   ! grid scale pressure at t
REAL               ,INTENT(IN)     :: PZZ(D%NIT,D%NKT)      ! height of model layer (m)
REAL               ,INTENT(IN)     :: PTKECLS(D%NIT)        ! TKE in the CLS  (m2/s2)
REAL               ,INTENT(IN)     :: PTT(D%NIT,D%NKT)      ! grid scale temperature at t
REAL               ,INTENT(IN)     :: PRVT(D%NIT,D%NKT)     ! grid scale water vapor
REAL               ,INTENT(IN)     :: PRCT(D%NIT,D%NKT)     ! grid scale r_c
REAL               ,INTENT(IN)     :: PRIT(D%NIT,D%NKT)     ! grid scale r_i
REAL               ,INTENT(IN)     :: PWT(D%NIT,D%NKT)      ! grid scale vertical velocity (m/s)
REAL               ,INTENT(INOUT)  :: PTTEN(D%NIT,D%NKT)    ! convective temperature tendency (K/s)
REAL               ,INTENT(INOUT)  :: PRVTEN(D%NIT,D%NKT)   ! convective r_v tendency (1/s)
REAL               ,INTENT(INOUT)  :: PRCTEN(D%NIT,D%NKT)   ! convective r_c tendency (1/s)
REAL               ,INTENT(INOUT)  :: PRITEN(D%NIT,D%NKT)   ! convective r_i tendency (1/s)
INTEGER            ,INTENT(INOUT)  :: KCLTOP(D%NIT)         ! cloud top level
INTEGER            ,INTENT(INOUT)  :: KCLBAS(D%NIT)         ! cloud base level
REAL               ,INTENT(INOUT)  :: PUMF(D%NIT,D%NKT)     ! updraft mass flux (kg/s m2)
LOGICAL            ,INTENT(IN)     :: OCH1CONV              ! include tracer transport
INTEGER            ,INTENT(IN)     :: KCH1                  ! number of species
REAL               ,INTENT(IN)     :: PCH1(D%NIT,D%NKT,KCH1)! grid scale chemical species
REAL               ,INTENT(INOUT)  :: PCH1TEN(D%NIT,D%NKT,KCH1)! species conv. tendency (1/s)
!
END SUBROUTINE SHALLOW_CONVECTION
!
END INTERFACE
!
END MODULE MODI_SHALLOW_CONVECTION
