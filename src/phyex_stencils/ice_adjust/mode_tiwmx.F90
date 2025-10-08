!@no_insert_drhook
!     ######spl
MODULE MODE_TIWMX
  !     ###############
  !
  !!****  *MODD_TIWMX_FUN* -
  !!
  !!    PURPOSE
  !!    -------
  !       The purpose of this  ...
  !
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
  !-------------------------------------------------------------------------------
  !
  !*       0.   DECLARATIONS
  !             ------------
  !

  IMPLICIT NONE

  CONTAINS
  !
  REAL FUNCTION ESATW (TT, cst_XALPW, cst_XBETAW, cst_XGAMW)
    REAL, INTENT(IN) :: TT
    REAL, INTENT(IN) :: cst_XALPW
    REAL, INTENT(IN) :: cst_XBETAW
    REAL, INTENT(IN) :: cst_XGAMW
    ESATW = EXP(cst_XALPW - cst_XBETAW / TT - cst_XGAMW*ALOG(TT))
  END FUNCTION ESATW
  !
  !     pure saturation pressure over ice for tt <0 C,
  !     esatw otherwise.
  !
  REAL FUNCTION ESATI (TT, cst_XALPI, cst_XALPW, cst_XBETAI, cst_XBETAW, cst_XGAMI, cst_XGAMW, cst_XTT)
    REAL, INTENT(IN) :: TT
    REAL, INTENT(IN) :: cst_XALPI
    REAL, INTENT(IN) :: cst_XALPW
    REAL, INTENT(IN) :: cst_XBETAI
    REAL, INTENT(IN) :: cst_XBETAW
    REAL, INTENT(IN) :: cst_XGAMI
    REAL, INTENT(IN) :: cst_XGAMW
    REAL, INTENT(IN) :: cst_XTT
    ESATI = (0.5 + SIGN(0.5, TT - cst_XTT))*EXP(cst_XALPW - cst_XBETAW / TT - cst_XGAMW*ALOG(TT)) - (SIGN(0.5, TT - cst_XTT) -  &
    & 0.5)*EXP(cst_XALPI - cst_XBETAI / TT - cst_XGAMI*ALOG(TT))
  END FUNCTION ESATI
  !
  !     pure saturation pressure over water

END MODULE MODE_TIWMX
