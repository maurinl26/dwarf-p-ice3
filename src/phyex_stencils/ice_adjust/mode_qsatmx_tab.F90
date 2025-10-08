MODULE MODE_QSATMX_TAB
  IMPLICIT NONE
  CONTAINS
  FUNCTION QSATMX_TAB (CST_XEPSILO, P, T, FICE, cst_XALPI, cst_XALPW, &
          &cst_XBETAI, cst_XBETAW, cst_XGAMI, cst_XGAMW, cst_XTT)

    USE MODE_TIWMX, ONLY: ESATI, ESATW

    IMPLICIT NONE

    REAL :: QSATMX_TAB
    REAL, INTENT(IN) :: P, T, FICE
    real, intent(in) :: cst_XALPI, cst_XALPW, cst_XBETAI, cst_XBETAW, cst_XGAMI, cst_XGAMW, cst_XTT

    REAL :: ZES
    REAL, INTENT(IN) :: CST_XEPSILO

    ZES = ESATI(T, cst_XALPI, cst_XALPW, cst_XBETAI, cst_XBETAW, cst_XGAMI, cst_XGAMW, cst_XTT)*FICE&
    & + ESATW(T, cst_XALPW, cst_XBETAW, cst_XGAMW)*(1. - FICE)
    IF (ZES >= P) THEN
      ! temp > boiling point, condensation not possible.
      ! Then this function lacks physical meaning,
      ! here set to one
      QSATMX_TAB = 1.
    ELSE
      QSATMX_TAB = CST_XEPSILO*ZES / (P - ZES)        !r
    END IF

  END FUNCTION QSATMX_TAB
END MODULE MODE_QSATMX_TAB
