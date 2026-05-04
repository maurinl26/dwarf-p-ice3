!@no_insert_drhook
!     ######spl
      MODULE MODE_TIWMX
!     ###############
!
!!****  *MODE_TIWMX* - OpenACC GPU-compatible version
!!
!!    PURPOSE
!!    -------
!!    Temperature-dependent lookup table functions for microphysics
!!    GPU-COMPATIBLE: All functions marked with !$acc routine seq
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
!!      OpenACC     20/12/25  Added !$acc routine seq directives for GPU compatibility
!-------------------------------------------------------------------------------
!
!*       0.   DECLARATIONS
!             ------------
!
USE MODD_TIWMX, ONLY: XNDEGR, TIWMX_t
IMPLICIT NONE

CONTAINS

  !$acc routine seq
  REAL FUNCTION ESATW(TIWMX, TT)
    !****  *ESATW* - Saturation vapor pressure over liquid water
    !!    Lookup table function called from GPU kernels in CONDENSATION
    TYPE(TIWMX_t),   INTENT(IN) :: TIWMX
    REAL,INTENT(IN) :: TT
    ESATW = TIWMX%ESTABW(NINT(XNDEGR*TT))
  END FUNCTION ESATW

  !$acc routine seq
  REAL FUNCTION DESDTW(TIWMX, TT)
    !****  *DESDTW* - Temperature derivative of saturation vapor pressure over liquid
    TYPE(TIWMX_t),   INTENT(IN) :: TIWMX
    REAL,INTENT(IN) :: TT
    DESDTW = TIWMX%DESTABW(NINT(XNDEGR*TT))
  END FUNCTION DESDTW

  !$acc routine seq
  REAL FUNCTION ESATI(TIWMX, TT)
    !****  *ESATI* - Saturation vapor pressure over ice
    !!    Lookup table function called from GPU kernels in CONDENSATION
    TYPE(TIWMX_t),   INTENT(IN) :: TIWMX
    REAL,INTENT(IN) :: TT
    ESATI = TIWMX%ESTABI(NINT(XNDEGR*TT))
  END FUNCTION ESATI

  !$acc routine seq
  REAL FUNCTION DESDTI(TIWMX, TT)
    !****  *DESDTI* - Temperature derivative of saturation vapor pressure over ice
    TYPE(TIWMX_t),   INTENT(IN) :: TIWMX
    REAL,INTENT(IN) :: TT
    DESDTI = TIWMX%DESTABI(NINT(XNDEGR*TT))
  END FUNCTION DESDTI

! Water droplet function:
  !$acc routine seq
  REAL FUNCTION AA2W(TIWMX, TT)
    !****  *AA2W* - Cloud droplet concentration parameter (liquid)
    TYPE(TIWMX_t),   INTENT(IN) :: TIWMX
    REAL,INTENT(IN) :: TT
    AA2W = TIWMX%A2WTAB(NINT(XNDEGR*TT))
  END FUNCTION AA2W

! Ice crystal function
  !$acc routine seq
  PURE REAL FUNCTION AA2(TIWMX, TT)
    !****  *AA2* - Ice crystal concentration parameter
    TYPE(TIWMX_t),   INTENT(IN) :: TIWMX
    REAL,INTENT(IN) :: TT
    AA2 = TIWMX%A2TAB(NINT(XNDEGR*TT))
  END FUNCTION AA2

! Meyers IN concentration function:
  !$acc routine seq
  PURE REAL FUNCTION AM3(TIWMX, TT)
    !****  *AM3* - Meyers ice nuclei concentration
    TYPE(TIWMX_t),   INTENT(IN) :: TIWMX
    REAL,INTENT(IN) :: TT
    AM3 = TIWMX%AM3TAB(NINT(XNDEGR*TT))
  END FUNCTION AM3

! Fletchers IN concentration function:
  !$acc routine seq
  PURE REAL FUNCTION AF3(TIWMX, TT)
    !****  *AF3* - Fletcher ice nuclei concentration
    TYPE(TIWMX_t),   INTENT(IN) :: TIWMX
    REAL,INTENT(IN) :: TT
    AF3 = TIWMX%AF3TAB(NINT(XNDEGR*TT))
  END FUNCTION AF3

! Ice crystal function
  !$acc routine seq
  PURE REAL FUNCTION BB3(TIWMX, TT)
    !****  *BB3* - Ice crystal concentration parameter
    TYPE(TIWMX_t),   INTENT(IN) :: TIWMX
    REAL,INTENT(IN) :: TT
    BB3 = TIWMX%BB3TAB(NINT(XNDEGR*TT))
  END FUNCTION BB3

! Water droplet function:
  !$acc routine seq
  REAL FUNCTION BB3W(TIWMX, TT)
    !****  *BB3W* - Cloud droplet concentration parameter (liquid)
    TYPE(TIWMX_t),   INTENT(IN) :: TIWMX
    REAL,INTENT(IN) :: TT
    BB3W = TIWMX%BB3WTAB(NINT(XNDEGR*TT))
  END FUNCTION BB3W

! Function for IN concentration reduction between 0 and -25 C:
  !$acc routine seq
  PURE REAL FUNCTION REDIN(TIWMX, TT)
    !****  *REDIN* - Ice nuclei concentration reduction factor
    TYPE(TIWMX_t),   INTENT(IN) :: TIWMX
    REAL,INTENT(IN) :: TT
    REDIN = TIWMX%REDINTAB(NINT(XNDEGR*TT))
  END FUNCTION REDIN
END MODULE MODE_TIWMX
