!SFX_LIC CeCILL-C
! CPU C-callable shim for SURFEX column physics.
!
! Exposes c_surfex_step as a plain C symbol callable from cffi / ctypes.
! All floating-point arguments are double precision (C_DOUBLE) to match the
! cffi declaration in surfex_jax.py:_CDEF.
!
! NOTE: the CPU path has no per-column tile-type argument (it was not included
! in the original cffi declaration).  A mixed ISBA/SEAFLUX/FLAKE neutral-limit
! bulk aerodynamic formula is applied to all columns using averaged parameters.
! When full SURFEX tile-aware coupling is added, the cffi ABI and this file
! must both be updated together.
!
! Build:
!   mkdir -p lib
!   gfortran -O2 -fPIC -shared surfex_c_api.F90 -o lib/libsurfex_offline.so
!   # macOS:
!   gfortran -O2 -fPIC -dynamiclib surfex_c_api.F90 -o lib/libsurfex_offline.dylib
!
! Physical basis
! --------------
! Neutral Monin-Obukhov bulk aerodynamic (Brutsaert 1982):
!   C_D = (kappa / ln(z/z0))^2
!   H/(rho*Cp) = C_H * |U| * (Ts - Ta)    [K m/s]
!   E/(rho*Lv) = C_E * |U| * (qs - qa)    [kg/kg m/s]
!   tau_u      = -C_D * |U| * u_a          [m^2/s^2]

MODULE surfex_c_api_mod
  USE ISO_C_BINDING, ONLY : C_INT, C_DOUBLE, C_PTR, C_F_POINTER
  IMPLICIT NONE

  ! Physical constants (double precision)
  REAL(C_DOUBLE), PARAMETER :: XKAPPA   = 0.4D0      ! von Karman constant
  REAL(C_DOUBLE), PARAMETER :: XG       = 9.81D0     ! gravity (m/s^2)

  ! Bulk aerodynamic parameters — land/sea averaged defaults
  ! These match the _bulk_aerodynamic_fallback() values in surfex_jax.py
  ! so that the test test_surfex_jax_cpu_bulk_fallback_consistency passes.
  REAL(C_DOUBLE), PARAMETER :: XZ_REF      = 10.0D0  ! reference height (m)
  REAL(C_DOUBLE), PARAMETER :: XZ0         = 0.05D0  ! roughness length (m)
  REAL(C_DOUBLE), PARAMETER :: XTS_DEFAULT = 295.0D0 ! default skin T (K)
  REAL(C_DOUBLE), PARAMETER :: XALB_DEF    = 0.20D0
  REAL(C_DOUBLE), PARAMETER :: XEMIS_DEF   = 0.98D0

CONTAINS

  ! ===========================================================================
  ! C-callable entry point
  ! Signature must match surfex_jax.py:_CDEF exactly (no tile_type argument).
  ! ===========================================================================
  SUBROUTINE c_surfex_step_wrap(                                   &
      n_cols, dt,                                                   &
      ptr_t_skin,                                                   &
      ptr_t_a, ptr_q_a, ptr_u_a, ptr_v_a,                          &
      ptr_p_a, ptr_rhodref,                                         &
      ptr_sw_down, ptr_lw_down,                                     &
      ptr_rain_rate, ptr_snow_rate,                                  &
      ptr_surf_flux_th, ptr_surf_flux_rv,                           &
      ptr_surf_flux_u,  ptr_surf_flux_v,                            &
      ptr_albedo, ptr_emissivity                                    &
  ) BIND(C, name="c_surfex_step")

    INTEGER(C_INT), VALUE, INTENT(IN) :: n_cols
    REAL(C_DOUBLE), VALUE, INTENT(IN) :: dt

    ! Netatmo-analysed skin temperature (double*).
    ! Sentinel: 0.0 means "use tile-type default" for that column.
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_t_skin

    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_t_a
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_q_a
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_u_a
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_v_a
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_p_a
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_rhodref
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_sw_down
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_lw_down
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_rain_rate
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_snow_rate

    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_surf_flux_th
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_surf_flux_rv
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_surf_flux_u
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_surf_flux_v
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_albedo
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_emissivity

    REAL(C_DOUBLE), POINTER, DIMENSION(:) :: f_t_skin
    REAL(C_DOUBLE), POINTER, DIMENSION(:) :: f_t_a, f_q_a, f_u_a, f_v_a
    REAL(C_DOUBLE), POINTER, DIMENSION(:) :: f_p_a, f_rhodref
    REAL(C_DOUBLE), POINTER, DIMENSION(:) :: f_sw_down, f_lw_down
    REAL(C_DOUBLE), POINTER, DIMENSION(:) :: f_rain_rate, f_snow_rate
    REAL(C_DOUBLE), POINTER, DIMENSION(:) :: f_surf_flux_th, f_surf_flux_rv
    REAL(C_DOUBLE), POINTER, DIMENSION(:) :: f_surf_flux_u, f_surf_flux_v
    REAL(C_DOUBLE), POINTER, DIMENSION(:) :: f_albedo, f_emissivity

    REAL(C_DOUBLE) :: WSPD, CD, QS_SAT, THETA_A, TS_COL
    INTEGER :: JC

    ! Associate C pointers with Fortran pointer arrays
    CALL C_F_POINTER(ptr_t_skin,       f_t_skin,       [n_cols])
    CALL C_F_POINTER(ptr_t_a,          f_t_a,          [n_cols])
    CALL C_F_POINTER(ptr_q_a,          f_q_a,          [n_cols])
    CALL C_F_POINTER(ptr_u_a,          f_u_a,          [n_cols])
    CALL C_F_POINTER(ptr_v_a,          f_v_a,          [n_cols])
    CALL C_F_POINTER(ptr_p_a,          f_p_a,          [n_cols])
    CALL C_F_POINTER(ptr_rhodref,      f_rhodref,      [n_cols])
    CALL C_F_POINTER(ptr_sw_down,      f_sw_down,      [n_cols])
    CALL C_F_POINTER(ptr_lw_down,      f_lw_down,      [n_cols])
    CALL C_F_POINTER(ptr_rain_rate,    f_rain_rate,    [n_cols])
    CALL C_F_POINTER(ptr_snow_rate,    f_snow_rate,    [n_cols])
    CALL C_F_POINTER(ptr_surf_flux_th, f_surf_flux_th, [n_cols])
    CALL C_F_POINTER(ptr_surf_flux_rv, f_surf_flux_rv, [n_cols])
    CALL C_F_POINTER(ptr_surf_flux_u,  f_surf_flux_u,  [n_cols])
    CALL C_F_POINTER(ptr_surf_flux_v,  f_surf_flux_v,  [n_cols])
    CALL C_F_POINTER(ptr_albedo,       f_albedo,       [n_cols])
    CALL C_F_POINTER(ptr_emissivity,   f_emissivity,   [n_cols])

    ! Neutral bulk aerodynamic, column-serial loop.
    ! Skin temperature is taken from the Netatmo analysis (f_t_skin > 0)
    ! or the module default XTS_DEFAULT when the sentinel 0.0 is passed.
    CD = (XKAPPA / LOG(XZ_REF / XZ0)) ** 2

    DO JC = 1, n_cols
      IF (f_t_skin(JC) > 0.0D0) THEN
        TS_COL = f_t_skin(JC)
      ELSE
        TS_COL = XTS_DEFAULT
      END IF
      ! Saturation specific humidity at TS_COL (Tetens formula)
      QS_SAT = 0.622D0 * 610.78D0 * EXP(17.27D0*(TS_COL-273.16D0) / &
                 (TS_COL-35.86D0)) / (101325.0D0 - 0.378D0 * 610.78D0 * &
                 EXP(17.27D0*(TS_COL-273.16D0)/(TS_COL-35.86D0)))
      WSPD    = MAX(SQRT(f_u_a(JC)**2 + f_v_a(JC)**2), 0.01D0)
      THETA_A = f_t_a(JC) * (1.0D5 / f_p_a(JC)) ** (287.05D0 / 1004.0D0)

      f_surf_flux_th(JC) = CD * WSPD * (TS_COL - THETA_A)
      f_surf_flux_rv(JC) = CD * WSPD * MAX(0.0D0, QS_SAT - f_q_a(JC))
      f_surf_flux_u(JC)  = -CD * WSPD * f_u_a(JC)
      f_surf_flux_v(JC)  = -CD * WSPD * f_v_a(JC)
      f_albedo(JC)       = XALB_DEF
      f_emissivity(JC)   = XEMIS_DEF
    END DO

  END SUBROUTINE c_surfex_step_wrap

END MODULE surfex_c_api_mod
