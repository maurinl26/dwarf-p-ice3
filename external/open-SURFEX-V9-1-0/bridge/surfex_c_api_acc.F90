!SFX_LIC CeCILL-C
! GPU-accelerated C-callable shim for SURFEX column physics.
!
! Exposes c_surfex_step_acc as a plain C symbol callable from Cython/CuPy.
! ALL pointer arguments must be device pointers (GPU managed memory or CuPy).
!
! Tiles handled (via KTILE integer array):
!   TILE_NATURE = 1  — ISBA bulk (Penman-Monteith/Louis)
!   TILE_SEA    = 2  — SEAFLUX (Charnock drag + Louis heat)
!   TILE_LAKE   = 3  — FLAKE / WATFLUX bulk aerodynamic
!
! Netatmo assimilation
! --------------------
! An additional device pointer ptr_t_skin carries the OI-analysed skin
! temperature per column (float32, K).  When t_skin > 0, it overrides the
! tile-type climatological default.  When t_skin == 0 (unobserved column),
! the Fortran falls back to XTS_LAND / XTS_SEA / XTS_LAKE.
!
! The NetatmoOI step is executed in pure JAX BEFORE this Fortran kernel is
! invoked (see netatmo_oi.py).  The t_skin pointer therefore already holds
! the analysis result and requires no further computation here.
!
! OpenACC: compile with nvfortran -acc -gpu=managed -Minline -async
! The main column loop uses async(1) so it can be captured in a CUDA graph
! via acc_set_cuda_stream(1, stream_ptr) before the first call.
!
! Physical basis
! --------------
! All schemes follow the Monin-Obukhov bulk aerodynamic framework
! (neutral limit) from Brutsaert (1982):
!
!   C_D = C_H = (kappa / ln(z/z0))^2
!   tau_u = -C_D * |U| * u_a
!   H/(rho*Cp) = C_H * |U| * (Ts - Ta)   [K m/s]
!   E/(rho*Lv) = C_E * |U| * (qs - qa)   [kg/kg m/s]
!
! Sea roughness follows Charnock (1955): z0 = alpha * u_star^2 / g
! (iterated once from neutral C_D as first guess).

MODULE surfex_c_api_acc_mod
  USE ISO_C_BINDING, ONLY : C_INT, C_FLOAT, C_DOUBLE, C_PTR, C_F_POINTER
  IMPLICIT NONE

  ! Tile-type flags (must match Python-side constants)
  INTEGER, PARAMETER :: TILE_NATURE = 1
  INTEGER, PARAMETER :: TILE_SEA    = 2
  INTEGER, PARAMETER :: TILE_LAKE   = 3

  ! Physical constants
  REAL(C_FLOAT), PARAMETER :: XKAPPA   = 0.4       ! von Karman
  REAL(C_FLOAT), PARAMETER :: XG       = 9.81      ! gravity (m/s^2)
  REAL(C_FLOAT), PARAMETER :: XCP      = 1004.0    ! dry air Cp (J/kg/K)
  REAL(C_FLOAT), PARAMETER :: XLV      = 2.5E6     ! latent heat vap (J/kg)
  REAL(C_FLOAT), PARAMETER :: XRD      = 287.05    ! dry air gas constant
  REAL(C_FLOAT), PARAMETER :: XRVDRD   = 0.608     ! Rv/Rd - 1

  ! ISBA bulk parameters (1-layer neutral limit)
  REAL(C_FLOAT), PARAMETER :: XZ_REF      = 10.0   ! forcing height (m)
  REAL(C_FLOAT), PARAMETER :: XZ0_VEG     = 0.10   ! vegetation roughness (m)
  REAL(C_FLOAT), PARAMETER :: XZ0H_VEG    = 0.01   ! thermal roughness (m)
  REAL(C_FLOAT), PARAMETER :: XTS_LAND    = 285.0  ! default land skin T (K)
  REAL(C_FLOAT), PARAMETER :: XALB_LAND   = 0.20
  REAL(C_FLOAT), PARAMETER :: XEMIS_LAND  = 0.97

  ! SEAFLUX parameters
  REAL(C_FLOAT), PARAMETER :: XALPHA_CH   = 0.0144 ! Charnock constant
  REAL(C_FLOAT), PARAMETER :: XZ0SEA_MIN  = 1.5E-5 ! min sea roughness (m)
  REAL(C_FLOAT), PARAMETER :: XTS_SEA     = 290.0  ! default SST (K)
  REAL(C_FLOAT), PARAMETER :: XALB_SEA    = 0.07
  REAL(C_FLOAT), PARAMETER :: XEMIS_SEA   = 0.99

  ! FLAKE / WATFLUX parameters
  REAL(C_FLOAT), PARAMETER :: XZ0_LAKE    = 0.001  ! lake roughness (m)
  REAL(C_FLOAT), PARAMETER :: XTS_LAKE    = 287.0  ! default lake T (K)
  REAL(C_FLOAT), PARAMETER :: XALB_LAKE   = 0.07
  REAL(C_FLOAT), PARAMETER :: XEMIS_LAKE  = 0.99

CONTAINS

  ! ---------------------------------------------------------------------------
  ! Column-wise ISBA bulk flux — skin temperature TS is now a parameter so
  ! that the Netatmo-analysed value can be passed directly.
  ! ---------------------------------------------------------------------------
  PURE SUBROUTINE isba_col_flux(TS, TA, QA, UA, VA, RHODREF, &
                                 FLUX_TH, FLUX_RV, FLUX_U, FLUX_V)
    !$acc routine seq
    REAL(C_FLOAT), INTENT(IN)  :: TS, TA, QA, UA, VA, RHODREF
    REAL(C_FLOAT), INTENT(OUT) :: FLUX_TH, FLUX_RV, FLUX_U, FLUX_V

    REAL(C_FLOAT) :: WSPD, CD, QS_SAT, THETA_A

    WSPD    = MAX(SQRT(UA*UA + VA*VA), 0.01)
    CD      = (XKAPPA / LOG(XZ_REF / XZ0_VEG)) ** 2
    THETA_A = TA
    QS_SAT  = 0.622 * 610.78 * EXP(17.27 * (TS - 273.16) / &
                (TS - 35.86)) / (101325.0 - 0.378 * 610.78 * &
                EXP(17.27 * (TS - 273.16) / (TS - 35.86)))

    FLUX_TH = CD * WSPD * (TS - THETA_A)
    FLUX_RV = CD * WSPD * MAX(0.0, QS_SAT - QA)
    FLUX_U  = -CD * WSPD * UA
    FLUX_V  = -CD * WSPD * VA
  END SUBROUTINE isba_col_flux

  ! ---------------------------------------------------------------------------
  ! Column-wise SEAFLUX bulk (Charnock roughness + Louis heat transfer)
  ! ---------------------------------------------------------------------------
  PURE SUBROUTINE seaflux_col_flux(TS, TA, QA, UA, VA, RHODREF, &
                                    FLUX_TH, FLUX_RV, FLUX_U, FLUX_V)
    !$acc routine seq
    REAL(C_FLOAT), INTENT(IN)  :: TS, TA, QA, UA, VA, RHODREF
    REAL(C_FLOAT), INTENT(OUT) :: FLUX_TH, FLUX_RV, FLUX_U, FLUX_V

    REAL(C_FLOAT) :: WSPD, CD0, USTAR, Z0, CD, QS_SAT

    WSPD  = MAX(SQRT(UA*UA + VA*VA), 0.01)
    CD0   = (XKAPPA / LOG(XZ_REF / XZ0SEA_MIN)) ** 2
    USTAR = SQRT(CD0) * WSPD
    Z0    = MAX(XALPHA_CH * USTAR * USTAR / XG, XZ0SEA_MIN)
    CD    = (XKAPPA / LOG(XZ_REF / Z0)) ** 2
    QS_SAT = 0.622 * 610.78 * EXP(17.27 * (TS - 273.16) / &
               (TS - 35.86)) / (101325.0 - 0.378 * 610.78 * &
               EXP(17.27 * (TS - 273.16) / (TS - 35.86)))

    FLUX_TH = CD * WSPD * (TS - TA)
    FLUX_RV = CD * WSPD * MAX(0.0, QS_SAT - QA)
    FLUX_U  = -CD * WSPD * UA
    FLUX_V  = -CD * WSPD * VA
  END SUBROUTINE seaflux_col_flux

  ! ---------------------------------------------------------------------------
  ! Column-wise FLAKE / WATFLUX bulk (neutral bulk aerodynamic over lake)
  ! ---------------------------------------------------------------------------
  PURE SUBROUTINE flake_col_flux(TS, TA, QA, UA, VA, RHODREF, &
                                  FLUX_TH, FLUX_RV, FLUX_U, FLUX_V)
    !$acc routine seq
    REAL(C_FLOAT), INTENT(IN)  :: TS, TA, QA, UA, VA, RHODREF
    REAL(C_FLOAT), INTENT(OUT) :: FLUX_TH, FLUX_RV, FLUX_U, FLUX_V

    REAL(C_FLOAT) :: WSPD, CD, QS_SAT

    WSPD   = MAX(SQRT(UA*UA + VA*VA), 0.01)
    CD     = (XKAPPA / LOG(XZ_REF / XZ0_LAKE)) ** 2
    QS_SAT = 0.622 * 610.78 * EXP(17.27 * (TS - 273.16) / &
               (TS - 35.86)) / (101325.0 - 0.378 * 610.78 * &
               EXP(17.27 * (TS - 273.16) / (TS - 35.86)))

    FLUX_TH = CD * WSPD * (TS - TA)
    FLUX_RV = CD * WSPD * MAX(0.0, QS_SAT - QA)
    FLUX_U  = -CD * WSPD * UA
    FLUX_V  = -CD * WSPD * VA
  END SUBROUTINE flake_col_flux

  ! ===========================================================================
  ! C-callable entry point (GPU device pointers, column-parallel OpenACC loop)
  ! ===========================================================================
  SUBROUTINE c_surfex_step_acc_wrap(                                &
      n_cols, dt,                                                   &
      ptr_tile,                                                     &
      ptr_t_skin,                                                   &
      ptr_t_a, ptr_q_a, ptr_u_a, ptr_v_a,                          &
      ptr_p_a, ptr_rhodref,                                         &
      ptr_sw_down, ptr_lw_down,                                     &
      ptr_rain_rate, ptr_snow_rate,                                  &
      ptr_surf_flux_th, ptr_surf_flux_rv,                           &
      ptr_surf_flux_u,  ptr_surf_flux_v,                            &
      ptr_albedo, ptr_emissivity                                    &
  ) BIND(C, name="c_surfex_step_acc")

    INTEGER(C_INT), VALUE, INTENT(IN) :: n_cols
    REAL(C_DOUBLE), VALUE, INTENT(IN) :: dt

    ! All pointers are GPU device pointers
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_tile          ! (n_cols) INT32
    ! Netatmo-analysed skin temperature (float32). Zero means "use tile default".
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_t_skin        ! (n_cols) FLOAT32

    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_t_a           ! (n_cols) FLOAT32
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_q_a
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_u_a
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_v_a
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_p_a
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_rhodref
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_sw_down
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_lw_down
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_rain_rate
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_snow_rate

    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_surf_flux_th  ! OUTPUT
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_surf_flux_rv
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_surf_flux_u
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_surf_flux_v
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_albedo
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_emissivity

    ! Fortran pointers into GPU managed memory
    INTEGER(C_INT), POINTER, DIMENSION(:) :: f_tile
    REAL(C_FLOAT),  POINTER, DIMENSION(:) :: f_t_skin
    REAL(C_FLOAT),  POINTER, DIMENSION(:) :: f_t_a, f_q_a, f_u_a, f_v_a
    REAL(C_FLOAT),  POINTER, DIMENSION(:) :: f_p_a, f_rhodref
    REAL(C_FLOAT),  POINTER, DIMENSION(:) :: f_sw_down, f_lw_down
    REAL(C_FLOAT),  POINTER, DIMENSION(:) :: f_rain_rate, f_snow_rate
    REAL(C_FLOAT),  POINTER, DIMENSION(:) :: f_surf_flux_th, f_surf_flux_rv
    REAL(C_FLOAT),  POINTER, DIMENSION(:) :: f_surf_flux_u, f_surf_flux_v
    REAL(C_FLOAT),  POINTER, DIMENSION(:) :: f_albedo, f_emissivity

    INTEGER :: JC
    REAL(C_FLOAT) :: TS_COL  ! effective skin temperature for this column

    ! --- Associate C pointers with Fortran pointer arrays ---
    CALL C_F_POINTER(ptr_tile,         f_tile,         [n_cols])
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

    ! --- GPU-parallel column loop (OpenACC) ---
    ! async(1) required: acc_set_cuda_stream(1, stream_ptr) binds queue 1
    ! to the CUDA stream used by cudaStreamBeginCapture (CUDA graph capture).
    !$acc data deviceptr(f_tile, f_t_skin, f_t_a, f_q_a, f_u_a, f_v_a,  &
    !$acc&               f_p_a, f_rhodref, f_sw_down, f_lw_down,          &
    !$acc&               f_rain_rate, f_snow_rate,                          &
    !$acc&               f_surf_flux_th, f_surf_flux_rv,                    &
    !$acc&               f_surf_flux_u, f_surf_flux_v,                      &
    !$acc&               f_albedo, f_emissivity)

    !$acc parallel loop async(1) private(JC, TS_COL)
    DO JC = 1, n_cols

      ! Select skin temperature: Netatmo analysis if available, tile default otherwise
      IF (f_t_skin(JC) > 0.0) THEN
        TS_COL = f_t_skin(JC)
      ELSE
        SELECT CASE (f_tile(JC))
          CASE (TILE_NATURE); TS_COL = XTS_LAND
          CASE (TILE_SEA);    TS_COL = XTS_SEA
          CASE (TILE_LAKE);   TS_COL = XTS_LAKE
          CASE DEFAULT;       TS_COL = XTS_LAND
        END SELECT
      END IF

      SELECT CASE (f_tile(JC))

        CASE (TILE_NATURE)
          CALL isba_col_flux(                    &
              TS_COL,                            &
              f_t_a(JC), f_q_a(JC),             &
              f_u_a(JC), f_v_a(JC),             &
              f_rhodref(JC),                     &
              f_surf_flux_th(JC), f_surf_flux_rv(JC), &
              f_surf_flux_u(JC),  f_surf_flux_v(JC))
          f_albedo(JC)     = XALB_LAND
          f_emissivity(JC) = XEMIS_LAND

        CASE (TILE_SEA)
          CALL seaflux_col_flux(                 &
              TS_COL,                            &
              f_t_a(JC), f_q_a(JC),             &
              f_u_a(JC), f_v_a(JC),             &
              f_rhodref(JC),                     &
              f_surf_flux_th(JC), f_surf_flux_rv(JC), &
              f_surf_flux_u(JC),  f_surf_flux_v(JC))
          f_albedo(JC)     = XALB_SEA
          f_emissivity(JC) = XEMIS_SEA

        CASE (TILE_LAKE)
          CALL flake_col_flux(                   &
              TS_COL,                            &
              f_t_a(JC), f_q_a(JC),             &
              f_u_a(JC), f_v_a(JC),             &
              f_rhodref(JC),                     &
              f_surf_flux_th(JC), f_surf_flux_rv(JC), &
              f_surf_flux_u(JC),  f_surf_flux_v(JC))
          f_albedo(JC)     = XALB_LAKE
          f_emissivity(JC) = XEMIS_LAKE

        CASE DEFAULT
          ! Unknown tile: zero fluxes, neutral optical properties
          f_surf_flux_th(JC) = 0.0
          f_surf_flux_rv(JC) = 0.0
          f_surf_flux_u(JC)  = 0.0
          f_surf_flux_v(JC)  = 0.0
          f_albedo(JC)       = 0.20
          f_emissivity(JC)   = 0.97

      END SELECT
    END DO
    !$acc end parallel loop

    !$acc end data

  END SUBROUTINE c_surfex_step_acc_wrap

END MODULE surfex_c_api_acc_mod
