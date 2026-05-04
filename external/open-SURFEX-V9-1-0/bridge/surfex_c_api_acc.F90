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
  USE isba_fluxes_acc_mod, ONLY : ISBA_FLUXES_ACC
  USE hydro_soil_acc_mod, ONLY : HYDRO_SOIL_ACC
  USE ice_soilfr_acc_mod, ONLY : ICE_SOILFR_ACC
  USE hydro_snow_acc_mod, ONLY : HYDRO_SNOW_ACC
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
  ! Mock ISBA-3L + D95 pipeline (Option C: hardcoded constants)
  ! ---------------------------------------------------------------------------
  PURE SUBROUTINE mock_isba_col_flux(DT, TS, TA, QA, UA, VA, P_A, RHODREF, &
                                     SW_DOWN, LW_DOWN, RAIN_RATE, SNOW_RATE, &
                                     WG1, WG2, WG3, WGI1, WGI2, TG1, TG2, &
                                     WSNOW1, RHO1, ALB_SN, &
                                     FLUX_TH, FLUX_RV, FLUX_U, FLUX_V, ALBEDO_OUT)
    !$acc routine seq
    REAL(C_FLOAT), INTENT(IN)  :: DT, TS, TA, QA, UA, VA, P_A, RHODREF
    REAL(C_FLOAT), INTENT(IN)  :: SW_DOWN, LW_DOWN, RAIN_RATE, SNOW_RATE
    REAL(C_FLOAT), INTENT(INOUT) :: WG1, WG2, WG3, WGI1, WGI2, TG1, TG2
    REAL(C_FLOAT), INTENT(INOUT) :: WSNOW1, RHO1, ALB_SN
    REAL(C_FLOAT), INTENT(OUT) :: FLUX_TH, FLUX_RV, FLUX_U, FLUX_V, ALBEDO_OUT

    ! Hardcoded physiography constants for a generic loamy soil with grass
    REAL(C_FLOAT) :: PWSAT=0.43, PWFC=0.32, PWWILT=0.17
    REAL(C_FLOAT) :: XC1=1.0, XC2=0.5, XWGEQ=0.2, XCT=1.0E-5, XCG=2.0E-6
    REAL(C_FLOAT) :: XWDRAIN=0.001, XC4B=5.0, XDG1=0.01, XDG2=1.0, PD_G3=2.0
    REAL(C_FLOAT) :: XC3_1=0.1, XC3_2=0.05, XC4REF=10.0
    REAL(C_FLOAT) :: XCPS=1.0E6, XRESA=50.0, XVEG=0.9, XPSNG=0.0, XPSNV=0.0
    REAL(C_FLOAT) :: XPSN=0.0, XHV=1.0, XRS=100.0, XFFROZEN=0.0, XFF=0.0
    REAL(C_FLOAT) :: XSRSFC=0.0, PALBT=0.2, PEMIST=0.97
    REAL(C_FLOAT) :: PEXNA, PEXNS, PHUG, PHUI, PLEG_DELTA, PLEGI_DELTA, PDELTA, PF5
    REAL(C_FLOAT) :: PCS, PTSM, PFROZEN1, PQSAT, PDQSAT, PSNOW_THRUFAL
    
    ! Flux outputs from ISBA_FLUXES_ACC
    REAL(C_FLOAT) :: PRN, PH, PLE, PLEG, PLEGI, PLEV, PLES, PLER, PLETR, PEVAP, PEPOT, PGFLUX
    REAL(C_FLOAT) :: PMELTADV, PMELT, PLE_FLOOD, PLEI_FLOOD, XTG1_OUT, WSNOW1_OUT
    REAL(C_FLOAT) :: PRUNOFF, PDRAIN, PPG, PEVAPCOR, PDWGI1, PDWGI2, PLEGI_FR

    ! 1. Mock inputs for ICE_SOILFR and ISBA_FLUXES
    PEXNA = (P_A / 100000.0) ** (287.05/1004.0)
    PEXNS = 1.0
    PHUG  = 1.0; PHUI = 1.0; PLEG_DELTA = 1.0; PLEGI_DELTA = 0.0; PDELTA = 0.0; PF5 = 1.0
    PCS = 2.0E6; PTSM = TG1; PFROZEN1 = 0.0; PSNOW_THRUFAL = 0.0
    PQSAT = QA; PDQSAT = 0.0 ! Simplification
    PPG = RAIN_RATE; PEVAPCOR = 0.0

    ! 2. Soil Freezing
    CALL ICE_SOILFR_ACC(DT, 1, TG1, TG2, WG1, WG2, WGI1, WGI2, PWSAT, &
                        XCG, XCT, XDG1, XDG2, PDWGI1, PDWGI2, PLEGI_FR)
                        
    ! 3. Soil Moisture
    CALL HYDRO_SOIL_ACC(DT, 0.0, 0.0, PPG, PEVAPCOR, PD_G3, PWSAT, PWFC, &
                        PDWGI1, PDWGI2, PLEGI_FR, WG3, PRUNOFF, PDRAIN, PWWILT, &
                        WG1, WG2, WGI1, WGI2, TG1, TG2, &
                        XC1, XC2, XWGEQ, XCT, XCG, XWDRAIN, XC4B, &
                        XDG1, XDG2, XC3_1, XC3_2, XC4REF, 2, 1)

    ! 4. Snow Scheme
    CALL HYDRO_SNOW_ACC(DT, 0, 0.0, SNOW_RATE, 0.0, PMELT, PPG, WSNOW1, ALB_SN, RHO1)

    ! 5. Energy Balance & Fluxes
    CALL ISBA_FLUXES_ACC(DT, SW_DOWN, LW_DOWN, TA, QA, RHODREF, &
                         PEXNS, PEXNA, PHUG, PHUI, PLEG_DELTA, PLEGI_DELTA, PDELTA, PF5, &
                         PCS, PTSM, PFROZEN1, PALBT, PEMIST, PQSAT, PDQSAT, PSNOW_THRUFAL, &
                         TG1, XCPS, XRESA, XVEG, XPSNG, 2.5E6, 2.8E6, XPSN, XPSNV, &
                         XHV, XRS, XFFROZEN, XFF, XCT, WSNOW1, XSRSFC, 1, 3, &
                         PRN, PH, PLE, PLEG, PLEGI, PLEV, PLES, PLER, PLETR, PEVAP, PEPOT, &
                         PGFLUX, PMELTADV, PMELT, PLE_FLOOD, PLEI_FLOOD, XTG1_OUT, WSNOW1_OUT)

    ! Convert H and LE into kinematic fluxes (approx)
    FLUX_TH = PH / (RHODREF * 1004.0)
    FLUX_RV = PLE / (RHODREF * 2.5E6)
    
    ! Drag is still mocked via bulk formulation
    CALL isba_col_flux(TS, TA, QA, UA, VA, RHODREF, FLUX_TH, FLUX_RV, FLUX_U, FLUX_V)

    ALBEDO_OUT = PALBT
  END SUBROUTINE mock_isba_col_flux

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
      ptr_albedo, ptr_emissivity,                                   &
      ptr_wg1, ptr_wg2, ptr_wg3, ptr_wgi1, ptr_wgi2,                &
      ptr_tg1, ptr_tg2, ptr_wsnow1, ptr_rho1, ptr_alb               &
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
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_wg1
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_wg2
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_wg3
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_wgi1
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_wgi2
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_tg1
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_tg2
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_wsnow1
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_rho1
    TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_alb

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
    REAL(C_FLOAT),  POINTER, DIMENSION(:) :: f_wg1, f_wg2, f_wg3
    REAL(C_FLOAT),  POINTER, DIMENSION(:) :: f_wgi1, f_wgi2
    REAL(C_FLOAT),  POINTER, DIMENSION(:) :: f_tg1, f_tg2
    REAL(C_FLOAT),  POINTER, DIMENSION(:) :: f_wsnow1, f_rho1, f_alb

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
    CALL C_F_POINTER(ptr_wg1,          f_wg1,          [n_cols])
    CALL C_F_POINTER(ptr_wg2,          f_wg2,          [n_cols])
    CALL C_F_POINTER(ptr_wg3,          f_wg3,          [n_cols])
    CALL C_F_POINTER(ptr_wgi1,         f_wgi1,         [n_cols])
    CALL C_F_POINTER(ptr_wgi2,         f_wgi2,         [n_cols])
    CALL C_F_POINTER(ptr_tg1,          f_tg1,          [n_cols])
    CALL C_F_POINTER(ptr_tg2,          f_tg2,          [n_cols])
    CALL C_F_POINTER(ptr_wsnow1,       f_wsnow1,       [n_cols])
    CALL C_F_POINTER(ptr_rho1,         f_rho1,         [n_cols])
    CALL C_F_POINTER(ptr_alb,          f_alb,          [n_cols])

    ! --- GPU-parallel column loop (OpenACC) ---
    ! async(1) required: acc_set_cuda_stream(1, stream_ptr) binds queue 1
    ! to the CUDA stream used by cudaStreamBeginCapture (CUDA graph capture).
    !$acc data deviceptr(f_tile, f_t_skin, f_t_a, f_q_a, f_u_a, f_v_a,  &
    !$acc&               f_p_a, f_rhodref, f_sw_down, f_lw_down,          &
    !$acc&               f_rain_rate, f_snow_rate,                          &
    !$acc&               f_surf_flux_th, f_surf_flux_rv,                    &
    !$acc&               f_surf_flux_u, f_surf_flux_v,                      &
    !$acc&               f_albedo, f_emissivity,                            &
    !$acc&               f_wg1, f_wg2, f_wg3, f_wgi1, f_wgi2,               &
    !$acc&               f_tg1, f_tg2, f_wsnow1, f_rho1, f_alb)

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
          CALL mock_isba_col_flux(               &
              REAL(dt, C_FLOAT), TS_COL,         &
              f_t_a(JC), f_q_a(JC),             &
              f_u_a(JC), f_v_a(JC),             &
              f_p_a(JC), f_rhodref(JC),         &
              f_sw_down(JC), f_lw_down(JC),     &
              f_rain_rate(JC), f_snow_rate(JC), &
              f_wg1(JC), f_wg2(JC), f_wg3(JC),  &
              f_wgi1(JC), f_wgi2(JC),           &
              f_tg1(JC), f_tg2(JC),             &
              f_wsnow1(JC), f_rho1(JC), f_alb(JC), &
              f_surf_flux_th(JC), f_surf_flux_rv(JC), &
              f_surf_flux_u(JC),  f_surf_flux_v(JC), &
              f_albedo(JC))
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
