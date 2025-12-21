MODULE phyex_bridge_acc
    USE ISO_C_BINDING
    ! Import the OpenACC GPU-accelerated routines
    USE MODI_ICE_ADJUST
    USE PARKIND1, ONLY : JPIM, JPRB
    USE MODD_DIMPHYEX, ONLY : DIMPHYEX_t
    USE MODD_CST, ONLY : CST_t, CST
    USE MODD_RAIN_ICE_PARAM_n
    USE MODD_RAIN_ICE_DESCR_n
    USE MODD_PARAM_ICE_n
    USE MODD_NEB_n, ONLY : NEB_t
    USE MODD_TURB_n, ONLY : TURB_t
    USE MODD_BUDGET, ONLY : TBUDGETCONF_t, TBUDGETDATA_PTR
    USE MODE_INI_CST, ONLY : INI_CST

    IMPLICIT NONE

CONTAINS

    !===========================================================================
    ! C-callable GPU wrapper for ICE_ADJUST_ACC
    !
    ! This subroutine expects ALL input arrays to be GPU device pointers.
    ! Use CuPy or other GPU array library to allocate and manage GPU memory.
    !
    ! Example Python/CuPy usage:
    !   import cupy as cp
    !   th_gpu = cp.asarray(th_cpu, dtype=np.float32)
    !   # Pass th_gpu.data.ptr to this function
    !===========================================================================
    SUBROUTINE c_ice_adjust_acc_wrap(                                          &
        nlon, nlev, krr, timestep,                                             &
        ptr_sigqsat, ptr_pabs, ptr_sigs, ptr_th, ptr_exn, ptr_exn_ref,        &
        ptr_rho_dry_ref, ptr_rv, ptr_rc, ptr_ri, ptr_rr, ptr_rs, ptr_rg,      &
        ptr_cf_mf, ptr_rc_mf, ptr_ri_mf,                                       &
        ptr_rvs, ptr_rcs, ptr_ris, ptr_ths,                                    &
        ptr_cldfr, ptr_icldfr, ptr_wcldfr                                      &
    ) BIND(C, name="c_ice_adjust_acc")

        !-----------------------------------------------------------------------
        ! Arguments (C-compatible)
        !-----------------------------------------------------------------------
        INTEGER(C_INT), VALUE, INTENT(IN) :: nlon, nlev, krr
        REAL(C_FLOAT), VALUE, INTENT(IN) :: timestep

        ! GPU device pointers for input arrays
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_sigqsat      ! 1D: (nlon) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_pabs         ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_sigs         ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_th           ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_exn          ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_exn_ref      ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_rho_dry_ref  ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_rv           ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_rc           ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_ri           ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_rr           ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_rs           ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_rg           ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_cf_mf        ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_rc_mf        ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_ri_mf        ! 2D: (nlon, nlev) - GPU

        ! GPU device pointers for input/output tendency arrays
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_rvs          ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_rcs          ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_ris          ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_ths          ! 2D: (nlon, nlev) - GPU

        ! GPU device pointers for output arrays
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_cldfr        ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_icldfr       ! 2D: (nlon, nlev) - GPU
        TYPE(C_PTR), VALUE, INTENT(IN) :: ptr_wcldfr       ! 2D: (nlon, nlev) - GPU

        !-----------------------------------------------------------------------
        ! Fortran pointers to GPU data (using C_FLOAT for single precision)
        !-----------------------------------------------------------------------
        REAL(KIND=C_FLOAT), POINTER, DIMENSION(:) :: f_sigqsat
        REAL(KIND=C_FLOAT), POINTER, DIMENSION(:,:) :: f_pabs, f_sigs, f_th, f_exn, f_exn_ref
        REAL(KIND=C_FLOAT), POINTER, DIMENSION(:,:) :: f_rho_dry_ref, f_rv, f_rc, f_ri
        REAL(KIND=C_FLOAT), POINTER, DIMENSION(:,:) :: f_rr, f_rs, f_rg
        REAL(KIND=C_FLOAT), POINTER, DIMENSION(:,:) :: f_cf_mf, f_rc_mf, f_ri_mf
        REAL(KIND=C_FLOAT), POINTER, DIMENSION(:,:) :: f_rvs, f_rcs, f_ris, f_ths
        REAL(KIND=C_FLOAT), POINTER, DIMENSION(:,:) :: f_cldfr, f_icldfr, f_wcldfr

        !-----------------------------------------------------------------------
        ! Local variables for PHYEX structures
        !-----------------------------------------------------------------------
        TYPE(DIMPHYEX_t) :: D
        TYPE(RAIN_ICE_PARAM_t) :: ICEP
        TYPE(NEB_t) :: NEBN
        TYPE(TURB_t) :: TURBN
        TYPE(PARAM_ICE_t) :: PARAMI
        TYPE(TBUDGETCONF_t) :: BUCONF
        TYPE(TBUDGETDATA_PTR), DIMENSION(0) :: TBUDGETS

        !-----------------------------------------------------------------------
        ! Additional required arrays (allocated on GPU)
        !-----------------------------------------------------------------------
        REAL(KIND=C_FLOAT), ALLOCATABLE, DIMENSION(:,:) :: PRHODJ, PZZ
        REAL(KIND=C_FLOAT), ALLOCATABLE, DIMENSION(:,:) :: PMFCONV
        REAL(KIND=C_FLOAT), ALLOCATABLE, DIMENSION(:,:) :: PWEIGHT_MF_CLOUD
        REAL(KIND=C_FLOAT), ALLOCATABLE, DIMENSION(:,:) :: PSSIO, PSSIU, PIFR
        REAL(KIND=C_FLOAT), ALLOCATABLE, DIMENSION(:,:) :: PSRCS
        LOGICAL :: LMFCONV, OCOMPUTE_SRC

        !-----------------------------------------------------------------------
        ! Convert C pointers to Fortran pointers
        !-----------------------------------------------------------------------
        CALL C_F_POINTER(ptr_sigqsat, f_sigqsat, [nlon])
        CALL C_F_POINTER(ptr_pabs, f_pabs, [nlon, nlev])
        CALL C_F_POINTER(ptr_sigs, f_sigs, [nlon, nlev])
        CALL C_F_POINTER(ptr_th, f_th, [nlon, nlev])
        CALL C_F_POINTER(ptr_exn, f_exn, [nlon, nlev])
        CALL C_F_POINTER(ptr_exn_ref, f_exn_ref, [nlon, nlev])
        CALL C_F_POINTER(ptr_rho_dry_ref, f_rho_dry_ref, [nlon, nlev])
        CALL C_F_POINTER(ptr_rv, f_rv, [nlon, nlev])
        CALL C_F_POINTER(ptr_rc, f_rc, [nlon, nlev])
        CALL C_F_POINTER(ptr_ri, f_ri, [nlon, nlev])
        CALL C_F_POINTER(ptr_rr, f_rr, [nlon, nlev])
        CALL C_F_POINTER(ptr_rs, f_rs, [nlon, nlev])
        CALL C_F_POINTER(ptr_rg, f_rg, [nlon, nlev])
        CALL C_F_POINTER(ptr_cf_mf, f_cf_mf, [nlon, nlev])
        CALL C_F_POINTER(ptr_rc_mf, f_rc_mf, [nlon, nlev])
        CALL C_F_POINTER(ptr_ri_mf, f_ri_mf, [nlon, nlev])
        CALL C_F_POINTER(ptr_rvs, f_rvs, [nlon, nlev])
        CALL C_F_POINTER(ptr_rcs, f_rcs, [nlon, nlev])
        CALL C_F_POINTER(ptr_ris, f_ris, [nlon, nlev])
        CALL C_F_POINTER(ptr_ths, f_ths, [nlon, nlev])
        CALL C_F_POINTER(ptr_cldfr, f_cldfr, [nlon, nlev])
        CALL C_F_POINTER(ptr_icldfr, f_icldfr, [nlon, nlev])
        CALL C_F_POINTER(ptr_wcldfr, f_wcldfr, [nlon, nlev])

        !-----------------------------------------------------------------------
        ! Initialize DIMPHYEX structure
        !-----------------------------------------------------------------------
        D%NIT = nlon
        D%NIB = 1
        D%NIE = nlon
        D%NJT = 1
        D%NJB = 1
        D%NJE = 1
        D%NKT = nlev
        D%NKL = 1        ! Ground to space ordering
        D%NKA = 1
        D%NKU = nlev
        D%NKB = 1
        D%NKE = nlev
        D%NKTB = 1
        D%NKTE = nlev
        D%NIBC = 1
        D%NJBC = 1
        D%NIEC = nlon
        D%NJEC = 1
        D%NIJT = nlon
        D%NIJB = 1
        D%NIJE = nlon
        D%NKLES = nlev
        D%NLESMASK = 0
        D%NLES_TIMES = 0

        !-----------------------------------------------------------------------
        ! Initialize physical constants (uses global CST module)
        !-----------------------------------------------------------------------
        CALL INI_CST()

        !-----------------------------------------------------------------------
        ! Initialize NEBN (nebulosity/cloud parameters) - AROME defaults
        !-----------------------------------------------------------------------
        NEBN%LSUBG_COND = .FALSE.     ! No subgrid condensation
        NEBN%LSIGMAS = .TRUE.         ! Use sigma_s
        NEBN%CFRAC_ICE_ADJUST = 'S'   ! Standard
        NEBN%CCONDENS = 'CB02'        ! Condensation scheme
        NEBN%CLAMBDA3 = 'CB'          ! Lambda3 formulation

        !-----------------------------------------------------------------------
        ! Initialize PARAMI (ice parameters)
        !-----------------------------------------------------------------------
        PARAMI%CSUBG_MF_PDF = 'NONE'  ! No mass flux PDF
        PARAMI%LOCND2 = .FALSE.       ! OCND2 option

        !-----------------------------------------------------------------------
        ! Initialize budget configuration (disable budgets)
        !-----------------------------------------------------------------------
        BUCONF%LBUDGET_TH = .FALSE.
        BUCONF%LBUDGET_RV = .FALSE.
        BUCONF%LBUDGET_RC = .FALSE.
        BUCONF%LBUDGET_RI = .FALSE.

        !-----------------------------------------------------------------------
        ! Allocate additional required arrays ON GPU
        !-----------------------------------------------------------------------
        ALLOCATE(PRHODJ(nlon, nlev))
        ALLOCATE(PZZ(nlon, nlev))
        ALLOCATE(PMFCONV(nlon, nlev))
        ALLOCATE(PWEIGHT_MF_CLOUD(nlon, nlev))
        ALLOCATE(PSSIO(nlon, nlev))
        ALLOCATE(PSSIU(nlon, nlev))
        ALLOCATE(PIFR(nlon, nlev))
        ALLOCATE(PSRCS(nlon, nlev))

        !-----------------------------------------------------------------------
        ! Initialize arrays (on GPU)
        !-----------------------------------------------------------------------
        !$acc data create(PRHODJ, PZZ, PMFCONV, PWEIGHT_MF_CLOUD, PSSIO, PSSIU, PIFR, PSRCS) &
        !$acc&     deviceptr(f_sigqsat, f_pabs, f_sigs, f_th, f_exn, f_exn_ref, f_rho_dry_ref) &
        !$acc&     deviceptr(f_rv, f_rc, f_ri, f_rr, f_rs, f_rg) &
        !$acc&     deviceptr(f_cf_mf, f_rc_mf, f_ri_mf) &
        !$acc&     deviceptr(f_rvs, f_rcs, f_ris, f_ths) &
        !$acc&     deviceptr(f_cldfr, f_icldfr, f_wcldfr)

        ! Initialize on GPU
        !$acc kernels
        PRHODJ = f_rho_dry_ref
        PZZ = 0.0_C_FLOAT
        PMFCONV = 0.0_C_FLOAT
        PWEIGHT_MF_CLOUD = 0.0_C_FLOAT
        PSSIO = 0.0_C_FLOAT
        PSSIU = 0.0_C_FLOAT
        PIFR = 0.0_C_FLOAT
        PSRCS = 0.0_C_FLOAT
        !$acc end kernels

        LMFCONV = .FALSE.
        OCOMPUTE_SRC = .FALSE.

        !-----------------------------------------------------------------------
        ! Call the GPU-accelerated ICE_ADJUST routine
        !-----------------------------------------------------------------------
        CALL ICE_ADJUST(                                                       &
            D, CST, ICEP, NEBN, TURBN, PARAMI, BUCONF, krr,                    &
            'BRID',                                                            &
            timestep, f_sigqsat,                                               &
            PRHODJ, f_exn_ref, f_rho_dry_ref, f_sigs, LMFCONV, PMFCONV,       &
            f_pabs, PZZ,                                                       &
            f_exn, f_cf_mf, f_rc_mf, f_ri_mf, PWEIGHT_MF_CLOUD,               &
            f_icldfr, f_wcldfr, PSSIO, PSSIU, PIFR,                           &
            f_rv, f_rc, f_rvs, f_rcs, f_th, f_ths,                            &
            OCOMPUTE_SRC, PSRCS, f_cldfr,                                     &
            f_rr, f_ri, f_ris, f_rs, f_rg, TBUDGETS, 0                        &
        )

        !$acc end data

        !-----------------------------------------------------------------------
        ! Cleanup
        !-----------------------------------------------------------------------
        DEALLOCATE(PRHODJ, PZZ, PMFCONV, PWEIGHT_MF_CLOUD)
        DEALLOCATE(PSSIO, PSSIU, PIFR, PSRCS)

    END SUBROUTINE c_ice_adjust_acc_wrap

END MODULE phyex_bridge_acc
