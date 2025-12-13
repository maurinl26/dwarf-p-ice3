MODULE phyex_bridge
    USE ISO_C_BINDING
    ! Import the original routine and precision module
    USE MODI_ICE_ADJUST, ONLY : ICE_ADJUST
    USE PARKIND1, ONLY : JPIM, JPRB

    IMPLICIT NONE

CONTAINS

    ! This function is callable from Python (C compatible)
    SUBROUTINE c_ice_adjust_wrap(nlon, nlev, ptr_pt, ptr_pq, ptr_pp) BIND(C, name="c_ice_adjust")
        ! 1. Define C-compatible arguments
        INTEGER(C_INT), VALUE, INTENT(IN) :: nlon, nlev
        TYPE(C_PTR), VALUE, INTENT(IN)    :: ptr_pt, ptr_pq, ptr_pp

        ! 2. Define Fortran Pointers to map the data
        REAL(KIND=JPRB), POINTER, DIMENSION(:,:) :: f_pt, f_pq, f_pp
        
        ! 3. Convert C Pointers to Fortran Arrays (No data copy!)
        CALL C_F_POINTER(ptr_pt, f_pt, [nlon, nlev])
        CALL C_F_POINTER(ptr_pq, f_pq, [nlon, nlev])
        CALL C_F_POINTER(ptr_pp, f_pp, [nlon, nlev])

        ! 4. Call the actual PHYEX routine
        ! We assume KIDIA=1, KFDIA=nlon for the whole array
        CALL ICE_ADJUST(1, nlon, nlon, nlev, f_pt, f_pq, f_pp)

    END SUBROUTINE c_ice_adjust_wrap

END MODULE phyex_bridge