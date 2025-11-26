!MNH_LIC Copyright 2018-2021 CNRS, Meteo-France and Universite Paul Sabatier
!MNH_LIC This is part of the Meso-NH software governed by the CeCILL-C licence
!MNH_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt
!MNH_LIC for details. version 1.
! Created for dwarf-p-ice3 reproducibility tests
! Simplified wrappers matching Python GT4Py implementations

module mode_ice4_stepping
   implicit none
   contains

   !---------------------------------------------------------------------------
   ! ICE4_STEPPING_HEAT: Compute thermodynamic variables
   ! Matches: ice3.stencils.ice4_stepping.ice4_stepping_heat
   !---------------------------------------------------------------------------
   subroutine ice4_stepping_heat(kproma, ksize, &
           &xcpd, xcpv, xcl, xci, &
           &xtt, xlvtt, xlstt, &
           &pexn, ptht, &
           &prvt, prct, prrt, prit, prst, prgt, &
           &pt, plsfact, plvfact)

      ! Arguments
      integer, intent(in) :: kproma, ksize
      real, intent(in) :: xcpd, xcpv, xcl, xci
      real, intent(in) :: xtt, xlvtt, xlstt
      real, dimension(kproma, ksize), intent(in) :: pexn, ptht
      real, dimension(kproma, ksize), intent(in) :: prvt, prct, prrt, prit, prst, prgt
      real, dimension(kproma, ksize), intent(out) :: pt, plsfact, plvfact
      
      ! Local variables
      real :: zcp
      integer :: jl, jk

      ! Compute for each grid point
      do jk = 1, ksize
        do jl = 1, kproma
          ! Specific heat capacity at constant pressure
          zcp = xcpd + xcpv*prvt(jl,jk) + xcl*(prct(jl,jk)+prrt(jl,jk)) &
                + xci*(prit(jl,jk)+prst(jl,jk)+prgt(jl,jk))
          
          ! Temperature from potential temperature
          pt(jl,jk) = ptht(jl,jk) * pexn(jl,jk)
          
          ! Sublimation latent heat factor Ls/Cp
          plsfact(jl,jk) = (xlstt + (xcpv-xci)*(pt(jl,jk)-xtt)) / zcp
          
          ! Vaporisation latent heat factor Lv/Cp
          plvfact(jl,jk) = (xlvtt + (xcpv-xcl)*(pt(jl,jk)-xtt)) / zcp
        enddo
      enddo

   end subroutine ice4_stepping_heat


   !---------------------------------------------------------------------------
   ! STATE_UPDATE: Update state variables after time integration
   ! Matches: ice3.stencils.ice4_stepping.state_update
   !---------------------------------------------------------------------------
   subroutine state_update(kproma, ksize, &
           &ptht, ptheta_b, ptheta_tnd_a, &
           &prct, prrt, prit, prst, prgt, &
           &prc_b, prr_b, pri_b, prs_b, prg_b, &
           &prc_tnd_a, prr_tnd_a, pri_tnd_a, prs_tnd_a, prg_tnd_a, &
           &pdelta_t_micro, ldmicro, pcit, pt_micro)

      ! Arguments
      integer, intent(in) :: kproma, ksize
      real, dimension(kproma, ksize), intent(inout) :: ptht
      real, dimension(kproma, ksize), intent(in) :: ptheta_b, ptheta_tnd_a
      real, dimension(kproma, ksize), intent(inout) :: prct, prrt, prit, prst, prgt
      real, dimension(kproma, ksize), intent(in) :: prc_b, prr_b, pri_b, prs_b, prg_b
      real, dimension(kproma, ksize), intent(in) :: prc_tnd_a, prr_tnd_a, pri_tnd_a, prs_tnd_a, prg_tnd_a
      real, dimension(kproma, ksize), intent(in) :: pdelta_t_micro
      logical, dimension(kproma, ksize), intent(in) :: ldmicro
      real, dimension(kproma, ksize), intent(inout) :: pcit, pt_micro
      
      ! Local variables
      integer :: jl, jk

      ! Update state variables: X(t+dt) = X(t) + dX/dt * dt + dX_process
      do jk = 1, ksize
        do jl = 1, kproma
          ! Update potential temperature
          ptht(jl,jk) = ptht(jl,jk) + ptheta_tnd_a(jl,jk) * pdelta_t_micro(jl,jk) + ptheta_b(jl,jk)
          
          ! Update mixing ratios
          prct(jl,jk) = prct(jl,jk) + prc_tnd_a(jl,jk) * pdelta_t_micro(jl,jk) + prc_b(jl,jk)
          prrt(jl,jk) = prrt(jl,jk) + prr_tnd_a(jl,jk) * pdelta_t_micro(jl,jk) + prr_b(jl,jk)
          prit(jl,jk) = prit(jl,jk) + pri_tnd_a(jl,jk) * pdelta_t_micro(jl,jk) + pri_b(jl,jk)
          prst(jl,jk) = prst(jl,jk) + prs_tnd_a(jl,jk) * pdelta_t_micro(jl,jk) + prs_b(jl,jk)
          prgt(jl,jk) = prgt(jl,jk) + prg_tnd_a(jl,jk) * pdelta_t_micro(jl,jk) + prg_b(jl,jk)
          
          ! Special case: if ice disappears, ice crystal concentration also disappears
          if (prit(jl,jk) <= 0.0 .and. ldmicro(jl,jk)) then
            pt_micro(jl,jk) = pt_micro(jl,jk) + pdelta_t_micro(jl,jk)
            pcit(jl,jk) = 0.0
          endif
        enddo
      enddo

   end subroutine state_update


   !---------------------------------------------------------------------------
   ! EXTERNAL_TENDENCIES_UPDATE: Remove external tendencies
   ! Matches: ice3.stencils.ice4_stepping.external_tendencies_update
   !---------------------------------------------------------------------------
   subroutine external_tendencies_update(kproma, ksize, &
           &ptht, ptheta_tnd_ext, &
           &prct, prrt, prit, prst, prgt, &
           &prc_tnd_ext, prr_tnd_ext, pri_tnd_ext, prs_tnd_ext, prg_tnd_ext, &
           &ldmicro, dt)

      ! Arguments
      integer, intent(in) :: kproma, ksize
      real, dimension(kproma, ksize), intent(inout) :: ptht
      real, dimension(kproma, ksize), intent(in) :: ptheta_tnd_ext
      real, dimension(kproma, ksize), intent(inout) :: prct, prrt, prit, prst, prgt
      real, dimension(kproma, ksize), intent(in) :: prc_tnd_ext, prr_tnd_ext, pri_tnd_ext, prs_tnd_ext, prg_tnd_ext
      logical, dimension(kproma, ksize), intent(in) :: ldmicro
      real, intent(in) :: dt
      
      ! Local variables
      integer :: jl, jk

      ! Remove external tendencies that were temporarily added
      do jk = 1, ksize
        do jl = 1, kproma
          if (ldmicro(jl,jk)) then
            ptht(jl,jk) = ptht(jl,jk) - ptheta_tnd_ext(jl,jk) * dt
            prct(jl,jk) = prct(jl,jk) - prc_tnd_ext(jl,jk) * dt
            prrt(jl,jk) = prrt(jl,jk) - prr_tnd_ext(jl,jk) * dt
            prit(jl,jk) = prit(jl,jk) - pri_tnd_ext(jl,jk) * dt
            prst(jl,jk) = prst(jl,jk) - prs_tnd_ext(jl,jk) * dt
            prgt(jl,jk) = prgt(jl,jk) - prg_tnd_ext(jl,jk) * dt
          endif
        enddo
      enddo

   end subroutine external_tendencies_update

end module mode_ice4_stepping
