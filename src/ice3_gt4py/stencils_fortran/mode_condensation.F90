module mode_condensation
contains
   subroutine condensation(nijb, nije, nktb, nkte, nijt, nkt,  &
       &xrv, xrd, xalpi, xbetai, xgami, xalpw, xbetaw, xgamw,  &
       &xtmaxmix, xtminmix,                                    &
       &osigmas, ocnd2, ouseri,                                &
       &hfrac_ice, hcondens, lstatnw,                          &
       &ppabs, pt,                                             &
       &prv_in, prc_in, pri_in,                                &
       &psigs,                                                 &
       &psigqsat,                                              &
       &plv, pls, pcph,                                        &
       &pt_out, prv_out, prc_out, pri_out, pcldfr, zq1)

      implicit none

      integer, intent(in) :: nijb, nije, nktb, nkte, nijt, nkt
      real, intent(in) :: xrv, xrd, xalpi, xbetai, xgami, xalpw, xbetaw, xgamw
      real, intent(in) :: xtmaxmix, xtminmix
      integer, intent(in)     :: hfrac_ice
      integer, intent(in)     :: hcondens
      logical, intent(in)     :: lstatnw
       logical, intent(in)    :: ouseri ! logical switch to compute both liquid and solid condensate (ouseri=.true.)or only solid condensate (ouseri=.false.)
      logical, intent(in)     :: osigmas! use present global sigma_s values or that from turbulence scheme
      logical, intent(in)     :: ocnd2  ! logical switch to sparate liquid and ice more rigid (defalt value : .false.)
      real, dimension(nijt, nkt), intent(in)    :: ppabs  ! pressure (pa)
      real, dimension(nijt, nkt), intent(in)    :: pt     ! grid scale t  (k)
      real, dimension(nijt, nkt), intent(in)    :: prv_in ! grid scale water vapor mixing ratio (kg/kg) in input
      real, dimension(nijt, nkt), intent(in)    :: prc_in ! grid scale r_c mixing ratio (kg/kg) in input
      real, dimension(nijt, nkt), intent(in)    :: pri_in ! grid scale r_i (kg/kg) in input
      real, dimension(nijt, nkt), intent(in)    :: psigs  ! sigma_s from turbulence scheme
      real, dimension(nijt), intent(in)    :: psigqsat ! use an extra "qsat" variance contribution (osigmas case) multiplied by psigqsat
      real, dimension(nijt, nkt), intent(in)    :: plv    ! latent heat l_v
      real, dimension(nijt, nkt), intent(in)    :: pls    ! latent heat l_s
      real, dimension(nijt, nkt), intent(in)    :: pcph   ! specific heat c_ph

      ! out
      real, dimension(nijt, nkt), intent(out)   :: pt_out
      real, dimension(nijt, nkt), intent(out)   :: prv_out! grid scale water vapor mixing ratio (kg/kg) in output
      real, dimension(nijt, nkt), intent(out)   :: prc_out! grid scale r_c mixing ratio (kg/kg) in output
      real, dimension(nijt, nkt), intent(out)   :: pri_out! grid scale r_i (kg/kg) in output
      real, dimension(nijt, nkt), intent(out)   :: pcldfr ! cloud fraction
      real, dimension(nijt, nkt), intent(out)   :: zq1
!
!
!*       0.2   declarations of local variables :
!
      integer :: jij, jk
      real, dimension(nijt, nkt) :: zrt     ! work arrays for t_l and total water mixing ratio
      real :: zlvs                                      ! thermodynamics
      real, dimension(nijt) :: zpv, zpiv, zqsl, zqsi ! thermodynamics
      real :: zah
      real, dimension(nijt) :: za, zb, zsbar, zsigma ! related to computation of sig_s
      real, dimension(nijt) :: zcond
      real, dimension(nijt) :: zfrac           ! ice fraction
      real :: zprifact
!
!
!*       0.3  definition of constants :
!
!-------------------------------------------------------------------------------

      real, dimension(-22:11), parameter :: ZSRC_1D = (/ &
                                            0., 0., 2.0094444e-04, 0.316670e-03, &
                                            4.9965648e-04, 0.785956e-03, 1.2341294e-03, 0.193327e-02, &
                                            3.0190963e-03, 0.470144e-02, 7.2950651e-03, 0.112759e-01, &
                                            1.7350994e-02, 0.265640e-01, 4.0427860e-02, 0.610997e-01, &
                                            9.1578111e-02, 0.135888e+00, 0.1991484, 0.230756e+00, &
                                            0.2850565, 0.375050e+00, 0.5000000, 0.691489e+00, &
                                            0.8413813, 0.933222e+00, 0.9772662, 0.993797e+00, &
                                            0.9986521, 0.999768e+00, 0.9999684, 0.999997e+00, &
                                            1.0000000, 1.000000/)
!
!-------------------------------------------------------------------------------
      pcldfr(:, :) = 0. ! initialize values
      prv_out(:, :) = 0. ! initialize values
      prc_out(:, :) = 0. ! initialize values
      pri_out(:, :) = 0. ! initialize values
!-------------------------------------------------------------------------------
! store total water mixing ratio
      do jk = nktb, nkte
         do jij = nijb, nije
            zrt(jij, jk) = prv_in(jij, jk) + prc_in(jij, jk) + pri_in(jij, jk)*zprifact
         end do
      end do
!-------------------------------------------------------------------------------
! preliminary calculations
! latent heat of vaporisation/sublimation
!-------------------------------------------------------------------------------
!
      do jk = nktb, nkte
         if (.not. ocnd2) then
            ! latent heats
            ! saturated water vapor mixing ratio over liquid water and ice
            do jij = nijb, nije
               zpv(jij) = min(exp(xalpw - xbetaw/pt(jij, jk) - xgamw*log(pt(jij, jk))), .99*ppabs(jij, jk))
               zpiv(jij) = min(exp(xalpi - xbetai/pt(jij, jk) - xgami*log(pt(jij, jk))), .99*ppabs(jij, jk))
            end do
         end if

         if (ouseri .and. .not. ocnd2) then
            do jij = nijb, nije
               if (prc_in(jij, jk) + pri_in(jij, jk) > 1.e-20) then
                  zfrac(jij) = pri_in(jij, jk)/(prc_in(jij, jk) + pri_in(jij, jk))
               end if
            end do
            do jij = nijb, nije
               if (hfrac_ice == 3) then
                  zfrac(jij) = max(0., min(1., zfrac(jij)))
               else if (hfrac_ice == 0 ) then
                  zfrac(jij) = max(0., min(1., (xtmaxmix - pt(jij, jk)) / (xtmaxmix - xtminmix)))
               end if
            end do
         end if

         do jij = nijb, nije
            zqsl(jij) = xrd/xrv*zpv(jij)/(ppabs(jij, jk) - zpv(jij))
            zqsi(jij) = xrd/xrv*zpiv(jij)/(ppabs(jij, jk) - zpiv(jij))

            ! interpolate between liquid and solid as function of temperature
            zqsl(jij) = (1.-zfrac(jij))*zqsl(jij) + zfrac(jij)*zqsi(jij)
            zlvs = (1.-zfrac(jij))*plv(jij, jk) + &
            & zfrac(jij)*pls(jij, jk)

            ! coefficients a and b
            zah = zlvs*zqsl(jij)/(xrv*pt(jij, jk)**2)*(xrv*zqsl(jij)/xrd + 1.)
            za(jij) = 1./(1.+zlvs/pcph(jij, jk)*zah)
            zb(jij) = zah*za(jij)
            zsbar(jij) = za(jij)*(zrt(jij, jk) - zqsl(jij) + &
            & zah*zlvs*(prc_in(jij, jk) + pri_in(jij, jk)*zprifact)/pcph(jij, jk))
         end do

         if (osigmas) then
            do jij = nijb, nije
               if (psigqsat(jij) /= 0.) then
                  if (.not. lstatnw) then
                     zsigma(jij) = sqrt((2*psigs(jij, jk))**2 + (psigqsat(jij)*zqsl(jij)*za(jij))**2)
                  end if
               else
                  if (.not. lstatnw) then
                     zsigma(jij) = 2*psigs(jij, jk)
                  end if
               end if
            end do
         end if

         do jij = nijb, nije
            zsigma(jij) = max(1.e-10, zsigma(jij))
            ! normalized saturation deficit
            zq1(jij, jk) = zsbar(jij)/zsigma(jij)
         end do

         ! 0 is for "cb02"
         ! 1 is for "gaus"
         if (hcondens == 0) then
            do jij = nijb, nije
               !total condensate
               if (zq1(jij, jk) > 0. .and. zq1(jij, jk) <= 2) then
                  zcond(jij) = min(exp(-1.) + .66*zq1(jij, jk) + .086*zq1(jij, jk)**2, 2.) ! we use the min function for continuity
               else if (zq1(jij, jk) > 2.) then
                  zcond(jij) = zq1(jij, jk)
               else
                  zcond(jij) = exp(1.2*zq1(jij, jk) - 1.)
               end if
               zcond(jij) = zcond(jij)*zsigma(jij)

               !cloud fraction
               if (zcond(jij) < 1.e-12) then
                  pcldfr(jij, jk) = 0.
               else
                  pcldfr(jij, jk) = max(0., min(1., 0.5 + 0.36*atan(1.55*zq1(jij, jk))))
               end if
               if (pcldfr(jij, jk) == 0.) then
                  zcond(jij) = 0.
               end if
            end do
         end if !hcondens

         if (.not. ocnd2) then
            do jij = nijb, nije
               prc_out(jij, jk) = (1.-zfrac(jij))*zcond(jij) ! liquid condensate
               pri_out(jij, jk) = zfrac(jij)*zcond(jij)   ! solid condensate
               pt_out(jij, jk) = pt(jij, jk) + ((prc_out(jij, jk) - prc_in(jij, jk))*plv(jij, jk) + &
                    &(pri_out(jij, jk) - pri_in(jij, jk))*pls(jij, jk)) &
                  & /pcph(jij, jk)
               prv_out(jij, jk) = zrt(jij, jk) - prc_out(jij, jk) - pri_out(jij, jk)*zprifact
            end do
         end if ! end ocnd2
      end do

   end subroutine condensation

   ! subroutine sigrc_computation(nijt, nkt, nkte, nktb, nije, nijb, hlambda3, zq1, src_1d, psigrc)

   !    integer, intent(in) :: nijt, nkt
   !    integer, intent(in) :: nije, nijb
   !    integer, intent(in) :: nkte, nktb
   !    real, dimension(nijt, nkt), intent(in) :: zq1
   !    real, dimension(nijt, nkt), intent(out) :: psigrc
   !    real, dimension(34), intent(in) :: src_1d

   !    integer, dimension(nijt, nkt) :: inq1
   !    real, dimension(nijt, nkt) ::  zinc

   !    character(len=6), intent(in) :: hlambda3

   !    do jk = nkte, nktb
   !       do jij = nijb, nije

   !          inq1 = min(max(-22, floor(min(100., max(-100., 2*zq1(jij, jk))))), 10)  !inner min/max prevents sigfpe when 2*zq1 does not fit into an int
   !          zinc = 2.*zq1(jij, jk) - inq1
   !          psigrc(jij, jk) = min(1., (1.-zinc)*ZSRC_1D(inq1) + zinc*ZSRC_1D(inq1 + 1))

   !          if (hlambda3 == "CB") then
   !             psigrc(jij, jk) = psigrc(jij, jk)*min(3., max(1., 1.-zq1(jij, jk)))
   !          end if

   !       end do
   !    end do

   ! end subroutine sigrc_computation

end module mode_condensation
