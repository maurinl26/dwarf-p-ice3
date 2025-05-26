! Created by  on 09/05/2025.

module mode_ice4_stepping
   implicit none

   subroutine time_integration_and_limiter(&
           &ldsigma_rc, ldaucv_adju, ldext_tend, &
         &kproma, kmicro, &
&ldmicro, &
&ptstep,&  ! double time step (single if cold start)
&krr, &     ! number of moist variable
&osave_micro, oelec, &         ! if true, cloud electricity is activated
   &pexn, prhodref, &! reference density
   &k1, k2, ppres, pcf, psigma_rc, pcit, &
   &pvart, & !packed variables
   &phlc_hrc, phlc_hcf, phli_hri, phli_hcf, &
   &prainfr, &
   &pextpk, pbu_sum, &
   &prrevav, platham_iaggs &
   &)

      logical, intent(in)    :: ldsigma_rc
      logical, intent(in)    :: ldaucv_adju
      logical, intent(in)    :: ldext_tend
      integer, intent(in)    :: kproma ! cache-blocking factor for microphysic loop
      integer, intent(in)    :: kmicro ! case r_x>0 locations
      logical, dimension(kproma), intent(in)  :: ldmicro
      real, intent(in)    :: ptstep  ! double time step (single if cold start)
      integer, intent(in)    :: krr     ! number of moist variable
      logical, intent(in)    :: osave_micro   ! if true, save the microphysical tendencies
      logical, intent(in)    :: oelec         ! if true, cloud electricity is activated
!
      real, dimension(kproma), intent(in)    :: pexn    ! exner function
      real, dimension(kproma), intent(in)    :: prhodref! reference density
      integer, dimension(kproma), intent(in)    :: k1, k2 ! used to replace the count and pack intrinsics on variables
      real, dimension(kproma), intent(in)    :: ppres
      real, dimension(kproma), intent(in)    :: pcf ! cloud fraction
      real, dimension(kproma), intent(inout) :: psigma_rc
      real, dimension(kproma), intent(inout) :: pcit
      real, dimension(kproma, 0:7), intent(inout) :: pvart !packed variables
      real, dimension(kproma), intent(inout) :: phlc_hrc
      real, dimension(kproma), intent(inout) :: phlc_hcf
      real, dimension(kproma), intent(inout) :: phli_hri
      real, dimension(kproma), intent(inout) :: phli_hcf
      real, dimension(d%nijt, d%nkt), intent(inout) :: prainfr
      real, dimension(kproma, 0:7), intent(inout) :: pextpk !to take into acount external tendencies inside the splitting
      real, dimension(kproma, ibunum - ibunum_extra), intent(out)   :: pbu_sum
      real, dimension(kproma), intent(out)   :: prrevav
      real, dimension(merge(kproma, 0, oelec)), intent(in)    :: platham_iaggs ! e function to simulate

      do jl = 1, kmicro
         if (llcompute(jl)) then
            zmaxtime(jl) = (ptstep - ztime(jl)) ! remaining time until the end of the timestep
         else
            zmaxtime(jl) = 0.
         end if
      end do
      !We need to adjust tendencies when temperature reaches 0
      if (parami%lfeedbackt) then

         do jl = 1, kmicro
            !is zb(:, ith) enough to change temperature sign?
            zx = cst%xtt/pexn(jl)
            if ((pvart(jl, ith) - zx)*(pvart(jl, ith) + zb(jl, ith) - zx) < 0.) then
               zmaxtime(jl) = 0.
            end if
            !can za(:, ith) make temperature change of sign?
            if (abs(za(jl, ith)) > 1.e-20) then
               ztime_threshold = (zx - zb(jl, ith) - pvart(jl, ith))/za(jl, ith)
               if (ztime_threshold > 0.) then
                  zmaxtime(jl) = min(zmaxtime(jl), ztime_threshold)
               end if
            end if
         end do
      end if

      !we need to adjust tendencies when a species disappears
      !when a species is missing, only the external tendencies can be negative (and we must keep track of it)
      do jv = 1, krr
         do jl = 1, kmicro
            if (za(jl, jv) < -1.e-20 .and. pvart(jl, jv) > zrsmin(jv)) then
               zmaxtime(jl) = min(zmaxtime(jl), -(zb(jl, jv) + pvart(jl, jv))/za(jl, jv))
               zmaxtime(jl) = max(zmaxtime(jl), cst%xmnh_tiny) !to prevent rounding errors
            end if
         end do
      end do
      !we stop when the end of the timestep is reached
      do jl = 1, kmicro
         if (ztime(jl) + zmaxtime(jl) >= ptstep) then
            llcompute(jl) = .false.
         end if
      end do
      !we must recompute tendencies when the end of the sub-timestep is reached
      if (parami%xtstep_ts /= 0.) then
         do jl = 1, kmicro
            if ((iiter(jl) < inb_iter_max) .and. (ztime(jl) + zmaxtime(jl) > ztime_lastcall(jl) + ztstep)) then
               zmaxtime(jl) = ztime_lastcall(jl) - ztime(jl) + ztstep
               llcompute(jl) = .false.
            end if
         end do
      end if

      !we must recompute tendencies when the maximum allowed change is reached
      !when a species is missing, only the external tendencies can be active and we do not want to recompute
      !the microphysical tendencies when external tendencies are negative (results won't change because species was already missing)
      if (parami%xmrstep /= 0.) then
         if (ll_any_iter) then
            ! in this case we need to remember the initial mixing ratios used to compute the tendencies
            ! because when mixing ratio has evolved more than a threshold, we must re-compute tendencies
            ! thus, at first iteration (ie when llcpz0rt=.true.) we copy pvart into z0rt
            do jv = 1, krr
               if (llcpz0rt) then
                  z0rt(1:kmicro, jv) = pvart(1:kmicro, jv)
               end if
               do jl = 1, kmicro
                  if (iiter(jl) < inb_iter_max .and. abs(za(jl, jv)) > 1.e-20) then
                     ztime_threshold1d(jl) = (sign(1., za(jl, jv))*parami%xmrstep + &
                                           &z0rt(jl, jv) - pvart(jl, jv) - zb(jl, jv))/za(jl, jv)
                  else
                     ztime_threshold1d(jl) = -1.
                  end if
               end do
               do jl = 1, kmicro
                  if (ztime_threshold1d(jl) >= 0 .and. ztime_threshold1d(jl) < zmaxtime(jl) .and. &
                     &(pvart(jl, jv) > zrsmin(jv) .or. za(jl, jv) > 0.)) then
                     zmaxtime(jl) = min(zmaxtime(jl), ztime_threshold1d(jl))
                     llcompute(jl) = .false.
                  end if
               end do
               if (jv == 1) then
                  do jl = 1, kmicro
                     zmaxb(jl) = abs(zb(jl, jv))
                  end do
               else
                  do jl = 1, kmicro
                     zmaxb(jl) = max(zmaxb(jl), abs(zb(jl, jv)))
                  end do
               end if
            end do
            llcpz0rt = .false.

            do jl = 1, kmicro
               if (iiter(jl) < inb_iter_max .and. zmaxb(jl) > parami%xmrstep) then
                  zmaxtime(jl) = 0.
                  llcompute(jl) = .false.
               end if
            end do
         end if ! ll_any_iter
      end if ! xmrstep/=0.
      !-------------------------------------------------------------------------------
      !
      !***       4.7 new values of variables for next iteration
      !
      !
      do jv = 0, krr
         do jl = 1, kmicro
            if (ldmicro(jl)) then
               pvart(jl, jv) = pvart(jl, jv) + za(jl, jv)*zmaxtime(jl) + zb(jl, jv)
            end if
         end do
      end do

      do jl = 1, kmicro
         if (pvart(jl, iri) <= 0. .and. ldmicro(jl)) pcit(jl) = 0.
         ztime(jl) = ztime(jl) + zmaxtime(jl)
      end do
      !-------------------------------------------------------------------------------
      !
      !***       4.8 mixing ratio change due to each process
      !
      if (buconf%lbu_enable .or. osave_micro) then
         !mixing ratio change due to a tendency
         do jv = 1, ibunum - ibunum_mr - ibunum_extra
            do jl = 1, kmicro
               pbu_sum(jl, jv) = pbu_sum(jl, jv) + zbu_inst(jl, jv)*zmaxtime(jl)
            end do
         end do

         !mixing ratio change due to a mixing ratio change

         do jv = ibunum - ibunum_mr - ibunum_extra + 1, ibunum - ibunum_extra
            do jl = 1, kmicro
               pbu_sum(jl, jv) = pbu_sum(jl, jv) + zbu_inst(jl, jv)
            end do
         end do

         !extra contribution as a mixing ratio change
         do jv = ibunum - ibunum_extra + 1, ibunum
            jjv = ibuextraind(jv)

            do jl = 1, kmicro
               pbu_sum(jl, jjv) = pbu_sum(jl, jjv) + zbu_inst(jl, jv)
            end do
         end do
      end if
      !-----------------------------

   end subroutine time_integration_and_limiter

end module mode_ice4_stepping
