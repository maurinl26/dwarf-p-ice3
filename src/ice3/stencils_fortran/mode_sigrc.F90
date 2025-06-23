module mode_sigrc
   implicit none
contains
   subroutine sigrc(nijt, nkt, zq1, psigrc, zsrc_1d)

      implicit none

      integer, intent(in) :: nijt, nkt
      real, dimension(nijt), intent(in) :: zq1
      real, dimension(-11:22), intent(in) :: zsrc_1d

      real, dimension(nijt, nkt), intent(inout) :: pcldfr

      real, dimension(nijt, nkt), intent(out) :: psigrc

      real, dimension(nijt) :: zcond, zsigma

      integer :: inq1
      real :: zinc

      integer :: jij, jk
      integer :: iktb, ikte
      integer :: iijb, iije

      do jk = iktb, ikte

         do jij = iijb, iije
            !total condensate
            if (zq1(jij) > 0. .and. zq1(jij) <= 2) then
               zcond(jij) = min(exp(-1.) + .66*zq1(jij) + .086*zq1(jij)**2, 2.) ! we use the min function for continuity
            else if (zq1(jij) > 2.) then
               zcond(jij) = zq1(jij)
            else
               zcond(jij) = exp(1.2*zq1(jij) - 1.)
            end if
            zcond(jij) = zcond(jij)*zsigma(jij)

            !cloud fraction
            if (zcond(jij) < 1.e-12) then
               pcldfr(jij, jk) = 0.
            else
               pcldfr(jij, jk) = max(0., min(1., 0.5 + 0.36*atan(1.55*zq1(jij))))
            end if
            if (pcldfr(jij, jk) == 0.) then
               zcond(jij) = 0.
            end if

            inq1 = min(max(-22, floor(min(100., max(-100., 2*zq1(jij))))), 10)  !inner min/max prevents sigfpe when 2*zq1 does not fit into an int
            zinc = 2.*zq1(jij) - inq1

            psigrc(jij, jk) = min(1., (1.-zinc)*zsrc_1d(inq1) + zinc*zsrc_1d(inq1 + 1))
         end do
      end do

   end subroutine sigrc
end module mode_sigrc

