module mode_ice4_rrhong
   implicit none
contains
   subroutine ice4_rrhong(kproma, ksize,  &
                        &xtt, r_rtmin, lfeedbackt, &
                        &ldcompute, &
                        &pexn, plvfact, plsfact, &
                        &pt, prrt, ptht, &
                        &prrhong_mr)
      !
      implicit none
      !
      real, intent(in) :: xtt, r_rtmin
      logical, intent(in) :: lfeedbackt
      integer, intent(in) :: kproma, ksize
      logical, dimension(kproma), intent(in)    :: ldcompute
      real, dimension(kproma), intent(in)    :: pexn     
      real, dimension(kproma), intent(in)    :: plvfact
      real, dimension(kproma), intent(in)    :: plsfact
      real, dimension(kproma), intent(in)    :: pt     
      real, dimension(kproma), intent(in)    :: prrt
      real, dimension(kproma), intent(in)    :: ptht
      real, dimension(size(pexn)), intent(out)   :: prrhong_mr

      integer :: jl
      !
      do jl = 1, ksize
         if (pt(jl) < xtt - 35.0 &
            .and. prrt(jl) > r_rtmin &
            .and. ldcompute(jl)) then
            prrhong_mr(jl) = prrt(jl)
            if (lfeedbackt) then
               prrhong_mr(jl) = min(prrhong_mr(jl), max(0., ((xtt - 35.)/pexn(jl) - ptht(jl))/(plsfact(jl) - plvfact(jl))))
            end if
         else
            prrhong_mr(jl) = 0.
         end if
      end do
      !
   end subroutine ice4_rrhong

   subroutine ice4_rrhong_postprocessing(ksize, kproma, &
            prrhong_mr, plsfact, plvfact, &
            prrt, prrg, &
            pexn, ptht, zt)

      integer, intent(in) :: ksize, kproma
      real, dimension(kproma), intent(in) :: pexn
      real, dimension(kproma), intent(in) :: plsfact
      real, dimension(kproma), intent(in) :: plvfact
      real, dimension(kproma), intent(inout) :: prrt
      real, dimension(kproma), intent(inout) :: prrg
      real, dimension(kproma), intent(inout) :: ptht
      real, dimension(kproma), intent(inout) :: prrhong_mr
      real, dimension(kproma), intent(out) :: zt

      integer :: jl

      do jl=1, ksize
         ptht(jl) = ptht(jl) + prrhong_mr(jl)*(plsfact(jl)-plvfact(jl))
         zt(jl) = ptht(jl) * pexn(jl)
         prrt(jl) = prrt(jl) - prrhong_mr(jl)
         prrg(jl)= prrg(jl)+ prrhong_mr(jl)
      enddo

   end subroutine ice4_rrhong_postprocessing

end module mode_ice4_rrhong
