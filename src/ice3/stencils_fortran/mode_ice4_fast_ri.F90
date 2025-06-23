module mode_ice4_fast_ri

    implicit none 
    contains

    subroutine ice4_fast_ri(kproma, ksize, ldsoft, ldcompute, &
       &xlbi, xlbexi, xdi, x0depi, x2depi, &
       &c_rtmin, i_rtmin, &
       &prhodref, &
       &pai, pcj, pcit, &
       &pssi, &
       &prct, prit, &
       &prcberi)
!
      implicit none
!
!*       0.1   declarations of dummy arguments :
!
      integer, intent(in)    :: kproma, ksize
      real, intent(in) :: xlbi, xlbexi, xdi, x0depi, x2depi
      real, intent(in) :: c_rtmin, i_rtmin
      logical, intent(in)    :: ldsoft
      logical, dimension(kproma), intent(in)    :: ldcompute
      real, dimension(kproma), intent(in)    :: prhodref ! reference density
      real, dimension(kproma), intent(in)    :: pai      ! thermodynamical function
      real, dimension(kproma), intent(in)    :: pcj      ! function to compute the ventilation coefficient
      real, dimension(kproma), intent(in)    :: pcit     ! pristine ice conc. at t
      real, dimension(kproma), intent(in)    :: pssi     ! supersaturation over ice
      real, dimension(kproma), intent(in)    :: prct     ! cloud water m.r. at t
      real, dimension(kproma), intent(in)    :: prit     ! pristine ice m.r. at t
      real, dimension(kproma), intent(inout) :: prcberi  ! bergeron-findeisen effect
!
!*       0.2  declaration of local variables
      integer :: jl
!
!-------------------------------------------------------------------------------
!*       7.2    bergeron-findeisen effect: rcberi
!
do jl=1, ksize
  if(pssi(jl)>0. &
      .and. prct(jl)>c_rtmin &
      .and. prit(jl)>i_rtmin &
      .and. pcit(jl)>1.e-20 &
      .and. ldcompute(jl)) then
    if(.not. ldsoft) then
      prcberi(jl) = min(1.e8, xlbi*(prhodref(jl)*prit(jl)/pcit(jl))**xlbexi) ! lbda_i
      prcberi(jl) = ( pssi(jl) / (prhodref(jl)*pai(jl)) ) * pcit(jl) * &
                    ( x0depi/prcberi(jl) + x2depi*pcj(jl)*pcj(jl)/prcberi(jl)**(xdi+2.0) )
    endif
  else
    prcberi(jl) = 0.
  endif
enddo
!
   end subroutine ice4_fast_ri

end module mode_ice4_fast_ri