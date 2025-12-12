module mode_ice4_nucleation
   implicit none
contains
   subroutine ice4_nucleation(ksize, &
                              kproma, &
                              xtt, v_rtmin, xalpw, xbetaw, xgamw, &
                              xalpi, xbetai, xgami, xepsilo, &
                              xnu10, xnu20, xalpha1, xalpha2, xbeta1, xbeta2, &
                              xmnu0, &
                              lfeedbackt, &
                              ldcompute, &
                              ptht, ppabst, prhodref, pexn, plsfact, pt, &
                              prvt, &
                              pcit, prvheni_mr)
      !
      implicit none
      !
      real, intent(in) :: xtt, v_rtmin, xalpw, xbetaw, xgamw
      real, intent(in) :: xalpi, xbetai, xgami, xepsilo
      real, intent(in) :: xnu10, xnu20, xalpha1, xalpha2, xbeta1, xbeta2
      real, intent(in) :: xmnu0
      logical, intent(in) :: lfeedbackt
      integer, intent(in)    :: ksize, kproma
      logical, dimension(kproma), intent(in)    :: ldcompute
      real, dimension(kproma), intent(in)    :: ptht    ! theta at t
      real, dimension(kproma), intent(in)    :: ppabst  ! absolute pressure at t
      real, dimension(kproma), intent(in)    :: prhodref! reference density
      real, dimension(kproma), intent(in)    :: pexn    ! exner function
      real, dimension(kproma), intent(in)    :: plsfact
      real, dimension(kproma), intent(in)    :: pt      ! temperature at time t
      real, dimension(kproma), intent(in)    :: prvt    ! water vapor m.r. at t
      real, dimension(kproma), intent(inout) :: pcit    ! pristine ice n.c. at t
      real, dimension(size(pt)), intent(out)   :: prvheni_mr ! mixing ratio change due to the heterogeneous nucleation
      !
      !*       0.2  declaration of local variables
      !
      real, dimension(kproma) :: zw ! work array
      logical, dimension(kproma) :: gnegt  ! test where to compute the hen process
      real, dimension(kproma)  :: zzw, & ! work array
                                 zusw, & ! undersaturation over water
                                 zssi        ! supersaturation over ice
      integer :: ji
      !-------------------------------------------------------------------------------
      !
      do ji = 1, ksize
         if (ldcompute(ji)) then
            gnegt(ji) = pt(ji) < xtt .and. prvt(ji) > v_rtmin
         else
            gnegt(ji) = .false.
         end if
      end do

      zusw(:) = 0.
      zzw(:) = 0.
      do ji = 1, ksize
         if (gnegt(ji)) then
            zzw(ji) = alog(pt(ji))
            zusw(ji) = exp(xalpw - xbetaw/pt(ji) - xgamw*zzw(ji))          ! es_w
            zzw(ji) = exp(xalpi - xbetai/pt(ji) - xgami*zzw(ji))           ! es_i
         end if
      end do

      zssi(:) = 0.
      do ji = 1, ksize
         if (gnegt(ji)) then
            zzw(ji) = min(ppabst(ji)/2., zzw(ji))             ! safety limitation
            zssi(ji) = prvt(ji)*(ppabst(ji) - zzw(ji))/(xepsilo*zzw(ji)) - 1.0
            ! supersaturation over ice
            zusw(ji) = min(ppabst(ji)/2., zusw(ji))            ! safety limitation
            zusw(ji) = (zusw(ji)/zzw(ji))*((ppabst(ji) - zzw(ji))/(ppabst(ji) - zusw(ji))) - 1.0
            ! supersaturation of saturated water vapor over ice
            !
            !*       3.1     compute the heterogeneous nucleation source rvheni
            !
            !*       3.1.1   compute the cloud ice concentration
            !
            zssi(ji) = min(zssi(ji), zusw(ji)) ! limitation of ssi according to ssw=0
         end if
      end do

      zzw(:) = 0.
      do ji = 1, ksize
         if (gnegt(ji)) then
            if (pt(ji) < xtt - 5.0 .and. zssi(ji) > 0.0) then
               zzw(ji) = xnu20*exp(xalpha2*zssi(ji) - xbeta2)
            elseif (pt(ji) <= xtt - 2.0 .and. pt(ji) >= xtt - 5.0 .and. zssi(ji) > 0.0) then
               zzw(ji) = max(xnu20*exp(-xbeta2), &
                             xnu10*exp(-xbeta1*(pt(ji) - xtt))*(zssi(ji)/zusw(ji))**xalpha1)
            end if
         end if
      end do
      do ji = 1, ksize
         if (gnegt(ji)) then
            zzw(ji) = zzw(ji) - pcit(ji)
            zzw(ji) = min(zzw(ji), 50.e3) ! limitation provisoire a 50 l^-1
         end if
      end do

      prvheni_mr(:) = 0.
      do ji = 1, ksize
         if (gnegt(ji)) then
            prvheni_mr(ji) = max(zzw(ji), 0.0)*xmnu0/prhodref(ji)
            prvheni_mr(ji) = min(prvt(ji), prvheni_mr(ji))
         end if
      end do
      if (lfeedbackt) then
         zw(:) = 0.
         do ji = 1, ksize
            if (gnegt(ji)) then
               zw(ji) = min(prvheni_mr(ji), &
                            max(0., (xtt/pexn(ji) - ptht(ji))/plsfact(ji)))/ &
                        max(prvheni_mr(ji), 1.e-20)
            end if
            prvheni_mr(ji) = prvheni_mr(ji)*zw(ji)
            zzw(ji) = zzw(ji)*zw(ji)
         end do
      end if
      do ji = 1, ksize
         if (gnegt(ji)) then
            pcit(ji) = max(zzw(ji) + pcit(ji), pcit(ji))
         end if
      end do
      !
   end subroutine ice4_nucleation
end module mode_ice4_nucleation
