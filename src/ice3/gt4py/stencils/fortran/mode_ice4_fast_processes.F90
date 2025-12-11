module mode_ice4_fast_processes

   implicit none
contains

   subroutine ice4_fast_rs(kproma, ksize, ldsoft, ldcompute, &
      &ngaminc, nacclbdas, nacclbdar, &
    &levlimit, lpack_interp, csnowriming, &
    &xcrimss, xexcrimss, xcrimsg, xexcrimsg, xexsrimcg2,&
    &xfraccss, &
       &s_rtmin, c_rtmin, r_rtmin, xepsilo, xalpi, xbetai, &
       &xgami, xtt, xlvtt, xcpv, xci, xcl, xlmtt, &
       &xestt, xrv, x0deps, x1deps, xex0deps, xex1deps, &
       &xlbraccs1, xlbraccs2, xlbraccs3, &
       &xcxs, xsrimcg2, xsrimcg3, xbs, &
       &xlbsaccr1, xlbsaccr2, xlbsaccr3, xfsaccrg, &
       &xsrimcg, xexsrimcg, xcexvt, &
       &xalpw, xbetaw, xgamw, xfscvmg, &
       &xker_raccss, xker_raccs,xker_saccrg, &
       &xgaminc_rim1, xgaminc_rim2, xgaminc_rim4, &
       &xrimintp1, xrimintp2, xaccintp1s, xaccintp2s, xaccintp1r, xaccintp2r, &
       &prhodref, ppres, &
       &pdv, pka, pcj, &
       &plbdar, plbdas, &
       &pt, prvt, prct, prrt, prst, &
       &priaggs, &
       &prcrimss, prcrimsg, prsrimcg, &
       &prraccss, prraccsg, prsaccrg, prsmltg, &
       &prcmltsr, &
       &prs_tend)
!
      implicit none
!
!*       0.1   declarations of dummy arguments :
!
      integer, intent(in)    :: kproma, ksize
      logical, intent(in) :: levlimit, lpack_interp
      logical, intent(in)    :: ldsoft
      integer, intent(in) :: ngaminc
      character(len=4), intent(in) :: csnowriming
      real, intent(in) :: s_rtmin, c_rtmin, r_rtmin, xepsilo, xalpi, xbetai
      real, intent(in) :: xgami, xtt, xlvtt, xcpv, xci, xcl, xlmtt
      real, intent(in) :: xestt, xrv, x0deps, x1deps, xex0deps, xex1deps
      real, intent(in) :: xcrimss, xexcrimss, xcrimsg, xexcrimsg
      real, intent(in) :: xsrimcg, xexsrimcg, xcexvt, xexsrimcg2
      real, intent(in) :: xfraccss, xlbraccs1, xlbraccs2, xlbraccs3
      real, intent(in) :: xcxs, xsrimcg2, xsrimcg3, xbs
      real, intent(in) :: xlbsaccr1, xlbsaccr2, xlbsaccr3, xfsaccrg
      real, intent(in) :: xalpw, xbetaw, xgamw, xfscvmg
      real, dimension(:,:), intent(in) :: xker_raccss, xker_raccs, xker_saccrg
      real, dimension(:) :: xgaminc_rim1, xgaminc_rim2, xgaminc_rim4
      real, intent(in) :: xrimintp1, xrimintp2
      integer, intent(in) :: nacclbdas, nacclbdar
      real, intent(in) :: xaccintp1s, xaccintp2s, xaccintp1r, xaccintp2r

      logical, dimension(kproma), intent(in)    :: ldcompute
      real, dimension(kproma), intent(in)    :: prhodref ! reference density
      real, dimension(kproma), intent(in)    :: ppres    ! absolute pressure at t
      real, dimension(kproma), intent(in)    :: pdv      ! diffusivity of water vapor in the air
      real, dimension(kproma), intent(in)    :: pka      ! thermal conductivity of the air
      real, dimension(kproma), intent(in)    :: pcj      ! function to compute the ventilation coefficient
      real, dimension(kproma), intent(in)    :: plbdar   ! slope parameter of the raindrop  distribution
      real, dimension(kproma), intent(in)    :: plbdas   ! slope parameter of the aggregate distribution
      real, dimension(kproma), intent(in)    :: pt       ! temperature
      real, dimension(kproma), intent(in)    :: prvt     ! water vapor m.r. at t
      real, dimension(kproma), intent(in)    :: prct     ! cloud water m.r. at t
      real, dimension(kproma), intent(in)    :: prrt     ! rain water m.r. at t
      real, dimension(kproma), intent(in)    :: prst     ! snow/aggregate m.r. at t
      real, dimension(kproma), intent(in)    :: priaggs  ! r_i aggregation on r_s
      real, dimension(kproma), intent(out)   :: prcrimss ! cloud droplet riming of the aggregates
      real, dimension(kproma), intent(out)   :: prcrimsg ! cloud droplet riming of the aggregates
      real, dimension(kproma), intent(out)   :: prsrimcg ! cloud droplet riming of the aggregates
      real, dimension(kproma), intent(out)   :: prraccss ! rain accretion onto the aggregates
      real, dimension(kproma), intent(out)   :: prraccsg ! rain accretion onto the aggregates
      real, dimension(kproma), intent(out)   :: prsaccrg ! rain accretion onto the aggregates
      real, dimension(kproma), intent(inout) :: prsmltg  ! conversion-melting of the aggregates
      real, dimension(kproma), intent(inout) :: prcmltsr ! cloud droplet collection onto aggregates by positive temperature
      real, dimension(kproma, 8), intent(inout) :: prs_tend ! individual tendencies
!
!*       0.2  declaration of local variables
!
      integer, parameter :: ircrims = 1, ircrimss = 2, irsrimcg = 3, irraccs = 4, irraccss = 5, irsaccrg = 6, &
       & ifreez1 = 7, ifreez2 = 8
      logical, dimension(kproma) :: grim, gacc
      integer :: igrim, igacc
      integer, dimension(kproma) :: ibuf1, ibuf2, ibuf3
      real, dimension(kproma) :: zbuf1, zbuf2, zbuf3
      real, dimension(kproma) :: zzw, zzw1, zzw2, zzw3, zfreez_rate
      integer :: jl
      real :: zzw0d
!
!*       5.0    maximum freezing rate
!
      do jl = 1, ksize
      if (prst(jl) > s_rtmin .and. ldcompute(jl)) then
      if (.not. ldsoft) then
         prs_tend(jl, ifreez1) = prvt(jl)*ppres(jl)/(xepsilo + prvt(jl)) ! vapor pressure
         if (levlimit) then
            prs_tend(jl, ifreez1) = min(prs_tend(jl, ifreez1), exp(xalpi - xbetai/pt(jl) - xgami*alog(pt(jl)))) ! min(ev, es_i(t))
         end if
         prs_tend(jl, ifreez1) = pka(jl)*(xtt - pt(jl)) +                              &
                 &(pdv(jl)*(xlvtt + (xcpv - xcl)*(pt(jl) - xtt)) &
                 &*(xestt - prs_tend(jl, ifreez1))/(xrv*pt(jl)))
         prs_tend(jl, ifreez1) = prs_tend(jl, ifreez1)*(x0deps*plbdas(jl)**xex0deps +     &
                 &                        x1deps*pcj(jl)*plbdas(jl)**xex1deps)/ &
                 &(prhodref(jl)*(xlmtt - xcl*(xtt - pt(jl))))

         prs_tend(jl, ifreez2) = (prhodref(jl)*(xlmtt + (xci - xcl)*(xtt - pt(jl))))/ &
                 &(prhodref(jl)*(xlmtt - xcl*(xtt - pt(jl))))
      end if

      zfreez_rate(jl) = max(0., max(0., prs_tend(jl, ifreez1) + &
                       &prs_tend(jl, ifreez2)*priaggs(jl)) - &
               priaggs(jl))
      else
      prs_tend(jl, ifreez1) = 0.
      prs_tend(jl, ifreez2) = 0.
      zfreez_rate(jl) = 0.
      end if
      end do
!
!*       5.1    cloud droplet riming of the aggregates
!
      do jl = 1, ksize
      if (prct(jl) > c_rtmin .and. prst(jl) > s_rtmin .and. ldcompute(jl)) then
         zzw(jl) = plbdas(jl)

         grim(jl) = .true.
      else
         grim(jl) = .false.
         prs_tend(jl, ircrims) = 0.
         prs_tend(jl, ircrimss) = 0.
         prs_tend(jl, irsrimcg) = 0.
      end if
      end do
!
! collection of cloud droplets by snow: this rate is used for riming (t<0) and for conversion/melting (t>0)
      if (.not. ldsoft) then
         call interp_micro_1d(kproma, ksize, zzw, ngaminc, xrimintp1, xrimintp2, &
                              lpack_interp, grim(:), ibuf1, ibuf2, zbuf1, zbuf2, &
                              igrim, &
                              xgaminc_rim1(:), zzw1(:), xgaminc_rim2(:), zzw2(:), xgaminc_rim4(:), zzw3(:))
         if (igrim > 0) then
!
!        5.1.4  riming of the small sized aggregates
!
!$mnh_expand_where(jl=1:ksize)
            where (grim(1:ksize))
               prs_tend(1:ksize, ircrimss) = xcrimss*zzw1(1:ksize)*prct(1:ksize) & ! rcrimss
                                             *plbdas(1:ksize)**xexcrimss &
                                             *prhodref(1:ksize)**(-xcexvt)

            end where
!$mnh_end_expand_where(jl=1:ksize)
!
!        5.1.6  riming-conversion of the large sized aggregates into graupeln
!
!
!$mnh_expand_where(jl=1:ksize)
            where (grim(1:ksize))
               prs_tend(1:ksize, ircrims) = xcrimsg*prct(1:ksize) & ! rcrims
                                            *plbdas(1:ksize)**xexcrimsg &
                                            *prhodref(1:ksize)**(-xcexvt)

            end where

            if (csnowriming == 'm90 ') then
!murakami 1990
               where (grim(1:ksize))
                  zzw(1:ksize) = prs_tend(1:ksize, ircrims) - prs_tend(1:ksize, ircrimss) ! rcrimsg
                  prs_tend(1:ksize, irsrimcg) = xsrimcg*plbdas(1:ksize)**xexsrimcg*(1.0 - zzw2(1:ksize))

                  prs_tend(1:ksize, irsrimcg) = zzw(1:ksize)*prs_tend(1:ksize, irsrimcg)/ &
                                                max(1.e-20, &
                                                    xsrimcg3*xsrimcg2*plbdas(1:ksize)**xexsrimcg2*(1.-zzw3(1:ksize)) - &
                                                    xsrimcg3*prs_tend(1:ksize, irsrimcg))

               end where
            else
               prs_tend(:, irsrimcg) = 0.
            end if
         end if
      end if
!
      do jl = 1, ksize
! more restrictive rim mask to be used for riming by negative temperature only
         if (grim(jl) .and. pt(jl) < xtt) then
            prcrimss(jl) = min(zfreez_rate(jl), prs_tend(jl, ircrimss))
            zfreez_rate(jl) = max(0., zfreez_rate(jl) - prcrimss(jl))
            zzw0d = min(1., zfreez_rate(jl)/max(1.e-20, prs_tend(jl, ircrims) - prcrimss(jl))) ! proportion we are able to freeze
            prcrimsg(jl) = zzw0d*max(0., prs_tend(jl, ircrims) - prcrimss(jl)) ! rcrimsg
            zfreez_rate(jl) = max(0., zfreez_rate(jl) - prcrimsg(jl))
            prsrimcg(jl) = zzw0d*prs_tend(jl, irsrimcg)

            prsrimcg(jl) = prsrimcg(jl)*max(0., -sign(1., -prcrimsg(jl)))
            prcrimsg(jl) = max(0., prcrimsg(jl))
         else
            prcrimss(jl) = 0.
            prcrimsg(jl) = 0.
            prsrimcg(jl) = 0.
         end if
      end do
!
!*       5.2    rain accretion onto the aggregates
!
      do jl = 1, ksize
      if (prrt(jl) > r_rtmin .and. prst(jl) > s_rtmin .and. ldcompute(jl)) then
         gacc(jl) = .true.
      else
         gacc(jl) = .false.
         prs_tend(jl, irraccs) = 0.
         prs_tend(jl, irraccss) = 0.
         prs_tend(jl, irsaccrg) = 0.
      end if
      end do
      if (.not. ldsoft) then
         prs_tend(:, irraccs) = 0.
         prs_tend(:, irraccss) = 0.
         prs_tend(:, irsaccrg) = 0.
         call interp_micro_2d(kproma, ksize, plbdas, plbdar, nacclbdas, nacclbdar, &
            &xaccintp1s, xaccintp2s, xaccintp1r, xaccintp2r,&
            &lpack_interp, gacc(:), ibuf1(:), ibuf2(:), ibuf3(:), zbuf1(:), zbuf2(:), zbuf3(:), &
            &igacc, &
            &xker_raccss(:, :), zzw1(:), xker_raccs(:, :), zzw2(:), xker_saccrg(:, :), zzw3(:))
         if (igacc > 0) then
!        5.2.4  raindrop accretion on the small sized aggregates
!
!$mnh_expand_where(jl=1:ksize)
            where (gacc(1:ksize))
               zzw(1:ksize) = & !! coef of rraccs
                  xfraccss*(plbdas(1:ksize)**xcxs)*(prhodref(1:ksize)**(-xcexvt - 1.)) &
                  *(xlbraccs1/((plbdas(1:ksize)**2)) + &
                    xlbraccs2/(plbdas(1:ksize)*plbdar(1:ksize)) + &
                    xlbraccs3/((plbdar(1:ksize)**2)))/plbdar(1:ksize)**4

               prs_tend(1:ksize, irraccss) = zzw1(1:ksize)*zzw(1:ksize)
            end where
!$mnh_end_expand_where(jl=1:ksize)
!
!$mnh_expand_where(jl=1:ksize)
            where (gacc(1:ksize))
               prs_tend(1:ksize, irraccs) = zzw2(1:ksize)*zzw(1:ksize)
            end where
!$mnh_end_expand_where(jl=1:ksize)
!
!        5.2.6  raindrop accretion-conversion of the large sized aggregates
!               into graupeln
!
!$mnh_expand_where(jl=1:ksize)
            where (gacc(1:ksize))
               prs_tend(1:ksize, irsaccrg) = xfsaccrg*zzw3(1:ksize)* & ! rsaccrg
                                             (plbdas(1:ksize)**(xcxs - xbs))*(prhodref(1:ksize)**(-xcexvt - 1.)) &
                                             *(xlbsaccr1/((plbdar(1:ksize)**2)) + &
                                               xlbsaccr2/(plbdar(1:ksize)*plbdas(1:ksize)) + &
                                               xlbsaccr3/((plbdas(1:ksize)**2)))/plbdar(1:ksize)

            end where
!$mnh_end_expand_where(jl=1:ksize)
         end if
      end if
!
      do jl = 1, ksize
! more restrictive acc mask to be used for accretion by negative temperature only
         if (gacc(jl) .and. pt(jl) < xtt) then
            prraccss(jl) = min(zfreez_rate(jl), prs_tend(jl, irraccss))
            zfreez_rate(jl) = max(0., zfreez_rate(jl) - prraccss(jl))
            zzw(jl) = min(1., zfreez_rate(jl)/max(1.e-20, prs_tend(jl, irraccs) - prraccss(jl))) ! proportion we are able to freeze
            prraccsg(jl) = zzw(jl)*max(0., prs_tend(jl, irraccs) - prraccss(jl))
            zfreez_rate(jl) = max(0., zfreez_rate(jl) - prraccsg(jl))
            prsaccrg(jl) = zzw(jl)*prs_tend(jl, irsaccrg)

            prsaccrg(jl) = prsaccrg(jl)*max(0., -sign(1., -prraccsg(jl)))
            prraccsg(jl) = max(0., prraccsg(jl))
         else
            prraccss(jl) = 0.
            prraccsg(jl) = 0.
            prsaccrg(jl) = 0.
         end if
      end do
!
!
!*       5.3    conversion-melting of the aggregates
!
      do jl = 1, ksize
      if (prst(jl) > s_rtmin .and. pt(jl) > xtt .and. ldcompute(jl)) then
      if (.not. ldsoft) then
         prsmltg(jl) = prvt(jl)*ppres(jl)/(xepsilo + prvt(jl)) ! vapor pressure
         if (levlimit) then
            prsmltg(jl) = min(prsmltg(jl), exp(xalpw - xbetaw/pt(jl) - xgamw*alog(pt(jl)))) ! min(ev, es_w(t))
         end if
         prsmltg(jl) = pka(jl)*(xtt - pt(jl)) +                                 &
         &(pdv(jl)*(xlvtt + (xcpv - xcl)*(pt(jl) - xtt)) &
         & *(xestt - prsmltg(jl))/(xrv*pt(jl)))
!
! compute rsmlt
!
         prsmltg(jl) = xfscvmg*max(0., (-prsmltg(jl)* &
                                        (x0deps*plbdas(jl)**xex0deps + &
                                         x1deps*pcj(jl)*plbdas(jl)**xex1deps) &
                                        - (prs_tend(jl, ircrims) + prs_tend(jl, irraccs))* &
                                        (prhodref(jl)*xcl*(xtt - pt(jl))) &
                                        )/(prhodref(jl)*xlmtt))
!

         prcmltsr(jl) = prs_tend(jl, ircrims) ! both species are liquid, no heat is exchanged
      end if
      else
      prsmltg(jl) = 0.
      prcmltsr(jl) = 0.
      end if
      end do
!
   contains
!
      include "interp_micro.func.h"
!
   end subroutine ice4_fast_rs

   subroutine ice4_fast_rg(kproma, ksize, krr,&
      &ldsoft, lcrflimit, levlimit, lnullwetg, lwetgpost, lpack_interp, ldcompute,& 
      &ndrylbdag, ndrylbdas, ndrylbdar, &
      &c_rtmin, i_rtmin, r_rtmin, g_rtmin, s_rtmin, &
      &xalpw, xbetaw, xgamw, xexrcfri, xdg, xepsilo, &
      &xicfrr, xexicfrr, xcexvt, xrcfri, xtt, xci, xcl, xlvtt, &
      &xcpv, xestt, x0depg, x1depg, xrv, xlmtt, &
      &xcxg, xfcdryg, xfidryg, xcolig, xcolexig, &
      &xlbsdryg1, xlbsdryg2, xlbsdryg3, xcolexsg, &
      &xfsdryg, xcolsg, xcxs, xbs, &
      &xfrdryg, xlbrdryg1, xlbrdryg2, xlbrdryg3, &
      &xex0depg, xex1depg, xalpi, xbetai, xgami, &
      &xdryintp1g, xdryintp2g, xdryintp1s, xdryintp2s, &
      &xdryintp1r, xdryintp2r, &
      &xker_sdryg, xker_rdryg, &
      &prhodref, ppres, &
      &pdv, pka, pcj, pcit, &
      &plbdar, plbdas, plbdag, &
      &pt, prvt, prct, prrt, prit, prst, prgt, &
      &prgsi, prgsi_mr, &
      &ldwetg, &
      &pricfrrg, prrcfrig, pricfrr, prcwetg, priwetg, prrwetg, prswetg, &
      &prcdryg, pridryg, prrdryg, prsdryg, prwetgh, prwetgh_mr, prgmltr, &
      &prg_tend)
!
!*      0. declarations
!          ------------
!
      implicit none
!
!*       0.1   declarations of dummy arguments :
!
      integer, intent(in) :: kproma, ksize
      logical, intent(in) :: ldsoft
      logical, intent(in) :: lcrflimit
      logical, intent(in) :: levlimit, lnullwetg, lwetgpost
      logical, dimension(kproma), intent(in)    :: ldcompute
      integer, intent(in)    :: krr      ! number of moist variable
      real, intent(in) :: c_rtmin, i_rtmin, r_rtmin, g_rtmin, s_rtmin
      real, intent(in) :: xfidryg, xcolig, xcolexig, xepsilo
      real, intent(in) :: xalpw, xbetaw, xgamw
      real, intent(in) :: xicfrr, xexicfrr, xcexvt, xrcfri, xtt, xci, xcl
      real, intent(in) :: xexrcfri, xlvtt, xdg, xcxg, xfcdryg
      real, intent(in) :: xfsdryg, xcolsg, xcxs, xbs
      real, intent(in) :: xlbsdryg1, xlbsdryg2, xlbsdryg3, xcolexsg
      real, intent(in) :: xcpv, xestt, x0depg, x1depg, xrv, xlmtt
      real, intent(in) :: xfrdryg, xlbrdryg1, xlbrdryg2, xlbrdryg3
      real, intent(in) :: xex0depg, xex1depg, xalpi, xbetai, xgami

      integer, intent(in) :: ndrylbdag, ndrylbdas, ndrylbdar
      real, intent(in) :: xdryintp1g, xdryintp2g, xdryintp1s, xdryintp2s
      real, intent(in) :: xdryintp1r, xdryintp2r
      logical, intent(in) :: lpack_interp

      real, dimension(:,:), intent(in) :: xker_sdryg, xker_rdryg

      real, dimension(kproma), intent(in)    :: prhodref ! reference density
      real, dimension(kproma), intent(in)    :: ppres    ! absolute pressure at t
      real, dimension(kproma), intent(in)    :: pdv      ! diffusivity of water vapor in the air
      real, dimension(kproma), intent(in)    :: pka      ! thermal conductivity of the air
      real, dimension(kproma), intent(in)    :: pcj      ! function to compute the ventilation coefficient
      real, dimension(kproma), intent(in)    :: pcit     ! pristine ice conc. at t
      real, dimension(kproma), intent(in)    :: plbdar   ! slope parameter of the raindrop  distribution
      real, dimension(kproma), intent(in)    :: plbdas   ! slope parameter of the aggregate distribution
      real, dimension(kproma), intent(in)    :: plbdag   ! slope parameter of the graupel   distribution
      real, dimension(kproma), intent(in)    :: pt       ! temperature
      real, dimension(kproma), intent(in)    :: prvt     ! water vapor m.r. at t
      real, dimension(kproma), intent(in)    :: prct     ! cloud water m.r. at t
      real, dimension(kproma), intent(in)    :: prrt     ! rain water m.r. at t
      real, dimension(kproma), intent(in)    :: prit     ! pristine ice m.r. at t
      real, dimension(kproma), intent(in)    :: prst     ! snow/aggregate m.r. at t
      real, dimension(kproma), intent(in)    :: prgt     ! graupel m.r. at t
      real, dimension(kproma), intent(in)    :: prgsi    ! graupel tendency by other processes
      real, dimension(kproma), intent(in)    :: prgsi_mr ! graupel mr change by other processes
      logical, dimension(kproma), intent(out)   :: ldwetg   ! .true. where graupel grows in wet mode
      real, dimension(kproma), intent(inout) :: pricfrrg ! rain contact freezing
      real, dimension(kproma), intent(inout) :: prrcfrig ! rain contact freezing
      real, dimension(kproma), intent(inout) :: pricfrr  ! rain contact freezing
      real, dimension(kproma), intent(out)   :: prcwetg  ! graupel wet growth
      real, dimension(kproma), intent(out)   :: priwetg  ! graupel wet growth
      real, dimension(kproma), intent(out)   :: prrwetg  ! graupel wet growth
      real, dimension(kproma), intent(out)   :: prswetg  ! graupel wet growth
      real, dimension(kproma), intent(out)   :: prcdryg  ! graupel dry growth
      real, dimension(kproma), intent(out)   :: pridryg  ! graupel dry growth
      real, dimension(kproma), intent(out)   :: prrdryg  ! graupel dry growth
      real, dimension(kproma), intent(out)   :: prsdryg  ! graupel dry growth
      real, dimension(kproma), intent(out)   :: prwetgh  ! conversion of graupel into hail
      real, dimension(kproma), intent(out)   :: prwetgh_mr ! conversion of graupel into hail, mr change
      real, dimension(kproma), intent(inout) :: prgmltr  ! melting of the graupel
      real, dimension(kproma, 8), intent(inout) :: prg_tend ! individual tendencies
!
!*       0.2  declaration of local variables
!
      integer, parameter :: ircdryg = 1, iridryg = 2, iriwetg = 3, irsdryg = 4, irswetg = 5, irrdryg = 6, &
       & ifreez1 = 7, ifreez2 = 8
      logical, dimension(kproma) :: gdry, lldryg
      integer :: igdry
      real, dimension(kproma) :: zbuf1, zbuf2, zbuf3
      integer, dimension(kproma) :: ibuf1, ibuf2, ibuf3
      real, dimension(kproma) :: zzw, &
                                 zrdryg_init, & !initial dry growth rate of the graupeln
                                 zrwetg_init !initial wet growth rate of the graupeln
      real :: zzw0d
      integer :: jl
!-------------------------------------------------------------------------------
!
!*       6.1    rain contact freezing
!
      do jl = 1, ksize
      if (prit(jl) > i_rtmin .and. prrt(jl) > r_rtmin .and. ldcompute(jl)) then
      if (.not. ldsoft) then
         pricfrrg(jl) = xicfrr*prit(jl) & ! ricfrrg
                        *plbdar(jl)**xexicfrr &
                        *prhodref(jl)**(-xcexvt)
         prrcfrig(jl) = xrcfri*pcit(jl) & ! rrcfrig
                        *plbdar(jl)**xexrcfri &
                        *prhodref(jl)**(-xcexvt - 1.)
         if (lcrflimit) then
!comparison between heat to be released (to freeze rain) and heat sink (rain and ice temperature change)
!zzw0d is the proportion of process that can take place
            zzw0d = max(0., min(1., (pricfrrg(jl)*xci + prrcfrig(jl)*xcl)*(xtt - pt(jl))/ &
                                max(1.e-20, xlvtt*prrcfrig(jl))))
            prrcfrig(jl) = zzw0d*prrcfrig(jl) !part of rain that can be freezed
            pricfrr(jl) = (1.-zzw0d)*pricfrrg(jl) !part of collected pristine ice converted to rain
            pricfrrg(jl) = zzw0d*pricfrrg(jl) !part of collected pristine ice that lead to graupel
         else
            pricfrr(jl) = 0.
         end if
      end if
      else
      pricfrrg(jl) = 0.
      prrcfrig(jl) = 0.
      pricfrr(jl) = 0.
      end if
      end do
!
!
!*       6.3    compute the graupel growth
!
! wet and dry collection of rc and ri on graupel
      do jl = 1, ksize
      if (prgt(jl) > g_rtmin .and. prct(jl) > c_rtmin .and. ldcompute(jl)) then
      if (.not. ldsoft) then
         prg_tend(jl, ircdryg) = plbdag(jl)**(xcxg - xdg - 2.)*prhodref(jl)**(-xcexvt)
         prg_tend(jl, ircdryg) = xfcdryg*prct(jl)*prg_tend(jl, ircdryg)
      end if
      else
      prg_tend(jl, ircdryg) = 0.
      end if

      if (prgt(jl) > g_rtmin .and. prit(jl) > i_rtmin .and. ldcompute(jl)) then
      if (.not. ldsoft) then
         prg_tend(jl, iridryg) = plbdag(jl)**(xcxg - xdg - 2.)*prhodref(jl)**(-xcexvt)
         prg_tend(jl, iridryg) = xfidryg*exp(xcolexig*(pt(jl) - xtt))*prit(jl)*prg_tend(jl, iridryg)
         prg_tend(jl, iriwetg) = prg_tend(jl, iridryg)/(xcolig*exp(xcolexig*(pt(jl) - xtt)))
      end if
      else
      prg_tend(jl, iridryg) = 0.
      prg_tend(jl, iriwetg) = 0.
      end if
      end do

! wet and dry collection of rs on graupel (6.2.1)
      do jl = 1, ksize
      if (prst(jl) > s_rtmin .and. prgt(jl) > g_rtmin .and. ldcompute(jl)) then
         gdry(jl) = .true.
      else
         gdry(jl) = .false.
         prg_tend(jl, irsdryg) = 0.
         prg_tend(jl, irswetg) = 0.
      end if
      end do

      if (.not. ldsoft) then
         call interp_micro_2d(kproma, ksize, plbdag(:), plbdas(:), ndrylbdag, ndrylbdas, &
             &xdryintp1g, xdryintp2g, xdryintp1s, xdryintp2s, &
             &lpack_interp, gdry(:), ibuf1(:), ibuf2(:), ibuf3(:), zbuf1(:), zbuf2(:), zbuf3(:), &
             &igdry, &
             &xker_sdryg(:, :), zzw(:))
         if (igdry > 0) then
            where (gdry(1:ksize))
               prg_tend(1:ksize, irswetg) = xfsdryg*zzw(1:ksize) & ! rsdryg
                                            /xcolsg &
                                            *(plbdas(1:ksize)**(xcxs - xbs))*(plbdag(1:ksize)**xcxg) &
                                            *(prhodref(1:ksize)**(-xcexvt - 1.)) &
                                            *(xlbsdryg1/(plbdag(1:ksize)**2) + &
                                              xlbsdryg2/(plbdag(1:ksize)*plbdas(1:ksize)) + &
                                              xlbsdryg3/(plbdas(1:ksize)**2))

               prg_tend(1:ksize, irsdryg) = prg_tend(1:ksize, irswetg)*xcolsg*exp(xcolexsg*(pt(1:ksize) - xtt))
            end where
         end if
      end if
!
!*       6.2.6  accretion of raindrops on the graupeln
!
      do jl = 1, ksize
      if (prrt(jl) > r_rtmin .and. prgt(jl) > g_rtmin .and. ldcompute(jl)) then
         gdry(jl) = .true.
      else
         gdry(jl) = .false.
         prg_tend(jl, irrdryg) = 0.
      end if
      end do
      if (.not. ldsoft) then
!
         call interp_micro_2d(kproma, ksize, plbdag(:), plbdar(:), ndrylbdag, ndrylbdar, &
             &xdryintp1g, xdryintp2g, xdryintp1r, xdryintp2r, &
             &lpack_interp, gdry(:), ibuf1(:), ibuf2(:), ibuf3(:), zbuf1(:), zbuf2(:), zbuf3(:), &
             &igdry, &
             &xker_rdryg(:, :), zzw(:))
         if (igdry > 0) then
            where (gdry(1:ksize))
               prg_tend(1:ksize, irrdryg) = xfrdryg*zzw(1:ksize) & ! rrdryg
                                            *(plbdar(1:ksize)**(-4))*(plbdag(1:ksize)**xcxg) &
                                            *(prhodref(1:ksize)**(-xcexvt - 1.)) &
                                            *(xlbrdryg1/(plbdag(1:ksize)**2) + &
                                              xlbrdryg2/(plbdag(1:ksize)*plbdar(1:ksize)) + &
                                              xlbrdryg3/(plbdar(1:ksize)**2))
            end where
         end if
      end if

      do jl = 1, ksize
         zrdryg_init(jl) = prg_tend(jl, ircdryg) + prg_tend(jl, iridryg) + &
         &prg_tend(jl, irsdryg) + prg_tend(jl, irrdryg)
      end do

      do jl = 1, ksize
      if (prgt(jl) > g_rtmin .and. ldcompute(jl)) then
         if (.not. ldsoft) then
            prg_tend(jl, ifreez1) = prvt(jl)*ppres(jl)/(xepsilo + prvt(jl)) ! vapor pressure
            if (levlimit) then
               prg_tend(jl, ifreez1) = min(prg_tend(jl, ifreez1), exp(xalpi - xbetai/pt(jl) - xgami*alog(pt(jl)))) ! min(ev, es_i(t))
            end if
            prg_tend(jl, ifreez1) = pka(jl)*(xtt - pt(jl)) + &
                                    (pdv(jl)*(xlvtt + (xcpv - xcl)*(pt(jl) - xtt)) &
                                     *(xestt - prg_tend(jl, ifreez1))/(xrv*pt(jl)))
            prg_tend(jl, ifreez1) = prg_tend(jl, ifreez1)*(x0depg*plbdag(jl)**xex0depg + &
                                                           x1depg*pcj(jl)*plbdag(jl)**xex1depg)/ &
                                    (prhodref(jl)*(xlmtt - xcl*(xtt - pt(jl))))
            prg_tend(jl, ifreez2) = (prhodref(jl)*(xlmtt + (xci - xcl)*(xtt - pt(jl))))/ &
                                    (prhodref(jl)*(xlmtt - xcl*(xtt - pt(jl))))
         end if
         zrwetg_init(jl) = max(prg_tend(jl, iriwetg) + prg_tend(jl, irswetg), &
             &max(0., prg_tend(jl, ifreez1) + &
             &        prg_tend(jl, ifreez2)*(prg_tend(jl, iriwetg) + prg_tend(jl, irswetg))))

         ldwetg(jl) = max(0., zrwetg_init(jl) - prg_tend(jl, iriwetg) - prg_tend(jl, irswetg)) <= &
         &max(0., zrdryg_init(jl) - prg_tend(jl, iridryg) - prg_tend(jl, irsdryg))

         if (lnullwetg) then
            ldwetg(jl) = ldwetg(jl) .and. zrdryg_init(jl) > 0.
         else
            ldwetg(jl) = ldwetg(jl) .and. zrwetg_init(jl) > 0.
         end if
         if (.not. lwetgpost) then
            ldwetg(jl) = ldwetg(jl) .and. pt(jl) < xtt
         end if

         lldryg(jl) = pt(jl) < xtt .and. zrdryg_init(jl) > 1.e-20 .and. &
         &max(0., zrwetg_init(jl) - prg_tend(jl, iriwetg) - prg_tend(jl, irswetg)) >&
         &max(0., zrdryg_init(jl) - prg_tend(jl, iridryg) - prg_tend(jl, irsdryg))
      else
         prg_tend(jl, ifreez1) = 0.
         prg_tend(jl, ifreez2) = 0.
         zrwetg_init(jl) = 0.
         ldwetg(jl) = .false.
         lldryg(jl) = .false.
      end if
      end do

      if (krr == 7) then
         where (ldwetg(1:ksize))
            prwetgh(1:ksize) = (max(0., prgsi(1:ksize) + pricfrrg(1:ksize) + prrcfrig(1:ksize)) + zrwetg_init(1:ksize))*&
             &zrdryg_init(1:ksize)/(zrwetg_init(1:ksize) + zrdryg_init(1:ksize))
            prwetgh_mr(1:ksize) = max(0., prgsi_mr(1:ksize))*zrdryg_init(1:ksize)/(zrwetg_init(1:ksize) + zrdryg_init(1:ksize))
         elsewhere
            prwetgh(1:ksize) = 0.
            prwetgh_mr(1:ksize) = 0.
         end where
      else
         prwetgh(:) = 0.
         prwetgh_mr(:) = 0.
      end if

      do jl = 1, ksize
!aggregated minus collected
         if (ldwetg(jl)) then
            prrwetg(jl) = -(prg_tend(jl, iriwetg) + prg_tend(jl, irswetg) +&
            &prg_tend(jl, ircdryg) - zrwetg_init(jl))
            prcwetg(jl) = prg_tend(jl, ircdryg)
            priwetg(jl) = prg_tend(jl, iriwetg)
            prswetg(jl) = prg_tend(jl, irswetg)
         else
            prrwetg(jl) = 0.
            prcwetg(jl) = 0.
            priwetg(jl) = 0.
            prswetg(jl) = 0.
         end if

         if (lldryg(jl)) then
            prcdryg(jl) = prg_tend(jl, ircdryg)
            prrdryg(jl) = prg_tend(jl, irrdryg)
            pridryg(jl) = prg_tend(jl, iridryg)
            prsdryg(jl) = prg_tend(jl, irsdryg)
         else
            prcdryg(jl) = 0.
            prrdryg(jl) = 0.
            pridryg(jl) = 0.
            prsdryg(jl) = 0.
         end if
      end do
!
!*       6.5    melting of the graupeln
!
      do jl = 1, ksize
      if (prgt(jl) > g_rtmin .and. pt(jl) > xtt .and. ldcompute(jl)) then
      if (.not. ldsoft) then
         prgmltr(jl) = prvt(jl)*ppres(jl)/(xepsilo + prvt(jl)) ! vapor pressure
         if (levlimit) then
            prgmltr(jl) = min(prgmltr(jl), exp(xalpw - xbetaw/pt(jl) - xgamw*alog(pt(jl)))) ! min(ev, es_w(t))
         end if
         prgmltr(jl) = pka(jl)*(xtt - pt(jl)) + &
                       pdv(jl)*(xlvtt + (xcpv - xcl)*(pt(jl) - xtt)) &
                       *(xestt - prgmltr(jl))/(xrv*pt(jl))
         prgmltr(jl) = max(0., (-prgmltr(jl)* &
                                (x0depg*plbdag(jl)**xex0depg + &
                                 x1depg*pcj(jl)*plbdag(jl)**xex1depg) - &
                                (prg_tend(jl, ircdryg) + prg_tend(jl, irrdryg))* &
                                (prhodref(jl)*xcl*(xtt - pt(jl))))/ &
                           (prhodref(jl)*xlmtt))
      end if
      else
      prgmltr(jl) = 0.
      end if
      end do
!
   end subroutine ice4_fast_rg

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!! Interpolation routines !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   subroutine interp_micro_1d(kproma, ksize, pin, knum, p1, p2, &
                           ldpack, ldmask, kbuf1, kbuf2, pbuf1, pbuf2, &
                           klen, &
                           plt1, pout1, plt2, pout2, plt3, pout3)

implicit none

integer,                    intent(in)  :: kproma       !array size
integer,                    intent(in)  :: ksize        !last usefull array index
real,    dimension(kproma), intent(in)  :: pin          !input array
integer,                    intent(in)  :: knum         !number of points in the look-up table
real,                       intent(in)  :: p1           !scaling factor
real,                       intent(in)  :: p2           !scaling factor
logical,                    intent(in)  :: ldpack       !.true. to perform packing
logical, dimension(kproma), intent(in)  :: ldmask       !computation mask
integer, dimension(kproma), intent(out) :: kbuf1, kbuf2 !buffer arrays
real,    dimension(kproma), intent(out) :: pbuf1, pbuf2 !buffer arrays
integer,                    intent(out) :: klen         !number of active points
real,    dimension(knum),   intent(in)            :: plt1  !look-up table
real,    dimension(kproma), intent(out)           :: pout1 !interpolated values
real,    dimension(knum),   intent(in) , optional :: plt2
real,    dimension(kproma), intent(out), optional :: pout2
real,    dimension(knum),   intent(in) , optional :: plt3
real,    dimension(kproma), intent(out), optional :: pout3

integer :: jl
integer :: iindex
real :: zindex

if (ldpack) then

  !pack input array
  klen=0
  do jl=1, ksize
    if (ldmask(jl)) then
      klen=klen+1
      pbuf1(klen)=pin(jl)
      kbuf1(klen)=jl
    endif
  enddo

  if (klen>0) then
    !index computation
    !$mnh_expand_array(jl=1:klen)
    pbuf1(1:klen) = max(1.00001, min(real(knum)-0.00001, p1 * log(pbuf1(1:klen)) + p2))
    kbuf2(1:klen) = int(pbuf1(1:klen))
    pbuf1(1:klen) = pbuf1(1:klen) - real(kbuf2(1:klen))
    !$mnh_end_expand_array(jl=1:klen)

    !interpolation and unpack
    !$mnh_expand_array(jl=1:klen)
    pbuf2(1:klen) = plt1(kbuf2(1:klen)+1) *  pbuf1(1:klen)       &
                  &-plt1(kbuf2(1:klen)  ) * (pbuf1(1:klen) - 1.0)
    !$mnh_end_expand_array(jl=1:klen)
    pout1(:)=0.
    do jl=1, klen
      pout1(kbuf1(jl))=pbuf2(jl)
    enddo

    !interpolation and unpack 2
    if(present(plt2)) then
      !$mnh_expand_array(jl=1:klen)
      pbuf2(1:klen) = plt2(kbuf2(1:klen)+1) *  pbuf1(1:klen)       &
                    &-plt2(kbuf2(1:klen)  ) * (pbuf1(1:klen) - 1.0)
      !$mnh_end_expand_array(jl=1:klen)
      pout2(:)=0.
      do jl=1, klen
        pout2(kbuf1(jl))=pbuf2(jl)
      enddo
    endif

    !interpolation and unpack 3
    if(present(plt3)) then
      !$mnh_expand_array(jl=1:klen)
      pbuf2(1:klen) = plt3(kbuf2(1:klen)+1) *  pbuf1(1:klen)       &
                    &-plt3(kbuf2(1:klen)  ) * (pbuf1(1:klen) - 1.0)
      !$mnh_end_expand_array(jl=1:klen)
      pout3(:)=0.
      do jl=1, klen
        pout3(kbuf1(jl))=pbuf2(jl)
      enddo
    endif

  endif

else

  klen=0
  do jl=1, ksize
    if (ldmask(jl)) then
      klen=klen+1

      !index computation
      zindex = max(1.00001, min(real(knum)-0.00001, p1 * log(pin(jl)) + p2))
      iindex = int(zindex)
      zindex = zindex - real(iindex)

      !interpolations
      pout1(jl) = plt1(iindex+1) *  zindex       &
                &-plt1(iindex  ) * (zindex - 1.0)

      if(present(plt2)) then
        pout2(jl) = plt2(iindex+1) *  zindex       &
                  &-plt2(iindex  ) * (zindex - 1.0)
      endif

      if(present(plt3)) then
        pout3(jl) = plt3(iindex+1) *  zindex       &
                  &-plt3(iindex  ) * (zindex - 1.0)
      endif

    else
      pout1(jl) = 0.
      if(present(plt2)) pout2(jl) = 0.
      if(present(plt3)) pout3(jl) = 0.
    endif
  enddo

endif
end subroutine interp_micro_1d

subroutine interp_micro_2d(kproma, ksize, pin1, pin2, knum1, knum2, p11, p12, p21, p22,&
                           ldpack, ldmask, kbuf1, kbuf2, kbuf3, pbuf1, pbuf2, pbuf3, &
                           klen, &
                           plt1, pout1, plt2, pout2, plt3, pout3)

implicit none

integer,                    intent(in)  :: kproma       !array size
integer,                    intent(in)  :: ksize        !last usefull array index
real,    dimension(kproma), intent(in)  :: pin1                !input array
real,    dimension(kproma), intent(in)  :: pin2                !input array
integer,                    intent(in)  :: knum1               !first dimension of the look-up table
integer,                    intent(in)  :: knum2               !second dimension of the look-up table
real,                       intent(in)  :: p11                 !scaling factor
real,                       intent(in)  :: p12                 !scaling factor
real,                       intent(in)  :: p21                 !scaling factor
real,                       intent(in)  :: p22                 !scaling factor
logical,                    intent(in)  :: ldpack              !.true. to perform packing
logical, dimension(kproma), intent(in)  :: ldmask              !computation mask
integer, dimension(kproma), intent(out) :: kbuf1, kbuf2, kbuf3 !buffer arrays
real,    dimension(kproma), intent(out) :: pbuf1, pbuf2, pbuf3 !buffer arrays
integer,                    intent(out) :: klen                !number of active points
real,    dimension(knum1, knum2),   intent(in)            :: plt1  !look-up table
real,    dimension(kproma),         intent(out)           :: pout1 !interpolated values from the first look-up table
real,    dimension(knum1, knum2),   intent(in) , optional :: plt2  !other look-up table
real,    dimension(kproma),         intent(out), optional :: pout2 !interpolated values from the second look-up table
real,    dimension(knum2, knum1),   intent(in) , optional :: plt3  !another look-up table **caution, table is reversed**
real,    dimension(kproma),         intent(out), optional :: pout3 !interpolated values from the third look-up table

integer :: jl
integer :: iindex1, iindex2
real :: zindex1, zindex2

if (ldpack) then

  !pack input array
  klen=0
  do jl=1, ksize
    if (ldmask(jl)) then
      klen=klen+1
      pbuf1(klen)=pin1(jl)
      pbuf2(klen)=pin2(jl)
      kbuf3(klen)=jl
    endif
  enddo

  if (klen>0) then
    !index computation
    !$mnh_expand_array(jl=1:klen)
    pbuf1(1:klen) = max(1.00001, min(real(knum1)-0.00001, p11 * log(pbuf1(1:klen)) + p12))
    kbuf1(1:klen) = int(pbuf1(1:klen))
    pbuf1(1:klen) = pbuf1(1:klen) - real(kbuf1(1:klen))

    pbuf2(1:klen) = max(1.00001, min(real(knum2)-0.00001, p21 * log(pbuf2(1:klen)) + p22))
    kbuf2(1:klen) = int(pbuf2(1:klen))
    pbuf2(1:klen) = pbuf2(1:klen) - real(kbuf2(1:klen))
    !$mnh_end_expand_array(jl=1:klen)

    !interpolation and unpack 1
    do jl=1, klen
      pbuf3(jl) = ( plt1(kbuf1(jl)+1,kbuf2(jl)+1)* pbuf2(jl)         &
                   -plt1(kbuf1(jl)+1,kbuf2(jl)  )*(pbuf2(jl) - 1.0)) *  pbuf1(jl) &
                 -( plt1(kbuf1(jl)  ,kbuf2(jl)+1)* pbuf2(jl)         &
                   -plt1(kbuf1(jl)  ,kbuf2(jl)  )*(pbuf2(jl) - 1.0)) * (pbuf1(jl) - 1.0)
    enddo
    pout1(:)=0.
    do jl=1, klen
      pout1(kbuf3(jl))=pbuf3(jl)
    enddo

    !interpolation and unpack 2
    if(present(plt2)) then
      do jl=1, klen
        pbuf3(jl) = ( plt2(kbuf1(jl)+1,kbuf2(jl)+1)* pbuf2(jl)         &
                     -plt2(kbuf1(jl)+1,kbuf2(jl)  )*(pbuf2(jl) - 1.0)) *  pbuf1(jl) &
                   -( plt2(kbuf1(jl)  ,kbuf2(jl)+1)* pbuf2(jl)         &
                     -plt2(kbuf1(jl)  ,kbuf2(jl)  )*(pbuf2(jl) - 1.0)) * (pbuf1(jl) - 1.0)
      enddo
      pout2(:)=0.
      do jl=1, klen
        pout2(kbuf3(jl))=pbuf3(jl)
      enddo
    endif

    !interpolation and unpack 3
    if(present(plt3)) then
      do jl=1, klen
        pbuf3(jl) = ( plt3(kbuf2(jl)+1,kbuf1(jl)+1)* pbuf1(jl)         &
                     -plt3(kbuf2(jl)+1,kbuf1(jl)  )*(pbuf1(jl) - 1.0)) *  pbuf2(jl) &
                   -( plt3(kbuf2(jl)  ,kbuf1(jl)+1)* pbuf1(jl)         &
                     -plt3(kbuf2(jl)  ,kbuf1(jl)  )*(pbuf1(jl) - 1.0)) * (pbuf2(jl) - 1.0)
      enddo
      pout3(:)=0.
      do jl=1, klen
        pout3(kbuf3(jl))=pbuf3(jl)
      enddo
    endif
  endif

else

  klen=0
  do jl=1, ksize
    if (ldmask(jl)) then
      klen=klen+1

      !indexes computation
      zindex1 = max(1.00001, min(real(knum1)-0.00001, p11 * log(pin1(jl)) + p12))
      iindex1 = int(zindex1)
      zindex1 = zindex1 - real(iindex1)
  
      zindex2 = max(1.00001, min(real(knum1)-0.00001, p21 * log(pin2(jl)) + p22))
      iindex2 = int(zindex2)
      zindex2 = zindex2 - real(iindex2)
  
      !interpolations
      pout1(jl) = ( plt1(iindex1+1,iindex2+1)* zindex2         &
                   -plt1(iindex1+1,iindex2  )*(zindex2 - 1.0)) *  zindex1 &
                 -( plt1(iindex1  ,iindex2+1)* zindex2         &
                   -plt1(iindex1  ,iindex2  )*(zindex2 - 1.0)) * (zindex1 - 1.0)

      if(present(plt2)) then
        pout2(jl) = ( plt2(iindex1+1,iindex2+1)* zindex2         &
                     -plt2(iindex1+1,iindex2  )*(zindex2 - 1.0)) *  zindex1 &
                   -( plt2(iindex1  ,iindex2+1)* zindex2         &
                     -plt2(iindex1  ,iindex2  )*(zindex2 - 1.0)) * (zindex1 - 1.0)
      endif

      if(present(plt3)) then
        pout3(jl) = ( plt3(iindex2+1,iindex1+1)* zindex1         &
                     -plt3(iindex2+1,iindex1  )*(zindex1 - 1.0)) *  zindex2 &
                   -( plt3(iindex2  ,iindex1+1)* zindex1         &
                     -plt3(iindex2  ,iindex1  )*(zindex1 - 1.0)) * (zindex2 - 1.0)
      endif

    else
      pout1(jl)=0.
      if(present(plt2)) pout2(jl)=0.
      if(present(plt3)) pout3(jl)=0.
    endif
  enddo

endif
end subroutine interp_micro_2d


end module mode_ice4_fast_processes

