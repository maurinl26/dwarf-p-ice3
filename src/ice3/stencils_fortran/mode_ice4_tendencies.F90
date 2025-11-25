module mode_ice4_tendencies
   implicit none
contains
   subroutine ice4_nucleation_post_processing(kproma, ksize, &
     &plsfact, pexn, &
     &rvheni_mr, tht, &
     &prvt, prit, zt)

      implicit none

      integer, intent(in) :: kproma, ksize
      real, dimension(kproma), intent(in) :: rvheni_mr
      real, dimension(kproma), intent(in) :: plsfact, pexn
      real, dimension(kproma), intent(inout) :: tht
      real, dimension(kproma), intent(inout) :: prvt, prit
      real, dimension(kproma), intent(out) :: zt

      integer :: jl

      do jl = 1, ksize
         tht(jl) = tht(jl) + rvheni_mr(jl)*plsfact(jl)
         zt(jl) = tht(jl)*pexn(jl)
         prvt(jl) = prvt(jl) - rvheni_mr(jl)
         prit(jl) = prit(jl) + rvheni_mr(jl)
      end do

   end subroutine ice4_nucleation_post_processing

   subroutine ice4_rrhong_post_processing(kproma, ksize, &
                                          pexn, prrhong_mr, plsfact, plvfact, &
                                          ptht, pt, prrt, prgt)

      implicit none

      integer, intent(in) :: kproma, ksize
      real, dimension(kproma), intent(in) :: plsfact, plvfact, pexn, prrhong_mr
      real, dimension(kproma), intent(inout) :: ptht
      real, dimension(kproma), intent(inout) :: pt
      real, dimension(kproma), intent(inout) :: prrt
      real, dimension(kproma), intent(inout) :: prgt


      integer :: jl

      do jl = 1, ksize
         ptht(jl) = ptht(jl) + prrhong_mr(jl)*(plsfact(jl) - plvfact(jl)) ! f(l_f*(rrhong))
         pt(jl) = ptht(jl)*pexn(jl)
         prrt(jl) = prrt(jl) - prrhong_mr(jl)
         prgt(jl) = prgt(jl) + prrhong_mr(jl)
      end do

   end subroutine ice4_rrhong_post_processing

   subroutine ice4_rimltc_post_processing(kproma, ksize, &
                                          pexn, primltc_mr, plsfact, plvfact, &
                                          ptht, pt, prct, prit)

      implicit none

      integer, intent(in) :: kproma, ksize
      real, dimension(kproma), intent(in) :: plsfact, plvfact, pexn, primltc_mr
      real, dimension(kproma), intent(inout) :: ptht, pt, prct, prit

      integer :: jl

      do jl = 1, ksize
         ptht(jl) = ptht(jl) - primltc_mr(jl)*(plsfact(jl) - plvfact(jl))
         pt(jl) = ptht(jl)*pexn(jl)
         prct(jl) = prct(jl) + primltc_mr(jl)
         prit(jl) = prit(jl) - primltc_mr(jl)
      end do

   end subroutine ice4_rimltc_post_processing

   subroutine ice4_fast_rg_pre_processing(kproma, ksize, &
                                          rvdepg, rsmltg, rraccsg, rsaccrg, &
                                          rcrimsg, rsrimcg, rrhong_mr, rsrimcg_mr, &
                                          zgrsi, zrgsi_mr)
      ! Compute instantaneous graupel sources before ice4_fast_rg call
      ! Reference: PHYEX-IAL_CY50T1/common/micro/mode_ice4_tendencies.F90:386-390
      
      implicit none

      integer, intent(in) :: kproma, ksize
      real, dimension(kproma), intent(in) :: rvdepg, rsmltg, rraccsg, rsaccrg
      real, dimension(kproma), intent(in) :: rcrimsg, rsrimcg, rrhong_mr, rsrimcg_mr
      real, dimension(kproma), intent(out) :: zgrsi, zrgsi_mr

      integer :: jl

      do jl = 1, ksize
         ! Sum of all graupel sources from various processes
         zgrsi(jl) = rvdepg(jl) + rsmltg(jl) + rraccsg(jl) + rsaccrg(jl) &
                     + rcrimsg(jl) + rsrimcg(jl)
         
         ! Mixing ratio sources
         zrgsi_mr(jl) = rrhong_mr(jl) + rsrimcg_mr(jl)
      end do

   end subroutine ice4_fast_rg_pre_processing

   subroutine ice4_increment_update(kproma, ksize, &
                                    plsfact, plvfact, prvheni_mr, primltc_mr, &
                                    prrhong_mr, prsrimcg_mr, &
                                    pth_inst, prv_inst, prc_inst, prr_inst, &
                                    pri_inst, prs_inst, prg_inst)

      implicit none

      integer, intent(in) :: kproma, ksize
      real, dimension(kproma), intent(in) :: plsfact, plvfact
      real, dimension(kproma), intent(in) :: prvheni_mr, primltc_mr, prrhong_mr, prsrimcg_mr
      real, dimension(kproma), intent(inout) :: pth_inst, prv_inst, prc_inst, prr_inst
      real, dimension(kproma), intent(inout) :: pri_inst, prs_inst, prg_inst

      integer :: jl

      do jl = 1, ksize
         pth_inst(jl) = pth_inst(jl) + prvheni_mr(jl)*plsfact(jl)
         pth_inst(jl) = pth_inst(jl) + prrhong_mr(jl)*(plsfact(jl) - plvfact(jl))
         pth_inst(jl) = pth_inst(jl) - primltc_mr(jl)*(plsfact(jl) - plvfact(jl))

         prv_inst(jl) = prv_inst(jl) - prvheni_mr(jl)

         prc_inst(jl) = prc_inst(jl) + primltc_mr(jl)

         prr_inst(jl) = prr_inst(jl) - prrhong_mr(jl)

         pri_inst(jl) = pri_inst(jl) + prvheni_mr(jl)
         pri_inst(jl) = pri_inst(jl) - primltc_mr(jl)

         prs_inst(jl) = prs_inst(jl) - prsrimcg_mr(jl)

         prg_inst(jl) = prg_inst(jl) + prrhong_mr(jl)
         prg_inst(jl) = prg_inst(jl) + prsrimcg_mr(jl)

      end do

   end subroutine ice4_increment_update

   subroutine ice4_derived_fields(kproma, ksize, &
                             xalpi, xbetai, xgami, &
                             xepsilo, xtt, xrv, xci, xlstt, xcpv, xp00, xscfac, &
                             zt, prvt, ppres, prhodref, &
                             zzw, pssi, zka, zai, zdv, zcj)

      implicit none

      integer, intent(in) :: kproma, ksize
      real, intent(in) :: xalpi, xbetai, xgami
      real, intent(in) :: xepsilo, xtt, xrv, xci, xlstt, xcpv, xp00, xscfac
      real, dimension(kproma), intent(in) :: zt, prvt, ppres, prhodref
      real, dimension(kproma), intent(out) :: zzw, pssi, zka, zai, zdv, zcj

      integer :: jl

      do jl = 1, ksize
         zzw(jl) = exp(xalpi - xbetai/zt(jl) - xgami*alog(zt(jl)))
         pssi(jl) = prvt(jl)*(ppres(jl) - zzw(jl))/(xepsilo*zzw(jl)) - 1.0
         zka(jl) = 2.38e-2 + 0.0071e-2*(zt(jl) - xtt) ! k_a
         zdv(jl) = 2.11e-5*(zt(jl)/xtt)**1.94*(xp00/ppres(jl)) ! d_v
         zai(jl) = (xlstt + (xcpv - xci)*(zt(jl) - xtt))**2/(zka(jl)**xrv*zt(jl)**2) &
                   + (xrv*zt(jl))/(zdv(jl)*zzw(jl))
         zcj(jl) = xscfac*prhodref(jl)**0.3/sqrt(1.718e-5 + 0.0049e-5*(zt(jl) - xtt))
      end do

   end subroutine ice4_derived_fields

   !TODO : rainfr mask
   subroutine ice4_slope_parameters(ksize, kproma, &
                               xlbr, xlbexr, r_rtmin, xlbg, &
                               xlbdas_min, xlbdas_max, xtrans_mp_gammas, &
                               s_rtmin, g_rtmin, xlbs, xlbexs, xlbexg, &
                               lsnow_t, &
                               prrt, prhodref, prst, prgt, zt, &
                               zlbdar, zlbdar_rf, zlbdas, zlbdag)

      implicit none

      integer, intent(in) :: ksize, kproma
      real, intent(in) :: xlbr, xlbexr, r_rtmin, s_rtmin, g_rtmin, xlbs, xlbexs, xlbg, xlbexg
      real, intent(in) :: xtrans_mp_gammas, xlbdas_max, xlbdas_min
      logical, intent(in) :: lsnow_t
      real, dimension(kproma), intent(in) :: prhodref, prrt, prst, prgt, zt
      real, dimension(kproma), intent(out) :: zlbdar
      real, dimension(kproma), intent(out) :: zlbdar_rf
      real, dimension(kproma), intent(out) :: zlbdas
      real, dimension(kproma), intent(out) :: zlbdag

      integer :: jl

      do jl = 1, ksize
         !zlbdar will be used when we consider rain diluted over the grid box
         if (prrt(jl) > 0.) then
            zlbdar(jl) = xlbr*(prhodref(jl)*max(prrt(jl), r_rtmin))**xlbexr
         else
            zlbdar(jl) = 0.
         end if
         !zlbdar_rf is used when we consider rain concentrated in its fraction

            zlbdar_rf(jl) = zlbdar(jl)
         if (lsnow_t) then
            if (prst(jl) > 0. .and. zt(jl) > 263.15) then
               zlbdas(jl) = max(min(xlbdas_max, 10**(14.554 - 0.0423*zt(jl))), xlbdas_min)*xtrans_mp_gammas
            else if (prst(jl) > 0. .and. zt(jl) <= 263.15) then
               zlbdas(jl) = max(min(xlbdas_max, 10**(6.226 - 0.0106*zt(jl))), xlbdas_min)*xtrans_mp_gammas
            else
               zlbdas(jl) = 0.
            end if
         else
            if (prst(jl) > 0.) then
               zlbdas(jl) = min(xlbdas_max, xlbs*(prhodref(jl)*max(prst(jl), s_rtmin))**xlbexs)
            else
               zlbdas(jl) = 0.
            end if
         end if
         if (prgt(jl) > 0.) then
            zlbdag(jl) = xlbg*(prhodref(jl)*max(prgt(jl), g_rtmin))**xlbexg
         else
            zlbdag(jl) = 0.
         end if
      end do

   end subroutine ice4_slope_parameters

   subroutine ice4_total_tendencies_update(kproma, ksize, &
                                      plsfact, plvfact, &
                                      pth_tnd, prv_tnd, prc_tnd, &
                                      prr_tnd, pri_tnd, prs_tnd, prg_tnd, &
                                      rvdepg, rchoni, rvdeps, rcrimsg, rcrimss, &
                                      ricfrr, rrwetg, rcwetg, &
                                      rcdryg, rrdryg, rcberi, rrevav, &
                                      rraccss, rraccsg, rgmltr, rcautr, rcaccr, &
                                      rcmltsr, rrcfrig, riaggs, riauts, ridryg, &
                                      riwetg, ricfrrg, rsrimcg, rsaccrg, rsmltg, &
                                      rsdryg, rswetg, rwetgh)

      implicit none

      integer, intent(in) :: kproma, ksize
      real, dimension(kproma), intent(in) :: plsfact, plvfact
      real, dimension(kproma), intent(in) :: rvdepg, rchoni, rvdeps, rcrimsg, rcrimss, rcberi
      real, dimension(kproma), intent(in) ::  rrevav, rraccss, rraccsg, rgmltr, rcautr, rcaccr
      real, dimension(kproma), intent(in) :: ricfrr, rrwetg, rcdryg, rrdryg, rcwetg
      real, dimension(kproma), intent(in) :: rcmltsr, rrcfrig, riaggs, riauts, riwetg, ridryg
      real, dimension(kproma), intent(in) :: ricfrrg, rsrimcg, rsaccrg, rsmltg, rsdryg, rswetg
      real, dimension(kproma), intent(in) :: rwetgh
      real, dimension(kproma), intent(inout) :: pth_tnd, prv_tnd, prc_tnd, prr_tnd, pri_tnd, prs_tnd, prg_tnd

      integer :: jl

      do jl = 1, ksize
         pth_tnd(jl) = pth_tnd(jl) + rvdepg(jl)*plsfact(jl)
         pth_tnd(jl) = pth_tnd(jl) + rchoni(jl)*(plsfact(jl) - plvfact(jl))
         pth_tnd(jl) = pth_tnd(jl) + rvdeps(jl)*plsfact(jl)
         pth_tnd(jl) = pth_tnd(jl) - rrevav(jl)*plvfact(jl)
         pth_tnd(jl) = pth_tnd(jl) + rcrimss(jl)*(plsfact(jl) - plvfact(jl))
         pth_tnd(jl) = pth_tnd(jl) + rcrimsg(jl)*(plsfact(jl) - plvfact(jl))
         pth_tnd(jl) = pth_tnd(jl) + rraccss(jl)*(plsfact(jl) - plvfact(jl))
         pth_tnd(jl) = pth_tnd(jl) + rraccsg(jl)*(plsfact(jl) - plvfact(jl))
         pth_tnd(jl) = pth_tnd(jl) + (rrcfrig(jl) - ricfrr(jl))*(plsfact(jl) - plvfact(jl))
         pth_tnd(jl) = pth_tnd(jl) + (rcwetg(jl) + rrwetg(jl))*(plsfact(jl) - plvfact(jl))
         pth_tnd(jl) = pth_tnd(jl) + (rcdryg(jl) + rrdryg(jl))*(plsfact(jl) - plvfact(jl))
         pth_tnd(jl) = pth_tnd(jl) - rgmltr(jl)*(plsfact(jl) - plvfact(jl))

         pth_tnd(jl) = pth_tnd(jl) + rcberi(jl)*(plsfact(jl) - plvfact(jl))

         prv_tnd(jl) = prv_tnd(jl) - rvdepg(jl)
         prv_tnd(jl) = prv_tnd(jl) - rvdeps(jl)
         prv_tnd(jl) = prv_tnd(jl) + rrevav(jl)

         prc_tnd(jl) = prc_tnd(jl) - rchoni(jl)
         prc_tnd(jl) = prc_tnd(jl) - rcautr(jl)
         prc_tnd(jl) = prc_tnd(jl) - rcaccr(jl)
         prc_tnd(jl) = prc_tnd(jl) - rcrimss(jl)
         prc_tnd(jl) = prc_tnd(jl) - rcrimsg(jl)
         prc_tnd(jl) = prc_tnd(jl) - rcmltsr(jl)
         prc_tnd(jl) = prc_tnd(jl) - rcwetg(jl)
         prc_tnd(jl) = prc_tnd(jl) - rcdryg(jl)
         prc_tnd(jl) = prc_tnd(jl) - rcberi(jl)

         prr_tnd(jl) = prr_tnd(jl) + rcautr(jl)
         prr_tnd(jl) = prr_tnd(jl) + rcaccr(jl)
         prr_tnd(jl) = prr_tnd(jl) - rrevav(jl)
         prr_tnd(jl) = prr_tnd(jl) - rraccss(jl)
         prr_tnd(jl) = prr_tnd(jl) - rraccsg(jl)
         prr_tnd(jl) = prr_tnd(jl) + rcmltsr(jl)
         prr_tnd(jl) = prr_tnd(jl) - rrcfrig(jl) + ricfrr(jl)
         prr_tnd(jl) = prr_tnd(jl) - rrwetg(jl)
         prr_tnd(jl) = prr_tnd(jl) - rrdryg(jl)
         prr_tnd(jl) = prr_tnd(jl) + rgmltr(jl)

         pri_tnd(jl) = pri_tnd(jl) + rchoni(jl)
         pri_tnd(jl) = pri_tnd(jl) - riaggs(jl)
         pri_tnd(jl) = pri_tnd(jl) - riauts(jl)
         pri_tnd(jl) = pri_tnd(jl) - ricfrrg(jl) - ricfrr(jl)
         pri_tnd(jl) = pri_tnd(jl) - riwetg(jl)
         pri_tnd(jl) = pri_tnd(jl) - ridryg(jl)

         pri_tnd(jl) = pri_tnd(jl) + rcberi(jl)

         prs_tnd(jl) = prs_tnd(jl) + rvdeps(jl)
         prs_tnd(jl) = prs_tnd(jl) + riaggs(jl)
         prs_tnd(jl) = prs_tnd(jl) + riauts(jl)
         prs_tnd(jl) = prs_tnd(jl) + rcrimss(jl)
         prs_tnd(jl) = prs_tnd(jl) - rsrimcg(jl)
         prs_tnd(jl) = prs_tnd(jl) + rraccss(jl)
         prs_tnd(jl) = prs_tnd(jl) - rsaccrg(jl)
         prs_tnd(jl) = prs_tnd(jl) - rsmltg(jl)
         prs_tnd(jl) = prs_tnd(jl) - rswetg(jl)
         prs_tnd(jl) = prs_tnd(jl) - rsdryg(jl)

         prg_tnd(jl) = prg_tnd(jl) + rvdepg(jl)
         prg_tnd(jl) = prg_tnd(jl) + rcrimsg(jl) + rsrimcg(jl)
         prg_tnd(jl) = prg_tnd(jl) + rraccsg(jl) + rsaccrg(jl)
         prg_tnd(jl) = prg_tnd(jl) + rsmltg(jl)
         prg_tnd(jl) = prg_tnd(jl) + ricfrrg(jl) + rrcfrig(jl)
         prg_tnd(jl) = prg_tnd(jl) + rcwetg(jl) + riwetg(jl) + rswetg(jl) + rrwetg(jl)
         prg_tnd(jl) = prg_tnd(jl) - rwetgh(jl)
         prg_tnd(jl) = prg_tnd(jl) + rcdryg(jl) + ridryg(jl) + rsdryg(jl) + rrdryg(jl)
         prg_tnd(jl) = prg_tnd(jl) - rgmltr(jl)
      end do
   end subroutine ice4_total_tendencies_update

end module mode_ice4_tendencies
