module mode_ice4_warm
    implicit none
    contains
    
    subroutine ice4_warm(kproma, ksize, ldsoft, &
          &xalpw, xbetaw, xgamw, xepsilo, &
          &xlvtt, xcpv, xcl, xtt, xrv, xcpd, &
          &xtimautc, xcriautc, xfcaccr, xexcaccr, &
          &x0evar, x1evar, xex0evar, xex1evar, &
          &c_rtmin, r_rtmin, xcexvt, &
          &ldcompute, hsubg_rr_evap, &
          &prhodref, pt, ppres, ptht, &
          &plbdar, plbdar_rf, pka, pdv, pcj, &
          &phlc_hcf, phlc_hrc, &
          &pcf, prf, &
          &prvt, prct, prrt, &
          &prcautr, prcaccr, prrevav)

    implicit none
    !
    !*       0.1   declarations of dummy arguments :
    !
    ! cst
    real, intent(in) :: xalpw, xbetaw, xgamw, xepsilo
    real, intent(in) :: xlvtt, xcpv, xcl, xtt, xrv, xcpd

    ! icep
    real, intent(in) :: xtimautc, xcriautc, xfcaccr, xexcaccr
    real, intent(in) :: x0evar, x1evar, xex0evar, xex1evar

    ! icep
    real, intent(in) :: c_rtmin, r_rtmin, xcexvt


    integer,                      intent(in)    :: kproma, ksize
    logical,                      intent(in)    :: ldsoft
    logical, dimension(kproma),   intent(in)    :: ldcompute
    character(len=80),            intent(in)    :: hsubg_rr_evap ! subgrid rr evaporation
    real, dimension(kproma),      intent(in)    :: prhodref ! reference density
    real, dimension(kproma),      intent(in)    :: pt       ! temperature
    real, dimension(kproma),      intent(in)    :: ppres    ! absolute pressure at t
    real, dimension(kproma),      intent(in)    :: ptht     ! theta at time t
    real, dimension(kproma),      intent(in)    :: plbdar   ! slope parameter of the raindrop  distribution
    real, dimension(kproma),      intent(in)    :: plbdar_rf!like plbdar but for the rain fraction part
    real, dimension(kproma),      intent(in)    :: pka      ! thermal conductivity of the air
    real, dimension(kproma),      intent(in)    :: pdv      ! diffusivity of water vapor in the air
    real, dimension(kproma),      intent(in)    :: pcj      ! function to compute the ventilation coefficient
    real, dimension(kproma),      intent(in)    :: phlc_hcf ! hlclouds : fraction of high cloud fraction in grid
    real, dimension(kproma),      intent(in)    :: phlc_hrc ! hlclouds : lwc that is high lwc in grid
    real, dimension(kproma),      intent(in)    :: pcf      ! cloud fraction
    real, dimension(kproma),      intent(in)    :: prf      ! rain fraction
    real, dimension(kproma),      intent(in)    :: prvt     ! water vapor m.r. at t
    real, dimension(kproma),      intent(in)    :: prct     ! cloud water m.r. at t
    real, dimension(kproma),      intent(in)    :: prrt     ! rain water m.r. at t
    real, dimension(kproma),      intent(inout) :: prcautr   ! autoconversion of r_c for r_r production
    real, dimension(kproma),      intent(inout) :: prcaccr  ! accretion of r_c for r_r production
    real, dimension(kproma),      intent(inout) :: prrevav  ! evaporation of r_r
    !
    !*       0.2  declaration of local variables
    !
    real :: zzw2, zzw3, zzw4
    real, dimension(kproma) :: zusw ! undersaturation over water
    real, dimension(kproma) :: zthlt    ! liquid potential temperature
    integer :: jl
    !
    !*       4.2    compute the autoconversion of r_c for r_r production: rcautr
    !
    do jl=1, ksize
      if(phlc_hrc(jl)>c_rtmin .and. phlc_hcf(jl)>0. .and. ldcompute(jl)) then
        if(.not. ldsoft) then
          prcautr(jl) = xtimautc*max(phlc_hrc(jl) - phlc_hcf(jl)*xcriautc/prhodref(jl), 0.0)
        endif
      else
        prcautr(jl) = 0.
      endif
    enddo
    !
    !
    !*       4.3    compute the accretion of r_c for r_r production: rcaccr
    !
      !cloud water and rain are diluted over the grid box
      do jl=1, ksize
        if(prct(jl)>c_rtmin .and. prrt(jl)>r_rtmin .and. ldcompute(jl)) then
          if(.not. ldsoft) then
            prcaccr(jl) = xfcaccr * prct(jl)                &
                        & * plbdar(jl)**xexcaccr    &
                        & * prhodref(jl)**(-xcexvt)
          endif
        else
          prcaccr(jl) = 0.
        endif
      enddo
    !
    !*       4.4    compute the evaporation of r_r: rrevav
    !
    if (hsubg_rr_evap=='none') then
      do jl=1, ksize
        if(prrt(jl)>r_rtmin .and. prct(jl)<=c_rtmin .and. ldcompute(jl)) then
          if(.not. ldsoft) then
            prrevav(jl) = exp(xalpw - xbetaw/pt(jl) - xgamw*alog(pt(jl))) ! es_w
            zusw(jl) = 1. - prvt(jl)*(ppres(jl)-prrevav(jl)) / (xepsilo * prrevav(jl)) ! undersaturation over water
            prrevav(jl) = (xlvtt+(xcpv-xcl)*(pt(jl)-xtt) )**2 / (pka(jl)*xrv*pt(jl)**2) &
                        &+(xrv*pt(jl)) / (pdv(jl)*prrevav(jl))
            prrevav(jl) = (max(0.,zusw(jl) )/(prhodref(jl)*prrevav(jl)) ) * &
                        & (x0evar*plbdar(jl)**xex0evar+x1evar*pcj(jl)*plbdar(jl)**xex1evar)
          endif
        else
          prrevav(jl)=0.
        endif
      enddo
    
    elseif (hsubg_rr_evap=='clfr' .or. hsubg_rr_evap=='prfr') then
      !attention
      !il faudrait recalculer les variables pka, pdv, pcj en tenant compte de la température t^u
      !ces variables devraient être sorties de rain_ice_slow et on mettrait le calcul de t^u, t^s
      !et plusieurs versions (comme actuellement, en ciel clair, en ciel nuageux) de pka, pdv, pcj dans rain_ice
      !on utiliserait la bonne version suivant l'option none, clfr... dans l'évaporation et ailleurs
    
      do jl=1, ksize
        !evaporation in clear sky part
        !with clfr, rain is diluted over the grid box
        !with prfr, rain is concentrated in its fraction
        !use temperature and humidity in clear sky part like bechtold et al. (1993)
        if (hsubg_rr_evap=='clfr') then
          zzw4=1. !precipitation fraction
          zzw3=plbdar(jl)
        else
          zzw4=prf(jl) !precipitation fraction
          zzw3=plbdar_rf(jl)
        endif
    
        if(prrt(jl)>r_rtmin .and. zzw4>pcf(jl) .and. ldcompute(jl)) then
          if(.not. ldsoft) then
            ! outside the cloud (environment) the use of t^u (unsaturated) instead of t
            ! bechtold et al. 1993
            !
            ! t_l
            zthlt(jl) = ptht(jl) - xlvtt*ptht(jl)/xcpd/pt(jl)*prct(jl)
            !
            ! t^u = t_l = theta_l * (t/theta)
            zzw2 =  zthlt(jl) * pt(jl) / ptht(jl)
            !
            ! es_w with new t^u
            prrevav(jl)  = exp(xalpw - xbetaw/zzw2 - xgamw*alog(zzw2))
            !
            ! s, undersaturation over water (with new theta^u)
            zusw(jl) = 1.0 - prvt(jl)*(ppres(jl)-prrevav(jl)) / (xepsilo * prrevav(jl))
            !
            prrevav(jl) = (xlvtt+(xcpv-xcl)*(zzw2-xtt))**2 / (pka(jl)*xrv*zzw2**2) &
                        &+(xrv*zzw2) / (pdv(jl)*prrevav(jl))
            !
            prrevav(jl) = max(0., zusw(jl))/(prhodref(jl)*prrevav(jl))  *      &
                        & (x0evar*zzw3**xex0evar+x1evar*pcj(jl)*zzw3**xex1evar)
            !
            prrevav(jl) = prrevav(jl)*(zzw4-pcf(jl))
          endif
        else
          prrevav(jl)=0.
        endif
      enddo
    
    end if
    !
    !
    end subroutine ice4_warm
    end module mode_ice4_warm
    