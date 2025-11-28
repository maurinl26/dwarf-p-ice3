! Created by  on 02/06/2025.

module mode_ice4_sedimentation_stat

    implicit none
contains
subroutine ice4_sedimentation_stat(d, cst, icep, iced, parami, &
                                  &ptstep, krr, pdzz, &
                                  &prhodref, ppabst, ptht, pt, prhodj, &
                                  &prs, prt, &
                                  &pinprc, pinprr, pinpri, pinprs, pinprg, &
                                  &psea, ptown, &
                                  &pinprh, pfpr)


use modd_dimphyex, only: dimphyex_t
use modd_cst, only: cst_t
use modd_rain_ice_descr_n, only: rain_ice_descr_t
use modd_rain_ice_param_n, only: rain_ice_param_t
use modd_param_ice_n,      only: param_ice_t
use modd_fields_address
use modi_gamma, only: gamma
!
implicit none
!
!*       0.1   declarations of dummy arguments :
!
integer, intent(in) :: nijt, 
real, intent(in) :: xnuc, xalphac, xnuc2, xalphac2
type(cst_t),                  intent(in)              :: cst
type(rain_ice_param_t),       intent(in)              :: icep
type(rain_ice_descr_t),       intent(in)              :: iced
type(param_ice_t),            intent(in)              :: parami
real,                         intent(in)              :: ptstep  ! double time step (single if cold start)
integer,                      intent(in)              :: krr     ! number of moist variable
real, dimension(d%nijt,d%nkt), intent(in)              :: pdzz    ! layer thikness (m)
real, dimension(d%nijt,d%nkt), intent(in)              :: prhodref! reference density
real, dimension(d%nijt,d%nkt), intent(in)              :: ppabst  ! absolute pressure at t
real, dimension(d%nijt,d%nkt), intent(in)              :: ptht    ! theta at time t
real, dimension(d%nijt,d%nkt), intent(in)              :: pt      ! temperature
real, dimension(d%nijt,d%nkt), intent(in)              :: prhodj  ! dry density * jacobian
real, dimension(d%nijt,d%nkt,krr), intent(inout)       :: prs     ! m.r. source
real, dimension(d%nijt,d%nkt,krr), intent(in)          :: prt     ! m.r. at t
real, dimension(d%nijt),     intent(out)             :: pinprc  ! cloud instant precip
real, dimension(d%nijt),     intent(out)             :: pinprr  ! rain instant precip
real, dimension(d%nijt),     intent(out)             :: pinpri  ! pristine ice instant precip
real, dimension(d%nijt),     intent(out)             :: pinprs  ! snow instant precip
real, dimension(d%nijt),     intent(out)             :: pinprg  ! graupel instant precip
real, dimension(d%nijt),     optional, intent(in)    :: psea    ! sea mask
real, dimension(d%nijt),     optional, intent(in)    :: ptown   ! fraction that is town
real, dimension(d%nijt),         optional, intent(out)   :: pinprh  ! hail instant precip
real, dimension(d%nijt,d%nkt,krr), optional, intent(out)   :: pfpr    ! upper-air precipitation fluxes
!
!*       0.2  declaration of local variables
!
logical :: llsea_and_town
integer :: jrr, jij, jk, ikb, ike,ikl, iijb, iije, iktb, ikte
integer :: ishift, ik, ikplus
real :: zinvtstep, zgac, zgc, zgac2, zgc2, zraydefo
real, dimension(d%nijt) :: ztsorhodz        ! timestep over (rhodref times delta z)
real, dimension(d%nijt,0:1,2:krr) :: zsed   ! sedimentation flux array for each species and for above and current levels
!
ikb=d%nkb
ike=d%nke
ikl=d%nkl
iijb=d%nijb
iije=d%nije
iktb=d%nktb
ikte=d%nkte
!
if ( present( pfpr ) ) then
 !set to 0. to avoid undefined values (in files)
 pfpr(:, : iktb, :) = 0.
 pfpr(:, ikte :, :) = 0.
end if

!-------------------------------------------------------------------------------
!
!*       1.    compute the fluxes
!
zinvtstep = 1./ptstep
zgac=gamma(iced%xnuc+1.0/xalphac)
zgc=gamma(xnuc)
zgac2=gamma(xnuc2+1.0/xalphac2)
zgc2=gamma(xnuc2)
zraydefo=max(1.,0.5*(zgac/zgc))
llsea_and_town=present(psea).and.present(ptown)

!
!*       2.    compute the fluxes
!
! start shift mechanism:
ishift=0
call shift

! initialize vertical loop
do jrr=irc,krr
  zsed(:,ikplus,jrr) = 0.
enddo

! calculation sedimentation flux
do jk = ike , ikb, -1*ikl

  do jij = iijb, iije
    ztsorhodz(jij) =ptstep/(prhodref(jij,jk)*pdzz(jij,jk))
  enddo
!
  do jrr=irc,krr

    if (jrr==irc) then

      !******* for cloud
      if (parami%lsedic) then
        call cloud(prt(:,jk,irc))
      else
        zsed(:,ik,jrr)=0.
      endif

    elseif (jrr==irr) then

      !*       2.2   for rain
      call other_species(icep%xfsedr,icep%xexsedr,prt(:,jk,irr))

    elseif (jrr==iri) then

      call pristine_ice(prt(:,jk,iri))

    elseif (jrr==irs) then

      !*       2.4   for aggregates/snow
      if(.not. icep%lnewcoeff) then
        call other_species(icep%xfseds,icep%xexseds,prt(:,jk,irs))
      else
        call snow(prt(:,jk,irs))
      endif

    elseif (jrr==irg) then

      !*       2.5   for graupeln
      call other_species(icep%xfsedg,icep%xexsedg,prt(:,jk,irg))

    elseif (jrr==irh) then

      !*       2.6   for hail
       call other_species(icep%xfsedh,icep%xexsedh,prt(:,jk,irh))

    endif

  enddo ! jrr

  ! wrap-up

  if(present(pfpr)) then
    do jrr=irc,krr
      pfpr(:,jk,jrr)=zsed(:,ik,jrr)
    enddo
  endif

  do jrr=irc, krr
    do jij = iijb, iije
      prs(jij,jk,jrr) = prs(jij,jk,jrr)+ztsorhodz(jij)*(zsed(jij,ikplus,jrr)-zsed(jij,ik,jrr))*zinvtstep
    enddo
  enddo

  if (jk==ikb) then
    do jij = iijb, iije
      if(parami%lsedic) pinprc(jij) = zsed(jij,ik,2)/cst%xrholw
      pinprr(jij) = zsed(jij,ik,3)/cst%xrholw
      pinpri(jij) = zsed(jij,ik,4)/cst%xrholw
      pinprs(jij) = zsed(jij,ik,5)/cst%xrholw
      pinprg(jij) = zsed(jij,ik,6)/cst%xrholw
      if (present(pinprh) .and. krr==7) then
        pinprh(jij) = zsed(jij,ik,7)/cst%xrholw
      endif
    enddo
  endif

  ! shift mechanism : current level now takes the place of previous one
  call shift

enddo ! jk

end subroutine ice4_sedimentation_stat

  subroutine cloud(prxt)

    real, intent(in)    :: prxt(d%nijt) ! mr of specy x

    real :: zlbc    ! xlbc weighted by sea fraction
    real :: zfsedc
    real :: zconc3d ! droplet condensation
    real :: zray    ! cloud mean radius
    real :: zzwlbda, zzwlbdc, zzcc
    integer :: jij
    real :: zqp
    real :: zwsedw1, zwsedw2

    !!real(kind=jphook) :: zhook_handle

    !!if (lhook) call dr_hook('ice4_sedimentation_stat:cloud',0,zhook_handle)

    do jij = iijb, iije
      !estimation of q' taking into account incoming zwsed from previous vertical level
      zqp=zsed(jij,ikplus,jrr)*ztsorhodz(jij)
      if ((prxt(jij) > iced%xrtmin(jrr)) .or. (zqp > iced%xrtmin(jrr))) then
        if (llsea_and_town) then
          zray   = max(1.,0.5*((1.-psea(jij))*zgac/zgc+psea(jij)*zgac2/zgc2))
          zlbc   = max(min(iced%xlbc(1),iced%xlbc(2)),(psea(jij)*iced%xlbc(2)+(1.-psea(jij))*iced%xlbc(1)) )
          zfsedc = max(min(icep%xfsedc(1),icep%xfsedc(2)), (psea(jij)*icep%xfsedc(2)+(1.-psea(jij))*icep%xfsedc(1)) )
          zconc3d= (1.-ptown(jij))*(psea(jij)*iced%xconc_sea+(1.-psea(jij))*iced%xconc_land) + &
                    ptown(jij)  *iced%xconc_urban
        else
          zray   = zraydefo
          zlbc   = iced%xlbc(1)
          zfsedc = icep%xfsedc(1)
          zconc3d= iced%xconc_land
        endif
        !calculation of w
        if(prxt(jij) > iced%xrtmin(jrr)) then
          zzwlbda=6.6e-8*(101325./ppabst(jij,jk))*(ptht(jij,jk)/293.15)
          zzwlbdc=(zlbc*zconc3d/(prhodref(jij,jk)*prxt(jij)))**iced%xlbexc
          zzcc=iced%xcc*(1.+1.26*zzwlbda*zzwlbdc/zray) !! zcc  : fall speed
          zwsedw1=prhodref(jij,jk)**(-iced%xcexvt ) * zzwlbdc**(-iced%xdc)*zzcc*zfsedc
        else
          zwsedw1=0.
        endif
        if ( zqp > iced%xrtmin(jrr) ) then
          zzwlbda=6.6e-8*(101325./ppabst(jij,jk))*(ptht(jij,jk)/293.15)
          zzwlbdc=(zlbc*zconc3d/(prhodref(jij,jk)*zqp))**iced%xlbexc
          zzcc=iced%xcc*(1.+1.26*zzwlbda*zzwlbdc/zray) !! zcc  : fall speed
          zwsedw2=prhodref(jij,jk)**(-iced%xcexvt ) * zzwlbdc**(-iced%xdc)*zzcc*zfsedc
        else
          zwsedw2=0.
        endif
      else
        zwsedw1=0.
        zwsedw2=0.
      endif
!- duplicated code -------------------------------------------------------------------------
      if (zwsedw2 /= 0.) then
        zsed(jij,ik,jrr)=fwsed1(zwsedw1,ptstep,pdzz(jij,jk),prhodref(jij,jk),prxt(jij),zinvtstep) &
         & + fwsed2(zwsedw2,ptstep,pdzz(jij,jk),zsed(jij,ikplus,jrr))
      else
        zsed(jij,ik,jrr)=fwsed1(zwsedw1,ptstep,pdzz(jij,jk),prhodref(jij,jk),prxt(jij),zinvtstep)
      endif
!-------------------------------------------------------------------------------------------
    enddo

    !!if (lhook) call dr_hook('ice4_sedimentation_stat:cloud',1,zhook_handle)

  end subroutine cloud

  subroutine pristine_ice(prxt)

    real, intent(in)    :: prxt(d%nijt) ! mr of specy x
    integer :: jij
    real :: zqp
    real :: zwsedw1, zwsedw2

    !!real(kind=jphook) :: zhook_handle

    !!if (lhook) call dr_hook('ice4_sedimentation_stat:pristine_ice',0,zhook_handle)

    ! ******* for pristine ice
    do jij = iijb, iije
      zqp=zsed(jij,ikplus,jrr)*ztsorhodz(jij)
      if ((prxt(jij) > iced%xrtmin(jrr)) .or. (zqp > iced%xrtmin(jrr))) then
        !calculation of w
        if ( prxt(jij) > max(iced%xrtmin(jrr),1.0e-7 ) ) then
          zwsedw1= icep%xfsedi *  &
                            & prhodref(jij,jk)**(-iced%xcexvt) * & !    mcf&h
                            & max( 0.05e6,-0.15319e6-0.021454e6* &
                            &      log(prhodref(jij,jk)*prxt(jij)) )**icep%xexcsedi
        else
          zwsedw1=0.
        endif
        if ( zqp > max(iced%xrtmin(jrr),1.0e-7 ) ) then
          zwsedw2= icep%xfsedi *  &
                            & prhodref(jij,jk)**(-iced%xcexvt) * & !    mcf&h
                            & max( 0.05e6,-0.15319e6-0.021454e6* &
                            &      log(prhodref(jij,jk)*zqp) )**icep%xexcsedi
        else
          zwsedw2=0.
        endif
      else
        zwsedw1=0.
        zwsedw2=0.
      endif
!- duplicated code -------------------------------------------------------------------------
      if (zwsedw2 /= 0.) then
        zsed(jij,ik,jrr)=fwsed1(zwsedw1,ptstep,pdzz(jij,jk),prhodref(jij,jk),prxt(jij),zinvtstep) &
         & + fwsed2(zwsedw2,ptstep,pdzz(jij,jk),zsed(jij,ikplus,jrr))
      else
        zsed(jij,ik,jrr)=fwsed1(zwsedw1,ptstep,pdzz(jij,jk),prhodref(jij,jk),prxt(jij),zinvtstep)
      endif
!-------------------------------------------------------------------------------------------
    enddo

    !!if (lhook) call dr_hook('ice4_sedimentation_stat:pristine_ice',1,zhook_handle)

  end subroutine pristine_ice

  subroutine snow(prxt)

    real, intent(in)    :: prxt(d%nijt) ! mr of specy x
    integer :: jij
    real :: zqp, zlbdas
    real :: zwsedw1, zwsedw2

    !!real(kind=jphook) :: zhook_handle

    !!if (lhook) call dr_hook('ice4_sedimentation_stat:snow',0,zhook_handle)

    ! ******* for snow
    do jij = iijb, iije
      zqp=zsed(jij,ikplus,jrr)*ztsorhodz(jij)
      if ((prxt(jij) > iced%xrtmin(jrr)) ) then
        !compute lambda_snow parameter
        if (parami%lsnow_t) then
          if(pt(jij,jk)>cst%xtt-10.0) then
            zlbdas = max(min(iced%xlbdas_max, 10**(14.554-0.0423*pt(jij,jk))),iced%xlbdas_min)*iced%xtrans_mp_gammas
          else
            zlbdas = max(min(iced%xlbdas_max, 10**(6.226-0.0106*pt(jij,jk))),iced%xlbdas_min)*iced%xtrans_mp_gammas
          end if
        else
          zlbdas  = max(min(iced%xlbdas_max,iced%xlbs*(prhodref(jij,jk)*prxt(jij))**iced%xlbexs),iced%xlbdas_min)
        end if
        !calculation of w
        if ( prxt(jij) > iced%xrtmin(jrr) ) then
          zwsedw1= icep%xfseds *  &
                        & prhodref(jij,jk)**(-iced%xcexvt) * &
                        & (1+(iced%xfvelos/zlbdas)**iced%xalphas)**(-iced%xnus+icep%xexseds/iced%xalphas)* &
                        & zlbdas**(iced%xbs+icep%xexseds)
        else
          zwsedw1=0.
        endif
        if ( zqp > iced%xrtmin(jrr) ) then
          zwsedw2= icep%xfseds *  &
                        & prhodref(jij,jk)**(-iced%xcexvt) * &
                        & (1+(iced%xfvelos/zlbdas)**iced%xalphas)**(-iced%xnus+icep%xexseds/iced%xalphas)* &
                        & zlbdas**(iced%xbs+icep%xexseds)
        else
          zwsedw2=0.
        endif
      else
        zwsedw1=0.
        zwsedw2=0.
      endif
!- duplicated code -------------------------------------------------------------------------
      if (zwsedw2 /= 0.) then
        zsed(jij,ik,jrr)=fwsed1(zwsedw1,ptstep,pdzz(jij,jk),prhodref(jij,jk),prxt(jij),zinvtstep) &
         & + fwsed2(zwsedw2,ptstep,pdzz(jij,jk),zsed(jij,ikplus,jrr))
      else
        zsed(jij,ik,jrr)=fwsed1(zwsedw1,ptstep,pdzz(jij,jk),prhodref(jij,jk),prxt(jij),zinvtstep)
      endif
!-------------------------------------------------------------------------------------------
    enddo

    !!if (lhook) call dr_hook('ice4_sedimentation_stat:snow',1,zhook_handle)

  end subroutine snow

  subroutine other_species(pfsed,pexsed,prxt)

    real, intent(in)    :: pfsed
    real, intent(in)    :: pexsed
    real, intent(in)    :: prxt(d%nijt) ! mr of specy x
    integer :: jij
    real :: zqp
    real :: zwsedw1, zwsedw2

    !!real(kind=jphook) :: zhook_handle

    !!if (lhook) call dr_hook('ice4_sedimentation_stat:other_species',0,zhook_handle)

    ! for all but cloud and pristine ice :
    do jij = iijb, iije
      zqp=zsed(jij,ikplus,jrr)*ztsorhodz(jij)
      if ((prxt(jij) > iced%xrtmin(jrr)) .or. (zqp > iced%xrtmin(jrr))) then
        !calculation of w
        if ( prxt(jij) > iced%xrtmin(jrr) ) then
          zwsedw1= pfsed *prxt(jij)**(pexsed-1)*prhodref(jij,jk)**(pexsed-iced%xcexvt-1)
        else
          zwsedw1=0.
        endif
        if ( zqp > iced%xrtmin(jrr) ) then
          zwsedw2= pfsed *zqp**(pexsed-1)*prhodref(jij,jk)**(pexsed-iced%xcexvt-1)
        else
          zwsedw2=0.
        endif
      else
        zwsedw1=0.
        zwsedw2=0.
      endif
!- duplicated code -------------------------------------------------------------------------
      if (zwsedw2 /= 0.) then
        zsed(jij,ik,jrr)=fwsed1(zwsedw1,ptstep,pdzz(jij,jk),prhodref(jij,jk),prxt(jij),zinvtstep) &
         & + fwsed2(zwsedw2,ptstep,pdzz(jij,jk),zsed(jij,ikplus,jrr))
      else
        zsed(jij,ik,jrr)=fwsed1(zwsedw1,ptstep,pdzz(jij,jk),prhodref(jij,jk),prxt(jij),zinvtstep)
      endif
!-------------------------------------------------------------------------------------------
    enddo

    !!if (lhook) call dr_hook('ice4_sedimentation_stat:other_species',1,zhook_handle)

  end subroutine other_species

  subroutine shift

    ikplus=ishift
    ik=1-ishift
    ishift=1-ishift

  end subroutine shift
!
!
elemental function fwsed1(pwsedw,ptstep1,pdzz1,prhodref1,prxt1,pinvtstep) result(pvar)
  real, intent(in) :: pwsedw,ptstep1,pdzz1,prhodref1,prxt1,pinvtstep
  real :: pvar
! 5 multiplications only => cost = 5x
  pvar = min(prhodref1*pdzz1*prxt1*pinvtstep,pwsedw*prhodref1*prxt1)
end function fwsed1
!
elemental function fwsed2(pwsedw,ptstep1,pdzz1,pwsedwsup) result(pvar)
  real, intent(in) :: pwsedw,ptstep1,pdzz1,pwsedwsup
  real :: pvar
  pvar = max(0.,1.-pdzz1/(ptstep1*pwsedw))*pwsedwsup
end function fwsed2

end module mode_ice4_sedimentation_stat