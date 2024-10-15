!MNH_LIC Copyright 1996-2021 CNRS, Meteo-France and Universite Paul Sabatier
!MNH_LIC This is part of the Meso-NH software governed by the CeCILL-C licence
!MNH_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt
!MNH_LIC for details. version 1.
!-----------------------------------------------------------------
module mode_cloud_fraction
  implicit none
  contains
!     ##########################################################################
      subroutine cloud_fraction(nijt, nkt, &
                            &nkte, nktb, &
                            &nijb, nije, &
                            &xcriautc, xcriauti, xacriauti, xbcriauti, xtt, &
                            &csubg_mf_pdf, &
                            &lsubg_cond, &
                            &zri, zrc, &
                            &ptstep,                                  &
                            &pexnref, prhodref, &
                            &pcf_mf, prc_mf, pri_mf,                     &
                            &prc, prvs, prcs, pths,                  &
                            &pcldfr,                      &
                            &pri, pris, &
                            &phlc_hrc, phlc_hcf, phli_hri, phli_hcf)

!     ########################################################################
!-------------------------------------------------------------------------------
!
!*       0.    declarations
!              ------------
implicit none
!*       0.1   declarations of dummy arguments :
!
!

integer, intent(in) :: nijt, nkt
integer, intent(in) :: nkte, nktb
integer, intent(in) :: nijb, nije
logical, intent(in) :: lsubg_cond
real, intent(in) :: xcriautc, xcriauti, xacriauti, xbcriauti, xtt
character(len=80), intent(in) :: csubg_mf_pdf

real, dimension(nijt, nkt), intent(in) :: zrc, zri


real,                     intent(in)   :: ptstep    ! double time step
                                                    ! (single if cold start)
!
real, dimension(nijt,nkt), intent(in)    ::  pexnref ! reference exner function
real, dimension(nijt,nkt), intent(in)    ::  prhodref
!
real, dimension(nijt,nkt), intent(in)    :: pcf_mf   ! convective mass flux cloud fraction
real, dimension(nijt,nkt), intent(in)    :: prc_mf   ! convective mass flux liquid mixing ratio
real, dimension(nijt,nkt), intent(in)    :: pri_mf   ! convective mass flux ice mixing ratio
!
real, dimension(nijt,nkt), intent(in)    :: prc     ! cloud water m.r. to adjust
real, dimension(nijt,nkt), intent(in)   ::  pri  ! cloud ice  m.r. to adjust

real, dimension(nijt,nkt), intent(inout) :: prvs    ! water vapor m.r. source
real, dimension(nijt,nkt), intent(inout) :: prcs    ! cloud water m.r. source
real, dimension(nijt,nkt), intent(inout) :: pths    ! theta source
real, dimension(nijt,nkt), intent(inout)::  pris ! cloud ice  m.r. at t+1

real, dimension(nijt,nkt), intent(out)  ::  pcldfr  ! cloud fraction
real, dimension(nijt,nkt), optional, intent(out)  ::  phlc_hrc
real, dimension(nijt,nkt), optional, intent(out)  ::  phlc_hcf
real, dimension(nijt,nkt), optional, intent(out)  ::  phli_hri
real, dimension(nijt,nkt), optional, intent(out)  ::  phli_hcf
!
!
!*       0.2   declarations of local variables :
!
!
real  :: zw1,zw2    ! intermediate fields
real, dimension(nijt,nkt) &
                         :: zt,   &  ! adjusted temperature
                            zcph, &  ! guess of the cph for the mixing
                            zlv,  &  ! guess of the lv at t+1
                            zls      ! guess of the ls at t+1
real :: zcriaut, & ! autoconversion thresholds
        zhcf, zhr
!
integer             :: jij, jk
integer :: iktb, ikte, iijb, iije
!
logical :: llnone, lltriangle, llhlc_h, llhli_h

! real(kind=jphook) :: zhook_handle
!
!-------------------------------------------------------------------------------
!
!*       1.     preliminaries
!               -------------
!
! if (lhook) call dr_hook('ice_adjust',0,zhook_handle)
!
iktb=nktb
ikte=nkte
iijb=nijb
iije=nije
!
!*       5.     compute the sources and stores the cloud fraction
!               -------------------------------------------------
!
!
do jk=iktb,ikte
  do jij=iijb,iije
    !
    !*       5.0    compute the variation of mixing ratio
    !
                                                         !         rc - rc*
    zw1 = (zrc(jij,jk) - prc(jij,jk)) / ptstep       ! pcon = ----------
                                                         !         2 delta t
    zw2 = (zri(jij,jk) - pri(jij,jk)) / ptstep       ! idem zw1 but for ri
    !
    !*       5.1    compute the sources
    !
    if( zw1 < 0.0 ) then
      zw1 = max ( zw1, -prcs(jij,jk) )
    else
      zw1 = min ( zw1,  prvs(jij,jk) )
    endif
    prvs(jij,jk) = prvs(jij,jk) - zw1
    prcs(jij,jk) = prcs(jij,jk) + zw1
    pths(jij,jk) = pths(jij,jk) +        &
                    zw1 * zlv(jij,jk) / (zcph(jij,jk) * pexnref(jij,jk))
    !
    if( zw2 < 0.0 ) then
      zw2 = max ( zw2, -pris(jij,jk) )
    else
      zw2 = min ( zw2,  prvs(jij,jk) )
    endif
    prvs(jij,jk) = prvs(jij,jk) - zw2
    pris(jij,jk) = pris(jij,jk) + zw2
    pths(jij,jk) = pths(jij,jk) +        &
                  zw2 * zls(jij,jk) / (zcph(jij,jk) * pexnref(jij,jk))
  enddo
  !
  !*       5.2    compute the cloud fraction pcldfr
  !
  if ( .not. lsubg_cond ) then
    do jij=iijb,iije
      if (prcs(jij,jk) + pris(jij,jk) > 1.e-12 / ptstep) then
        pcldfr(jij,jk)  = 1.
      else
        pcldfr(jij,jk)  = 0.
      endif
    enddo
  else !nebn%lsubg_cond case
    ! tests on characters strings can break the vectorization, or at least they would
    ! slow down considerably the performance of a vector loop. one should use tests on
    ! reals, integers or booleans only. rek.
    llnone=csubg_mf_pdf=='none'
    lltriangle=csubg_mf_pdf=='triangle'
    llhlc_h=present(phlc_hrc).and.present(phlc_hcf)
    llhli_h=present(phli_hri).and.present(phli_hcf)
    do jij=iijb,iije
      !we limit prc_mf+pri_mf to prvs*ptstep to avoid negative humidity
      zw1=prc_mf(jij,jk)/ptstep
      zw2=pri_mf(jij,jk)/ptstep
      if(zw1+zw2>prvs(jij,jk)) then
        zw1=zw1*prvs(jij,jk)/(zw1+zw2)
        zw2=prvs(jij,jk)-zw1
      endif
      pcldfr(jij,jk)=min(1.,pcldfr(jij,jk)+pcf_mf(jij,jk))
      prcs(jij,jk)=prcs(jij,jk)+zw1
      pris(jij,jk)=pris(jij,jk)+zw2
      prvs(jij,jk)=prvs(jij,jk)-(zw1+zw2)
      pths(jij,jk) = pths(jij,jk) + &
                    (zw1 * zlv(jij,jk) + zw2 * zls(jij,jk)) / zcph(jij,jk) / pexnref(jij,jk)
      !
      if(llhlc_h) then
        zcriaut=xcriautc/prhodref(jij,jk)
        if(llnone)then
          if(zw1*ptstep>pcf_mf(jij,jk) * zcriaut) then
            phlc_hrc(jij,jk)=phlc_hrc(jij,jk)+zw1*ptstep
            phlc_hcf(jij,jk)=min(1.,phlc_hcf(jij,jk)+pcf_mf(jij,jk))
          endif
        elseif(lltriangle)then
          !zhcf is the precipitating part of the *cloud* and not of the grid cell
          if(zw1*ptstep>pcf_mf(jij,jk)*zcriaut) then
            zhcf=1.-.5*(zcriaut*pcf_mf(jij,jk) / max(1.e-20, zw1*ptstep))**2
            zhr=zw1*ptstep-(zcriaut*pcf_mf(jij,jk))**3 / &
                                        &(3*max(1.e-20, zw1*ptstep)**2)
          elseif(2.*zw1*ptstep<=pcf_mf(jij,jk) * zcriaut) then
            zhcf=0.
            zhr=0.
          else
            zhcf=(2.*zw1*ptstep-zcriaut*pcf_mf(jij,jk))**2 / &
                       &(2.*max(1.e-20, zw1*ptstep)**2)
            zhr=(4.*(zw1*ptstep)**3-3.*zw1*ptstep*(zcriaut*pcf_mf(jij,jk))**2+&
                        (zcriaut*pcf_mf(jij,jk))**3) / &
                      &(3*max(1.e-20, zw1*ptstep)**2)
          endif
          zhcf=zhcf*pcf_mf(jij,jk) !to retrieve the part of the grid cell
          phlc_hcf(jij,jk)=min(1.,phlc_hcf(jij,jk)+zhcf) !total part of the grid cell that is precipitating
          phlc_hrc(jij,jk)=phlc_hrc(jij,jk)+zhr
        endif
      endif
      if(llhli_h) then
        zcriaut=min(xcriauti,10**(xacriauti*(zt(jij,jk)-xtt)+xbcriauti))
        if(llnone)then
          if(zw2*ptstep>pcf_mf(jij,jk) * zcriaut) then
            phli_hri(jij,jk)=phli_hri(jij,jk)+zw2*ptstep
            phli_hcf(jij,jk)=min(1.,phli_hcf(jij,jk)+pcf_mf(jij,jk))
          endif
        elseif(lltriangle)then
          !zhcf is the precipitating part of the *cloud* and not of the grid cell
          if(zw2*ptstep>pcf_mf(jij,jk)*zcriaut) then
            zhcf=1.-.5*(zcriaut*pcf_mf(jij,jk) / (zw2*ptstep))**2
            zhr=zw2*ptstep-(zcriaut*pcf_mf(jij,jk))**3/(3*(zw2*ptstep)**2)
          elseif(2.*zw2*ptstep<=pcf_mf(jij,jk) * zcriaut) then
            zhcf=0.
            zhr=0.
          else
            zhcf=(2.*zw2*ptstep-zcriaut*pcf_mf(jij,jk))**2 / (2.*(zw2*ptstep)**2)
            zhr=(4.*(zw2*ptstep)**3-3.*zw2*ptstep*(zcriaut*pcf_mf(jij,jk))**2+&
                        (zcriaut*pcf_mf(jij,jk))**3)/(3*(zw2*ptstep)**2)
          endif
          zhcf=zhcf*pcf_mf(jij,jk) !to retrieve the part of the grid cell
          phli_hcf(jij,jk)=min(1.,phli_hcf(jij,jk)+zhcf) !total part of the grid cell that is precipitating
          phli_hri(jij,jk)=phli_hri(jij,jk)+zhr
        endif
      endif
    enddo
    !
  endif !nebn%lsubg_cond
enddo

end subroutine cloud_fraction
end module mode_cloud_fraction
