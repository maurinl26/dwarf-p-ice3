! ######spl
module mode_aro_adjust
    implicit none
    contains
      subroutine  aro_adjust(klon,kidia,kfdia,klev,krr,  &
        &xcpv, xcpd, xcl, xci, xtt, xlvtt, xlstt, &
        &cmicro, &
        &ptstep, &
        &pzzf, pexnref,&
        &ptht, prc, prr, pri, prs, prg, &
        &pths, prcs, prrs, pris, prss, prgs)
!     ##########################################################################
!
!!****  * -  compute the  resolved clouds and precipitation
!!
!!    purpose
!!    -------
!!      the purpose of this routine is to compute the  microphysical sources
!!    related to the resolved clouds and precipitation
!!
!!
!!**  method
!!    ------
!!      the main actions of this routine is to call the routines computing the
!!    microphysical sources. before that:
!!        - it computes the real absolute pressure,
!!        - negative values of the current guess of all mixing ratio are removed.
!!          this is done by a global filling algorithm based on a multiplicative
!!          method (rood, 1987), in order to conserved the total mass in the
!!          simulation domain.
!!        - sources are transformed in physical tendencies, by removing the
!!          multiplicative term rhod*j.
!!        - external points values are filled owing to the use of cyclic
!!          l.b.c., in order to performe computations on the full domain.
!!      after calling to microphysical routines, the physical tendencies are
!!    switched back to prognostic variables.
!!
!!
!!    external
!!    --------
!!      subroutine fmlook: to recover the logical unit number linked to a fmfile
!!      subroutine slow_terms: computes the explicit microphysical sources
!!      subroutine fast_terms: performs the saturation adjustment for l
!!      subroutine rain_ice  : computes the explicit microphysical sources for i
!!      subroutine ice_adjust: performs the saturation adjustment for i+l
!!      min_ll,sum3d_ll : distributed functions equivalent to min and sum
!!
!!
!!    implicit arguments
!!    ------------------
!!      module modd_parameters : contains declarations of parameter variables
!!         jphext       : horizontal external points number
!!         jpvext       : vertical external points number
!!      module modd_cst
!!          xp00               ! reference pressure
!!          xrd                ! gaz  constant for dry air
!!          xcpd               ! cpd (dry air)
!!
!!    reference
!!    ---------
!!
!!      documentation arome
!!
!!    author
!!    ------
!!    s.malardel and y.seity
!!
!!    modifications
!!    -------------
!!      original    10/03/03
!!      t. kovacic  11-05-05, call to budgets for nega1_
!!      s. riette ice for edkf
!!      2012-02 y. seity,  add possibility to run with reversed vertical levels
!!      2016-11 s. riette: new ice_adjust interface, add old3/old4 schemes
!!      2018-02 k.i ivarsson : more outputs from ocnd2 option
!!      2020-12 u. andrae : introduce spp for harmonie-arome
!!     r. el khatib 24-aug-2021 optimizations
!!
!-------------------------------------------------------------------------------
!
!*       0.    declarations
!              ------------
!
use iso_fortran_env, only : real64, int32
!
implicit none
!
!*       0.1   declarations of dummy arguments :
!
! type(phyex_t),            intent(in)   :: phyex
integer(kind=int32),                  intent(in)   :: klon ! array length (nproma)
integer(kind=int32),                  intent(in)   :: kidia    !start index (=1)
integer(kind=int32),                  intent(in)   :: kfdia    !end index (=klon only if block is full)
integer(kind=int32),                  intent(in)   :: klev     !number of vertical levels
integer(kind=int32),                  intent(in)   :: krr      ! number of moist variables
character (len=4),        intent(in)   :: cmicro  ! microphysics scheme
real(kind=real64),                     intent(in)   :: ptstep   ! time step

real(kind=real64), intent(in) :: xcpv, xcpd, xcl, xci, xtt, xlvtt, xlstt
!
!
real(kind=real64), dimension(klon,1,klev), intent(in)   :: pexnref ! reference exner function
real(kind=real64), dimension(klon,1,klev), intent(in)   :: pzzf
!
!
real(kind=real64), dimension(klon,1,klev),   intent(in)   :: ptht    ! theta at time t

! replace prt
real(kind=real64), dimension(klon, 1, klev), intent(inout) :: prc ! jrr = 2
real(kind=real64), dimension(klon, 1, klev), intent(inout) :: prr ! jrr = 3
real(kind=real64), dimension(klon, 1, klev), intent(inout) :: pri ! jrr = 4
real(kind=real64), dimension(klon, 1, klev), intent(inout) :: prs ! jrr = 5
real(kind=real64), dimension(klon, 1, klev), intent(inout) :: prg ! jrr = 6

! replace prs
real(kind=real64), dimension(klon,1,klev), intent(inout) :: pths ! jrr = 1
real(kind=real64), dimension(klon,1,klev), intent(inout) :: prcs ! jrr = 2
real(kind=real64), dimension(klon,1,klev), intent(inout) :: prrs ! jrr = 3
real(kind=real64), dimension(klon,1,klev), intent(inout) :: pris ! jrr = 4
real(kind=real64), dimension(klon,1,klev), intent(inout) :: prss ! jrr = 5
real(kind=real64), dimension(klon,1,klev), intent(inout) :: prgs ! jrr = 6
! !
!
!*       0.2   declarations of local variables :
!
integer(kind=int32) :: jrr           ! loop index for the moist and scalar variables
integer(kind=int32) :: jlon, jlev
real(kind=real64) :: zt, ztwotstep
real(kind=real64), dimension(klon) :: zlv,zls,zcph
logical :: ll(klon)

real(kind=real64), dimension(klon,1,klev) :: zths   ! krr = 1
real(kind=real64), dimension(klon,1,klev) :: zrc    ! krr = 2
real(kind=real64), dimension(klon,1,klev) :: zrr    ! krr = 3
real(kind=real64), dimension(klon,1,klev) :: zri    ! krr = 4
real(kind=real64), dimension(klon,1,klev) :: zrs    ! krr = 5
real(kind=real64), dimension(klon,1,klev) :: zrg    ! krr = 6


real(kind=real64), dimension(klon,1,klev) :: zzz ! model layer height
real(kind=real64)  :: zmasstot                   ! total mass  for one water category ! including the negative values
real(kind=real64)  :: zmasspos                   ! total mass  for one water category ! after removing the negative values
real(kind=real64)  :: zratio                     ! zmasstot / zmasscor
real(kind=real64)  :: zcor(klon)                 ! for the correction of negative rv
!
real(kind=real64), dimension(klon,1) :: zsigqsat, zice_cld_wgt

!
!------------------------------------------------------------------------------
!*       2.     transformation into physical tendencies
!               ---------------------------------------
!
!
! personal comment:  tranfering these variables to the
!                    microphysical routines would save
!                    computing time

! well, getting rid of array syntax already saves a lot ;-) .rek.
!
!
!
!*       3.     remove negative values
!               ----------------------
!
!*       3.1    non local correction for precipitating species (rood 87)

if (cmicro == 'ice3') then

    ! rain
    if ( minval( prrs(kidia:kfdia,:,:)) < 0.0 ) then
        prrs(kidia:kfdia,:,:) = max( 0., prrs(kidia:kfdia,:,:) )
    end if

    ! snow
    if ( minval( prss(kidia:kfdia,:,:)) < 0.0 ) then
        prss(kidia:kfdia,:,:) = max( 0., prss(kidia:kfdia,:,:) )
    end if

    ! graupel
    if ( minval( prgs(kidia:kfdia,:,:)) < 0.0 ) then
            prcs(kidia:kfdia,:,:) = max( 0., prcs(kidia:kfdia,:,:) )
    end if

end if
!
!*       3.2    adjustement for liquid and solid cloud
!

ztwotstep=2.*ptstep

select case ( cmicro )
!
!
case('ice2','ice3','ice4')

    do jlev=1,klev
        do jlon=kidia,kfdia
            zt = ptht(jlon,1,jlev)*pexnref(jlon,1,jlev)
            zlv(jlon)=xlvtt +(xcpv-xcl) *(zt-xtt)
            zls(jlon)=xlstt +(xcpv-xci) *(zt-xtt)
            zcph(jlon)=xcpd +xcpv*2.*ptstep*prs(jlon,1,jlev,1)
        enddo

    do jlon=kidia,kfdia
        if (pris(jlon,1,jlev) < 0.) then
            prs(jlon,1,jlev,1) = prs(jlon,1,jlev,1) + pris(jlon,1,jlev)
            pths(jlon,1,jlev)  = pths(jlon,1,jlev)  - pris(jlon,1,jlev) * zls(jlon) / zcph(jlon) / pexnref(jlon,1,jlev)
            pris(jlon,1,jlev) = 0.
        endif
    enddo
!
!   cloud
    do jlon=kidia,kfdia
        if (prs(jlon,1,jlev,2) < 0.) then
            prs(jlon,1,jlev,1) = prs(jlon,1,jlev,1) + prs(jlon,1,jlev,2)
            pths(jlon,1,jlev)  = pths(jlon,1,jlev)  - prs(jlon,1,jlev,2) * zlv(jlon) / zcph(jlon) / pexnref(jlon,1,jlev)
            prs(jlon,1,jlev,2) = 0.
        endif
    enddo
!
! if rc or ri are positive, we can correct negative rv
!   cloud
    do jlon=kidia,kfdia
        ll(jlon) = (prs(jlon,1,jlev,1) <0.) .and. (prs(jlon,1,jlev,2)> 0.)
        if (ll(jlon)) then
            zcor(jlon)=min(-prs(jlon,1,jlev,1),prs(jlon,1,jlev,2))
        endif
    enddo
    do jlon=kidia,kfdia
        if (ll(jlon)) then
            prs(jlon,1,jlev,1) = prs(jlon,1,jlev,1) + zcor(jlon)
            pths(jlon,1,jlev)  = pths(jlon,1,jlev)  - zcor(jlon) * zlv(jlon) / zcph(jlon) / pexnref(jlon,1,jlev)
            prs(jlon,1,jlev,2) = prs(jlon,1,jlev,2) - zcor(jlon)
        endif
    enddo

!   ice
    if (krr > 3) then
        do jlon=kidia,kfdia
            ll(jlon) = (prs(jlon,1,jlev,1) < 0.).and.(pris(jlon,1,jlev) > 0.)
            if (ll(jlon)) then
                zcor(jlon)=min(-prs(jlon,1,jlev,1),pris(jlon,1,jlev))
            endif
        enddo
        do jlon=kidia,kfdia
            if (ll(jlon)) then
                prs(jlon,1,jlev,1) = prs(jlon,1,jlev,1) + zcor(jlon)
                pths(jlon,1,jlev)  = pths(jlon,1,jlev)  - zcor(jlon) * zls(jlon) / zcph(jlon) / pexnref(jlon,1,jlev)
                pris(jlon,1,jlev) = pris(jlon,1,jlev) - zcor(jlon)
            endif
        enddo
    endif

enddo ! jlev
!
end select
!
!
!*       3.3  store the budget terms
!            ----------------------

!*       9.     mixed-phase microphysical scheme (with 3 ice species)
!               -----------------------------------------------------
!

zths(kidia:kfdia,:,:) = pths(kidia:kfdia,:,:)*2.*ptstep
zrc(kidia:kfdia,:,:) = prc(kidia:kfdia,:,:)*2.*ptstep
zrr(kidia:kfdia,:,:) = prr(kidia:kfdia,:,:)*2.*ptstep
zri(kidia:kfdia,:,:) = prc(kidia:kfdia,:,:)*2.*ptstep
zrs(kidia:kfdia,:,:) = prc(kidia:kfdia,:,:)*2.*ptstep
zrg(kidia:kfdia,:,:) = prc(kidia:kfdia,:,:)*2.*ptstep


zzz(kidia:kfdia,:,:) =  pzzf(kidia:kfdia,:,:)
!
!
!
!-------------------------------------------------------------------------------
!
end subroutine aro_adjust
end module aro_adjust
