! module mode_condensation
! implicit none
! contains
!     ######spl
    subroutine condensation(nijb, nije, nktb, nkte, nijt, nkt,  &
        &xrv, xrd, xalpi, xbetai, xgami, xalpw, xbetaw, xgamw,  &
        &osigmas, ocnd2, ouseri,                                &                                            
        &hcondens, hlambda3, lstatnw,                           &
        &ppabs, pt,                                             &              
        &prv_in, prv_out, prc_in, prc_out, pri_in, pri_out,     &
        &psigs, pcldfr, psigrc,                                 &
        &psigqsat,                                              &
        &plv, pls, pcph                                                                 &
    )
    USE ISO_FORTRAN_ENV, ONLY: REAL64, INT32 ! <- Get a float64 type.

    implicit none

    integer(kind=int32), intent(in) :: nijb, nije, nktb, nkte, nijt, nkt
    real(kind=real64), intent(in) :: xrv, xrd, xalpi, xbetai, xgami, xalpw, xbetaw, xgamw
    character(len=4),             intent(in)    :: hcondens
    character(len=*),             intent(in)    :: hlambda3 ! formulation for lambda3 coeff
    logical, intent(in) :: lstatnw
    real(kind=real64), dimension(nijt,nkt), intent(in)    :: ppabs  ! pressure (pa)
    real(kind=real64), dimension(nijt,nkt), intent(inout) :: pt     ! grid scale t  (k)
    real(kind=real64), dimension(nijt,nkt), intent(in)    :: prv_in ! grid scale water vapor mixing ratio (kg/kg) in input
    real(kind=real64), dimension(nijt,nkt), intent(out)   :: prv_out! grid scale water vapor mixing ratio (kg/kg) in output
    real(kind=real64), dimension(nijt,nkt), intent(in)    :: prc_in ! grid scale r_c mixing ratio (kg/kg) in input
    real(kind=real64), dimension(nijt,nkt), intent(out)   :: prc_out! grid scale r_c mixing ratio (kg/kg) in output
    real(kind=real64), dimension(nijt,nkt), intent(in)    :: pri_in ! grid scale r_i (kg/kg) in input
    real(kind=real64), dimension(nijt,nkt), intent(out)   :: pri_out! grid scale r_i (kg/kg) in output
    real(kind=real64), dimension(nijt,nkt), intent(in)    :: psigs  ! sigma_s from turbulence scheme
    real(kind=real64), dimension(nijt,nkt), intent(out)   :: pcldfr ! cloud fraction
    real(kind=real64), dimension(nijt,nkt), intent(out)   :: psigrc ! s r_c / sig_s^2

    logical, intent(in)                         :: ouseri ! logical switch to compute both liquid and solid condensate (ouseri=.true.)or only solid condensate (ouseri=.false.)
    logical, intent(in)                         :: osigmas! use present global sigma_s values or that from turbulence scheme
    logical, intent(in)                         :: ocnd2  ! logical switch to sparate liquid and ice more rigid (defalt value : .false.)
    real(kind=real64), dimension(nijt),       intent(in)    :: psigqsat ! use an extra "qsat" variance contribution (osigmas case) multiplied by psigqsat
    real(kind=real64), dimension(nijt,nkt), intent(in)    :: plv    ! latent heat l_v
    real(kind=real64), dimension(nijt,nkt), intent(in)    :: pls    ! latent heat l_s
    real(kind=real64), dimension(nijt,nkt), intent(in)    :: pcph   ! specific heat c_ph
!
!
!*       0.2   declarations of local variables :
!
integer :: jij, jk 
real(kind=real64), dimension(nijt,nkt) :: zrt     ! work arrays for t_l and total water mixing ratio
real(kind=real64) :: zlvs                                      ! thermodynamics
real(kind=real64), dimension(nijt) :: zpv, zpiv, zqsl, zqsi ! thermodynamics
real(kind=real64) :: zah
real(kind=real64), dimension(nijt) :: za, zb, zsbar, zsigma, zq1 ! related to computation of sig_s
real(kind=real64), dimension(nijt) :: zcond
real(kind=real64), dimension(nijt) :: zfrac           ! ice fraction
integer  :: inq1
real(kind=real64) :: zinc
! related to ocnd2 noise check :
real(kind=real64) :: zprifact
! end ocnd2

! lhgt_qs:
real(kind=real64) :: zdzfact
! lhgt_qs end
!
!
!*       0.3  definition of constants :
!
!-------------------------------------------------------------------------------

real(kind=real64), dimension(-22:11),parameter :: zsrc_1d = (/                         &
0.           ,  0.           ,  2.0094444e-04,   0.316670e-03,    &
4.9965648e-04,  0.785956e-03 ,  1.2341294e-03,   0.193327e-02,    &
3.0190963e-03,  0.470144e-02 ,  7.2950651e-03,   0.112759e-01,    &
1.7350994e-02,  0.265640e-01 ,  4.0427860e-02,   0.610997e-01,    &
9.1578111e-02,  0.135888e+00 ,  0.1991484    ,   0.230756e+00,    &
0.2850565    ,  0.375050e+00 ,  0.5000000    ,   0.691489e+00,    &
0.8413813    ,  0.933222e+00 ,  0.9772662    ,   0.993797e+00,    &
0.9986521    ,  0.999768e+00 ,  0.9999684    ,   0.999997e+00,    &
1.0000000    ,  1.000000     /)
!
!-------------------------------------------------------------------------------
pcldfr(:,:) = 0. ! initialize values
psigrc(:,:) = 0. ! initialize values
prv_out(:,:)= 0. ! initialize values
prc_out(:,:)= 0. ! initialize values
pri_out(:,:)= 0. ! initialize values
!-------------------------------------------------------------------------------
! store total water mixing ratio
do jk=nktb,nkte
    do jij=nijb,nije
        zrt(jij,jk)  = prv_in(jij,jk) + prc_in(jij,jk) + pri_in(jij,jk)*zprifact
    end do
end do
!-------------------------------------------------------------------------------
! preliminary calculations
! latent heat of vaporisation/sublimation
!-------------------------------------------------------------------------------
!
do jk=nktb,nkte
    if (.not. ocnd2) then
        ! latent heats
        ! saturated water vapor mixing ratio over liquid water and ice
        do jij=nijb,nije
            zpv(jij)  = min(exp( xalpw - xbetaw / pt(jij,jk) - xgamw * log( pt(jij,jk) ) ), .99*ppabs(jij,jk))
            zpiv(jij) = min(exp( xalpi - xbetai / pt(jij,jk) - xgami * log( pt(jij,jk) ) ), .99*ppabs(jij,jk))
        end do
    endif
    
    do jij=nijb,nije
        zqsl(jij)   = xrd / xrv * zpv(jij) / ( ppabs(jij,jk) - zpv(jij) )
        zqsi(jij)   = xrd / xrv * zpiv(jij) / ( ppabs(jij,jk) - zpiv(jij) )

       ! interpolate between liquid and solid as function of temperature
        zqsl(jij) = (1. - zfrac(jij)) * zqsl(jij) + zfrac(jij) * zqsi(jij)
        zlvs = (1. - zfrac(jij)) * plv(jij,jk) + &
        & zfrac(jij)      * pls(jij,jk)

        ! coefficients a and b
        zah  = zlvs * zqsl(jij) / ( xrv * pt(jij,jk)**2 ) * (xrv * zqsl(jij) / xrd + 1.)
        za(jij)   = 1. / ( 1. + zlvs/pcph(jij,jk) * zah )
        zb(jij)   = zah * za(jij)
        zsbar(jij) = za(jij) * ( zrt(jij,jk) - zqsl(jij) + &
        & zah * zlvs * (prc_in(jij,jk)+pri_in(jij,jk)*zprifact) / pcph(jij,jk))
    end do

    if ( osigmas ) then
        do jij=nijb,nije
            if (psigqsat(jij)/=0.) then
                zdzfact = 1.
                if (.not. lstatnw) then
                    zsigma(jij) = sqrt((2*psigs(jij,jk))**2 + (psigqsat(jij)*zqsl(jij)*za(jij))**2)
                endif
            else
                if (.not. lstatnw) then
                    zsigma(jij) = 2*psigs(jij,jk)
                endif
            end if
        end do
    end if

    do jij=nijb,nije
        zsigma(jij)= max( 1.e-10, zsigma(jij) )
        ! normalized saturation deficit
        zq1(jij)   = zsbar(jij)/zsigma(jij)
    end do

    if(hcondens == 'cb02')then
        do jij=nijb,nije
            !total condensate
            if (zq1(jij) > 0. .and. zq1(jij) <= 2) then
                zcond(jij) = min(exp(-1.)+.66*zq1(jij)+.086*zq1(jij)**2, 2.) ! we use the min function for continuity
            else if (zq1(jij) > 2.) then
                zcond(jij) = zq1(jij)
            else
                zcond(jij) = exp( 1.2*zq1(jij)-1. )
            endif
                zcond(jij) = zcond(jij) * zsigma(jij)

            !cloud fraction
            if (zcond(jij) < 1.e-12) then
                pcldfr(jij,jk) = 0.
            else
                pcldfr(jij,jk) = max( 0., min(1.,0.5+0.36*atan(1.55*zq1(jij))) )
            endif
            if (pcldfr(jij,jk)==0.) then
                zcond(jij)=0.
            endif

            inq1 = min( max(-22,floor(min(100., max(-100., 2*zq1(jij)))) ), 10)  !inner min/max prevents sigfpe when 2*zq1 does not fit into an int
            zinc = 2.*zq1(jij) - inq1

            psigrc(jij,jk) =  min(1.,(1.-zinc)*zsrc_1d(inq1)+zinc*zsrc_1d(inq1+1))
        end do
    end if !hcondens

    if(.not. ocnd2) then
        do jij=nijb,nije
            prc_out(jij,jk) = (1.-zfrac(jij)) * zcond(jij) ! liquid condensate
            pri_out(jij,jk) = zfrac(jij) * zcond(jij)   ! solid condensate
            pt(jij,jk) = pt(jij,jk) + ((prc_out(jij,jk)-prc_in(jij,jk))*plv(jij,jk) + &
                 &(pri_out(jij,jk)-pri_in(jij,jk))*pls(jij,jk)   ) &
               & /pcph(jij,jk)
            prv_out(jij,jk) = zrt(jij,jk) - prc_out(jij,jk) - pri_out(jij,jk)*zprifact
        end do
    end if ! end ocnd2
    if(hlambda3=='cb')then
        do jij=nijb,nije
            psigrc(jij,jk) = psigrc(jij,jk)* min( 3. , max(1.,1.-zq1(jij)) )
        end do
    end if
end do

end subroutine condensation
! end module mode_condensation
