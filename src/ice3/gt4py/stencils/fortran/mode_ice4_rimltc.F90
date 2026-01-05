module mode_ice4_rimltc
    implicit none
    contains
    
    subroutine ice4_rimltc(kproma, ksize, &
                        &xtt, lfeedbackt, &
                        &ldcompute, &
                           &pexn, plvfact, plsfact, &
                           &pt, &
                           &ptht, prit, &
                           &primltc_mr)
    !
    implicit none
    !
    !*       0.1   declarations of dummy arguments :
    real, intent(in) :: xtt
    logical, intent(in) :: lfeedbackt


    integer,                      intent(in)    :: kproma, ksize
    logical, dimension(kproma),    intent(in)    :: ldcompute
    real, dimension(kproma),       intent(in)    :: pexn     ! exner function
    real, dimension(kproma),       intent(in)    :: plvfact  ! l_v/(pi_ref*c_ph)
    real, dimension(kproma),       intent(in)    :: plsfact  ! l_s/(pi_ref*c_ph)
    real, dimension(kproma),       intent(in)    :: pt       ! temperature
    real, dimension(kproma),       intent(in)    :: ptht     ! theta at t
    real, dimension(kproma),       intent(in)    :: prit     ! cloud ice at t
    real, dimension(kproma),       intent(out)   :: primltc_mr ! mixing ratio change due to cloud ice melting
    !
    !*       0.2  declaration of local variables
    !
    ! real(kind=jphook) :: zhook_handle
    integer :: jl
    !
    !-------------------------------------------------------------------------------
    ! if (lhook) call dr_hook('ice4_rimltc',0,zhook_handle)
    !
    !*       7.1    cloud ice melting
    !
    do jl=1, ksize
      if(prit(jl)>0. .and. pt(jl)>xtt .and. ldcompute(jl)) then
        primltc_mr(jl)=prit(jl)
        if(lfeedbackt) then
          !limitation due to 0 crossing of temperature
          primltc_mr(jl)=min(primltc_mr(jl), max(0., (ptht(jl)-xtt/pexn(jl)) / (plsfact(jl)-plvfact(jl))))
        endif
      else
        primltc_mr(jl)=0.
      endif
    enddo    
    !
    end subroutine ice4_rimltc
    end module mode_ice4_rimltc
    