!MNH_LIC Copyright 1994-2021 CNRS, Meteo-France and Universite Paul Sabatier
!MNH_LIC This is part of the Meso-NH software governed by the CeCILL-C licence
!MNH_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt
!MNH_LIC for details. version 1.
!-----------------------------------------------------------------
module mode_ice4_rrhong
    implicit none
    contains
    subroutine ice4_rrhong(xtt, r_rtmin, lfeedbackt, kproma, ksize, ldcompute, &
                           &pexn, plvfact, plsfact, &
                           &pt,   prrt, &
                           &ptht, &
                           &prrhong_mr)
    !!
    !!**  purpose
    !!    -------
    !!      computes the rrhong process
    !!
    !!    author
    !!    ------
    !!      s. riette from the splitting of rain_ice source code (nov. 2014)
    !!
    !!    modifications
    !!    -------------
    !!
    !
    !
    !*      0. declarations
    !          ------------
    !
    ! use modd_cst,            only: cst_t
    ! use modd_param_ice_n,      only: param_ice_t
    ! use modd_rain_ice_descr_n, only: rain_ice_descr_t
    ! use yomhook , only : lhook, dr_hook, jphook
    !
    implicit none
    !
    !*       0.1   declarations of dummy arguments :
    !
    real, intent(in) :: xtt
    real, intent(in) :: r_rtmin
    logical, intent(in) :: lfeedbackt
    ! type(cst_t),              intent(in)    :: cst
    ! type(param_ice_t),        intent(in)    :: parami
    ! type(rain_ice_descr_t),   intent(in)    :: iced
    integer, intent(in) :: kproma, ksize
    logical, dimension(KPROMA),    intent(in)    :: ldcompute
    real, dimension(KPROMA),       intent(in)    :: pexn     ! exner function
    real, dimension(KPROMA),       intent(in)    :: plvfact  ! l_v/(pi_ref*c_ph)
    real, dimension(KPROMA),       intent(in)    :: plsfact  ! l_s/(pi_ref*c_ph)
    real, dimension(KPROMA),       intent(in)    :: pt       ! temperature
    real, dimension(KPROMA),       intent(in)    :: prrt     ! rain water m.r. at t
    real, dimension(KPROMA),       intent(in)    :: ptht     ! theta at t
    real, dimension(:),       intent(out)   :: prrhong_mr ! mixing ratio change due to spontaneous freezing
    !
    !*       0.2  declaration of local variables
    !
    integer :: jl
    !
    !-------------------------------------------------------------------------------    !
    !*       3.3     compute the spontaneous freezing source: rrhong
    !
    do jl=1, ksize
      if(pt(jl)<xtt-35.0 .and. prrt(jl)>r_rtmin .and. ldcompute(jl)) then
        prrhong_mr(jl)=prrt(jl)
        if(lfeedbackt) then
          !limitation due to -35 crossing of temperature
          prrhong_mr(jl)=min(prrhong_mr(jl), max(0., ((xtt-35.)/pexn(jl)-ptht(jl))/(plsfact(jl)-plvfact(jl))))
        endif
      else
        prrhong_mr(jl)=0.
      endif
    enddo
    !
    end subroutine ice4_rrhong
    end module mode_ice4_rrhong
