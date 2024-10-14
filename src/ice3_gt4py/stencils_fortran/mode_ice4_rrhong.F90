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

    use iso_fortran_env, only: real64, int32
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
    !*      0. declarations
    !          ------------
    !
    implicit none
    !
    !*       0.1   declarations of dummy arguments :
    !
    real(kind=real64), intent(in) :: xtt
    real(kind=real64), intent(in) :: r_rtmin
    logical, intent(in) :: lfeedbackt
    integer(kind=INT32), intent(in) :: kproma, ksize
    logical, dimension(kproma),    intent(in)    :: ldcompute
    real(kind=real64), dimension(kproma),       intent(in)    :: pexn     ! exner function
    real(kind=real64), dimension(kproma),       intent(in)    :: plvfact  ! l_v/(pi_ref*c_ph)
    real(kind=real64), dimension(kproma),       intent(in)    :: plsfact  ! l_s/(pi_ref*c_ph)
    real(kind=real64), dimension(kproma),       intent(in)    :: pt       ! temperature
    real(kind=real64), dimension(kproma),       intent(in)    :: prrt     ! rain water m.r. at t
    real(kind=real64), dimension(kproma),       intent(in)    :: ptht     ! theta at t
    real(kind=real64), dimension(:),            intent(out)   :: prrhong_mr ! mixing ratio change due to spontaneous freezing
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
