! Created by  on 02/06/2025.

module mode_ice4_fast_rg

    implicit none
    contains

    subroutine rain_contact_freezing(ksize, &
            &ldcompute)

    integer, intent(in) :: ksize
    logical, dimension(ksize), intent(in) :: ldcompute

    integer :: jl

    DO JL=1, KSIZE
  IF(PRIT(JL)>I_RTMIN .AND. PRRT(JL)>R_RTMIN .AND. LDCOMPUTE(JL)) THEN
    IF(.NOT. LDSOFT) THEN
      PRICFRRG(JL) = XICFRR*PRIT(JL)                & ! RICFRRG
                                *PLBDAR(JL)**XEXICFRR    &
                                *PRHODREF(JL)**(-XCEXVT)
      PRRCFRIG(JL) = XRCFRI*PCIT(JL)                & ! RRCFRIG
                                * PLBDAR(JL)**XEXRCFRI    &
                                * PRHODREF(JL)**(-XCEXVT-1.)
      IF(PARAMI%LCRFLIMIT) THEN
        !Comparison between heat to be released (to freeze rain) and heat sink (rain and ice temperature change)
        !ZZW0D is the proportion of process that can take place
        ZZW0D=MAX(0., MIN(1., (PRICFRRG(JL)*XCI+PRRCFRIG(JL)*XCL)*(XTT-PT(JL)) / &
                              MAX(1.E-20, XLVTT*PRRCFRIG(JL))))
        PRRCFRIG(JL) = ZZW0D * PRRCFRIG(JL) !Part of rain that can be freezed
        PRICFRR(JL) = (1.-ZZW0D) * PRICFRRG(JL) !Part of collected pristine ice converted to rain
        PRICFRRG(JL) = ZZW0D * PRICFRRG(JL) !Part of collected pristine ice that lead to graupel
      ELSE
        PRICFRR(JL) = 0.
      ENDIF
    ENDIF
  ELSE
    PRICFRRG(JL)=0.
    PRRCFRIG(JL)=0.
    PRICFRR(JL)=0.
  ENDIF
ENDDO
!$acc end kernels
!
!
!*       6.3    compute the graupel growth
!
! Wet and dry collection of rc and ri on graupel
!$acc kernels
!$acc loop independent
DO JL=1, KSIZE
  IF(PRGT(JL)>G_RTMIN .AND. PRCT(JL)>C_RTMIN .AND. LDCOMPUTE(JL)) THEN
    IF(.NOT. LDSOFT) THEN
      PRG_TEND(JL, IRCDRYG)=PLBDAG(JL)**(ICED%XCXG-ICED%XDG-2.) * PRHODREF(JL)**(-ICED%XCEXVT)
      PRG_TEND(JL, IRCDRYG)=ICEP%XFCDRYG * PRCT(JL) * PRG_TEND(JL, IRCDRYG)
    ENDIF
  ELSE
    PRG_TEND(JL, IRCDRYG)=0.
  ENDIF

  IF(PRGT(JL)>G_RTMIN .AND. PRIT(JL)>I_RTMIN .AND. LDCOMPUTE(JL)) THEN
    IF(.NOT. LDSOFT) THEN
      PRG_TEND(JL, IRIDRYG)=PLBDAG(JL)**(XCXG-XDG-2.) * PRHODREF(JL)**(-XCEXVT)
      PRG_TEND(JL, IRIDRYG)=XFIDRYG*EXP(XCOLEXIG*(PT(JL)-XTT))*PRIT(JL)*PRG_TEND(JL, IRIDRYG)
      PRG_TEND(JL, IRIWETG)=PRG_TEND(JL, IRIDRYG) / (XCOLIG*EXP(XCOLEXIG*(PT(JL)-XTT)))
    ENDIF
  ELSE
    PRG_TEND(JL, IRIDRYG)=0.
    PRG_TEND(JL, IRIWETG)=0.
  ENDIF
ENDDO
!$acc end kernels

    end subroutine rain_contact_freezing

end module mode_ice4_fast_rg