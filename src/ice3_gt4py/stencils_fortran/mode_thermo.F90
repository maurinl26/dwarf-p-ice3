
module mode_thermo
    implicit none
    contains

    subroutine latent_heat(nkt, nijt, nktb, nkte, nijb, nije, &
        &xlvtt, xlstt, xcpv, xci, xcl, xtt, xcpd, krr, &
        &prv_in, prc_in, pri_in, prr, prs, prg,&
        &pth, pexn,&
        &zt, zls, zlv, zcph)

    implicit none

    integer, intent(in) :: nktb, nkte, nkt, nijb, nije, nijt, krr
    real, intent(in) :: xlvtt, xlstt, xcl, xci, xtt, xcpv, xcpd

    real, dimension(nijt, nkt), intent(in) :: pth
    real, dimension(nijt, nkt), intent(in) :: pexn
    real, dimension(nijt, nkt), intent(in) :: prv_in
    real, dimension(nijt, nkt), intent(in) :: prc_in
    real, dimension(nijt, nkt), intent(in) :: pri_in
    real, dimension(nijt, nkt), intent(in) :: prr
    real, dimension(nijt, nkt), intent(in) :: prs
    real, dimension(nijt, nkt), intent(in) :: prg

    real, dimension(nijt, nkt), intent(out) :: zt
    real, dimension(nijt, nkt), intent(out) :: zlv
    real, dimension(nijt, nkt), intent(out) :: zls
    real, dimension(nijt, nkt), intent(out) :: zcph

    integer :: jk, jij

  do jk=nktb,nkte
    do jij=nijb,nije
      zt(jij,jk) = pth(jij,jk) * pexn(jij,jk)
      zlv(jij,jk) = xlvtt + ( xcpv - xcl ) * ( zt(jij,jk) - xtt )
      zls(jij,jk) = xlstt + ( xcpv - xci ) * ( zt(jij,jk) - xtt )
    enddo
  enddo

  do jk=nktb,nkte
    do jij=nijb,nije
      select case(krr)
        case(6)
          zcph(jij,jk) = xcpd + xcpv * prv_in(jij,jk)                             &
                                  + xcl  * (prc_in(jij,jk) + prr(jij,jk))             &
                                  + xci  * (pri_in(jij,jk) + prs(jij,jk) + prg(jij,jk))
        case(5)
          zcph(jij,jk) = xcpd + xcpv * prv_in(jij,jk)                             &
                                  + xcl  * (prc_in(jij,jk) + prr(jij,jk))             &
                                  + xci  * (pri_in(jij,jk) + prs(jij,jk))
        case(3)
          zcph(jij,jk) = xcpd + xcpv * prv_in(jij,jk)               &
                                  + xcl  * (prc_in(jij,jk) + prr(jij,jk))
        case(2)
          zcph(jij,jk) = xcpd + xcpv * prv_in(jij,jk) &
                                  + xcl  * prc_in(jij,jk)
      end select
    enddo
  enddo

end subroutine latent_heat
end module mode_thermo
