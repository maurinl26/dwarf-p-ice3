
module mode_thermo
    implicit none
    contains

    subroutine latent_heat(nkt, nijt, nktb, nkte, nijb, nije, &
        &xlvtt, xlstt, xcpv, xci, xcl, xtt, xcpd, krr, &
        &prv_in, prc_in, pri_in, prr, prs, prg,&
        &pth, pexn,&
        &zt, zls, zlv, zcph)

    use iso_fortran_env, only: real64, int64
    implicit none

    integer(kind=int64), intent(in) :: nktb, nkte, nijb, nije, nkt, nijt, krr
    real(kind=real64), intent(in) :: xlvtt, xlstt, xcl, xci, xtt, xcpv, xcpd

    real(kind=real64), dimension(nijt, nkt), intent(in) :: pth
    real(kind=real64), dimension(nijt, nkt), intent(in) :: pexn
    real(kind=real64), dimension(nijt, nkt), intent(in) :: prv_in
    real(kind=real64), dimension(nijt, nkt), intent(in) :: prc_in
    real(kind=real64), dimension(nijt, nkt), intent(in) :: pri_in
    real(kind=real64), dimension(nijt, nkt), intent(in) :: prr
    real(kind=real64), dimension(nijt, nkt), intent(in) :: prs
    real(kind=real64), dimension(nijt, nkt), intent(in) :: prg

    real(kind=real64), dimension(nijt, nkt), intent(out) :: zt
    real(kind=real64), dimension(nijt, nkt), intent(out) :: zlv
    real(kind=real64), dimension(nijt, nkt), intent(out) :: zls
    real(kind=real64), dimension(nijt, nkt), intent(out) :: zcph

    integer(kind=int64) jk, jij


print *, "xlvtt : ", xlvtt
print *, "xlstt : ", xlstt
print *, "xcl : ", xcl
print *, "xci : ", xci
print *, "xtt : ", xtt
print *, "xcpv : ", xcpv
print *, "xcpd : ", xcpd

do jk=nktb, nkte
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
