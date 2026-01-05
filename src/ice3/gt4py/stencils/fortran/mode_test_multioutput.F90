module mode_test_multioutput

    implicit none
    contains

subroutine multioutput(nijt, nkt, a, b, c)

    use iso_fortran_env, only: real64, int64
    implicit none

    integer(kind=int64), intent(in) :: nijt, nkt
    real(kind=real64), dimension(nijt, nkt), intent(in) ::  a
    real(kind=real64), dimension(nijt, nkt), intent(out) :: b, c

    integer(kind=real64) :: jij, jk

    do jij=1, nijt
        do jk=1, nkt
            b = 2.0 * a
            c = 3.0 * a
        end do
    end do

end subroutine multioutput

end module mode_test_multioutput