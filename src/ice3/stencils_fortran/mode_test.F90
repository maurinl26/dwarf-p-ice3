module mode_test

    implicit none
    contains
    subroutine multiply_ab2c(nijt, nkt, a, b, c)

        use iso_fortran_env, only : real64, int64
        implicit none

        integer(kind=int64), intent(in) :: nijt, nkt
        real(kind=real64), dimension(nijt, nkt), intent(in) :: a
        real(kind=real64), dimension(nijt, nkt), intent(in) :: b
        real(kind=real64), dimension(nijt, nkt), intent(out) :: c

        integer(kind=real64) :: jij, jk

        do jij=1, nijt
            do jk = 1,  nkt
                c = a * b
            end do
        end do

    end subroutine multiply_ab2c

    subroutine double_a(nijt, nkt, a, c)

        use iso_fortran_env, only: real64, int64
        implicit none

        integer(kind=int64), intent(in) :: nijt, nkt
        real(kind=real64), dimension(nijt, nkt), intent(in) :: a
        real(kind=real64), dimension(nijt, nkt), intent(out) :: c

        integer(kind=real64) :: jij, jk

        do jij=1, nijt
            do  jk = 1, nkt
                c = 2.0 * a
            end do
        end do
    
    end subroutine double_a

    subroutine mutliply_oned_array(nijkt, a, c)

        use iso_fortran_env, only: real64, int64
        implicit none

        integer(kind=int64), intent(in) :: nijkt
        real(kind=real64), dimension(nijkt), intent(in) :: a
        real(kind=real64), dimension(nijkt), intent(out) :: c

        integer(kind=int64) :: jij

        do jij=1, nijkt
            c = 2.0 * a
        end do

    end subroutine mutliply_oned_array

end module mode_test

