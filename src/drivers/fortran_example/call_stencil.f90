program example
use callpy_mod

implicit none

real(8) :: a(10)
real(8) :: b(10)
real(8) :: c(10)

a = 1.0
b = 1.0
c = 0.0

call set_state("a", a)
call set_state("b", b)
call set_state("c", c)
call call_stencil("multiply_ab2c")
! read any changes from "a" back into a.
call get_state("c", c)

end program example