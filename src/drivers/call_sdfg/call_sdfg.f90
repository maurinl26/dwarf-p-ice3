module SDFGWrapper
   use, intrinsic :: iso_c_binding
   implicit none

   interface
         ! from add_main.cpp
         integer(c_int) function main_c() bind(c, name='main_driver')
            use, intrinsic :: iso_c_binding
         end function main_c
   end interface

contains

   subroutine call_sdfg(I, J, K)

      integer, intent(in) :: I
      integer, intent(in) :: J
      integer, intent(in) :: K

      real(8), dimension(I, J, K) :: a
      real(8), dimension(I, J, K) :: b
      real(8), dimension(I, J, K) :: c

      integer :: error



      a(:, :, :) = 1.0
      b(:, :, :) = 1.0
      c(:, :, :) = 0.0


      error =  main_c()

      print *, "c:", sum(c)/(I + J + K)

   end subroutine call_sdfg

end module SDFGWrapper

