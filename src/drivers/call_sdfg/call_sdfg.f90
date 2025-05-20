module SDFGWrapper
   use, intrinsic :: iso_c_binding
   implicit none

   interface
          ! from add.h
         type(c_funptr) function c_dace_init_add(I, J, K) bind(c, name='__dace_init_add')
            use, intrinsic :: iso_c_binding, only: c_int, c_funptr
            integer(c_int), intent(in) :: I
            integer(c_int), intent(in) :: J
            integer(c_int), intent(in) :: K
         end function c_dace_init_add

         subroutine c_program_add(handle, c, a, b, I, J, K) bind(c, name='__program_add')
            use, intrinsic :: iso_c_binding, only: c_funptr, c_ptr, c_float, c_int
            type(c_funptr), intent(in) :: handle
            integer(c_int), value :: I
            integer(c_int), value :: J
            integer(c_int), value :: K
            real(c_float), dimension(I, J, K), intent(in) :: c
            real(c_float), dimension(I, J, K), intent(in) :: a
            real(c_float), dimension(I, J, K), intent(in) :: b

        end subroutine c_program_add

        subroutine c_dace_exit_add(handle, err) bind(c, name='__dace_exit_add')
            use, intrinsic :: iso_c_binding, only: c_funptr, c_int
            type(c_funptr), intent(in) :: handle
            integer(c_int), intent(out) :: err
        end subroutine c_dace_exit_add
   end interface

contains

   subroutine call_sdfg(I, J, K)
      use iso_c_binding, only: c_associated, c_loc, c_ptr, c_int, c_float, c_funptr

      integer(c_int), intent(in) :: I
      integer(c_int), intent(in) :: J
      integer(c_int), intent(in) :: K

      real(c_float), dimension(I, J, K) :: a
      real(c_float), dimension(I, J, K) :: b
      real(c_float), dimension(I, J, K) :: c

      type(c_funptr) :: handle
      integer :: err

      a(:, :, :) = 1.0
      b(:, :, :) = 1.0
      c(:, :, :) = 0.0

      print *, "c:", sum(c)/(I+J+K)

      handle = c_dace_init_add(I, J, K)
      call c_program_add(handle, c, a, b, I, J, K)
!      call c_dace_exit_add(handle, err)
      print *, "handle :", handle

      print *, "error :", err
      print *, "c:", sum(c)/(I * J * K)

   end subroutine call_sdfg

end module SDFGWrapper

