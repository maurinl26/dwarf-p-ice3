! Created by  on 09/05/2025.

module mode_ice4_stepping
   implicit none
   contains

   subroutine ice4_stepping_heat(kmicro, &
           &xcpd, xcpv, xcl, xci, &
           &xtt, xlvtt, xlstt, &
           &pexn, ptht, &
           &prvt, prct, prrt, prit, prsrt, prgt)

      integer, intent(in) :: kmicro
      real, intent(in) :: xcpd, xcpv, xcl, xci
      real, intent(in) :: xtt, xlvtt, xlstt
      real, dimension(kmicro), intent(in) :: pexn, ptht
      real, dimension(kmicro), intent(in) :: prvt, prct, prrt
      real, dimension(kmicro) :: zsum2

      integer :: jl

    DO JV=IRI+1,KRR
      DO JL=1, KMICRO
        ZSUM2(JL)=PRIT(JL)+PRST(JL)+PRGT(JL)
      ENDDO
    ENDDO

      DO JL=1, KMICRO
      ZDEVIDE=(XCPD + XCPV*PRVT(JL) + XCL*(PRCT(JL)+PRRT(JL)) + XCI*ZSUM2(JL)) * PEXN(JL)
      ZZT(JL) = PTHT(JL) * PEXN(JL)
      ZLSFACT(JL)=(XLSTT+(XCPV-XCI)*(ZZT(JL)-XTT)) / ZDEVIDE
      ZLVFACT(JL)=(XLVTT+(XCPV-XCL)*(ZZT(JL)-XTT)) / ZDEVIDE
    ENDDO

   end subroutine ice4_stepping_heat

end module mode_ice4_stepping
