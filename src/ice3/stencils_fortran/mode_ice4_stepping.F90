! Created by  on 09/05/2025.

module mode_ice4_stepping
   implicit none
   contains

   subroutine ice4_stepping_heat(kmicro, &
           &xcpd, xcpv, xcl, xci, &
           &xtt, xlvtt, xlstt, xmnh_tiny, &
           &xmrstep, &
           &pexn, ptht, &
           &prvt, prct, prrt, prit, prsrt, prgt)

      integer, intent(in) :: kmicro
      real, intent(in) :: xcpd, xcpv, xcl, xci
      real, intent(in) :: xtt, xlvtt, xlstt
      real, intent(in) :: xmrstep, xmnh_tiny
      real, dimension(kmicro), intent(in) :: pexn, ptht
      real, dimension(kmicro), intent(in) :: prvt, prct, prrt
      real, dimension(kmicro), intent(out) :: zzt, zlsfact, zlvfact
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

    subroutine time_integration(kmicro, &
            &ptstep, lfeedbackt, &
            &xtstep_ts, xtt, &
            &llcompute, zmaxtime, ztime, &
            &ptht, ath, bth)

    integer :: jl

        DO JL=1, KMICRO
      IF(LLCOMPUTE(JL)) THEN
        ZMAXTIME(JL)=(PTSTEP-ZTIME(JL)) ! Remaining time until the end of the timestep
      ELSE
        ZMAXTIME(JL)=0.
      ENDIF
    ENDDO
    !We need to adjust tendencies when temperature reaches 0
    IF(LFEEDBACKT) THEN
      DO JL=1, KMICRO
        !Is ZBTH(:) enough to change temperature sign?
        ZX=XTT/PEXN(JL)
        IF ((PTHT(JL) - ZX) * (PTHT(JL) + ZBTH(JL) - ZX) < 0.) THEN
          ZMAXTIME(JL)=0.
        ENDIF
        !Can ZATH(:) make temperature change of sign?
        IF (ABS(ZATH(JL)) > 1.E-20 ) THEN
          ZTIME_THRESHOLD=(ZX - ZBTH(JL) - PTHT(JL))/ZATH(JL)
          IF (ZTIME_THRESHOLD > 0.) THEN
            ZMAXTIME(JL)=MIN(ZMAXTIME(JL), ZTIME_THRESHOLD)
          ENDIF
        ENDIF
      ENDDO
    ENDIF

    !We need to adjust tendencies when a species disappears
    !When a species is missing, only the external tendencies can be negative (and we must keep track of it)
    DO JV=1, KRR
      DO JL=1, KMICRO
        IF (ZA(JL, JV) < -1.E-20 .AND. PRT(JL, JV) > ZRSMIN(JV)) THEN
          ZMAXTIME(JL)=MIN(ZMAXTIME(JL), -(ZB(JL, JV)+PRT(JL, JV))/ZA(JL, JV))
          ZMAXTIME(JL)=MAX(ZMAXTIME(JL), XMNH_TINY) !to prevent rounding errors
        ENDIF
      ENDDO
    ENDDO
    !We stop when the end of the timestep is reached
    DO JL=1, KMICRO
      IF (ZTIME(JL)+ZMAXTIME(JL) >= PTSTEP) THEN
        LLCOMPUTE(JL)=.FALSE.
      ENDIF
    ENDDO
    !We must recompute tendencies when the end of the sub-timestep is reached
    IF (XTSTEP_TS/=0.) THEN
      DO JL=1, KMICRO
        IF ((IITER(JL) < INB_ITER_MAX) .AND. (ZTIME(JL)+ZMAXTIME(JL) > ZTIME_LASTCALL(JL)+ZTSTEP)) THEN
          ZMAXTIME(JL)=ZTIME_LASTCALL(JL)-ZTIME(JL)+ZTSTEP
          LLCOMPUTE(JL)=.FALSE.
        ENDIF
      ENDDO
    ENDIF

    !We must recompute tendencies when the maximum allowed change is reached
    !When a species is missing, only the external tendencies can be active and we do not want to recompute
    !the microphysical tendencies when external tendencies are negative (results won't change because species was already missing)
    IF (XMRSTEP/=0.) THEN
      IF (LL_ANY_ITER) THEN
        ! In this case we need to remember the initial mixing ratios used to compute the tendencies
        ! because when mixing ratio has evolved more than a threshold, we must re-compute tendencies
        ! Thus, at first iteration (ie when LLCPZ0RT=.TRUE.) we copy PRT into Z0RT
        DO JV=1,KRR
          IF (LLCPZ0RT) THEN
            Z0RT(1:KMICRO, JV)=PRT(1:KMICRO, JV)
          ENDIF
          DO JL=1, KMICRO
            IF (IITER(JL)<INB_ITER_MAX .AND. ABS(ZA(JL,JV))>1.E-20) THEN
              ZTIME_THRESHOLD1D(JL)=(SIGN(1., ZA(JL, JV))*XMRSTEP+ &
                                    &Z0RT(JL, JV)-PRT(JL, JV)-ZB(JL, JV))/ZA(JL, JV)
            ELSE
              ZTIME_THRESHOLD1D(JL)=-1.
            ENDIF
          ENDDO
          DO JL=1, KMICRO
            IF (ZTIME_THRESHOLD1D(JL)>=0 .AND. ZTIME_THRESHOLD1D(JL)<ZMAXTIME(JL) .AND. &
               &(PRT(JL, JV)>ZRSMIN(JV) .OR. ZA(JL, JV)>0.)) THEN
              ZMAXTIME(JL)=MIN(ZMAXTIME(JL), ZTIME_THRESHOLD1D(JL))
              LLCOMPUTE(JL)=.FALSE.
            ENDIF
          ENDDO
          IF (JV == 1) THEN
            DO JL=1, KMICRO
              ZMAXB(JL)=ABS(ZB(JL, JV))
            ENDDO
          ELSE
            DO JL=1, KMICRO
              ZMAXB(JL)=MAX(ZMAXB(JL), ABS(ZB(JL, JV)))
            ENDDO
          ENDIF
        ENDDO
        LLCPZ0RT=.FALSE.
        DO JL=1, KMICRO
          IF (IITER(JL)<INB_ITER_MAX .AND. ZMAXB(JL)>XMRSTEP) THEN
            ZMAXTIME(JL)=0.
            LLCOMPUTE(JL)=.FALSE.
          ENDIF
        ENDDO
      ENDIF ! LL_ANY_ITER
    ENDIF ! XMRSTEP/=0.

    end subroutine time_integration

end module mode_ice4_stepping
