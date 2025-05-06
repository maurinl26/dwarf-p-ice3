module mode_cloud_fraction_split
    implicit none
contains
    subroutine cloud_fraction_1(nijt, nkt,            &
            &nkte, nktb,                &
            &nijb, nije,                &
            &zri, zrc,                  &
            &ptstep,                    &
            &pexnref, zcph,             &
            &zlv, zls,               &
            &prc,                       &
            &pri,                       &
            &pths, prvs, prcs, pris)

        implicit none

        integer, intent(in) :: nijt, nkt
        integer, intent(in) :: nkte, nktb
        integer, intent(in) :: nijb, nije

        real, dimension(nijt, nkt), intent(in) :: zrc, zri

        real, intent(in)   :: ptstep    ! double time step
        real, dimension(nijt, nkt), intent(in)    ::  pexnref ! reference exner function
        real, dimension(nijt, nkt), intent(in)    :: zcph
        real, dimension(nijt, nkt), intent(in)    :: zlv
        real, dimension(nijt, nkt), intent(in)    :: zls

        real, dimension(nijt, nkt), intent(in)    :: prc     ! cloud water m.r. to adjust
        real, dimension(nijt, nkt), intent(in)   ::  pri  ! cloud ice  m.r. to adjust

        real, dimension(nijt, nkt), intent(inout) :: pths    ! theta source
        real, dimension(nijt, nkt), intent(inout) :: prvs    ! water vapor m.r. source
        real, dimension(nijt, nkt), intent(inout) :: prcs    ! cloud water m.r. source
        real, dimension(nijt, nkt), intent(inout) ::  pris ! cloud ice  m.r. at t+1

        real :: zw1, zw2

        integer :: jk, jij

        do jk = nktb, nkte
            do jij = nijb, nije

                ! *       5.0   compute the variation of mixing ratio
                zw1 = (zrc(jij, jk) - prc(jij, jk)) / ptstep       ! pcon = ----------
                zw2 = (zri(jij, jk) - pri(jij, jk)) / ptstep       ! idem zw1 but for ri

                ! *       5.1   compute the sources
                if (zw1 < 0.0) then
                    zw1 = max(zw1, - prcs(jij, jk))
                else
                    zw1 = min(zw1, prvs(jij, jk))
                end if
                prvs(jij, jk) = prvs(jij, jk) - zw1
                prcs(jij, jk) = prcs(jij, jk) + zw1
                pths(jij, jk) = pths(jij, jk) + &
                    zw1 * zlv(jij, jk) / (zcph(jij, jk) * pexnref(jij, jk))

                if (zw2 < 0.0) then
                    zw2 = max(zw2, - pris(jij, jk))
                else
                    zw2 = min(zw2, prvs(jij, jk))
                end if
                prvs(jij, jk) = prvs(jij, jk) - zw2
                pris(jij, jk) = pris(jij, jk) + zw2
                pths(jij, jk) = pths(jij, jk) + &
                    zw2 * zls(jij, jk) / (zcph(jij, jk) * pexnref(jij, jk))

            end do
        end do

    end subroutine cloud_fraction_1

    subroutine cloud_fraction_2(nijt, nkt,                              &
            &nkte, nktb,                                   &
            &nijb, nije,                                   &
            &xcriautc, xcriauti, xacriauti, xbcriauti, xtt,&
            &csubg_mf_pdf,                                 &
            &lsubg_cond,                                   &
            &ptstep,                                       &
            &pexnref, prhodref, zcph,                      &
            &zlv, zls, zt,                                 &
            &pcf_mf, prc_mf, pri_mf,                       &
            &pths, prvs, prcs, pris,                       &
            &pcldfr,                                       &
            &phlc_hrc, phlc_hcf, phli_hri, phli_hcf)

        implicit none

        integer, intent(in) :: nijt, nkt
        integer, intent(in) :: nkte, nktb
        integer, intent(in) :: nijb, nije
        logical, intent(in) :: lsubg_cond
        real, intent(in) :: xcriautc, xcriauti, xacriauti, xbcriauti, xtt
        integer, intent(in) :: csubg_mf_pdf

        real, intent(in)   :: ptstep    ! double time step
        real, dimension(nijt, nkt), intent(in)    ::  pexnref ! reference exner function
        real, dimension(nijt, nkt), intent(in)    ::  prhodref
        real, dimension(nijt, nkt), intent(in)    :: zcph
        real, dimension(nijt, nkt), intent(in)    :: zlv
        real, dimension(nijt, nkt), intent(in)    :: zls
        real, dimension(nijt, nkt), intent(in)    :: zt

        real, dimension(nijt, nkt), intent(in)    :: pcf_mf   ! convective mass flux cloud fraction
        real, dimension(nijt, nkt), intent(in)    :: prc_mf   ! convective mass flux liquid mixing ratio
        real, dimension(nijt, nkt), intent(in)    :: pri_mf   ! convective mass flux ice mixing ratio

        real, dimension(nijt, nkt), intent(inout) :: pths     ! theta source
        real, dimension(nijt, nkt), intent(inout) :: prvs     ! water vapor m.r. source
        real, dimension(nijt, nkt), intent(inout) :: prcs     ! cloud water m.r. source
        real, dimension(nijt, nkt), intent(inout) ::  pris     ! cloud ice  m.r. at t+1

        real, dimension(nijt, nkt), intent(out)  ::  pcldfr   ! cloud fraction
        real, dimension(nijt, nkt), optional, intent(out)  ::  phlc_hrc
        real, dimension(nijt, nkt), optional, intent(out)  ::  phlc_hcf
        real, dimension(nijt, nkt), optional, intent(out)  ::  phli_hri
        real, dimension(nijt, nkt), optional, intent(out)  ::  phli_hcf

        real :: zw1, zw2, zhcf, zhr
        logical :: llnone, lltriangle
        real :: zcriaut


        integer :: jij, jk

        do jk = nkte, nktb

            if (.not. lsubg_cond) then
                do jij = nijb, nije
                    if ((prcs(jij, jk) + pris(jij, jk)) * ptstep > 1.e-12) then
                        pcldfr(jij, jk) = 1.
                    else
                        pcldfr(jij, jk) = 0.
                    end if
                end do
            else

                llnone = csubg_mf_pdf == 0
                lltriangle = csubg_mf_pdf == 1
                do jij = nijb, nije
                    zw1 = prc_mf(jij, jk) / ptstep
                    zw2 = pri_mf(jij, jk) / ptstep

                    if (zw1 + zw2 > prvs(jij, jk)) then
                        zw1 = zw1 * prvs(jij, jk) / (zw1 + zw2)
                        zw2 = prvs(jij, jk) - zw1
                    end if

                    pcldfr(jij, jk) = min(1., pcldfr(jij, jk) + pcf_mf(jij, jk))
                    prcs(jij, jk) = prcs(jij, jk) + zw1
                    pris(jij, jk) = pris(jij, jk) + zw2
                    prvs(jij, jk) = prvs(jij, jk) - (zw1 + zw2)
                    pths(jij, jk) = pths(jij, jk) + &
                        (zw1 * zlv(jij, jk) + zw2 * zls(jij, jk)) / (zcph(jij, jk) * pexnref(jij, jk))
                    !
                    ! cloud droplets
                    zcriaut = xcriautc / prhodref(jij, jk)
                    if (llnone) then
                        if (zw1 * ptstep > pcf_mf(jij, jk) * zcriaut) then
                            phlc_hrc(jij, jk) = phlc_hrc(jij, jk) + zw1 * ptstep
                            phlc_hcf(jij, jk) = min(1., phlc_hcf(jij, jk) + pcf_mf(jij, jk))
                        end if
                    else if (lltriangle) then
                        if (zw1 * ptstep > pcf_mf(jij, jk) * zcriaut) then
                            zhcf = 1.-.5 * (zcriaut * pcf_mf(jij, jk) / max(1.e-20, zw1 * ptstep)) ** 2
                            zhr = zw1 * ptstep - (zcriaut * pcf_mf(jij, jk)) ** 3 / &
                                &(3 * max(1.e-20, zw1 * ptstep) ** 2)
                        else if (2.* zw1 * ptstep <= pcf_mf(jij, jk) * zcriaut) then
                            zhcf = 0.
                            zhr = 0.
                        else
                            zhcf = (2.* zw1 * ptstep - zcriaut * pcf_mf(jij, jk)) ** 2 / &
                                &(2.* max(1.e-20, zw1 * ptstep) ** 2)
                            zhr = (4.* (zw1 * ptstep) ** 3 - 3.* zw1 * ptstep * (zcriaut * pcf_mf(jij, jk)) ** 2 + &
                                (zcriaut * pcf_mf(jij, jk)) ** 3) / &
                                &(3 * max(1.e-20, zw1 * ptstep) ** 2)
                        end if
                        zhcf = zhcf * pcf_mf(jij, jk) ! to retrieve the part of the grid cell
                        phlc_hcf(jij, jk) = min(1., phlc_hcf(jij, jk) + zhcf) ! total part of the grid cell that is precipitating
                        phlc_hrc(jij, jk) = phlc_hrc(jij, jk) + zhr
                    end if
                    ! cloud droplet

                    ! ice
                    zcriaut = min(xcriauti, 10 ** (xacriauti * (zt(jij, jk) - xtt) + xbcriauti))
                    if (llnone) then
                        if (zw2 * ptstep > pcf_mf(jij, jk) * zcriaut) then
                            phli_hri(jij, jk) = phli_hri(jij, jk) + zw2 * ptstep
                            phli_hcf(jij, jk) = min(1., phli_hcf(jij, jk) + pcf_mf(jij, jk))
                        end if
                    else if (lltriangle) then
                        if (zw2 * ptstep > pcf_mf(jij, jk) * zcriaut) then
                            zhcf = 1.-.5 * (zcriaut * pcf_mf(jij, jk) / (zw2 * ptstep)) ** 2
                            zhr = zw2 * ptstep - (zcriaut * pcf_mf(jij, jk)) ** 3 / (3 * (zw2 * ptstep) ** 2)
                        else if (2.* zw2 * ptstep <= pcf_mf(jij, jk) * zcriaut) then
                            zhcf = 0.
                            zhr = 0.
                        else
                            zhcf = (2.* zw2 * ptstep - zcriaut * pcf_mf(jij, jk)) ** 2 / (2.* (zw2 * ptstep) ** 2)
                            zhr = (4.* (zw2 * ptstep) ** 3 - 3.* zw2 * ptstep * (zcriaut * pcf_mf(jij, jk)) ** 2 + &
                                (zcriaut * pcf_mf(jij, jk)) ** 3) / (3 * (zw2 * ptstep) ** 2)
                        end if
                        zhcf = zhcf * pcf_mf(jij, jk) ! to retrieve the part of the grid cell
                        phli_hcf(jij, jk) = min(1., phli_hcf(jij, jk) + zhcf) ! total part of the grid cell that is precipitating
                        phli_hri(jij, jk) = phli_hri(jij, jk) + zhr
                    end if
                end do
            end if
        end do

    end subroutine cloud_fraction_2

end module mode_cloud_fraction_split
