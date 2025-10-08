!MNH_LIC Copyright 1996-2021 CNRS, Meteo-France and Universite Paul Sabatier
!MNH_LIC This is part of the Meso-NH software governed by the CeCILL-C licence
!MNH_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt
!MNH_LIC for details. version 1.
!-----------------------------------------------------------------
!     ##########################################################################
subroutine ice_adjust (d_nijb, d_nije, d_nijt, d_nkb, d_nke, d_nkl, d_nkt, d_nktb, d_nkte, cst_xalpi, cst_xalpw, cst_xbetai,  &
& cst_xbetaw, cst_xci, cst_xcl, cst_xcpd, cst_xcpv, cst_xepsilo, cst_xg, cst_xgami, cst_xgamw, cst_xlstt, cst_xlvtt, cst_xpi,  &
& cst_xrd, cst_xrv, cst_xtt, icep_xacriauti, icep_xbcriauti, icep_xcriautc, icep_xcriauti, icep_xfrmin, nebn_ccondens,  &
& nebn_cfrac_ice_adjust, nebn_clambda3, nebn_lcondborn, nebn_lhgt_qs, nebn_lsigmas, nebn_lstatnw, nebn_lsubg_cond,  &
& nebn_xtmaxmix, nebn_xtminmix, parami_csubg_mf_pdf, parami_locnd2, krr, hbuname, ptstep, psigqsat, prhodj, pexnref,  &
& prhodref, psigs, lmfconv, pmfconv, ppabst, pzz, pexn, pcf_mf, prc_mf, pri_mf, pweight_mf_cloud, picldfr, pwcldfr, pssio,  &
& pssiu, pifr, prv, prc, prvs, prcs, pth, pths, ocompute_src, psrcs, pcldfr, prr, pri, pris, prs, prg, pice_cld_wgt, prh,  &
& pout_rv, pout_rc, pout_ri, pout_th, phlc_hrc, phlc_hcf, phli_hri, &
        &phli_hcf, phlc_hrc_mf, phlc_hcf_mf, phli_hri_mf, phli_hcf_mf)

  !     #############################################TURBN############################
  !
  !!****  *ICE_ADJUST* -  compute the ajustment of water vapor in mixed-phase
  !!                      clouds
  !!
  !!    PURPOSE
  !!    -------
  !!    The purpose of this routine is to compute the fast microphysical sources
  !!    through a saturation ajustement procedure in case of mixed-phase clouds.
  !!
  !!
  !!**  METHOD
  !!    ------
  !!    Langlois, Tellus, 1973 for the cloudless version.
  !!    When cloud water is taken into account, refer to book 1 of the
  !!    documentation.
  !!
  !!
  !!
  !!    EXTERNAL
  !!    --------
  !!      None
  !!
  !!
  !!    IMPLICIT ARGUMENTS
  !!    ------------------
  !!      Module MODD_CST
  !!         XP00               ! Reference pressure
  !!         XMD,XMV            ! Molar mass of dry air and molar mass of vapor
  !!         XRD,XRV            ! Gaz constant for dry air, gaz constant for vapor
  !!         XCPD,XCPV          ! Cpd (dry air), Cpv (vapor)
  !!         XCL                ! Cl (liquid)
  !!         XCI                ! Ci (ice)
  !!         XTT                ! Triple point temperature
  !!         XLVTT              ! Vaporization heat constant
  !!         XLSTT              ! Sublimation  heat constant
  !!         XALPW,XBETAW,XGAMW ! Constants for saturation vapor over liquid
  !!                            !  pressure  function
  !!         XALPI,XBETAI,XGAMI ! Constants for saturation vapor over ice
  !!                            !  pressure  function
  !!      Module  MODD_CONF
  !!         CCONF
  !!      Module MODD_BUDGET:
  !!         NBUMOD
  !!         CBUTYPE
  !!         LBU_RTH
  !!         LBU_RRV
  !!         LBU_RRC
  !!         LBU_RRI
  !!
  !!
  !!    REFERENCE
  !!    ---------
  !!      Book 1 and Book2 of documentation ( routine ICE_ADJUST )
  !!      Langlois, Tellus, 1973
  !!
  !!    AUTHOR
  !!    ------
  !!      J.-P. Pinty    * Laboratoire d'Aerologie*
  !!
  !!
  !!    MODIFICATIONS
  !!    -------------
  !!      Original    06/12/96
  !!      M. Tomasini 27/11/00 Change CND and DEP fct of the T instead of rc and ri
  !!                           Avoid the sub- and super-saturation before the ajustment
  !!                           Avoid rc>0 if T<T00 before the ajustment
  !!      P Bechtold 12/02/02  change subgrid condensation
  !!      JP Pinty   29/11/02  add ICE2 and IC4 cases
  !!      (P. Jabouille) 27/05/04 safety test for case where esw/i(T)> pabs (~Z>40km)
  !!      J.Pergaud and S.Malardel Add EDKF case
  !!      S. Riette ice for EDKF
  !!      2012-02 Y. Seity,  add possibility to run with reversed vertical levels
  !!      J.Escobar : 15/09/2015 : WENO5 & JPHEXT <> 1
  !!      2016-07 S. Riette: adjustement is now realized on state variables (PRV, PRC, PRI, PTH)
  !!                         whereas tendencies are still applied on S variables.
  !!                         This modification allows to call ice_adjust on T variable
  !!                         or to call it on S variables
  !!      2016-11 S. Riette: all-or-nothing adjustment now uses condensation
  !  P. Wautelet 05/2016-04/2018: new data structures and calls for I/O
  !!      2018-02 K.I.Ivarsson : More outputs for OCND2 option
  !  P. Wautelet    02/2020: use the new data structures and subroutines for budgets
  !!      2020-12 U. Andrae : Introduce SPP for HARMONIE-AROME
  !!     R. El Khatib 24-Aug-2021 Optimizations
  !!     R. El Khatib 24-Oct-2023 Re-vectorize ;-)
  !!     A. Marcel Jan 2025: bi-Gaussian PDF and associated subgrid precipitation
  !!      A. Marcel Jan 2025: relaxation of the small fraction assumption
  !!
  !-------------------------------------------------------------------------------
  !
  !*       0.    DECLARATIONS
  !              ------------
  !
  !
  !
  use iso_c_binding
  use modi_condensation
  !
  implicit none
  !
  !
  !*       0.1   declarations of dummy arguments :
  !
  !
  integer(c_int), intent(in) :: krr  ! number of moist variables
  character(c_char), intent(in) :: hbuname  ! name of the budget
  real(c_float), intent(in) :: ptstep  ! double time step
  ! (single if cold start)
  real(c_float), intent(in), dimension(d_nijt) :: psigqsat  ! coeff applied to qsat variance contribution
  !
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: prhodj  ! dry density * jacobian
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: pexnref  ! reference exner function
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: prhodref
  !
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: psigs  ! sigma_s at time t
  logical, intent(in) :: lmfconv  ! =size(pmfconv)!=0
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: pmfconv  ! convective mass flux
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: ppabst  ! absolute pressure at t
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: pzz  ! height of model layer
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: pexn  ! exner function
  !
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: pcf_mf  ! convective mass flux cloud fraction
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: prc_mf  ! convective mass flux liquid mixing ratio
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: pri_mf  ! convective mass flux ice mixing ratio
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: pweight_mf_cloud  ! weight coefficient for the mass-flux cloud
  !
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: prv  ! water vapor m.r. to adjust
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: prc  ! cloud water m.r. to adjust
  real(c_float), intent(inout), dimension(d_nijt, d_nkt) :: prvs  ! water vapor m.r. source
  real(c_float), intent(inout), dimension(d_nijt, d_nkt) :: prcs  ! cloud water m.r. source
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: pth  ! theta to adjust
  real(c_float), intent(inout), dimension(d_nijt, d_nkt) :: pths  ! theta source
  logical, intent(in) :: ocompute_src
  real(c_float), intent(out), dimension(d_nijt, d_nkt) :: psrcs  ! second-order flux
  ! s'rc'/2sigma_s2 at time t+1
  ! multiplied by lambda_3
  real(c_float), intent(out), dimension(d_nijt, d_nkt) :: pcldfr  ! cloud fraction
  real(c_float), intent(out), dimension(d_nijt, d_nkt) :: picldfr  ! ice cloud fraction
  real(c_float), intent(out), dimension(d_nijt, d_nkt) :: pwcldfr  ! water or mixed-phase cloud fraction
  real(c_float), intent(out), dimension(d_nijt, d_nkt) :: pssio  ! super-saturation with respect to ice in the
  ! supersaturated fraction
  real(c_float), intent(out), dimension(d_nijt, d_nkt) :: pssiu  ! sub-saturation with respect to ice in the
  ! subsaturated fraction
  real(c_float), intent(out), dimension(d_nijt, d_nkt) :: pifr  ! ratio cloud ice moist part to dry part
  !
  real(c_float), intent(inout), dimension(d_nijt, d_nkt) :: pris  ! cloud ice  m.r. at t+1
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: prr  ! rain water m.r. to adjust
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: pri  ! cloud ice  m.r. to adjust
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: prs  ! aggregate  m.r. to adjust
  real(c_float), intent(in), dimension(d_nijt, d_nkt) :: prg  ! graupel    m.r. to adjust
  real(c_float), optional, intent(in), dimension(d_nijt) :: pice_cld_wgt
  real(c_float), optional, intent(in), dimension(d_nijt, d_nkt) :: prh  ! hail       m.r. to adjust
  real(c_float), optional, intent(out), dimension(d_nijt, d_nkt) :: pout_rv  ! adjusted value
  real(c_float), optional, intent(out), dimension(d_nijt, d_nkt) :: pout_rc  ! adjusted value
  real(c_float), optional, intent(out), dimension(d_nijt, d_nkt) :: pout_ri  ! adjusted value
  real(c_float), optional, intent(out), dimension(d_nijt, d_nkt) :: pout_th  ! adjusted value
  real(c_float), optional, intent(out), dimension(d_nijt, d_nkt) :: phlc_hrc
  real(c_float), optional, intent(out), dimension(d_nijt, d_nkt) :: phlc_hcf
  real(c_float), optional, intent(out), dimension(d_nijt, d_nkt) :: phli_hri
  real(c_float), optional, intent(out), dimension(d_nijt, d_nkt) :: phli_hcf
  real(c_float), optional, intent(in), dimension(d_nijt, d_nkt) :: phlc_hrc_mf
  real(c_float), optional, intent(in), dimension(d_nijt, d_nkt) :: phlc_hcf_mf
  real(c_float), optional, intent(in), dimension(d_nijt, d_nkt) :: phli_hri_mf
  real(c_float), optional, intent(in), dimension(d_nijt, d_nkt) :: phli_hcf_mf
  !
  !
  !*       0.2   declarations of local variables :
  !
  !
  real(c_float) :: zw1, zw2  ! intermediate fields
  real(c_float), dimension(d_nijt, d_nkt) :: zt, zrv, zrc, zri, zcph, zlv, zls
  ! adjusted temperature
  ! adjusted state
  ! guess of the cph for the mixing
  ! guess of the lv at t+1
  ! guess of the ls at t+1
  real(c_float) :: zcriaut, zhcf, zhr
  ! autoconversion thresholds
  !
  integer(c_int) :: jiter, itermax  ! iterative loop for first order adjustment
  integer(c_int) :: jij, jk
  integer(c_int) :: iktb, ikte, iijb, iije
  !
  real(c_float), dimension(d_nijt, d_nkt) :: zsigs, zsrcs
  real(c_float), dimension(d_nijt) :: zsigqsat
  logical(c_bool) :: llnone, lltriangle, llbiga, llhlc_h, llhli_h
  integer(c_int), intent(in) :: d_nijb
  integer(c_int), intent(in) :: d_nije
  integer(c_int), intent(in) :: d_nijt
  integer(c_int), intent(in) :: d_nkb
  integer(c_int), intent(in) :: d_nke
  integer(c_int), intent(in) :: d_nkl
  integer(c_int), intent(in) :: d_nkt
  integer(c_int), intent(in) :: d_nktb
  integer(c_int), intent(in) :: d_nkte
  real(c_float), intent(in) :: cst_xalpi
  real(c_float), intent(in) :: cst_xalpw
  real(c_float), intent(in) :: cst_xbetai
  real(c_float), intent(in) :: cst_xbetaw
  real(c_float), intent(in) :: cst_xci
  real(c_float), intent(in) :: cst_xcl
  real(c_float), intent(in) :: cst_xcpd
  real(c_float), intent(in) :: cst_xcpv
  real(c_float), intent(in) :: cst_xepsilo
  real(c_float), intent(in) :: cst_xg
  real(c_float), intent(in) :: cst_xgami
  real(c_float), intent(in) :: cst_xgamw
  real(c_float), intent(in) :: cst_xlstt
  real(c_float), intent(in) :: cst_xlvtt
  real(c_float), intent(in) :: cst_xpi
  real(c_float), intent(in) :: cst_xrd
  real(c_float), intent(in) :: cst_xrv
  real(c_float), intent(in) :: cst_xtt
  real(c_float), intent(in) :: icep_xacriauti
  real(c_float), intent(in) :: icep_xbcriauti
  real(c_float), intent(in) :: icep_xcriautc
  real(c_float), intent(in) :: icep_xcriauti
  real(c_float), intent(in) :: icep_xfrmin(:)
  character(c_char), intent(in) :: nebn_ccondens
  character(c_char), intent(in) :: nebn_cfrac_ice_adjust
  character(c_char), intent(in) :: nebn_clambda3
  logical, intent(in) :: nebn_lcondborn
  logical, intent(in) :: nebn_lhgt_qs
  logical, intent(in) :: nebn_lsigmas
  logical, intent(in) :: nebn_lstatnw
  logical, intent(in) :: nebn_lsubg_cond
  real(c_float), intent(in) :: nebn_xtmaxmix
  real(c_float), intent(in) :: nebn_xtminmix
  character(c_char), intent(in) :: parami_csubg_mf_pdf
  logical, intent(in) :: parami_locnd2
  !
  !-------------------------------------------------------------------------------
  !
  !*       1.     PRELIMINARIES
  !               -------------
  !
  !
  IKTB = D_NKTB
  IKTE = D_NKTE
  IIJB = D_NIJB
  IIJE = D_NIJE
  !
  ITERMAX = 1
  !
  !-------------------------------------------------------------------------------
  !
  !*       2.     COMPUTE QUANTITIES WITH THE GUESS OF THE FUTURE INSTANT
  !               -------------------------------------------------------
  !
  !
  !    beginning of the iterative loop (to compute the adjusted state)
  !
  DO JITER=1,ITERMAX
    !
    !*       2.3    compute the latent heat of vaporization Lv(T*) at t+1
    !                   and the latent heat of sublimation  Ls(T*) at t+1
    !


    DO JK=IKTB,IKTE
      DO JIJ=IIJB,IIJE
        IF (JITER == 1) ZT(JIJ, JK) = PTH(JIJ, JK)*PEXN(JIJ, JK)
        ZLV(JIJ, JK) = CST_XLVTT + (CST_XCPV - CST_XCL)*(ZT(JIJ, JK) - CST_XTT)
        ZLS(JIJ, JK) = CST_XLSTT + (CST_XCPV - CST_XCI)*(ZT(JIJ, JK) - CST_XTT)
      END DO
    END DO

    !
    !*       2.4   Iterate
    !
    IF (JITER == 1) THEN
      ! compute with input values
      ! [Loki] inlined child subroutine: ITERATION
      ! =========================================
      !
      !*       2.4    compute the specific heat for moist air (Cph) at t+1
      !


      DO JK=IKTB,IKTE
        DO JIJ=IIJB,IIJE
          SELECT CASE (KRR)
          CASE (7)
            ZCPH(JIJ, JK) = CST_XCPD + CST_XCPV*PRV(JIJ, JK) + CST_XCL*(PRC(JIJ, JK) + PRR(JIJ, JK)) + CST_XCI*(PRI(JIJ, JK) +  &
            & PRS(JIJ, JK) + PRG(JIJ, JK) + PRH(JIJ, JK))
          CASE (6)
            ZCPH(JIJ, JK) = CST_XCPD + CST_XCPV*PRV(JIJ, JK) + CST_XCL*(PRC(JIJ, JK) + PRR(JIJ, JK)) + CST_XCI*(PRI(JIJ, JK) +  &
            & PRS(JIJ, JK) + PRG(JIJ, JK))
          CASE (5)
            ZCPH(JIJ, JK) =  &
            & CST_XCPD + CST_XCPV*PRV(JIJ, JK) + CST_XCL*(PRC(JIJ, JK) + PRR(JIJ, JK)) + CST_XCI*(PRI(JIJ, JK) + PRS(JIJ, JK))
          CASE (3)
            ZCPH(JIJ, JK) = CST_XCPD + CST_XCPV*PRV(JIJ, JK) + CST_XCL*(PRC(JIJ, JK) + PRR(JIJ, JK))
          CASE (2)
            ZCPH(JIJ, JK) = CST_XCPD + CST_XCPV*PRV(JIJ, JK) + CST_XCL*PRC(JIJ, JK)
          END SELECT
        END DO
      END DO

      !
      IF (NEBN_LSUBG_COND) THEN
        !
        !*       3.     SUBGRID CONDENSATION SCHEME
        !               ---------------------------
        !
        !   ZSRC= s'rci'/Sigma_s^2
        !   ZT is INOUT
        CALL CONDENSATION(D_NIJB, D_NIJE, D_NIJT, D_NKB, D_NKE, D_NKL, D_NKT, D_NKTB, D_NKTE, CST_XALPI, CST_XALPW, CST_XBETAI,  &
        & CST_XBETAW, CST_XCI, CST_XCL, CST_XCPD, CST_XCPV, CST_XEPSILO, CST_XG, CST_XGAMI, CST_XGAMW, CST_XLSTT, CST_XLVTT,  &
        & CST_XPI, CST_XRD, CST_XRV, CST_XTT, ICEP_XACRIAUTI, ICEP_XBCRIAUTI, ICEP_XCRIAUTC, ICEP_XCRIAUTI, ICEP_XFRMIN,  &
        & NEBN_LCONDBORN, NEBN_LHGT_QS, NEBN_LSTATNW, NEBN_XTMAXMIX, NEBN_XTMINMIX, NEBN_CFRAC_ICE_ADJUST, NEBN_CCONDENS,  &
        & NEBN_CLAMBDA3, PPABST, PZZ, PRHODREF, ZT, PRV(:, :), ZRV(:, :), PRC(:, :), ZRC(:, :), PRI(:, :), ZRI(:, :), PRR, PRS,  &
        & PRG, PSIGS, LMFCONV, PMFCONV, PCLDFR, ZSRCS, .true., NEBN_LSIGMAS, PARAMI_LOCND2, PICLDFR, PWCLDFR, PSSIO, PSSIU,  &
        & PIFR, PSIGQSAT, PLV=ZLV, PLS=ZLS, PCPH=ZCPH, PHLC_HRC=PHLC_HRC, PHLC_HCF=PHLC_HCF, PHLI_HRI=PHLI_HRI,  &
        & PHLI_HCF=PHLI_HCF, PICE_CLD_WGT=PICE_CLD_WGT)
      ELSE
        !
        !*       4.     ALL OR NOTHING CONDENSATION SCHEME
        !                            FOR MIXED-PHASE CLOUD
        !               -----------------------------------------------
        !

        ZSIGS(:, :) = 0.
        ZSIGQSAT(:) = 0.

        !ZT is INOUT
        CALL CONDENSATION(D_NIJB, D_NIJE, D_NIJT, D_NKB, D_NKE, D_NKL, D_NKT, D_NKTB, D_NKTE, CST_XALPI, CST_XALPW, CST_XBETAI,  &
        & CST_XBETAW, CST_XCI, CST_XCL, CST_XCPD, CST_XCPV, CST_XEPSILO, CST_XG, CST_XGAMI, CST_XGAMW, CST_XLSTT, CST_XLVTT,  &
        & CST_XPI, CST_XRD, CST_XRV, CST_XTT, ICEP_XACRIAUTI, ICEP_XBCRIAUTI, ICEP_XCRIAUTC, ICEP_XCRIAUTI, ICEP_XFRMIN,  &
        & NEBN_LCONDBORN, NEBN_LHGT_QS, NEBN_LSTATNW, NEBN_XTMAXMIX, NEBN_XTMINMIX, NEBN_CFRAC_ICE_ADJUST, NEBN_CCONDENS,  &
        & NEBN_CLAMBDA3, PPABST, PZZ, PRHODREF, ZT, PRV(:, :), ZRV(:, :), PRC(:, :), ZRC(:, :), PRI(:, :), ZRI(:, :), PRR, PRS,  &
        & PRG, ZSIGS, LMFCONV, PMFCONV, PCLDFR, ZSRCS, .true., .true., PARAMI_LOCND2, PICLDFR, PWCLDFR, PSSIO, PSSIU, PIFR,  &
        & ZSIGQSAT, PLV=ZLV, PLS=ZLS, PCPH=ZCPH, PHLC_HRC=PHLC_HRC, PHLC_HCF=PHLC_HCF, PHLI_HRI=PHLI_HRI, PHLI_HCF=PHLI_HCF,  &
        & PICE_CLD_WGT=PICE_CLD_WGT)
      END IF

      ! =========================================
    ELSE
      ! compute with updated values
      ! [Loki] inlined child subroutine: ITERATION
      ! =========================================
      !
      !*       2.4    compute the specific heat for moist air (Cph) at t+1
      !


      DO JK=IKTB,IKTE
        DO JIJ=IIJB,IIJE
          SELECT CASE (KRR)
          CASE (7)
            ZCPH(JIJ, JK) = CST_XCPD + CST_XCPV*ZRV(JIJ, JK) + CST_XCL*(ZRC(JIJ, JK) + PRR(JIJ, JK)) + CST_XCI*(ZRI(JIJ, JK) +  &
            & PRS(JIJ, JK) + PRG(JIJ, JK) + PRH(JIJ, JK))
          CASE (6)
            ZCPH(JIJ, JK) = CST_XCPD + CST_XCPV*ZRV(JIJ, JK) + CST_XCL*(ZRC(JIJ, JK) + PRR(JIJ, JK)) + CST_XCI*(ZRI(JIJ, JK) +  &
            & PRS(JIJ, JK) + PRG(JIJ, JK))
          CASE (5)
            ZCPH(JIJ, JK) =  &
            & CST_XCPD + CST_XCPV*ZRV(JIJ, JK) + CST_XCL*(ZRC(JIJ, JK) + PRR(JIJ, JK)) + CST_XCI*(ZRI(JIJ, JK) + PRS(JIJ, JK))
          CASE (3)
            ZCPH(JIJ, JK) = CST_XCPD + CST_XCPV*ZRV(JIJ, JK) + CST_XCL*(ZRC(JIJ, JK) + PRR(JIJ, JK))
          CASE (2)
            ZCPH(JIJ, JK) = CST_XCPD + CST_XCPV*ZRV(JIJ, JK) + CST_XCL*ZRC(JIJ, JK)
          END SELECT
        END DO
      END DO

      !
      IF (NEBN_LSUBG_COND) THEN
        !
        !*       3.     SUBGRID CONDENSATION SCHEME
        !               ---------------------------
        !
        !   ZSRC= s'rci'/Sigma_s^2
        !   ZT is INOUT
        CALL CONDENSATION(D_NIJB, D_NIJE, D_NIJT, D_NKB, D_NKE, D_NKL, D_NKT, D_NKTB, D_NKTE, CST_XALPI, CST_XALPW, CST_XBETAI,  &
        & CST_XBETAW, CST_XCI, CST_XCL, CST_XCPD, CST_XCPV, CST_XEPSILO, CST_XG, CST_XGAMI, CST_XGAMW, CST_XLSTT, CST_XLVTT,  &
        & CST_XPI, CST_XRD, CST_XRV, CST_XTT, ICEP_XACRIAUTI, ICEP_XBCRIAUTI, ICEP_XCRIAUTC, ICEP_XCRIAUTI, ICEP_XFRMIN,  &
        & NEBN_LCONDBORN, NEBN_LHGT_QS, NEBN_LSTATNW, NEBN_XTMAXMIX, NEBN_XTMINMIX, NEBN_CFRAC_ICE_ADJUST, NEBN_CCONDENS,  &
        & NEBN_CLAMBDA3, PPABST, PZZ, PRHODREF, ZT, ZRV(:, :), ZRV(:, :), ZRC(:, :), ZRC(:, :), ZRI(:, :), ZRI(:, :), PRR, PRS,  &
        & PRG, PSIGS, LMFCONV, PMFCONV, PCLDFR, ZSRCS, .true., NEBN_LSIGMAS, PARAMI_LOCND2, PICLDFR, PWCLDFR, PSSIO, PSSIU,  &
        & PIFR, PSIGQSAT, PLV=ZLV, PLS=ZLS, PCPH=ZCPH, PHLC_HRC=PHLC_HRC, PHLC_HCF=PHLC_HCF, PHLI_HRI=PHLI_HRI,  &
        & PHLI_HCF=PHLI_HCF, PICE_CLD_WGT=PICE_CLD_WGT)
      ELSE
        !
        !*       4.     ALL OR NOTHING CONDENSATION SCHEME
        !                            FOR MIXED-PHASE CLOUD
        !               -----------------------------------------------
        !

        ZSIGS(:, :) = 0.
        ZSIGQSAT(:) = 0.

        !ZT is INOUT
        CALL CONDENSATION(D_NIJB, D_NIJE, D_NIJT, D_NKB, D_NKE, D_NKL, D_NKT, D_NKTB, D_NKTE, CST_XALPI, CST_XALPW, CST_XBETAI,  &
        & CST_XBETAW, CST_XCI, CST_XCL, CST_XCPD, CST_XCPV, CST_XEPSILO, CST_XG, CST_XGAMI, CST_XGAMW, CST_XLSTT, CST_XLVTT,  &
        & CST_XPI, CST_XRD, CST_XRV, CST_XTT, ICEP_XACRIAUTI, ICEP_XBCRIAUTI, ICEP_XCRIAUTC, ICEP_XCRIAUTI, ICEP_XFRMIN,  &
        & NEBN_LCONDBORN, NEBN_LHGT_QS, NEBN_LSTATNW, NEBN_XTMAXMIX, NEBN_XTMINMIX, NEBN_CFRAC_ICE_ADJUST, NEBN_CCONDENS,  &
        & NEBN_CLAMBDA3, PPABST, PZZ, PRHODREF, ZT, ZRV(:, :), ZRV(:, :), ZRC(:, :), ZRC(:, :), ZRI(:, :), ZRI(:, :), PRR, PRS,  &
        & PRG, ZSIGS, LMFCONV, PMFCONV, PCLDFR, ZSRCS, .true., .true., PARAMI_LOCND2, PICLDFR, PWCLDFR, PSSIO, PSSIU, PIFR,  &
        & ZSIGQSAT, PLV=ZLV, PLS=ZLS, PCPH=ZCPH, PHLC_HRC=PHLC_HRC, PHLC_HCF=PHLC_HCF, PHLI_HRI=PHLI_HRI, PHLI_HCF=PHLI_HCF,  &
        & PICE_CLD_WGT=PICE_CLD_WGT)
      END IF

      ! =========================================
    END IF
  END DO
  ! end of the iterative loop
  !
  !*       5.     COMPUTE THE SOURCES AND STORES THE CLOUD FRACTION
  !               -------------------------------------------------
  !
  !
  ! Apply a ponderation between condensation and mas flux cloud
  LLHLC_H = PRESENT(PHLC_HRC) .and. PRESENT(PHLC_HCF)
  LLHLI_H = PRESENT(PHLI_HRI) .and. PRESENT(PHLI_HCF)


  DO JK=IKTB,IKTE
    DO JIJ=IIJB,IIJE
      ZRC(JIJ, JK) = ZRC(JIJ, JK)*(1. - PWEIGHT_MF_CLOUD(JIJ, JK))
      ZRI(JIJ, JK) = ZRI(JIJ, JK)*(1. - PWEIGHT_MF_CLOUD(JIJ, JK))
      PCLDFR(JIJ, JK) = PCLDFR(JIJ, JK)*(1. - PWEIGHT_MF_CLOUD(JIJ, JK))
      ZSRCS(JIJ, JK) = ZSRCS(JIJ, JK)*(1. - PWEIGHT_MF_CLOUD(JIJ, JK))
      IF (LLHLC_H) THEN
        PHLC_HRC(JIJ, JK) = PHLC_HRC(JIJ, JK)*(1. - PWEIGHT_MF_CLOUD(JIJ, JK))
        PHLC_HCF(JIJ, JK) = PHLC_HCF(JIJ, JK)*(1. - PWEIGHT_MF_CLOUD(JIJ, JK))
      END IF
      IF (LLHLI_H) THEN
        PHLI_HRI(JIJ, JK) = PHLI_HRI(JIJ, JK)*(1. - PWEIGHT_MF_CLOUD(JIJ, JK))
        PHLI_HCF(JIJ, JK) = PHLI_HCF(JIJ, JK)*(1. - PWEIGHT_MF_CLOUD(JIJ, JK))
      END IF
    END DO
  END DO

  !


  DO JK=IKTB,IKTE
    DO JIJ=IIJB,IIJE
      !
      !*       5.0    compute the variation of mixing ratio
      !
      !         Rc - Rc*
      ZW1 = (ZRC(JIJ, JK) - PRC(JIJ, JK)) / PTSTEP        ! Pcon = ----------
      !         2 Delta t
      ZW2 = (ZRI(JIJ, JK) - PRI(JIJ, JK)) / PTSTEP        ! idem ZW1 but for Ri
      !
      !*       5.1    compute the sources
      !
      IF (ZW1 < 0.0) THEN
        ZW1 = MAX(ZW1, -PRCS(JIJ, JK))
      ELSE
        ZW1 = MIN(ZW1, PRVS(JIJ, JK))
      END IF
      PRVS(JIJ, JK) = PRVS(JIJ, JK) - ZW1
      PRCS(JIJ, JK) = PRCS(JIJ, JK) + ZW1
      PTHS(JIJ, JK) = PTHS(JIJ, JK) + ZW1*ZLV(JIJ, JK) / (ZCPH(JIJ, JK)*PEXNREF(JIJ, JK))
      !
      IF (ZW2 < 0.0) THEN
        ZW2 = MAX(ZW2, -PRIS(JIJ, JK))
      ELSE
        ZW2 = MIN(ZW2, PRVS(JIJ, JK))
      END IF
      PRVS(JIJ, JK) = PRVS(JIJ, JK) - ZW2
      PRIS(JIJ, JK) = PRIS(JIJ, JK) + ZW2
      PTHS(JIJ, JK) = PTHS(JIJ, JK) + ZW2*ZLS(JIJ, JK) / (ZCPH(JIJ, JK)*PEXNREF(JIJ, JK))
    END DO
  END DO

  !
  !*       5.2    compute the cloud fraction PCLDFR
  !
  IF (.not.NEBN_LSUBG_COND) THEN


    DO JK=IKTB,IKTE
      DO JIJ=IIJB,IIJE
        IF (PRCS(JIJ, JK) + PRIS(JIJ, JK) > 1.E-12 / PTSTEP) THEN
          PCLDFR(JIJ, JK) = 1.
        ELSE
          PCLDFR(JIJ, JK) = 0.
        END IF
        ZSRCS(JIJ, JK) = PCLDFR(JIJ, JK)
      END DO
    END DO

  ELSE
    !NEBN%LSUBG_COND case

    ! Tests on characters strings can break the vectorization, or at least they would
    ! slow down considerably the performance of a vector loop. One should use tests on
    ! reals, integer(c_int)s or booleans only. REK.
    LLNONE = PARAMI_CSUBG_MF_PDF == 'NONE'
    LLTRIANGLE = PARAMI_CSUBG_MF_PDF == 'TRIANGLE'
    LLBIGA = PARAMI_CSUBG_MF_PDF == 'BIGA'

    DO JK=IKTB,IKTE
      DO JIJ=IIJB,IIJE
        !We limit PRC_MF+PRI_MF to PRVS*PTSTEP to avoid negative humidity
        ZW1 = PRC_MF(JIJ, JK) / PTSTEP
        ZW2 = PRI_MF(JIJ, JK) / PTSTEP
        IF (ZW1 + ZW2 > PRVS(JIJ, JK)) THEN
          ZW1 = ZW1*PRVS(JIJ, JK) / (ZW1 + ZW2)
          ZW2 = PRVS(JIJ, JK) - ZW1
        END IF
        PCLDFR(JIJ, JK) = MIN(1., PCLDFR(JIJ, JK) + PCF_MF(JIJ, JK))
        PRCS(JIJ, JK) = PRCS(JIJ, JK) + ZW1
        PRIS(JIJ, JK) = PRIS(JIJ, JK) + ZW2
        PRVS(JIJ, JK) = PRVS(JIJ, JK) - (ZW1 + ZW2)
        PTHS(JIJ, JK) = PTHS(JIJ, JK) + (ZW1*ZLV(JIJ, JK) + ZW2*ZLS(JIJ, JK)) / ZCPH(JIJ, JK) / PEXNREF(JIJ, JK)
        !
        IF (LLHLC_H) THEN
          ZCRIAUT = ICEP_XCRIAUTC / PRHODREF(JIJ, JK)
          IF (LLNONE) THEN
            IF (ZW1*PTSTEP > PCF_MF(JIJ, JK)*ZCRIAUT) THEN
              PHLC_HRC(JIJ, JK) = PHLC_HRC(JIJ, JK) + ZW1*PTSTEP
              PHLC_HCF(JIJ, JK) = MIN(1., PHLC_HCF(JIJ, JK) + PCF_MF(JIJ, JK))
            END IF
          ELSE IF (LLTRIANGLE) THEN
            !ZHCF is the precipitating part of the *cloud* and not of the grid cell
            IF (ZW1*PTSTEP > PCF_MF(JIJ, JK)*ZCRIAUT) THEN
              ZHCF = 1. - .5*(ZCRIAUT*PCF_MF(JIJ, JK) / MAX(1.E-20, ZW1*PTSTEP))**2
              ZHR = ZW1*PTSTEP - (ZCRIAUT*PCF_MF(JIJ, JK))**3 / (3*MAX(1.E-20, ZW1*PTSTEP)**2)
            ELSE IF (2.*ZW1*PTSTEP <= PCF_MF(JIJ, JK)*ZCRIAUT) THEN
              ZHCF = 0.
              ZHR = 0.
            ELSE
              ZHCF = (2.*ZW1*PTSTEP - ZCRIAUT*PCF_MF(JIJ, JK))**2 / (2.*MAX(1.E-20, ZW1*PTSTEP)**2)
              ZHR = (4.*(ZW1*PTSTEP)**3 - 3.*ZW1*PTSTEP*(ZCRIAUT*PCF_MF(JIJ, JK))**2 + (ZCRIAUT*PCF_MF(JIJ, JK))**3) /  &
              & (3*MAX(1.E-20, ZW1*PTSTEP)**2)
            END IF
            ZHCF = ZHCF*PCF_MF(JIJ, JK)              !to retrieve the part of the grid cell
            PHLC_HCF(JIJ, JK) = MIN(1., PHLC_HCF(JIJ, JK) + ZHCF)              !total part of the grid cell that is precipitating
            PHLC_HRC(JIJ, JK) = PHLC_HRC(JIJ, JK) + ZHR
          ELSE IF (LLBIGA) THEN
            PHLC_HCF(JIJ, JK) = MIN(1., PHLC_HCF(JIJ, JK) + PHLC_HCF_MF(JIJ, JK))
            PHLC_HRC(JIJ, JK) = PHLC_HRC(JIJ, JK) + PHLC_HRC_MF(JIJ, JK)
          END IF
        END IF
        IF (LLHLI_H) THEN
          ZCRIAUT = MIN(ICEP_XCRIAUTI, 10**(ICEP_XACRIAUTI*(ZT(JIJ, JK) - CST_XTT) + ICEP_XBCRIAUTI))
          IF (LLNONE) THEN
            IF (ZW2*PTSTEP > PCF_MF(JIJ, JK)*ZCRIAUT) THEN
              PHLI_HRI(JIJ, JK) = PHLI_HRI(JIJ, JK) + ZW2*PTSTEP
              PHLI_HCF(JIJ, JK) = MIN(1., PHLI_HCF(JIJ, JK) + PCF_MF(JIJ, JK))
            END IF
          ELSE IF (LLTRIANGLE) THEN
            !ZHCF is the precipitating part of the *cloud* and not of the grid cell
            IF (ZW2*PTSTEP > PCF_MF(JIJ, JK)*ZCRIAUT) THEN
              ZHCF = 1. - .5*(ZCRIAUT*PCF_MF(JIJ, JK) / (ZW2*PTSTEP))**2
              ZHR = ZW2*PTSTEP - (ZCRIAUT*PCF_MF(JIJ, JK))**3 / (3*(ZW2*PTSTEP)**2)
            ELSE IF (2.*ZW2*PTSTEP <= PCF_MF(JIJ, JK)*ZCRIAUT) THEN
              ZHCF = 0.
              ZHR = 0.
            ELSE
              ZHCF = (2.*ZW2*PTSTEP - ZCRIAUT*PCF_MF(JIJ, JK))**2 / (2.*(ZW2*PTSTEP)**2)
              ZHR = (4.*(ZW2*PTSTEP)**3 - 3.*ZW2*PTSTEP*(ZCRIAUT*PCF_MF(JIJ, JK))**2 + (ZCRIAUT*PCF_MF(JIJ, JK))**3) /  &
              & (3*(ZW2*PTSTEP)**2)
            END IF
            ZHCF = ZHCF*PCF_MF(JIJ, JK)              !to retrieve the part of the grid cell
            PHLI_HCF(JIJ, JK) = MIN(1., PHLI_HCF(JIJ, JK) + ZHCF)              !total part of the grid cell that is precipitating
            PHLI_HRI(JIJ, JK) = PHLI_HRI(JIJ, JK) + ZHR
          ELSE IF (LLBIGA) THEN
            PHLI_HCF(JIJ, JK) = MIN(1., PHLI_HCF(JIJ, JK) + PHLI_HCF_MF(JIJ, JK))
            PHLI_HRI(JIJ, JK) = PHLI_HRI(JIJ, JK) + PHLI_HRI_MF(JIJ, JK)
          END IF
        END IF
        !
        IF (PRESENT(POUT_RV) .or. PRESENT(POUT_RC) .or. PRESENT(POUT_RI) .or. PRESENT(POUT_TH)) THEN
          ZW1 = PRC_MF(JIJ, JK)
          ZW2 = PRI_MF(JIJ, JK)
          IF (ZW1 + ZW2 > ZRV(JIJ, JK)) THEN
            ZW1 = ZW1*ZRV(JIJ, JK) / (ZW1 + ZW2)
            ZW2 = ZRV(JIJ, JK) - ZW1
          END IF
          ZRC(JIJ, JK) = ZRC(JIJ, JK) + ZW1
          ZRI(JIJ, JK) = ZRI(JIJ, JK) + ZW2
          ZRV(JIJ, JK) = ZRV(JIJ, JK) - (ZW1 + ZW2)
          ZT(JIJ, JK) = ZT(JIJ, JK) + (ZW1*ZLV(JIJ, JK) + ZW2*ZLS(JIJ, JK)) / ZCPH(JIJ, JK)
        END IF
      END DO
    END DO

  END IF
  !NEBN%LSUBG_COND
  !
  IF (OCOMPUTE_SRC) PSRCS = ZSRCS
  IF (PRESENT(POUT_RV)) POUT_RV = ZRV
  IF (PRESENT(POUT_RC)) POUT_RC = ZRC
  IF (PRESENT(POUT_RI)) POUT_RI = ZRI
  IF (PRESENT(POUT_TH)) POUT_TH = ZT / PEXN(:, :)
  !
  !
  !*       6.  STORE THE BUDGET TERMS
  !            ----------------------
  !
  !------------------------------------------------------------------------------
  !
  !
  !

END SUBROUTINE ICE_ADJUST
