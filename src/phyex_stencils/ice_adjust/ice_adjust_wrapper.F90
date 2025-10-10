module modi_ice_adjust_wrapper
   use iso_c_binding
   implicit none
contains
!
   subroutine ice_adjust_wrapper(d_nijb, d_nije, d_nijt, d_nkb, d_nke, d_nkl, d_nkt, d_nktb, d_nkte, &
   & cst_xalpi, cst_xalpw, cst_xbetai, cst_xbetaw, cst_xci, cst_xcl, cst_xcpd, cst_xcpv, cst_xepsilo, &
   & cst_xg, cst_xgami, cst_xgamw, cst_xlstt, cst_xlvtt, cst_xpi, cst_xrd, cst_xrv, cst_xtt, &
   & icep_xacriauti, icep_xbcriauti, icep_xcriautc, icep_xcriauti, icep_xfrmin, nebn_ccondens,  &
   & nebn_cfrac_ice_adjust, nebn_clambda3, nebn_lcondborn, nebn_lhgt_qs, nebn_lsigmas, nebn_lstatnw, &
   & nebn_lsubg_cond, nebn_xtmaxmix, nebn_xtminmix, parami_csubg_mf_pdf, parami_locnd2, krr, hbuname, ptstep, &
   & psigqsat, prhodj, pexnref, prhodref, psigs, lmfconv, pmfconv, ppabst, pzz, pexn, pcf_mf, prc_mf, pri_mf, &
   & pweight_mf_cloud, picldfr, pwcldfr, pssio, pssiu, pifr, prv, prc, prvs, prcs, pth, pths, ocompute_src, &
   & psrcs, pcldfr, prr, pri, pris, prs, prg, pice_cld_wgt, prh, pout_rv, pout_rc, pout_ri, pout_th, phlc_hrc, &
   & phlc_hcf, phli_hri, phli_hcf, phlc_hrc_mf, phlc_hcf_mf, phli_hri_mf, phli_hcf_mf) bind(c)

      use modi_condensation
      implicit none

      integer(c_int), intent(in) :: d_nijb, d_nije, d_nijt, d_nkb, d_nke, d_nkl, d_nkt, d_nktb, d_nkte
      real(c_double), intent(in) :: cst_xalpi, cst_xalpw, cst_xbetai, cst_xbetaw, cst_xci, cst_xcl, cst_xcpd
      real(c_double), intent(in) :: cst_xcpv, cst_xepsilo, cst_xg, cst_xgami, cst_xgamw, cst_xlstt, cst_xlvtt
      real(c_double), intent(in) :: cst_xpi, cst_xrd, cst_xrv, cst_xtt, icep_xacriauti, icep_xbcriauti, icep_xcriautc
      real(c_double), intent(in) :: icep_xcriauti, icep_xfrmin
      character(c_char), intent(in) :: nebn_ccondens, nebn_cfrac_ice_adjust, nebn_clambda3
      logical(c_bool), intent(in) :: nebn_lcondborn, nebn_lhgt_qs, nebn_lsigmas, nebn_lstatnw, nebn_lsubg_cond
      real(c_double), intent(in) :: nebn_xtmaxmix, nebn_xtminmix
      character(c_char), intent(in) :: parami_csubg_mf_pdf
      real(c_double), intent(in) :: parami_locnd2
      integer(c_int), intent(in) :: krr  ! number of moist variables
      character(c_char), intent(in) :: hbuname  ! name of the budget
      real(c_double), intent(in) :: ptstep  ! double time step
      real(c_double), intent(in) :: psigqsat(d_nijt) ! coeff applied to qsat variance contribution
      real(c_double), intent(in) :: prhodj(d_nijt, d_nkt)  ! dry density * jacobian
      real(c_double), intent(in) :: pexnref(d_nijt, d_nkt)  ! reference exner function
      real(c_double), intent(in) :: prhodref(d_nijt, d_nkt)
      real(c_double), intent(in) :: psigs(d_nijt, d_nkt)  ! sigma_s at time t
      logical(c_bool), intent(in) :: lmfconv  ! =size(pmfconv)!=0
      real(c_double), intent(in) :: pmfconv(d_nijt, d_nkt)  ! convective mass flux
      real(c_double), intent(in) :: ppabst(d_nijt, d_nkt)  ! absolute pressure at t
      real(c_double), intent(in) :: pzz(d_nijt, d_nkt)  ! height of model layer
      real(c_double), intent(in) :: pexn(d_nijt, d_nkt)  ! exner function
      !
      real(c_double), intent(in) :: pcf_mf(d_nijt, d_nkt)  ! convective mass flux cloud fraction
      real(c_double), intent(in) :: prc_mf(d_nijt, d_nkt)  ! convective mass flux liquid mixing ratio
      real(c_double), intent(in) :: pri_mf(d_nijt, d_nkt)  ! convective mass flux ice mixing ratio
      real(c_double), intent(in) :: pweight_mf_cloud(d_nijt, d_nkt)  ! weight coefficient for the mass-flux cloud
      !
      real(c_double), intent(in) :: prv(d_nijt, d_nkt)  ! water vapor m.r. to adjust
      real(c_double), intent(in) :: prc(d_nijt, d_nkt)  ! cloud water m.r. to adjust
      real(c_double), intent(inout) :: prvs(d_nijt, d_nkt)  ! water vapor m.r. source
      real(c_double), intent(inout) :: prcs(d_nijt, d_nkt)  ! cloud water m.r. source
      real(c_double), intent(in) :: pth(d_nijt, d_nkt)  ! theta to adjust
      real(c_double), intent(inout) :: pths(d_nijt, d_nkt)  ! theta source
      logical(c_bool), intent(in) :: ocompute_src
      real(c_double), intent(out) :: psrcs(d_nijt, d_nkt)  ! second-order flux
      ! s'rc'/2sigma_s2 at time t+1
      ! multiplied by lambda_3
      real(c_double), intent(out) :: pcldfr(d_nijt, d_nkt)  ! cloud fraction
      real(c_double), intent(out) :: picldfr(d_nijt, d_nkt)  ! ice cloud fraction
      real(c_double), intent(out) :: pwcldfr(d_nijt, d_nkt)  ! water or mixed-phase cloud fraction
      real(c_double), intent(out) :: pssio(d_nijt, d_nkt)  ! super-saturation with respect to ice in the
      ! supersaturated fraction
      real(c_double), intent(out) :: pssiu(d_nijt, d_nkt)  ! sub-saturation with respect to ice in the
      ! subsaturated fraction
      real(c_double), intent(out) :: pifr(d_nijt, d_nkt)  ! ratio cloud ice moist part to dry part
      !
      real(c_double), intent(inout) :: pris(d_nijt, d_nkt)  ! cloud ice  m.r. at t+1
      real(c_double), intent(in) :: prr(d_nijt, d_nkt)  ! rain water m.r. to adjust
      real(c_double), intent(in) :: pri(d_nijt, d_nkt)  ! cloud ice  m.r. to adjust
      real(c_double), intent(in) :: prs(d_nijt, d_nkt)  ! aggregate  m.r. to adjust
      real(c_double), intent(in) :: prg(d_nijt, d_nkt)  ! graupel    m.r. to adjust

      !! optional fields
      real(c_double), intent(in) :: pice_cld_wgt(d_nijt)
      real(c_double), intent(in) :: prh(d_nijt, d_nkt)  ! hail       m.r. to adjust
      real(c_double), intent(out) :: pout_rv(d_nijt, d_nkt)  ! adjusted value
      real(c_double), intent(out) :: pout_rc(d_nijt, d_nkt)  ! adjusted value
      real(c_double), intent(out) :: pout_ri(d_nijt, d_nkt)  ! adjusted value
      real(c_double), intent(out) :: pout_th(d_nijt, d_nkt)  ! adjusted value
      real(c_double), intent(out) :: phlc_hrc(d_nijt, d_nkt)
      real(c_double), intent(out) :: phlc_hcf(d_nijt, d_nkt)
      real(c_double), intent(out) :: phli_hri(d_nijt, d_nkt)
      real(c_double), intent(out) :: phli_hcf(d_nijt, d_nkt)
      real(c_double), intent(in) :: phlc_hrc_mf(d_nijt, d_nkt)
      real(c_double), intent(in) :: phlc_hcf_mf(d_nijt, d_nkt)
      real(c_double), intent(in) :: phli_hri_mf(d_nijt, d_nkt)
      real(c_double), intent(in) :: phli_hcf_mf(d_nijt, d_nkt)

      call ice_adjust(d_nijb, d_nije, d_nijt, d_nkb, d_nke, d_nkl, d_nkt, d_nktb, d_nkte, &
   & cst_xalpi, cst_xalpw, cst_xbetai, cst_xbetaw, cst_xci, cst_xcl, cst_xcpd, cst_xcpv, cst_xepsilo, &
   & cst_xg, cst_xgami, cst_xgamw, cst_xlstt, cst_xlvtt, cst_xpi, cst_xrd, cst_xrv, cst_xtt, &
   & icep_xacriauti, icep_xbcriauti, icep_xcriautc, icep_xcriauti, icep_xfrmin, nebn_ccondens,  &
   & nebn_cfrac_ice_adjust, nebn_clambda3, nebn_lcondborn, nebn_lhgt_qs, nebn_lsigmas, nebn_lstatnw, &
   & nebn_lsubg_cond, nebn_xtmaxmix, nebn_xtminmix, parami_csubg_mf_pdf, parami_locnd2, krr, hbuname, ptstep, &
   & psigqsat, prhodj, pexnref, prhodref, psigs, lmfconv, pmfconv, ppabst, pzz, pexn, pcf_mf, prc_mf, pri_mf, &
   & pweight_mf_cloud, picldfr, pwcldfr, pssio, pssiu, pifr, prv, prc, prvs, prcs, pth, pths, ocompute_src, &
   & psrcs, pcldfr, prr, pri, pris, prs, prg, pice_cld_wgt, prh, pout_rv, pout_rc, pout_ri, pout_th, phlc_hrc, &
   & phlc_hcf, phli_hri, phli_hcf, phlc_hrc_mf, phlc_hcf_mf, phli_hri_mf, phli_hcf_mf)
!
   end subroutine ice_adjust_wrapper
!
!
end module modi_ice_adjust_wrapper
