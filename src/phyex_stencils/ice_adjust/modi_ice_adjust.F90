!     ######################
module modi_ice_adjust
!     ######################
!
   implicit none
   interface
!
      subroutine ice_adjust(d_nijb, d_nije, d_nijt, d_nkb, d_nke, d_nkl, d_nkt, d_nktb, d_nkte, &
      & cst_xalpi, cst_xalpw, cst_xbetai, cst_xbetaw, cst_xci, cst_xcl, cst_xcpd, cst_xcpv, cst_xepsilo, &
      & cst_xg, cst_xgami, cst_xgamw, cst_xlstt, cst_xlvtt, cst_xpi,  cst_xrd, cst_xrv, cst_xtt, &
      & icep_xacriauti, icep_xbcriauti, icep_xcriautc, icep_xcriauti, icep_xfrmin, nebn_ccondens,  &
      & nebn_cfrac_ice_adjust, nebn_clambda3, nebn_lcondborn, nebn_lhgt_qs, nebn_lsigmas, nebn_lstatnw, &
      & nebn_lsubg_cond, nebn_xtmaxmix, nebn_xtminmix, parami_csubg_mf_pdf, parami_locnd2, krr, hbuname, ptstep, &
      & psigqsat, prhodj, pexnref, prhodref, psigs, lmfconv, pmfconv, ppabst, pzz, pexn, pcf_mf, prc_mf, pri_mf, &
      & pweight_mf_cloud, picldfr, pwcldfr, pssio, pssiu, pifr, prv, prc, prvs, prcs, pth, pths, ocompute_src, &
      & psrcs, pcldfr, prr, pri, pris, prs, prg, pice_cld_wgt, prh, pout_rv, pout_rc, pout_ri, pout_th, phlc_hrc, &
      & phlc_hcf, phli_hri, phli_hcf, phlc_hrc_mf, phlc_hcf_mf, phli_hri_mf, phli_hcf_mf)

         use modi_condensation
         implicit none
         !
         !
         !*       0.1   declarations of dummy arguments :

         integer, intent(in) :: d_nijb, d_nije, d_nijt, d_nkb, d_nke, d_nkl, d_nkt, d_nktb, d_nkte
         real, intent(in) :: cst_xalpi, cst_xalpw, cst_xbetai, cst_xbetaw, cst_xci, cst_xcl, cst_xcpd
         real, intent(in) :: cst_xcpv, cst_xepsilo, cst_xg, cst_xgami, cst_xgamw, cst_xlstt, cst_xlvtt
         real, intent(in) :: cst_xpi, cst_xrd, cst_xrv, cst_xtt, icep_xacriauti, icep_xbcriauti, icep_xcriautc
         real, intent(in) :: icep_xcriauti, icep_xfrmin
         character(len=4), intent(in) :: nebn_ccondens, nebn_cfrac_ice_adjust, nebn_clambda3
         logical, intent(in) :: nebn_lcondborn, nebn_lhgt_qs, nebn_lsigmas, nebn_lstatnw, nebn_lsubg_cond
         real, intent(in) :: nebn_xtmaxmix, nebn_xtminmix
         character(len=4), intent(in) :: parami_csubg_mf_pdf
         real, intent(in) :: parami_locnd2
         !
         !
         integer, intent(in) :: krr  ! number of moist variables
         character(len=4), intent(in) :: hbuname  ! name of the budget
         real, intent(in) :: ptstep  ! double time step
         ! (single if cold start)
         real, intent(in), dimension(d_nijt) :: psigqsat  ! coeff applied to qsat variance contribution
         !
         real, intent(in), dimension(d_nijt, d_nkt) :: prhodj  ! dry density * jacobian
         real, intent(in), dimension(d_nijt, d_nkt) :: pexnref  ! reference exner function
         real, intent(in), dimension(d_nijt, d_nkt) :: prhodref
         !
         real, intent(in), dimension(d_nijt, d_nkt) :: psigs  ! sigma_s at time t
         logical, intent(in) :: lmfconv  ! =size(pmfconv)!=0
         real, intent(in), dimension(d_nijt, d_nkt) :: pmfconv  ! convective mass flux
         real, intent(in), dimension(d_nijt, d_nkt) :: ppabst  ! absolute pressure at t
         real, intent(in), dimension(d_nijt, d_nkt) :: pzz  ! height of model layer
         real, intent(in), dimension(d_nijt, d_nkt) :: pexn  ! exner function
         !
         real, intent(in), dimension(d_nijt, d_nkt) :: pcf_mf  ! convective mass flux cloud fraction
         real, intent(in), dimension(d_nijt, d_nkt) :: prc_mf  ! convective mass flux liquid mixing ratio
         real, intent(in), dimension(d_nijt, d_nkt) :: pri_mf  ! convective mass flux ice mixing ratio
         real, intent(in), dimension(d_nijt, d_nkt) :: pweight_mf_cloud  ! weight coefficient for the mass-flux cloud
         !
         real, intent(in), dimension(d_nijt, d_nkt) :: prv  ! water vapor m.r. to adjust
         real, intent(in), dimension(d_nijt, d_nkt) :: prc  ! cloud water m.r. to adjust
         real, intent(inout), dimension(d_nijt, d_nkt) :: prvs  ! water vapor m.r. source
         real, intent(inout), dimension(d_nijt, d_nkt) :: prcs  ! cloud water m.r. source
         real, intent(in), dimension(d_nijt, d_nkt) :: pth  ! theta to adjust
         real, intent(inout), dimension(d_nijt, d_nkt) :: pths  ! theta source
         logical, intent(in) :: ocompute_src
         real, intent(out), dimension(d_nijt, d_nkt) :: psrcs  ! second-order flux
         ! s'rc'/2sigma_s2 at time t+1
         ! multiplied by lambda_3
         real, intent(out), dimension(d_nijt, d_nkt) :: pcldfr  ! cloud fraction
         real, intent(out), dimension(d_nijt, d_nkt) :: picldfr  ! ice cloud fraction
         real, intent(out), dimension(d_nijt, d_nkt) :: pwcldfr  ! water or mixed-phase cloud fraction
         real, intent(out), dimension(d_nijt, d_nkt) :: pssio  ! super-saturation with respect to ice in the
         ! supersaturated fraction
         real, intent(out), dimension(d_nijt, d_nkt) :: pssiu  ! sub-saturation with respect to ice in the
         ! subsaturated fraction
         real, intent(out), dimension(d_nijt, d_nkt) :: pifr  ! ratio cloud ice moist part to dry part
         !
         real, intent(inout), dimension(d_nijt, d_nkt) :: pris  ! cloud ice  m.r. at t+1
         real, intent(in), dimension(d_nijt, d_nkt) :: prr  ! rain water m.r. to adjust
         real, intent(in), dimension(d_nijt, d_nkt) :: pri  ! cloud ice  m.r. to adjust
         real, intent(in), dimension(d_nijt, d_nkt) :: prs  ! aggregate  m.r. to adjust
         real, intent(in), dimension(d_nijt, d_nkt) :: prg  ! graupel    m.r. to adjust
         real, optional, intent(in), dimension(d_nijt) :: pice_cld_wgt
         real, optional, intent(in), dimension(d_nijt, d_nkt) :: prh  ! hail       m.r. to adjust
         real, optional, intent(out), dimension(d_nijt, d_nkt) :: pout_rv  ! adjusted value
         real, optional, intent(out), dimension(d_nijt, d_nkt) :: pout_rc  ! adjusted value
         real, optional, intent(out), dimension(d_nijt, d_nkt) :: pout_ri  ! adjusted value
         real, optional, intent(out), dimension(d_nijt, d_nkt) :: pout_th  ! adjusted value
         real, optional, intent(out), dimension(d_nijt, d_nkt) :: phlc_hrc
         real, optional, intent(out), dimension(d_nijt, d_nkt) :: phlc_hcf
         real, optional, intent(out), dimension(d_nijt, d_nkt) :: phli_hri
         real, optional, intent(out), dimension(d_nijt, d_nkt) :: phli_hcf
         real, optional, intent(in), dimension(d_nijt, d_nkt) :: phlc_hrc_mf
         real, optional, intent(in), dimension(d_nijt, d_nkt) :: phlc_hcf_mf
         real, optional, intent(in), dimension(d_nijt, d_nkt) :: phli_hri_mf
         real, optional, intent(in), dimension(d_nijt, d_nkt) :: phli_hcf_mf
!
      end subroutine ice_adjust
!
   end interface
!
end module modi_ice_adjust

