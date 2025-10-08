!     ######spl
      MODULE MODI_CONDENSATION
!     ########################
!
IMPLICIT NONE
INTERFACE
!
    SUBROUTINE CONDENSATION (D_NIJB, D_NIJE, D_NIJT, D_NKB, D_NKE, D_NKL, D_NKT, D_NKTB, D_NKTE, CST_XALPI, CST_XALPW,&
            &CST_XBETAI, CST_XBETAW, CST_XCI, CST_XCL, CST_XCPD, CST_XCPV, CST_XEPSILO, CST_XG, CST_XGAMI, CST_XGAMW,&
            &CST_XLSTT, CST_XLVTT, CST_XPI, CST_XRD, CST_XRV, CST_XTT, ICEP_XACRIAUTI, ICEP_XBCRIAUTI, ICEP_XCRIAUTC, &
            &ICEP_XCRIAUTI, ICEP_XFRMIN, NEBN_LCONDBORN, NEBN_LHGT_QS, NEBN_LSTATNW, NEBN_XTMAXMIX, NEBN_XTMINMIX, &
            &HFRAC_ICE, HCONDENS, HLAMBDA3, PPABS, PZZ, PRHODREF, PT, PRV_IN, PRV_OUT, PRC_IN, PRC_OUT, PRI_IN, &
            &PRI_OUT, PRR, PRS, PRG, PSIGS, LMFCONV, PMFCONV, PCLDFR, PSIGRC, OUSERI, OSIGMAS,  OCND2, PICLDFR, &
            &PWCLDFR, PSSIO, PSSIU, PIFR, PSIGQSAT, PLV, PLS, PCPH, PHLC_HRC, PHLC_HCF, PHLI_HRI, PHLI_HCF, &
            &PICE_CLD_WGT)
  !   ################################################################################
  !*       0.    DECLARATIONS
  !              ------------
  !
  USE MODE_TIWMX, ONLY: ESATW, ESATI
  USE MODE_ICECLOUD, ONLY: ICECLOUD
  !
  IMPLICIT NONE
  !
  !*       0.1   Declarations of dummy arguments :

  integer, intent(in) :: D_NIJB, D_NIJE, D_NIJT, D_NKB, D_NKE, D_NKL, D_NKT, D_NKTB, D_NKTE
  real, intent(in) :: CST_XALPI, CST_XALPW, CST_XBETAI, CST_XBETAW, CST_XCI, CST_XCL, CST_XCPD, CST_XCPV, CST_XEPSILO
  real, intent(in) :: CST_XG, CST_XGAMI, CST_XGAMW, CST_XLSTT, CST_XLVTT, CST_XPI, CST_XRD, CST_XRV, CST_XTT
  real, intent(in) :: ICEP_XACRIAUTI, ICEP_XBCRIAUTI, ICEP_XCRIAUTC, ICEP_XCRIAUTI
  real, dimension(25), intent(in) :: ICEP_XFRMIN
  real, intent(in) :: NEBN_XTMINMIX,NEBN_XTMAXMIX
  logical, intent(in) ::  NEBN_LCONDBORN, NEBN_LHGT_QS, NEBN_LSTATNW

  !
  !
  CHARACTER(LEN=1), INTENT(IN) :: HFRAC_ICE
  CHARACTER(LEN=4), INTENT(IN) :: HCONDENS
  CHARACTER(LEN=4), INTENT(IN) :: HLAMBDA3  ! formulation for lambda3 coeff
  REAL, INTENT(IN), DIMENSION(D_NIJT, D_NKT) :: PPABS  ! pressure (Pa)
  REAL, INTENT(IN), DIMENSION(D_NIJT, D_NKT) :: PZZ  ! height of model levels (m)
  REAL, INTENT(IN), DIMENSION(D_NIJT, D_NKT) :: PRHODREF
  REAL, INTENT(INOUT), DIMENSION(D_NIJT, D_NKT) :: PT  ! grid scale T  (K)
  REAL, INTENT(IN), DIMENSION(D_NIJT, D_NKT) :: PRV_IN  ! grid scale water vapor mixing ratio (kg/kg) in input
  REAL, INTENT(OUT), DIMENSION(D_NIJT, D_NKT) :: PRV_OUT  ! grid scale water vapor mixing ratio (kg/kg) in output
  REAL, INTENT(IN), DIMENSION(D_NIJT, D_NKT) :: PRC_IN  ! grid scale r_c mixing ratio (kg/kg) in input
  REAL, INTENT(OUT), DIMENSION(D_NIJT, D_NKT) :: PRC_OUT  ! grid scale r_c mixing ratio (kg/kg) in output
  REAL, INTENT(IN), DIMENSION(D_NIJT, D_NKT) :: PRI_IN  ! grid scale r_i (kg/kg) in input
  REAL, INTENT(OUT), DIMENSION(D_NIJT, D_NKT) :: PRI_OUT  ! grid scale r_i (kg/kg) in output
  REAL, INTENT(IN), DIMENSION(D_NIJT, D_NKT) :: PRR  ! grid scale mixing ration of rain (kg/kg)
  REAL, INTENT(IN), DIMENSION(D_NIJT, D_NKT) :: PRS  ! grid scale mixing ration of snow (kg/kg)
  REAL, INTENT(IN), DIMENSION(D_NIJT, D_NKT) :: PRG  ! grid scale mixing ration of graupel (kg/kg)
  REAL, INTENT(IN), DIMENSION(D_NIJT, D_NKT) :: PSIGS  ! Sigma_s from turbulence scheme
  LOGICAL, INTENT(IN) :: LMFCONV  !
  REAL, INTENT(IN), DIMENSION(D_NIJT, D_NKT) :: PMFCONV  ! convective mass flux (kg /s m^2)
  REAL, INTENT(OUT), DIMENSION(D_NIJT, D_NKT) :: PCLDFR  ! cloud fraction
  REAL, INTENT(OUT), DIMENSION(D_NIJT, D_NKT) :: PSIGRC  ! s r_c / sig_s^2

  LOGICAL, INTENT(IN) :: OUSERI  ! logical switch to compute both
  ! liquid and solid condensate (OUSERI=.TRUE.)
  ! or only solid condensate (OUSERI=.FALSE.)
  LOGICAL, INTENT(IN) :: OSIGMAS  ! use present global Sigma_s values
  ! or that from turbulence scheme
  LOGICAL, INTENT(IN) :: OCND2  ! logical switch to sparate liquid and ice
  ! more rigid (DEFALT value : .FALSE.)
  REAL, INTENT(OUT), DIMENSION(D_NIJT, D_NKT) :: PICLDFR  ! ice cloud fraction
  REAL, INTENT(OUT), DIMENSION(D_NIJT, D_NKT) :: PWCLDFR  ! water or mixed-phase cloud fraction
  REAL, INTENT(OUT), DIMENSION(D_NIJT, D_NKT) :: PSSIO  ! Super-saturation with respect to ice in the
  ! supersaturated fraction
  REAL, INTENT(OUT), DIMENSION(D_NIJT, D_NKT) :: PSSIU  ! Sub-saturation with respect to ice in the
  ! subsaturated fraction
  REAL, INTENT(OUT), DIMENSION(D_NIJT, D_NKT) :: PIFR  ! Ratio cloud ice moist part
  REAL, INTENT(IN), DIMENSION(D_NIJT) :: PSIGQSAT  ! use an extra "qsat" variance contribution (OSIGMAS case)
  ! multiplied by PSIGQSAT

  REAL, OPTIONAL, INTENT(IN), DIMENSION(D_NIJT, D_NKT) :: PLV  ! Latent heat L_v
  REAL, OPTIONAL, INTENT(IN), DIMENSION(D_NIJT, D_NKT) :: PLS  ! Latent heat L_s
  REAL, OPTIONAL, INTENT(IN), DIMENSION(D_NIJT, D_NKT) :: PCPH  ! Specific heat C_ph
  REAL, OPTIONAL, INTENT(OUT), DIMENSION(D_NIJT, D_NKT) :: PHLC_HRC
  REAL, OPTIONAL, INTENT(OUT), DIMENSION(D_NIJT, D_NKT) :: PHLC_HCF  ! cloud fraction
  REAL, OPTIONAL, INTENT(OUT), DIMENSION(D_NIJT, D_NKT) :: PHLI_HRI
  REAL, OPTIONAL, INTENT(OUT), DIMENSION(D_NIJT, D_NKT) :: PHLI_HCF
  REAL, OPTIONAL, INTENT(IN), DIMENSION(D_NIJT) :: PICE_CLD_WGT
!
END SUBROUTINE CONDENSATION
!
END INTERFACE
!
END MODULE MODI_CONDENSATION
