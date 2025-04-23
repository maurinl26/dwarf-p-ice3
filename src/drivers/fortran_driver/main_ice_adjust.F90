program main_ice_adjust

   use xrd_getoptions, only: initoptions, getoption, checkoptions
   use getdata_ice_adjust_mod, only: getdata_ice_adjust
   use compute_diff, only: diff
   use modi_ice_adjust
   use modd_dimphyex, only: dimphyex_t
   use modd_phyex, only: phyex_t
   use stack_mod, only: stack
   use mode_mnh_zwork, only: zmnh_stack, imnh_block, ymnh_stack, inumpin
   use omp_lib
   use yomhook, only: lhook, dr_hook, jphook

! replacing misc structure
   use modd_budget, only: nbudget_rh, tbudgetdata, tbudgetconf_t, &
   & tbuconf_associate, nbudget_ri, tbuconf, lbu_enable, lbudget_u, lbudget_v, lbudget_w, lbudget_th, &
   & lbudget_tke, lbudget_rv, lbudget_rc, lbudget_rr, lbudget_ri, lbudget_rs, lbudget_rg, lbudget_rh, lbudget_sv
   use modi_ini_phyex, only: ini_phyex

   use modd_io, only: tfiledata

   implicit none

   logical           :: lmfconv
   integer           :: krr
   character(len=4)  :: hbuname
   real              :: ptstep
   logical           :: ocompute_src
   type(tbudgetdata), dimension(nbudget_rh) :: ylbudget
   integer                  :: nbudget

   ! for comparisons
   real :: mean_field, std_field

   integer      :: klev

   real, allocatable   :: prhodj(:, :, :)
   real, allocatable   :: pexnref(:, :, :)
   real, allocatable   :: prhodref(:, :, :)
   real, allocatable   :: ppabsm(:, :, :)
   real, allocatable   :: ptht(:, :, :)
   real, allocatable   :: psigs(:, :, :)
   real, allocatable   :: pmfconv(:, :, :)
   real, allocatable   :: prc_mf(:, :, :)
   real, allocatable   :: pri_mf(:, :, :)
   real, allocatable   :: pcf_mf(:, :, :)
   real, allocatable   :: pths(:, :, :)
   real, allocatable   :: prs(:, :, :, :)
   real, allocatable   :: psrcs(:, :, :)
   real, allocatable   :: pcldfr(:, :, :)
   real, allocatable   :: phlc_hrc(:, :, :)
   real, allocatable   :: phlc_hcf(:, :, :)
   real, allocatable   :: phli_hri(:, :, :)
   real, allocatable   :: phli_hcf(:, :, :)
   real, allocatable   :: zrs(:, :, :, :)
   real, allocatable   :: zzz(:, :, :)
   real, allocatable   :: zsigqsat(:, :)
   real, allocatable   :: zice_cld_wgt(:, :)
   real, allocatable   :: zdum1(:, :, :)
   real, allocatable   :: zdum2(:, :, :)
   real, allocatable   :: zdum3(:, :, :)
   real, allocatable   :: zdum4(:, :, :)
   real, allocatable   :: zdum5(:, :, :)

   real, allocatable   :: prs_out(:, :, :, :)
   real, allocatable   :: psrcs_out(:, :, :)
   real, allocatable   :: pcldfr_out(:, :, :)
   real, allocatable   :: phlc_hrc_out(:, :, :)
   real, allocatable   :: phlc_hcf_out(:, :, :)
   real, allocatable   :: phli_hri_out(:, :, :)
   real, allocatable   :: phli_hcf_out(:, :, :)

   integer :: nproma, ngpblks, nflevg
   integer :: jlon, jlev
   integer, target :: ibl

   type(dimphyex_t)         :: d, d0
   type(phyex_t)            :: phyex
   logical                  :: llcheck
   logical                  :: llcheckdiff
   logical                  :: lldiff
   integer                  :: iblock1, iblock2
   integer                  :: istsz(2), jblk1, jblk2
   integer                  :: ntid, itid

   real, allocatable, target :: pstack8(:, :)
   real(kind=4), allocatable, target :: pstack4(:, :)
   type(stack), target :: ylstack

   real(kind=8) :: ts, te
   real(kind=8) :: tsc, tec, tsd, ted, ztc, ztd
   integer :: itime, ntime
   integer :: irank, isize
   logical :: llverbose, llstat, llbind
   real(kind=jphook) :: zhook_handle

!!!!!!!!! dummy allocation !!!!!!!!!!!!!!!!!!!!!!!!
   integer :: nit = 50
   integer :: nkt = 15
   real(kind=8) :: vsigqsat = 0.02

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! replacing misc
   type(tfiledata)          :: tpfile
!

!-----------------------------------------------------------------------
!    local variables
   integer :: iulout, jrr
   real :: zdzmin
   character(len=6) :: cprogram
   character(len=4) :: cmicro, csconv, cturb
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   ! todo : dsl declare
   

   print *, "debug : main_ice_adjust.f90 - launching main_ice_adjust"

   call initoptions()
   ngpblks = 296
   call getoption("--blocks", ngpblks)
   nproma = 32
   call getoption("--nproma", nproma)
   nflevg = -1
   call getoption("--nflevg", nflevg)
   call getoption("--check", llcheck)
   llcheckdiff = .false.
   call getoption("--checkdiff", llcheckdiff)
   iblock1 = 1
   call getoption("--check-block-1", iblock1)
   iblock2 = ngpblks
   call getoption("--check-block-2", iblock2)
   call getoption("--stat", llstat)
   ntime = 1
   call getoption("--times", ntime)
   llverbose = .true. ! default behaviour
   call getoption("--verbose", llverbose)
   call getoption("--bind", llbind)
   call checkoptions()

   print *, "debug : main_ice_adjust.f90 - cli args"
   print *, "debug : main_ice_adjust.f90 - blocks=", ngpblks, ";nproma=", nproma, ";nflevg=", nflevg, ";times=", ntime

   lldiff = .false.

   irank = 0
   isize = 1
   if (llbind) then
      call linux_bind(irank, isize)
      call linux_bind_dump(irank, isize)
   end if

   print *, "debug : main_ice_adjust.F90 - call GETDATA_ICE_ADJUST"

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!! GETDATA_ICE_ADJUST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   call getdata_ice_adjust(nproma, ngpblks, nflevg, prhodj, pexnref, prhodref, ppabsm, ptht, zice_cld_wgt,     &
   & zsigqsat, psigs, pmfconv, prc_mf, pri_mf, pcf_mf, zdum1, zdum2, zdum3, zdum4, zdum5, &
   & pths, prs, psrcs, pcldfr, phlc_hrc, phlc_hcf, &
   & phli_hri, phli_hcf, zrs, zzz, prs_out, psrcs_out, pcldfr_out, phlc_hrc_out, phlc_hcf_out,       &
   & phli_hri_out, phli_hcf_out, llverbose)

   klev = size(prs, 2)
   krr = size(prs, 3)

   print *, "debug : main_ice_adjust.f90 - klev = ", klev, " krr = ", krr, " nflevg=", nflevg
   print *, "debug : main_ice_adjust.F90 - NPROMA = ", NPROMA, " NGPBLKS = ", NGPBLKS

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!! CALL INIT_PHYEX(KRR, PHYEX)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   IULOUT = 20
   CPROGRAM = 'AROME'
   ZDZMIN = 20.
   CMICRO = 'ICE3'
   CSCONV = 'NONE'
   CTURB = 'TKEL'
   PTSTEP = 50.000000000000000
   TPFILE%NLU = 0

   print *, "debug : main_ice_adjust.F90 - start ini_phyex, allocating default values"

   call ini_phyex(hprogram=cprogram, tpfile=tpfile, ldneednam=.true., kluout=iulout, kfrom=0, kto=1, &
                 &ptstep=ptstep, pdzmin=zdzmin, &
                 &cmicro=cmicro, cturb=cturb, csconv=csconv, ldchangemodel=.false., &
                 &lddefaultval=.true., ldreadnam=.false., ldcheck=.false., kprint=0, ldinit=.true., &
                 &phyex_out=phyex)

   print *, "debug : main_ice_adjust.F90 - end ini_phyex, allocating default values"

!control parameters
! phyex misc removed
   hbuname = 'depi'
   lmfconv = .true.
   ocompute_src = .true.

!emulate the namelist reading
   phyex%param_icen%lcriauti = .true.
   phyex%param_icen%xcriauti_nam = 0.2e-3
   phyex%param_icen%xt0criauti_nam = -5.
   phyex%param_icen%xcriautc_nam = 0.1e-2
   phyex%param_icen%locnd2 = .false.
   phyex%param_icen%csedim = 'stat'
   phyex%param_icen%lwarm = .true.
   phyex%param_icen%lsedic = .true.
   phyex%param_icen%csnowriming = 'm90 '
   phyex%param_icen%xfracm90 = 0.1 ! fraction used for the murakami 1990 formulation
   phyex%param_icen%lconvhg = .true. ! true to allow the conversion from hail to graupel
   phyex%param_icen%lcrflimit = .true. !true to limit rain contact freezing to possible heat exchange
   phyex%param_icen%lfeedbackt = .true. ! when .true. feed back on temperature is taken into account
   phyex%param_icen%levlimit = .true.   ! when .true. water vapour pressure is limited by saturation
   phyex%param_icen%lnullwetg = .true.  ! when .true. graupel wet growth is activated with null rate (to allow water shedding)
   phyex%param_icen%lwetgpost = .true.  ! when .true. graupel wet growth is activated with positive temperature (to allow water shedding)
   phyex%param_icen%lnullweth = .true.  ! same as lnullwetg but for hail
   phyex%param_icen%lwethpost = .true.  ! same as lwetgpost but for hail
   phyex%param_icen%lsedim_after = .false. ! sedimentation done after microphysics
   phyex%param_icen%xsplit_maxcfl = 0.8
   phyex%param_icen%ldeposc = .false.  ! water deposition on vegetation
   phyex%param_icen%xvdeposc = 0.02    ! deposition speed (2 cm.s-1)
   phyex%param_icen%csubg_rc_rr_accr = 'none'
   phyex%param_icen%csubg_rr_evap = 'none'
   phyex%param_icen%csubg_pr_pdf = 'sigm'
   phyex%nebn%lsubg_cond = .true.
   phyex%nebn%lsigmas = .true.
   phyex%nebn%cfrac_ice_adjust = 't' ! ice/liquid partition rule to use in adjustment
   phyex%nebn%cfrac_ice_shallow_mf = 's' ! ice/liquid partition rule to use in shallow_mf

!budgets
   call tbuconf_associate
   do jrr = 1, nbudget
      ylbudget(jrr)%nbudget = jrr
   end do
 

! MISC removed
   TBUCONF = TBUCONF
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!! END CALL INIT(PHYEX, KRR)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   D0%NIT = NPROMA
   D0%NIB = 1
   D0%NIE = NPROMA
   D0%NJT = 1
   D0%NJB = 1
   D0%NJE = 1
   D0%NIJT = D0%NIT*D0%NJT
   D0%NIJB = 1
   D0%NIJE = NPROMA
   D0%NKL = -1
   D0%NKT = KLEV
   D0%NKA = KLEV
   D0%NKU = 1
   D0%NKB = KLEV
   D0%NKE = 1
   D0%NKTB = 1
   D0%NKTE = KLEV

   DO ITIME = 1, NTIME

      DO IBL = JBLK1, JBLK2
         

        print *, "debug : main_ice_adjust.F90 - call ice_adjust without USE_STACK"
        print *, "debug : main_ice_adjust.F90 - IBL = ", IBL
        print *, "debug : main_ice_adjust.F90 - size of pexnref ", size(PEXNREF)

            
        print *, "debug : main_ice_adjust.F90 - end ice_adjust"
      
      END DO
   END DO


   TE = OMP_GET_WTIME()

   STOP

END PROGRAM