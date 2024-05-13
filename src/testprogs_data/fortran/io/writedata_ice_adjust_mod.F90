MODULE WRITEDATA_ICE_ADJUST_MOD
    use netcdf
    USE OMP_LIB
    USE ARRAYS_MANIP, ONLY: SETUP, REPLICATE, NPROMIZE, INTERPOLATE, SET
    USE PARKIND1, ONLY: JPRD

    CONTAINS

    SUBROUTINE WRITEDATA_ICE_ADJUST (NPROMA, NGPBLKS, NFLEVG, LDVERBOSE)

    IMPLICIT NONE

    INTEGER, PARAMETER :: IFILE = 77

    INTEGER      :: KLON
    INTEGER      :: KIDIA
    INTEGER      :: KFDIA
    INTEGER      :: KLEV
    INTEGER      :: KRR
    INTEGER      :: KDUM

    LOGICAL, INTENT(IN) :: LDVERBOSE

    ! REAL, INTENT(OUT), ALLOCATABLE   :: PRHODJ_B       (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PEXNREF_B      (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PRHODREF_B     (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PPABSM_B       (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PTHT_B         (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: ZICE_CLD_WGT_B (:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: ZSIGQSAT_B     (:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PSIGS_B        (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PMFCONV_B      (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PRC_MF_B       (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PRI_MF_B       (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PCF_MF_B       (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: ZDUM1_B        (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: ZDUM2_B        (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: ZDUM3_B        (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: ZDUM4_B        (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: ZDUM5_B        (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PTHS_B         (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PRS_B          (:,:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PRS_OUT_B      (:,:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PSRCS_B        (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PSRCS_OUT_B    (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PCLDFR_B       (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PCLDFR_OUT_B   (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PHLC_HRC_B     (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PHLC_HRC_OUT_B (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PHLC_HCF_B     (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PHLC_HCF_OUT_B (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PHLI_HRI_B     (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PHLI_HRI_OUT_B (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PHLI_HCF_B     (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: PHLI_HCF_OUT_B (:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: ZRS_B          (:,:,:,:)
    ! REAL, INTENT(OUT), ALLOCATABLE   :: ZZZ_B          (:,:,:)

    REAL(KIND=JPRD), ALLOCATABLE   :: PRHODJ         (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PEXNREF        (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PRHODREF       (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PPABSM         (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PTHT           (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PSIGS          (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PMFCONV        (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PRC_MF         (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PRI_MF         (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PCF_MF         (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PTHS           (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PRS            (:,:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PRS_OUT        (:,:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PSRCS_OUT      (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PCLDFR_OUT     (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PHLC_HRC_OUT   (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PHLC_HCF_OUT   (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PHLI_HRI_OUT   (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: PHLI_HCF_OUT   (:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: ZRS            (:,:,:,:)
    REAL(KIND=JPRD), ALLOCATABLE   :: ZZZ            (:,:,:)

    INTEGER, INTENT(IN) :: NPROMA, NGPBLKS
    INTEGER :: NGPTOT
    INTEGER, INTENT(INOUT) :: NFLEVG
    INTEGER :: IOFF, IBL
    LOGICAL :: LLEXIST
    CHARACTER(LEN=32) :: CLFILE

    character (len = *), parameter :: FILE_NAME = "ref_fortran.nc"

    ! NIJ, NK
    integer, parameter :: NDIMS = 2

    ! When we create netCDF files, variables and dimensions, we get back
    ! an ID for each one.
    integer :: ncid, varid, dimids(NDIMS)
    integer :: ij_dimid, k_dimid, krr_dimids

    CALL SETUP()

    KRR=6
    NGPTOT = NPROMA * NGPBLKS

    IBL = 1
    WRITE (CLFILE, '("data/",I8.8,".dat")') IBL
    OPEN (IFILE, FILE=TRIM (CLFILE), FORM='UNFORMATTED')
    READ (IFILE) KLON, KDUM, KLEV
    CLOSE (IFILE)

    IF (NFLEVG < 0) NFLEVG = KLEV

    IOFF = 0
    IBL = 0
    LLEXIST = .TRUE.

    DO WHILE(LLEXIST)
      IBL = IBL + 1
      WRITE (CLFILE, '("data/",I8.8,".dat")') IBL

      INQUIRE (FILE=TRIM (CLFILE), EXIST=LLEXIST)

      IF (LDVERBOSE) PRINT *, TRIM (CLFILE)

      IF (.NOT. LLEXIST) EXIT

      OPEN (IFILE, FILE=TRIM (CLFILE), FORM='UNFORMATTED')

      READ (IFILE) KLON, KDUM, KLEV

      IF (IBL == 1) THEN
        ALLOCATE (PRHODJ       (NGPTOT,KLEV,1))
        ALLOCATE (PEXNREF      (NGPTOT,KLEV,1))
        ALLOCATE (PRHODREF     (NGPTOT,KLEV,1))
        ALLOCATE (PPABSM       (NGPTOT,KLEV,1))
        ALLOCATE (PTHT         (NGPTOT,KLEV,1))
        ALLOCATE (PSIGS        (NGPTOT,KLEV,1))
        ALLOCATE (PMFCONV      (NGPTOT,KLEV,1))
        ALLOCATE (PRC_MF       (NGPTOT,KLEV,1))
        ALLOCATE (PRI_MF       (NGPTOT,KLEV,1))
        ALLOCATE (PCF_MF       (NGPTOT,KLEV,1))
        ALLOCATE (PTHS         (NGPTOT,KLEV,1))
        ALLOCATE (PRS          (NGPTOT,KLEV,KRR,1))
        ALLOCATE (PRS_OUT      (NGPTOT,KLEV,KRR,1))
        ALLOCATE (PSRCS_OUT    (NGPTOT,KLEV,1))
        ALLOCATE (PCLDFR_OUT   (NGPTOT,KLEV,1))
        ALLOCATE (ZRS          (NGPTOT,KLEV,0:KRR,1))
        ALLOCATE (ZZZ          (NGPTOT,KLEV,1))
        ALLOCATE (PHLC_HRC_OUT (NGPTOT,KLEV,1))
        ALLOCATE (PHLC_HCF_OUT (NGPTOT,KLEV,1))
        ALLOCATE (PHLI_HRI_OUT (NGPTOT,KLEV,1))
        ALLOCATE (PHLI_HCF_OUT (NGPTOT,KLEV,1))
      ENDIF

      IF (IOFF+KLON > NGPTOT) THEN
        EXIT
      ENDIF

      READ (IFILE) PRHODJ       (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) PEXNREF      (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) PRHODREF     (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) PSIGS        (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) PMFCONV      (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) PPABSM       (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) ZZZ          (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) PCF_MF       (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) PRC_MF       (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) PRI_MF       (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) ZRS          (IOFF+1:IOFF+KLON,:,:,1)
      READ (IFILE) PRS          (IOFF+1:IOFF+KLON,:,:,1)
      READ (IFILE) PTHS         (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) PRS_OUT      (IOFF+1:IOFF+KLON,:,:,1)
      READ (IFILE) PSRCS_OUT    (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) PCLDFR_OUT   (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) PHLC_HRC_OUT (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) PHLC_HCF_OUT (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) PHLI_HRI_OUT (IOFF+1:IOFF+KLON,:,1)
      READ (IFILE) PHLI_HCF_OUT (IOFF+1:IOFF+KLON,:,1)

      CLOSE (IFILE)

      IOFF = IOFF + KLON

    ENDDO

    ! WRITE to netCDF

    ! Create file
    call check( nf90_create(FILE_NAME, NF90_CLOBBER, ncid) )

    !
    call check( nf90_def_dim(ncid, "IJ", NGPTOT, ij_dimid) )
    call check( nf90_def_dim(ncid, "K", KLEV, k_dimid) )
    ! call check( nf90_def_dim(ncid, "S", KRR, krr_dimid))

    call check( nf90_def_var(ncid, "PRHODJ", NF90_REAL, dimids, varid) )
    ! call check( nf90_def_var(ncid, "PEXNREF", NF90_REAL, dimids, varid + 1) )
    ! call check( nf90_def_var(ncid, "PSIGS", NF90_REAL, dimids, varid + 2) )
    ! call check( nf90_def_var(ncid, "PMFCONV", NF90_REAL, dimids, varid + 3) )
    ! call check( nf90_def_var(ncid, "PPABSM", NF90_REAL, dimids, varid + 4) )
    ! call check( nf90_def_var(ncid, "ZZZ", NF90_REAL, dimids, varid + 5) )
    ! call check( nf90_def_var(ncid, "PCF_MF", NF90_REAL, dimids, varid + 6) )
    ! call check( nf90_def_var(ncid, "PRC_MF", NF90_REAL, dimids, varid + 7) )
    ! call check( nf90_def_var(ncid, "PRI_MF", NF90_REAL, dimids, varid + 8) )
    ! ! call check( nf90_def_var(ncid, "ZRS", NF90_REAL, dimids, varid + 9) )          ! KRR + 1
    ! ! call check( nf90_def_var(ncid, "PRS", NF90_REAL, dimids, varid + 10) )         ! KRR
    ! call check( nf90_def_var(ncid, "PTHS", NF90_REAL, dimids, varid + 11) )
    ! ! call check( nf90_def_var(ncid, "PRS_OUT", NF90_REAL, dimids, varid + 12) )     ! KRR
    ! call check( nf90_def_var(ncid, "PSCRS_OUT", NF90_REAL, dimids, varid + 13) )
    ! call check( nf90_def_var(ncid, "PCLDFR_OUT", NF90_REAL, dimids, varid + 14) )
    ! call check( nf90_def_var(ncid, "PHLC_HRC_OUT", NF90_REAL, dimids, varid + 15) )
    ! call check( nf90_def_var(ncid, "PHLC_HCF_OUT", NF90_REAL, dimids, varid + 16) )
    ! call check( nf90_def_var(ncid, "PHLI_HRI_OUT", NF90_REAL, dimids, varid + 17) )
    ! call check( nf90_def_var(ncid, "PHLI_HCF_OUT", NF90_REAL, dimids, varid + 18) )



    ! End define mode. This tells netCDF we are done defining metadata.
    call check( nf90_enddef(ncid) )

    ! Write the pretend data to the file. Although netCDF supports
    ! reading and writing subsets of data, in this case we write all the
    ! data in one operation.
    call check( nf90_put_var(ncid, varid, PRHODJ) )
    ! call check( nf90_put_var(ncid, varid + 1, PEXNREF))
    ! call check( nf90_put_var(ncid, varid + 2, PSIGS) )
    ! call check( nf90_put_var(ncid, varid + 3, PMFCONV) )
    ! call check( nf90_put_var(ncid, varid + 4, PPABSM) )
    ! call check( nf90_put_var(ncid, varid + 5, ZZZ) )
    ! call check( nf90_put_var(ncid, varid + 6, PCF_MF) )
    ! call check( nf90_put_var(ncid, varid + 7, PRC_MF) )
    ! call check( nf90_put_var(ncid, varid + 8, PRI_MF) )
    ! ! call check( nf90_put_var(ncid, varid + 9, ZRS) )      ! KRR + 1
    ! ! call check( nf90_put_var(ncid, varid + 10, PRS) )      ! KRR
    ! call check( nf90_put_var(ncid, varid + 11, PTHS) )
    ! ! call check( nf90_put_var(ncid, varid + 12, PRS_OUT))   ! KRR
    ! call check( nf90_put_var(ncid, varid + 13, PSRCS_OUT))
    ! call check( nf90_put_var(ncid, varid + 14, PCLDFR_OUT))
    ! call check( nf90_put_var(ncid, varid + 15, PHLC_HRC_OUT))
    ! call check( nf90_put_var(ncid, varid + 16, PHLC_HCF_OUT))
    ! call check( nf90_put_var(ncid, varid + 17, PHLI_HRI_OUT))
    ! call check( nf90_put_var(ncid, varid + 18, PHLI_HCF_OUT))


    ! Close the file. This frees up any internal netCDF resources
    ! associated with the file, and flushes any buffers.
    call check( nf90_close(ncid) )

    END SUBROUTINE

    END  MODULE
