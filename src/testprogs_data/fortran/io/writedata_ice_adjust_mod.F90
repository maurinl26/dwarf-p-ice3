MODULE WRITEDATA_ICE_ADJUST_MOD
    use netCDF
    USE OMP_LIB
    USE ARRAYS_MANIP, ONLY: SETUP, REPLICATE, NPROMIZE, INTERPOLATE, SET
    USE PARKIND1, ONLY: JPRD

    CONTAINS

    SUBROUTINE WRITEDATA_ICE_ADJUST (NPROMA, NGPBLKS, NFLEVG, PRHODJ_B, PEXNREF_B, PRHODREF_B, PPABSM_B, PTHT_B, ZICE_CLD_WGT_B, &
    & ZSIGQSAT_B, PSIGS_B, PMFCONV_B, PRC_MF_B, PRI_MF_B, PCF_MF_B, ZDUM1_B, ZDUM2_B, ZDUM3_B, ZDUM4_B, ZDUM5_B, PTHS_B, PRS_B, PSRCS_B, PCLDFR_B, PHLC_HRC_B, PHLC_HCF_B,   &
    & PHLI_HRI_B, PHLI_HCF_B, ZRS_B, ZZZ_B, PRS_OUT_B, PSRCS_OUT_B, PCLDFR_OUT_B, PHLC_HRC_OUT_B, PHLC_HCF_OUT_B,         &
    & PHLI_HRI_OUT_B, PHLI_HCF_OUT_B, LDVERBOSE)

    IMPLICIT NONE

    INTEGER, PARAMETER :: IFILE = 77

    INTEGER      :: KLON
    INTEGER      :: KIDIA
    INTEGER      :: KFDIA
    INTEGER      :: KLEV
    INTEGER      :: KRR
    INTEGER      :: KDUM

    LOGICAL, INTENT(IN) :: LDVERBOSE

    REAL, INTENT(OUT), ALLOCATABLE   :: PRHODJ_B       (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PEXNREF_B      (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PRHODREF_B     (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PPABSM_B       (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PTHT_B         (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: ZICE_CLD_WGT_B (:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: ZSIGQSAT_B     (:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PSIGS_B        (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PMFCONV_B      (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PRC_MF_B       (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PRI_MF_B       (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PCF_MF_B       (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: ZDUM1_B        (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: ZDUM2_B        (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: ZDUM3_B        (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: ZDUM4_B        (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: ZDUM5_B        (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PTHS_B         (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PRS_B          (:,:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PRS_OUT_B      (:,:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PSRCS_B        (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PSRCS_OUT_B    (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PCLDFR_B       (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PCLDFR_OUT_B   (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PHLC_HRC_B     (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PHLC_HRC_OUT_B (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PHLC_HCF_B     (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PHLC_HCF_OUT_B (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PHLI_HRI_B     (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PHLI_HRI_OUT_B (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PHLI_HCF_B     (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: PHLI_HCF_OUT_B (:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: ZRS_B          (:,:,:,:)
    REAL, INTENT(OUT), ALLOCATABLE   :: ZZZ_B          (:,:,:)

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

    ALLOCATE (ZSIGQSAT_B      (NPROMA,NGPBLKS))
    ALLOCATE (ZICE_CLD_WGT_B  (NPROMA,NGPBLKS))
    ALLOCATE (PSRCS_B         (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PCLDFR_B        (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PHLC_HRC_B      (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PHLC_HCF_B      (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PHLI_HRI_B      (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PHLI_HCF_B      (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PRHODJ_B        (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PEXNREF_B       (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PRHODREF_B      (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PPABSM_B        (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PTHT_B          (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PSIGS_B         (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PMFCONV_B       (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PRC_MF_B        (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PRI_MF_B        (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PCF_MF_B        (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (ZDUM1_B         (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (ZDUM2_B         (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (ZDUM3_B         (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (ZDUM4_B         (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (ZDUM5_B         (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PTHS_B          (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PRS_B           (NPROMA,NFLEVG,KRR,NGPBLKS))
    ALLOCATE (PRS_OUT_B       (NPROMA,NFLEVG,KRR,NGPBLKS))
    ALLOCATE (PSRCS_OUT_B     (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PCLDFR_OUT_B    (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (ZRS_B           (NPROMA,NFLEVG,0:KRR,NGPBLKS))
    ALLOCATE (ZZZ_B           (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PHLC_HRC_OUT_B  (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PHLC_HCF_OUT_B  (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PHLI_HRI_OUT_B  (NPROMA,NFLEVG,NGPBLKS))
    ALLOCATE (PHLI_HCF_OUT_B  (NPROMA,NFLEVG,NGPBLKS))

    CALL SET (ZSIGQSAT_B    )
    CALL SET (ZICE_CLD_WGT_B)
    CALL SET (PSRCS_B       )
    CALL SET (PCLDFR_B      )
    CALL SET (PHLC_HRC_B    )
    CALL SET (PHLC_HCF_B    )
    CALL SET (PHLI_HRI_B    )
    CALL SET (PHLI_HCF_B    )
    CALL SET (PRHODJ_B      )
    CALL SET (PEXNREF_B     )
    CALL SET (PRHODREF_B    )
    CALL SET (PPABSM_B      )
    CALL SET (PTHT_B        )
    CALL SET (PSIGS_B       )
    CALL SET (PMFCONV_B     )
    CALL SET (PRC_MF_B      )
    CALL SET (PRI_MF_B      )
    CALL SET (PCF_MF_B      )
    CALL SET (PTHS_B        )
    CALL SET (PRS_B         )
    CALL SET (PRS_OUT_B     )
    CALL SET (PSRCS_OUT_B   )
    CALL SET (PCLDFR_OUT_B  )
    CALL SET (ZRS_B         )
    CALL SET (ZZZ_B         )
    CALL SET (PHLC_HRC_OUT_B)
    CALL SET (PHLC_HCF_OUT_B)
    CALL SET (PHLI_HRI_OUT_B)
    CALL SET (PHLI_HCF_OUT_B)



    ZSIGQSAT_B     = 2.0000000000000000E-002
    ZICE_CLD_WGT_B = 1.5

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

    call check( nf90_def_var(ncid, "PRHODJ", NF90_INT, dimids, varid) )
    call check( nf90_def_var(ncid, "PEXNREF", NF90_INT, dimids, varid + 1) )
    call check( nf90_def_var(ncid, "PSIGS", NF90_INT, dimids, varid + 2) )
    call check( nf90_def_var(ncid, "PMFCONV", NF90_INT, dimids, varid + 3) )
    call check( nf90_def_var(ncid, "PPABSM", NF90_INT, dimids, varid + 4) )
    call check( nf90_def_var(ncid, "ZZZ", NF90_INT, dimids, varid + 5) )
    call check( nf90_def_var(ncid, "PCF_MF", NF90_INT, dimids, varid + 6) )
    call check( nf90_def_var(ncid, "PRC_MF", NF90_INT, dimids, varid + 7) )
    call check( nf90_def_var(ncid, "PRI_MF", NF90_INT, dimids, varid + 8) )
    ! call check( nf90_def_var(ncid, "ZRS", NF90_INT, dimids, varid + 9) )          ! KRR + 1
    ! call check( nf90_def_var(ncid, "PRS", NF90_INT, dimids, varid + 10) )         ! KRR
    call check( nf90_def_var(ncid, "PTHS", NF90_INT, dimids, varid + 11) )
    ! call check( nf90_def_var(ncid, "PRS_OUT", NF90_INT, dimids, varid + 12) )     ! KRR
    call check( nf90_def_var(ncid, "PSCRS_OUT", NF90_INT, dimids, varid + 13) )
    call check( nf90_def_var(ncid, "PCLDFR_OUT", NF90_INT, dimids, varid + 14) )
    call check( nf90_def_var(ncid, "PHLC_HRC_OUT", NF90_INT, dimids, varid + 15) )
    call check( nf90_def_var(ncid, "PHLC_HCF_OUT", NF90_INT, dimids, varid + 16) )
    call check( nf90_def_var(ncid, "PHLI_HRI_OUT", NF90_INT, dimids, varid + 17) )
    call check( nf90_def_var(ncid, "PHLI_HCF_OUT", NF90_INT, dimids, varid + 18) )



    ! End define mode. This tells netCDF we are done defining metadata.
    call check( nf90_enddef(ncid) )

    ! Write the pretend data to the file. Although netCDF supports
    ! reading and writing subsets of data, in this case we write all the
    ! data in one operation.
    call check( nf90_put_var(ncid, varid, PRHODJ) )
    call check( nf90_put_var(ncid, varid + 1, PEXNREF))
    call check( nf90_put_var(ncid, varid + 2, PSIGS) )
    call check( nf90_put_var(ncid, varid + 3, PMFCONV) )
    call check( nf90_put_var(ncid, varid + 4, PPABSM) )
    call check( nf90_put_var(ncid, varid + 5, ZZZ) )
    call check( nf90_put_var(ncid, varid + 6, PCF_MF) )
    call check( nf90_put_var(ncid, varid + 7, PRC_MF) )
    call check( nf90_put_var(ncid, varid + 8, PRI_MF) )
    ! call check( nf90_put_var(ncid, varid + 9, ZRS) )      ! KRR + 1
    ! call check( nf90_put_var(ncid, varid + 10, PRS) )      ! KRR
    call check( nf90_put_var(ncid, varid + 11, PTHS) )
    ! call check( nf90_put_var(ncid, varid + 12, PRS_OUT))   ! KRR
    call check( nf90_put_var(ncid, varid + 13, PSRCS_OUT))
    call check( nf90_put_var(ncid, varid + 14, PCLDFR_OUT))
    call check( nf90_put_var(ncid, varid + 15, PHLC_HRC_OUT))
    call check( nf90_put_var(ncid, varid + 16, PHLC_HCF_OUT))
    call check( nf90_put_var(ncid, varid + 17, PHLI_HRI_OUT))
    call check( nf90_put_var(ncid, varid + 18, PHLI_HCF_OUT))


    ! Close the file. This frees up any internal netCDF resources
    ! associated with the file, and flushes any buffers.
    call check( nf90_close(ncid) )

    ! IF (NFLEVG /= KLEV) THEN
    !   CALL INTERPOLATE (NFLEVG, IOFF, PRHODJ      )
    !   CALL INTERPOLATE (NFLEVG, IOFF, PEXNREF     )
    !   CALL INTERPOLATE (NFLEVG, IOFF, PRHODREF    )
    !   CALL INTERPOLATE (NFLEVG, IOFF, PSIGS       )
    !   CALL INTERPOLATE (NFLEVG, IOFF, PMFCONV     )
    !   CALL INTERPOLATE (NFLEVG, IOFF, PPABSM      )
    !   CALL INTERPOLATE (NFLEVG, IOFF, ZZZ         )
    !   CALL INTERPOLATE (NFLEVG, IOFF, PCF_MF      )
    !   CALL INTERPOLATE (NFLEVG, IOFF, PRC_MF      )
    !   CALL INTERPOLATE (NFLEVG, IOFF, PRI_MF      )
    !   CALL INTERPOLATE (NFLEVG, IOFF, ZRS         )
    !   CALL INTERPOLATE (NFLEVG, IOFF, PRS         )
    !   CALL INTERPOLATE (NFLEVG, IOFF, PTHS        )
    !   CALL INTERPOLATE (NFLEVG, IOFF, PRS_OUT     )
    !   CALL INTERPOLATE (NFLEVG, IOFF, PSRCS_OUT   )
    !   CALL INTERPOLATE (NFLEVG, IOFF, PCLDFR_OUT  )
    !   CALL INTERPOLATE (NFLEVG, IOFF, PHLC_HRC_OUT)
    !   CALL INTERPOLATE (NFLEVG, IOFF, PHLC_HCF_OUT)
    !   CALL INTERPOLATE (NFLEVG, IOFF, PHLI_HRI_OUT)
    !   CALL INTERPOLATE (NFLEVG, IOFF, PHLI_HCF_OUT)
    ! ENDIF

    ! CALL REPLICATE (IOFF, PRHODJ       (:, :, 1))
    ! CALL REPLICATE (IOFF, PEXNREF      (:, :, 1))
    ! CALL REPLICATE (IOFF, PRHODREF     (:, :, 1))
    ! CALL REPLICATE (IOFF, PSIGS        (:, :, 1))
    ! CALL REPLICATE (IOFF, PMFCONV      (:, :, 1))
    ! CALL REPLICATE (IOFF, PPABSM       (:, :, 1))
    ! CALL REPLICATE (IOFF, ZZZ          (:, :, 1))
    ! CALL REPLICATE (IOFF, PCF_MF       (:, :, 1))
    ! CALL REPLICATE (IOFF, PRC_MF       (:, :, 1))
    ! CALL REPLICATE (IOFF, PRI_MF       (:, :, 1))
    ! CALL REPLICATE (IOFF, ZRS          (:, :, :, 1))
    ! CALL REPLICATE (IOFF, PRS          (:, :, :, 1))
    ! CALL REPLICATE (IOFF, PTHS         (:, :, 1))
    ! CALL REPLICATE (IOFF, PRS_OUT      (:, :, :, 1))
    ! CALL REPLICATE (IOFF, PSRCS_OUT    (:, :, 1))
    ! CALL REPLICATE (IOFF, PCLDFR_OUT   (:, :, 1))
    ! CALL REPLICATE (IOFF, PHLC_HRC_OUT (:, :, 1))
    ! CALL REPLICATE (IOFF, PHLC_HCF_OUT (:, :, 1))
    ! CALL REPLICATE (IOFF, PHLI_HRI_OUT (:, :, 1))
    ! CALL REPLICATE (IOFF, PHLI_HCF_OUT (:, :, 1))

    ! CALL NPROMIZE (NPROMA, PRHODJ      ,  PRHODJ_B        )
    ! CALL NPROMIZE (NPROMA, PEXNREF     ,  PEXNREF_B       )
    ! CALL NPROMIZE (NPROMA, PRHODREF    ,  PRHODREF_B      )
    ! CALL NPROMIZE (NPROMA, PSIGS       ,  PSIGS_B         )
    ! CALL NPROMIZE (NPROMA, PMFCONV     ,  PMFCONV_B       )
    ! CALL NPROMIZE (NPROMA, PPABSM      ,  PPABSM_B        )
    ! CALL NPROMIZE (NPROMA, ZZZ         ,  ZZZ_B           )
    ! CALL NPROMIZE (NPROMA, PCF_MF      ,  PCF_MF_B        )
    ! CALL NPROMIZE (NPROMA, PRC_MF      ,  PRC_MF_B        )
    ! CALL NPROMIZE (NPROMA, PRI_MF      ,  PRI_MF_B        )
    ! CALL NPROMIZE (NPROMA, ZRS         ,  ZRS_B           )
    ! CALL NPROMIZE (NPROMA, PRS         ,  PRS_B           )
    ! CALL NPROMIZE (NPROMA, PTHS        ,  PTHS_B          )
    ! CALL NPROMIZE (NPROMA, PRS_OUT     ,  PRS_OUT_B       )
    ! CALL NPROMIZE (NPROMA, PSRCS_OUT   ,  PSRCS_OUT_B     )
    ! CALL NPROMIZE (NPROMA, PCLDFR_OUT  ,  PCLDFR_OUT_B    )
    ! CALL NPROMIZE (NPROMA, PHLC_HRC_OUT,  PHLC_HRC_OUT_B  )
    ! CALL NPROMIZE (NPROMA, PHLC_HCF_OUT,  PHLC_HCF_OUT_B  )
    ! CALL NPROMIZE (NPROMA, PHLI_HRI_OUT,  PHLI_HRI_OUT_B  )
    ! CALL NPROMIZE (NPROMA, PHLI_HCF_OUT,  PHLI_HCF_OUT_B  )

    END SUBROUTINE

    END  MODULE
