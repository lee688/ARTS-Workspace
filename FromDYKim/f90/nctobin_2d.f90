      PROGRAM NctoBin

      USE Nctobin_core_2d

      IMPLICIT NONE

      INTEGER*4 :: nlon, nlat, i, j
      INTEGER*8 :: time

      REAL*4, ALLOCATABLE :: dat(:,:)

      CHARACTER(LEN=120) :: ncfile, vname
      CHARACTER(LEN=120) :: outfile

      NAMELIST / file_info / ncfile, vname, outfile
      OPEN(99,FILE='nctobin_2d.nml')
      READ(99,file_info)
     
      PRINT*,'Reading ',TRIM(ncfile)
      PRINT*,'Varname ',TRIM(vname)

      CALL nc2bin_2d( TRIM(ncfile), dat, vname)

      PRINT*, 'Min : ', MINVAL(dat,dat/=-999)
      PRINT*, 'Max : ', MAXVAL(dat)

      OPEN(2,FILE=TRIM(outfile),FORM='unformatted')
      WRITE(2) dat(:,:)

      DEALLOCATE(dat)

      END PROGRAM NctoBin
