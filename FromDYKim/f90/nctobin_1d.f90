      PROGRAM NctoBin

      USE Nctobin_core_1d

      IMPLICIT NONE

      INTEGER*4 :: nlon, nlat, i, j
      INTEGER*8 :: time

      REAL*4, ALLOCATABLE :: dat(:)

      CHARACTER(LEN=120) :: ncfile, vname
      CHARACTER(LEN=120) :: outfile

      NAMELIST / file_info / ncfile, vname, outfile
      OPEN(99,FILE='nctobin_1d.nml')
      READ(99,file_info)
     
      PRINT*,'Reading ',TRIM(ncfile)
      PRINT*,'Varname ',TRIM(vname)

      CALL nc2bin_1d( TRIM(ncfile), dat, vname)

      PRINT*, 'T00 : ', dat

      OPEN(2,FILE=TRIM(outfile),FORM='unformatted')
      WRITE(2) dat

      DEALLOCATE(dat)

      END PROGRAM NctoBin
