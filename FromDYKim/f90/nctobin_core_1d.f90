      MODULE Nctobin_core_1d

      CONTAINS

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      SUBROUTINE nc2bin_1d(ncfile, dat, vname)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      USE netcdf

      IMPLICIT NONE

      INTEGER                 :: i, j, k, z
      INTEGER                 :: ncid, nlon, nlat, nlev
      INTEGER                 :: time
      INTEGER                 :: istat, varid, dimid
      INTEGER*2, ALLOCATABLE  :: int16(:)
      INTEGER*2               :: badpix 

      REAL,PARAMETER          :: thefillvalue = -999.  ! marker value for the bad pixels.
      REAL*4                  :: offset, rate
      REAL*4, ALLOCATABLE     :: flt(:)
      REAL*4, ALLOCATABLE, OPTIONAL, INTENT(OUT) :: dat(:)

      CHARACTER*(*), INTENT(IN) :: ncfile
      CHARACTER*(*), INTENT(IN) :: vname

      ! open the netCDF file and obtain the fileID and main dimenstions.
        istat = nf90_open( TRIM(ncfile), nf90_nowrite, ncid)
          IF (istat /= nf90_noerr) STOP 'Error opening netCDF file'

      ! dimensions: 
        istat = nf90_inq_dimid(ncid, "lev", dimid)
          IF (istat /= nf90_noerr) STOP 'Error finding "time" dimension'
        istat = nf90_inquire_dimension(ncid, dimid, LEN=nlev)
          IF (istat /= nf90_noerr) STOP 'Error reading "time" dimension' 

        write(*,*) trim(vname)
        write(*,*) "# of nlev=",nlev
        
      ! dat
        istat = nf90_inq_varid(ncid, TRIM(vname), varid)
          IF(istat/=nf90_NoErr) STOP 'Error finding variable'
        print*, nlev
        ALLOCATE(dat(nlev))
        ALLOCATE(flt(nlev))

        istat = nf90_get_var(ncid, varid, flt)
          IF(istat/=nf90_NoErr) STOP 'Error reading variable'
        DO z=1,nlev
           dat(z)=flt(z)
        END DO

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      END SUBROUTINE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      END MODULE nctobin_core_1d
