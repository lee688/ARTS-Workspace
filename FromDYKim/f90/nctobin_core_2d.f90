      MODULE Nctobin_core_2d

      CONTAINS

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      SUBROUTINE nc2bin_2d(ncfile, dat, vname)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      USE netcdf

      IMPLICIT NONE

      INTEGER                 :: i, j, k, z
      INTEGER                 :: ncid, nlon, nlat, nlev
      INTEGER                 :: istat, varid, dimid
      INTEGER*2, ALLOCATABLE  :: int16(:,:)
      INTEGER*2               :: badpix 

      REAL,PARAMETER          :: thefillvalue = -999.  ! marker value for the bad pixels.
      REAL*4                  :: offset, rate
      REAL*4, ALLOCATABLE     :: flt(:,:)
      REAL*4, ALLOCATABLE, OPTIONAL, INTENT(OUT) :: dat(:,:)

      CHARACTER*(*), INTENT(IN) :: ncfile
      CHARACTER*(*), INTENT(IN) :: vname

      ! open the netCDF file and obtain the fileID and main dimenstions.
        istat = nf90_open( TRIM(ncfile), nf90_nowrite, ncid)
          IF (istat /= nf90_noerr) STOP 'Error opening netCDF file'

        ! dimensions:
        istat = nf90_inq_dimid(ncid, "lon", dimid)
          IF (istat /= nf90_noerr) STOP 'Error finding "lon" dimension'
        istat = nf90_inquire_dimension(ncid, dimid, LEN=nlon)
          IF (istat /= nf90_noerr) STOP 'Error reading "lon" dimension'
        istat = nf90_inq_dimid(ncid, "lat", dimid)
          IF (istat /= nf90_noerr) STOP 'Error finding "lat" dimension'
        istat = nf90_inquire_dimension(ncid, dimid, LEN=nlat)
          IF (istat /= nf90_noerr) STOP 'Error reading "lat" dimension'

        write(*,*) trim(vname)
        write(*,*) "# of lon=",nlon
        write(*,*) "# of lat=",nlat
   
      ! dat
        istat = nf90_inq_varid(ncid, TRIM(vname), varid)
          IF(istat/=nf90_NoErr) STOP 'Error finding variable "t"'
        ALLOCATE(dat(nlon,nlat))
        ALLOCATE(flt(nlon,nlat))
        istat = nf90_get_var(ncid, varid, flt)
          IF(istat/=nf90_NoErr) STOP 'Error reading variable "t"'
        DO j=1,nlat
        DO i=1,nlon
           dat(i,j)=flt(i,j)
        END DO
        END DO

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      END SUBROUTINE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      END MODULE nctobin_core_2d
