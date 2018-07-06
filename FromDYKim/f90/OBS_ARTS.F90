       PROGRAM RTmodl

       IMPLICIT NONE

       INTEGER, PARAMETER  :: nx = 248, ny = 223, nz = 24
       INTEGER, PARAMETER  :: isf= 12, isz = 1, isd = 1   !!(frequency, altitude, scan deg.)
       INTEGER, PARAMETER  :: pz = 1
       INTEGER             :: i, j, k

       REAL, PARAMETER     :: Er = 6378000.
       REAL                :: SVP, AVP, zen, value, QF, QL, CTEMP, hPa
       REAL, ALLOCATABLE, DIMENSION(:)     :: P, LON, LAT
       REAL, ALLOCATABLE, DIMENSION(:,:)   :: ST, SP, SZ, NUL
       REAL, ALLOCATABLE, DIMENSION(:,:)   :: TBB1, TBB2, TBB3
       REAL, ALLOCATABLE, DIMENSION(:,:,:) :: T, H
       REAL, ALLOCATABLE, DIMENSION(:,:,:) :: QCL, QCF, Mix_QCL, Mix_QCF, QVAPO, ND_WAT, ND_ICE, ND_TOT 

       REAL f_grid(isf)
       data f_grid/ 18.700e9, 22.235e9,                     &
                    37.000e9, 40.000e9,                     & 
                    50.300e9, 52.300e9, 53.600e9, 54.550e9, &
                    55.750e9, 57.000e9, 58.400e9, 59.800e9/ 
       REAL     :: sat_lat, sat_lon, sat_alt, deg2rad, rad2deg, psat_lat,psat_lon, rdeg 
       REAL     :: min_dist, dist, temp(100), sec, conv_lat, conv_lon
       INTEGER  :: loc_i, loc_j, kk, date, hour, min

!       REAL sat_alt(isz)
!       data sat_alt/ 0.1e3,  1.0e3,  2.0e3,  3.0e3,  4.0e3, &
!                     5.0e3,  6.0e3,  7.0e3,  8.0e3,  9.0e3/
       REAL, ALLOCATABLE, DIMENSION(:)   :: sat_zen, sat_azi
       REAL, ALLOCATABLE, DIMENSION(:)   :: tlon, tlat, angdist
!       REAL, PARAMETER     :: sat_lat = 36.77661, sat_lon = 125.49391
!       REAL     :: sat_lat, sat_lon
       NAMELIST / file_info / sat_lat, sat_lon, sat_alt, zen, &
                              date, hour, min, sec, &
                              psat_lat, psat_lon
       real(8), parameter :: pi = 4*atan(1.0d0)

       OPEN(9,FILE='../DAIO/obs_art_input.nml',status='old')
       READ(9,file_info) 

       OPEN(10,FILE='../DAIO/lev.bin',FORM='unformatted')       !z level  

       OPEN(20,FILE='../DAIO/sp.bin',FORM='unformatted')             !2D Surface Pressure 
       OPEN(30,FILE='../DAIO/t_2.bin',FORM='unformatted')            !2D Surface Temp.
       OPEN(31,FILE='../DAIO/lon.bin',FORM='unformatted')            !2D longitude
       OPEN(32,FILE='../DAIO/lat.bin',FORM='unformatted')            !2D latitude

       OPEN(40,FILE='../DAIO/t.bin',FORM='unformatted')              !3D Temp.   
       OPEN(50,FILE='../DAIO/gh.bin',FORM='unformatted')             !3D Geopotential Height 
       OPEN(70,FILE='../DAIO/param194.1.0.bin',FORM='unformatted')   !3D RH wrt. ice 
       OPEN(80,FILE='../DAIO/r.bin',FORM='unformatted')              !3D RH wrt. water 

       ALLOCATE( P(nz) )
       ALLOCATE( LON(nx) )
       ALLOCATE( LAT(ny) )

       ALLOCATE( SP(nx, ny) )
       ALLOCATE( ST(nx, ny) )
       ALLOCATE( SZ(nx, ny) )
       ALLOCATE( NUL(nx, ny) )

       ALLOCATE( T(nx, ny, nz)  )
       ALLOCATE( H(nx, ny, nz)  )
       ALLOCATE( QCF(nx, ny, nz)  )
       ALLOCATE( QCL(nx, ny, nz)  )
       ALLOCATE( Mix_QCF(nx, ny, nz)  )
       ALLOCATE( Mix_QCL(nx, ny, nz)  )
       ALLOCATE( QVAPO(nx, ny, nz)  )
       ALLOCATE( ND_WAT(nx, ny, nz)  )
       ALLOCATE( ND_ICE(nx, ny, nz)  )
       ALLOCATE( ND_TOT(nx, ny, nz)  )

       ALLOCATE( sat_zen(isd) )
       ALLOCATE( sat_azi(isd) )
       ALLOCATE( angdist(isd) )
       ALLOCATE( tlat(isd) )
       ALLOCATE( tlon(isd) )

       ALLOCATE( TBB1(isf,isd) )
       ALLOCATE( TBB2(isf,isd) )
       ALLOCATE( TBB3(isf,isd) )

       if(zen <= 180) then
        sat_zen(1)=zen
        sat_azi(1)=90
       else
        sat_zen(1)=360.-zen
        sat_azi(1)=-90
       endif
       deg2rad=pi/180.
       rad2deg=180./pi
       rdeg= (pi/2.) - atan2( (sat_lat - psat_lat), (sat_lon - psat_lon) )   ! -1 * angle from old to new , which to transform coordination
       angdist(1) = sat_alt * 0.001 * tan( (180.- zen)*deg2rad )
       conv_lon= 1. / (2.*3.14*6400.*cos(sat_lat * deg2rad)/360.)
       conv_lat= 1. / (2.*3.14*6400./360.)

       tlon(1)= sat_lon + angdist(1)*conv_lon*cos(rdeg)
       tlat(1)= sat_lat + angdist(1)*conv_lat*sin(rdeg)

       write(*,*) i+1, sat_lon, sat_lat, tlon(1), tlat(1)
       write(*,*) i+1, rdeg, angdist(1), conv_lon, conv_lat 


       H  = 0 
       T  = 0 
       P  = 0 
 
       READ(10) P
       READ(20) SP
       READ(30) ST
       READ(31) LON
       READ(32) LAT

       CLOSE(10)
       CLOSE(20)
       CLOSE(30)
       CLOSE(31)
       CLOSE(32)

       DO k = 1, nz
       READ(40) T(:,:,k)
       READ(50) H(:,:,k)
       READ(70) QCF(:,:,k)
       READ(80) QCL(:,:,k)
       END DO

       CLOSE(40)
       CLOSE(50)
       CLOSE(70)
       CLOSE(80)

         DO j = 1, ny
         DO i = 1, nx
          NUL(i,j) = 0.
!          WRITE(*,*) i,j, SP(i,j)

         DO k = 1, nz 
          !! Pressure, Temp. Etc  convert
          H(i,j,k)=  H(i,j,k)/0.98                                        ! Geopotential Height -> Dynamic Height
          ND_WAT(i,j,k) = QCL(i,j,k)*1.60771704e6*P(k)*100*(18.01528/1000) &    ! ???? 
                          / (8.314472*T(i,j,k))
          ND_ICE(i,j,k) = QCF(i,j,k)*1.60771704e6*P(k)*100*(18.01528/1000) &    ! ????
                          / (8.314472*T(i,j,k))
          if ( ND_WAT(i,j,k) < 0 ) ND_WAT(i,j,k) = 0
          if ( ND_ICE(i,j,k) < 0 ) ND_ICE(i,j,k) = 0
          ND_TOT(i,j,k)= ND_WAT(i,j,k)+ND_ICE(i,j,k)
!          write(*,*) i,j,k,QCF(i,j,k),QCL(i,j,k)
          if (i <= 3 .or. i >= nx-2 ) ND_TOT(i,j,k)=0.
          if (j <= 3 .or. j >= ny-2 ) ND_TOT(i,j,k)=0.
          if (k <= 3 .or. k >= nz-2 ) ND_TOT(i,j,k)=0.

          CTEMP=T(i,j,k)-273.15
          hPa=P(k)/100 
          SVP = 6.112**(17.62*CTEMP/(243.12+CTEMP))*(1.0016+hPa*3.15*0.000001-0.074/hPa) ! saturation vapor pressure(https://planetcalc.com/2167/) 
          Mix_QCF(i,j,k)= SVP*QCF(i,j,k)/(461.5*T(i,j,k))
          Mix_QCL(i,j,k)= SVP*QCL(i,j,k)/(461.5*T(i,j,k))

         END DO
          if ( SP(i,j) < 100100 ) SP(i,j) = 100100. 
         END DO
         END DO

!FIX-2D/3D!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          !!lon  (LON)
          OPEN(101,FILE='../DAIO/lon_grid.xml')
          WRITE(101,*) '<?xml version="1.0"?>'
          WRITE(101,*) '<arts format="ascii" version="1">'
          WRITE(101,*) '<Vector nelem="248">'
             WRITE(101,'(f20.10)') LON(2)-20
          DO k = 2, nx-1
             WRITE(101,'(f20.10)') LON(k)
          END DO
             WRITE(101,'(f20.10)') LON(nx-1)+20
          WRITE(101,*) '</Vector>'
          WRITE(101,*) '</arts>'
          CLOSE(101)

          !!lat  (LAT)
          OPEN(102,FILE='../DAIO/lat_grid.xml')
          WRITE(102,*) '<?xml version="1.0"?>'
          WRITE(102,*) '<arts format="ascii" version="1">'
          WRITE(102,*) '<Vector nelem="223">'
             WRITE(102,'(f20.10)') LAT(2)-20
          DO k = 2, ny-1
             WRITE(102,'(f20.10)') LAT(k)
          END DO
             WRITE(102,'(f20.10)') LAT(ny-1)+20
          WRITE(102,*) '</Vector>'
          WRITE(102,*) '</arts>'
          CLOSE(102)

          !! z_surface.xml
          OPEN(103,FILE='../DAIO/z_surface.xml')
          WRITE(103,*) '<?xml version="1.0"?>'
          WRITE(103,*) '<arts format="ascii" version="1">'
          WRITE(103,*) '<Matrix npages="1" nrows="223" ncols="248">'
          DO j = 1, ny
          WRITE(103,'(248f10.1)') (NUL(i,j),i=1,248)
          END DO
          WRITE(103,*) '</Matrix>'
          WRITE(103,*) '</arts>'
          CLOSE(103)
    
          !! t_surface.xml
          OPEN(104,FILE='../DAIO/t_surface.xml')
          WRITE(104,*) '<?xml version="1.0"?>'
          WRITE(104,*) '<arts format="ascii" version="1">'
          WRITE(104,*) '<Matrix npages="1" nrows="223" ncols="248">'
          DO j = 1, ny
          WRITE(104,'(248f10.1)') (ST(i,j),i=1,248)
          END DO
          WRITE(104,*) '</Matrix>'
          WRITE(104,*) '</arts>'
          CLOSE(104)

          !! t.xml
          OPEN(105,FILE='../DAIO/t_field.xml')
          WRITE(105,*) '<?xml version="1.0"?>'
          WRITE(105,*) '<arts format="ascii" version="1">'
          WRITE(105,*) '<Tensor3 npages="25" nrows="223" ncols="248">'
          DO j = 1, ny
          WRITE(105,'(248f10.1)') (ST(i,j),i=1,248)
          END DO
          DO k = 1, nz
          DO j = 1, ny
          WRITE(105,'(248f10.1)') (T(i,j,k),i=1,248)
          ENDDO
          ENDDO
          WRITE(105,*) '</Tensor3>'
          WRITE(105,*) '</arts>'
          CLOSE(105)

          !! z.xml
          OPEN(106,FILE='../DAIO/z_field.xml')
          WRITE(106,*) '<?xml version="1.0"?>'
          WRITE(106,*) '<arts format="ascii" version="1">'
          WRITE(106,*) '<Tensor3 npages="25" nrows="223" ncols="248">'
          DO j = 1, ny
          WRITE(106,'(248f10.1)') (NUL(i,j),i=1,248)
          END DO
          DO k = 1, nz
          DO j = 1, ny
          WRITE(106,'(248f10.1)') (H(i,j,k),i=1,248)
          ENDDO
          ENDDO
          WRITE(106,*) '</Tensor3>'
          WRITE(106,*) '</arts>'
          CLOSE(106)

          !! pnd particle number density(?)
          OPEN(107,FILE='../DAIO/pnd_field.xml')
          WRITE(107,*) '<?xml version="1.0"?>'
          WRITE(107,*) '<arts format="ascii" version="1">'
          WRITE(107,*) '<Tensor4 nbooks="1" npages="23" nrows="221" ncols="246">'
          DO k = 1, nz-1
          DO j = 2, ny-1
          WRITE(107,'(246e)') (ND_TOT(i,j,k),i=2,247)
          ENDDO
          ENDDO
          WRITE(107,*) '</Tensor4>'
          WRITE(107,*) '</arts>'
          CLOSE(107)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! find i, j location
      if (sat_lat < 32.45000 .or. sat_lat > 35.44700) then
      write(*,*) "out of simulation range sat_lat 32.45000 ~ 35.44700 :",sat_lat 
      stop
      end if
      if (sat_lon < 125.5000 .or. sat_lat > 128.8345) then
      write(*,*) "out of simulation range sat_lon 125.5000 ~ 128.8345 :",sat_lon 
      stop
      end if
      if (sat_alt < 100.0000 .or. sat_alt > 30000.00) then
      write(*,*) "out of simulation range sat_alt 100.00 ~ 30000.00 :",sat_alt
      stop
      end if

      min_dist=10000000.
      DO I = 1, NX
      DO J = 1, NY
      CALL HAVERSINE_FORMULA(LON(I),LAT(J),SAT_LON, SAT_LAT,DIST)
      if(dist < min_dist) then
        min_dist=dist
        LOC_I=I
        LOC_J=J
      end if  
!      write(*,*) i,j,min_dist,dist,loc_i,loc_j
      END DO
      END DO

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! build input data 

      j = LOC_J
      i = LOC_I
          !!p_grid
          OPEN(111,FILE='../DAIO/p_grid.xml')
          WRITE(111,*) '<?xml version= "1.0"?>'
          WRITE(111,*) '<arts format="ascii" version="1">'
          WRITE(111,*) '<Vector nelem="25">'
          WRITE(111,'(f10.1)') SP(i,j)
          DO k = 1, nz
             WRITE(111,'(f10.1)') P(k)
          END DO
          WRITE(111,*) '</Vector>'
          WRITE(111,*) '</arts>'
          CLOSE(111)

          !! t.xml
          OPEN(112,FILE='../DAIO/tropical.t.xml')
          WRITE(112,*) '<?xml version="1.0"?>'
          WRITE(112,*) '<arts format="ascii" version="1">'
          WRITE(112,*) '<GriddedField3>'
          WRITE(112,*) '<Vector nelem="25" name="Pressure">'
          WRITE(112,'(f10.1)') SP(i,j)
          DO k = 1, nz
             WRITE(112,'(f10.1)') P(k)
          END DO
          WRITE(112,*) '</Vector>'
          WRITE(112,*) '<Vector nelem="1" name="Latitude">'
          WRITE(112,*) 0
          WRITE(112,*) '</Vector>'
          WRITE(112,*) '<Vector nelem="1" name="Longitude">'
          WRITE(112,*) 0
          WRITE(112,*) '</Vector>'
          WRITE(112,*) '<Tensor3 npages="25" nrows="1" ncols="1">'
          WRITE(112,'(f10.1)') ST(i,j)
          DO k = 1, nz
             WRITE(112,'(f10.1)') T(i,j,k)
          END DO
          WRITE(112,*) '</Tensor3>'
          WRITE(112,*) '</GriddedField3>'
          WRITE(112,*)  '</arts>'
          CLOSE(112)
 
          !! z.xml
          OPEN(113,FILE='../DAIO/tropical.z.xml')
          WRITE(113,*) '<?xml version="1.0"?>'
          WRITE(113,*) '<arts format="ascii" version="1">'
          WRITE(113,*) '<GriddedField3>'
          WRITE(113,*) '<Vector nelem="25" name="Pressure">'
          WRITE(113,'(f10.1)') SP(i,j)
          DO k = 1, nz
             WRITE(113,'(f10.1)') P(k)
          END DO
          WRITE(113,*) '</Vector>'
          WRITE(113,*) '<Vector nelem="1" name="Latitude">'
          WRITE(113,*) 0
          WRITE(113,*) '</Vector>'
          WRITE(113,*) '<Vector nelem="1" name="Longitude">'
          WRITE(113,*) 0
          WRITE(113,*) '</Vector>'
          WRITE(113,*) '<Tensor3 npages="25" nrows="1" ncols="1">'
          WRITE(113,'(f10.1)') 0.
          DO k = 1, nz
             WRITE(113,'(f10.1)') H(i,j,k)
          END DO
          WRITE(113,*) '</Tensor3>'
          WRITE(113,*) '</GriddedField3>'
          WRITE(113,*)  '</arts>' 
          CLOSE(113)

          !!p_grid
          OPEN(114,FILE='../DAIO/tropical.H2O.xml')
          WRITE(114,*) '<?xml version= "1.0"?>'
          WRITE(114,*) '<arts format="ascii" version="1">'
          WRITE(114,*) '<GriddedField3>'
          WRITE(114,*) '<Vector nelem="25" name="Pressure">'
          WRITE(114,'(f10.1)') SP(i,j)
          DO k = 1, nz
             WRITE(114,'(f10.1)') P(k)
          END DO
          WRITE(114,*) '</Vector>'
          WRITE(114,*) '<Vector nelem="1" name="Latitude">'
          WRITE(114,*) '                   0'
          WRITE(114,*) '</Vector>'
          WRITE(114,*) '<Vector nelem="1" name="Longitude">'
          WRITE(114,*) '                   0'
          WRITE(114,*) '</Vector>'
          WRITE(114,*) '<Tensor3 npages="25" nrows="1" ncols="1">'
          WRITE(114,*) Mix_QCL(i,j,1)+Mix_QCF(i,j,1)
          DO k = 1, nz
             WRITE(114,*) Mix_QCL(i,j,k)+Mix_QCF(i,j,k) 
          END DO
          WRITE(114,*) '</Tensor3>'
          WRITE(114,*) '</GriddedField3>'
          WRITE(114,*) '</arts>' 
          CLOSE(114)


          write(*,*) "Cal. prepare end"

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! CAL ARTS

       OPEN(21,FILE='../DAOU/OBS_ARTS_ClearSky3D_OUT.dat',status='unknown')
       OPEN(22,FILE='../DAOU/OBS_ARTS_MC3D_OUT.dat',status='unknown')
!       WRITE(21,'(A)') "  I     J  DATE     HOUR  MIN   SAT_LON      SAT_LAT    SAT_ALT  |==> TB1 ~ 12 (Zen 0 ~ 360 : int 12deg)"
!       WRITE(22,'(A)') "  I     J  DATE     HOUR  MIN   SAT_LON      SAT_LAT    SAT_ALT  |==> TB1 ~ 12 (Zen 0 ~ 360 : int 12deg)"

          !! f_grid.xml
          OPEN(120,FILE='../DAIO/f_grid.xml')
          WRITE(120,*) '<?xml version="1.0"?>'
          WRITE(120,*) '<arts format="ascii" version="1">'
          WRITE(120,*) '<Vector nelem="12">'
          DO i = 1, isf
          WRITE(120,'(e)') f_grid(i)
          END DO
          WRITE(120,*) '</Vector>'
          WRITE(120,*) '</arts>'
          CLOSE(120)

          !! sensor_pos.xml
          OPEN(120,FILE='../DAIO/sensor_pos.xml')
          WRITE(120,*) '<?xml version="1.0"?>'
          WRITE(120,*) '<arts format="ascii" version="1">'
          WRITE(120,*) '<Matrix nrows="1" ncols="3">'
          DO j = 1, isd
          WRITE(120,'(3f15.5)') sat_alt, sat_lat, sat_lon
          END DO
          WRITE(120,*) '</Matrix>'
          WRITE(120,*) '</arts>'
          CLOSE(120)

          !! sensor_los.xml
          OPEN(120,FILE='../DAIO/sensor_los.xml')
          WRITE(120,*) '<?xml version="1.0"?>'
          WRITE(120,*) '<arts format="ascii" version="1">'
          WRITE(120,*) '<Matrix nrows="1" ncols="2">'
          DO j = 1, isd
          WRITE(120,'(2f15.5)') sat_zen(j), sat_azi(j)
          END DO
          WRITE(120,*) '</Matrix>'
          WRITE(120,*) '</arts>'
          CLOSE(120)

          CALL SYSTEM('../PROG/arts2.2 -r000 OBS_ARTS_ClearSky3D.arts') 
          OPEN(1000,FILE='OBS_ARTS_ClearSky3D.xml')
          READ(1000,*)
          READ(1000,*)
          READ(1000,*)
          DO j = 1, isd
          DO i = 1, isf
          READ(1000,*) TBB1(i,j)
!          WRITE(21,'(2i5,i9,2i5,4f12.5,e12.5,3f12.5)') LOC_I,LOC_J,DATE,HOUR,MIN,SEC,sat_lon,sat_lat,sat_alt,f_grid(i),tlon(j),tlat(j),TBB1(i,j)
          END DO
          END DO
          WRITE(21,'(2i5,i9,2i5,4f12.5,3f12.5,12f12.5)') LOC_I,LOC_J,DATE,HOUR,MIN,SEC,sat_lon,sat_lat,sat_alt,zen,tlon(1),tlat(1),(TBB1(i,1),i=1,isf)

          CLOSE(1000)


          CALL SYSTEM('../PROG/arts2.2 -r000 OBS_ARTS_MC3D.arts') 
          OPEN(1001,FILE='OBS_ARTS_MC3D.xml')
          READ(1001,*)
          READ(1001,*)
          READ(1001,*)
          DO j = 1, isd
          DO i = 1, isf
          READ(1001,*) TBB2(i,j)
!          WRITE(22,'(2i5,i9,2i5,4f12.5,e12.5,3f12.5)') LOC_I,LOC_J,DATE,HOUR,MIN,SEC,sat_lon,sat_lat,sat_alt,f_grid(i),tlon(j),tlat(j),TBB2(i,j)
          END DO
          END DO
          CLOSE(1001)
          WRITE(22,'(2i5,i9,2i5,4f12.5,3f12.5,12f12.5)') LOC_I,LOC_J,DATE,HOUR,MIN,SEC,sat_lon,sat_lat,sat_alt,zen,tlon(1),tlat(1),(TBB2(i,1),i=1,isf)


!          WRITE(21,'(2i5,i9,2i5,3f12.5,360f12.5)') LOC_I,LOC_J,DATE,HOUR,MIN,sat_lon,sat_lat,sat_alt,((TBB1(i,j),i=1,isf),j=1,isd)
!          WRITE(22,'(2i5,i9,2i5,3f12.5,360f12.5)') LOC_I,LOC_J,DATE,HOUR,MIN,sat_lon,sat_lat,sat_alt,((TBB2(i,j),i=1,isf),j=1,isd)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
9999   write(*,*) "TIP Cal. end"


       END PROGRAM RTmodl 

      subroutine haversine_formula(lon1,lat1,lon2,lat2,dist)
      implicit none
      real,intent(in)::lon1,lon2,lat1,lat2
      real,intent(out)::dist
      real,parameter::pi=3.141592,mean_earth_radius=6371.0088
      real::lonr1,lonr2,latr1,latr2
      real::delangl,dellon,dellat,a
      lonr1=lon1*(pi/180.);lonr2=lon2*(pi/180.)
      latr1=lat1*(pi/180.);latr2=lat2*(pi/180.)
      dellon=lonr2-lonr1
      dellat=latr2-latr1
      a=(sin(dellat/2))**2+cos(latr1)*cos(latr2)*(sin(dellon/2))**2
      delangl=2*asin(sqrt(a)) !2*asin(sqrt(a))
      dist=delangl*mean_earth_radius
      end subroutine

