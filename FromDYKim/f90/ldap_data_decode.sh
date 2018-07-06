#!/bin/bash

#=========== directory path and input data file setting =======
src_path=../PROG
out_path=../DAIO
data_path=../DATA
f_date=$1
f_hour=$2
#f_date=2017042800
#f_hour=000
#f3d_gb_name=l015_v070_erlo_pres_h${f_hour}.${f_date}.gb2
#f2d_gb_name=l015_v070_erlo_unis_h${f_hour}.${f_date}.gb2
#f3d_nc_name=l015_v070_erlo_pres_h${f_hour}.${f_date}.gb2.nc
#f2d_nc_name=l015_v070_erlo_unis_h${f_hour}.${f_date}.gb2.nc
#f3d_re_name=l015_v070_erlo_pres_h${f_hour}.${f_date}.rec.nc
#f2d_re_name=l015_v070_erlo_unis_h${f_hour}.${f_date}.rec.nc
f3d_gb_name=l015v070erlopresh${f_hour}.${f_date}.gb2
f2d_gb_name=l015v070erlounish${f_hour}.${f_date}.gb2
f3d_nc_name=l015v070erlopresh${f_hour}.${f_date}.gb2.nc
f2d_nc_name=l015v070erlounish${f_hour}.${f_date}.gb2.nc
f3d_re_name=l015v070erlopresh${f_hour}.${f_date}.rec.nc
f2d_re_name=l015v070erlounish${f_hour}.${f_date}.rec.nc
#=========== directory path and input data file setting =======

rm -rf ${data_path}/${f2d_nc_name}
rm -rf ${data_path}/${f3d_nc_name}
## convert grib2 -> nc
cdo -s -f nc setgridtype,curvilinear ${data_path}/${f2d_gb_name} ${data_path}/${f2d_nc_name}
cdo -s -f nc setgridtype,curvilinear ${data_path}/${f3d_gb_name} ${data_path}/${f3d_nc_name}
## crop and rectangular regrid
cdo -s remapbil,./ldaps_crop_rec.grid  ${data_path}/${f2d_nc_name} ${data_path}/${f2d_re_name}
cdo -s remapbil,./ldaps_crop_rec.grid  ${data_path}/${f3d_nc_name} ${data_path}/${f3d_re_name}


#=========== 1 vertical dimension netcdf data decoding =================
for var in lev
do
   echo "&file_info"                            >  ./nctobin_1d.nml
   echo "ncfile='${data_path}/${f3d_re_name}'"  >> ./nctobin_1d.nml
   echo "outfile='${out_path}/${var}.bin'"      >> ./nctobin_1d.nml
   echo "vname='${var}'"                        >> ./nctobin_1d.nml
   echo "/"                                     >> ./nctobin_1d.nml
   ${src_path}/nctobin_1d.exe
done 

#=========== 1 horizontal dimension netcdf data decoding =================
for var in lat lon
do
   echo "&file_info"                            >  ./nctobin_1d.nml
   echo "ncfile='${data_path}/${f3d_re_name}'"  >> ./nctobin_1d.nml
   echo "outfile='${out_path}/${var}.bin'"      >> ./nctobin_1d.nml
   echo "vname='${var}'"                        >> ./nctobin_1d.nml
   echo "/"                                     >> ./nctobin_1d.nml
   ${src_path}/nctobin_1d_2.exe
done 

#=========== 2 dimension netcdf data decoding =================
#for var in TSK PSFC U10 V10 LANDMASK
for var in t_2 sp 10u 10v lsm 
do
   echo "&file_info"                            >  ./nctobin_2d.nml
   echo "ncfile='${data_path}/${f2d_re_name}'"  >> ./nctobin_2d.nml
   echo "outfile='${out_path}/${var}.bin'"      >> ./nctobin_2d.nml
   echo "vname='${var}'"                        >> ./nctobin_2d.nml
   echo "/"                                     >> ./nctobin_2d.nml
   ${src_path}/nctobin_2d.exe
done

#=========== 3 dimension netcdf data(pres 24 levels) decoding ======
#for var in T P PB QVAPOR QCLOUD QICE QRAIN QSNOW QGRAUP QHAIL
#          T  GH humidity1, 2   u-wind v-wind w-wind
for var in t  gh r param194.1.0 u v param9.2.0
do
   echo "&file_info"                            >  ./nctobin_3d.nml
   echo "ncfile='${data_path}/${f3d_re_name}'"  >> ./nctobin_3d.nml
   echo "outfile='${out_path}/${var}.bin'"      >> ./nctobin_3d.nml
   echo "vname='${var}'"                        >> ./nctobin_3d.nml
   echo "/"                                     >> ./nctobin_3d.nml
   ${src_path}/nctobin_3d.exe
done

\rm -v ./nctobin_1d.nml ./nctobin_2d.nml ./nctobin_3d.nml ./nctobin_3d_2.nml
