{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os, re, sys, glob\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from matplotlib import pyplot as plt\n",
    "import inspect"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define a function to see the source code of a function. \n",
    "def showfunc(functiontoshow):\n",
    "    print(inspect.getsource(functiontoshow))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/user/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.\n",
      "  from ._conv import register_converters as _register_converters\n"
     ]
    }
   ],
   "source": [
    "import typhon as tp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Assume WGS 1984 for the reference Ellipsoid.\n",
    "R_eq = 6378137 # Earth's equatorial radius, in meters\n",
    "iFlttn = 298.257223563 # Inverse flattening\n",
    "R_polar = R_eq * (1-1/iFlttn) # Earth's polar radius\n",
    "eccnty = (2/iFlttn - (1/iFlttn)**2)**0.5 # Eccentricity of the ellipsoid "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data month and day\n",
    "dada = '07-03'\n",
    "\n",
    "# Data hour of day\n",
    "daho = '09'\n",
    "\n",
    "# Files location\n",
    "dain = os.path.join(os.getcwd(),dada)\n",
    "\n",
    "# Observation/simulation time\n",
    "TimeOfInterest = pd.Timestamp('2018-' + dada + ' ' + daho + ':00:00+0000',tz='UTC')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Read in radimeters data. \n",
    "\n",
    "# Functions for reading in radiometers data. Written by Dr. Choi. \n",
    "class radiometrics:\n",
    "    def read_lv0_data(self,f, *args):\n",
    "\n",
    "        if not f:\n",
    "            print(\"Missing argument FN, Please provide input filename.\")\n",
    "        if len(args) == 0:\n",
    "            tmp_dir = os.path.dirname(f)\n",
    "            dat_idx = '15'\n",
    "        else:\n",
    "            if len(args) == 1:\n",
    "                dat_idx = args[0]\n",
    "                tmp_dir = os.path.dirname(f)\n",
    "            else:\n",
    "                dat_idx = args[0]\n",
    "                tmp_dir = args[1]\n",
    "\n",
    "        dain = os.path.dirname(f)\n",
    "        fn = os.path.basename(f)\n",
    "\n",
    "        fn_sep = re.split('[. _]', fn)\n",
    "        if fn_sep[2] != 'lv0':\n",
    "            print(\"=============================================================================\")\n",
    "            print(\" Given file is not Level2 data. Please try again with lv0 file. Returning...\")\n",
    "            print(\"=============================================================================\")\n",
    "            return -1\n",
    "\n",
    "        f1 = \"_\".join(fn_sep[:len(fn_sep)-1]) + \"_{0}\".format( dat_idx ) + \".csv\"\n",
    "        file_exists = os.path.isfile( os.path.join(dain, f1) )\n",
    "        print(file_exists) \n",
    "        if not file_exists:\n",
    "            bsmwr = radiometrics()\n",
    "            bsmwr.prepare_original(f)\n",
    "\n",
    "        fin = os.path.join(dain, f1)\n",
    "\n",
    "        delimiters = (\"+\", \"-\", \"%\", \"/\", '|', \":\", \" \", \"(\", \")\")\n",
    "        # Get headers (column titles)\n",
    "        regexPattern = '|'.join(map(re.escape, delimiters))\n",
    "        with open( fin, 'r' ) as d:\n",
    "                line = d.readline()     # Read the first line in file\n",
    "                h0 = re.split(',|\\n', line)\n",
    "        d.close()\n",
    "        # Read data\n",
    "        df = pd.read_csv(fin, skiprows=1,names=h0)\n",
    "\n",
    "        # Make time understood in Pandas\n",
    "        df[\"DateTime\"] = pd.to_datetime(df[\"DateTime\"], format='%m/%d/%Y %H:%M:%S', utc=True)\n",
    "\n",
    "        self.df = df\n",
    "        return df\n",
    "\n",
    "    def read_lv1_data(self,f, *args):\n",
    "\n",
    "        if not f:\n",
    "            print(\"Missing argument FN, Please provide input filename.\")\n",
    "        if len(args) == 0:\n",
    "            tmp_dir = os.path.dirname(f)\n",
    "            dat_idx = '50'\n",
    "        else:\n",
    "            if len(args) == 1:\n",
    "                dat_idx = args[0]\n",
    "                tmp_dir = os.path.dirname(f)\n",
    "            else:\n",
    "                dat_idx = args[0]\n",
    "                tmp_dir = args[1]\n",
    "\n",
    "        dain = os.path.dirname(f)\n",
    "        fn = os.path.basename(f)\n",
    "\n",
    "        fn_sep = re.split('[. _]', fn)\n",
    "        if fn_sep[2] != 'lv1':\n",
    "            print(\"=============================================================================\")\n",
    "            print(\" Given file is not Level2 data. Please try again with lv1 file. Returning...\")\n",
    "            print(\"=============================================================================\")\n",
    "            return -1\n",
    "\n",
    "        f1 = \"_\".join(fn_sep[:len(fn_sep)-1]) + \"_{0}\".format( dat_idx ) + \".csv\"\n",
    "        file_exists = os.path.isfile( os.path.join(dain, f1) )\n",
    "        print(file_exists) \n",
    "        if not file_exists:\n",
    "            bsmwr = radiometrics()\n",
    "            bsmwr.prepare_original(f)\n",
    "\n",
    "        fin = os.path.join(dain, f1)\n",
    "\n",
    "        delimiters = (\"+\", \"-\", \"%\", \"/\", '|', \":\", \" \", \"(\", \")\")\n",
    "        # Get headers (column titles)\n",
    "        regexPattern = '|'.join(map(re.escape, delimiters))\n",
    "        with open( fin, 'r' ) as d:\n",
    "                line = d.readline()     # Read the first line in file\n",
    "                h0 = re.split(',|\\n', line)\n",
    "        d.close()\n",
    "        # Read data\n",
    "        df = pd.read_csv(fin, skiprows=1,names=h0)\n",
    "\n",
    "        # Make time understood in Pandas\n",
    "        df[\"DateTime\"] = pd.to_datetime(df[\"DateTime\"], format='%m/%d/%y %H:%M:%S', utc=True)\n",
    "\n",
    "        self.df = df\n",
    "        return df\n",
    "\n",
    "    def read_lv2_data(self,f,*args):\n",
    "\n",
    "        if not f:\n",
    "            print(\"Missing argument FN, Please provide input filename.\")\n",
    "        if len(args) == 0:\n",
    "            tmp_dir = os.path.dirname(f)\n",
    "            dat_idx = '400'\n",
    "        else:\n",
    "            if len(args) == 1:\n",
    "                dat_idx = args[0]\n",
    "                tmp_dir = os.path.dirname(f)\n",
    "            else:\n",
    "                dat_idx = args[0]\n",
    "                tmp_dir = args[1]\n",
    "\n",
    "        dain = os.path.dirname(f)\n",
    "        fn = os.path.basename(f)\n",
    "\n",
    "        fn_sep = re.split('[. _]', fn)\n",
    "        if fn_sep[2] != 'lv2':\n",
    "            print(\"=============================================================================\")\n",
    "            print(\" Given file is not Level2 data. Please try again with lv2 file. Returning...\")\n",
    "            print(\"=============================================================================\")\n",
    "            return -1\n",
    "\n",
    "        f1 = \"_\".join(fn_sep[:len(fn_sep)-1]) + \"_{0}\".format( dat_idx ) + \".csv\"\n",
    "        file_exists = os.path.isfile( os.path.join(dain, f1) )\n",
    "        print(file_exists) \n",
    "        if not file_exists:\n",
    "            bsmwr = radiometrics()\n",
    "            bsmwr.prepare_original(f)\n",
    "\n",
    "        fin = os.path.join(dain, f1)\n",
    "\n",
    "        delimiters = (\"+\", \"-\", \"%\", \"/\", '|', \":\", \" \", \"(\", \")\")\n",
    "        # Get headers (column titles)\n",
    "        regexPattern = '|'.join(map(re.escape, delimiters))\n",
    "        with open( fin, 'r' ) as d:\n",
    "                line = d.readline()     # Read the first line in file\n",
    "                h0 = re.split(',|\\n', line)\n",
    "        d.close()\n",
    "        # Read data\n",
    "        df = pd.read_csv(fin, skiprows=1,names=h0)\n",
    "\n",
    "        # Make time understood in Pandas\n",
    "        df[\"DateTime\"] = pd.to_datetime(df[\"DateTime\"], format='%m/%d/%y %H:%M:%S', utc=True)\n",
    "\n",
    "        self.df = df\n",
    "        return df\n",
    "    \n",
    "    def prepare_original(self, fn, *args):\n",
    "\n",
    "        if not fn:\n",
    "            print(\"Missing argument FN, Please provide input filename.\")\n",
    "        if len(args) == 0:\n",
    "            tmp_dir = os.path.dirname(fn)\n",
    "        else:\n",
    "            tmp_dir = args[0]\n",
    "        fn_sep = fn.split(\".\")\n",
    "\n",
    "        dataidx = []\n",
    "        datanum = []\n",
    "        delimiters = (\"+\", \"-\", \"%\", \"/\", '|', \":\", \" \", \"(\", \")\")\n",
    "        regexPattern = '|'.join(map(re.escape, delimiters))\n",
    "        with open( fn, 'r' ) as d:\n",
    "            while True:\n",
    "                line = d.readline()     # Read the first line in file\n",
    "                if not line: \n",
    "                    self.dataidx = [0]\n",
    "                    return dataidx\n",
    "                    break\n",
    "                h0 = re.split(',|\\n', line)\n",
    "                nh = len( h0 ) - 1      # Number of headers. Ignore last header since it's '/n'\n",
    "                # \n",
    "                # Remove inappropriate letters for BIN filename\n",
    "                # \n",
    "                h1 = [ \"\".join( re.split(regexPattern, h0[i]) ) for i in range(0,nh) ]\n",
    "                                        #---------------------------\n",
    "                if h1[0] == \"Record\":   # Header for each data type\n",
    "                                        #---------------------------\n",
    "                    dataidx.append( h1[2] )\n",
    "                    datanum.append(1)\n",
    "\n",
    "                    foun = \".\".join(fn_sep[:len(fn_sep)-1]) + \"_{0}\".format( h1[2] ) + '.csv'\n",
    "                    fou = open( os.path.join(tmp_dir, foun), 'w' )                     # Open input file\n",
    "                    fou.write( \",\".join(h1) + \"\\n\" )                                # Write data (Add \\n for new line\n",
    "                    fou.close()                                                     # Close input file\n",
    "                                        #-------------------------\n",
    "                else:                   # Data for each data type\n",
    "                                        #-------------------------\n",
    "                    if h1[2] == '99':\n",
    "                        if dataidx.count('99') == 0:\n",
    "                            dataidx.append( h1[2] )\n",
    "                            datanum.append(1)\n",
    "                            file_op_index = 'w'\n",
    "                        else:\n",
    "                            datanum[dataidx.index('99')] = datanum[dataidx.index('99')] + 1\n",
    "                            file_op_index = 'a'\n",
    "\n",
    "                            foun = \".\".join(fn_sep[:len(fn_sep)-1]) + \"_{0}\".format( \"99\" ) + '.csv'\n",
    "                            fou = open( os.path.join(tmp_dir, foun), file_op_index )   # Open input file\n",
    "                            fou.write( line )                                       # Write data\n",
    "                            fou.close()                                             # Close input file\n",
    "                    else:\n",
    "                        for i in range( 0, len(dataidx) ):\n",
    "                            line_index = int(h1[2])\n",
    "                            data_index = int(dataidx[i])\n",
    "                            if (line_index > data_index and line_index <= data_index + 5):\n",
    "                                #\n",
    "                                # There are types of data line ends with comma(,), which Python code recognise\n",
    "                                # as null('') at re.split. In this case, comma separated array shows one more\n",
    "                                # elements than it should. If last element of comma separated array contains\n",
    "                                # null(''), it should drop as bellow. \n",
    "                                #\n",
    "                                h1len = len(h1)\n",
    "                                if h1[h1len-1] == '':\n",
    "                                    h1 = h1[0:h1len-1]\n",
    "                                    nh = nh - 1\n",
    "\n",
    "                                datanum[dataidx.index(dataidx[i])] = datanum[dataidx.index(dataidx[i])] + 1\n",
    "\n",
    "                                foun = \".\".join(fn_sep[:len(fn_sep)-1]) + \"_{0}\".format( dataidx[i] ) + '.csv'\n",
    "                                fou = open( os.path.join(tmp_dir, foun), 'a' )         # Open input file\n",
    "                                fou.write( line )                                   # Write data\n",
    "                                fou.close()                                         # Close input file\n",
    "\n",
    "        d.close()\n",
    "        self.dataidx = dataidx\n",
    "        return dataidx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True\n"
     ]
    }
   ],
   "source": [
    "# Read in brightness temperature data, contained in Level1 data.\n",
    "f_radmtr_lv1 = glob.glob(dain + '/*lv1.csv')[0]\n",
    "df_radmtr_lv1 = radiometrics()\n",
    "df_radmtr_lv1.read_lv1_data(f_radmtr_lv1)\n",
    "\n",
    "# Radiometer channels\n",
    "radmtr_channels = df_radmtr_lv1.df.loc[:,'Ch22.000':'Ch58.800'].dropna(axis=1).columns.str.replace('Ch','')\n",
    "radmtr_channels = radmtr_channels.values.astype(np.float64) * 1e9"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2018-07-03 09:00:13+00:00\n"
     ]
    }
   ],
   "source": [
    "# Radiometer observations (brightness temperatures) for the specific time \n",
    "BosungObs_radmtr = df_radmtr_lv1.df.loc[(df_radmtr_lv1.df.DateTime - TimeOfInterest).abs().idxmin()]\n",
    "print(BosungObs_radmtr.DateTime)\n",
    "BosungObs_radmtr = BosungObs_radmtr.loc['Ch22.000':'Ch58.800'].dropna().values.flatten()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save as .xml files. \n",
    "tp.arts.xml.save(radmtr_channels, './ClearSky_1D_f_grid.xml')\n",
    "tp.arts.xml.save(BosungObs_radmtr, './BosungObservations.xml')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Bosung radiometer's Gaussian optical antenna characteristics. \n",
    "\n",
    "# Full width at half maximum:\n",
    "FWHM_22GHz = 6.3 ;\n",
    "FWHM_30GHz = 4.9 ;\n",
    "FWHM_51GHz = 2.5 ;\n",
    "FWHM_59GHz = 2.4 ;\n",
    "# Linear interpolation\n",
    "FWHM_22to30GHz = np.interp(radmtr_channels[0:8], np.array([22, 30])*1e9, [FWHM_22GHz, FWHM_30GHz]) ;\n",
    "FWHM_51to59GHz = np.interp(radmtr_channels[8:22], np.array([51, 59])*1e9, [FWHM_51GHz, FWHM_59GHz]) ;\n",
    "FWHM = np.append(FWHM_22to30GHz, FWHM_51to59GHz)\n",
    "\n",
    "# Antenna response\n",
    "xwidth_si = 3; # Default value in ARTS. See \"antenna_responseGaussian\".\n",
    "dx_si = 0.1; # Default values in ARTS. See \"antenna_responseGaussian\".\n",
    "Zenith_angle = np.arange(-xwidth_si, xwidth_si + dx_si, dx_si) * FWHM_22GHz / (2*(2*np.log(2))**0.5)\n",
    "anthenna_response = np.zeros((1, len(radmtr_channels), len(Zenith_angle), 1))\n",
    "for i in range(len(radmtr_channels)):\n",
    "    std_FWHM = FWHM[i]/(2*(2*np.log(2))**0.5)\n",
    "    anthenna_response[0,i,:,0] = 1/(std_FWHM*(2*np.pi)**0.5)*np.exp(-4*np.log(2)*Zenith_angle**2/(FWHM[i]**2))\n",
    "\n",
    "# Define ARTS variable \"mblock_dlos_grid\". \n",
    "mblock_dlos_grid = np.array([np.linspace(Zenith_angle[0],Zenith_angle[len(Zenith_angle)-1],20)]).T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save antenna_response as GriddedField4 .xml file. \n",
    "antenna_response_GF4 = tp.arts.griddedfield.GriddedField4()\n",
    "antenna_response_GF4.name = 'Antenna response'\n",
    "antenna_response_GF4.data = anthenna_response\n",
    "antenna_response_GF4.grids = [['NaN'], radmtr_channels, Zenith_angle, np.array([0])]\n",
    "antenna_response_GF4.gridnames = ['Polarisation', 'Frequency', 'Zenith angle', 'Azimuth angle']\n",
    "tp.arts.xml.save(antenna_response_GF4, './ClearSky_1D_antenna_response.xml')\n",
    "\n",
    "# Save mblock_dlos_grid as .xml file. \n",
    "tp.arts.xml.save(mblock_dlos_grid, './ClearSky_1D_mblock_dlos_grid.xml')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sensor LOS and geolocation \n",
    "tp.arts.xml.save(np.array([[0]]), './ClearSky_1D_sensor_los.xml')\n",
    "tp.arts.xml.save(np.array([[0]]), './ClearSky_1D_sensor_pos.xml')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Read in LDAPS data. \n",
    "\n",
    "# Pressure (pres) data\n",
    "fn_pres = \"ldps_v070_erlo_pres_BSWO_h000.\" + TimeOfInterest.strftime('%Y%m%d%H') + \".txt\"\n",
    "f_pres = os.path.join(dain,fn_pres)\n",
    "df_pres = pd.read_csv(f_pres, skiprows=0, \n",
    "                 names=['Index', '?(GridPosition)', 'Type', 'Pressure', 'Longitude', 'Latitude', 'Value'], \n",
    "                 sep=' mb:lon=|,lat=|,val=|:', \n",
    "                 engine='python')\n",
    "pres_P = df_pres.loc[df_pres.Type=='HGT'].Pressure.values * 100\n",
    "pres_GH = df_pres.loc[df_pres.Type=='HGT'].Value.values\n",
    "pres_T = df_pres.loc[df_pres.Type=='TMP'].Value.values\n",
    "pres_RH = df_pres.loc[df_pres.Type=='RH'].Value.values\n",
    "pres_Lat = df_pres.Latitude[0]\n",
    "\n",
    "# Surface (unis) data\n",
    "fn_unis = \"ldps_v070_erlo_unis_BSWO_h000.\" + TimeOfInterest.strftime('%Y%m%d%H') + \".txt\"\n",
    "f_unis = os.path.join(dain,fn_unis)\n",
    "df_unis = pd.read_csv(f_unis, skiprows=0, \n",
    "                 names=['Index', '?(GridPosition)', 'Type', 'Altitude', 'Longitude', 'Latitude', 'Value'], \n",
    "                 sep=':lon=|,lat=|,val=|:',\n",
    "                 engine='python')\n",
    "unis_P = df_unis.loc[df_unis.Type=='PRMSL'].Value.values # df_unis.loc[df_unis.Type=='PRES'].Value.values\n",
    "unis_T = df_unis.loc[(df_unis.Type=='TMP') & (df_unis.Altitude=='surface')].Value.values\n",
    "unis_RH = df_unis.loc[df_unis.Type=='RH'].Value.values\n",
    "unis_Alt = 0 # df_unis.loc[df_unis.Type=='DIST'].Value.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 99216.8, 100000. ,  97500. ,  95000. ,  92500. ,  90000. ,\n",
       "        87500. ,  85000. ,  80000. ,  75000. ,  70000. ,  65000. ,\n",
       "        60000. ,  55000. ,  50000. ,  45000. ,  40000. ,  35000. ,\n",
       "        30000. ,  25000. ,  20000. ,  15000. ,  10000. ,   7000. ,\n",
       "         5000. ])"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Pressure\n",
    "\n",
    "# Combine the unis and pres variables. \n",
    "LDAPS_p = np.append(unis_P, pres_P)\n",
    "LDAPS_p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([    0.        ,   -69.76623271,   154.67407637,   383.91628152,\n",
       "         617.96162349,   856.24846643,  1099.55879501,  1349.37783693,\n",
       "        1867.36340307,  2413.47197498,  2993.09837264,  3608.33534798,\n",
       "        4264.78654767,  4970.01668976,  5734.95859054,  6566.94765423,\n",
       "        7479.55164005,  8488.58759331,  9619.91067809, 10912.34232256,\n",
       "       12426.37388924, 14278.75810695, 16769.54768633, 18892.22202478,\n",
       "       20961.61062189])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Altitude\n",
    "\n",
    "# Convert geopotential height to geometric height. \n",
    "# Reference (accessed 2018-07-02): \n",
    "# http://glossary.ametsoc.org/wiki/Geopotential_height\n",
    "# http://glossary.ametsoc.org/wiki/Acceleration_of_gravity \n",
    "g0 = 9.80665 # Standard gravity at sea level \n",
    "g_lat = 0.01*(980.6160*(\n",
    "    1 - 0.0026373*np.cos(np.pi/180 * 2*pres_Lat) + 0.0000059*(\n",
    "        np.cos(np.pi/180 * 2*pres_Lat)**2))) # Sea-level gravity at given latitude\n",
    "Cg = 0.01*(3.085462*(10**-4) + 2.27*(10**-7)*np.cos(np.pi/180*2*pres_Lat)) # The coefficient in the gravity equation. \n",
    "\n",
    "# Solve for geometric height, using the quadratic formula.\n",
    "a = Cg/2\n",
    "b = -g_lat\n",
    "c = g0*pres_GH\n",
    "pres_Alt = (-b - (b**2 - 4*a*c)**0.5)/(2*a)\n",
    "# Here, the geopotential height is given based on 국지예보모델, so the calculated z field may be based on a spherical coordinates system.\n",
    "# ARTS requires z field that is defined in terms of the geometrical altitude, \n",
    "# which is the distance between the ellipsoid's surface and the point along the line passing through the Earth's center and the point. \n",
    "# For now, assume that the difference between the two systems in this regard is negligible. \n",
    "\n",
    "# Combine the unis and pres variables. \n",
    "LDAPS_z = np.append(unis_Alt, pres_Alt)\n",
    "LDAPS_z"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([298.551, 300.676, 299.203, 297.599, 295.598, 293.497, 291.862,\n",
       "       290.625, 287.838, 285.863, 283.212, 280.   , 276.75 , 274.25 ,\n",
       "       270.473, 266.089, 260.575, 253.387, 245.711, 236.376, 225.224,\n",
       "       213.75 , 203.723, 205.599, 211.875])"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Temperature\n",
    "\n",
    "# Combine the unis and pres variables.\n",
    "LDAPS_t = np.append(unis_T, pres_T)\n",
    "LDAPS_t"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2.52791807e-02, 2.76052410e-02, 2.53786666e-02, 2.38140014e-02,\n",
       "       2.31836276e-02, 2.29237782e-02, 2.25266325e-02, 2.19945752e-02,\n",
       "       2.00840898e-02, 1.82425177e-02, 1.55772143e-02, 1.37603168e-02,\n",
       "       1.26806777e-02, 1.00787162e-02, 8.28979669e-03, 5.66460950e-03,\n",
       "       3.87530683e-03, 2.86846764e-03, 1.74603883e-03, 7.50496609e-04,\n",
       "       8.94729832e-05, 2.85144034e-06, 1.75388594e-06, 3.86814252e-06,\n",
       "       4.73871350e-06])"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Water VMR\n",
    "\n",
    "# Combine the unis and pres variables.\n",
    "LDAPS_RH = np.append(unis_RH, pres_RH)\n",
    "\n",
    "# Convert RH to VMR. \n",
    "LDAPS_watervmr = tp.physics.relative_humidity2vmr(LDAPS_RH * 0.01, LDAPS_p, LDAPS_t)\n",
    "LDAPS_watervmr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get rid of invalid values at the pressure levels higher than the surface pressure. \n",
    "flag_validpressurelevels = (LDAPS_p <= LDAPS_p[0])\n",
    "LDAPS_p = LDAPS_p[flag_validpressurelevels]\n",
    "LDAPS_z = LDAPS_z[flag_validpressurelevels]\n",
    "LDAPS_t = LDAPS_t[flag_validpressurelevels]\n",
    "LDAPS_watervmr = LDAPS_watervmr[flag_validpressurelevels]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Read in radiosondes data. \n",
    "\n",
    "fn_radsnd = \"UPP_LV2_RS92-SGP_47258_\" + TimeOfInterest.strftime('%Y%m%d%H%M') + \".txt\"\n",
    "f_radsnd = os.path.join(dain,fn_radsnd)\n",
    "df_radsnd = pd.read_csv(f_radsnd, sep=\",\")\n",
    "#print(*df_radsnd.time.values, sep='\\n') # Print all values.\n",
    "#print(df_radsnd.columns) % Data types. \n",
    "#print(df_radsnd.loc[1865:1868]) # Pressure duplicates\n",
    "\n",
    "# Solicit useful variables. \n",
    "df_radsnd_useful = df_radsnd[['HGT', 'time', 'P', 'Temp', 'RH', 'MixR', 'Lon', 'Lat', 'Alt']]\n",
    "df_radsnd_useful = df_radsnd_useful.dropna().reset_index().drop('index',axis=1)\n",
    "df_radsnd_useful.loc[1:,] = df_radsnd_useful.loc[1:,].astype(float).values \n",
    "df_radsnd_useful_size = len(df_radsnd_useful.loc[:,'P'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Interpolate the data to fewer vertical grids. Use the nearest neighbor interpolation.\n",
    "df_radsnd_useful_interp_size = 50;\n",
    "df_radsnd_useful_interp = pd.DataFrame( \n",
    "    {'P' : \n",
    "     np.linspace(df_radsnd_useful.loc[1,'P'], df_radsnd_useful.loc[df_radsnd_useful_size-1,'P'], df_radsnd_useful_interp_size)} )\n",
    "df_radsnd_useful_interp = df_radsnd_useful_interp.reindex(df_radsnd_useful.columns, axis=1)\n",
    "\n",
    "# Interpolation\n",
    "for i in range(df_radsnd_useful_interp_size):\n",
    "    nearneigindex = (df_radsnd_useful.loc[1:,'P'] - df_radsnd_useful_interp.loc[i,'P']).astype(float).abs().idxmin()\n",
    "    df_radsnd_useful_interp.loc[i,] = df_radsnd_useful.loc[nearneigindex,].astype('float')\n",
    "\n",
    "# Unit conversions\n",
    "df_radsnd_useful_interp.loc[:,'P'] = df_radsnd_useful_interp.loc[:,'P'] * 100\n",
    "df_radsnd_useful_interp.loc[:,'Temp'] = df_radsnd_useful_interp.loc[:,'Temp'] + 273.15\n",
    "df_radsnd_useful_interp.loc[:,'RH'] = df_radsnd_useful_interp.loc[:,'RH'] * 1e-2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Necessary variables for ARTS simulations \n",
    "radsnd_P = df_radsnd_useful_interp.loc[:,'P'].values\n",
    "radsnd_T = df_radsnd_useful_interp.loc[:,'Temp'].values\n",
    "radsnd_WaterVMR = tp.physics.relative_humidity2vmr(df_radsnd_useful_interp.loc[:,'RH'].values, radsnd_P, radsnd_T)\n",
    "radsnd_HGT = df_radsnd_useful_interp.loc[:,'HGT'].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([    0.,   180.,   370.,   550.,   740.,   940.,  1130.,  1330.,\n",
       "        1530.,  1740.,  1950.,  2170.,  2390.,  2620.,  2850.,  3090.,\n",
       "        3340.,  3590.,  3840.,  4100.,  4380.,  4650.,  4940.,  5240.,\n",
       "        5540.,  5860.,  6190.,  6530.,  6880.,  7250.,  7630.,  8020.,\n",
       "        8440.,  8870.,  9330.,  9810., 10310., 10840., 11410., 12020.,\n",
       "       12670., 13380., 14160., 15020., 16010., 17160., 18590., 20510.,\n",
       "       23440., 29400.])"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "radsnd_HGT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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xo0b573runXrVi1cuFCrV6/W4MGDddVVV2nWrFlas2aNfv75Z23fvl3t2rXTzTffrKKiIt177736+OOP1bRpU82YMUOPPfaYpk2b5tP3YNy4cXr44YfVpUsX5eTkaNCgQcrMzJQkrV+/XvPmzVNGRoZ69Oihjz/+WJMmTdIVV1yhzz//3HtH+LTTTtOSJUs0bdo0jR8/XrNnzz7udVesWKEFCxYoIiJCBw4c0JdffqmIiAitXr1aN954oxYvXqxnnnlGL730kvd78vrrrx/zPeTn56tDhw6aOHGiCgsL1bt3b82ZM0dNmjRRSkqK/vKXv+jVV1/16fsRTMjNAJxC/q66/N2tWzctWLBAt956q7KzszVixAi98sorkkqK4UcffVSSNGzYMN12222SSm48v/HGG7r33ns1ePBgDRo0SFdddZUkqU+fPnr55ZfVpk0bLV68WHfffbe+/vprSdKvv/6qr776SqGhoWViGDBggG699Vbt3r1bUVFRSk1N1b333us9boxR3759NXfuXOXn52vw4MFav359uf92n3/+uYYOHerTv7O/KvU5w547xMsknaOSO8VZkvKstUfm0XMltfB83kLSJkmy1hYbY/IlRXnGF5W6bOnXbPrDeOdKeBsAoIMHDyoxMVE5OTnq2LGj+vXrJ6nk7u2f//xnLViwQCEhIdq8ebO2b9+u7777TldeeaX3DujgwYOPuuaaNWsUGxurc889V5J04403asqUKRo7dqwiIiJ066236vLLL/cWkD/88IM++ugjSdL111+vhx9+2HutoUOHKiQkRO3atfPOai5YsECjRo1SaGiozjzzTF1yySXer5uZmel9Dy6XS82bN/f5e/HVV1+VuVO+Z88eHTx4UJJ02WWXKSwsTHFxcZLk/RpxcXHKycnxvmbUqFGSpNGjR2vChAknvO6QIUMUEREhSSosLNTYsWOVkZGhsLAwZWVl+Rz7EbVq1dKVV14pSVq1apVWrlzpXfrmcrkUHR3t9zWDBbkZQE1SE/N39+7d9cwzz2j9+vWKiYlRRESErLUqKCjQsmXLlJxccu8yMzNTjz/+uPLy8lRQUKBLL730qGsVFBQoLS1NI0aM8I4dmQWXpBEjRhxVCEsleXbw4MGaOXOmhg8frvT0dPXv37/MOSNHjtTkyZOVn5+vSZMm6amnnipzfPTo0dq/f79cLpeW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bmaGgYBF/la3bt20YMEC3XrrrcrOztaIESO8y7DT0tL06KOPSpKGDRum2267TVLJjec33nhD9957rwYPHqxBgwbpqquukiT16dNHL7/8stq0aaPFixfr7rvv1tdffy1J+vXXX/XVV18pNLTspNCAAQN06623avfu3YqKilJqaqruvfde73FjjPr27au5c+cqPz9fgwcP1vr168v9nn3++ecaOnSoX9/nQEMxjGpn028HNCdji27qFqPIurWcDgc1zJG9QFLJ0qKrr75akjRy5EjdcsstZZJp7969FRoaqvj4eE2cOFGhoaH6/PPP9eOPP2revHm6//77tWzZMj3xxBM+fe2vvvpKa9as8f59z549OnjwoCTpsssuU1hYmOLi4iRJ/fr1k1SyHConJ8f7mlGjRkmSRo8erQkTJpzwukOGDFFERMm+/MLCQo0dO1YZGRkKCwtTVlaWT3GXVqtWLV155ZWSpFWrVmnlypXq27evpJJl09HR0X5fEzXHkScIhBjjcCQAgk1V5O/u3bvrmWee0fr16xUTE6OIiAhZa1VQUKBly5YpOTlZkpSZmanHH39ceXl5Kigo0KWXXnpUvAUFBUpLS9OIESO8Y4WFhd7PR4wYcVQhLJXk2cGDB2vmzJkaPny40tPT1b9//zLnjBw5UpMnT1Z+fr4mTZqkp556qszx0aNHa//+/XK5XFq+fPmJvrUBjWIY1c4bC9crxEi39PB9eSmCSAXdAa4sK1asUKdOnSSVLLHavn27UlJSJElbtmzR2rVr1aZNG0kle46aNGlS5vXGGCUnJys5Odm7/NjXYthaqyVLlqhWraNvAtWuXVuSFBISUuZ4SEhImf1Eppwi4njXrVevnvfzSZMmqWXLlnr33XdVVFRUbtMSSQoLCyvTIOTQoUPez+vUqeONwVqr+Ph4fffdd8d8z0BpR4rhMJZJA4GH/K02bdpoz549+uSTT9S1a1dJUseOHfXmm28qNjbWmzfHjBmj2bNnKyEhQW+99Za++eabo+J1u92KjIw8ZgOr0vn5j0aNGqWJEyfKWqshQ4YoPDy8zPHk5GRlZmaqTp06Ovfcc496fUpKihISEjRhwgTdc889+uijj475tQId2QLVyu6CQqX+uFFDE1uoeaM6ToeDGubDDz/UF198oVGjRmnNmjXav3+/Nm/erJycHOXk5OjRRx89biOpLVu2lLnDmp6errPP9n2pf9++fTVlypQyr/fXjBkzJJX8ItC9e3e/rpufn6/mzZvLGKO3337be5e9QYMG2rdvn/e8mJgYrVixQtZa5eTkaNmyZeVer127dtq8ebOWLFkiSTp8+LBWrlzp93tCzeGdGea3GwB+qMr83bVrV7344oveYrhr16564YUXvPuFJWnfvn1q3ry5ioqKvAW5VDafNmzYULGxsfrggw8kldxAzsjI8On99u7dW2vXrtWUKVO8K8L+6Omnnz5qRri08PBwTZw4UYsWLfI2BQtGpAtUK2//sEGHity6o1crp0NBNXPgwAFFR0d7P/79739Lkp5//nnvoxneffddff3112ratKmmT5/uXe57xPDhw8vtSnlEUVGRHnzwQZ133nlKTEzUjBkz9OKLL/oc45QpU/T9998rPj5e7dq102uvvXZS7zM5OVlTp07VpEmT/Lru2LFj9frrr6tLly7asGGDdzY6KSlJLpdLCQkJmjx5snr16qUWLVooLi5OEyZMOGZjjtq1a2vmzJkaP368EhISlJSU5G3KBZTnSAMtZoYBHBFo+bt79+7atGmTdxa6a9euys7OLlMM//Of/1Tnzp3Vr18/nXfeed7xkSNH6tlnn1VSUpKysrKUkpKiN954QwkJCbrggguOepzTsYSEhGj48OHavXu3evbsWe45AwcOVO/evY97nTp16uiBBx7Qc88959PXDUSm9Pr4mqBTp0526dKlToeBSrC/sFjdnvlanWMb69UbOjkdDirQqlWrdP755zsdRrUXHR2tzMxMRUZGOh1KwCrvZ9EYs8xay386p6CicvO3v+7UjdOW6MO7uqrj2Y0rIDIAp4L8japwKrmZW6eoNqYv2aj8g0W68+LWTocCAHCA282jlQAAvqOBFqqFw8VuvbFwvTrHNlaHs05zOhwgKOXm5jodAnBKvI9Wops0AMAH3DpFtfBx+mZtzT+ku5gVrrZq2pYOBB5+BgMfDbSAwMP/nahMp/rzRbpA0HO7rV5ZkK3zmzdUr3ObOh0OKkFERIR2795NQoVjrLXavXu395nKCEw00AICC/kblakicjPLpBH0vlq1Xet2FOjFkYnlPiMVwS86Olq5ubnauXOn06GgBouIiFB0dLTTYeA4vMukqYWBgED+RmU71dxMMYygZq3V1G+z1LJxHV0e19zpcFBJwsPDFRsb63QYAAIcDbSAwEL+RqAjWyCoLVn/m1ZszNPtPVopjKkAAKjRaKAFAPAH1QOC2tRvsxRVr5ZGdGrpdCgAAIe5aaAFAPAD6QJBa9XWvfpmzU7d1D1GEeGhTocDAHDYkZlhGmgBAHxBtkDQevnbLNWrFarru8Q4HQoAIAC4LDPDAADfkS4QlDb9dkD//Wmrru18lhrVDXc6HABAAHAzMwwA8APZAkHpte+yFWKkWy5q5XQoAIAAQQMtAIA/KIYRdHYVFGrGj5t0ZVILNWt08g/ZBgBUL95HK4VSDAMAToxiGEHn7bQcHXa5dXvP1k6HAgAIIMwMAwD8QTGMoFJQWKy303J0abtmOuf0+k6HAwAIIG4aaAEA/EC6QFBJXbJRew8V686LmRUGAJTlooEWAMAPZAsEjcPFbr3+3Xp1bRWlxJaRTocDAAgwR5ZJh7BKGgDgA4phBI3Z6Zu1be8hZoUBAOVyu61CQ4wMe4YBAD6gGEZQcLutXv42S+2aN1TPNk2cDgcAEICK3ZbmWQAAn1EMIyh88ct2Ze/cr7subs0dfwBAudzW0jwLAOAzUgYCnrVWU7/N0lmN62pg+2ZOhwMACFDFLkvzLACAz8gYCHiLsn9TxqY83d6zlcJC+ZEFAJTPbS3NswAAPqOyQMB7+dssNalfS1d1jHY6FABAAHO5LTdNAQA+I2MgoK3ckq9vf92pm7rHKiI81OlwAAABrNhtFUJfCQCAjyiGEdBe/jZb9WuH6bouZzsdCgAgwLndVmGskwYA+IhiGAFr4+4D+r+ftmh057PUqE640+EAAAJcsec5wwAA+IJiGAHr1e+yFBYSopsvinU6FABAEODRSgAAf5AyEJB27ivUB0tzNaxDC53RMMLpcAAAQcDl5tFKAADfkTEQkN5KW6/DLrdu79nK6VAAAEHC5ebRSgAA31EMI+DsO1Skd37YoAEXNFOrpvWdDgcAECSYGQYA+IOMgYAzfclG7T1UrDt7tXY6FABAECl2W4UwNQwA8BHFMAJKYbFLr3+3Xt3PiVJCy0inwwEABBG3tQrlNxsAgI9IGQgos1ds1o59hcwKAwD8VvJoJX61AQD4hoyBgOFyW73ybbbat2ioi85p4nQ4AIAg43ZbhbJKGgDgI4phBIwvf9mm7F37dWev1jKG32YAAP6hgRYAwB9kDAQEa62mfpOls6PqamD75k6HAwAIQi63FbUwAMBXpAwEhB+ydisjN1+392ylUDqBAgBOgssyMwwA8B0ZAwFh6rdZalK/toZ3iHY6FABAkOLRSgAAf1AMw3GZm/P13dpduuWiWEWEhzodDgAgSNFACwDgD4phOO7lb7PUoHaYRnc5y+lQAABBzMWjlQAAfiBjwFEbdu/Xpz9v1eguZ6thRLjT4QAAglhJMex0FACAYEHKgKNeXZCtsJAQ3dw9xulQAABBjgZaAAB/kDHgmB37DumDZbka3jFapzeMcDocAECQc9FACwDgB4phOObN73NU7HLrjp6tnA4FAFANuNzlqmUfAAAgAElEQVRWYRTDAAAfUQzDEXsPFendHzZoYPvmimlSz+lwAADVgMttFWIohgEAvqEYhiPeW7xR+wqLdWev1k6HAgCoJmigBQDwBykDVe5QkUtvLFyvi85porjoRk6HAwCoJlyWRysBAHxHxkCVm7Vis3buK9RdFzMrDACoOMwMAwD8QcpAlXK5rV5dkK24Fo3UrXWU0+EAAKqRkgZa/GoDAPANGQNVau7KbVq/a7/uuri1DE1OAAAViAZaAAB/UAyjylhrNfWbLMU2qadLL2jmdDgAgGqGZdIAAH+QMlBl0rJ26+fN+bq9ZyuF8hxIAEAFo4EWAMAfZAxUmanfZOn0BrU1rEMLp0MBAFRDzAwDAPxBykCV+Dk3XwvX7dLNF8Wqdlio0+EAAKoZa62nGOZXGwCAb8gYqBIvf5ulBhFhGt35LKdDAQBUQ25b8mcoDbQAAD6iGEalW79rvz7N3Krru5ytBhHhTocDAKiGXJ5qOCyUYhgA4BuKYVS6VxdkKzw0RDd1j3U6FABANXWkGObRSgAAX1EMo1Lt2HtIHy7L1YiO0WraoLbT4QAAqimXLSmGaaAFAPAVKQOVatr3OSp2u3V7z1ZOhwIAqMaOzAzTQAsA4CsyBirN3kNFSlm0QZfFNdfZUfWcDgcAUI15i2FWSQMAfEQxjEqTsmij9hUW685erZ0OBQBQzXmLYdZJAwB8VGkZwxjT0hgz3xizyhiz0hjzJ8/4E8aYzcaYdM/HZaVe86gxZp0xZo0x5tJS4wM8Y+uMMRNKjccaYxYbY9YaY2YYY2pV1vuBfw4VufTGwvXq0aaJ2rdo5HQ4AABV79z8+8wwU8MAAN9U5u3TYkkPWGvPl9RF0j3GmHaeY89baxM9H59KkufYSEkXSBog6T/GmFBjTKikKZIGSmonaVSp6/w/z7XaSNoj6ZZKfD/ww4fLc7WroFB3XcysMAAEkGqbm2mgBQDwV6WlDGvtVmvtcs/n+yStktTiOC8ZIinVWltorV0vaZ2kZM/HOmtttrX2sKRUSUOMMUbSJZJmel7/tqShlfNu4A+X2+rVBdlKiG6krq2inA4HAOBRnXOzy0UDLQCAf6okYxhjYiQlSVrsGRprjPnJGDPNGHOaZ6yFpE2lXpbrGTvWeJSkPGtt8R/G4bDPMrdqw+4Duuvi1jIsVwOAgFTdcjMzwwAAf1V6yjDG1Jf0oaT7rLV7JU2V1FpSoqStkiYdObWcl9uTGC8vhtuNMUuNMUt37tzp5zuAP6y1evnbLLVqUk/92jVzOhwAQDmqY27m0UoAAH9VasYwxoSrJNmmWGs/kiRr7XZrrcta65b0mkqWWkkld49blnp5tKQtxxnfJSnSGBP2h/GjWGtftdZ2stZ2atq0acW8OZRr4bpdyty8V3f0aqXQEGaFASDQVNfcTAMtAIC/KrObtJH0hqRV1tp/lxpvXuq0KyVlej6fI2mkMaa2MSZWUhtJSyT9KKmNpztlLZU08phjrbWS5ku6yvP6GyV9XFnvB76Z+k2WzmhYW0OTWLEOAIGmOufm32eGKYYBAL4JO/EpJ627pOsl/WyMSfeM/VklHScTVbJsKkfSHZJkrV1pjHlf0i8q6XZ5j7XWJUnGmLGS5koKlTTNWrvSc71HJKUaYyZKWqGSBA+HZGzKU1rWbv35svNUOyzU6XAAAEertrmZYhgA4K9KK4attQtV/t6hT4/zmiclPVnO+Kflvc5am63fl3LBYS9/m6WGEWEalXyW06EAAMpRnXMzDbQAAP4iZaBCZO8s0Ocrt+n6rmerQUS40+EAAGoYGmgBAPxFxkCFeHVBtmqFhmhMt1inQwEA1EA00AIA+ItiGKds+95D+mj5Zo3oFK2mDWo7HQ4AoAZizzAAwF8Uwzhl0xauV7Hbrdt7tHY6FABADUUxDADwF8UwTkn+wSKlLN6oQfFn6qyouk6HAwCooWigBQDwFykDp+TdRRtUUFisO3q1cjoUAEAN5nK7JdFACwDgOzIGTtqhIpfe/H69ep3bVBec2cjpcAAANZirpBamgRYAwGcUwzhpM5flalfBYd3Zi73CAABnsWcYAOAvimGclGKXW68uyFZiy0h1adXY6XAAADUcxTAAwF8Uwzgpn2Vu08bfDuiui1vLsCQNAOCw3xtokZMAAL6hGIbfrLWa+k2WWjetp37nn+F0OAAAlGqgRTEMAPANxTD8tmDtLv2yda/u6NVaIfzSAQAIADTQAgD4i2IYfnv5myw1axihoYktnA4FAABJkvvInuFQimEAgG8ohuGX9E15+iF7t27tEataYfz4AAACQ/GRYpiZYQCAj6hm4JeXv8lSw4gwjUw+y+lQAADwooEWAMBfFMPw2bodBZr7yzbd2C1G9WuHOR0OAABeLhcNtAAA/qEYhs9eXZCl2mEhGtMtxulQAAAow1UyMcwyaQCAzyiG4ZNt+Yc0a8VmXd2ppaLq13Y6HAAAyvA+WokGWgAAH1EMwydvLMyW20q39WjldCgAAByFRysBAPxFMYwTyj9QpPcWb9Sg+OZq2biu0+EAAHAUNw20AAB+ohjGCb2VlqP9h126s1drp0MBAKBcxS6KYQCAfyiGcVzpm/L00vy1Gti+mc5v3tDpcAAAKNeRmWFqYQCAryiGcUx79h/WPSnLdXqDCD09LM7pcAAAOCZPM2kZ9gwDAHzEw2JRLrfb6v7307VzX6Fm3tVVkXVrOR0SAAAAAFQYZoZRrv98s07frNmpv1zRTvHRkU6HAwDA8Vl74nMAACiFYhhH+X7dLv37y181JPFMXdf5LKfDAQDghKwkVkgDAPxBMYwytuUf0p9SV6hV0/p66so49l4BAIIGGQsA4A/2DMOryOXWvdOX68Bhl1Jv76B6tfnxAAAEB1ZJAwD8RbUDr2fnrtGPOXv04shEnXN6A6fDAQDAL6xmAgD4g2XSkCR9nrlNry7I1vVdztaQxBZOhwMAgF+smBoGAPiHYhjasHu/HvogQ/HRjfT4oPOdDgcAAL9Zy55hAIB/KIZruENFLt317nKFhBhNubaDaoeFOh0SAAAnhVXSAAB/sGe4hntizkr9snWvpo3ppJaN6zodDgAAJ4VF0gAAfzEzXIN9sHSTUn/cpHt6t9Yl553hdDgAAJy0kmXSTA0DAHxHMVxDrdq6V3/5OFNdW0Xp/r7nOh0OAACnjloYAOAHiuEaaN+hIt2dslwNI8L14qhEhYXyYwAACG50kwYA+Is9wzWMtVaPfPiTNv52QO/d2lmnN4hwOiQAAE4d3aQBAH5iSrCGefP7HH368zY9fGlbdW4V5XQ4AABUGLpJAwD8QTFcgyzbsEdPfbpK/dqdodt7tnI6HAAAKgyLpAEA/qIYriF2FxRq7HvL1TwyQs+NSJDh9jkAoJqhmzQAwB/sGa4BXG6r+2aka/f+w/rorm5qVCfc6ZAAAKhQ1jI3DADwDzPDNcD/fL1W363dpb8PvkDtWzRyOhwAACqctewZBgD4h2K4mlvw6069OG+thnVooZEXtnQ6HAAAKg21MADAHxTD1diWvIP6U+oKnXt6A00c2p59wgCAaotF0gAAf1EMV1NFLrfGvrdch4vd+s91HVS3FtvDAQDVV8kyaW76AgB8R4VUTT3z2Wot35inl65NUuum9Z0OBwCASkcpDADwBzPD1dBnP2/VGwvXa0y3GA2KP9PpcAAAqHSWhdIAAD9RDFcz63ft10Mzf1Jiy0j9+bLznQ4HAICqw9QwAMAPFMPVyMHDLt317jKFhxpNGd1BtcL45wUA1Aw8ZhgA4C/2DFcjf/04U2u279ObYy5Ui8g6TocDAECVYmIYAOAPpg6rifd/3KQPluXq3kva6OK2pzsdDgAAVY5u0gAAf1AMVwMrt+TrLx9n6qJzmuhPfdo4HQ4AAFXOsk4aAOAniuEgt/dQke5OWa7T6tbSiyMTFRrCXXEAQM1jJTExDADwB3uGg5i1Vg99kKHNew4q9fYuiqpf2+mQAABwDLUwAMAfzAwHsTcWrtfclds1YeB56hTT2OlwAABwDKukAQD+ohgOUj/m/KanP1utARc00y0XxTodDgAAjqOBFgDAHxTDQWhXQaHGvrdcLU+ro3+NiCf5AwBqPCumhgEA/qEYDjIut9WfUlco70CR/jO6oxpGhDsdEgAAjrOWPcMAAP/QQCvIvPjVr/p+3W7966p4tTuzodPhAAAQMFgoBQDwBzPDQWT+mh2a/PU6jegYras7tXQ6HAAAAgaLpAEA/qIYDhKb8w7q/hnpOq9ZA/1jSHunwwEAIKCUdJNmahgA4DuK4SBwuNitu1OWq9hlNfW6jqpTK/Tok/Ztr/rAAAAIICyTBgD4g2I4CDz16SplbMrTs1fFK7ZJvaNPyP5GeiFOyppf5bEBABAIBm/+t74sutHpMAAAQYRiOMB9krFFb6Xl6JaLYjUwrvnRJxzKl2bfI0W2lFp2rvoAAQAIACyTBgD4i27SAWzdjgJN+PAndTz7NE0YeF75J33+qLRvi3TzF1KtulUbIAAAAAAEKWaGA9SBw8W6O2WZaoeH6qVrkxQeWs4/1epPpfQU6aL7pZYXVn2QAAAAABCkmBkOQNZaPTYrU2t3FOh/b05W80Z1jj5p/y7pk3HSGXFSrwlVHyQAAAAABDGK4QA0fckmzVqxWff3PVc92jQ9+gRrpf/eX7Jf+PrZUlitqg8SAIAAYnnSMADATyyTDjA/5+briTkr1fPcprr3knOOcdIH0qo5Uu8/S8145jAAAAAA+ItiOIDkHyzS3e8tU1T9WnrhmkSFhJTTFXPvFunTB0s6R3cbV/VBAgAAAEA1wDLpAPLCV79q856DmnlXNzWuV87SZ2ulj8dKriJp6FQpJLTqgwQAAACAaoBiOEDk7jmglEUbNaJjS3U467TyT1o6TcqaJ132nBTVumoDBAAAAIBqhGXSAeKFr9ZKRvpT3zbln/BbtvTFX6RWvaULb63a4AAAAACgmqEYDgBrt+/TR8tzdUOXs3VmZDmPUXK7pFl3SSFh0pApkilnLzEAADWZpZs0AMA/LJMOAJO++FV1a4Xp7t7H6B79w0vSpkXSla9IjVpUbXAAAAQJK24WAwB8V2kzw8aYlsaY+caYVcaYlcaYP3nGGxtjvjTGrPX8eZpn3BhjJhtj1hljfjLGdCh1rRs95681xtxYaryjMeZnz2smGxN8U6YZm/L0+cpturVHbPlNs7b/In09UTpvkBR/TdUHCACoNsjNAAD8rjKXSRdLesBae76kLpLuMca0kzRB0jxrbRtJ8zx/l6SBktp4Pm6XNFUqSdCS/iaps6RkSX87kqQ959xe6nUDKvH9VIp/zV2txvVq6dYerY4+WHxYmnWHVLuhdMWLLI8GAJwqcjMAAB6VVgxba7daa5d7Pt8naZWkFpKGSHrbc9rbkoZ6Ph8i6X9tiUWSIo0xzSVdKulLa+1v1to9kr6UNMBzrKG19gdrrZX0v6WuFRS+X7dL36/brXt6n6P6tctZsb7gWWnbTyWFcL0mVR8gAKBaITcDAPC7KmmgZYyJkZQkabGkM6y1W6WSpCzpdM9pLSRtKvWyXM/Y8cZzyxkPCtZa/WvuGp3ZKEKjO5919Ambl0nfTZISRknnD6r6AAEA1Rq5GQBQ01V6MWyMqS/pQ0n3WWv3Hu/UcsbsSYyXF8PtxpilxpilO3fuPFHIVWLuyu3K2JSn+/qeq4jw0LIHiw5Ks+6UGjSTBjzjTIAAgGqL3AwAQCUXw8aYcJUk2xRr7Uee4e2eZVTy/LnDM54rqWWpl0dL2nKC8ehyxo9irX3VWtvJWtupadOmp/amKoDLbfXcF2vUumk9DetQzg3zef+Qdv1a8hilOpFVHyAAoNoiNwMAUKIyu0kbSW9IWmWt/XepQ3MkHek6eaOkj0uN3+DpXNlFUr5nqdZcSf2NMad5mnP0lzTXc2yfMaaL52vdUOpaAe2j5blat6NAD/Zvq7DQP/wTrP9OWvQf6cLbpNa9nQkQAFAtkZsBAPhdZT5nuLuk6yX9bIxJ94z9WdIzkt43xtwiaaOkEZ5jn0q6TNI6SQck3SRJ1trfjDH/lPSj57x/WGt/83x+l6S3JNWR9JnnI6AVFrv0wldrFR/dSAPaNyt78NBeafbdUuNWUr+/OxMgAKA6IzcDAOBRacWwtXahyt87JEl9yjnfSrrnGNeaJmlaOeNLJbU/hTCr3HuLN2pz3kE9MzxORz16ce6fpb250s1zpVr1nAkQAFBtkZsBAPhdlXSTRomCwmK99PU6dW0VpYvO+cOjkn6dK614R+r+J6llsjMBAgAAAEANQTFchaYtXK/d+w/r4QFty84KH/hNmnOvdEZ76eJHnQsQAIAgdazpbgAAjqUy9wyjlD37D+u1Bdnq3+4MJZ11WtmD/ze+pCC+7kMprLYzAQIAAABADcLMcBWZ+m2WCg4X68FL25Y98PNMaeUs6eIJUrM4Z4IDAAAAgBqGYrgKbM0/qLfScnRlUgude0aD3w/s3Sr93wNS9IVS9/ucCxAAAAAAahiK4Sowed5aWWt1f99zfx+0tmSfcHGhNPRlKZQV6wAAAABQVajAKln2zgK9vzRX13c5Wy0b1/39wPK3pXVfSgP/JTU5x7kAAQAAAKAGYma4kv37y19VOyxE9/QuVfDuyZHmPibF9pQuvM2x2AAAAACgpqIYrkSZm/P135+26ubusWrawNMl2u2WZt8tmRBpyH+kEP4JAAAAAKCqsUy6Ej07d40i64br9l6tfh9c9B9pw/clhXBkS+eCAwAAAIAajGnJSrI4e7e+/XWn7urVWg0jwksGd6yW5v1DanuZlHitswECAAAAQA1GMVwJrLX619w1OqNhbd3YLaZk0FUkzbpDql1fuuJFyRhHYwQAAACAmoxiuBLMW7VDyzbs0bg+bRQRHloy+N0kaWu6NOh5qf7pzgYIAAAAADUcxXAFc7utnvtijWKi6urqTp49wYfypQXPSu2vktoNcTZAAACqJet0AACAIEMxXMHmZGzR6m37NL5/W4WHer69WzMkd7GUMMrZ4AAAqMas2IIEAPAdxXAFOlzs1r+//FXtmjfUoLjmvx/Ykl7y55mJzgQGAAAAACiDYrgCzfhxozb+dkAPXdpWISGl7k5vTZcaRkv1mjgXHAAAAADAi2K4ghw4XKzJX69TckxjXdy2admDW9KZFQYAoBIZ9gwDAPxEMVxBZq3YrJ37CvXQgLYypR+bdChf+i1Lak4xDABAZWLPMADAHxTDFWRpzh6d3qC2Op19WtkDW38q+ZOZYQAAAAAIGBTDFWTFxj1KbBlZdlZYKtkvLDEzDAAAAAABhGK4AuzZf1g5uw8o8azIow9uSZcatpDqNz36GAAAAADAERTDFSA9N0+SlNiynGJ4azqzwgAAVDoaaAEA/EMxXAHSN+bJGCk++g/F8KG90u517BcGAKAK0EALAOAPiuEKkL4pT+ee3kD1a4eVPbDN0zyLmWEAAAAACCgUw6fIWquM3DwlHWu/sMTMMAAAAAAEGIrhU5Sz+4DyDhQde79wgzOl+qdXfWAAANQgJQuk2TcMAPAdxfApSt+0R5KO3UmaWWEAAKoEe4YBAP6gGD5FKzbmqV6tULU5vUHZA4X7SppnsV8YAAAAAAIOxfApSt+Up7joRgoN+cPd6G0/S7LMDAMAAABAAKIYPgWHilxatXWvEluedvTBI82zmBkGAAAAgIBDMXwKVm7ZqyKXPU7zrOZSgzOqPjAAAGoYY2meBQDwD8XwKUjflCdJx36sErPCAABUIRpoAQB8RzF8CtI35enMRhE6o2FE2QOFBdKuX9kvDAAAAAABimL4FKRv2lP+I5WONM9iZhgAAAAAAhLF8EnaVVCoTb8dPPZ+YYmZYQAAqogx7BkGAPiHYvgkpW8s2S98zE7S9ZtJDZpVcVQAANRclMMAAH9QDJ+k9E15Cg0ximvR6OiDW9OZFQYAAACAAEYxfJLSN+Wp7RkNVKdWaNkDh/eXNM9ivzAAAAAABCyK4ZPgdltlbMo7dvMs62ZmGAAAAAACGMXwScjeVaB9hcVKKq951hZP8yxmhgEAqDqWHcMAAP9QDJ+EFZ7mWUnlzQxvTZfqnyE1bF7FUQEAULNZGadDAAAEEYrhk7BuR4GMkZo2iDj64JZ0ZoUBAAAAIMBRDJ+EgXHNZST96/PVZQ8c3i/tWsN+YQAAqphhVhgA4CeK4ZOQ2DJSN3ePVcrijfoha/fvB7ZlljTPYmYYAIAqxp5hAIB/KIZP0gP92+qsxnX16Ec/6eBhV8ngVk/zLGaGAQCocuwZBgD4g2L4JNWpFapnhscpZ/cBPf/VryWDW9KleqdLDWieBQAAAACBjGL4FHRr3USjks/S699lK2NTXsnM8JmJkuHONAAAAAAEMorhU/ToZefp9AYR+uvMJbI7V7NfGAAAAACCAMXwKWoYEa6JQ9vr1+37VBQSIa35VDqY53RYAAAAAIDjoBiuAH3bnaF+Ca10e+GfZHeukVKvlYoOOR0WAAA1BjuUAAD+ohiuIH+7op1+qt1Rk+rdL234XvrwFsntcjosAAAAAEA5KIYrSFT92vrbFe300s5E/dDmIWn1f6X/Gy9ZnnsIAAAAAIGGYrgCDU44U33PP103re6o/I73Ssvekr552umwAAAAAAB/QDFcgYwxmjg0TuEhIbpzy+WyiddJ3/4/aclrTocGAAAAACiFYriCNWsUoT9ffr5+WP+bZjR7QDp3oPTpQ9LK2U6HBgAAAADwOGExbIzpaoyZYoz5yRiz0xiz0RjzqTHmHmNMo6oIMtiMvLCluraK0pOfrdW2/v+RWnaWPrpNyv7W6dAAANUAuRkAgFN33GLYGPOZpFslzZU0QFJzSe0kPS4pQtLHxpjBlR1ksDHG6JnhcSpyu/X4/2XJjpouNW4tpY6WtmY4HR4AIIiRmwEAqBgnmhm+3lp7i7V2jrV2i7W22FpbYK1dbq2dZK29WFJaFcQZdM6OqqcH+7fVV6t26JO1h6TrPpQiGknvXiX9lu10eACA4EVuBgCgAhy3GLbW7ir9d2NMQ2NM4yMf5Z2D393UPVYJLSP1xJyV+i2sqXT9LMldJL0zTCrY4XR4AIAgRG4+NiMeZwgA8J1PDbSMMXcYY7ZL+knSMs/H0soMrDoIDTH61/B47TtUpL9/slJqeq40eqZUsF16d7h0aK/TIQIAghS5uSwr43QIAIAg42s36QclXWCtjbHWxno+WlVmYNVF22YNdE/vc/Rx+hZ9+ct2KbqTdPX/Sjt+kWaMlooLnQ4RABCcyM0AAJwCX4vhLEkHKjOQ6uzui89Ru+YN9dDMDG3NPyi16ScNmSKtXyB9dLvkdjkdIgAg+JCbAQA4BWE+nveopDRjzGJJ3qlMa+24SomqmqkVFqKXrk3SoP9ZqD9NT9d7t3VWWMJIaf9O6YvHpc+aSpc9KxmWeAEAfEZuBgDgFPhaDL8i6WtJP0tyV1441VerpvX11JVxum9GuibPW6vx/dtK3e4t2T+c9j9S/TOkXg85HSYAIHiQmwEAOAW+FsPF1trxlRpJDTA0qYW+X7dL/zN/nTq3ilL3c5pIff8hFeyU5k+U6jeVOo5xOkwAQHAgNwMAcAp83TM83xhzuzGm+R8f3wD//H3IBWrdtL7um5GunfsKpZAQachL0jn9pP/eL636xOkQAQDBgdwMAMAp8LUYvlaevUni8Q2npG6tME25toP2HizS+PfT5XZbKTRcuvpt6cwO0sxbpJzvnQ4TABD4yM1/wFOGAQD+8KkYLvXIhlge33Dq2jZroCcGX6Dv1u7SywuySgZr1ZNGfyCddrY0fZS0LdPZIAEAAY3cXBY9KAEA/jpuMWyMuegExxsaY9pXbEg1w8gLW2pQfHNN+uJXLc35rWSwbmPpuo9KCuN3h0t7chyNEQAQeMjNAABUjBPNDA83xqQZY/5qjLncGJNsjOlpjLnZGPOOpP9KqlMFcVY7xhg9PSxOLSLraNz0Fco7cLjkQGRL6fqPpOKD0jvDpP27nA0UABBoyM3HYFgoDQDww3GLYWvt/ZIul7RV0ghJ/5Q0XlIbSa9Ya3taa3+s9CirqQYR4Xrp2iTtLCjUgx/8JGs9Sfz086Vr35f2bpZSrpIKC5wNFAAQMMjN5bNinTQAwD8nfLSStXaPpNc8H6hg8dGRmjDwfP3zv7/orbQc3dQ9tuTAWV2kEW9JqaOlGdeVFMdhtRyNFQAQGMjNAACcOl+7SaMS3dw9Rn3PP0NPf7paP+fm/36g7UBp8GQpe740+y7J7XYuSAAAAACoRiiGA4AxRs9eFa+o+rU0dvpy7TtU9PvBpOukPn+TMmdKc//8/9m7z+ioqrcN49eeVAgJEAi99yIgEHoNHQTpXRCUIk0UscBr/auogKBIkSYo0qSD9F5DCR3pVXrvENLO+2EiRqWEkpyU+7dWVpI9Zyb3rMVi58ne59lg6X4oERGRf9MmaREReVIxVgwbY34yxlwwxuyJMvapMea0MWZH5EedKI/1McYcNsYcMMbUjDJeK3LssDHmgyjj2Y0xm4wxh4wxU40x8XoPcUovd4a0LMqpq3fpO2vP3/cPA5R/G0p1gU0jYP139oUUEZF4TXOziIjI36JVDBtjkhpjPjLGjI78Prcxpu5jnjYeqPWA8cGWZb0Y+bEg8vUKAC2AgpHPGW6McTHGuADDgNpAAaBl5LUA30S+Vm7gKvB6dN5LXFYimy+9qudh3s4zTN1y8u8HjIGa/eCFJrDsU9j+q20ZRUQkbtDcLCIi8myiuzI8DrgHlIn8/hTwxaOeYFnWGuBKNF+/PjDFsqx7lmUdAw4DJSM/DluWddSyrBBgClDfGGOAKsD0yOf/DDSI5s+K07pUykn5XKn5dN4fHDh38+8HHA5oMAJyBMDcN+HAQvtCiohIXKC5WURE5BlEtxjOaVlWfyAUwLKsuzz97f2efwsAACAASURBVDndjTG7IrdqpYwcywhEWQrlVOTYw8ZTAdcsywr713i853AYBjUvQjIPN1qP2cS8nWf+3jLt6g7NJ0D6wjCtHfy5ydasIiJiK83NIiIizyC6xXCIMSYJOE+zN8bkxPnX6Cc1AsgJvIjzfMRvI8cfNHlbTzH+QMaYTsaYIGNM0MWLF58ssQ3SeHvya4eSpE/uSY/J23lt/BZOXb3jfNDDG1pPB5+MMKkZXNhnb1gREbGL5mYREZFnEN1i+BNgEZDZGDMRWA6896Q/zLKs85ZlhVuWFYHzbMSSkQ+dAjJHuTQTcOYR45eAFMYY13+NP+znjrIsy9+yLH8/P78njW2LfOl8mN2tHB/XLcCmY1eoPmgNY9YeJSw8ArxSQ5tZ4OoJExrBtZOPf0EREUloNDf/izpKi4jIk4hWMWxZ1lKgEdAOmAz4W5a16kl/mDEmfZRvGwJ/dbOcC7QwxngYY7IDuYHNwBYgd2R3SnecjTzmWs59wyuBJpHPfxWY86R54joXh+G18tlZ2qsSZXOm4ov5+2gwfL3zLOKUWeGVGRByG35tBHeiewuYiIgkBJqb/02lsIiIPBnXRz1ojCn2r6GzkZ+zGGOyWJa17RHPnQxUBlIbY07h/At2ZWPMizi3TR0HOgNYlvWHMeY3YC8QBnSzLCs88nW6A4sBF+Any7L+iPwR7wNTjDFfANuBsdF6x/FQxhRJGPOqPwv3nOPTuX9Qf9g62pfLTq/q+fBqORkmNISJTeHVueDuZXdcERGJQZqbRUREng/zj/Ns//2gMSsjv/QE/IGdOP/0WhjYZFlW+RhP+Jz5+/tbQUFBdsd4ajeCQ+m/aD+/bvyTjCmS8HmDglSxtsBvbSBnVWg5GVzc7I4pIpJoGGO2WpblH4s/T3PzAwQNeYUsV9aT5tNjzymViIjEV9Gdmx+5TdqyrADLsgKAE0CxyHt7igNFcR6xILHMx9ONLxoUYkaXMnh5uPDa+CC6bcvAjar94fBSmNMdIiLsjikiIjFEc7OIiMjzEd0GWvksy9r91zeWZe3B2XVSbFI8qy+/96hA7xp5WLrvPOWWZWFn7u6wawos+8TueCIiEvM0N4uIiDyD6BbD+4wxY4wxlY0xlYwxowGd6WMzd1cH3avkZvFbFXkhQ3Lq7y7DwiT1YMMQ2PCD3fFERCRmaW4WERF5Bo9soBVFe6AL0DPy+zU4zyWUOCB7ai8mdSzFjG2n+fB3F6yIC9RZ8iEhnqlwL9bK7ngiIhIzNDeLiIg8g2gVw5ZlBQODIz8kDjLG0KR4JgLy+vH1vFQk3/sWpeZ2Z+8tdwpUbPL4FxARkXhFc7OIiMiziVYxbIw5hvPIhX+wLCvHc08kzyRVMg8GtCxJ4N6JHJ3eiGzLuzLkWChtmjQmpZe73fFEROQ50dwsIiLybKK7TTpqW2pPoCng+/zjyPNSpkB2gnss4O6P1WhztDc9v71Cw3r1afBiRowxdscTEZFnp7lZRETkGUSrgZZlWZejfJy2LOs7oEoMZ5Nn5JkyPSk7zyOZdwrGRXzI4emf8erYQE5cvm13NBEReUaam/8p3LjiRpjdMUREJB6J7jbpYlG+deD8a7R3jCSS58s3B27d1hPx+9u8+8dvbDm5m7aDu9Gocmk6VMiOl0d0NweIiEhcorn5n0IdnngQYncMERGJR6JbCX0b5esw4BjQ7PnHkRiRJAWOJj9B7uoUn9+bheEf0HvFa1QMrEi3gFy0KpUFTzcXu1OKiMiT0dwcRajDA0/ugWWBbgcSEZFoiG4x/LplWUejDhhjssdAHokpxsCLrXBkLkXSGR0YfmYIqzwO0PX35oxZe5Se1XLTuFgmXF2ie/S0iIjYTHNzFKEOTxxYEHYP3DztjiMiIvFAdCuf6dEck7guVU54fQlUeIfKd5aw1e9/lEnyJ+/P2E2NwWuYt/MMERH/aU4qIiJxj+bmKEIcHs4vQu/YG0REROKNR64MG2PyAQWB5MaYRlEe8sHZuVLiIxc3qPox5KxCkpmdGHijN2+UeIseJyrQY/J2Rqw6wrs181I5r586T4uIxDGamx8s1BH51kPv2htERETijcdtk84L1AVSAPWijN8EOsZUKIkl2crDG+swv79F7t0DWZhtE4tLf0a/dTdoP34L/llT8m7NvJTKkcrupCIi8jfNzQ8Qen9lWMWwiIhEzyOLYcuy5gBzjDFlLMsKjKVMEpuS+kLTn2H7r5iF71HrXCOq1R3ClFsvMmT5IZqP2kjFPH68WyMvhTIltzutiEiip7n5wULMXyvD2iYtIiLR87ht0u9ZltUfaGWMafnvxy3LejPGkknsMQaKtYEsZWDG67hOb8srxV6lyVtf8MvWiwxfdYR6Q9dRp1A6elXPQ640ifbkDhER22lufjBtkxYRkSf1uG3S+yI/B8V0EIkDUueC15fCyi9h/fd4nthAp8ZjaFkygDFrjzFm7VEW7TlHo2KZ6Fk1N5l9k9qdWEQkMdLc/AAhRg20RETkyTxum/S8yC/vWJY1LepjxpimMZZK7OPqDtU/g5xVYFZnGFMN72qf8HbVbrQtk5URq47wy8YTzNlxmtalstI1ICdpvBNtvxYRkVinufnBtDIsIiJPKrpHK/WJ5pgkFDkqQZcNkKcmLPkQfm1EqogrfFi3AKvfrUyT4pmZsPEElfqvov+i/Vy/E2p3YhGRxEZzcxR/F8NaGRYRkeh53D3DtYE6QEZjzJAoD/kAYTEZTOKApL7Q/FfYOh4W9YERZaH+MNLnq8NXjQrRuWIOBi87yIjVR/h14wk6V8pJ+3LZSOr+uN33IiLytDQ3P9g9o27SIiLyZB63MnwG2AoER37+62MuUDNmo0mcYAz4t4fOayB5JpjSEn7vBSF3yJbai+9bFGXBmxUomd2XAYsPULH/SsavP8a9sHC7k4uIJFSamx9A26RFRORJPe6e4Z3ATmPMr5ZlJdq/Ngvglwc6LIPl/4PAoXB8HTQZC+kKkT+9D2NeLcHWE1cZsHg/n87by+i1x/ikXgFqFExnd3IRkQRFc/ODaZu0iIg8qUeuDBtjdhtjdgHbjDG7onz8NS6JiasH1PwS2syC4OswugoEDoOICACKZ03J5I6lmfB6SZInceONX7cyfespm0OLiCQsmpsfLMy4O7/QyrCIiETT427urBsrKSR+yVkFuqyHOd1hcV84vBwajADvtBhjqJDbD/8uvnSaEETvaTsJCYugVaksdqcWEUkoNDc/gGUcBOOOp1aGRUQkmh65MmxZ1okHfQCZgPdiJ6LESV6poeVkeOlbOLHe2Vzr4OL7Dydxd2F0W38C8vrRd9Zuxq0/ZmNYEZGEQ3PzwwXjoZVhERGJtugerYQx5kVjTH9jzHHgC2B/jKWS+MEYKNEBOq0G73QwqRksePf+LyKebi6MbONPzYJp+WzeXkauPmJzYBGRhEVz8z+pGBYRkSfxuKOV8gAtgJbAZWAqYCzLCoiFbBJfpMkHHZbD8s9g43Bnc63GYyFtAdxdHQxtVYy3p+7gq4X7uRcWQY8quTDG2J1aRCRe0tz8cMG4q4GWiIhE2+NWhvcDVYF6lmWVtyzrB0Bn5sh/uXlCra+g9Qy4fQlGVYZNI8GycHNx8H2LojQqlpFBSw8ycMkBLMuyO7GISHylufkBjNHKsIiIPJnHFcONgXPASmPMaGNMVUBLevJwuatBlw2QoxIsfM+5dfrWRVwchoFNitCyZGaGrTzCl/P3qSAWEXk6mpsfwlkM37Y7hoiIxBOPa6A1y7Ks5kA+YBXwNpDWGDPCGFMjFvJJfJTMD1r9BrUHwNHVzuZah5bhcBj6NSxEu7LZGLPuGB/P+YOICBXEIiJPQnPzwwUbrQyLiEj0RauBlmVZty3LmmhZVl2c3Sp3AB/EaDKJ34yBUp2g00pn5+mJjWFRH0zYPT6pV4BOFXMwYeMJ+s7aTbgKYhGRJ6a5+b+c9wyrGBYRkeiJdjfpv1iWdcWyrJGWZVWJiUCSwKQtCB1XQMnOzuZaY6piLh6gT+189KiSiylbTtJ72k7CwiPsTioiEm9pbnbuE3duk1YDLRERiZ4nLoZFnphbEqjT37l1+uY5GFUJEzSWd6rn4Z3qeZi1/TQ9p+wgVAWxiIg8AzXQEhGRJ6FiWGJPnprO5lrZysP8d2BKK3qU9qVvnXzM332WLr9u425Iom+IKiIiT0lHK4mIyJNQMSyxyzsttJoGNb+Cw8tgRBk6ZTzB/+oXZPn+8zQbGci568F2pxQRkXgmVTIProW6YmllWEREoknFsMQ+hwPKdHXeS5wkJUxoSNsboxnTqjBHL97i5aHr2HXqmt0pRUQkHqmYOzV3LA9MeAiEh9kdR0RE4gEVw2KfdIWg40rwfx0Ch1J1fSvmtkiDm4uDZiMDmb/rrN0JRUQknngxcwoi3Dyd34RpdVhERB5PxbDYyz0p1B0ELSbD9VPknFmHReUO8EI6L7pN2saQ5YewLB29JCIij+bq4iCTXyoArBDdNywiIo+nYljihnx1oGsgZC2D9/IP+M31E3rmu8GgpQd5c8oOgkPVWEtERB4tY9acAJzcv9nmJCIiEh+oGJa4wzsdvDITGo3BcfM0bx3vwvzsM1i78wDNRwZy4YYaa4mIyMPlLlOfG1ZSgoMm2R1FRETiARXDErcYA4WbQvcgTOmuFDw3m80+71PowlwaDF3LntPX7U4oIiJxVLpUKVjnUYEsF5bDvVt2xxERkThOxbDETZ4+UKsfdF6De7r8fOEYyajQPnzy40QW7TlndzoREYmjLudsiKd1j3u7Z9sdRURE4jgVwxK3pXsB2i+EhiMpkOQq01z6cmFKd8Ys2abGWiIi8h85i1Xjzwg/bm2ZaHcUERGJ41QMS9xnDBRpgaPHViJKdOQV1+U0WF+fySO/Ijgk1O50IiIShxTP7ss8U4mU5wPh+mm744iISBymYljijyQpcH1pAKbzKkKSZ6PVuW841r8Cu4PWapVYREQA8HB14XTmejiwYPdvdscREZE4TMWwxDsmfREyvLWancX7kTbsNAXm1WNB/7as3nmIiAgVxSIiiV2OvIUJishD2PbJoD+WiojIQ6gYlvjJ4aBIvW4keXsHh7I0o9bd3ykwsyoDB3zGrG0nCQuPsDuhiIjYJF86H2aGV8D18gE4u9PuOCIiEkepGJZ4LUnyVOR7fRQRHVbg4puV9+4OJuPsxrzW/2d+CTxOcGi43RFFRCSW5Uvvze/hpQg3brBzit1xREQkjlIxLAmCW6ai+PZYTUS9IbzoeZ5x93oROv8Danz1O8NWHub6XTXaEhFJLFIn88A9mS9/eJeFPdMhXHOAiIj8l4phSTgcDhzFX8X9re04irfjNddFzOUtDiz9iXJfL+erhfu4cDPY7pQiIhIL8qXzYY5VEW5fhCMr7I4jIiJxkIphSXiS+mLqDcZ0XE6KdFkZ4j6MWUn7sWrtGsp/s5K+s3Zz4vJtu1OKiEgMypfOm6lX82Il8YWdk+2OIyIicZCKYUm4MhaHDsuh7nfk5k8WefZlTLrZLAw6RMDAVfSYvJ29Z27YnVJERGJAvvQ+3ApzcCNXfdi/AO5eszuSiIjEMSqGJWFzuIB/e+i+FfNiaypemkJQij4MKniElfvPU2fIWtqP28zmY1fsTioiIs9RvnTeAOxOVRvC78HeOTYnEhGRuEbFsCQOXqng5SHQYTkuPmlpcPgjdmQbSr/y7uw6dZ1mIwNpMmIDy/edx9KZlCIi8V6uNMlwGNh0Lyukyg27ptodSURE4hgVw5K4ZPKHjivhpW9xPb+LVttasLHEGvq9lJ2z14N5/ecgan23lnk7z6goFhGJxzzdXMjhl4z9529BkRZwYj1cPW53LBERiUNUDEvi43CBEh2gxzYo0gK3jUNotbkxq1+6xuBmhbGw6DF5O01+DGT3qet2pxURkadUPEtK1hy8yPGMdZ0Du36zN5CIiMQpKoYl8fJKDfWHwWtLwCsVrjPa03BPdxa1Tk//xoU5cfk2Lw9bx/vTd3Hp1j2704qIyBN6p0Yekri70HPRZSKyloedU0C7fkREJJKKYZEspaDjKqg9AE5vx/FjWZpd/4kVb5agY4UczNx+ioABqxi95ighYRF2pxURkWhK4+PJFw1eYOfJa6xwrwJXjsCpILtjiYhIHKFiWATAxRVKdYIeQVCoKawbhM+YcvTNdojFPSvgny0lXy7YR63v17DywAW704qISDTVLZyBl4tkoPcfWYlw8dCZwyIicp+KYZGokqWBhiOg/ULwTA6/tSHH4lcZ93IqxrUrARa0H7eF18Zv4ejFW3anFRGRaPhf/YK4e6VglaMU1h8zIUy3voiIiIphkQfLWhY6r4FaX8PJzTC8NAFnRrGomz996+Rj87Er1PxuDf0W7ONmcKjdaUVE5BFSJHWnf5PC/Hy7DObuVTi0xO5IIiISB6gYFnkYF1co3cW5dbpgQ1gzAPcfy9ApzX5WvFORhkUzMnrtUQIGrua3oJNERKgpi4hIXFU5bxoy+9fmopWcKxt+sTuOiIjEASqGRR7HOx00GgXt5oO7F0xpRZp5belfxZs53cqRxTcJ703fRYPh69l64qrdaUVE5CH6vFSIFW6V8D65gltX1f9BRCSxUzEsEl3ZysMba6HGl3BiAwwrTeHDI5nRsTiDmxfh/I1gGo/YwNtTd3D+RrDdaUVE5F+8PFwpWKczboSxfkp/u+OIiIjNVAyLPAkXNyjbHboHQb6XYFU/zIiyNPQ5xIp3KtO1ck7m7zpLwMBVjFx9hNBwHcUkIhKXvFCsPAdSVqLSufFsDNpsdxwREbGRimGRp+GTHpqOg1dmAhZMaIDXvE68Vy4Fy3pVomzOVHy1cD8vDVnLpqOX7U4rIiJRZGs7nDDjjvv8t7hxV52lRUQSKxXDIs8iV1XoEgiVPoB982BoCbIc/pUxbYoxuq0/t++F03zURt75bSeXbukXLhGRuMAjZSaulv+IYtYfLJ0wwO44IiJiExXDIs/KzRMC+jiL4ozFYOG7MLoK1ZOfZmmvinStnJO5O09T9dvVTNx0Ql2nRUTigMxV3+CET3Gqnx5K4I7ddscREREbqBgWeV5S54I2s6HxWLh5FkZXIenSD3ivUjoW9qxA/vTe/N+sPTQcsYE9p6/bnVZEJHEzhnSvjMTdhBM6921u3A2xO5GIiMQyFcMiz5MxUKgJdN8CJTtB0FgYWoJc5xYxuUMpvmv+Iqev3uHloev4dO4f3AgOtTuxiEii5ZEmN5dL9qZixBZ+nzzc7jgiIhLLVAyLxATP5FCnP3RcAckzwcwOmAkNaJD5DsvfqcwrpbPyc+Bxqn67mjk7TmNZ2jotImKHjDXf4axXPqqf+JYNuw/aHUdERGKRimGRmJShKHRYBnUGwpkdMKIsyQP78786OZnbrTzpk3vSc8oOWo/ZxOELt+xOKyKS+Li44ttyFCnNba7Neo+b2rEjIpJoqBgWiWkOFyjZ0bl1ukADWNMfhpem0N0tzOpajs8bvMDu09ep/f0aBizez92QcLsTi4gkKh6ZinCxSBfqRKxk2m+/2B1HRERiiYphkdjinRYaj4a2c8HhChMb4zK9HW0KuLHincrUK5yBYSuPUG3Qan7fdUZbp0VEYlH6uh9x2TMrNY70Y90fx+2OIyIisUDFsEhsy1EJumyAgA/h4CIYWgK/PWMZ1OQFpnQqjbenK90nbafpj4HsOnXN7rQiIomDmyfJmo0gk7nEyRl9uXJb3aVFRBI6FcMidnD1gErvQteNkKUMLO4DoytT2u0o89+swNeNCnH88m1eHrqeXr/t4Nz1YLsTi4gkeB45ynGlQFuahy9g5MQp2qEjIpLAxVgxbIz5yRhzwRizJ8qYrzFmqTHmUOTnlJHjxhgzxBhz2BizyxhTLMpzXo28/pAx5tUo48WNMbsjnzPEGGNi6r2IxBjf7NB6GjT7BW5fhrHVcZn/Fi1eSMbK3pV5o1JOft95loCBq/h+2SHdTywiz0Rz8+P5vvwldzzT0vjU10wNPGJ3HBERiUExuTI8Hqj1r7EPgOWWZeUGlkd+D1AbyB350QkYAc4JGvgEKAWUBD75a5KOvKZTlOf9+2eJxA/GQIH60H0zlO4K2ybA0BJ475/OB7XysqxXJQLy+TF42UGqfLuK2dtPExGh1QoReSrj0dz8aJ4+JG00hDyO01xa1E+d/kVEErAYK4Yty1oDXPnXcH3g58ivfwYaRBn/xXLaCKQwxqQHagJLLcu6YlnWVWApUCvyMR/LsgIt5x6mX6K8lkj85OENtfpB59XOFePZb8D4umSJOMnw1sWZ2qk0qZK589bUHTQasYFtf161O7GIxDOam6PHkbcmwfka0dkxm29/nU1IWITdkUREJAbE9j3DaS3LOgsQ+TlN5HhG4GSU605Fjj1q/NQDxh/IGNPJGBNkjAm6ePHiM78JkRiVrhC8tgTqfQ/n98CIcrDsM0plSsLcbuUZ0KQwZ67dpdHwDbw5eTunr921O7GIxG+amx/As94ALHdvOl0bzKAle+2OIyIiMSCuNNB60D1F1lOMP5BlWaMsy/K3LMvfz8/vKSOKxCKHA4q3gx5boVBTWDcIhpfCcWgxTf0zs7J3ZXpUycXiP85RZeAqvl1ygNv3wuxOLSIJS+Kem71S4153IEUdh7m34Uc2HL5kdyIREXnOYrsYPh+5jYrIzxcix08BmaNclwk485jxTA8YF0lYvFJDwxHQbgG4JYXJzWFKa7zunuWdGnlZ0bsyNQum44cVhwkYuIqZ206p+6mIPCnNzQ9TqAnhOavznutvDJi6hEu37tmdSEREnqPYLobnAn91nXwVmBNlvG1k58rSwPXIrVqLgRrGmJSRzTlqAIsjH7tpjCkd2amybZTXEkl4spWDzmuh6idweDkMKwXrh5DR25UhLYsyo0tZ0qdIQq/fdtJ6zCaOXbptd2IRiT80Nz+MMbjUG4y7myu9Q4bT/McNOupORCQBicmjlSYDgUBeY8wpY8zrwNdAdWPMIaB65PcAC4CjwGFgNNAVwLKsK8DnwJbIj/9FjgF0AcZEPucIsDCm3otInODqDhV6QbdNkL0iLP0IRlaEPzdSPGtKZnUpyxcNXmD36evU/G4NQ5Yf4l6YjmISkb9pbn4KKTLjUv0zypndlLq5hGYjAzl55Y7dqURE5DkwiW1Lpb+/vxUUFGR3DJFnt38+LHgPbpyCoq9A9c8hqS8XbgTz2e97mb/rLLnSJKNfw0KUzO5rd1qRBMsYs9WyLH+7c8RncX5ujoiA8XUIO7eXaiEDuOeRmokdSpHDL5ndyURE5AGiOzfHlQZaIvKk8r3kPJu4XE/YOQWGlYQ/ZpPGx5NhrYoxrl0J7oaE02xkIO9P38W1OyF2JxYRiZ8cDqg3BNfwYOanH0dE6D2ajdzI/nM37E4mIiLPQMWwSHzm7gXV/wedVoFPBpj2KkxtAzfPE5AvDUt7VaRzxRxM33aKqt+uZvb202qwJSLyNPzywMs/4HU2kCX5fsfFWLQYtZFdp67ZnUxERJ6SimGRhCBdIeiwwtlg6+Bi5yrxjskkdXOhT538zOtenky+SXlr6g7a/rSZE5fVYEtE5IkVaQ7l3yb53oksLLOPZB6utBq9iS3Hrzz+uSIiEueoGBZJKFxcnQ223lgHfnlh9hswsSlcP0WBDD7M7FKW/9UvyPY/r1Fj8BqGrTxMSFiE3alFROKXKh9D3pfwXfsJc2rfI423B23HbmbdIZ1DLCIS36gYFklo/PJA+4VQ6xs4sR6GlYagn3DBom2ZbCzrVYkq+dIwYPEBXhqylq0nrtqdWEQk/nA4oNFI8MtPqvmdmd4kNVlTJeW1n7ewcPdZu9OJiMgTUDEskhA5XKD0G9BlA2QsBr+/Db+8DFeOki65JyNeKc6Ytv7cvhdGkx838OX8vQSH6hgmEZFo8fCGVlPAxQ3fuW2Y2iYv+dP70GXiNt6euoOrt9WwUEQkPlAxLJKQ+WaHtnOg3hA4uxOGl4XAYRARTrUCaVnSqxItS2Zh9Npj1Bmylm1/apVYRCRaUmSBFhPh+imS/96BaR386Vk1N/N2nqH64NVaJRYRiQdUDIskdMZA8Veh60bIUQkW94WfasKF/STzcKVfw0JMeL0kwSHhNBmxga8W7NMqsYhIdGQpDfW+h2NrcF/ah7er52Fu9/KkS+5Jl4nb6DpxKxdv3rM7pYiIPISKYZHEInlGaDkFGo2By0dgZAVYMwDCQ6mQ24/Fb1ekeYnMjFxzlJeGrGW7VolFRB7vxVZQ9k0IGgubR1Mggw+zu5bjvVp5Wbb3AjUGr2bODh1rJyISF6kYFklMjIHCTaHbZsj3Eqz4AkYHwNmdeHu68VWjwvz8WknuhITTeMQGvl64X6vEIiKPU+1TyFMLFr4PR1bi6uKga+VcLOhZnmypveg5ZQcdfwni3PVgu5OKiEgUKoZFEqNkftB0PDSfCLcuwKgAWP4/CA2mUh7nKnHT4pn5cfUR6v2wjp0nr9mdWEQk7nK4QKPRkDoPTHsVLh0GIFcab6a/UZaP6hZg3eFLVB+8mt+2nNQqsYhIHKFiWCQxy18Xum2CIi1g7bfOrdMnN+Pj6cY3TQozvn0JbgaH0WjEBvov2s+9MK0Si4g8kKePs8O0wxUmN4e7zltNXByG18tnZ1HPihRI78N7M3bR9qfNnLp6x+bAIiKiYlgksUuSEhoMh1dmQOhdGFsDFvWBkNtUzpuGxW9XpFHRjAxf5Vwl1rnEIiIPkTIbNP8Vrp6Aae0gPOz+Q9lSezG5Y2k+b/AC205cpcbgNczafsq2qCIiomJYRP6Sqxp0DYQSr8PG4TCiLBxdTfIkbgxoWoRx7ZyrxE1+3MD/zdrN9buhdicWEYl7spaFuoPh6Cpn9/4oHA5Dm9JZWfx2RQplTM7bU3fSf9F+IiK0bVpExA4qhkXkbx7eSGESnAAAIABJREFU8NK30G4+GAf88jLM6wnB1wnIl4alvSrxWrnsTN78J9UGrWbezjO6901E5N+KtYEy3WHzSAj66T8PZ0qZlF87lKJlycwMX3WErhO3cSck7AEvJCIiMUnFsIj8V7by8MZ6KNsDtv0Cw0rDwcUk83Dlo7oFmNOtPGl9POgxeTvtxm3h5BXd+yYi8g/V/we5qsOCd+HYmv887ObioF/DQnxUtwBL9p6j6Y+BnL1+14agIiKJl4phEXkw96RQ4wt4fRl4JodJzWBmJ7hzhUKZkjO7azk+rluAoONXqD54NSNWHSE0PMLu1CIicYPDBZqMBd+cMLWN83z3fzHG2Vxr7KslOHH5DvWHrlf3fhGRWKRiWEQeLVNx6LwaKr0Pe2bAsJLwx2xcXRy8Vj47S3tVomJuP75ZtF8NtkREovJM7uwwbQxMbgHB1x94WUC+NMzoUhZ3VwfNRgYyf9fZWA4qIpI4qRgWkcdz9YCAvtBpFfhkcJ6jOfUVuHmODCmSMKqtPyPbFOf63VCa/LiBD2erwZaICAC+OaDZBLhyFKa1/0eH6ajypvNmdrdyFMqYnG6TtjFk+SH1ZBARiWEqhkUk+tIVgg4roOoncHAJDC0JW8ZARAQ1C6Zjaa9KtCubjUmbnA22lvxxzu7EIiL2y17B2ZzwyHJY+tFDL0udzIOJHUvRqGhGBi09SM8pO3T7iYhIDFIxLCJPxsUVKvSCLhsgQxGY/w78VAPO7SGZhyuf1CvInG7lSePtQacJWxm05ICODRERKd4OSnd1Hl0XNO6hl3m4uvBtsyK8WzMvc3ee4f9m7dYKsYhIDFExLCJPJ3UuaDsXGo50bv8bWRGWfAQhtymUKTkzupSlafFMDFlxmE4TtnIzWNumRSSRq/55ZIfp3g/sMP0XYwzdAnLxZpVc/BZ0iiHLD8diSBGRxEPFsIg8PWOgSAvoHgRFW8OGIc5jmA4swtPNhf5NCvPZywVZeeACDYdv4Nil23YnFhGxj4urs8N0qlwP7TAd1dvV89C4WCYGLzvItKCTsRRSRCTxUDEsIs8uqS+8/AO0X+g8kmlyc5jaBnPzLK+WzcaE10ty+dY96g9dx6oDF+xOKyJiH8/k0HIKGAdMag53H96B3xjDV40KUT5XavrM3M2agxdjMaiISMKnYlhEnp+sZaHzWqjyERyKbLC1aSRls6dkbvfyZEiRhNfGb2Hk6iO6B05EEi/f7ND8V7h6HKa1g/CH30bi7upgxCvFyJUmGV0nbmPvmRuxFlNEJKFTMSwiz5erO1TsDV0DIXMJWPgejKlK5uCDzOxaltovpOerhfvpOWUHd0PC7U4rImKPbOWg7mA4ugoW9Xnkpd6eboxvXxJvT1faj9/MmWt3YyejiEgCp2JYRGKGbw54ZSY0HgvXT8PoAJKu+IihjXPxbs28zNt1hiY/buDsdf1SJyKJVLE2ULYHbBkNm0c/8tJ0yT0Z374kd0LCaTduM7fuPfi8YhERiT4VwyISc4yBQk2g+xbnsSIbh2OGl6Zbuv2MfdWfPy/fodHwDRy+cNPupCIi9qj2GeSpBQvfhyMrHnlp7jTJ8M+akoPnb3H04q1YCigiknCpGBaRmJckhXM74OtLwTMFTG1Nle1vM711FkLDLZr8GMi2Px/eREZEJMFyuEDjMeCXF35rBxcPPvAyy7L4bN4frDxwkXdr5qVwphSxm1NEJAFSMSwisSdzSei8Gqr/D46uJO+0qiwtvQtfTwetR29ipTpNi0hi5OHt7DDt4ubsxn/nyn8u+X75IX4OPEGH8tnpWjmnDSFFRBIeFcMiErtc3KBcT+i6EbKVJ+W6z1js9Qm1Up6i489BzN5+2u6EIiKxL2VWaDERrp+C39r+o8P0+PXH+G7ZIZoUz8T/vZQfY4yNQUVEEg4VwyJij5RZodVUaPYLbsGXGXSjN8NSTOKjqRsYs/ao3elERGJfltJQbwgcXwsLeoNlMXv7aT6dt5caBdLydaNCKoRFRJ4jV7sDiEgiZgwUqA85AjArvqDG5lGs9dpA34Wt+fpmK96vnU+/+IlI4vJiS7h0ANYN5qCViXc2FqR0Dl+GtCyKq4vWMEREnif9ryoi9vP0gTr9MR2Xk9wvI8Pdh1Ay8A16jZrL5mP/vXdORCRBq/IxN7PVJOfWL3kl1UFGt/XH083F7lQiIgmOimERiTsyFsd0XIlV40squB+g39kOLB/Tl2bD17Bs73kiIiy7E4qIxDjLGLoHv8Fhk4VP7g3E+8YRuyOJiCRIKoZFJG5xccWU7Y7bm0G456lOH7fJ9LvYnaETplDr+zXM3HaK0PAIu1OKiMSYBbvPsfr4XfZUGoXDLYmzw/TN83bHEhFJcFQMi0jclDwTLq0mQYtJ5PQOY6bnZ7QMnkbv37ZTecAqxq8/xt2QcLtTiog8V3dDwvly/l7yp/ehQaVS0GIS3DwHw0rAlrEQof/3RESeFxXDIhK35XsJ0zUQR4H6tL83gaCsw8if7A6fzttLuW9W8P2yQ1y7E2J3ShGR52LE6iOcuR7Mp/UK4OIwkLkEdF4L6YvA/F4wphqc2W53TBGRBEHFsIjEfZ7JoclP8PIP+F7ewZi7b7G4XghFM6dg8LKDlP16BZ//vpez1+/anVRE5KmdvHKHH1cfoV6RDJTKkervB/zyQNu50Hgs3DgNowJg/jtw95p9YUVEEgAVwyISPxgDxdpCp1Xg5Ufepe0Ym2Eui3qUombBdIzfcJyK/Vfy7rSdHL5wy+60IiJPrN+CfbgYQ5/a+f77oDFQqAl03wKlOkPQTzDUH3ZOAUvNBUVEnoaKYRGJX9Lkg44rwP812DCEfAuaMbh6clb1rkyrklmYt+sM1QevpvOEIHac1KqJiMQfu05dp2R2XzKkSPLwizyTQ+1vnH8YTJEVZnWG8S/BhX2xFVNEJMFQMSwi8Y9bEqg7GJr+DJcOw8iKZD6ziM/qv8D696vQPSAXgUcu02DYelqO2siagxextHIiInFcjYJpCTxyOXp9ENIXgdeXQr3v4cJe+LE8LP0Y7mlnjIhIdKkYFpH4q2ADeGMt+OWF6e1h7pukcg/nnRp52dCnKv9XJz9HL92i7U+bqfvDOn7fdYZwnVUsInFU42KZCAmPYN6us9F7gsMBxdtB961QpCWs/x6GlYS9c7V1WkQkGlQMi0j8ljIrtF8I5d+GbT/D6AA4v5dkHq50rJiDNe8F8E3jQtwNCaf7pO1U/XYVkzb9SXCojicRkbilYAYf8qXzZsbWU0/2RK9UUH8ovLYEkqSE39rAxKZw5WjMBBURSSBUDItI/OfiBtU+hVdmwp3LzoI4aBxYFh6uLjQvkYWlvSoxonUxvD3d6DtrNxX6r+TH1Ue4GRxqd3oREQCMMTQulokdJ69x5OJTbHfOUgo6rYaaX8GfG2FYaVj1DYQGP/+wIiIJgIphEUk4clWFN9ZDljLw+1swrd39o0dcHIbahdIzt3s5JnYoRd603ny9cD9lv15B/0X7uXUvzN7sIiJA/aIZcBgYv/44N57mj3UurlCmq7PrdP66sKofjCgDh5c9/7AiIvGcSWxNZfz9/a2goCC7Y4hITIqIgA1DYMXn4JMBmoyDTP7/uWzXqWv8uPoIC3af461quXmrWh4bwkp8Z4zZalnWf/+BSbRpbv6nNyZsZdEf5wDIlDIJBdL7kD+9DwUy+FAgvQ+ZUibBGBO9FzuyEhb0hsuHoUB956px8owxmF5ExH7RnZtVDItIwnVyC8x4DW6cgSofQdk3nQ1n/qVUv2VUzpOGb5oUtiGkxHcqhp+d5uZ/Cg4NJ/DoZfaeucHeszfYd/YGxy7dvt8Ty9vT1VkcR37kT+9D7rTJ8HRzefALht1z/oFwzUAwLhDQB0q94bzFREQkAYru3OwaG2FERGyRuQR0Xgvz3oRln8Cx1dBwJCRL84/LfDzduH5X9w6LSNzg6eZCQN40BOT9+/+qOyFhHDh3k71nb7D3jLNA/i3oJHdCnM0AXRyGnH5ezgI5g7NAzp/eh9TJPMDVAyq+C4WawsL3YcmHsGMSvDQIspax622KiNhOxbCIJGxJUjjPI946HhZ9ACPKQaNRkDPg/iXJk7g93b15IiKxJKm7K0WzpKRolpT3xyIiLE5cuXO/ON579gYbj15h9o4z969J4+1xvzgukN6H/NXGkr3oalwWvQ/jasGLraHaZ5DMz463JSJiKxXDIpLwGQP+7SFzKed5xBMaOo9iCugLLm74JHHj/A11WxWR+MXhMGRP7UX21F68VDj9/fErt0PYF7m9+q+t1usOXSIs8px1TzcHRdJ+zxt+s6i4czLW3nmEB3yMR6nXwPGQrdYiIgmQimERSTzSFoCOK50rxOsGwclN0GwCyZO4ceDcTSzLin5TGhGROMrXy51yuVJTLlfq+2P3wsI5dP7W/RXkfWdv0PNMPfzuFebzsHGUXdyb00sGstW3NldyNqZq6eJk9k1q47sQEYl5aqAlIonTrt9gTnfwycCkXAPouzaECrlT82WDQmRJpV8AJfrUQOvZaW62h2VZnLkezN7T1wnZPYucx6eQL3gHEZYh0HqBm/maU7lBezyTJrM7qojIE1E36YfQhCsi953cDFNaYYXdY0nB/vQKSkm4ZdGzah46VMiOm4uOYpfHUzH87DQ3xyFXj3Nz4y+EbP2VVGHnuUlSbuR8mYwBHSFjcedtJyIicVx052b9piciiVfmktBxBSZ5Jmpu78aGaseplMePbxbtp94P69j+51W7E4qIxK6U2fCu/TGp+u5nd9VfCXQtie/hmTCmKqFDSsC67+DmObtTiog8FyqGRSRxS5EFXl8CuaqRfMX7jEw9jdGti3DtTiiNRmzg4zl71GlaRBIfh4NCFepR+YNZTKq4jI8iOrH7ioFln2ANKgATm8HeORAWYndSEZGnpm3SIiIAEeGw5CPYOAxyVedWvVEMXH2WnwOPk8bbg0/rFaTWC+nUYEv+Q9ukn53m5rjv7PW7fDF/H/t2b+U1r0Cauq3F4+4FSOILhZvBi60gfRG7Y4qIALpn+KE04YrII20dD/PfgVS5odUUdt5KQZ+Zu9l79gZV86Xh/dr5yJPW2+6UEoeoGH52mpvjj/WHL/HxnD0cu3iTd3KepmuKjZj98yE8BNIWgqKtoVAz8Epld1QRScR0z7CIyNMo3g5emQk3z8DoKhSx9jO3ezn61snH5mNXqPndGnpN3cGfl+/YnVREJNaVy5WahT0r0rFiLgYcycwW/0HwzgGoMxBcXJ1H132bF6a+AgcWQXiY3ZFFRB5KxbCIyL/lqAQdVoBnCvi5Hq57ptGpYk7WvBdAp4o5mL/7LFW+XcX/zdrNuevBdqcVEYlV7q4O3qyam6TuLszYegqS+kLJjtBpFXTZAKU6w58bYXJzGJQflnwIF/bbHVtE5D9UDIuIPEjqXNBhGWQuBbM6wfLPSZnElT6187PmvQBalszC1C0nqTRgJf0W7OPKbTWREZHEw8vDldovpGf+7rPcDQn/+4G0BaHml9BrH7SY5Ozav3EEDC8Fo6vAlrFw95p9wUVEolAxLCLyMEl9nVumi7WFtQNhejsIuUNaH08+b/ACK96pzEuF0zN67VEq9l/J4KUHuanO0yKSSDQunpFb98JY/McDjlpycYN8L0GLidBrP9TsB6HBML8XDMwD01+Dw8udzQtFRGyiBloiIo9jWRA4zLnVL8OL0GIy+KS///DB8zcZtOQgi/44R8qkbnSpnJO2ZbLh6eZiY2iJLWqg9ew0N8dPEREWFfqvJFvqpPzcviSuLo9ZY7EsOLsDtk+E3dMg+Br4ZIQiLZ3dqFPljJ3gIpLgqZv0Q2jCFZGndmAhTH8dPJNDqyn/OUZk16lrfLNoP+sPX6Z7QC5618xrU1CJTSqGn53m5vhr8NKDfL/8EK4OQ5ZUScmROhk5/bzI6ZeMHH5e5PBLhq+X+3+fGBoMBxc6C+Mjy8GKgCxlnd2oC9QHD3XtF5GnF9252TU2woiIJAh5a8Pri2FSC/ipFjQaDfnr3n+4cKYUFM/qy/rDlwnIl8bGoCIisaNHlVxkS52Ug+dvcfTiLY5evM2agxcJCY+4f02KpG7kSO0sjHP4ed0vmLPkfRmPgg3hxhnYOQV2TIQ53WDBe86CuGhryFoOdL67iMQQFcMiIk8iXSHouAKmtHQeHVKzH5TpCsDZ63cZteYIdQunp3jWlDYHFRGJea4uDhoWzfSPsbDwCE5fu8vRi7c5cvEWRy7e5ujFW6w+eJHpW0/dv85hILNv0shCuRY5SjSiMIfIdXo2nvvmYHZOgjQFIaAP5KurolhEnjsVwyIiT8o7LbSbDzM7wuI+cPMsVPuMAYsPEGHB+7Xy2Z1QRMQ2ri4OsqbyImsqr//skrkRHMqxi7c5esm5ivxXwbzhyGXuhf21mlwHP4+atPbeRqur00gz9RWupSjAjdLv4Ve0Hkk89OuriDwf+t9ERORpuCWBpj/DwvdgwxCuXjjJvD0Neb1SXjL7JrU7nYhInOTj6UaRzCkokjnFP8YjIizOXL8bWSDf4uil2wRdTMuMC2UpGbKcN6/MJOuidmxfkItfPFpzKW1Zcqbxvr/tOoefF+l8PHE4tHosItGnYlhE5Gk5XKDOQPBOR8oVX/CL51FeKDvb7lQiIvGOw2HIlDIpmVImpWIev388diekCscuvMvOoF/JtW84g+99xh/nCjDwz8aMD8l//7okbi7k8POiWJaUlM6RilI5fEmdzCO234qIxCPqJi0i8hyMG/o5bS4NgrSFcG0zA5L5Pf5JkiCom/Sz09ws0RZ2D7b9Amu/hZtnuZe5HAfzv8kOR36OXrzFofO32PbnVe6EOM8vzpUmGaVz+DqL4+yp8PNWcSySGOhopYfQhCsiMeHAuZsM+OE7Rrj/gFuKDNBmJvjmsDuWxAIVw89Oc7M8sdBg2DoO1g6C2xcgZxUI+BAyFSc0PII9p6+z8egVNh69TNDxK9yOLI5z+nlROkeq+yvHabw9bX4jIhITVAw/hCZcEYkpfWft5mDQCqYmG4SLqxu0ngYZitodS2KYiuFnp7lZnlrIbdgyBtZ9B3evQJ5aULkPZHjx/iVh4RHsOXODjUcvs+noZbYcv8qte2EA5IhSHJfO7ksaHxXHIgmBiuGH0IQrIjHl4s17BAxcRYNMt/ni1ifOX8yaT3CuWEiCpWL42Wlulmd27yZsGgkbfoDga86jmAL6QtqC/7k0LDyCP87cYNOxy2w8eoUtx65w86/iOLUXpXKkur+1Oq2KY5F4ScXwQ2jCFZGYNHzVYfovOsC4xhkJCOoGF/dDgxFQuJnd0SSGqBh+dpqb5bkJvg4bR0DgMLh3Awo2dK4U++V96FPCIyz2Rq4cbzx6mc3Hr3Az2FkcZ0/tRe0X0tGubDatGovEIyqGH0ITrojEpODQcJr+GMjeszcYWDcbDQ++B8fXQo0voGwPu+NJDFAx/Ow0N8tzd+eKsyDe9COE3oFCTaHS+5Aq52OfGh5hse+sszhed/gSqw9exNVhaPBiRjpWzEGetN6x8AZE5FlEd252xEaYfzPGHDfG7DbG7DDGBEWO+RpjlhpjDkV+Thk5bowxQ4wxh40xu4wxxaK8zquR1x8yxrxqx3sREYnK082FyZ1KUzZnKt6ee4yhGb7BKtAAlnwIi/pCRITdEUUeSHOzJChJfaHqR9BzF5TpDnvnwtASMLsbXD3+yKe6OAwvZExOhwo5GN++JKt6V6ZlySzM23WGGoPX0G7cZjYcvkRiW1ASSYhsWRk2xhwH/C3LuhRlrD9wxbKsr40xHwApLct63xhTB+gB1AFKAd9bllXKGOMLBAH+gAVsBYpblnX1UT9bf30WkdgQGh7B+zN2MXPbaVqVyMgXnhNxbBkFLzSBBsPBVcd7JBQJZWVYc7MkaDfPw/rvYMtYsMKh6CtQ8V1IninaL3H1dgi/bjzBz4HHuXQrhIIZfOhUMQd1CqXHzcWW9SUReYg4vTL8EPWBnyO//hloEGX8F8tpI5DCGJMeqAkstSzrSuQkuxSoFduhRUQexM3FwbdNi9A9IBeTtpymw4WmhAR8DHumw8SmEHzD7ogi0aG5WRIG77RQ6yvouQOKt4PtE2FIUVjwLlw/Fa2XSOnlTo+quVn3fhW+blSI4NBwek7ZQaX+Kxmz9ig3g0Nj9j2IyHNnVzFsAUuMMVuNMZ0ix9JalnUWIPJzmsjxjMDJKM89FTn2sPH/MMZ0MsYEGWOCLl68+BzfhojIwxlj6F0zL182fIFVBy/SZHcpzgUMhuPrYGprCA+zO6JIVJqbJeHzyQAvfQtvbociLSHoJxhcEMZUd3aivnrisS/h6eZCi5JZWPp2Jca+6k9m36R8MX8fZb9awZTNf8bCmxCR58WuYricZVnFgNpAN2NMxUdcax4wZj1i/L+DljXKsix/y7L8/fz8njytiMgzaF0qKyPb+HPi8h3KL07HvGx94NgaWPaJ3dFEotLcLIlHiszw8hDosRUCPoSwu87eDt8XhpEVYc1AuHjwkS/hcBiq5k/L1M5lmNOtHHnSefP573u5oRVikXjDlmLYsqwzkZ8vALOAksD5yC1WRH6+EHn5KSBzlKdnAs48YlxEJM6pXiAtK/6/vfsOj6pM+zj+vVOpSQgk9E6Q3gmiIiKK4qJYULD3uta1l93Vdd13LaurYlu7rqKoiyKoIEUQFBCQXiQUpRN6C6nP+8c5kYgZSEiZSeb3ua65Mjlzyj1PTuae+5znPOfOvgzp3ohblrZjZMRA+H44bsHIYIcmAig3S5iq1Qz63g03TPPOFp/6N4iIhkmPwgs94YVeMOkx2LQQDjPOTufGCTxyVnv2ZeUy8oe1AecTkdBS7sWwmVU3s5r5z4EBwCJgNJA/6uTlwGf+89HAZf7IlccCu/yuWuOAAWZWyx/dcoA/TUQkJNWuEcs/z+vEqJuO4/2EG5iZ14asUTfz8+IZwQ5NwpxyswiQ2AKOvw2unQh3LIGBT0C1OvDtU/DyCd41xuP/DOtmF1oYd2gYT2rzRN76bg25eRppWqQiCMaZ4brANDObD8wCxjrnvgL+CZxqZiuAU/3fAb4AVgFpwKvATQDOue3Ao8AP/uNv/jQRkZDWtUktPrm5L+tOeYkdrgaRIy/mX6O+0+ArEkzKzSIFxTeEXtfDlWPhzp9g0L+9YnnGi/Baf+864y/vhTXTIS/318WuOr4563ZkMH7xpiAGLyJFFZRbKwWTbt8gIqFkd9pMqr43iJm5KdwV81ceO68L/dvWDXZYUgyV5dZKwaTcLBVGxg5Y/hUsHQ1pEyE3E6onQZtB0PZMcpv2od8z06kaHcnrV/SgUa1qwY5YJCxVxFsriYiEnbhWvYg+6xlOiFjM3ZEjuPrt2Tw4aiEZWblHXlhERMpX1VrQ5UK4cATcsxKGvAHNToAFI+G/5xL5rxQ+SH6bVju+5cxnJvDujJ/JU5dpkZAVFewARETCXtdLYMM8zvvhVSLbd+WOWfD9qm08O7QrHRvFBzs6EREpTGxN6HCe98jOgJWTYMloGiz/khciRpFhVZkwtjPPzjyZ8y64kib1k4+8ThEpV+omLSISCnKz4e2zYMOPzBswkhsnZpO+J5M7Tm3NDX1bEhlR2B1rJBSom3TJKTdLpZKTBWum4paMJnPR51TJ2s4BF82m5BNocvwwIo45HaomBDtKkUqtqLlZxbCISKjYuwVe6QuRUey+dAIPjFvPmAUb6dmsFk9f0IXGibr2LBSpGC455WaptPJy2b7kG34c9zbtd0+lnu3ARURjLfpC27OgzR+gep1gRylS6eiaYRGRiqZGMgz9L+zZTNyYa3l+aEeeGdqZZRv3cNbwaXy/cluwIxQRkeKIiCSxQ39O/tPbfD94KpfwGK9ln8aOX5bA57fCUynw1iCY9Srs3hjsaEXCjophEZFQ0qg7DHoaVk/BJjzMOV0b8fktJ1C7RiyXvj6T92f+EuwIRUSkmMyMc7o14d93XcuqrvfSbc+TDIt4koUtrsHt3QJf3AVPt4HXToXvnoeMncEOWSQsqBgWEQk1XS+BntfA98Ph67/SrHY1/nfTcZyQUocHRi3k4dGLycnNC3aUIiJSTHVqxPJ/53bi85v7kFe3I2cuPokzcv/FvLPGQ7+HIOcAjH8IXj4BfpkR7HBFKj0VwyIioWjgE9D9Spj+bxh1A3HR8PrlPbm2T3Pe+m4NV7z5A7v2Zwc7ShEROQodGsbz4XXH8sJF3didkc3ZI7dy49qTWXvBOLh6AkREwpsDYcoTkKdb7YmUFRXDIiKhKCISBj0D/R6EBR/A+xcQmb2XB//QjieGdGLm6m2c/eJ00vdkBjtSERE5CmbGHzrVZ+Kdfbnz1NZ8szyd/k9P4V9L48i6Zop3y6bJj8HbZ8Ku9cEOV6RSUjEsIhKqzKDvPXDmc7BqijfIyt4tXNCjMa9c2p3VW/fx7Yr0YEcpIiIlUCU6klv6pzDprr6c0aEez09K45zXF5J2wtNw9suwYR68fDwsHRPsUEUqHRXDIiKhrvvlMOx9SF8Orw+AbStpmODdZik2KjLIwYmISGmoH1+Vfw/ryn8u7c6GnRkMGj6ddzN6466fCglN4cOLYcyfIDsj2KGKVBoqhkVEKoJjTofLP4cDu+D1AbgNcwGoFqtiWESkMhnQvh7jbj+R1Oa1+fNni7l6zA7Sh46B426B2a/Df/rB5iXBDlOkUlAxLCJSUTTuCVePh5hqpHwxjBMj5lMtWsWwiEhlkxxXhbeu6MnDZ7ZjWtpWBg6fwVcNbibnoo9h/1Z4tR/88Do4F+xQRSo0FcMiIhVJnRS4+msO1GzK69FPsWz8q+zK0KjSIiKVTUSEccXxzRlzywnUqRHLDf+dQ/t3criy6r9Jq9YZxv6J/e8Ow+3fHuxQRSosc2F2RKlHjx5u9uzZwQ5DRKRE3IEmu7LxAAAgAElEQVRdrH1pCE12zeKFyEvoMuxhjk9JCnZYYcnM5jjnegQ7jopMuVnk8DJzcpmwZAvz1u5g3tqdLFq/g4vyxnJv1AdstwTeqns/VVNOonPjeLo0TiChWkywQxYJqqLmZhXDIiIVVU4WO96/mlqrRvNmzmmsTX2Iewa2p4q6TpcrFcMlp9wsUjzZuXks37SHtYu/o8fsO0nM2sjwnLN5LucccomkeZ3qdG7kFcadGyfQrkGcBlyUsKJiOAAlXBGpVPLyyB73ENEzX2BsbirDE+7hyWGpdGgYH+zIwoaK4ZJTbhYpgcw98MU9MP99did15/PmDzJlazzz1u5ki38v+pjICNo2iKNLo3i6NEmgS+NaNKtdDTMLcvAiZUPFcABKuCJSKX33PIx/iPnWhmsz7+CcPp259eQUqsdGBTuySk/FcMkpN4uUggUfwZg7IGsPtOiH63Elm+r1Y976fcxbu5N5a3eycP0u9mflAtAgvgrn92jMBT0b0zChapCDFyldKoYDUMIVkUpr8SjcqBvYbolcsPcO9tVsyUOD2vKHjvV19L8MqRguOeVmkVKyZxPMfQfmvA2710GNutD1Uu9+9QlNyMnNY8WWvcxbu5MvFm5kWtpWAPq2TmJYzyb0b5tMdKTG15WKT8VwAEq4IlKprZsDI4aRk53BX2Pv5b30FhzXsjZ/G9yeVsk1gx1dpaRiuOSUm0VKWV4urPga5rwJP43zpqWcCj2ugpQBEOFdP7x2+35Gzl7LyNlr2bw7k6SasQzp3ohhPRvTtHb1IL4BkZJRMRyAEq6IVHo7f4H3h+HSlzGj7YNcv6QD+7NyufqE5tzSP4Ua6jpdqlQMl5xys0gZ2rnWO1s89x3YuwniGkK3y6HbpRDXAICc3Dy+WZ7OiFm/MHn5FvIcHNeyNsNSm3Ba+7oafEsqHBXDASjhikhYOLAbPr4K0r4mo8eNPJIxlA/mbKBeXBWev6grPZslBjvCSkPFcMkpN4uUg9xs+OkrmP0GrJwEFgnHDITuV0LLkyHC6x69cVcGH81ex4c/rGX9zgxqVYumXYM4kmtWIalmLMk1Y0nyH8k1q5AcF0vN2ChdjiMhRcVwAEq4IhI2cnNg3AMw6xU45g/MS32SO0atYP2ODJ48vxODuzQMdoSVgorhklNuFiln21d51xX/+F/YvxUSmnrXFXe9FGokA5CX5/g2bSuf/rieNdv2kb4nky17MsnKyfvd6qpER/xaHCfViCU57mDRXLCIrl0jlsgIFc1S9lQMB6CEKyJhZ+Z/4Kt7oW4Hdp3zX67/bAMzVm3n9lNSuK1/io7ml5CK4ZJTbhYJkpwsWPY5zH4T1nwLEVHQZhD0uBKanfjr2eJ8zjl2Z+SQvvcAW3Z7xbFXJB8o8DyTLbsPsPtAzu82F2GQWN0rjJPjYgsUzgcL5vznVWPUNVuOnorhAJRwRSQs/TQePr4SYmuSfcEI7vs+gk/mruPsLg14fEgnXQ9WAiqGS065WSQEbF0Bc96Cee9Bxg5IbAndr4AuF0P12sVe3YHs3F+L4/Q9maTvOXCwWC5QRG/dm0Vu3u/rkZqxUST9WjBXKXCmuUAX7ZqxJFSL1kFd+R0VwwEo4YpI2Nq8GN4fCvu34c59lRc3teHJccvp2awWb1+VSrUYDax1NFQMl5xys0gIyT4ASz7zri1eOwMiY6DdYG8k6ia9oZQLz9w8x479Wf6Z5gOHFNAHzzpv2Z1JRnbu75aPjjSSasSSFOd10U6pW4Pb+qdQJVoHecOZiuEAlHBFJKzt2QwfXAjr58KAR/ms6jnc9uF8bu2fwp9ObR3s6CokFcMlp9wsEqI2L/FuzzT/A8jcDXWO8YrizkOhaq1yD2dvZo5XIO8+pFu2X0Sn78lk2aY9XHFcMx4+q325xyehQ8VwAEq4IhL2sjNg1A2w5FNIvZ5bdlzA18vSmXzXSdSPrxrs6CocFcMlp9wsEuKy9sGi/3mF8fo5EBkLTY6F5idCi5OgfheIDI3eRY98vpg3p6/h9ct70L9t3WCHI0FS1NwccaQZRESkkomuCkPehN43w6xX+L+YN3Aujye/Wh7syEREJBTFVPfuS3ztJLh+KvS8BvZthUmPwmv94Ynm8P4w+P5F2LQI8n4/4nR5uW9gG9rWj+PujxewZfeBoMUhFUNoHMIREZHyFREBA/4OUVWo8e1TfFxvB4N/vIjLjmtGl8YJwY5ORERCVf3O3gNgb7o3CvXqKbB6Kvz0pTe9Wh1o3gea9/XOHie2KPVrjQOJjYrk+Qu7MOj5adz2wTxevLgbtarHlMu2peJRMSwiEq7MoP+fISqWjpMf4+Vqu/jTiCqMuuUk4qtGBzs6EREJdTWSoMO53gNg5y+wukBxvHiUNz2+sVcU5xfHcfXLNKxWyTV5dHAH7v1kASc99Q239k/h0mObEhOlTrHyWyqGRUTCXd97IDKGARP+itubxd0fVOfly3sTEaFbVYiISDEkNIGuF3sP52BbGqz6xiuMl3/h3bYJoE5rvzg+EZr1gWqJpR7K+T0a07FRPI+NXcqjY5bw7vdruP+MtgxoV1e3YpJfaQAtERHxzHgZvrqXibldWXbicP54aodgR1QhaACtklNuFgkDeXmweSGs8s8a//wdZO8DDOp1hBZ9vTPHTXpDbI1S26xzjm9+SuexsUtJ27KXY1sk8tAf2tGhYXypbUNCj0aTDkAJV0QkMPfDG9jYO/g2ryN7Br/NGd1bBjukkKdiuOSUm0XCUE4WbJh7sDheNwtysyAiChr28Eeq7guNekJUbMk3l5vHiFm/8MyEFezYn8Xp7evRJyWJXi0SaVGnus4WVzIqhgNQwhURObys2e8SNeYWZuW1YVrqC9w+sCtRkbrOKhAVwyWn3CwiZO2HtTO8wnjVFNg4D1weRFUtcBunvt5tnCIij3ozuzKyeXFyGp/MXc/WvZkA1KkRQ2rzRFKbJZLavDZt6tXUpUIVnIrhAJRwRUSOLHveh0R8eiM/5rXkpYb/5ImLT6B2jZIfma+MVAyXnHKziPxOxk74efrB4jh9qTc9Nh6aHX9wMK7ktkc1UrVzjtVb9zFr9XZmrd7OzNXbWb8zA4C4KlH0bJboFcjNE+nQMJ5oHRSuUFQMB6CEKyJSREs+I++jq1ic14S7Yv/K45eepNsuFULFcMkpN4vIEe3d4hXG+SNV71jjTa+edHCk6lb9Ib7RUW9i3Y79/LBmOzNXeQXyqq37AKgWE0n3prX8M8eJdG6cQJXooz87LWVPxXAASrgiIsWw/EvyPryMha4lF2U9yKPndeXcbkf/RaMyUjFccsrNIlJsO37+bXG8d7M3venx0PF8aDe4xKNUb9lzgB9W72DW6m3MXL2dZZv2ABATGUHXJgmc2DqJvq2TaFc/Tt2qQ4yK4QCUcEVEimnhx/DJ1UyodgbXbL+Eu087hptOaqnBRnwqhktOuVlESsQ5SF8OS0fDgpGwbQVEREPKqdBxCLQeCDHVSryZnfuzmL1mBzNXb2N62jaWbNwNQO3qMfRJqcOJrZPok5JEUk1dVhRsRc3Nus+wiIgcXschsHkRp0x7hieaHMM942DTrgM8fFZ7InUkXEREgs0Mktt4jxPvho3zYeFH3sHc5V9ATA1oe6aXz5qfBJFHVwIlVIvhlHZ1OaVdXcA7c/ztT1uZuiKdqSu28um8DQC0qx9H32OSODElie5NaxETpeuNQ5XODIuIyJHl5cKIC3ErJ/Je6+d4aF4CA9rV5bkLu4b9dVM6M1xyys0iUibycmHNNFg4EpZ8Dpm7vGuM258LnS6Aht2PavCtQjeV51i8YTdTV6QzZXk6c3/ZQU6eo3pMJL1b1v61S3XT2tVLZXtyeOomHYASrojIUTqwC147BfZv4+Nu73D3xF30bJbIG1f0pEZs+HY0UjFccsrNIlLmsg/AivHeGeOfxkFuJtRq7l1f3OkCqJNSqpvbcyCb71ZuY+pP6Uxdkc7a7d5I1a3r1uC09vU4rX092jeI0yVHZUTFcABKuCIiJbA1DV49GRIa80Xq29zyyU90bhTPW1elElclOtjRBYWK4ZJTbhaRcpWxE5Z+7hXGq6cCzrt/ccfzocN5EFe/VDfnnGPNtv1MWraF8Ys38cOa7eQ5aJhQlVPb1eW09vXo2awWUbp9U6lRMRyAEq6ISAmlTYD3zoe2Z/Jlm39yywfzaN8wnneuTCW+WvgVxCqGS065WUSCZvdGWPw/b+CtjfMAg+Z9oOMF0O4sqBJf6pvctjeTiUu3MH7JJqau2EpWTh61qkVzStu6DGhfjz4pdcL+EqSSUjEcgBKuiEgp+O55GP8Q9HuQr5Mu56b35tC6bk3+e3UvalWPCXZ05UrFcMkpN4tISEj/CRZ97BXGO1ZDZCy0HuAVxikDILpKqW9yX2YOU35KZ9ziTUxatoU9B3KoFhNJ39ZJnN6hHv3b1g3rS5GOlorhAJRwRURKgXMw6npY8CEMfY/JEalc/+4cWtetwXvXHEt81fA5Q6xiuOSUm0UkpDgH6+d43agXfQL70iE2Htqd6RXGzU6AiNI/c5uVk8eMVdsYt3gTXy/ZzJY9mcRGRdC/bTKDOjXg5DbJOmNcRCqGA1DCFREpJdkZ8OYZkL4MznudSXTn+nfn0KFhPO9e3StsjmSrGC455WYRCVm5ObB6ilcYL/0csvZC9WRoO8i7XVOzPhBZ+geA8/Icc37ZwefzN/DFwo1s3ZtF9ZhITm1Xl0GdGtCndR1io1QYB6JiOAAlXBGRUrRnM4wYChvmQf8/81XCRfxxxI90b1qLt69MpWpM5U/UKoZLTrlZRCqErP3w01ew5FNY8TVk74eqteCYM6DtWdCyH0TFlvpmc3LzmLl6O2MWbODLRZvYuT+buCpRnNa+Hmd2bsBxLWtr8K1DqBgOQAlXRKSUZWfAZ3/0upJ1PJ8xzR/g1o+Wkto8kVcu6VHpB9VSMVxyys0iUuFk7YeVE72zxcu/8u5hHFMTWp/mDbzV6hSIKf17Cmfl5DE9bSufL9jA+MWb2ZuZQ50asfz97A6c3qFeqW+volIxHIASrohIGXAOvv0XTHoUGnRjXMenuXnMRhonVuONy3vSrE7pfyEIFSqGS065WUQqtJwsryv1ks9g2VjI2A5RVaFVf2g32CuQy2BU6gPZuUz5KZ3hk9JYuH4Xl/VuygNntNV1xagYDkgJV0SkDC0bC59cC1XiWHziS1zyZTYOeOWS7vRqUTvY0ZUJFcMlp9wsIpVGbg78PN07Y7z0c9i7CSJjoMVJXlfqNn+AaomlusmsnDyeHLeMV79dTZt6NRl+UVdaJdcs1W1UNCqGA1DCFREpY5sXw4hhsHcL6f2eYuiMxqzdvp9/nNOR83s0DnZ0pU7FcMkpN4tIpZSXB+t+gKWjYclo2PULWCQ0O94rjNueCTVLr2vz5GVbuPOj+WRk5fLnQe0Y2rMxkRFWauuvSFQMB6CEKyJSDvZthZGXwc/TOXDs7Vyz9jSmrdzB0B6Nefis9pVqYC0VwyWn3CwilZ5zsHH+wcJ42wrAoHEv7xrjtmdCQpMSb2bz7gPc8eE8vlu5jRZJ1bm5XyvO6twg7AbYUjEcgBKuiEg5ycmCL++GOW+R13ogzyXczbPfbqJVUg1euLgbretWji5cKoZLTrlZRMKKc95tCZeM9orjzYu86fW7+IXxYKjT6qhXn5fnGLd4E89OXMGyTXtoWrsaf+zXinO6NiQ6TIpiFcMBKOGKiJQj52DWq/DVfZB0DLN6v8BNY7exNzOHv57ZnmE9G2NWsbtwqRguOeVmEQlr21b61xiPhvVzvGnJ7byu1F0uglpNj2q1eXmOCUs389ykFSxav5tGtaryx36tOK9bI2KiKndRrGI4ACVcEZEgWDkZProCImPYfs573PJNHtPTtnFm5wb845wO1KxScW+/pGK45JSbRUR8u9Z5hfGS0fDL9xAZDb1ugD53QtWEo1qlc47Jy7fw7MQ05q/dycltknnjip6lHHhoKWpurtyHBEREJDS07AdXfw1RsSSOPId3+u7nrgGtGbtgA4Oen8bCdbuCHaGIiEjwxTeCY2+Eq76EOxZBhyHw3fPwfDevp1VudrFXaWac3KYuo248joRq0VSPjSqDwCsmFcMiIlI+klp7BXGtZkSOOJ+ba8/lw+t7k5WTx7kvTef1aasJt95KIiIiAcU3gnNeguu+8bpNf3EXvHQcLP/KuwypmBas38XO/dn0b5Nc6qFWVCqGRUSk/MTVhyu/gCa9YdR19Fz/Ll/ccgJ9Wyfz6JglXPfuHA5k5wY7ShERkdDRoAtc/jkMGwF5uTBiKLxzFmxcUKzVTFq2hQiDvq2TyijQikfFsIiIlK8q8XDJJ9D+HPj6L9QaezWvntuYh/7QlglLN3PTe3PJyskLdpQiIiKhwwzanAE3zYDTH4dNC+E/fWHKE16BXAy7Morf1bqyUjEsIiLlLyoWznsD+v8VfhqHvdiLa+Jn89jgDkxatoU7Rs4jN09dpkVERH4jKgaOvQFu/RE6nAeTH4N3BsPuDUdc9JJeTYiKjODlKSvLIdCKQcWwiIgER0QE9PkT3DANareC/13LRavu5bH+tRm7YCN3fzyffZk5wY5SREQk9FStBee+CoNf9G7H9NLx3rXEh5EcV4WhPRrzydx1bNyVUU6BhjYVwyIiElxJx8BV42DA32HVZC6ecwGvdFzO/+auo88Tk3llykr2Z6koFhER+Q0z6HoxXD8V4ht61xJ/eR/kZAZcpGPDeLJzHavS95VjoKFLxbCIiARfRCQcdwvcMB2S23HaikdY1ORfDE1cwf99uZQTn5jMa9+u0uBaIiIih6qTAtdM9O5HPPMlePccyP79md8D2bk8O3EF7RvE0btF7SAEGnpUDIuISOio0wqu+AIG/ZsaBzZzb/r9LGzyDOcmpPH3sUvo88Rk3py+WkWxiIhIQVGxMPBxr+v0z9/Bx1dB7m97Vb05fQ3rd2bw4BltiYiwIAUaWlQMi4hIaImIgB5XeoODnPEUNTM28MDW+1jQ5N+cWTONRz5fQu//m8ijY5awYvOeYEcrIiISOjpdAAOfgOVfwJjbfr0f8Y59Wbw4OY3+bZI5rlWdIAcZOqKCHYCIiEihomIh9VroeinMfYe4aU/zlz33clvTXrwRNZQXv8/h9Wmr6d60FkN7NmZQp/pUi1FaExGRMNfrOtiXDlOfgOpJcMrDvDx1JXuzcrh3YJtgRxdSdGZYRERCW3QVL7HfOg8GPkH8vp+5Y/2fWNzyRZ49Ppsd+7O45+MFpD42kQdGLWTBup04p9syiYhIGOv3AHS/EqY9Q+abZ7Hqu1Gc27k+revWDHZkIcXC7QtDjx493OzZs4MdhoiIHK3sDJj9Jkx7Gval41qfxuJjbuXNlTUZu3ADB7Lz6NQonltPTqF/22TMyva6KDOb45zrUaYbqeSUm0VEykBeLnz3HHumDKdm9lZ212hB3Em3QudhEF012NGVqaLmZhXDIiJSMWXuhVmvwPRn4cAuaHc2e467l0/XVuPVb1fzy/b9dGgYx239W3NKGRbFKoZLTrlZRKTsfLd8A+NHvsyQ7M/oELGGnCqJRKVeDT2vhZp1gx1emVAxHIASrohIJZOxE74fDjNeguz90GkY2X3u4dM1UQyfnMbP2/bTvkEct59SNkWxiuGSU24WESlbB7JzeWv6amZ9M5oLc8fQP3IuFhmNdRgCx94A9Tp59y2uJFQMB6CEKyJSSe3bCtOegVmvAg5OvJuc3rcyakH6r0XxKW2Tee3ynqW6WRXDJafcLCJSPnbtz+bFKWlMmv49l9uXDImcQhUyyYyszr64VuQltaNqo45Ua9wRq9sBqiUGO+SjomI4ACVcEZFKbvcGGP8QLPoEktrAmc+S3TCV4/45iZZJ1fngut6lujkVwyWn3CwiUr427Mzg9Wmr+WX9elqmT6J+ZhrH2FqOsbXUsr2/zrcrMpHtNVqRldiGqPodiGvWmdpNOxIRWz2I0R9ZUXOz7kEhIiKVS1wDGPIGdBoGY/8Eb5zG7jaXcGDPSQw+tXWwoxMREQm6BglV+fOgdkA74FQyc3JZuz2DuVv3smXjL2RvWEjM9mUk7Emj4Y7VpOz8kSqrs+E7yHPGusj6bK7Skn3xrXHJ7ajWuCN1m7ajQWJ1oiIrzg2LVAyLiEjl1HoANJ0Bk/9BrRkvMSF2DDVinwOaBDsyERGRkBIbFUmr5Bq0Sq4B7eoBqb++lpObx8Yd+9j881L2rV2AbVlC9V0/kZyxigb7phG50cF8OOCiWeYasS6mGTtrpGANu9Gwy6l0a5pAtZjQLDtDM6piMLPTgWeBSOA159w/gxySiIiEitgacPo/GJnRk7PnXUv0p1fi2q7DYqoFO7JKTblZRKTyiIqMoHGdmjSukwrdU3/zmsvaz7afF7JzzXxyNiyi+vZlHLt3AQk7J8NOGDD3cVZZEzo2iqdX89r0apFIj6a1qFklOkjv5rcqdDFsZpHAC8CpwDrgBzMb7ZxbEtzIREQkZOTlMdQmYJbNY1kXETHhZ+4b2KbM7z8crpSbRUTCh8VUo3ZKL2qn9Do4MTcbhvcgN6oq9588mFk/72Lmqm289u0qXp6ykgiD9g3iSW2eSK/mifRslkit6jFBib9CF8N45+/TnHOrAMzsA2AwoIQrIiLgHIy7H5v3X9yJ93Bg92DenboKQAVx2VFuFhEJZ3Pfhh1riLxoJP1a16df2/oAZGTlMveXHcxcvZ2Zq7bx7oyfeX3aagDa1KtJ32OSuH9g23INtaIXww2BtQV+Xwf0OnQmM7sOuA6gSRNdKyYiEjZys2FbGhz7R6zfA/zNn5y2ZS85eY7oSBXDZUC5WUQknO1aB836QMqA30yuGhPJ8a3qcHyrOgBk5uQyf+0uZq3exszV21mzdV+5h1rRi+HCvsX87l5Rzrn/AP8B7/YNZR2UiIiEiKgYGDYCIqPBDAP+Nrg92bmO6Ao02mUFo9wsIhLOTnkYcnPgCL2vYqMiSW2eSGrzRG4ul8B+r6IXw+uAxgV+bwRsCFIsIiISiqJ+ex2SmRETpTPCZUi5WUQk3EVWjDKzoh8W/wFIMbPmZhYDDANGBzkmERGRcKbcLCIiFULFKNkDcM7lmNnNwDi82ze84ZxbHOSwREREwpZys4iIVBQVuhgGcM59AXwR7DhERETEo9wsIiIVQUXvJi0iIiIiIiJSbCqGRUREREREJOyoGBYREREREZGwo2JYREREREREwo6KYREREREREQk7KoZFREREREQk7KgYFhERERERkbCjYlhERERERETCjophERERERERCTsqhkVERERERCTsqBgWERERERGRsKNiWERERERERMKOimEREREREREJOyqGRUREREREJOyoGBYREREREZGwo2JYREREREREwo6KYREREREREQk7KoZFREREREQk7KgYFhERERERkbBjzrlgx1CuzCwd+PkIs9UBtpZDOJWF2qt41F7Fo/YqHrVX0ZVWWzV1ziWVwnrCVhFzc1Fo/w9MbVM4tUvh1C6BqW0KF2rtUqTcHHbFcFGY2WznXI9gx1FRqL2KR+1VPGqv4lF7FZ3aqvLR3zQwtU3h1C6FU7sEprYpXEVtF3WTFhERERERkbCjYlhERERERETCjorhwv0n2AFUMGqv4lF7FY/aq3jUXkWntqp89DcNTG1TOLVL4dQugaltClch20XXDIuIiIiIiEjY0ZlhERERERERCTthWQybWWMzm2xmS81ssZnd5k9PNLOvzWyF/7OWP93M7DkzSzOzBWbWLbjvoHwdpr2eNLNlfpuMMrOEAsvc77fXcjM7LXjRl69AbVXg9bvMzJlZHf937VsB2svMbvH3n8Vm9kSB6WG5b8Fh/xe7mNkMM5tnZrPNLNWfHu77VxUzm2Vm8/32esSf3tzMZvqf9R+aWYw/Pdb/Pc1/vVkw4xcws9P9//U0M7uvkNcD/s0CfVYcaZ0VQRm1yxozW5j/OVI+76R0HW27mFlt/7N1r5kNP2SZ7n67pPmfp1Y+76Z0lVHbfOOvc57/SC6fd1N6StAup5rZ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\n",
      "text/plain": [
       "<Figure size 1152x720 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visualize the LDAPS and radiosondes data. \n",
    "plt.figure()\n",
    "\n",
    "plt.subplot(1,2,1)\n",
    "plt.plot(radsnd_T, radsnd_HGT, LDAPS_t, LDAPS_z)\n",
    "plt.xlabel('Temperature (K)')\n",
    "plt.ylabel('Altitude (m)')\n",
    "#plt.ylabel('Pressure (Pa)')\n",
    "#plt.gca().invert_yaxis()\n",
    "plt.legend(['Radiosonde Temperature', 'LDAPS Temperature'])\n",
    "\n",
    "plt.subplot(1,2,2)\n",
    "plt.plot(radsnd_WaterVMR, radsnd_HGT, LDAPS_watervmr, LDAPS_z)\n",
    "plt.xlabel('Volume mixing ratio')\n",
    "plt.ylabel('Altitude (m)')\n",
    "#plt.ylabel('Pressure (Pa)')\n",
    "#plt.gca().invert_yaxis()\n",
    "plt.legend(['Radiosonde Water VMR', 'LDAPS Water VMR'])\n",
    "\n",
    "plt.gcf().set_size_inches(16,10)\n",
    "\n",
    "# Save the figure.\n",
    "plt.savefig(TimeOfInterest.strftime('%Y_%m_%d_%H-%M-%S_') + 'T&WaterVMR' + '.png')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ARTS forward model with the radiosonde data.\n",
    "# Save the radiosonde variables as the input atmopsheric profiles. \n",
    "\n",
    "# Save pressure grid as .xml files.\n",
    "tp.arts.xml.save(radsnd_P, './ClearSky_1D_p_grid.xml')\n",
    "\n",
    "# Save z_field as GriddedField3 xml file. \n",
    "z_field_GF3 = tp.arts.griddedfield.GriddedField3()\n",
    "z_field_GF3.data = np.reshape(radsnd_HGT,(len(radsnd_P),1,1))\n",
    "z_field_GF3.grids = [radsnd_P, np.array([0]), np.array([0])]\n",
    "z_field_GF3.gridnames = ['Pressure', 'Latitude', 'Longitude']\n",
    "tp.arts.xml.save(z_field_GF3, './ClearSky_1D.z.xml')\n",
    "\n",
    "# Save t_field as GriddedField3 xml file. \n",
    "# Remove temperature values greater than 300 K, due to partition functions error in ARTS. \n",
    "# radsnd_T[radsnd_T > 300] = 300\n",
    "t_field_GF3 = tp.arts.griddedfield.GriddedField3()\n",
    "t_field_GF3.data = np.reshape(radsnd_T,(len(radsnd_P),1,1))\n",
    "t_field_GF3.grids = [radsnd_P, np.array([0]), np.array([0])]\n",
    "t_field_GF3.gridnames = ['Pressure', 'Latitude', 'Longitude']\n",
    "tp.arts.xml.save(t_field_GF3, './ClearSky_1D.t.xml')\n",
    "\n",
    "# Save Water VMR as GriddedField3 xml file. \n",
    "VMR_H2O_GF3 = tp.arts.griddedfield.GriddedField3()\n",
    "VMR_H2O_GF3.data = np.reshape(radsnd_WaterVMR,(len(radsnd_P),1,1))\n",
    "VMR_H2O_GF3.grids = [radsnd_P, np.array([0]), np.array([0])]\n",
    "VMR_H2O_GF3.gridnames = ['Pressure', 'Latitude', 'Longitude']\n",
    "tp.arts.xml.save(VMR_H2O_GF3, './ClearSky_1D.H2O.xml')\n",
    "\n",
    "# Run ARTS. \n",
    "tp.arts.run_arts(controlfile='./ClearSky_1D_ARTSvdev.arts');\n",
    "\n",
    "# ARTS forward model results. \n",
    "Tb_radsnd = tp.arts.xml.load(\"./ClearSky_1D_Tb.xml\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ARTS forward model with the LDAPS data.\n",
    "# Save the LDAPS variables as the input atmopsheric profiles. \n",
    "\n",
    "# Save pressure grid as .xml files.\n",
    "tp.arts.xml.save(LDAPS_p, './ClearSky_1D_p_grid.xml')\n",
    "\n",
    "# Save z_field as GriddedField3 xml file. \n",
    "z_field_GF3 = tp.arts.griddedfield.GriddedField3()\n",
    "z_field_GF3.data = np.reshape(LDAPS_z,(len(LDAPS_p),1,1))\n",
    "z_field_GF3.grids = [LDAPS_p, np.array([0]), np.array([0])]\n",
    "z_field_GF3.gridnames = ['Pressure', 'Latitude', 'Longitude']\n",
    "tp.arts.xml.save(z_field_GF3, './ClearSky_1D.z.xml')\n",
    "\n",
    "# Save t_field as GriddedField3 xml file. \n",
    "# Remove temperature values greater than 300 K, due to partition functions error in ARTS. \n",
    "# LDAPS_t[LDAPS_t > 300] = 300\n",
    "t_field_GF3 = tp.arts.griddedfield.GriddedField3()\n",
    "t_field_GF3.data = np.reshape(LDAPS_t,(len(LDAPS_p),1,1))\n",
    "t_field_GF3.grids = [LDAPS_p, np.array([0]), np.array([0])]\n",
    "t_field_GF3.gridnames = ['Pressure', 'Latitude', 'Longitude']\n",
    "tp.arts.xml.save(t_field_GF3, './ClearSky_1D.t.xml')\n",
    "\n",
    "# Save Water VMR as GriddedField3 xml file. \n",
    "VMR_H2O_GF3 = tp.arts.griddedfield.GriddedField3()\n",
    "VMR_H2O_GF3.data = np.reshape(LDAPS_watervmr,(len(LDAPS_p),1,1))\n",
    "VMR_H2O_GF3.grids = [LDAPS_p, np.array([0]), np.array([0])]\n",
    "VMR_H2O_GF3.gridnames = ['Pressure', 'Latitude', 'Longitude']\n",
    "tp.arts.xml.save(VMR_H2O_GF3, './ClearSky_1D.H2O.xml')\n",
    "\n",
    "# Run ARTS. \n",
    "tp.arts.run_arts(controlfile='./ClearSky_1D_ARTSvdev.arts');\n",
    "\n",
    "# ARTS forward model results. \n",
    "Tb_LDAPS = tp.arts.xml.load(\"./ClearSky_1D_Tb.xml\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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mY3/I3BZHnPLyPaC5RzP0XvvvUd5eTl77rknb731Fp/61KeATK/ad7/7XcrKyjjuuON4/fXXWblyJU8++SSnnXYaO++8M7vuumtTodjciy++yD777MMBBxwAwOTJk5kzZw677rorffr04atf/SoPPvggO++8MwDz5s3jC1/4AgBf+tKXmDt3btO1PvvZz1JUVMQhhxzCyux/yXPmzOHzn/88vXr14qMf/SjHHnts032fe+45PvWpT1FeXs4111xDdXX1ZvkdeeSRnHvuudx22200NDS0+r18+tOfpri4mNLSUhoaGjjhhBMAKC0tZdmyZTl/v9XV1UycOJHS0lJuuOEGlixZ0m77tWvXsmbNGsaPH9/iu9vo9NNPB+Cwww5ryuOII47g2muv5brrrmP58uX07ds35/wkSZK6ip40VFHbJ5+zei5KKY1MKZWllIanlK7Oxl9NKY1JKf1dSumMlNIH2fj72eO/y77/ar5yazJ06NbFc7TxGb/ly5dTX1/f9IxfZWUlq1atYsGCBVRVVTFo0KCm9d62NIFpW52fvXv35o9//COf+9zn+K//+q+mYmpTza+/0047tXrd1nJIKXHooYdSVVVFVVUVixcv5rHHHtus3fTp07nmmmt47bXXKC8vZ/Xq1Zu12XjfoqIiiouLm+5XVFTEhg0bmj5PY2MjQJtr4X3rW9/iggsuYPHixfz4xz/e7jXzNubVq1evpjy+8IUv8PDDD9O3b18mTpzI73//++26hyRJUiF0tV68QdRuVRxgUFHr83q0FVfrOuUZvy5r2jTI9pA12XnnTLwD7Lbbbtxyyy3ceOONrF+/nrVr1/KRj3yE4uJiZs+ezfLlywEYN24cs2bN4r333mPdunU88sgjm13roIMOYtmyZbz88ssA3HXXXYwfP5533nmHtWvXcuKJJ3LzzTdTVVUFwCc+8QnuvfdeIFNwfvKTn2w313HjxnHvvffS0NBATU0Ns2fPBuDAAw9k1apVzJs3D8gM/Wyth+2VV15h7NixXH311QwYMIDXXnttm76zkpISFixYAMADDzzQapu1a9ey996ZyWDvuOOOpnj//v1Zt27dZu132203dt9996bn9zZ+d+159dVX2Xfffbnwwgs59dRTWbRo0TZ9HkmSJP1N7bDD+fpJwYf+JeDKzP4bJwa1ww5v85yFP5tJn6mZ9hu3vlODqp/9ot17Ddz13a2K93Q7duE3aRL85CcwbFimz3vYsMzxpEkddouRI0cyYsQI7r33XiZNmsT8+fMZPXo0lZWVHHTQQQCMGjWKs846i/Lycj73uc9t9nwaZJ4PnDFjBmeccQalpaUUFRVx/vnns27dOk4++WTKysoYP348N910EwC33HILM2bMoKysjLvuuosf/OAH7eZ52mmnsf/++1NaWsrXv/71psLoQx/6EPfffz+XXnopI0aMoLy8vMXMmRtdcskllJaWMnz4cMaNG8eIESO26fv63ve+x0UXXcRRRx1Fr169Wm1z5ZVXcsYZZ3DUUUcxYMCApvgpp5zCrFmzmiZ3ae6OO+7gkksuoaysjKqqKq644op285g5cybDhw+nvLycF154wdlBJUmSmhnY682tim9Uc9V3mFEO9dmpHut7w4yRUHvVJW2eUzFgKY3FvVvEGop7UzFgabv3+vufX8KHrixuUTB+6Mpizvj5P7d7XnN9P7y2jfjbOV+jq4h8z5+ST6NHj07z589vEXv++ec5+ODWVo2QOo9/h5IkqSto72miTcuAwb1WsbJx81nzBxWtorahZfwbNx3HT996vKmAA/jQBvjqHhO49eK216D+xqPf4Kfzb6OeDX87j958dfQUbj2p9SXQRv54JFW1VZvFyweXs/C8hW3ea1vP6+hr5FtELEgpjd5Su95baiBJkiSp6xg8uPVn9AYNgtq2H5XbooU/m8m+L32L94v/Fuu7Hqr2/3/AN1u0nbfzauo36fSq7w1P7bz5PA8tzque16LoA6hnA09Vbz6irCmvbSywOqIw6yrFXUew8JMkSZK6ka2ZsGVQUdu9eNAyXjFgKY3LekOzwmzjkMpN++IKWYxp21j4SZIkST3UVvXibUNvnLoPCz9JkiSph+qMXjx1Dzv2rJ6SJElSD2Yvnjayx0+SJEnqoezF00b2+OVBv379NotdeeWV7L333pSXl7P//vtz+umns3Rpy7VHVq1aRXFxMT/+8Y9bxEtKSigtLWXEiBEcf/zx1Gana/rZz35GaWkpZWVlDB8+nIceeiin/ObPn8+FF164jZ+updtvv50LLrig3TZ/+MMfWqz9N336dO68884Oub8kSdKOJjMxS+5xCSz8AKhZV8P428dT+852zH+bg4svvpiqqipeeuklzjrrLI499lhWrfrbf6D33Xcfhx9+OPfcc89m586ePZtnn32W0aNHc+2111JdXc20adOYO3cuixYt4umnn6asrCynPEaPHs0tt9zSYZ9rSzYt/M4//3wXRJckSdpGtWkQb/QL+kzNLEred2pQ0y+oTYMKnZq6MAs/oGJOBXNXzKXiiYpOu+dZZ53F8ccfz89//vOm2D333MN//Md/UF1dzeuvv97qeePGjePll1/mzTffpH///k29i/369WOfffbZrP19993H8OHDGTFiBOPGjQMyhdjJJ58MZHoiJ0+ezPHHH09JSQkPPvgg//zP/0xpaSknnHAC69evBzK9jn/5y1+ATI/h0Ucfvdm9HnnkEcaOHcvIkSM57rjjWLlyJcuWLWP69OncdNNNlJeX8+STT3LllVdy4403AlBVVcXhhx9OWVkZp512Gn/9618BOProo7n00ksZM2YMBxxwAE8++SQAS5YsYcyYMZSXl1NWVsZLL7201d+9JElStzZ0KBXjoTG7OHtDQMW4TFxqyw5f+NWsq2FG1QwaUyMzqmbkvdevuVGjRvHCCy8A8Nprr1FbW8uYMWM488wzmTlzZqvn/PKXv2wa9jlo0CD22Wcf/uEf/oFHHnmk1fZXX301v/nNb3j22Wd5+OGHW23zyiuv8Oijj/LQQw/xxS9+kWOOOYbFixfTt29fHn300Zw/zyc/+UmefvppFi5cyNlnn831119PSUkJ559/flNv51FHHdXinHPOOYfrrruORYsWUVpaylVXXdX03oYNG/jjH//IzTff3BSfPn06F110EVVVVcyfP58hQ4bknJ8kSVJPUHPVd5hRnlkwHTL7GSOh9qpLCpuYurQdvvCrmFNBY2oEoCE1dGqvX0qp6fW9997LmWeeCcDZZ5+92XDPY445hvLyct5++20uv/xyevXqxa9//Wvuv/9+DjjgAC6++GKuvPLKze5x5JFHcu6553LbbbfR0NDQah6f/vSnKS4uprS0lIaGBk444QQASktLWbZsWc6fp7q6mokTJ1JaWsoNN9zAkiVL2m2/du1a1qxZw/jx4wGYPHkyc+bMaXr/9NNPB+Cwww5ryuOII47g2muv5brrrmP58uX07ds35/wkSZJ6gooBS2ksbjlH48YlGqS27NCF38bevvqGegDqG+o7tddv4cKFHHzwwUBmmOftt99OSUkJp556Ks8++2yLYYyzZ8+mqqqKO++8kw9/+MMARARjxozh8ssv59577+WBBx7Y7B7Tp0/nmmuu4bXXXqO8vJzVq1dv1mannXYCoKioiOLiYiKi6XjDhsz0v71796axMVMgv//++61+nm9961tccMEFLF68mB//+MdttsvVxrx69erVlMcXvvAFHn74Yfr27cvEiRP5/e9/v133kCRJKrTBgyFi823w4Nbbu0SDtsUOXfg17+3bqLN6/R544AEee+wxPv/5z/Piiy/y7rvv8vrrr7Ns2TKWLVvWVMy15Y033uCZZ55pOq6qqmLYsGGbtXvllVcYO3YsV199NQMGDOC1117bpnxLSkpYsGBBU+6tWbt2LXvvvTcAd9xxR1O8f//+rFu3brP2u+22G7vvvnvT83t33XVXU+9fW1599VX23XdfLrzwQk499VQWLVq0TZ9HkiSpq1i5cuviC89bSPpe2mxz6Qa1Z4cu/OZVz2vq7duovqF+u/9vSV1dHUOGDGnavv/97wM0TXCy//77c/fdd/P73/+egQMHcs8993Daaae1uMbnPve5Vmf33Gj9+vV85zvf4aCDDqK8vJyZM2fygx/8YLN2l1xyCaWlpQwfPpxx48YxYsSIbfpM3/ve97jooos46qij6NWrV6ttrrzySs444wyOOuooBgwY0BQ/5ZRTmDVrVtPkLs3dcccdXHLJJZSVlVFVVcUVV1zRbh4zZ85k+PDhlJeX88ILLzg7qCRJkpSDaP6cWXczevToNH/+/Bax559/vmn4pFQo/h1KkqRcZZ+yaVU3/qe6OklELEgpjd5Sux26x0+SJEmSdgQWfpIkSZLUw1n4SZIkSQU0aNDWxaVt0XvLTSRJkiTlS23nrCSmHZw9fpIkSZLUw1n4SZIkSVIPZ+GXB7169aK8vJzhw4dzyimnsGbNmq06/8orr+TGG28E4IorruB3v/tdPtIEYM2aNfzwhz/c7uuUlJTwl7/8ZbP4ueeey/3337/d199eN998M3V1dU3HJ5544lb/LpIkSXlTWQklJVBUlNlXVhY6I/UwO3ThN3hwZt2UTbfBg7fvun379qWqqornnnuOPfbYg1tvvXWbr3X11Vdz3HHHbV9C7diWwi+lRGNjY54y2rKGhoatPmfTwu9Xv/oVH/7whzsyLUmSpG1TWQlTplCzejnjJydqVy+HKVMs/tShdujCb+XKrYtviyOOOILXX38dgHfeeYcJEyYwatQoSktLeeihh5raTZs2jQMPPJDjjjuOF198sSnevMfs8ccfZ+TIkZSWlvLlL3+ZDz74AMj0tn33u9/liCOOYPTo0TzzzDNMnDiR/fbbj+nTpzdd64YbbuDjH/84ZWVlfO973wPgsssu45VXXqG8vJxLLrmkzXbLli3j4IMP5hvf+AajRo3itdde2+yz3nDDDYwZM4YxY8bw8ssvN8V/97vfcdRRR3HAAQfwy1/+EoAlS5YwZswYysvLKSsr46WXXgLg7rvvboqfd955TUVev379uOKKKxg7dizXXnstZ555ZtP1//CHP3DKKacA8PWvf53Ro0dz6KGHNuV+yy238MYbb3DMMcdwzDHHNH1nG3sov//97zN8+HCGDx/OzTff3OLzfu1rX+PQQw/l+OOP57333mu63iGHHEJZWRlnn312Dn8FkiRJ7Zg6FerqqBgPc4dCxTigri4TlzpKSqnbbocddlja1NKlSzeLtQXa3rbHLrvsklJKacOGDenv//7v03//93+nlFJav359Wrt2bUoppVWrVqX99tsvNTY2pvnz56fhw4end999N61duzbtt99+6YYbbkgppTR58uR03333pffeey8NGTIkvfjiiymllL70pS+lm266KaWU0rBhw9IPf/jDlFJK3/72t1NpaWl6++2305tvvpkGDhyYUkrpN7/5Tfra176WGhsbU0NDQzrppJPSE088kf785z+nQw89tCn39tpFRJo3b16rn3nYsGHpmmuuSSmldMcdd6STTjqpKf+JEyemhoaG9H//939p7733Tu+991664IIL0t13351SSumDDz5IdXV1aenSpenkk09O9fX1KaWUvv71r6c77rgj+1uRZs6c2fQ9fuxjH0vvvPNOSiml888/P911110ppZRWr17d9N2PHz8+Pfvss035rVq1qkW+q1atavru33nnnbRu3bp0yCGHpGeeeSb9+c9/Tr169UoLFy5MKaV0xhlnNN1jr732Su+//35KKaW//vWvrX4fW/N3KEmSdnAR6Y1+pD5TSVxJ6juVVNOPlCIKnZm6AWB+yqF22qF7/PLlvffeo7y8nD333JO33nqLT33qU0CmyP7ud79LWVkZxx13HK+//jorV67kySef5LTTTmPnnXdm11135dRTT93smi+++CL77LMPBxxwAACTJ09mzpw5Te9vPKe0tJSxY8fSv39/Bg4cSJ8+fVizZg2PPfYYjz32GCNHjmTUqFG88MILTb1szbXXbtiwYRx++OFtfu7Pf/7zTft58+Y1xc8880yKiorYf//92XfffXnhhRc44ogjuPbaa7nuuutYvnw5ffv25fHHH2fBggV8/OMfp7y8nMcff5xXX30VyDw3+bnPfQ6A3r17c8IJJ/DII4+wYcMGHn30UT7zmc8A8Itf/IJRo0YxcuRIlixZwtKlS9v9rebOnctpp53GLrvsQr9+/Tj99NN58sknAdhnn30oLy8H4LDDDmPZsmUAlJWVMWnSJO6++25693ZFFEmStJ2GDqViPDRG5rAhsr1+Q4cWNC31LP6rNQ82PuO3du1aTj75ZG699VYuvPBCKisrWbVqFQsWLKC4uJiSkhLef/99ACKi3Wtmivm27bTTTgAUFRU1vd54vGHDBlJKXH755Zx33nktzttYzDS/T1vtdtlll3ZzaP4Z2nq98fgLX/gCY8eO5dFHH2XixIn853/+JyklJk+ezL/9279tdu0+ffrQq1evpuOzzjqLW2+9lT322IOPf/zj9O/fnz//+c/ceOON/OlPf2L33Xfn3HPPbfp+29Le99r8e+zVq1fTUM9HH32UOXPm8PDDD1NRUcGSJUssACVJ0jarueo7zHjpW9Rn/zlR3xtmjIR/PfMStnPqCamJPX55tNtuu3HLLbdw4403sn79etauXctHPvIRiouLmT17NsuXLwdg3LhxzJo1i/fee49169bxyCOPbHatgw46iGXLljU9O3fXXXcxfvz4nHOZOHEiP/vZz3jnnXcAeP3113nzzTfp378/69at22K7XMycObNpf8QRRzTF77vvPhobG3nllVd49dVXOfDAA3n11VfZd999ufDCCzn11FNZtGgREyZM4P7772+631tvvdX0HW3q6KOP5plnnuG2227jrLPOAuDtt99ml112YbfddmPlypX893//d1P7TT/nRuPGjeO//uu/qKur491332XWrFkcddRRbX7GxsZGXnvtNY455hiuv/561qxZ0/RdSZIkbYuKAUtpLG75P5EbintTMaD9kUvS1tihuykGDWp9IpdBgzruHiNHjmTEiBHce++9TJo0iVNOOYXRo0dTXl7OQQcdBMCoUaM466yzKC8vZ9iwYa0WHn369GHGjBmcccYZbNiwgY9//OOcf/75Oedx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\n",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Compare brightness temperatures between the radiometer observations, the LDAPS simulations, and the radiosonde simulations. \n",
    "plt.plot(radmtr_channels, Tb_radsnd, 'ro', \n",
    "         radmtr_channels, Tb_LDAPS, 'g^', \n",
    "         radmtr_channels, BosungObs_radmtr, 'bs')\n",
    "plt.xlabel('Frequency (Hz)')\n",
    "plt.ylabel('Brightness Temperature (K)')\n",
    "plt.legend(['Radiosonde simulations', 'LDAPS simulations', 'Radiometer bbservations'])\n",
    "plt.gcf().set_size_inches(15,5)\n",
    "\n",
    "# Save the figure.\n",
    "plt.savefig(TimeOfInterest.strftime('%Y_%m_%d_%H-%M-%S_') + 'ForwardModels_v_Observations' + '.png')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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efP00EMP6bDDDmu2b8aMGcrPz9eePXskSdu2bVNFRUWLc0yaNEmvvvqqdu7cqdraWj388MM666yzuhXXsGHDNHr0aD322GOS6pLMd955p1vnBAAA6O3SAgGty85WWpN/bAcGInoAo2TJkiWqqKjQd7/73WbteXl5uvTSS/Xuu+9q8uTJkqRDDz1Uy5cvV/xB/xqVnJyshQsX6uyzz5a7a+bMmZo9e3a3Y/v973+v7373u7r99tu1f/9+XXbZZTrttNO6fV4AAAAAvZt1d5hhb5CVleUN6+A1ePfdd3XqqafGKCIA6Bn8rgMAAJJkZmvdPauj4xgCCgAAAAADBAkgAAAAAAwQJIAAAAAAMECQAAIAAADAAEECCAAAAAADBAlgE8GSoArGFShYEozI+eLj45WZmalx48Zp1qxZ2rVrV6fef+utt2rx4sWSpJtvvlkvvPBCROLqjLlz5+rxxx/v1jkKCwu1YMGCiMSzbNkyXXfdde0e88orr+jNN99sfL1kyRI99NBDEbl+V0XyM4iknnzGdu3apfvuuy9q5x81apR27twZtfOH4+B7/OijjzRnzpwYRgQAANAcCWC92mCtimcWq2p9lYpzilUbrO32OQ855BAVFRVp3bp1Ouqoo/SrX/2qy+e67bbbNG3atG7HFAtZWVm69957e+x6ByeAV199tb75zW/22PVb05OfwYEDB7r0vmg/Y11JAN1doVAoShG1r7a2878DDr7HY489ttv/gAIAABBJJID1SnNLVVNRI7lUs71GpfNLI3r+yZMna9u2bZKkPXv2aOrUqZowYYLS09P1zDPPNB53xx136JRTTtG0adO0YcOGxvamPXEvvviixo8fr/T0dOXm5mrfvn2SpB/96EcaO3asMjIy9IMf/ECStHnzZk2dOlUZGRmaOnWq/vGPfzSeb8GCBfrCF76glJSUxnO7u6677jqNHTtWOTk5qqioaIxh7dq1OuusszRx4kTNmDFD5eXlLe7zscce07hx43TaaafpzDPPlFSXkJ1//vmS6nqcrrzySk2fPl2jRo3Sk08+qRtuuEHp6ek699xztX//fknNe3MKCws1ZcqUFtdauXKlJk2apPHjx2vatGnavn27Nm3apCVLlujuu+9WZmamXn/99Wa9XEVFRTr99NOVkZGhCy+8UJ988okkacqUKbrxxhuVnZ2tMWPG6PXXX5cklZSUKDs7W5mZmcrIyNDGjRtbxHHooYfqxhtv1MSJEzVt2jQVFBRoypQpSklJ0bPPPtvqZ5Cbm9t4TENiuGnTJo0bN67xvIsXL9att94qSbr33nsbf7aXXXZZixiWLVumSy65RLNmzdL06dOj9oyNGjVKN910kyZPnqysrCy9/fbbmjFjhk466SQtWbKk8VyLFi3S5z//eWVkZOiWW26RVPd8fvDBB8rMzNQPf/jDNo/btGmTTj31VF1zzTWaMGGCtmzZ0uxe24qt4XzZ2dnKzs7W+++/L6n1Z7K2tlY//OEPG699//33N/6czj77bF1++eVKT0/XjTfe2Cyhu/XWW/WLX/yizc/34Hts+jOtrq7WvHnzlJ6ervHjx+vll19u/NlddNFFOvfcc3XyySfrhhtuaIxx7ty5GjdunNLT03X33Xe3+LkDAAB0mrv3+W3ixIl+sPXr17doa8tHSz/yVwOv+st6uXF7NfFV/2jpR2GfozWBQMDd3Q8cOOBz5szx5557zt3d9+/f77t373Z39x07dvhJJ53koVDICwsLfdy4cR4MBn337t1+0kkn+aJFi9zd/corr/THHnvM9+7d68cdd5xv2LDB3d2vuOIKv/vuu72ystLHjBnjoVDI3d0/+eQTd3c///zzfdmyZe7uvnTpUp89e3bj+ebMmeO1tbVeUlLiJ510kru7P/HEEz5t2jQ/cOCAb9u2zQ8//HB/7LHHvKamxidPnuwVFRXu7r5ixQqfN29ei3seN26cb926tVkML7/8sufk5Li7+y233OJnnHGG19TUeFFRkR9yyCG+evVqd3f/yle+4k899ZS7u5944om+Y8cOd3f/61//6meddZa7uz/44IN+7bXXurv7xx9/3Hi/DzzwgH//+99vvEbD53bw6/T0dH/llVfc3f3HP/6x/+u//qu7u5911lmN71+1apVPnTrV3d2vu+46X758ubu779u3z6uqqlrcs6Rm93DOOec03t9pp53W6mcwefJkr66u9h07dvhRRx3lNTU1/uGHH3paWlrjeRctWuS33HKLu7snJyd7dXV1s8+1qQcffNBHjhzplZWV7h6dZ6zh53Lfffe5u/v111/v6enp/umnn3pFRYUPHz7c3d2ff/55//a3v+2hUMhra2s9JyfHX3311Rb3195xZuZvvfVWi/vsKLbbb7/d3d1/+9vfNn7erT2T999/v//0pz91d/fq6mqfOHGil5WV+csvv+yJiYleVlbm7u5vv/22n3nmmY3XP/XUU33z5s1tfr4H32PT14sXL/a5c+e6u/u7777rxx9/vO/du9cffPBBHz16tO/atcv37t3rJ5xwgv/jH//wwsJCnzZtWuO5Wvu5u3fudx0AAOi/JBV6GLkTPYCSyvLKFAo2H2YWqgqpLK+sW+fdu3evMjMzlZSUpI8//ljnnHOOpLqk+6abblJGRoamTZumbdu2afv27Xr99dd14YUXKjExUcOGDdMFF1zQ4pwbNmzQ6NGjNWbMGEnSlVdeqddee03Dhg3T0KFD9a1vfUtPPvmkEhMTJUlvvfWWLr/8cknSFVdcoTfeeKPxXF/5ylcUFxensWPHavv27ZKk1157TV/72tcUHx+vY489Vl/+8pcbr7tu3Tqdc845yszM1O23366tW7e2iO+MM87Q3Llz9cADD7Q5hO68887T4MGDlZ6ertraWp177rmSpPT0dG3atCnsz3fr1q2aMWOG0tPTtWjRIpWUlLR7/O7du7Vr1y6dddZZzT67BhdddJEkaeLEiY1xTJ48WXfeead+9rOfafPmzTrkkENanDchIaHZPZx11lmN99fW/eTk5GjIkCE6+uijdcwxxzR+/m3JyMjQ17/+dS1fvlyDBg1q9ZhzzjlHRx11lKToPGMNGt6Tnp6uSZMm6bDDDtPw4cM1dOhQ7dq1S2vWrNGaNWs0fvx4TZgwQaWlpa32nLZ33IknnqjTTz+907F97Wtfa/z61ltvSWr9mVyzZo0eeughZWZmatKkSaqsrGy8dnZ2tkaPHi1JGj9+vCoqKvTRRx/pnXfe0ZFHHqkTTjihzc+3PW+88YauuOIKSVJqaqpOPPFEvffee5KkqVOn6vDDD9fQoUM1duxYbd68WSkpKSorK9P3vvc9/fGPf9SwYcPaPT8AAEA4SAAlpSxMUVyg+UcRlxinlLtSunXehjmAmzdvVk1NTeMcwN///vfasWOH1q5dq6KiIo0YMULV1dWSJDNr95x1yX1LgwYNUkFBgS6++GI9/fTTjQnJwZqef8iQIa2et7UY3F1paWkqKipSUVGR/v73v2vNmjUtjluyZIluv/12bdmyRZmZmaqsrGxxTMN14+LiNHjw4MbrxcXFNc5fGzRoUOPcr4bP5mDf+973dN111+nvf/+77r///jaPC1dDXPHx8Y1xXH755Xr22Wd1yCGHaMaMGXrppZdavO/ge2h6f23Nx2v62Tdcr+k9S83ve9WqVbr22mu1du1aTZw4sdXzBgKBxu+j8YwdHHvTe214feDAAbm78vLyGp+V999/X/Pnz2/1Om0d1/ReOhNb03tr+L61Z9Ld9ctf/rLx2h9++KGmT5/e6rXnzJmjxx9/XI888kjj8Nv2Pt+2tBd7a8/DkUceqXfeeUdTpkzRr371K33rW99q9/wAAADhIAGUlJybrKScJNnQuj8YbagpaVaSkuclR+T8hx9+uO69914tXrxY+/fv1+7du3XMMcdo8ODBevnll7V582ZJ0plnnqmnnnpKe/fu1WeffaaVK1e2OFdqaqo2bdrUOL/pd7/7nc466yzt2bNHu3fv1syZM3XPPfeoqKhIkvSFL3xBK1askFT3R+sXv/jFdmM988wztWLFCtXW1qq8vLxxntIpp5yiHTt2NPaq7N+/v9Uetw8++ECTJk3SbbfdpqOPPrrF/K1wjRo1SmvXrpUkPfHEE60es3v3bo0cOVKS9Nvf/rax/bDDDtNnn33W4vjDDz9cRx55ZOP8vobPrj1lZWVKSUnRggULdMEFF6i4uLhL9xOOESNGqKKiQpWVldq3b5/+8Ic/SJJCoZC2bNmis88+Wz//+c+1a9cu7dmzp91zReMZC9eMGTOUn5/fGOO2bdtUUVHR4ufS1nHt6Si2Rx55pPHr5MmTJbX+TM6YMUO//vWvG+ecvvfeewoGW6/+e9lll2nFihV6/PHHGyt6tvX5tvXsSXWf/e9///vG6/3jH//QKaec0ua97ty5U6FQSBdffLF++tOf6u233273swEAAAhH62PJBqDU/FQVjC3Qvi37lDAiQalLUyN6/vHjx+u0007TihUr9PWvf12zZs1SVlaWMjMzlZpad60JEybo0ksvVWZmpk488UR96UtfanGeoUOH6sEHH9Qll1yiAwcO6POf/7yuvvpqffzxx5o9e7aqq6vl7o0FI+69917l5uZq0aJFGj58uB588MF247zwwgv10ksvKT09XWPGjGn84zohIUGPP/64FixYoN27d+vAgQO6/vrrlZaW1uz9P/zhD7Vx40a5u6ZOnarTTjtNr776aqc/r1tuuUXz58/XnXfeqUmTJrV6zK233qpLLrlEI0eO1Omnn64PP/xQkjRr1izNmTNHzzzzjH75y182e89vf/tbXX311aqqqlJKSkqHn8cjjzyi5cuXa/Dgwfrc5z6nm2++udP3Eq7Bgwfr5ptv1qRJkzR69OjG56K2tlbf+MY3tHv3brm7/u3f/k1HHHFEu+eKxjMWrunTp+vdd99tTMAOPfRQLV++XCeddJLOOOMMjRs3Tuedd54WLVrU6nHx8fFtnruj2Pbt26dJkyYpFArp4YcfltT6M5mRkaFNmzZpwoQJcncNHz5cTz/9dKvXTEtL02effaaRI0cqOTm53c83KSmp2T1ee+21jee55pprdPXVVys9PV2DBg3SsmXLmvX8HWzbtm2aN29eY6/wwoULO/zsAQAAOmIdDanqC7KysrywsLBZ27vvvqtTTz21U+cJlgRVcmmJ0h5JUyCt9SFoANCbdOV3HQAA6H/MbK27Z3V0HD2ATQTSAspelx3rMAAAAAAgKpgDCAAAgD4pWBJUwbgCBUtan8cNoCUSQAAAAPQ5tcFaFc8sVtX6KhXnFKs22PryUwCaIwEEAABAn1OaW6qaihrJpZrtNSqdXxrrkIA+gQQQAAAAfUp5frkqV1XKq+uKGXq1q3Jlpcrzy2McGdD7kQACAACgTynLK1MoGGrWFqoKqSyvLEYRAX0HCWATJcGgxhUUqKSNBaEBAAAQeykLUxQXaP5nbFxinFLuSolRREDfQQJYL1hbq5nFxVpfVaWc4mIFa7s/kfjQQw9t0Xbrrbdq5MiRyszM1Mknn6yLLrpI69evb3bMjh07NHjwYN1///3N2keNGqX09HSddtppmj59uv73f/9XkpSfn6/09HRlZGRo3LhxeuaZZ8KKr7CwUAsWLOji3TW3bNkyXXfdde0e88orr+jNN99sfL1kyRI99NBDEbl+V0XyM4ikW2+9VYsXL5Yk3XzzzXrhhReidq1du3bpvvvui9r5R40apZ07d0bt/OE4+B4/+ugjzZkzJ4YRAQC6Izk3WUk5SbKhJkmyoaakWUlKnpcc48iA3o8EsF5uaakqamrkkrbX1Gh+afQmEv/bv/2bioqKtHHjRl166aX68pe/rB07djTuf+yxx3T66afr4YcfbvHel19+We+8846ysrJ05513auvWrbrjjjv0xhtvqLi4WH/5y1+UkZERVhxZWVm69957I3ZfHTk4Abz66qv1zW9+s8eu35qe/AwOHDjQpffddtttmjZtWoSj+aeuJIDurlAo1PGBUVDbhX+cOfgejz32WD3++OORDAsA0MNS81O1dfwgzcuXto4fpNSlqbEOCegTSAAl5ZeXa1Vlpaq9biJxtbtWVlYqvzz6E4kvvfRSTZ8+Xf/zP//T2Pbwww/rF7/4hbZu3apt27a1+r4zzzxT77//vioqKnTYYYc19jYeeuihGj16dIvjH3vsMY0bN06nnXaazjzzTEl1Cdn5558vqa7H6corr9T06dM1atQoPfnkk7rhhhuUnp6uc889V/v375fUvDensLBQU6ZMaXGtlStXatKkSRo/frymTZum7du3a9OmTVqyZInuvvtuZWZm6vXXX2/Wy1VUVKTTTz9dGRkZuvDCC/XJJ59IkqZMmaIbb7xR2dnZGjNmjF5//XVJUklJibKzs5WZmamMjAxt3LixRRyHHnqobrzxRk2cOFHTpk1TQUGBpkyZopSUFD377LOtfga5ubmNxzQkhps2bdK4ceMaz7t48WLdeuutkqR7771XY8eOVUZGhi677LIWMSxbtkyXXHKJZs2apenTp2vPnj2aOnWqJkyYoPT09Ga9tXfccYdOOeUUTZs2TRs2bGhsnzt3bmOy8uKLL2r8/2/v3uOjrO59j39+QCCAoMhNGjBcFEFICLdEDkeRgqCbQqui4K2AtPtQEW+70EKPYtWttujmnO22arUgFlpQ3Fi8bMS+CF563EXAoCKiCAECPYJBAkISJPntP+Zh9mQykwwWMoH5vl+veWXmedY8z289a1Zm1qzL9O1LVlYWN998M+Xl5eFymTVrFoMGDWLAgAGsX7+ekSNH0q1bN5588snwsebMmcPAgQPJzs5m9uzZAPz85z/n888/Jycnh+nTp8dNV1hYSM+ePbnlllvo168fO3furJLXeLEdO15ubi65ubls2bIFiP2arKioYPr06eFzH+sFX716NUOHDuX6668nKyuLn/3sZ1UadPfeey+PPvpo3OsbncfIMi0rK2PSpElkZWXRt29f8vPzw2V31VVXcfnll3P++eczY8aMcIwTJ06kd+/eZGVlMXfu3GrlLiIiJ19ZOsx82NieCbMeNsrSkx2RyCnC3U/5W//+/T3axx9/XG1bPO3eecfJz692a/fOOwkfI5bmzZtX2zZ79myfM2dOlW1z5871KVOmuLv7jh07/LzzznN395kzZ/qjjz4aTpeZmel79+51d/epU6f6jBkz/OjRoz5ixAjv1KmTT5w40ZcvXx4zlt69e3tRUZG7u3/11Vfu7p6fn++jRo0KxzV48GA/cuSIFxQUeNOmTf21115zd/cf/OAHvmzZsmoxvPfeez5kyBB3d58/f75PnTrV3d337dvnlZWV7u7+9NNP+1133RUz75GPs7KyfPXq1e7ufvfdd/vtt9/u7u5DhgwJP//VV1/1YcOGubv7rbfe6gsXLnR39/Lycj98+HC1PANV8nDZZZeF89enT5+Y12DQoEFeVlbme/fu9bPPPtuPHDni27Zt8169eoWPO2fOHJ89e7a7u3fo0MHLysqqXNdI8+fP94yMDC8uLnZ392+++cZLSkrc3X3v3r3erVs3r6ys9LVr13rv3r390KFDXlJS4t26dQtfmwkTJvgLL7zgpaWl3rFjR9+8ebO7u9900/0rFjMAABPLSURBVE0+d+7ccLn85je/cXf3O+64w7OysvzAgQO+Z88eb9u2rbu7v/766/7jH//YKysrvaKiwkeNGuVvvvlmtfzVlM7M/N13362Wz9pie+CBB9zdfcGCBeHrHes1+dRTT/n999/v7u5lZWXev39/37p1q+fn53uzZs1869at7u6+fv16v+SSS8Ln79mzp2/fvj3u9Y3OY+TjRx55xCdOnOju7ps2bfJOnTp5aWmpz58/37t06eL79+/30tJSP/fcc33Hjh2+du1aHz58ePhYscq9Lh3P/zoRkdPJtR995OmrVzv5+Z6+erWP++ijZIckklTAWk+g7aQeQOChrl1p3qDqpWjWoAEPd62bicQe9DwCLF68mGuvvRaA8ePHVxsGOnToUHJycjhw4AAzZ86kYcOGrFixgqVLl9K9e3fuvPPOcO9UpMGDBzNx4kSefvrpuEPorrjiCtLS0sjKyqKiooLLL78cgKysLAoLCxPOT1FRESNHjiQrK4s5c+awcePGGtOXlJSwf/9+hgwZAsCECRN46623wvuvuuoqAPr37x+OY9CgQTz44IP86le/Yvv27TRt2rTacRs3blwlD0OGDAnnL15+Ro0aRZMmTWjTpg3t2rXjiy++qDH27OxsbrjhBhYuXEijRo1iprnssss4++yzgVBZz5o1i+zsbIYPH86uXbv44osvePvtt7nyyitp1qwZLVu2ZMyYMdWOs3nzZrp06UL3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\n",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot the difference between the two. \n",
    "plt.plot(radmtr_channels, np.zeros(radmtr_channels.shape),'k--', \n",
    "        radmtr_channels, Tb_radsnd - BosungObs_radmtr, 'md', \n",
    "        radmtr_channels, Tb_LDAPS - BosungObs_radmtr, 'cd')\n",
    "plt.xlabel('Frequency (Hz)')\n",
    "plt.ylabel('Brightness Temperature (K)')\n",
    "plt.legend(['Zero line', 'Radiosonde simulations minus radiometer observations', 'LDAPS simulations minus radiometer observations'])\n",
    "plt.gcf().set_size_inches(15,5)\n",
    "\n",
    "# Save the figure.\n",
    "plt.savefig(TimeOfInterest.strftime('%Y_%m_%d_%H-%M-%S_') + 'ForwardModels_v_Observations_Diff' + '.png')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
