# Collection of additional functions to be used. 
# By B Lee



import os, re, inspect
import numpy as np
import pandas as pd
import numpy.matlib as npm
import scipy.interpolate as spi
import copy 



# A function to see the source code of a function. 
def showfunc(functiontoshow):
    print(inspect.getsource(functiontoshow))



# A function to generate covariance matrices based on correlation functions. 
# 2018-10-25   Below copied/modified from MATLAB to Python by B Lee
'''
COVMAT1D_FROM_CFUN   Correlation function based covariance matrix

   This function sets up a covariance matrix from a defined correlation 
   function. The correlation function is specified by giving a functional
   form (such as exponential decreasing) and correlation lengths. The
   correlation length is throughout defined as the distance where the 
   correlation has dropped to exp(-1). For off-diagonal values, the 
   correlation length is averaged between the two involved positions.

   Correlation matrices are obtained by setting *Std* to [].

FORMAT   S = covmat1d_from_cfun( xp, Std, cfun, Cl, [, cco, mapfun] )
       
OUT   S      The covariance matrix
IN    xp     The data abscissa.
      Std    Standard deviations. Given as a two vector matrix. First column
             holds position in same unit as *xp*. The second column is the 
             standard deviation at the postions of the first column. These
             values are then interpolated to *xp*, extrapolating end
             values to +-Inf (in a "nearest" manner).
             If set to a scalar, this value is applied for all *xp*.
             If set to [], unit standard deviation is assumed.
      cfun   Correlation function. Possible choices are
              'drc' : Dirac. No correlation. Any given correlation length
                      is ignored here.
              'lin' : Linearly decreasing (down to zero).
              'exp' : Exponential decreasing (exp(-dx/cl)).
              'gau' : Gaussian (normal) deceasing (exp(-(dx/cl))^2).
OPT   Cl     Correlation lengths. Given as a column matrix as *Std*.
             Must be given for all *cfun* beside 'drc'. Extrapolation as
             for *Std*. Scalar input is allowed.
      cco    Correlation cut-off. All values below this limit are set to 0.
      mapfun Mapping function from grid unit to unit for corrleation
             lengths. For example, if correlation lengths are given in
             pressure decades, while the basic coordinate is Pa, this is
             *mapfun* handled by setting *mapfun* to @log10. 

2005-05-20   Created by Patrick Eriksson.
'''
def covmat1d_from_cfun(xp1,Std1,cfun1,Cl1=None,cco1=0,mapfun1=None):        
    
    xp = copy.deepcopy(xp1)
    Std = copy.deepcopy(Std1)
    cfun = copy.deepcopy(cfun1)
    Cl = copy.deepcopy(Cl1)
    cco = copy.deepcopy(cco1)
    mapfun = copy.deepcopy(mapfun1)
    
    def handle_expand(x,xi):
        
        v1 = np.min( x );
        i1 = ( xi<v1 );
        
        v2 = np.max( x );
        i2 = ( xi>v2 );
        
        if np.sum(v1) > 0:
            xi[i1,] = v1;
          
        if np.sum(v2) > 0:
            xi[i2,] = v2;
        
        return xi
    
    # Determine standard deviations
    #
    n  = len( xp );
    #
    if np.isscalar(Std):
        si = npm.repmat( Std, n, 1 );
    else:
        f = spi.interp1d(Std[:,0], Std[:,1])
        si = f(handle_expand(Std[:,0],xp));
        si = np.array([si]).T
    
    # Handle diagonal matrices separately (note return)
    #
    if cfun.lower() == 'drc':
        S = np.zeros([n,n])
        for i in range(0,n):
            S[i,i] = si[i]**2 ;
        return S
    
    # Conversion of length unit
    #
    if mapfun != None:
        if not callable(mapfun):
            Exception('Input *mapfun* must be empty or a function handle.')
        xp = mapfun(xp)
        if not np.isscalar(Cl):
            Cl[:,0] = mapfun( Cl[:,0] );
    
    # Distance matrix
    #
    [X1, X2] = npm.meshgrid(xp, xp);
    D = np.abs(X1-X2)
    
    # Correlation length matrix
    #
    if np.isscalar(Cl):
        L = Cl;
    else:
        f = spi.interp1d(Cl[:,0], Cl[:,1])
        cl = f(handle_expand(handle_expand(Cl[:,0],xp)));
        [X1,X2] = npm.meshgrid( cl, cl );
        L       = ( X1 + X2 ) / 2;
    
    # Create correlation matrix
    #
    if cfun.lower() == 'lin':
        S = 1 - (1-np.exp(-1)) * ( D/L );
        # Negativa values removed by cco
    elif cfun.lower() == 'exp':
        S = np.exp(-D/L)
    elif cfun.lower() == 'gau':
        S = np.exp( -(D/L)**2 );
    else:
        Exception('Unknown correlation function: ' + cfun)
    
    # Remove values below correlation cut-off limit, convert to sparse and
    # include standard deviations
    #
    S[S < cco] = 0;
    S       = (si @ si.T) * S;

    return S



# A class to read in radiometer data. Original provided by Dr. Choi. 
'''
Created on Jul 24, 2018
@author: reno
'''
class radiometrics:
    def read_lv0_data(self,f, *args):

        if not f:
            print("Missing argument FN, Please provide input filename.")
        if len(args) == 0:
            tmp_dir = os.path.dirname(f)
            dat_idx = '15'
        else:
            if len(args) == 1:
                dat_idx = args[0]
                tmp_dir = os.path.dirname(f)
            else:
                dat_idx = args[0]
                tmp_dir = args[1]

        dain = os.path.dirname(f)
        fn = os.path.basename(f)

        fn_sep = re.split('[. _]', fn)
        if fn_sep[2] != 'lv0':
            print("=============================================================================")
            print(" Given file is not Level2 data. Please try again with lv0 file. Returning...")
            print("=============================================================================")
            return -1

        f1 = "_".join(fn_sep[:len(fn_sep)-1]) + "_{0}".format( dat_idx ) + ".csv"
        file_exists = os.path.isfile( os.path.join(dain, f1) )
        print(file_exists) 
        if not file_exists:
            bsmwr = radiometrics()
            bsmwr.prepare_original(f)

        fin = os.path.join(dain, f1)

        delimiters = ("+", "-", "%", "/", '|', ":", " ", "(", ")")
        # Get headers (column titles)
        regexPattern = '|'.join(map(re.escape, delimiters))
        with open( fin, 'r' ) as d:
                line = d.readline()     # Read the first line in file
                h0 = re.split(',|\n', line)
        d.close()
        # Read data
        df = pd.read_csv(fin, skiprows=1,names=h0)

        # Make time understood in Pandas
        df["DateTime"] = pd.to_datetime(df["DateTime"], format='%m/%d/%Y %H:%M:%S', utc=True)

        self.df = df
        return df

    def read_lv1_data(self,f, *args):

        if not f:
            print("Missing argument FN, Please provide input filename.")
        if len(args) == 0:
            tmp_dir = os.path.dirname(f)
            dat_idx = '50'
        else:
            if len(args) == 1:
                dat_idx = args[0]
                tmp_dir = os.path.dirname(f)
            else:
                dat_idx = args[0]
                tmp_dir = args[1]

        dain = os.path.dirname(f)
        fn = os.path.basename(f)

        fn_sep = re.split('[. _]', fn)
        if fn_sep[2] != 'lv1':
            print("=============================================================================")
            print(" Given file is not Level2 data. Please try again with lv1 file. Returning...")
            print("=============================================================================")
            return -1

        f1 = "_".join(fn_sep[:len(fn_sep)-1]) + "_{0}".format( dat_idx ) + ".csv"
        file_exists = os.path.isfile( os.path.join(dain, f1) )
        print(file_exists) 
        if not file_exists:
            bsmwr = radiometrics()
            bsmwr.prepare_original(f)

        fin = os.path.join(dain, f1)

        delimiters = ("+", "-", "%", "/", '|', ":", " ", "(", ")")
        # Get headers (column titles)
        regexPattern = '|'.join(map(re.escape, delimiters))
        with open( fin, 'r' ) as d:
                line = d.readline()     # Read the first line in file
                h0 = re.split(',|\n', line)
        d.close()
        # Read data
        df = pd.read_csv(fin, skiprows=1,names=h0)

        # Make time understood in Pandas
        df["DateTime"] = pd.to_datetime(df["DateTime"], format='%m/%d/%y %H:%M:%S', utc=True)

        self.df = df
        return df

    def read_lv2_data(self,f,*args):

        if not f:
            print("Missing argument FN, Please provide input filename.")
        if len(args) == 0:
            tmp_dir = os.path.dirname(f)
            dat_idx = '400'
        else:
            if len(args) == 1:
                dat_idx = args[0]
                tmp_dir = os.path.dirname(f)
            else:
                dat_idx = args[0]
                tmp_dir = args[1]

        dain = os.path.dirname(f)
        fn = os.path.basename(f)

        fn_sep = re.split('[. _]', fn)
        if fn_sep[2] != 'lv2':
            print("=============================================================================")
            print(" Given file is not Level2 data. Please try again with lv2 file. Returning...")
            print("=============================================================================")
            return -1

        f1 = "_".join(fn_sep[:len(fn_sep)-1]) + "_{0}".format( dat_idx ) + ".csv"
        file_exists = os.path.isfile( os.path.join(dain, f1) )
        print(file_exists) 
        if not file_exists:
            bsmwr = radiometrics()
            bsmwr.prepare_original(f)

        fin = os.path.join(dain, f1)

        delimiters = ("+", "-", "%", "/", '|', ":", " ", "(", ")")
        # Get headers (column titles)
        regexPattern = '|'.join(map(re.escape, delimiters))
        with open( fin, 'r' ) as d:
                line = d.readline()     # Read the first line in file
                h0 = re.split(',|\n', line)
        d.close()
        # Read data
        df = pd.read_csv(fin, skiprows=1,names=h0)

        # Make time understood in Pandas
        df["DateTime"] = pd.to_datetime(df["DateTime"], format='%m/%d/%y %H:%M:%S', utc=True)

        self.df = df
        return df
    
    def prepare_original(self, fn, *args):

        if not fn:
            print("Missing argument FN, Please provide input filename.")
        if len(args) == 0:
            tmp_dir = os.path.dirname(fn)
        else:
            tmp_dir = args[0]
        fn_sep = fn.split(".")

        dataidx = []
        datanum = []
        delimiters = ("+", "-", "%", "/", '|', ":", " ", "(", ")")
        regexPattern = '|'.join(map(re.escape, delimiters))
        with open( fn, 'r' ) as d:
            while True:
                line = d.readline()     # Read the first line in file
                if not line: 
                    self.dataidx = [0]
                    return dataidx
                    break
                h0 = re.split(',|\n', line)
                nh = len( h0 ) - 1      # Number of headers. Ignore last header since it's '/n'
                # 
                # Remove inappropriate letters for BIN filename
                # 
                h1 = [ "".join( re.split(regexPattern, h0[i]) ) for i in range(0,nh) ]
                                        #---------------------------
                if h1[0] == "Record":   # Header for each data type
                                        #---------------------------
                    dataidx.append( h1[2] )
                    datanum.append(1)

                    foun = ".".join(fn_sep[:len(fn_sep)-1]) + "_{0}".format( h1[2] ) + '.csv'
                    fou = open( os.path.join(tmp_dir, foun), 'w' )                     # Open input file
                    fou.write( ",".join(h1) + "\n" )                                # Write data (Add \n for new line
                    fou.close()                                                     # Close input file
                                        #-------------------------
                else:                   # Data for each data type
                                        #-------------------------
                    if h1[2] == '99':
                        if dataidx.count('99') == 0:
                            dataidx.append( h1[2] )
                            datanum.append(1)
                            file_op_index = 'w'
                        else:
                            datanum[dataidx.index('99')] = datanum[dataidx.index('99')] + 1
                            file_op_index = 'a'

                            foun = ".".join(fn_sep[:len(fn_sep)-1]) + "_{0}".format( "99" ) + '.csv'
                            fou = open( os.path.join(tmp_dir, foun), file_op_index )   # Open input file
                            fou.write( line )                                       # Write data
                            fou.close()                                             # Close input file
                    else:
                        for i in range( 0, len(dataidx) ):
                            line_index = int(h1[2])
                            data_index = int(dataidx[i])
                            if (line_index > data_index and line_index <= data_index + 5):
                                #
                                # There are types of data line ends with comma(,), which Python code recognise
                                # as null('') at re.split. In this case, comma separated array shows one more
                                # elements than it should. If last element of comma separated array contains
                                # null(''), it should drop as bellow. 
                                #
                                h1len = len(h1)
                                if h1[h1len-1] == '':
                                    h1 = h1[0:h1len-1]
                                    nh = nh - 1

                                datanum[dataidx.index(dataidx[i])] = datanum[dataidx.index(dataidx[i])] + 1

                                foun = ".".join(fn_sep[:len(fn_sep)-1]) + "_{0}".format( dataidx[i] ) + '.csv'
                                fou = open( os.path.join(tmp_dir, foun), 'a' )         # Open input file
                                fou.write( line )                                   # Write data
                                fou.close()                                         # Close input file

        d.close()
        self.dataidx = dataidx
        return dataidx




    
    
    
