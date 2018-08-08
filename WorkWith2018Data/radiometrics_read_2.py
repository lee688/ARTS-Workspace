'''
Created on Jul 24, 2018

@author: reno
'''
import os, re, sys, glob
import pandas as pd
import numpy as np

class radiometrics:
    def read_lv0_data(self,f, *args):

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
        df["DateTime"] = pd.to_datetime(df["DateTime"], format='%d/%m/%y %H:%M:%S', utc=True)

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
        df["DateTime"] = pd.to_datetime(df["DateTime"], format='%d/%m/%y %H:%M:%S', utc=True)

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
        df["DateTime"] = pd.to_datetime(df["DateTime"], format='%d/%m/%y %H:%M:%S', utc=True)

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
    