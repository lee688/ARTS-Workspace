'''
Created on Jul 31, 2018

@author: reno
'''
import os, re
import pandas as pd
import numpy as np


dain = "/Users/reno/Google Drive/[NIMS18]IOP/ldps.01/201807/01"
fn = "ldps_v070_erlo_etas_h000.2018070100.gb2"
f = os.path.join(dain,fn)
print('filename =', f)

names = ['Index', 'Type', 'Altitude', '_0', '_1', 'Lon', 'Lat', 'Val']
sep=':| m above ground|lon=|lat=|val='
df = pd.read_csv(f, skiprows=0, names=names, sep=sep, engine='python')


print(df.loc[lambda df: df.Type == 'POT', :])