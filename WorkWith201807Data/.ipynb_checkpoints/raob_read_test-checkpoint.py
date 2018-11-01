'''
Created on Aug 3, 2018

@author: reno
'''
import os
import pandas as pd

dain = "/Users/reno/Google Drive/[NIMS18]IOP/raob.01/BSWO/Messages.lv2"
fn = "UPP_LV2_RS92-SGP_47258_201807031800.txt"
f = os.path.join(dain,fn)

d = pd.read_csv(f, sep=",")

print(d)

header = list(d)
print(header)
