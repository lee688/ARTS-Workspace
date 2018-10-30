'''
Created on Jul 26, 2018

@author: reno
'''
import os, re, sys, glob
import pandas as pd
import numpy as np
import imp

dain = "/Users/reno/Google Drive/[NIMS]Radiometer/sample_data"
#fn = "2018-04-10_00-04-09_lv0.csv"
fn = "2018-04-10_00-04-09_lv1.csv"
#fn = "2018-04-10_00-04-09_lv2.csv"
f = os.path.join(dain,fn)

module = imp.load_source('radiometrics', '/Users/reno/Google Drive/_Programming/Python/Development/radiometrics/radiometrics_read.py')
bsmwr  = module.radiometrics()
#out = bsmwr.prepare_original(f)

df = bsmwr.read_lv1_data(f)
print('df =', df)
sf = df[df["50"] == 51]
print('sf =', sf)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ts = pd.DataFrame(sf["Ch23.000"], index=sf["Record"])

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,3)))
plt.plot(sf["DateTime"], sf["Ch22.234"], label='22.2')
plt.plot(sf["DateTime"], sf["Ch57.288"], label='52.2')
plt.plot(sf["DateTime"], sf["Ch58.800"], label='58.8')
plt.xlabel('Date-Time')
plt.ylabel('Tb')
plt.title("Title")
plt.grid(True)
plt.legend()
plt.show()
