# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:20:48 2018

@author: James Rig
"""
import matplotlib.pyplot as plt
import numpy as np

import numpy.polynomial.polynomial as poly

import JM_custom_figs as jmfig
import JM_general_functions as jmf


from scipy import signal

def FPfftfilt(blue, uv):
    [c,d]=signal.butter(9,0.012,btype='low',output='ba')
    
    pt = len(blue)
    X = np.fft.fft(blue,pt)
    Y = np.fft.fft(uv,pt)
    Ynet = Y-X
    
    blue_new = np.fft.ifft(Ynet)
    blue_filt = signal.filtfilt(c, d, np.real(blue_new))
  
    return(np.real(blue_filt))
    


x = sessions_to_add['PPP3-8_s10']

#type(x.left['licks-forced'])





#end = len(x.t2sMap)-60000
#
#datarange = range(60000, end)
#
#xdata = x.t2sMap[datarange]
#ydata = x.data[datarange]
#ydataUV = x.dataUV[datarange]
#
#blue_filt = FPfftfilt(ydata,ydataUV)

#timelock = x.left['sipper']
#timelock_events = [licks for licks in x.left['lickdata']['rStart'] if licks in x.left['licks-forced']]
#
#blue_filt_trials,_ =  jmf.snipper(blue_filt, timelock_events, fs = x.fs, t2sMap = x.t2sMap, preTrial=10, trialLength=30,
#                 adjustBaseline = True,
#                 bins = 300)

f, ax = plt.subplots()
jmfig.shadedError(ax, x.right['snips_sipper']['blue_z'])

    

    



#
#coefs, stats = poly.polyfit(xdata, ydata, 2, full=True)
#
#print(coefs)
#print(stats)
#
#fitted = poly.Polynomial.fit(xdata, ydata, 2)
#
#print(fitted.coef)


#fit = np.polynomial.polynomial.Polynomial.fit(x.t2sMap[datarange], x.data[datarange], 2)
#
#np.polynomial.polynomial.Polynomial.fit(xdata, ydataUV, 2)

