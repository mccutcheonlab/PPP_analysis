# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:41:31 2019

@author: jmc010
"""
# Import modules
import tdt
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

import timeit


def correctforbaseline(blue, uv):
    pt = len(blue)
    X = np.fft.rfft(uv, pt)
    Y = np.fft.rfft(blue, pt)
    Ynet = Y-X

    datafilt = np.fft.irfft(Ynet)

    datafilt = sig.detrend(datafilt)

    b, a = sig.butter(9, 0.012, 'low', analog=True)
    datafilt = sig.filtfilt(b, a, datafilt)
    
    return datafilt


# Set variables - hardcoded here but could be read in via metafile
tdtfile='C:\\Github\\PPP_analysis\\data\\Eelke-171027-111329\\'
tdtfile='C:\\Github\\PPP_analysis\\data\\Giulia-190930-102422\\'
SigBlue='Dv1B'
SigUV='Dv2B'

ttl_licksL='LL1_'
ttl_licksR='RL1_'

tic = timeit.default_timer()
#tmp=tdt.read_block(tdtfile, nodata=True)
tmp = tdt.read_block(tdtfile, t2=2, evtype=['streams'])
toc = timeit.default_timer()

print(toc-tic)
print(tmp.streams)


# Read streamed data in
tmp = tdt.read_block(tdtfile, evtype=['streams'], store=[SigBlue])
#data_blue = getattr(tmp.streams, SigBlue)['data']
#fs = getattr(tmp.streams, SigBlue)['fs']
#
#tmp = tdt.read_block(tdtfile, evtype=['streams'], store=[SigUV])
#data_uv = getattr(tmp.streams, SigUV)['data']
#
#
## Perform baseline correction
#data_filt = correctforbaseline(data_blue, data_uv)
#
#
#
## Read in TTLs
ttls = tdt.read_block(tdtfile, evtype=['epocs']).epocs
#
#lt = getattr(ttls, ttl_licksL)
#rt = getattr(ttls, ttl_licksR)
#
## Get ticks if not ticks saved
sc = tdt.read_block(tdtfile, evtype=['scalars']).scalars
#scalars.scalars.Pars.ts(1:2:end)
#
## Plots data - x-axis of lower panel does not match upper two panels but you
## get the idea
#f, ax = plt.subplots(nrows=3, figsize=(8,8))
#ax[0].plot(data_blue, color='blue')
#ax[0].plot(data_uv, color='pink')
#
#ax[1].plot(data_filt, color='green')
#
#ax[2].scatter(lt.onset, [1]*len(lt.onset))
#ax[2].scatter(rt.onset, [2]*len(rt.onset))
#ax[2].set_ylim([0, 3])