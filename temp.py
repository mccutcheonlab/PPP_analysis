# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:39:12 2018

@author: jaimeHP
"""

import matplotlib.pyplot as plt
import numpy as np

s = 's10'
rat = 'PPP1.7'
x = rats[rat].sessions[s]
data = x.malt['snips_licks_forced']['blue']
data = x.cas['snips_licks_forced']['blue']


ntrials = np.shape(data)[0]
xvals = np.linspace(-9.9,20,300)
yvals = np.arange(1, ntrials+2)
xx, yy = np.meshgrid(xvals, yvals)

#data = [[1,2,3,4], [3,4,5,6], [5,4,3,2]]
f,ax = plt.subplots(1)
##heatmapFig(ax, data)
ax.pcolormesh(xx, yy, data, shading = 'flat')
ax.set_ylabel('Trials')
ax.set_yticks([1, ntrials])
ax.invert_yaxis()



