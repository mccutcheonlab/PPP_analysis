# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:39:12 2018

@author: jaimeHP
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def heatmapFig(ax, data):
    ntrials = np.shape(data)[0]
    xvals = np.linspace(-9.9,20,300)
    yvals = np.arange(1, ntrials+2)
    xx, yy = np.meshgrid(xvals, yvals)
    
    mesh = ax.pcolormesh(xx, yy, data, cmap='RdBu', vmin = -0.15, vmax=0.22, shading = 'flat')
    ax.set_ylabel('Trials')
    ax.set_yticks([1, ntrials])
    ax.invert_yaxis()
    
    return ax, mesh

def removenoise(snipdata):
    # returns blue snips with noisey ones removed
    new_snips = [snip for (snip, noise) in zip(snipdata['blue'], snipdata['noise']) if not noise]
    return new_snips

s = 's10'
rat = 'PPP1.7'
x = rats[rat].sessions[s]

data_cas = removenoise(x.cas['snips_licks_forced'])
data_malt = removenoise(x.malt['snips_licks_forced'])

f,(ax1, ax2) = plt.subplots(2,1, sharex=True)

ax, mesh = heatmapFig(ax1, data_cas)
ax, mesh = heatmapFig(ax2, data_malt)

f.subplots_adjust(right=0.8)
cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])

f.colorbar(mesh, cax=cbar_ax)






