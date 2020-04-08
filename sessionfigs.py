# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:15:26 2017

PPP1 session figs, for individual rats when assembling data

@author: jaimeHP
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
import JM_custom_figs as jmfig
import numpy as np

mpl.rc('lines', linewidth=0.5)

def makeBehavFigs(x, pdf_pages):
    # Initialize figure
    behavFig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    gs1 = gridspec.GridSpec(5, 2)
    gs1.update(left=0.10, right= 0.9, wspace=0.5, hspace = 0.7)
    plt.suptitle('Rat ' + x.rat + ': Session ' + x.session)
    
    ax = plt.subplot(gs1[0, :])
    sessionlicksFig(x, ax)

    if x.left['exist'] == True:
        behavFigsCol(gs1, 0, x.left)
        
    if x.right['exist'] == True:
        behavFigsCol(gs1, 1, x.right)
        
    ax = plt.subplot(gs1[4, 0])
    jmfig.latencyFig(ax, x)

    pdf_pages.savefig(behavFig)

def behavFigsCol(gs1, col, side):
    ax = plt.subplot(gs1[1, col])
    jmfig.licklengthFig(ax, side['lickdata'], color=side['color'])
    
    ax = plt.subplot(gs1[2, col])
    jmfig.iliFig(ax, side['lickdata'], color=side['color'])
    
    ax = plt.subplot(gs1[3, col])
    jmfig.cuerasterFig(ax, side['sipper'], side['lickdata']['licks'])
    
def sessionlicksFig(x, ax):
    if x.left['exist'] == True:
        licks = x.left['lickdata']['licks']
        ax.hist(licks, range(0, 3600, 60), color=x.left['color'], alpha=0.4)          
        yraster = [ax.get_ylim()[1]] * len(licks)
        ax.scatter(licks, yraster, s=50, facecolors='none', edgecolors=x.left['color'])

    if x.right['exist'] == True:
        licks = x.right['lickdata']['licks']
        ax.hist(licks, range(0, 3600, 60), color=x.right['color'], alpha=0.4)          
        yraster = [ax.get_ylim()[1]] * len(licks)
        ax.scatter(licks, yraster, s=50, facecolors='none', edgecolors=x.right['color'])           
    
    ax.set_xticks(np.multiply([0, 10, 20, 30, 40, 50, 60],60))
    ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'])
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Licks per min')
 
    
def autoscale_y(vector, zoom=1):
    med = np.median(vector)
    sd = np.std(vector)
    
    return [med-(sd*zoom), med+(sd*zoom)]

def sessionFig(x, ax, filtered=True):
    if filtered:
        ax.plot(x.data_filt, color='blue', linewidth=0.1)
        
        ax.set_ylim(autoscale_y(x.data_filt))
    else:
        ax.plot(x.data, color='blue', linewidth=0.1)
        try:
            ax.plot(x.dataUV, color='m', linewidth=0.1)
        except:
            print('No UV data.')
        ax.set_ylim(autoscale_y(x.data))
            
    ax.set_xticks(np.multiply([0, 10, 20, 30, 40, 50, 60],60*x.fs))
    ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'])
    ax.set_xlabel('Time (min)')
    ax.set_title('Rat ' + x.rat + ': Session ' + x.session)

def makePhotoFigs(x, pdf_pages):
    # Initialize photometry figure
    photoFig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    gs1 = gridspec.GridSpec(7, 4)
    gs1.update(left=0.125, right= 0.9, wspace=0.4, hspace = 0.8)
    plt.suptitle('Rat ' + x.rat + ': Session ' + x.session)

    ax = plt.subplot(gs1[0, :])
    sessionFig(x, ax, filtered=False)
    
    ax = plt.subplot(gs1[1, :])
    sessionFig(x, ax)

    if x.left['exist'] == True:
        photoFigsCol(gs1, 0, x.pps,
                     x.left['snips_sipper'],
                     x.left['snips_licks_forced'])

    if x.right['exist'] == True:
        photoFigsCol(gs1, 2, x.pps,
                     x.right['snips_sipper'],
                     x.right['snips_licks_forced'])
        
    if x.left['exist'] == True and x.right['exist'] == True:
        ax = plt.subplot(gs1[6, 0])
        jmfig.trialsMultShadedFig(ax, [x.left['snips_sipper']['filt_z'], x.right['snips_sipper']['filt_z']],
                                  x.pps,
                                  linecolor=[x.left['color'], x.right['color']],
                                  eventText = 'Sipper')

        ax = plt.subplot(gs1[6, 2])
        jmfig.trialsMultShadedFig(ax, [x.left['snips_licks']['filt_z'], x.right['snips_licks']['filt_z']],
                                  x.pps,
                                  linecolor=[x.left['color'], x.right['color']],
                                  eventText = 'Lick')
        
#    plt.savefig(userhome + '/Dropbox/Python/photometry/output-thph1-lp/' + x.rat + '.eps', format='eps', dpi=1000)
    pdf_pages.savefig(photoFig)
    
def photoFigsCol(gs1, col, pps, snips_sipper, snips_licks):
    ax = plt.subplot(gs1[2, col])
    jmfig.trialsFig(ax, snips_sipper['filt_z'], pps, noiseindex = snips_sipper['noise'],
                    eventText = 'Sipper',
                    ylabel = 'Delta F / F0')
    
    ax = plt.subplot(gs1[3, col])
    jmfig.trialsMultShadedFig(ax, [snips_sipper['uv'], snips_sipper['blue']],
                              pps, noiseindex = snips_sipper['noise'],
                              eventText = 'Sipper')
    
    ax = plt.subplot(gs1[2, col+1])
    data = [snip for snip, noise in zip(snips_sipper['filt_z'], snips_sipper['noise']) if not noise]
    jmfig.shadedError(ax, data)
    
    ax = plt.subplot(gs1[4, col])
    jmfig.trialsFig(ax, snips_licks['filt_z'], pps, noiseindex=snips_licks['noise'],
                    eventText = 'First Lick',
                    ylabel = 'Delta F / F0')
    
    ax = plt.subplot(gs1[5, col])
    jmfig.trialsMultShadedFig(ax, [snips_licks['uv'], snips_licks['blue']],
                              pps, noiseindex=snips_licks['noise'],
                              eventText = 'First Lick')
    
    ax = plt.subplot(gs1[4, col+1])
    data = [snip for snip, noise in zip(snips_licks['filt_z'], snips_licks['noise']) if not noise]
    jmfig.shadedError(ax, data)
    

    
