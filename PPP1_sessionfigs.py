# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:15:26 2017

PPP1 session figs, for individual rats when assembling data

@author: jaimeHP
"""

def makeBehavFigs(x):
    # Initialize figure
    behavFig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    gs1 = gridspec.GridSpec(5, 2)
    gs1.update(left=0.10, right= 0.9, wspace=0.5, hspace = 0.7)
    plt.suptitle('Rat ' + x.rat + ': Session ' + x.session)
    
    ax = plt.subplot(gs1[0, :])
    x.sessionlicksFig(ax)

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