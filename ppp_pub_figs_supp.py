# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:11:25 2018

@author: James Rig
"""

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import JM_custom_figs as jmfig
import JM_general_functions as jmf

from ppp_pub_figs_settings import *

def sacc_behav_fig(df):
    f, ax = plt.subplots(figsize=(7.2, 2.5), ncols=2)
    
    scattersize = 50
    
    x = [[df.xs('NR', level=1)['latx1'], df.xs('NR', level=1)['latx2'], df.xs('NR', level=1)['latx3']],
     [df.xs('PR', level=1)['latx1'], df.xs('PR', level=1)['latx2'], df.xs('PR', level=1)['latx3']]]
    
    jmfig.barscatter(x, paired=True, unequal=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [col['np_cas'], col['np_malt'], col['lp_cas'], col['lp_malt']],
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel=['NR', 'PR'],
                 barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                 scattersize = scattersize,
                 ylim=[-1,20],
                 ax=ax[0])

    x = [[df.xs('NR', level=1)['missed1'], df.xs('NR', level=1)['missed2'], df.xs('NR', level=1)['missed3']],
     [df.xs('PR', level=1)['missed1'], df.xs('PR', level=1)['missed2'], df.xs('PR', level=1)['missed3']]]

    jmfig.barscatter(x, paired=True, unequal=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [col['np_cas'], col['np_malt'], col['lp_cas'], col['lp_malt']],
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel=['NR', 'PR'],
                 barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                 scattersize = scattersize,
                 ylim=[-5,50],
                 ax=ax[1])

    return f

def cond_licks_fig(ax, df, diet):
    
    scattersize = 50
    
    if diet == 'NR':
        cols = [col['np_cas'], col['np_cas'], col['np_malt'], col['np_malt']]
        title = 'Non-restricted'
    else:
        cols = [col['lp_cas'], col['lp_cas'], col['lp_malt'], col['lp_malt']]
        title = 'Protein restricted'

    x = [[df.xs(diet, level=1)['cond1-cas1-licks'], df.xs(diet, level=1)['cond1-cas2-licks']],
     [df.xs(diet, level=1)['cond1-malt1-licks'], df.xs(diet, level=1)['cond1-malt2-licks']]]

    jmfig.barscatter(x, paired=True, unequal=True,
             barfacecoloroption = 'individual',
             barfacecolor = cols,
             scatteredgecolor = ['xkcd:charcoal'],
             scatterlinecolor = 'xkcd:charcoal',
             grouplabel=['Cas', 'Malt'],
             barlabels=['1', '2', '1', '2'],
             scattersize = scattersize,
             
#             ylim=[-5,50],
             ax=ax)
    ax.set_yticks([0,1000,2000,3000,4000])
    ax.set_title(title)

def cond_photo_fig(ax, df, diet, keys, event='',
                 color=[almost_black, 'xkcd:bluish grey'],
                 errorcolors=['xkcd:silver', 'xkcd:silver'],
                 yerror=True):

    if diet == 'NR':
        color=[almost_black, 'xkcd:bluish grey']
        errorcolors=['xkcd:silver', 'xkcd:silver']
        title = 'Non-restricted'
    else:
        color=[green, light_green]
        errorcolors=['xkcd:silver', 'xkcd:silver']
        title = 'Protein restricted'
    
    df = df.xs(diet, level=1)
    
    # Plots casein and maltodextrin shaded erros
    jmfig.shadedError(ax, df[keys[0]], linecolor=color[0], errorcolor=errorcolors[0])
    jmfig.shadedError(ax, df[keys[1]], linecolor=color[1], errorcolor=errorcolors[1])
    
    
    
    #ax.legend(['Casein', 'Maltodextrin'], fancybox=True)    
    ax.axis('off')

# Marks location of event on graph with arrow    
    arrow_y = ax.get_ylim()[1]
    ax.plot([100, 150], [arrow_y, arrow_y], color='xkcd:silver', linewidth=3)
    ax.annotate(event, xy=(125, arrow_y), xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom')

# Adds y scale bar
    if yerror:
        y = [y for y in ax.get_yticks() if y>0][:2]
        l = y[1] - y[0]
        scale_label = '{0:.0f}% \u0394F'.format(l*100)
        ax.plot([50,50], [y[0], y[1]], c=almost_black)
        ax.text(40, y[0]+(l/2), scale_label, va='center', ha='right')

# Adds x scale bar   
    y = ax.get_ylim()[0]
    ax.plot([251,300], [y, y], c=almost_black, linewidth=2)
    ax.annotate('5 s', xy=(276,y), xycoords='data',
                xytext=(0,-5), textcoords='offset points',
                ha='center',va='top')
    
def cond_photobar_fig(ax, df, diet, keys):
    
    scattersize = 50
    
    if diet == 'NR':
        cols = [col['np_cas'], col['np_cas'], col['np_malt'], col['np_malt']]
        title = 'Non-restricted'
    else:
        cols = [col['lp_cas'], col['lp_cas'], col['lp_malt'], col['lp_malt']]
        title = 'Protein restricted'
    
    x = [[df.xs(diet, level=1)[keys[0][0]], df.xs(diet, level=1)[keys[0][1]]],
          [df.xs(diet, level=1)[keys[1][0]], df.xs(diet, level=1)[keys[1][1]]]]
    
    jmfig.barscatter(x, paired=True, unequal=True,
         barfacecoloroption = 'individual',
         barfacecolor = cols,
         scatteredgecolor = ['xkcd:charcoal'],
         scatterlinecolor = 'xkcd:charcoal',
         grouplabel=['Cas', 'Malt'],
         barlabels=['1', '2', '1', '2'],
         scattersize = scattersize,
         
#             ylim=[-5,50],
         ax=ax)

def longtracefig(f, gs, longtrace):
    
    inner = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs[0,:],
                                             height_ratios=[1,8],
                                             hspace=0.05)
    pre = longtrace['pre']
    post = longtrace['post']
    fs = longtrace['fs']
    
    blue_sig = longtrace['blue'] / longtrace['f0'][0]
    uv_sig = longtrace['uv'] / longtrace['f0'][1]   
    
    ax1 = f.add_subplot(inner[1,0])    
    ax1.axis('off')
    
    ax1.plot(blue_sig, c='xkcd:azure', linewidth=1)
    ax1.plot(uv_sig, c='xkcd:amethyst',linewidth=1)
    
    w = (pre+post) * fs
    h = ax1.get_ylim()[1] - ax1.get_ylim()[0]

    for event in longtrace['all_events']:
        start = event - (pre * fs)
        rect = patches.Rectangle((start,ax1.get_ylim()[0]),w,h,
                                 linewidth=1, linestyle='dashed', edgecolor='xkcd:silver', facecolor='none')
        ax1.add_patch(rect)
        
    old_ax = ax1.get_ylim()
    ax1.set_ylim([old_ax[0]-0.005, old_ax[1]+0.001])
    
    l = 0.03
    yrange = ax1.get_ylim()[1] - ax1.get_ylim()[0]        
    y1 = yrange * 0.6 + ax1.get_ylim()[0]
    y2 = y1 + l        
    scale_label = '{0:.0f}% \u0394F'.format(l*100)
    ax1.plot([0, 0], [y1, y2], c=almost_black, linewidth=1.5)
    ax1.text(-fs, y1 + (l/2), scale_label, va='center', ha='right')

    y = ax1.get_ylim()[0]
    xvals = [ax1.get_xlim()[1] - 10*fs, ax1.get_xlim()[1]]
    ax1.plot(xvals, [y, y], c=almost_black, linewidth=2)
    text_x = ((xvals[1]-xvals[0]) / 2) + xvals[0]
    ax1.annotate('10 s', xy=(text_x,y), xycoords='data',
                xytext=(0,-5), textcoords='offset points',
                ha='center',va='top')
        
#    ax2 = f.add_subplot(inner[0,0], sharex=ax1)
#    ax2.axis('off')

def reptrace(f, gs, gsx, tracedata, yscale=False, xscale=False, event_text=False, ax_to_match=[]):
    
    pre = longtrace['pre']
    post = longtrace['post']
    fs = longtrace['fs']
    
    text_offset= 1 * fs
    
    blue_sig = tracedata['blue'] / longtrace['f0'][0]
    uv_sig = tracedata['uv'] / longtrace['f0'][1]  
    
    inner = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs[1,gsx],
                                             height_ratios=[1,8],
                                             hspace=0.05)
    try:
        ax1 = f.add_subplot(inner[1,0], sharey=ax_to_match)
    except:
        ax1 = f.add_subplot(inner[1,0])
    
    ax1.axis('off')
    ax1.plot(blue_sig, c='xkcd:azure', linewidth=1)
    ax1.plot(uv_sig, c='xkcd:amethyst', linewidth=1)
    
    if yscale:
        l = 0.03
        yrange = ax1.get_ylim()[1] - ax1.get_ylim()[0]        
        y1 = yrange * 0.6 + ax1.get_ylim()[0]
        y2 = y1 + l        
        scale_label = '{0:.0f}% \u0394F'.format(l*100)
        xvals = [0, 0]
        ax1.plot(xvals, [y1, y2], c=almost_black, linewidth=1.5)
        ax1.text(-fs, y1 + (l/2), scale_label, va='center', ha='right')
    
    # Adds x scale bar
    if xscale:
        y = ax1.get_ylim()[0]
        xvals = [ax1.get_xlim()[1] - 2*fs, ax1.get_xlim()[1]]
        ax1.plot(xvals, [y, y], c=almost_black, linewidth=1.5)
        text_x = ((xvals[1]-xvals[0]) / 2) + xvals[0]
        ax1.annotate('2 s', xy=(text_x,y), xycoords='data',
                    xytext=(0,-5), textcoords='offset points',
                    ha='center',va='top')

    ax2 = f.add_subplot(inner[0,0], sharex=ax1)
    ax2.axis('off')
    
    sipper_x = pre * fs
    ax2.plot(sipper_x, 1, 'v', color='xkcd:silver')
    licks_x = [(lick+pre) * fs for lick in tracedata['licks']]
    
    yvals = [0]*len(licks_x)
    ax2.plot(licks_x,yvals,linestyle='None',marker='|',markersize=5, color='xkcd:silver')
    
    if event_text:
        ax2.annotate('Licks', xy=(sipper_x-text_offset,0), va='center', ha='right')
        ax2.annotate('Sipper', xy=(sipper_x-text_offset, 1), xytext=(0,0), textcoords='offset points',
                    ha='right', va='center')
        
    return ax1

def figS2_rep(longtrace):
    gs = gridspec.GridSpec(2, 3, wspace=0.5, hspace=0.3, bottom=0.1)
    f = plt.figure(figsize=(5,4))
    
    longtracefig(f, gs, longtrace)
    
    ax_trace1 = reptrace(f, gs, 0, longtrace['event1'], yscale=True, event_text=True)    
    ax_trace2 = reptrace(f, gs, 1, longtrace['event2'], ax_to_match=ax_trace1)    
    ax_trace3 = reptrace(f, gs, 2, longtrace['event3'], xscale=True, ax_to_match=ax_trace1)
    
    return f