# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:22:32 2020

@author: admin
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.transforms as transforms

import pandas as pd

from ppp_pub_figs_settings import *
from ppp_pub_figs_fx import *
from ppp_pub_figs_supp import *

import pandas as pd

import trompy as tp

def makeheatmap(ax, data, events=None, ylabel='Trials', xscalebar=False, sort=True):
    ntrials = np.shape(data)[0]
    xvals = np.linspace(-9.9,20,300)
    yvals = np.arange(1, ntrials+2)
    xx, yy = np.meshgrid(xvals, yvals)
    
    if sort == True:
        try:
            inds = np.argsort(events)
            data = [data[i] for i in inds]
            events = [events[i] for i in inds]
        except:
            print("Events cannot be sorted")
            
    mesh = ax.pcolormesh(xx, yy, data, cmap=heatmap_color_scheme, shading = 'flat')
    
    # events = [-e for e in events]
    
    if events:
        ax.vlines(events, yvals[:-1], yvals[1:], color='w')
    else:
        print('No events')
        
    ax.set_ylabel(ylabel, rotation=90, labelpad=2)
    # ax.set_yticks([1, ntrials])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylim([ntrials+1, 1])
    
    if xscalebar:
        ax.plot([15, 19.9], [ntrials+2, ntrials+2], linewidth=2, color='k', clip_on=False)
        ax.text(17.5, ntrials+3, "5 s", va="top", ha="center")
    
    return ax, mesh

def fig1_heatmap_panel(df, diet):
    
    if diet == 'NR':
        color = [almost_black, 'xkcd:bluish grey']
        errorcolors = ['xkcd:silver', 'xkcd:silver']
        rat = 'PPP1-7'
        clims = [-3,4]
    else:
        color = [col['pr_cas'], col['pr_malt']]
        errorcolors = ['xkcd:silver', 'xkcd:silver']
        rat = 'PPP1-4'
        clims =  [-3,3]

    data_cas = df['pref1_cas'][rat]
    data_malt = df['pref1_malt'][rat]
    event_cas = df['pref1_cas_event'][rat]
    event_malt = df['pref1_malt_event'][rat]    

    gs = gridspec.GridSpec(nrows=2, ncols=1, wspace=0.45, left=0.25, bottom=0.08, right=0.8,
                                             height_ratios=[0.05,1],
                                             hspace=0.0)
    f = plt.figure(figsize=(1.4, 2.2))
    
    plots_gs = gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec=gs[1,0],
                                             width_ratios=[12,1],
                                             wspace=0.05)
    
    marker_gs = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[0,0],
                                             width_ratios=[12,1],
                                             wspace=0.05)

    ax1 = f.add_subplot(plots_gs[0,0])
    ax, mesh = makeheatmap(ax1, data_cas, events=event_cas, ylabel='Casein trials')
    mesh.set_clim(clims)
    
    ax2 = f.add_subplot(plots_gs[1,0], sharex=ax1)
    ax, mesh = makeheatmap(ax2, data_malt, events=event_malt, ylabel='Malt. trials', xscalebar=True)
    mesh.set_clim(clims)
    
    ax0 = f.add_subplot(marker_gs[0,0], sharex=ax1)
    ax0.axis('off')

    ax0.plot([0,5], [0,0], color='xkcd:silver', linewidth=3)
    ax0.annotate("Licks", xy=(2.5, 0), xytext=(0,5), textcoords='offset points',
        ha='center', va='bottom')
        
    ax1.set_xlim([-10,20])
    
    cbar_ax = f.add_subplot(plots_gs[0,1])   
    cbar = f.colorbar(mesh, cax=cbar_ax, ticks=[clims[0], 0, clims[1]])
    
    ## for labels with raw blue signal 
    # cbar_labels = ['{0:.0f}%'.format(clims[0]*100),
    #                '0% \u0394F',
    #                '{0:.0f}%'.format(clims[1]*100)]
    
    cbar_labels = ['{0:.0f}'.format(clims[0]),
                   '0 Z',
                   '{0:.0f}'.format(clims[1])]
    cbar.ax.set_yticklabels(cbar_labels)
    
    ax3 = f.add_subplot(plots_gs[2,0])
 
    tp.shadedError(ax3, data_cas, linecolor=color[0], errorcolor=errorcolors[0])
    tp.shadedError(ax3, data_malt, linecolor=color[1], errorcolor=errorcolors[1])
    
    ax3.axis('off')

    y = [y for y in ax3.get_yticks() if y>0][:2]
    l = y[1] - y[0]
    ## for use with raw blue signal
    # scale_label = '{0:.0f}% \u0394F'.format(l*100)
    
    scale_label = '{0:.0f} Z'.format(l)
    
    ax3.plot([50,50], [y[0], y[1]], c=almost_black)
    ax3.text(40, y[0]+(l/2), scale_label, va='center', ha='right')

# Adds x scale bar   
    y = ax3.get_ylim()[0]
    ax3.plot([251,300], [y, y], c=almost_black, linewidth=2)
    ax3.annotate('5 s', xy=(276,y), xycoords='data',
                xytext=(0,-5), textcoords='offset points',
                ha='center',va='top')
    return f


# def fig1_new(df_behav, df_heatmaps, df_photo):
    
#     print("Woo yeah")
    
#     # f1 = fig1_forced_behavior(df_behav, df_photo)
    
#     # f2 = fig1_heatmap_panel(df_heatmap, "NR")
    
#     # f3 = fig1_heatmap_panel(df_heatmap, "PR")
    
#     f1, f2, f3 = [], [], []
    
#     f4 = fig1_photogroup_panel(df_photo)
    
#     return f1, f2, f3, f4

f2 = fig1_heatmap_panel(df_heatmap, "NR")

f3 = fig1_heatmap_panel(df_heatmap, "PR")

f2.savefig(savefolder+"f2_heatmap_NR.pdf")
f3.savefig(savefolder+"f2_heatmap_PR.pdf")