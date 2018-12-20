# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 10:27:38 2018

@author: James Rig
"""
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import JM_custom_figs as jmfig
import JM_general_functions as jmf

import ppp_pub_figs_fx as pppfig

import timeit
tic = timeit.default_timer()

#Colors
green = mpl.colors.to_rgb('xkcd:kelly green')
light_green = mpl.colors.to_rgb('xkcd:light green')
almost_black = mpl.colors.to_rgb('#262626')

## Colour scheme
col={}
col['np_cas'] = 'xkcd:silver'
col['np_malt'] = 'white'
col['lp_cas'] = 'xkcd:kelly green'
col['lp_malt'] = 'xkcd:light green'

def makeheatmap(ax, data, events=None, ylabel='Trials'):
    ntrials = np.shape(data)[0]
    xvals = np.linspace(-9.9,20,300)
    yvals = np.arange(1, ntrials+2)
    xx, yy = np.meshgrid(xvals, yvals)
    
    mesh = ax.pcolormesh(xx, yy, data, cmap='summer', shading = 'flat')
    
    if events:
        ax.vlines(events, yvals[:-1], yvals[1:], color='w')
    else:
        print('No events')
        
    ax.set_ylabel(ylabel)
    ax.set_yticks([1, ntrials])
    ax.set_xticks([])
    ax.invert_yaxis()
    
    return ax, mesh

def heatmapCol(f, df, gs, diet, session, rat, event='', reverse=False, clims=[0,1], colorgroup='control'):
    
    if colorgroup == 'control':
        color = [almost_black, 'xkcd:bluish grey']
        errorcolors = ['xkcd:silver', 'xkcd:silver']
    else:
        color = [green, light_green]
        errorcolors = ['xkcd:silver', 'xkcd:silver']
   
    data_cas = df[session+'_cas'][rat]
    data_malt = df[session+'_malt'][rat]
    event_cas = df[session+'_cas_event'][rat]
    event_malt = df[session+'_malt_event'][rat]
    
    if reverse:
            event_cas = [-event for event in event_cas]
            event_malt = [-event for event in event_malt]

    inner = gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec=gs[:,0],
                                             width_ratios=[12,1],
                                             wspace=0.05)
    ax1 = f.add_subplot(inner[0,0])
    ax, mesh = makeheatmap(ax1, data_cas, events=event_cas, ylabel='Casein trials')
    mesh.set_clim(clims)
    ax1.plot([0], [-1], 'v', color='xkcd:silver')
    
    ax2 = f.add_subplot(inner[1,0], sharex=ax1)
    ax, mesh = makeheatmap(ax2, data_malt, events=event_malt, ylabel='Malt. trials')
    mesh.set_clim(clims)
    
    if event == 'Sipper':
        print(event)
        
    for ax in [ax1, ax2]:
        ax.set_xlim([-10,20])
    
    cbar_ax = f.add_subplot(inner[0,1])   
    cbar = f.colorbar(mesh, cax=cbar_ax, ticks=[clims[0], 0, clims[1]])
    cbar_labels = ['{0:.0f}%'.format(clims[0]*100),
                   '0% \u0394F',
                   '{0:.0f}%'.format(clims[1]*100)]
    cbar.ax.set_yticklabels(cbar_labels)
    
    ax3 = f.add_subplot(inner[2,0])
 
    jmfig.shadedError(ax3, data_cas, linecolor=color[0], errorcolor=errorcolors[0])
    jmfig.shadedError(ax3, data_malt, linecolor=color[1], errorcolor=errorcolors[1])
    
    ax3.axis('off')

    y = [y for y in ax3.get_yticks() if y>0][:2]
    l = y[1] - y[0]
    scale_label = '{0:.0f}% \u0394F'.format(l*100)
    ax3.plot([50,50], [y[0], y[1]], c=almost_black)
    ax3.text(40, y[0]+(l/2), scale_label, va='center', ha='right')

# Adds x scale bar   
    y = ax3.get_ylim()[0]
    ax3.plot([251,300], [y, y], c=almost_black, linewidth=2)
    ax3.annotate('5 s', xy=(276,y), xycoords='data',
                xytext=(0,-5), textcoords='offset points',
                ha='center',va='top')

def averagetrace(ax, df, diet, keys, event='', fullaxis=True, colorgroup='control', ylabel=True):
    
    if colorgroup == 'control':
        color=[almost_black, 'xkcd:bluish grey']
        errorcolors=['xkcd:silver', 'xkcd:silver']
    else:
        color=[green, light_green]
        errorcolors=['xkcd:silver', 'xkcd:silver']
# Selects diet group to plot                
    df = df.xs(diet, level=1)

# Plots casein and maltodextrin shaded erros
    jmfig.shadedError(ax, df[keys[0]], linecolor=color[0], errorcolor=errorcolors[0])
    jmfig.shadedError(ax, df[keys[1]], linecolor=color[1], errorcolor=errorcolors[1])
    
    #ax.legend(['Casein', 'Maltodextrin'], fancybox=True)    
    if fullaxis == False:
        ax.axis('off')

# Adds y scale bar
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

    else:
        ax.set_xticks([0, 100, 200, 300])
        ax.set_xticklabels(['-10', '0', '10', '20'])
        ax.set_xlabel('Time from first lick (s)')
        
    if ylabel:
        ax.set_ylabel('Z-Score')
        
# Marks location of event on graph with arrow    
    arrow_y = ax.get_ylim()[1]
    ax.plot([100, 150], [arrow_y, arrow_y], color='xkcd:silver', linewidth=3)
    ax.annotate(event, xy=(125, arrow_y), xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom')

def peakbargraph(ax, df, diet, keys, peaktype='average', epoch=[100, 109],
                 sc_color='w', colorgroup='control', ylabel=True,
                 ylim=[-0.05, 0.1], grouplabeloffset=0):
    
    if colorgroup == 'control':
        bar_colors=['xkcd:silver', 'w']
    else:
        bar_colors=[green, light_green]
    
    epochrange = range(epoch[0], epoch[1])
    
    df = df.xs(diet, level=1)
    
    if peaktype == 'average':
        a1 = [np.mean(rat[epochrange]) for rat in df[keys[0]]]
        a2 = [np.mean(rat[epochrange]) for rat in df[keys[1]]]
        ylab = 'Mean Z-Score'
        
    if peaktype == 'auc':
        a1 = [np.trapz(rat[epochrange]) for rat in df[keys[0]]]
        a2 = [np.trapz(rat[epochrange]) for rat in df[keys[1]]]
        ylab = 'AUC'
        
    a = [a1, a2]
    x = jmf.data2obj1D(a)
    
    ax, x, _, _ = jmfig.barscatter(x, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = bar_colors,
                 scatteredgecolor = [almost_black],
                 scatterlinecolor = almost_black,
                 scatterfacecolor = [sc_color],
                 grouplabel=['Cas', 'Malt'],
                 grouplabeloffset=grouplabeloffset,
                 scattersize = 50,
                 ax=ax)

#    ax.set_yticks([-0.05,0,0.05, 0.1])
#    ax.set_yticklabels(['5%', '0%', '5%', '10%'])
#    ax.set_ylim(ylim)
    
    if ylabel:
        ax.set_ylabel(ylab)

def averageCol(f, df_photo, gs, diet, keys_traces, keys_lats, peaktype='average', epoch=[100,119], event=''):
    
    inner = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs[:,1])
    
    if diet == 'NR':
        colors = 'control'
    else:
        colors = 'exptl'
    
    ax1 = f.add_subplot(inner[0,0])
    averagetrace(ax1, df_photo, diet, keys_traces, event=event, fullaxis=True, colorgroup=colors)
    ax1.set_ylim([-1.5, 3.5])
    
    ax2 = f.add_subplot(gs[1,1]) 
    peakbargraph(ax2, df_photo, diet, keys_traces, peaktype=peaktype, epoch=epoch,
                 colorgroup=colors, ylim=[-0.04,0.12], grouplabeloffset=0.07)
    ax2.set_ylim([-25, 105])

def fig1_photo(df_heatmap, df_photo, diet, session, clims=[[0,1], [0,1]],
                 peaktype='average', epoch=[100,119],
                 keys_traces = ['pref1_cas_licks_forced', 'pref1_malt_licks_forced'],
                 keys_lats = ['pref1_cas_lats_all', 'pref1_malt_lats_all'],
                 event='Licks'):
    
    if diet == 'NR':
        rat='PPP1-7'
        colors = 'control'
    else:
        rat='PPP1-4'
        colors = 'exptl'
    
    gs = gridspec.GridSpec(2, 2, wspace=0.7, width_ratios=[1, 0.8], hspace=0.6, left=0.12, right=0.98)
    f = plt.figure(figsize=(3.2,3.2))
    
    heatmapCol(f, df_heatmap, gs, diet, session, rat, event=event, clims=clims, reverse=True, colorgroup=colors)
    
    averageCol(f, df_photo, gs, diet, keys_traces, keys_lats, peaktype=peaktype, epoch=epoch,  event=event)
        
    return f

def fig2_photo(df_photo, peaktype='average', epoch=[100,119],
               keys_traces = ['pref2_cas_licks_forced', 'pref2_malt_licks_forced'],
               event='Licks'):
    
    gs = gridspec.GridSpec(2, 2, hspace=0.6, wspace=0.7, left=0.20, right=0.98)
    f = plt.figure(figsize=(3.2,3.2))
    
    ax1 = f.add_subplot(gs[0,0])
    averagetrace(ax1, df_photo, 'NR', keys_traces, event=event, colorgroup='exptl')
    
    ax2 = f.add_subplot(gs[0,1], sharey=ax1)
    averagetrace(ax2, df_photo, 'PR', keys_traces, event=event, ylabel=False)
    ax2.set_ylim([-1.5, 2.5])
    
    ax3 = f.add_subplot(gs[1,0]) 
    peakbargraph(ax3, df_photo, 'NR', keys_traces, peaktype=peaktype, epoch=epoch,
                 colorgroup='exptl',
                 grouplabeloffset=0.1)
    
    ax4 = f.add_subplot(gs[1,1], sharey=ax3) 
    peakbargraph(ax4, df_photo, 'PR', keys_traces, peaktype=peaktype, epoch=epoch,
                 ylabel=False, grouplabeloffset=0.1)
    ax4.set_ylim([-25, 100])

    return f
    