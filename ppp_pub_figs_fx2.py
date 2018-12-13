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

def makeheatmap(ax, data, ylabel='Trials'):
    ntrials = np.shape(data)[0]
    xvals = np.linspace(-9.9,20,300)
    yvals = np.arange(1, ntrials+2)
    xx, yy = np.meshgrid(xvals, yvals)
    
    mesh = ax.pcolormesh(xx, yy, data, cmap='copper', shading = 'flat')
    ax.set_ylabel(ylabel)
    ax.set_yticks([1, ntrials])
    ax.set_xticks([])
    ax.invert_yaxis()
    
    return ax, mesh

def heatmapFig(f, df, gs, diet, session, rat, event='', clims=[0,1]):
    
    if diet == 'NR':
        color = [almost_black, 'xkcd:bluish grey']
        errorcolors = ['xkcd:silver', 'xkcd:silver']
    else:
        color = [green, light_green]
        errorcolors = ['xkcd:silver', 'xkcd:silver']
   
    data_cas = df[session+'_cas'][rat]
    data_malt = df[session+'_malt'][rat]

    inner = gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec=gs[:,0],
                                             width_ratios=[12,1],
                                             wspace=0.05)
    ax1 = f.add_subplot(inner[0,0])
    ax, mesh = makeheatmap(ax1, data_cas, ylabel='Casein trials')
    mesh.set_clim(clims)
    ax1.plot([0], [-1], 'v', color='xkcd:silver')
    
    ax2 = f.add_subplot(inner[1,0], sharex=ax1)
    ax, mesh = makeheatmap(ax2, data_malt, ylabel='Malt. trials')
    mesh.set_clim(clims)
    
    if event == 'Sipper':
        print(event)
    
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

def averagetrace(ax, df, diet, keys, event=''):
    
    if diet == 'NR':
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
    ax.axis('off')

# Marks location of event on graph with arrow    
    arrow_y = ax.get_ylim()[1]
    ax.plot([100, 150], [arrow_y, arrow_y], color='xkcd:silver', linewidth=3)
    ax.annotate(event, xy=(125, arrow_y), xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom')

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

def peakbargraph(ax, df, diet, keys, sc_color='w'):
    
    if diet == 'NR':
        bar_colors=['xkcd:silver', 'w']
    else:
        bar_colors=[green, light_green]
    
    df = df.xs(diet, level=1)
    a = [df[keys[0]], df[keys[1]]]
    x = jmf.data2obj1D(a)
    
    ax, x, _, _ = jmfig.barscatter(x, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = bar_colors,
                 scatteredgecolor = [almost_black],
                 scatterlinecolor = almost_black,
                 scatterfacecolor = [sc_color],
                 grouplabel=['Cas', 'Malt'],
                 scattersize = 50,
                 ax=ax)

    ax.set_ylabel('Peak (\u0394F)')
#    ax.set_ylim([-0.04, 0.14])
    plt.yticks([0,0.05, 0.1], ['0%', '5%', '10%'])

def fabphotofig(df_heatmap, df_photo, diet, session, clims=[[0,1], [0,1]],
                 keys_traces = ['pref1_cas_licks_forced', 'pref1_malt_licks_forced'],
                 keys_lats = ['pref1_cas_lats_all', 'pref1_malt_lats_all'],
                 keys_bars = ['pref1_cas_licks_peak', 'pref1_malt_licks_peak'],
                 event='Licks'):
    
    if diet == 'NR':
        rat='PPP1-7'
    else:
        rat='PPP1-4'
    
    gs = gridspec.GridSpec(2, 2, wspace=0.7, hspace=0.6, left=0.12, right=0.98)
    f = plt.figure(figsize=(3.2,3.2))
    
    heatmapFig(f, df_heatmap, gs, diet, session, 'PPP1-7', event=event, clims=clims)
    
    if event == 'Sipper':
        pppfig.averagetrace_sipper(f, gs, 0, 1, df_photo, diet, keys_traces, keys_lats, event=event)
    else:
        ax = f.add_subplot(gs[0,1])
        averagetrace(ax, df_photo, diet, keys_traces, event=event)
        
    ax = f.add_subplot(gs[1,1]) 
    peakbargraph(ax, df_photo, diet, keys_bars)
    
    return f