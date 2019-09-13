# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:37:11 2019

@author: jmc010
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

import numpy as np

# install at an anaconda command prompt using:  conda install -c anaconda dill
import dill

#Colors
green = mpl.colors.to_rgb('xkcd:kelly green')
light_green = mpl.colors.to_rgb('xkcd:light green')
almost_black = mpl.colors.to_rgb('#262626')

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
 
    col_gs = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs[:,0],
                                             height_ratios=[0.05,1],
                                             hspace=0.0)
    
    plots_gs = gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec=col_gs[1,0],
                                             width_ratios=[12,1],
                                             wspace=0.05)
    
    marker_gs = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=col_gs[0,0],
                                             width_ratios=[12,1],
                                             wspace=0.05)

    ax1 = f.add_subplot(plots_gs[0,0])
    ax, mesh = makeheatmap(ax1, data_cas, events=event_cas, ylabel='Casein trials')
    mesh.set_clim(clims)
    
    ax2 = f.add_subplot(plots_gs[1,0], sharex=ax1)
    ax, mesh = makeheatmap(ax2, data_malt, events=event_malt, ylabel='Malt. trials')
    mesh.set_clim(clims)
    
    ax0 = f.add_subplot(marker_gs[0,0], sharex=ax1)
    ax0.axis('off')
    if event == 'Sipper':
        ax0.plot(0,0, 'v', color='xkcd:silver')
        ax0.annotate(event, xy=(0, 0), xytext=(0,5), textcoords='offset points',
            ha='center', va='bottom')
    elif event == 'Licks':
        ax0.plot([0,5], [0,0], color='xkcd:silver', linewidth=3)
        ax0.annotate(event, xy=(2.5, 0), xytext=(0,5), textcoords='offset points',
            ha='center', va='bottom')
        
    ax1.set_xlim([-10,20])
    
    cbar_ax = f.add_subplot(plots_gs[0,1])   
    cbar = f.colorbar(mesh, cax=cbar_ax, ticks=[clims[0], 0, clims[1]])
    cbar_labels = ['{0:.0f}%'.format(clims[0]*100),
                   '0% \u0394F',
                   '{0:.0f}%'.format(clims[1]*100)]
    cbar.ax.set_yticklabels(cbar_labels)
    
    ax3 = f.add_subplot(plots_gs[2,0])
 
    shadedError(ax3, data_cas, linecolor=color[0], errorcolor=errorcolors[0])
    shadedError(ax3, data_malt, linecolor=color[1], errorcolor=errorcolors[1])
    
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

def makeheatmap(ax, data, events=None, ylabel='Trials'):
    ntrials = np.shape(data)[0]
    xvals = np.linspace(-9.9,20,300)
    yvals = np.arange(1, ntrials+2)
    xx, yy = np.meshgrid(xvals, yvals)
    
    mesh = ax.pcolormesh(xx, yy, data, cmap='jet', shading = 'flat')
    
    if events:
        ax.vlines(events, yvals[:-1], yvals[1:], color='w')
    else:
        print('No events')
        
    ax.set_ylabel(ylabel)
    ax.set_yticks([1, ntrials])
    ax.set_xticks([])
    ax.invert_yaxis()
    
    return ax, mesh

def shadedError(ax, yarray, linecolor='black', errorcolor = 'xkcd:silver', linewidth=1):
    yarray = np.array(yarray)
    y = np.mean(yarray, axis=0)
    yerror = np.std(yarray)/np.sqrt(len(yarray))
    x = np.arange(0, len(y))
    ax.plot(x, y, color=linecolor, linewidth=1)
    ax.fill_between(x, y-yerror, y+yerror, color=errorcolor, alpha=0.4)
    
    return ax

# get data, change pickle_folder if needed
pickle_folder = '..\\data\\'

pickle_in = open(pickle_folder + 'ppp_dfs_pref.pickle', 'rb')
df_behav, df_photo, df_reptraces, df_heatmap, df_reptraces_sip, df_heatmap_sip, longtrace = dill.load(pickle_in)


# initialize figure
gs = gridspec.GridSpec(2, 1, wspace=0.8, hspace=0.3, left=0.2, right=0.8, bottom=0.05, top=0.95)
f = plt.figure(figsize=(4,6))

diet='NR'
session='pref1'
rat='PPP1-7'
event ='Licks'
clims=[-0.05,0.17]
reverse=True
colors='control'

heatmapCol(f, df_heatmap, gs, diet, session, rat, event=event, clims=clims, reverse=reverse, colorgroup=colors)
f.savefig('EBBS_heatmap.jpg')