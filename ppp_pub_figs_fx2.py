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

from ppp_pub_figs_settings import *

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


def lickplot(ax, licks, sipper,
             ylabel=True,
             style='raster',
             showsipper=True,
             text_offset=30,
             lick_pos=3.5,
             sip_pos=1):        
    # Removes axes and spines
    jmfig.invisible_axes(ax)

    licks_x = [(x+10)*10 for x in licks]
    if style == 'histo':
        hist, bins = np.histogram(licks_x, bins=30, range=(0,300))
        center = (bins[:-1] + bins[1:]) / 2
        width = 1 * (bins[1] - bins[0])   
        ax.bar(center, hist, align='center', width=width, color='xkcd:silver')
    
    if style == 'raster':
        yvals = [lick_pos]*len(licks)
        ax.plot(licks_x,yvals,linestyle='None',marker='|',markersize=5, color='xkcd:silver')
        
    else:
        print('Not a valid style for plotting licks')

    if showsipper == True:
        sipper_x = (sipper+10)*10
        ax.plot(sipper_x, sip_pos, 'v', color='xkcd:silver')

    if ylabel == True:
        ax.annotate('Licks', xy=(licks_x[0]-text_offset,lick_pos), va='center', ha='right')
        ax.annotate('Sipper', xy=(sipper_x-text_offset, sip_pos), xytext=(0,5), textcoords='offset points',
                    ha='right', va='top')
    ax.set_ylim([min([lick_pos, sip_pos])-1, max([lick_pos, sip_pos])+1])


def repFig(ax, df, session, plot_licks=False, color=almost_black, yscale=True, xscale=True, legend=False):

# Plots data
    datauv = df[session+'_photo_uv']
    datablue = df[session+'_photo_blue']
    
    uv_color = jmfig.lighten_color(color, amount=0.3)
    
    #lines for blue/violet style
    uv_color='xkcd:azure'
    color='xkcd:amethyst'
    
    ax.plot(datauv, c=uv_color)
    ax.plot(datablue, c=color)
       
    # Adds x scale bar
    if xscale == True:
        y = ax.get_ylim()[0]
        ax.plot([251,300], [y, y], c=almost_black, linewidth=2)
        ax.annotate('5 s', xy=(276,y), xycoords='data',
                    xytext=(0,-5), textcoords='offset points',
                    ha='center',va='top')
    
    # Removes axes and spines
    jmfig.invisible_axes(ax)
    
    if yscale == True:
        l = 0.05
        y1 = [y for y in ax.get_yticks() if y>0][0]
        y2 = y1 + l        
        scale_label = '{0:.0f}% \u0394F'.format(l*100)
        ax.plot([50,50], [y1, y2], c=almost_black)
        ax.text(40, y1 + (l/2), scale_label, va='center', ha='right')

    if legend == True:
        ax.annotate('470 nm', xy=(310,datablue[299]), color=color, va='center')
        ax.annotate('405 nm', xy=(310,datauv[299]), color=uv_color, va='center')
    
    return ax

def reptracesFig(f, df, index, session, gs, gsx, gsy, title=False, color=almost_black):
    
    inner = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs[gsx,gsy],
                                             wspace=0.05, hspace=0,
                                             height_ratios=[1,8])    
    ax1 = f.add_subplot(inner[1,0])
    repFig(ax1, df.loc[index[0]], session, color=color, xscale=False)
    ax2 = f.add_subplot(inner[1,1], sharey=ax1)
    repFig(ax2, df.loc[index[1]], session, color=color, yscale=False, legend=True)

    ax3 = f.add_subplot(inner[0,0], sharex=ax1)
    lickplot(ax3, df.loc[index[0]][session+'_licks'], df.loc[index[0]][session+'_sipper'][0])
    ax4 = f.add_subplot(inner[0,1], sharey=ax3, sharex=ax2)
    lickplot(ax4, df.loc[index[1]][session+'_licks'], df.loc[index[1]][session+'_sipper'][0], ylabel=False)
    
    if title == True:
        ax3.set_title('Casein')
        ax4.set_title('Maltodextrin')

def makeheatmap(ax, data, events=None, ylabel='Trials'):
    ntrials = np.shape(data)[0]
    xvals = np.linspace(-9.9,20,300)
    yvals = np.arange(1, ntrials+2)
    xx, yy = np.meshgrid(xvals, yvals)
    
    mesh = ax.pcolormesh(xx, yy, data, cmap='YlGnBu', shading = 'flat')
    
    if events:
        ax.vlines(events, yvals[:-1], yvals[1:], color='w')
    else:
        print('No events')
        
    ax.set_ylabel(ylabel)
    ax.set_yticks([1, ntrials])
    ax.set_xticks([])
    ax.invert_yaxis()
    
    return ax, mesh

def heatmapFig(f, df, gs, gsx, gsy, session, event, rat, reverse=False, clims=[0,1]):
    
    data_cas = df[session+'_cas'][rat]
    data_malt = df[session+'_malt'][rat]
    event_cas = df[session+'_cas_event'][rat]
    event_malt = df[session+'_malt_event'][rat]
    
    if reverse:
            event_cas = [-event for event in event_cas]
            event_malt = [-event for event in event_malt]

    inner = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs[gsx,gsy],
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
        
    for ax in [ax1, ax2]:
        ax.set_xlim([-10,20])
    
    cbar_ax = f.add_subplot(inner[:,1])   
    cbar = f.colorbar(mesh, cax=cbar_ax, ticks=[clims[0], 0, clims[1]])
    cbar_labels = ['{0:.0f}%'.format(clims[0]*100),
                   '0% \u0394F',
                   '{0:.0f}%'.format(clims[1]*100)]
    cbar.ax.set_yticklabels(cbar_labels)

def averagetrace(ax, df, diet, keys, event='',
                 color=[almost_black, 'xkcd:bluish grey'],
                 errorcolors=['xkcd:silver', 'xkcd:silver']):

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
    
# Adds legend
    ax.annotate('Casein', xy=(100,100), xycoords='axes fraction',
                ha='right', va='center')

def averagetrace_sipper(f, gs, gsx, gsy, df, diet, keys, keys_lats, event='',
                 color=[almost_black, 'xkcd:bluish grey'],
                 errorcolors=['xkcd:silver', 'xkcd:silver']):
    
    inner = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs[gsx,gsy],
                                             wspace=0.05, hspace=0.2,
                                             height_ratios=[1,7])    
   
    df = df.xs(diet, level=1)
    
    ax1 = f.add_subplot(inner[1,0])
    
    jmfig.shadedError(ax1, df[keys[0]], linecolor=color[0], errorcolor=errorcolors[0])
    jmfig.shadedError(ax1, df[keys[1]], linecolor=color[1], errorcolor=errorcolors[1])
    
    #ax.legend(['Casein', 'Maltodextrin'], fancybox=True)    
    ax1.axis('off')
    
    arrow_y = ax1.get_ylim()[1]
    ax1.plot([100], [arrow_y], 'v', color='xkcd:silver')
    ax1.annotate(event, xy=(100, arrow_y), xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom')

    y = [y for y in ax1.get_yticks() if y>0][:2]
    l = y[1] - y[0]
    scale_label = '{0:.0f}% \u0394F'.format(l*100)
    ax1.plot([50,50], [y[0], y[1]], c=almost_black)
    ax1.text(40, y[0]+(l/2), scale_label, va='center', ha='right')
   
    y = ax1.get_ylim()[0]
    ax1.plot([251,300], [y, y], c=almost_black, linewidth=2)
    ax1.annotate('5 s', xy=(276,y), xycoords='data',
                xytext=(0,-5), textcoords='offset points',
                ha='center',va='top')
    
    ax_lats = f.add_subplot(inner[0,0])
    
    lat_data = []
    for key in keys_lats:
        lats = jmf.flatten_list(df[key])
        lat_data.append([l for l in lats if not np.isnan(l)])
        
    stats = jmfig.get_violinstats(lat_data, points=200)
    
    x=stats[0]['coords']
    y=stats[0]['vals']
    
    x2=stats[1]['coords']
    y2=-stats[1]['vals']
    
    ax_lats.fill_between(x, y, color=color[0])
    ax_lats.fill_between(x2, y2, color=color[1])
    ax_lats.set_xlim([-10, 20])
    jmfig.invisible_axes(ax_lats)

    ax_lats.annotate('Latencies', xy=(0,0), ha='right', va='center')

def peakbargraph(ax, df, diet, keys, bar_colors=['xkcd:silver', 'w'], sc_color='w'):
    
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

def mainphotoFig(df_reptraces, df_heatmap, df_photo, session='pref1', clims=[[0,1], [0,1]],
                 keys_traces = ['cas1_licks_forced', 'malt1_licks_forced'],
                 keys_lats = ['pref1_cas_lats_all', 'pref1_malt_lats_all'],
                 keys_bars = ['cas1_licks_peak', 'malt1_licks_peak'],
                 event='Licks',
                 dietswitch=False):

    gs = gridspec.GridSpec(2, 7, width_ratios=[1.3,0.3,1,0.05,1,0.05,0.4], wspace=0.3, hspace=0.6, left=0.04, right=0.98)
    f = plt.figure(figsize=(7.2,5))
    
    rowcolors = [[almost_black, 'xkcd:bluish grey'], [green, light_green]]
    rowcolors_bar = [['xkcd:silver', 'w'], [green, light_green]]
    
    if dietswitch == True:
        rowcolors.reverse()
        rowcolors_bar.reverse()

    # Non-restricted figures, row 0
    reptracesFig(f, df_reptraces, ['NR_cas', 'NR_malt'], session, gs, 0, 0, title=True, color=rowcolors[0][0])
    
    heatmapFig(f, df_heatmap, gs, 0, 2, session, event, 'PPP1-7', clims=clims[0])

    if event == 'Sipper':
        averagetrace_sipper(f, gs, 0, 4, df_photo, 'NR', keys_traces, keys_lats, event=event, color=rowcolors[0])
    else:
        ax3 = f.add_subplot(gs[0,4])
        averagetrace(ax3, df_photo, 'NR', keys_traces, event=event, color=rowcolors[0])
        
    ax7 = f.add_subplot(gs[0,6]) 
    peakbargraph(ax7, df_photo, 'NR', keys_bars, bar_colors=rowcolors_bar[0], sc_color='w')
   
    # Protein-restricted figures, row 1
    reptracesFig(f, df_reptraces, ['PR_cas', 'PR_malt'], session, gs, 1, 0, color=rowcolors[1][0])
    heatmapFig(f, df_heatmap, gs, 1, 2, session, event, 'PPP1-4', clims=clims[1])

    if event == 'Sipper':
            averagetrace_sipper(f, gs, 1, 4, df_photo, 'PR', keys_traces, keys_lats, event=event, color=rowcolors[1])
    else:
        ax6 = f.add_subplot(gs[1,4])
        averagetrace(ax6, df_photo, 'PR', keys_traces, event=event, color=rowcolors[1])
            
    ax8 = f.add_subplot(gs[1,6])
    peakbargraph(ax8, df_photo, 'PR', keys_bars, bar_colors=rowcolors_bar[1], sc_color=almost_black)
     
    return f


def reduced_photofig(df_photo, df_behav, session=2, event='Licks', dietswitch=True,
                     keys_traces = ['pref2_cas_licks_forced', 'pref2_malt_licks_forced'],
                     keys_bars = ['pref2_cas_licks_peak', 'pref2_malt_licks_peak']):
    
   
    
    rowcolors = [[almost_black, 'xkcd:bluish grey'], [green, light_green]]
    rowcolors_bar = [['xkcd:silver', 'w'], [green, light_green]]
    
    if dietswitch == True:
        rowcolors.reverse()
        rowcolors_bar.reverse()
    
    gs_main = gridspec.GridSpec(2, 1, hspace=0.4, left=0.04, right=0.98)
    f = plt.figure(figsize=(7.2,5))
        
    behav_panel = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[0,0],
                                         wspace=0.65)    
    
    axes = []
    for i, ax in enumerate(behav_panel):
        axes.append(f.add_subplot(ax))

    pref_behav_fig(axes, df_behav, df_photo, prefsession=session, dietswitch=dietswitch)
    
    barpanelwidth=0.25
    
    photo_panel = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[1,0],
                                         width_ratios=[1, barpanelwidth, 1, barpanelwidth],
                                         wspace=0.5)
    
    ax1 = f.add_subplot(photo_panel[0,0])
    averagetrace(ax1, df_photo, 'NR', keys_traces, event=event, color=rowcolors[0])
    
    ax2 = f.add_subplot(photo_panel[0,1])
    peakbargraph(ax2, df_photo, 'NR', keys_bars, bar_colors=rowcolors_bar[0], sc_color=almost_black)
    
    ax3 = f.add_subplot(photo_panel[0,2])
    averagetrace(ax3, df_photo, 'PR', keys_traces, event=event, color=rowcolors[1])
    
    ax4 = f.add_subplot(photo_panel[0,3])
    peakbargraph(ax4, df_photo, 'PR', keys_bars, bar_colors=rowcolors_bar[1], sc_color=almost_black)
    
    return f




    