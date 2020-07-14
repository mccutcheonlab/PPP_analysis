# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:16:41 2018

@author: jaimeHP
"""
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import trompy as tp

from ppp_pub_figs_settings import *

def pref_behav_fig(ax, df_behav, df_photo, prefsession=1, dietswitch=False, barlabeloffset=[], gs=[], f=[], scattersize=50):

    forced_cas_key = 'pref' + str(prefsession) + '_cas_forced'
    forced_malt_key = 'pref' + str(prefsession) + '_malt_forced'
    free_cas_key = 'pref' + str(prefsession) + '_cas_free'
    free_malt_key = 'pref' + str(prefsession) + '_malt_free'
    choice_cas_key = 'pref' + str(prefsession) + '_ncas'
    choice_malt_key = 'pref' + str(prefsession) + '_nmalt'
    lat_cas_key = 'pref' + str(prefsession) + '_cas_lats_fromsip'
    lat_malt_key = 'pref' + str(prefsession) + '_malt_lats_fromsip'
    pref_key = 'pref' + str(prefsession)

    if len(barlabeloffset) < 4:
        barlabeloffset = [0.025, 0.025, 0.025, 0.025]

    if dietswitch == True:
        grouplabel=['NR \u2192 PR', 'PR \u2192 NR']
        barfacecolor = [col['pr_cas'], col['pr_malt'], col['nr_cas'], col['nr_malt']]
    else:
        grouplabel=['NR', 'PR']
        barfacecolor = [col['nr_cas'], col['nr_malt'], col['pr_cas'], col['pr_malt']]
    
#panel 1 - forced choice licks    
    x = [[df_behav.xs('NR', level=1)[forced_cas_key], df_behav.xs('NR', level=1)[forced_malt_key]],
         [df_behav.xs('PR', level=1)[forced_cas_key], df_behav.xs('PR', level=1)[forced_malt_key]]]
    tp.barscatter(x, paired=True, unequal=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = barfacecolor,
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel=grouplabel,
                 barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                 barlabeloffset=barlabeloffset[0],
                 scattersize = scattersize,
                 ylim=[-50,1050],
                 xfontsize=6,
                 ax=ax[0])

    ax[0].set_ylabel('Licks')
    ax[0].set_yticks([0, 500, 1000])

#panel 2 - latency for forced choice
    x = [[df_photo.xs('NR', level=1)[lat_cas_key], df_photo.xs('NR', level=1)[lat_malt_key]],
         [df_photo.xs('PR', level=1)[lat_cas_key], df_photo.xs('PR', level=1)[lat_malt_key]]]
    tp.barscatter(x, paired=True, unequal=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = barfacecolor,
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel=grouplabel,
                 barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                 barlabeloffset=barlabeloffset[1],
                 scattersize = scattersize,
                 ylim=[-0.5,10],
                 xfontsize=6,
                 ax=ax[1])
    ax[1].set_ylabel('Latency (s)')
    ax[1].set_yticks([0, 2, 4, 6, 8, 10])

#panel 2 - free choice licks
    x = [[df_behav.xs('NR', level=1)[free_cas_key], df_behav.xs('NR', level=1)[free_malt_key]],
         [df_behav.xs('PR', level=1)[free_cas_key], df_behav.xs('PR', level=1)[free_malt_key]]]
    tp.barscatter(x, paired=True, unequal=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = barfacecolor,
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel = grouplabel,
                 barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                 barlabeloffset=barlabeloffset[2],
                 scattersize = scattersize,
                 ylim=[-50, 800],
                 xfontsize=6,
                 ax=ax[2])

    ax[2].set_ylabel('Licks')
    ax[2].set_yticks([0, 250, 500, 750])

    if prefsession == 1:
        ax[3].axis('off')
    
        x = [df_behav.xs('NR', level=1)[pref_key], df_behav.xs('PR', level=1)[pref_key]]
        tp.barscatter(x, paired=False, unequal=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [barfacecolor[0], barfacecolor[2]],
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel = grouplabel,
                 grouplabeloffset = 0,
                 barlabels=[],
                 barlabeloffset=barlabeloffset[3],
                 scattersize = scattersize/3,
                 ylim=[-0.03, 1.1],
                 barwidth = .75,
                 groupwidth = .5,
                 xfontsize=6,
                 spaced=True,
                 xspace=0.2,
                 ax=ax[4])
            
        ax[4].plot(ax[4].get_xlim(), [0.5, 0.5], linestyle='dashed',color='k', alpha=0.3)
        ax[4].set_ylabel('Casein preference')
        ax[4].set_yticks([0, 0.5, 1])
    else:
        inner = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[1,1],
                                             width_ratios=[0.3,1,0.2])
        print(f)
        print(type(f))
        new_ax = f.add_subplot(inner[1])
        x = [df_behav.xs('NR', level=1)[pref_key], df_behav.xs('PR', level=1)[pref_key]]
        tp.barscatter(x, paired=False, unequal=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [barfacecolor[0], barfacecolor[2]],
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabeloffset = 0,
                 scattersize = scattersize/3,
                 ylim=[-0.03, 1.1],
                 barwidth = .75,
                 groupwidth = .5,
                 xfontsize=6,
                 spaced=True,
                 xspace=0.2,
                 ax=new_ax)
        new_ax.plot(new_ax.get_xlim(), [0.5, 0.5], linestyle='dashed',color='k', alpha=0.3)
        new_ax.set_ylabel('Casein preference')
        new_ax.set_yticks([0, 0.5, 1])
        new_ax.set_ylim([-0.05, 1.1])

def fig1_photo(df_heatmap, df_photo, diet, session, clims=[[0,1], [0,1]],
                 peaktype='average', epoch=[100,149],
                 keys_traces = ['pref1_cas_licks_forced', 'pref1_malt_licks_forced'],
                 keys_lats = ['pref1_cas_lats_all', 'pref1_malt_lats_all'],
                 event='Licks',
                 scattersize=50):
    
    if diet == 'NR':
        rat='PPP1-7'
        colors = 'control'
    else:
        rat='PPP1-4'
        colors = 'exptl'
    
    if event == 'Sipper':
        reverse = False
    else:
        reverse = True
    
    gs = gridspec.GridSpec(2, 2, wspace=0.8, width_ratios=[1, 0.8], hspace=0.3, left=0.12, right=0.90) # change back to right=0.98 if not showing t-vals
    f = plt.figure(figsize=(3.2,3.2))
    
    heatmapCol(f, df_heatmap, gs, diet, session, rat, event=event, clims=clims, reverse=reverse, colorgroup=colors)
    
    averageCol(f, df_photo, gs, diet, keys_traces, keys_lats, peaktype=peaktype, epoch=epoch,  event=event, scattersize=scattersize)
        
    return f

def fig2_photo(df_photo, peaktype='average', epoch=[100,149],
               keys_traces = ['pref2_cas_licks_forced', 'pref2_malt_licks_forced'],
               peakkey='peakdiff_2',
               event='Licks',
               plot_tvals=False,
               scattersize=50):
    
    gs = gridspec.GridSpec(2, 2, hspace=0.6, wspace=0.7, left=0.20, right=0.90)
    f = plt.figure(figsize=(3.2,3.2))
    
    ax1 = f.add_subplot(gs[0,0])
    averagetrace(ax1, df_photo, 'NR', keys_traces, event=event, colorgroup='exptl')
    ax1.set_title('NR \u2192 PR')
    
    ax2 = f.add_subplot(gs[0,1], sharey=ax1)
    averagetrace(ax2, df_photo, 'PR', keys_traces, event=event, ylabel=False)
    ax2.set_ylim([-1.5, 2.5])
    ax2.set_title('PR \u2192 NR')
    
    
    if plot_tvals:
        innerNR = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[1,0],
                                         width_ratios=[1,0.5],
                                         hspace=0.0)
        
        ax31 = f.add_subplot(innerNR[0,0])
        peakbargraph(ax31, df_photo, 'NR', keys_traces, peaktype=peaktype, epoch=epoch,
                         colorgroup='exptl',
                         grouplabeloffset=0.1)
        ax31.set_ylim([-3, 7.5])

        ax32 = f.add_subplot(innerNR[0,1])
        tvaluegraph(ax32, df_photo, 'NR', peakkey)
        ax32.set_ylabel('')
        ax32.set_ylim([-8.5, 8.5])
        
        innerPR = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[1,1],
                                         width_ratios=[1,0.5],
                                         hspace=0.0)
        
        ax41 = f.add_subplot(innerPR[0,0])
        peakbargraph(ax41, df_photo, 'PR', keys_traces, peaktype=peaktype, epoch=epoch,
                         colorgroup='exptl',
                         ylabel=False, 
                         grouplabeloffset=0.1)
        
        ax41.set_ylim([-3, 7.5])

        ax42 = f.add_subplot(innerPR[0,1])
        tvaluegraph(ax42, df_photo, 'PR', peakkey)
        ax42.set_ylim([-8.5, 8.5])

    else:
        ax3 = f.add_subplot(gs[1,0]) 
        peakbargraph(ax3, df_photo, 'NR', keys_traces, peaktype=peaktype, epoch=epoch,
                     colorgroup='exptl',
                     grouplabeloffset=0.1,
                     scattersize=scattersize)
        ax3.set_ylim([-3, 7.5])
 
        ax4 = f.add_subplot(gs[1,1], sharey=ax3) 
        peakbargraph(ax4, df_photo, 'PR', keys_traces, peaktype=peaktype, epoch=epoch,
                      ylabel=False, grouplabeloffset=0.1,
                      scattersize=scattersize)
        ax4.set_ylim([-3, 7.5])

    return f

def makeheatmap(ax, data, events=None, ylabel='Trials', xscalebar=False):
    ntrials = np.shape(data)[0]
    xvals = np.linspace(-9.9,20,300)
    yvals = np.arange(1, ntrials+2)
    xx, yy = np.meshgrid(xvals, yvals)
    
    mesh = ax.pcolormesh(xx, yy, data, cmap=heatmap_color_scheme, shading = 'flat')
    
    events = [-e for e in events]
    
    if events:
        ax.vlines(events, yvals[:-1], yvals[1:], color='w')
    else:
        print('No events')
        
    ax.set_ylabel(ylabel, rotation=270, labelpad=10)
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

def heatmapCol(f, df, gs, diet, session, rat, event='', reverse=False, clims=[0,1], colorgroup='control'):
    
    if colorgroup == 'control':
        color = [almost_black, 'xkcd:bluish grey']
        errorcolors = ['xkcd:silver', 'xkcd:silver']
    else:
        color = [col['pr_cas'], col['pr_malt']]
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
    ax, mesh = makeheatmap(ax1, data_cas, events=event_cas, ylabel='Casein trials \u2192')
    mesh.set_clim(clims)
    
    ax2 = f.add_subplot(plots_gs[1,0], sharex=ax1)
    ax, mesh = makeheatmap(ax2, data_malt, events=event_malt, ylabel='Malt. trials \u2192', xscalebar=True)
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
    
    scale_label = '{0:.0f} Z-score'.format(l)
    
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
        color=[col['pr_cas'], col['pr_malt']]
        errorcolors=['xkcd:silver', 'xkcd:silver']
# Selects diet group to plot                
    df = df.xs(diet, level=1)
    
# Removes empty arrays
    cas = [trace for trace in df[keys[0]] if len(trace) > 0]
    malt = [trace for trace in df[keys[1]] if len(trace) > 0]

# Plots casein and maltodextrin shaded erros
    tp.shadedError(ax, cas, linecolor=color[0], errorcolor=errorcolors[0], linewidth=2)
    tp.shadedError(ax, malt, linecolor=color[1], errorcolor=errorcolors[1], linewidth=2)
    
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
        
    

def peakbargraph(ax, df, diet, keys, peaktype='average', epoch=[100, 149],
                 sc_color='w', colorgroup='control', ylabel=True,
                 ylim=[-0.05, 0.1], grouplabeloffset=0,
                 scattersize=50):
    
    if colorgroup == 'control':
        bar_colors=['xkcd:silver', 'w']
    else:
        bar_colors=[col['pr_cas'], col['pr_malt']]
    
    epochrange = range(epoch[0], epoch[1])
    
    df = df.xs(diet, level=1)
    
    if peaktype == 'average':
        a1 = [np.mean(rat[epochrange]) for rat in df[keys[0]]]
        a2 = [np.mean(rat[epochrange]) for rat in df[keys[1]]]
        ylab = 'Mean Z-Score'
        
    elif peaktype == 'auc':
        a1 = [np.trapz(rat[epochrange])/10 for rat in df[keys[0]]]
        a2 = [np.trapz(rat[epochrange])/10 for rat in df[keys[1]]]
        ylab = 'AUC'
        
    elif peaktype == 'calcd': # for already calculated
        a1 = df[keys[0]]
        a2 = df[keys[1]]
        ylab = 'AUC'
        
    a = [a1, a2]
    
    ax, x, _, _ = tp.barscatter(a, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = bar_colors,
                 scatteredgecolor = [almost_black],
                 scatterlinecolor = almost_black,
                 scatterfacecolor = [sc_color],
                 grouplabel=['Cas', 'Malt'],
                 grouplabeloffset=grouplabeloffset,
                 scattersize = scattersize,
                 xfontsize=6,
                 barwidth=0.75,
                 ax=ax)

#    ax.set_yticks([-0.05,0,0.05, 0.1])
#    ax.set_yticklabels(['5%', '0%', '5%', '10%'])
#    ax.set_ylim(ylim)
    
    if ylabel:
        ax.set_ylabel(ylab)

def tvaluegraph(ax, df, diet, key):
    #ax.set_zorder(-20)
    
    df = df.xs(diet, level=1)
    data = df[key]
    
    xvals, yvals = tp.xyspacer(ax, 1, list(data), bindist=10, space=0.4)

    for x, y in zip(xvals, yvals):
        if np.abs(y) > 2.02:
            ax.plot(x, y, 'o', markerfacecolor='xkcd:light grey', markeredgecolor='k', markersize = 5)
        else:
            ax.plot(x, y, 'o', markerfacecolor='w', markeredgecolor='k', markersize = 5, zorder=20)
    ax.set_ylim([-7.5, 7.5])
    ax.set_yticks([-5, 0, 5])
    ax.set_xlim([0, 2])
    ax.set_xticks([])
    ax.plot([0, 2], [2.02, 2.02], linestyle='dashed',color='k', alpha=0.5, zorder=-20)
    ax.plot([0, 2], [-2.02, -2.02], linestyle='dashed',color='k', alpha=0.5, zorder=-20)
    
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    
    ax.set_ylabel('t value (Casein vs. Malt.)')
    ax.spines['bottom'].set_position('zero')
    
    
def averageCol(f, df_photo, gs, diet, keys_traces, keys_lats, peaktype='average', epoch=[100,149], event='',
               plot_tvals=False,
               scattersize=50):
    
    inner = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs[0,1],
                                             height_ratios=[0.15,1],
                                             hspace=0.0)
    
    if diet == 'NR':
        colors = 'control'
    else:
        colors = 'exptl'
    
    ax1 = f.add_subplot(inner[1,0])
    averagetrace(ax1, df_photo, diet, keys_traces, event=event, fullaxis=True, colorgroup=colors)
    ax1.set_ylim([-1.5, 3.2])
    for xval in epoch:
        ax1.axvline(xval, linestyle='--', color='k', alpha=0.3)
    
    ax0 = f.add_subplot(inner[0,0], sharex=ax1)
    ax0.axis('off')
    if event == 'Sipper':
        ax0.plot(100,0, 'v', color='xkcd:silver')
        ax0.annotate(event, xy=(100, 0), xytext=(0,5), textcoords='offset points',
            ha='center', va='bottom')
    elif event == 'Licks':
        ax0.plot([100,150], [0,0], color='xkcd:silver', linewidth=3)
        ax0.annotate(event, xy=(125, 0), xytext=(0,5), textcoords='offset points',
            ha='center', va='bottom')
        
    if plot_tvals:
        inner = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[1,1],
                                         width_ratios=[1,0.5],
                                         hspace=0.0)
        
        ax2 = f.add_subplot(inner[0,0]) 
        peakbargraph(ax2, df_photo, diet, keys_traces, peaktype=peaktype, epoch=epoch,
                     colorgroup=colors, ylim=[-0.04,0.12], grouplabeloffset=0.07,
                     scattersize=scattersize)
        ax2.set_ylim([-3, 7.5])
        
        ax3 = f.add_subplot(inner[0,1])
        tvaluegraph(ax3, df_photo, diet, key='peakdiff_1')
        
    else:
    
        ax2 = f.add_subplot(gs[1,1]) 
        peakbargraph(ax2, df_photo, diet, keys_traces, peaktype=peaktype, epoch=epoch,
                     colorgroup=colors, ylim=[-0.04,0.12], grouplabeloffset=0.07,
                     scattersize=scattersize)
        # ax2.set_ylim([-3, 7.5]) # for epoch=[100:119]
        ax2.set_ylim([-15, 30]) # for epoch=[100:149]

        

# To make summary figure

def summary_subfig_bars(ax, df, keys, scattersize=50):
    
    df_NR = df.xs('NR', level=1)
    df_PR = df.xs('PR', level=1)
       
    a = [[df_NR[keys[0]], df_NR[keys[1]], df_NR[keys[2]]],
          [df_PR[keys[0]], df_PR[keys[1]], df_PR[keys[2]]]]
    
    x = tp.data2obj2D(a)
    
    cols = ['xkcd:silver', col['pr_cas']]
    
    xlabels = ['NR \u2192 PR', 'PR \u2192 NR']
    
    tp.barscatter(x, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [cols[0], cols[1], cols[1], cols[1], cols[0], cols[0]],
                 scatteredgecolor = [almost_black],
                 scatterlinecolor = almost_black,
                 scattersize = scattersize,
                 grouplabel = xlabels,
                 ax=ax)
    
def summary_subfig_bars_new(ax, df, diet, keys, datatype='behav', scattersize=50):
    
    df = df.xs(diet, level=1)
       
    a = [df[keys[0]], df[keys[1]], df[keys[2]]]
    
    cols = ['xkcd:silver', col['pr_cas']]
    
    if diet == 'PR':
        cols.reverse()
        
    if datatype == 'behav':
        barlabeloffset=0.05
    else:
        barlabeloffset=0.35
    
    tp.barscatter(a, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [cols[0], cols[1], cols[1]],
                 scatteredgecolor = [almost_black],
                 scatterlinecolor = almost_black,
                 scattersize = scattersize,
                 barlabels=['1', '2', '3'],
                 barlabeloffset=barlabeloffset,
                 xfontsize=6,
                 ax=ax)
    
    if datatype == 'behav':
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['0', '0.5', '1'])
    else:
        ax.set_ylim([-2, 4.5])
        ax.set_yticks([-2, 0, 2, 4])

def summary_subfig_casmalt(ax, df, diet, keys, epoch=[100,149]):
    
    epochrange = range(epoch[0], epoch[1])
    
    cols = ['xkcd:silver', col['pr_cas']]
    
    if diet == 'NR':
        cols = ['xkcd:silver', col['pr_cas'], col['pr_cas']]
    else:
        cols = [col['pr_cas'], 'xkcd:silver', 'xkcd:silver']
        
    df = df.xs(diet, level=1)
    
    cas_auc = []
    malt_auc = []
    
    for key in keys:
        cas_auc.append([np.trapz(x[epochrange])/10 for x in df[key[0]]])
        malt_auc.append([np.trapz(x[epochrange])/10 for x in df[key[1]]])
    
    xvals = [1,2,3]
    
    cas_data = [np.mean(day) for day in cas_auc]
    cas_sem = [np.std(day)/np.sqrt(len(day)) for day in cas_auc]
    
    print(cas_auc)

    malt_data = [np.mean(day) for day in malt_auc]
    malt_sem = [np.std(day)/np.sqrt(len(day)) for day in malt_auc] 
        
    
        
    # print(malt_auc)
    
    
    ax.errorbar(xvals, malt_data, yerr=malt_sem, capsize=0, c=almost_black, linewidth=0.75, zorder=-1)
    ax.scatter(xvals, malt_data, marker='o', c='white', edgecolors='k', s=20)
    
    ax.errorbar(xvals, cas_data, yerr=cas_sem, capsize=0, c=almost_black, linewidth=0.75, zorder=-1)
    ax.scatter(xvals, cas_data, marker='o', c=cols, edgecolors='k', s=20)
    
    ax.set_ylabel('Z-score AUC')
    # ax.set_ylim([0, 5.5])
    
    ax.set_xticks([])
    for x in xvals:
        #ax.text(x, 0.05, str(x), va='top', ha='center', fontsize=8, transform=ax.transAxes)
        ax.text(x, -0.25, str(x), va='top', ha='center', fontsize=6)
    # ax.set_xlim([0.5, 3.5])
    

def find_delta(df, keys_in, epoch=[100,149]):
    
    epochrange = range(epoch[0], epoch[1])
    
    keys_out = ['delta_1', 'delta_2', 'delta_3']
        
    for k_in, k_out in zip(keys_in, keys_out):
        cas_auc = [np.trapz(x[epochrange])/10 for x in df[k_in[0]]]
        malt_auc = [np.trapz(x[epochrange])/10 for x in df[k_in[1]]]
        df[k_out] = [c-m for c, m in zip(cas_auc, malt_auc)]
    
    return df


def summary_subfig_correl(ax, df_behav, df_photo, diet, use_zscore_diff=True):
    
    dfy = df_behav.xs(diet, level=1)
    dfx = df_photo.xs(diet, level=1)
        
    yvals = [dfy['pref1'], dfy['pref2'], dfy['pref3']]
    if use_zscore_diff:
        xvals = [dfx['delta_1'], dfx['delta_2'], dfx['delta_3']]
    else:   
        xvals = [dfx['peakdiff_1'], dfx['peakdiff_2'], dfx['peakdiff_3']]
    
    yvals_mean = np.mean(yvals, axis=1)
    y_sem = np.std(yvals, axis=1)/np.sqrt(len(yvals))
    
    xvals_mean = np.mean(xvals, axis=1)
    x_sem = np.std(xvals, axis=1)/np.sqrt(len(xvals))

#    ax.plot(xvals, yvals, color='k', alpha=0.2)
    
    ax.errorbar(xvals_mean, yvals_mean, yerr=y_sem, xerr=x_sem, capsize=0, c=almost_black, linewidth=0.75, zorder=-1)
    ax.scatter(xvals_mean, yvals_mean, marker='o', c=['w', 'grey', 'k'], edgecolors='k', s=20)
    
    ax.set_ylabel('Protein preference')
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0, 0.5, 1.0]) 
    ax.set_yticklabels(['0', '0.5', '1'])

    if use_zscore_diff:
        ax.set_xlabel('Diff. in z-score (Casein - Malt.)')
        # ax.set_xlim([-2.9, 2.9])
        ax.set_xticks([-2, -1, 0, 1, 2])
    else:
        ax.set_xlabel('T-value (Casein vs. Malt.)')
        ax.set_xlim([-5.9, 5.9])
        ax.set_xticks([-4, -2, 0, 2, 4])
    
    ax.plot(ax.get_xlim(), [0.5, 0.5], linestyle='dashed',color='k', alpha=0.2)
    ax.plot([0, 0], [-0.1, 1.1], linestyle='dashed',color='k', alpha=0.2)

def makesummaryFig(df_behav, df_photo, peaktype='auc', epoch=[100, 149], use_zscore_diff=True,
                   scattersize=50):
    gs = gridspec.GridSpec(2, 2, wspace=0.5, hspace=0.4, bottom=0.1)
    mpl.rcParams['figure.subplot.left'] = 0.10
    mpl.rcParams['figure.subplot.top'] = 0.85
    mpl.rcParams['axes.labelpad'] = 4
    f = plt.figure(figsize=(5,4))
    
    ax0 = f.add_subplot(gs[0, 0])
    summary_subfig_bars(ax0, df_behav, ['pref1', 'pref2', 'pref3'],
                        scattersize=scattersize)
    ax0.set_ylabel('Protein preference')
    ax0.set_ylim([-0.1, 1.2])
    ax0.set_yticks([0, 0.5, 1.0]) 
    ax0.set_yticklabels(['0', '0.5', '1'])
    ax0.set_title('Behaviour')

    ax0.plot(ax0.get_xlim(), [0.5, 0.5], linestyle='dashed',color='k', alpha=0.5, zorder=-20)
    
    ax1 = f.add_subplot(gs[0, 1])
    
    summary_photo_keys = [['pref1_cas_licks_forced', 'pref1_malt_licks_forced'],
                          ['pref2_cas_licks_forced', 'pref2_malt_licks_forced'],
                          ['pref3_cas_licks_forced', 'pref3_malt_licks_forced']]
    
    df_delta = find_delta(df_photo, summary_photo_keys, epoch=epoch)
    
    if use_zscore_diff:
        summary_subfig_bars(ax1, df_delta, ['delta_1', 'delta_2', 'delta_3'],
                            scattersize=scattersize)
        ax1.set_ylabel('Diff. in z-score (Casein - Malt.)')
    else:
        summary_subfig_bars(ax1, df_photo, ['peakdiff_1', 'peakdiff_2', 'peakdiff_3'],
                            scattersize=scattersize)
        ax1.set_ylabel('T-value (Casein vs. Malt.)')
        xlims=ax1.get_xlim()
        ax1.plot(xlims, [2.02, 2.02], linestyle='dashed',color='k', alpha=0.3, zorder=-20)
        ax1.plot(xlims, [-2.02, -2.02], linestyle='dashed',color='k', alpha=0.3, zorder=-20)

    
#    ax1.set_ylim([-0.035, 0.11])
    #ax1.set_yticks([-20, 0, 20, 40])
    ax1.set_title('Photometry')
    
    ax2 = f.add_subplot(gs[1,0])
    summary_subfig_correl(ax2, df_behav, df_delta, 'NR', use_zscore_diff=use_zscore_diff)
    ax2.set_title('NR \u2192 PR rats')
    
    ax3 = f.add_subplot(gs[1,1])
    summary_subfig_correl(ax3, df_behav, df_delta, 'PR', use_zscore_diff=use_zscore_diff)
    ax3.set_title('PR \u2192 NR rats')
    
    return f

def makesummaryFig_new(df_behav, df_photo, peaktype='auc', epoch=[100, 149], use_zscore_diff=True,
                                scattersize=50):
    print('Bløah')
    
    summary_photo_keys = [['pref1_cas_licks_forced', 'pref1_malt_licks_forced'],
                      ['pref2_cas_licks_forced', 'pref2_malt_licks_forced'],
                      ['pref3_cas_licks_forced', 'pref3_malt_licks_forced']]
    
    df_delta = find_delta(df_photo, summary_photo_keys, epoch=epoch)
    
    gs = gridspec.GridSpec(2, 3, wspace=0.7, hspace=0.6, bottom=0.1, width_ratios=[1,1,1.8])
    mpl.rcParams['figure.subplot.left'] = 0.10
    mpl.rcParams['figure.subplot.top'] = 0.85
    mpl.rcParams['axes.labelpad'] = 4
    f = plt.figure(figsize=(5,4))
    
    ax0 = f.add_subplot(gs[0, 0])
    summary_subfig_bars_new(ax0, df_behav, 'NR', ['pref1', 'pref2', 'pref3'],
                    scattersize=scattersize)
    ax0.set_ylabel('Protein preference')
    
    ax1 = f.add_subplot(gs[0, 1])
    # summary_subfig_bars_new(ax1, df_delta, 'NR', ['delta_1', 'delta_2', 'delta_3'],
    #                         scattersize=scattersize, datatype='photo')
    # ax1.set_ylabel('Diff. in z-score (Casein - Malt.)')
    
    summary_subfig_casmalt(ax1, df_photo, 'NR', summary_photo_keys, epoch=epoch)
    
    ax2 = f.add_subplot(gs[0,2])
    summary_subfig_correl(ax2, df_behav, df_delta, 'NR', use_zscore_diff=use_zscore_diff)
    
    
    ax3 = f.add_subplot(gs[1, 0])
    summary_subfig_bars_new(ax3, df_behav, 'PR', ['pref1', 'pref2', 'pref3'],
                    scattersize=scattersize)
    ax3.set_ylabel('Protein preference')
    
    
    ax4 = f.add_subplot(gs[1, 1])
    # summary_subfig_bars_new(ax4, df_delta, 'PR', ['delta_1', 'delta_2', 'delta_3'],
    #                         scattersize=scattersize, datatype='photo')
    # ax4.set_ylabel('Diff. in z-score (Casein - Malt.)')
    
    summary_subfig_casmalt(ax4, df_photo, 'PR', summary_photo_keys, epoch=epoch)
    
    ax5 = f.add_subplot(gs[1,2], sharex=ax2)
    summary_subfig_correl(ax5, df_behav, df_delta, 'PR', use_zscore_diff=use_zscore_diff)

    # ax0.set_title('NR \u2192 PR rats')
    # ax3.set_title('PR \u2192 NR rats')
    
    return f

def make_fig2_and_3(df_behav, df_photo, diet, dietswitch=False,
                      peaktype='auc', epoch=[], peakkey='peakdiff_2',
                      scattersize=50):
    
    if diet == 'PR': # takes into account diet switch
        colorgroup='control'
    else:
        colorgroup='exptl'
    
    
    f = plt.figure(figsize=(7.2, 5.5), constrained_layout=False)
    gs=f.add_gridspec(nrows=2, ncols=6, left=0.10, right=0.9, top=0.95, bottom=0.1, wspace=0.7, hspace=1.2,
                      width_ratios=[1, 1, 1, 0.6, 1.5, 1])
    
    ax1 = f.add_subplot(gs[0,0])
    behavbargraph(ax1, df_behav, diet,
                  ['pref2_cas_forced', 'pref2_malt_forced'],
                  colorgroup=colorgroup,
                  ylabel="Licks")
    
    ax2 = f.add_subplot(gs[0,1])
    behavbargraph(ax2, df_photo, diet, ['pref2_cas_lats_fromsip', 'pref2_malt_lats_fromsip'],
                  colorgroup=colorgroup,
                  ylabel="Latency (s)")
    
    ax3 = f.add_subplot(gs[0,2])
    behavbargraph(ax3, df_behav, diet,
              ['pref2_cas_free', 'pref2_malt_free'],
              colorgroup=colorgroup,
              ylabel="Licks")
    
    # ax4 = f.add_subplot(gs[0,3])
    # behavbargraph(ax4, df_behav, diet,
    #           ['pref2_ncas', 'pref2_nmalt'],
    #           colorgroup=colorgroup,
    #           ylabel="Choices out of 20")
    
    ax4 = f.add_subplot(gs[0,3])
    onecolbehavbargraph(ax4, df_behav, diet,
              'pref2',
              colorgroup=colorgroup,
              ylabel="Choices out of 20")

    ax5 = f.add_subplot(gs[0,4])
    averagetrace(ax5, df_photo, diet, ['pref2_cas_licks_forced', 'pref2_malt_licks_forced'],
                 event='Licks', fullaxis=True, colorgroup=colorgroup, ylabel=True)
    for xval in epoch:
        ax5.axvline(xval, linestyle='--', color='k', alpha=0.3)
       
    ax6 = f.add_subplot(gs[0,5])
    peakbargraph(ax6, df_photo, diet, ['pref2_cas_licks_forced', 'pref2_malt_licks_forced'],
                 peaktype='auc', epoch=[100, 119],
                 sc_color='w', colorgroup=colorgroup,
                 scattersize=scattersize)

    ax7 = f.add_subplot(gs[1,0], sharey=ax1)
    behavbargraph(ax7, df_behav, diet,
                  ['pref3_cas_forced', 'pref3_malt_forced'],
                  colorgroup=colorgroup,
                  ylabel="Licks")
                     
    ax8 = f.add_subplot(gs[1,1], sharey=ax2)
    behavbargraph(ax8, df_photo, diet, ['pref3_cas_lats_fromsip', 'pref3_malt_lats_fromsip'],
                  colorgroup=colorgroup,
                  ylabel="Latency (s)")
    
    ax9 = f.add_subplot(gs[1,2], sharey=ax3)
    behavbargraph(ax9, df_behav, diet,
              ['pref3_cas_free', 'pref3_malt_free'],
              colorgroup=colorgroup,
              ylabel="Licks")
    
    # ax10 = f.add_subplot(gs[1,3], sharey=ax4)
    # behavbargraph(ax10, df_behav, diet,
    #           ['pref3_ncas', 'pref3_nmalt'],
    #           colorgroup=colorgroup,
    #           ylabel="Choices out of 20")
    
    ax10 = f.add_subplot(gs[1,3], sharey=ax4)
    onecolbehavbargraph(ax10, df_behav, diet,
              'pref3',
              colorgroup=colorgroup,
              ylabel="Choices out of 20")


    ax11 = f.add_subplot(gs[1,4], sharey=ax5)
    averagetrace(ax11, df_photo, diet, ['pref3_cas_licks_forced', 'pref3_malt_licks_forced'],
                 event='Licks', fullaxis=True, colorgroup=colorgroup, ylabel=True)
    ax11.set_ylim([-1.6, 2.4])
    ax11.set_yticks([-1, 0, 1, 2])
    for xval in epoch:
        ax11.axvline(xval, linestyle='--', color='k', alpha=0.3)                     
    
    ax12 = f.add_subplot(gs[1,5], sharey=ax6)
    peakbargraph(ax12, df_photo, diet, ['pref3_cas_licks_forced', 'pref3_malt_licks_forced'],
                 peaktype=peaktype, epoch=[100, 119],
                 sc_color='w', colorgroup=colorgroup,
                 scattersize=scattersize)
    ax12.set_ylim([-2, 7])
    ax12.set_yticks([-2, 0, 2, 4, 6])
    
    return f

def behavbargraph(ax, df, diet, keys,
                 sc_color='w', colorgroup='control', ylabel="",
                 ylim=[-0.05, 0.1], grouplabeloffset=0,
                 scattersize=30):
    
    if colorgroup == 'control':
        bar_colors=['xkcd:silver', 'w']
    else:
        bar_colors=[col['pr_cas'], col['pr_malt']]
    
    df = df.xs(diet, level=1)
    
    a = [df[keys[0]], df[keys[1]]]

    ax, x, _, _ = tp.barscatter(a, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = bar_colors,
                 scatteredgecolor = [almost_black],
                 scatterlinecolor = almost_black,
                 scatterfacecolor = [sc_color],
                 grouplabel=['Cas', 'Malt'],
                 grouplabeloffset=grouplabeloffset,
                 scattersize = scattersize,
                 xfontsize=6,
                 barwidth=0.75,
                 ax=ax)
    ax.set_clip_on(False)
    ax.set_ylabel(ylabel)
    # ax.set_xlim([0, 6])

def onecolbehavbargraph(ax, df, diet, keys,
                 sc_color='w', colorgroup='control', ylabel="Casein preference",
                 ylim=[-0.05, 0.1], grouplabeloffset=0,
                 scattersize=30):
    
    if colorgroup == 'control':
        bar_colors=['xkcd:silver']
    else:
        bar_colors=[col['pr_cas']]
    
    df = df.xs(diet, level=1)
    
    a = [df[keys]]

    ax, x, _, _ = tp.barscatter(a,
                 barfacecoloroption = 'individual',
                 barfacecolor = bar_colors,
                 scatteredgecolor = [almost_black],
                 scatterlinecolor = almost_black,
                 scatterfacecolor = [sc_color],
                 # grouplabel=['Cas', 'Malt'],
                 # grouplabeloffset=grouplabeloffset,
                 scattersize = scattersize/3,
                 xfontsize=6,
                 barwidth=0.6,
                 spaced=True,
                 xspace=0.2,
                 ax=ax)
    

    ax.set_xlim([0,2])
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim([-0.1, 1.1])
    ax.plot(ax.get_xlim(), [0.5, 0.5], linestyle='dashed',color='k', alpha=0.3)
    ax.set_ylabel(ylabel)