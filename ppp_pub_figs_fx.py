# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:16:41 2018

NEED TO RUN ppp1_grouped.py first to load data and certain functions into memory.
Trying to do this using import statement - but at the moment not importing modules.

@author: jaimeHP
"""
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import JM_custom_figs as jmfig
import JM_general_functions as jmf

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

def toc():
    tc = timeit.default_timer()
    print(tc-tic)

def inch(mm):
    result = mm*0.0393701
    return result

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

    x = [[df.xs(diet, level=1)['cond1-cas1'], df.xs(diet, level=1)['cond1-cas2']],
     [df.xs(diet, level=1)['cond1-malt1'], df.xs(diet, level=1)['cond1-malt2']]]

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
    ax.set_title(title)

def pref_behav_fig(ax, df_behav, df_photo, prefsession=1, dietswitch=False):

    forced_cas_key = 'pref' + str(prefsession) + '_cas_forced'
    forced_malt_key = 'pref' + str(prefsession) + '_malt_forced'
    free_cas_key = 'pref' + str(prefsession) + '_cas_free'
    free_malt_key = 'pref' + str(prefsession) + '_malt_free'
    choice_cas_key = 'pref' + str(prefsession) + '_ncas'
    choice_malt_key = 'pref' + str(prefsession) + '_nmalt'
    lat_cas_key = 'pref' + str(prefsession) + '_cas_lats'
    lat_malt_key = 'pref' + str(prefsession) + '_malt_lats'
    
    scattersize = 50
 
    if dietswitch == True:
        grouplabel=['NR \u2192 PR', 'PR \u2192 NR']
        barfacecolor = [col['lp_cas'], col['lp_malt'], col['np_cas'], col['np_malt']]
    else:
        grouplabel=['NR', 'PR']
        barfacecolor = [col['np_cas'], col['np_malt'], col['lp_cas'], col['lp_malt']]
    
#panel 1 - forced choice licks    
    x = [[df_behav.xs('NR', level=1)[forced_cas_key], df_behav.xs('NR', level=1)[forced_malt_key]],
         [df_behav.xs('PR', level=1)[forced_cas_key], df_behav.xs('PR', level=1)[forced_malt_key]]]
    jmfig.barscatter(x, paired=True, unequal=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = barfacecolor,
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel=grouplabel,
                 barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                 scattersize = scattersize,
                 ylim=[-50,1050],
                 ax=ax[0])

    ax[0].set_ylabel('Licks')
    ax[0].set_yticks([0, 500, 1000])

#panel 2 - latency for forced choice
    x = [[df_photo.xs('NR', level=1)[lat_cas_key], df_photo.xs('NR', level=1)[lat_malt_key]],
         [df_photo.xs('PR', level=1)[lat_cas_key], df_photo.xs('PR', level=1)[lat_malt_key]]]
    jmfig.barscatter(x, paired=True, unequal=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = barfacecolor,
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel=grouplabel,
                 barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                 scattersize = scattersize,
                 ylim=[-0.5,10],
                 ax=ax[1])
    ax[1].set_ylabel('Latency (s)')

#panel 2 - free choice licks
    x = [[df_behav.xs('NR', level=1)[free_cas_key], df_behav.xs('NR', level=1)[free_malt_key]],
         [df_behav.xs('PR', level=1)[free_cas_key], df_behav.xs('PR', level=1)[free_malt_key]]]
    jmfig.barscatter(x, paired=True, unequal=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = barfacecolor,
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel = grouplabel,
                 barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                 scattersize = scattersize,
                 ylim=[-50, 800],
                 ax=ax[2])

    ax[2].set_ylabel('Licks')
    ax[2].set_yticks([0, 250, 500, 750])

#panel 3 - free choice, choices   
    x = [[df_behav.xs('NR', level=1)[choice_cas_key], df_behav.xs('NR', level=1)[choice_malt_key]],
         [df_behav.xs('PR', level=1)[choice_cas_key], df_behav.xs('PR', level=1)[choice_malt_key]]]
    jmfig.barscatter(x, paired=True, unequal=True,
             barfacecoloroption = 'individual',
             barfacecolor = barfacecolor,
             scatteredgecolor = ['xkcd:charcoal'],
             scatterlinecolor = 'xkcd:charcoal',
             grouplabel = grouplabel,
             barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
             scattersize = scattersize,
             ylim=[-2,22],
             ax=ax[3])
    
    ax[3].set_ylabel('Choices (out of 20)')
    ax[3].set_yticks([0, 10, 20])

def lickplot(ax, licks, ylabel=True, style='raster'):        
    # Removes axes and spines
    jmfig.invisible_axes(ax)

 
    licks_x = [(x+10)*10 for x in licks]
    if style == 'histo':
        hist, bins = np.histogram(licks_x, bins=30, range=(0,300))
        center = (bins[:-1] + bins[1:]) / 2
        width = 1 * (bins[1] - bins[0])   
        ax.bar(center, hist, align='center', width=width, color='xkcd:silver')
    
    if style == 'raster':
        yvals = [1]*len(licks)
        ax.plot(licks_x,yvals,linestyle='None',marker='|',markersize=5, color='xkcd:silver')
        
    else:
        print('Not a valid style for plotting licks')

    if ylabel == True:
        ax.annotate('Licks', xy=(90,1), va='center', ha='right')


def repFig(ax, df, session, plot_licks=False, color=almost_black, yscale=True, xscale=True, legend=False):

# Plots data
    datauv = df[session+'_photo_uv']
    datablue = df[session+'_photo_blue']
    
    uv_color = jmfig.lighten_color(color, amount=0.3)
    
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
                                             wspace=0.05, hspace=0.00,
                                             height_ratios=[1,8])    
    ax1 = f.add_subplot(inner[1,0])
    repFig(ax1, df.loc[index[0]], session, color=color, xscale=False)
    ax2 = f.add_subplot(inner[1,1], sharey=ax1)
    repFig(ax2, df.loc[index[1]], session, color=color, yscale=False, legend=True)

    ax3 = f.add_subplot(inner[0,0], sharex=ax1)
    lickplot(ax3, df.loc[index[0]][session+'_licks'])
    ax4 = f.add_subplot(inner[0,1], sharey=ax3, sharex=ax2)
    lickplot(ax4, df.loc[index[1]][session+'_licks'], ylabel=False)
    
    if title == True:
        ax3.set_title('Casein')
        ax4.set_title('Maltodextrin')

def makeheatmap(ax, data, ylabel='Trials'):
    ntrials = np.shape(data)[0]
    xvals = np.linspace(-9.9,20,300)
    yvals = np.arange(1, ntrials+2)
    xx, yy = np.meshgrid(xvals, yvals)
    
    mesh = ax.pcolormesh(xx, yy, data, cmap='YlGnBu', shading = 'flat')
    ax.set_ylabel(ylabel)
    ax.set_yticks([1, ntrials])
    ax.set_xticks([])
    ax.invert_yaxis()
    
    return ax, mesh

def heatmapFig(f, df, gs, gsx, gsy, session, rat, clims=[0,1]):
    
    data_cas = df[session+'_cas'][rat]
    data_malt = df[session+'_malt'][rat]

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
    
    cbar_ax = f.add_subplot(inner[:,1])   
    cbar = f.colorbar(mesh, cax=cbar_ax, ticks=[clims[0], 0, clims[1]])
    cbar_labels = ['{0:.0f}%'.format(clims[0]*100),
                   '0% \u0394F',
                   '{0:.0f}%'.format(clims[1]*100)]
    cbar.ax.set_yticklabels(cbar_labels)

def averagetrace(f, gs, gsx, gsy, df, diet, keys, keys_lats, event='',
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
        lat_data.append(jmf.flatten_list(df[key]))
        
    stats = jmfig.get_violinstats(lat_data, points=200)
    
    x=stats[0]['coords']
    y=stats[0]['vals']
    
    x2=stats[1]['coords']
    y2=-stats[1]['vals']
    
    ax_lats.fill_between(x, y, color=color[0])
    ax_lats.fill_between(x2, y2, color=color[1])
    ax_lats.set_xlim([-10, 20])
    jmfig.invisible_axes(ax_lats)
    
#    arrow_y = ax_lats.get_ylim()[1]
#    ax_lats.plot([0], [arrow_y], 'v', color='xkcd:silver')
#    ax_lats.annotate(event, xy=(0, arrow_y), xytext=(0,5), textcoords='offset points',
#                ha='center', va='bottom')
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
                 event='sipper',
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
    heatmapFig(f, df_heatmap, gs, 0, 2, session, 'PPP1-7', clims=clims[0])
#    # average traces NR cas v malt
#    ax3 = f.add_subplot(gs[0,4])
    
    averagetrace(f, gs, 0, 4, df_photo, 'NR', keys_traces, keys_lats, event=event, color=rowcolors[0])

    ax7 = f.add_subplot(gs[0,6]) 
    peakbargraph(ax7, df_photo, 'NR', keys_bars, bar_colors=rowcolors_bar[0], sc_color='w')
#   
#    # Protein-restricted figures, row 1
    reptracesFig(f, df_reptraces, ['PR_cas', 'PR_malt'], session, gs, 1, 0, color=rowcolors[1][0])
    heatmapFig(f, df_heatmap, gs, 1, 2, session, 'PPP1-4', clims=clims[1])
#    heatmapFig(f, gs, 1, 2, 's10', 'PPP1.3', clims=clim_pr)
#    # average traces NR cas v malt
#    ax6 = f.add_subplot(gs[1,4])
    averagetrace(f, gs, 1, 4, df_photo, 'PR', keys_traces, keys_lats, event=event, color=rowcolors[1])

    ax8 = f.add_subplot(gs[1,6])
    peakbargraph(ax8, df_photo, 'PR', keys_bars, bar_colors=rowcolors_bar[1], sc_color=almost_black)
     
    return f














#
#
#
#
#
#
#
#
## To make summary figure
#
#
#def peakresponsebargraph(ax, df, keys, ylabels=True, dietswitch=False, xlabels=[]):
#    dietmsk = df.diet == 'NR'
#    
#    a = [[df[keys[0]][dietmsk], df[keys[1]][dietmsk]],
#          [df[keys[0]][~dietmsk], df[keys[1]][~dietmsk]]]
#
#    x = data2obj2D(a)
#    if dietswitch == True:
#        cols = [green, light_green, 'xkcd:silver', 'w']
#    else:        
#        cols = ['xkcd:silver', 'w', green, light_green]
#    
#    jmfig.barscatter(x, paired=True,
#                 barfacecoloroption = 'individual',
#                 barfacecolor = [cols[0], cols[1], cols[2], cols[3]],
#                 scatteredgecolor = [almost_black],
#                 scatterlinecolor = almost_black,
#                 scattersize = 100,
#                 ax=ax)
#    ax.set_xticks([])
#    
#    for x,label in enumerate(xlabels):
#        ax.text(x+1, -0.0175, label, ha='center')
#    
#    ax.set_ylim([-.02, 0.135])
#    yticks = [0, 0.05, 0.1]
#    ax.set_yticks(yticks)
#    
#    if ylabels == True:
#        yticklabels = ['{0:.0f}%'.format(x*100) for x in yticks]
#        ax.set_yticklabels(yticklabels)
#        ax.set_ylabel('\u0394F', rotation=0)
#    else:
#        ax.set_yticklabels([])
#
#
#
#def makesummaryFig2(df_pref, df_photo):
#    gs = gridspec.GridSpec(1, 2, wspace=0.5)
#    mpl.rcParams['figure.subplot.left'] = 0.10
#    mpl.rcParams['figure.subplot.top'] = 0.85
#    mpl.rcParams['axes.labelpad'] = 4
#    f = plt.figure(figsize=(5,2))
#    
#    ax0 = f.add_subplot(gs[0])
#    choicefig(df_pref, ['pref1', 'pref2', 'pref3'], ax0)
#    ax0.set_ylabel('Casein preference')
#    ax0.set_yticks([0, 0.5, 1.0]) 
#    ax0.set_yticklabels(['0', '0.5', '1'])
#    ax0.set_title('Behaviour')
#    ax1 = f.add_subplot(gs[1])
#    choicefig(df_photo, ['pref1_peak_delta', 'pref2_peak_delta', 'pref3_peak_delta'], ax1)
#    ax1.set_ylabel('\u0394F (Casein - Malt.)')
#    
#    ax1.set_ylim([-0.035, 0.09])
#    ax1.set_yticks([-0.02, 0, 0.02, 0.04, 0.06, 0.08])
#    ax1.set_yticklabels([-0.02, 0, 0.02, 0.04, 0.06, 0.08])
#    ax1.set_title('Photometry')
#
#    return f
#
#
#
#
#def peakresponsebargraph(df, keys, ax):
#    dietmsk = df.diet == 'NR'
#    
#    a = [[df[keys[0]][dietmsk], df[keys[1]][dietmsk]],
#          [df[keys[0]][~dietmsk], df[keys[1]][~dietmsk]]]
#
#    x = data2obj2D(a)
#    
#    cols = ['xkcd:silver', 'w', 'xkcd:kelly green', 'xkcd:light green']
#    
#    ax, x, _, _ = jmfig.barscatter(x, paired=True,
#                 barfacecoloroption = 'individual',
#                 barfacecolor = [cols[0], cols[1], cols[2], cols[3]],
#                 scatteredgecolor = ['xkcd:charcoal'],
#                 scatterlinecolor = 'xkcd:charcoal',
#                 grouplabel=['NR', 'PR'],
#                 scattersize = 100,
#                 ax=ax)
#    ax.set_ylim([-.02, 0.15])
#    ax.set_yticks([0, 0.05, 0.1, 0.15])
##    ax.set_ylabel('\u0394F')
#
#def behav_vs_photoFig(ax, xdata, ydata, diet):
#    for x, y, d in zip(xdata, ydata, diet):
#        if d == 'NR':
#            color = 'k'
#        else:
#            color = 'g'
#        ax.scatter(x, y, c=color)
#
#
#def makesummaryFig():
#    gs = gridspec.GridSpec(1, 2, width_ratios=[1,3], wspace=0.3)
#    mpl.rcParams['figure.subplot.left'] = 0.10
#    mpl.rcParams['figure.subplot.top'] = 0.90
#    mpl.rcParams['axes.labelpad'] = 4
#    f = plt.figure(figsize=(inch(300), inch(120)))
#    
#    adjust = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs[0],
#                                             wspace=0.05,
#                                             height_ratios=[18,1])
#    
#    ax0 = f.add_subplot(adjust[0])
#    choicefig(df1, ['pref1', 'pref2', 'pref3'], ax0)
#    ax0.set_ylabel('Casein preference')
#    plt.yticks([0, 0.5, 1.0])
#    ax_ = f.add_subplot(adjust[1])
#    jmfig.invisible_axes(ax_)
#    
#    inner = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[1],
#                                             wspace=0.15)
#    ax1 = f.add_subplot(inner[0])
#    ax2 = f.add_subplot(inner[1])
#    ax3 = f.add_subplot(inner[2])
#    
#    peakresponsebargraph(ax1, df4, ['cas1_licks_peak', 'malt1_licks_peak'],
#                         xlabels=['NR', 'PR'])
#    peakresponsebargraph(ax2, df4, ['cas2_licks_peak', 'malt2_licks_peak'],
#                         xlabels=['NR \u2192 PR', 'PR \u2192 NR'],
#                         ylabels=False, dietswitch=True)
#    peakresponsebargraph(ax3, df4, ['cas3_licks_peak', 'malt3_licks_peak'],
#                         xlabels=['NR \u2192 PR', 'PR \u2192 NR'],
#                         ylabels=False, dietswitch=True)
#    
#    titles = ['Preference test 1', 'Preference test 2', 'Preference test 3']
#    for ax, title in zip([ax1, ax2, ax3], titles):
#        ax.set_title(title)
#    
#    return f
#
