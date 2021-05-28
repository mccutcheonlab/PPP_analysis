# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:16:41 2018

The function estimation_plot uses modified source code from the dabest Python
package used under conditions of a BSD 3-Clause Clear License.
Copyright (c) 2016-2020 Joses W. Ho. All rights reserved.
https://github.com/ACCLAB/DABEST-python

@author: jaimeHP
"""
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

import numpy as np
import pandas as pd
import dabest as db

import trompy as tp

from scipy.stats import linregress
from scipy.stats import gaussian_kde

from settings4figs import *

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

def heatmap_panel(df, diet):
    
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
        
def photogroup_panel(df_photo, keys, dietgroup, colorgroup="control"):
    
    epoch=[100,149]
    
    event='Licks'
    
    f = plt.figure(figsize=(1.5, 2))
    
    
    gs = gridspec.GridSpec(2,1,
                                         height_ratios=[0.15,1],
                                         hspace=0.0,
                                         left=0.3, right=0.8, top=0.9, bottom=0.2)
    
    
    ax1 = f.add_subplot(gs[1,0])
    averagetrace(ax1, df_photo, dietgroup, keys, event=event, fullaxis=True, colorgroup=colorgroup)
    ax1.set_ylim([-1.5, 3.2])
    for xval in epoch:
        ax1.axvline(xval, linestyle='--', color='k', alpha=0.3)

        
    ax2 = f.add_subplot(gs[0,0], sharex=ax1)
    

    ax2.axis('off')
    if event == 'Sipper':
        ax2.plot(100,0, 'v', color='xkcd:silver')
        ax2.annotate(event, xy=(100, 0), xytext=(0,5), textcoords='offset points',
            ha='center', va='bottom')
    elif event == 'Licks':
        ax2.plot([100,150], [0,0], color='xkcd:silver', linewidth=3)
        ax2.annotate(event, xy=(125, 0), xytext=(0,5), textcoords='offset points',
            ha='center', va='bottom')
    
    return f

def summary_subfig_casmalt(df, diet, keys, epoch=[100,149]):
    
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

    malt_data = [np.mean(day) for day in malt_auc]
    malt_sem = [np.std(day)/np.sqrt(len(day)) for day in malt_auc]
    
    f, ax = plt.subplots(figsize=(1.3, 1.75),
                         gridspec_kw={"left": 0.35, "right": 0.95, "bottom": 0.2})

    ax.errorbar(xvals, malt_data, yerr=malt_sem, capsize=0, c=almost_black, linewidth=0.75, zorder=-1)
    ax.scatter(xvals, malt_data, marker='o', c='white', edgecolors='k', s=20)
    
    ax.errorbar(xvals, cas_data, yerr=cas_sem, capsize=0, c=almost_black, linewidth=0.75, zorder=-1)
    ax.scatter(xvals, cas_data, marker='o', c=cols, edgecolors='k', s=20)
    
    ax.set_ylabel('Z-score AUC')
    # ax.set_ylim([0, 5.5])
    
    trans = transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
    ax.set_xticks([1,2,3])
    ax.set_xlabel("Pref. test")

    # for x in xvals:
    #     #ax.text(x, 0.05, str(x), va='top', ha='center', fontsize=8, transform=ax.transAxes)
    #     ax.text(x, -0.05, str(x), va='top', ha='center', fontsize=6, transform=trans)
    # ax.set_xlim([0.5, 3.5])
    # ax.text(0.5, -0.12, "Pref. test", va='top', ha='center', fontsize=6, transform=ax.transAxes)
    
    return f

def summary_subfig_correl(df_behav, df_photo, diet, use_zscore_diff=True):
    
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
    f, ax = plt.subplots(figsize=(1.6, 1.75),
                         gridspec_kw={"left": 0.25, "right": 0.95, "bottom": 0.2})
    
    ax.errorbar(xvals_mean, yvals_mean, yerr=y_sem, xerr=x_sem, capsize=0, c=almost_black, linewidth=0.75, zorder=-1)
    ax.scatter(xvals_mean, yvals_mean, marker='o', c=['w', 'grey', 'k'], edgecolors='k', s=20)
    
    ax.set_ylabel('Casein preference')
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0, 0.5, 1.0]) 
    ax.set_yticklabels(['0', '0.5', '1'])

    if use_zscore_diff:
        ax.set_xlabel('Diff. in z-score (Casein - Malt.)')
        ax.set_xlim([-2.1, 7])
        ax.set_xticks([-2, 0, 2, 4, 6])
    else:
        ax.set_xlabel('T-value (Casein vs. Malt.)')
        ax.set_xlim([-5.9, 5.9])
        ax.set_xticks([-4, -2, 0, 2, 4])
    
    ax.plot(ax.get_xlim(), [0.5, 0.5], linestyle='dashed',color='k', alpha=0.2)
    ax.plot([0, 0], [-0.1, 1.1], linestyle='dashed',color='k', alpha=0.2)
    
    return f
    
def halfviolin(v, half='right', fill_color='k', alpha=1,
                line_color='k', line_width=0):
    import numpy as np

    for b in v['bodies']:
        V = b.get_paths()[0].vertices

        mean_vertical = np.mean(V[:, 0])
        mean_horizontal = np.mean(V[:, 1])

        if half == 'right':
            V[:, 0] = np.clip(V[:, 0], mean_vertical, np.inf)
        elif half == 'left':
            V[:, 0] = np.clip(V[:, 0], -np.inf, mean_vertical)
        elif half == 'bottom':
            V[:, 1] = np.clip(V[:, 1], -np.inf, mean_horizontal)
        elif half == 'top':
            V[:, 1] = np.clip(V[:, 1], mean_horizontal, np.inf)

        b.set_color(fill_color)
        b.set_alpha(alpha)
        b.set_edgecolor(line_color)
        b.set_linewidth(line_width)
        
def equalize_y(ax):
    lim = np.max(np.abs(ax.get_ylim()))
    ax.set_ylim([-lim, lim])

def prep4estimationstats(df, groups, keys, id_col="rat"):
    
    df1 = df.xs(groups[0], level=1)[keys]
    df1.reset_index(inplace=True)
    df1.columns = [id_col, "control1", "test1"]
    
    df2 = df.xs(groups[1], level=1)[keys]
    df2.reset_index(inplace=True)
    df2.columns = [id_col, "control2", "test2"]
    
    df_to_return = pd.concat([df1, df2], sort=True)
    
    data_to_return = np.array([[df1["control1"], df1["test1"]], [df2["control2"], df2["test2"]]], dtype=object)

    return data_to_return, df_to_return

def prep4estimationstats_1group(df, groups, keys, id_col="rat"):
    
    df1 = df.xs(groups[0], level=1)[keys]
    df1.reset_index(inplace=True)
    df1.columns = [id_col, "control1", "test1"]
    
    df_to_return = df1
    
    data_to_return = [df1["control1"].tolist(), df1["test1"].tolist()]
    
    return data_to_return, df_to_return

def prep4estimationstats_summary(df, groups, keys, id_col="rat"):
    
    df1 = df.xs(groups[0], level=1)[keys]
    df1.reset_index(inplace=True)
    df1.columns = [id_col, "control1", "test1", "test2"]
    
    df_to_return = df1
    
    data_to_return = [df1["control1"].tolist(), df1["test1"].tolist(), df1["test2"].tolist()]
        
    return data_to_return, df_to_return
        
def barscatter_plus_estimation(data, df, ylabel="", stats_args={}):
    
    data = np.array(data, dtype=object)
    
    f, [ax1, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(1.3, 1.75),
                         gridspec_kw={"height_ratios": [1, 0.5], "left": 0.35, "right": 0.95, "bottom": 0.1})
    
    grouplabel=['NR', 'PR']
    barfacecolor = [col['nr_malt'], col['nr_cas'], col['pr_malt'], col['pr_cas']]
    
    _, barx, _, _ = tp.barscatter(data, ax=ax1, paired=True,
                                  barfacecoloroption = 'individual',
                                  barfacecolor = barfacecolor,
                                  scatteredgecolor = ['xkcd:charcoal'],
                                  scatterlinecolor = 'xkcd:charcoal',
                                  # grouplabel=grouplabel,
                                  # barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                                  scattersize = scattersize)
    
    ax1.set_ylabel(ylabel, fontsize=6)
    
    estimation_plot(df, barx=barx, ax=ax2, stats_args=stats_args, idx=(("control1", "test1"), ("control2", "test2")), plottype="twogroup")
    
    equalize_y(ax2)
    
    trans = transforms.blended_transform_factory(
                    ax1.transData, ax1.transAxes)
    barlabels = ['Malt', 'Cas', 'Malt', 'Cas']
    
    for xtick, label in zip(barx, barlabels):
        ax1.text(xtick, -0.03, label, ha="center", va="top", transform=trans, fontsize=5)
        
    f.align_ylabels()
    
    return f

def barscatter_plus_estimation_1group(data, df, colors="control", ylabel="", stats_args={}):
        
    f, [ax1, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(1, 1.75),
                         gridspec_kw={"height_ratios": [1, 0.5], "left": 0.43, "right": 0.95, "bottom": 0.1})
    
    # grouplabel=['NR', 'PR']
    if colors == "expt":
        barfacecolor = [col['pr_malt'], col['pr_cas']]
    else:
        barfacecolor = [col['nr_malt'], col['nr_cas']]
        
        
    _, barx, _, _ = tp.barscatter(data, ax=ax1, paired=True,
                                  barfacecoloroption = 'individual',
                                  barfacecolor = barfacecolor,
                                  scatteredgecolor = ['xkcd:charcoal'],
                                  scatterlinecolor = 'xkcd:charcoal',
                                  barwidth = .75,
                                  groupwidth = .5,
                                  # grouplabel=grouplabel,
                                  # barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                                  scattersize = scattersize)
    
    ax1.set_ylabel(ylabel, fontsize=6)
    
    estimation_plot(df, barx=barx, ax=ax2, stats_args=stats_args, idx=("control1", "test1"), plottype="onegroup")
    equalize_y(ax2)
    
    ax1.set_xlim([0.2,2.8])
    
    trans = transforms.blended_transform_factory(
                    ax1.transData, ax1.transAxes)
    barlabels = ['Malt', 'Cas']
    
    for xtick, label in zip(barx, barlabels):
        ax1.text(xtick, -0.03, label, ha="center", va="top", transform=trans, fontsize=6)
        
    f.align_ylabels()
    
    return f

def barscatter_plus_estimation_summary(data, df, colors="control", ylabel="", stats_args={}):

    f, [ax1, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(1.3, 1.75),
                         gridspec_kw={"height_ratios": [1, 0.5], "left": 0.35, "right": 0.95, "bottom": 0.1})
    
    if colors == "control":
        barfacecolor = [col['nr_cas'], col['pr_cas'], col['pr_cas']]
    else:
        barfacecolor = [col['pr_cas'], col['nr_cas'], col['nr_cas']]
    
    _, barx, _, _ = tp.barscatter(data, ax=ax1, paired=True,
                                  barfacecoloroption = 'individual',
                                  barfacecolor = barfacecolor,
                                  scatteredgecolor = ['xkcd:charcoal'],
                                  scatterlinecolor = 'xkcd:charcoal',
                                  # grouplabel=grouplabel,
                                  # barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                                  scattersize = scattersize)
    
    ax1.set_ylabel(ylabel, fontsize=6)
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.set_xlim([0.2, 3.8])
    
    estimation_plot(df, barx=barx, ax=ax2, stats_args=stats_args, idx=("control1", "test1", "test2"), plottype="summary")
    
    equalize_y(ax2)
    
    trans = transforms.blended_transform_factory(
                    ax1.transData, ax1.transAxes)
    barlabels = ["1", "2", "3"]
    
    for xtick, label in zip(barx, barlabels):
        ax1.text(xtick, -0.04, label, ha="center", va="top", transform=trans, fontsize=6)
    
    f.align_ylabels()
    
    return f

def barscatter_plus_estimation_vs50_2col(data, df, ylabel="", stats_args={}):
    
    data[:,0] = data[:,1]
    data[:,1] = [[], []]
    
    data = np.array(data, dtype=object)
    
    grouplabel=['NR', 'PR']
    barfacecolor = [col['nr_cas'], col['nr_cas'], col['pr_cas'], col['pr_cas']]
    
    f, ax1 = plt.subplots(figsize=(1.9, 1.75),
                         gridspec_kw={"left": 0.35, "right": 0.85, "bottom": 0.1})

    
    _, barx, _, _ = tp.barscatter(data, ax=ax1, paired=False,
                                  barfacecoloroption = 'individual',
                                  barfacecolor = barfacecolor,
                                  scatteredgecolor = ['xkcd:charcoal'],
                                  scatterlinecolor = 'xkcd:charcoal',
                                  # grouplabel=grouplabel,
                                  # barlabels=['Cas', 'Malt', 'Cas', 'Malt'],
                                  scattersize = 10,
                                  spaced=True,
                                  xspace=0.1,)

    ax1.set_ylabel(ylabel, fontsize=6)
    
    ax2 = ax1.twinx() 
    
    estimation_plot(df, barx=barx, ax=ax2, stats_args=stats_args, idx=(("control1", "test1"), ("control2", "test2")), plottype="horiz")
    
    ax1.set_ylim([-0.03, 1.1])
    ax2.set_ylim([-0.53, 0.6])
    
    ax1.set_yticks([0, 0.5, 1])
    # ax2.set_yticks([-0.5, 0, 0.5])
    ax2.set_yticks([])

    
    trans = transforms.blended_transform_factory(
                    ax1.transData, ax1.transAxes)
    
    xticks = [1, 2]
        
    for xtick, label in zip(xticks, grouplabel):
        ax1.text(xtick, -0.03, label, ha="center", va="top", transform=trans, fontsize=6)
        
    # ax1.set_xlim([0,4])

    return f

def estimation_plot(df, barx=[], ax=[], stats_args={}, idx=("control1", "test1"), plottype="onegroup"):
    """
    This function uses modified source code from the dabest Python package, 
    Copyright (c) 2016-2020 Joses W. Ho
    All rights reserved.
    https://github.com/ACCLAB/DABEST-python
    """
    
    if plottype == "summary":
        est_stats = db.load(df, idx=idx, id_col="rat", paired=False)
    else:
        est_stats = db.load(df, idx=idx, id_col="rat", paired=True)
        
    e = est_stats.mean_diff
    results = e.results
    
    try:
        with pd.ExcelWriter(stats_args["file"], mode="a", engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name=stats_args["sheet"])
    except:
        print("No stats file to write to.")

    contrast_xtick_labels = []
    
    dabest_obj  = e.dabest_obj
    plot_data   = e._plot_data
    xvar        = e.xvar
    yvar        = e.yvar
    is_paired   = e.is_paired

    all_plot_groups = dabest_obj._all_plot_groups
    idx             = dabest_obj.idx
    
    default_violinplot_kwargs = {'widths':0.5, 'vert':True,
                               'showextrema':False, 'showmedians':False}
        
    violinplot_kwargs = default_violinplot_kwargs
    
    ticks_to_skip   = np.cumsum([len(t) for t in idx])[:-1].tolist()
    ticks_to_skip.insert(0, 0)
    
    # Then obtain the ticks where we have to plot the effect sizes.
    ticks_to_plot = [t for t in range(0, len(all_plot_groups))
                    if t not in ticks_to_skip]

    ticks_to_plot = [barx[tick] for tick in ticks_to_plot]
    
    fcolors = [col['nr_cas'], col['pr_cas']]
    
    for j, tick in enumerate(ticks_to_plot):
        current_group     = results.test[j]
        current_control   = results.control[j]
        current_bootstrap = results.bootstraps[j]
        current_effsize   = results.difference[j]
        current_ci_low    = results.bca_low[j]
        current_ci_high   = results.bca_high[j]

        v = ax.violinplot(current_bootstrap[~np.isinf(current_bootstrap)],
                                      positions=[tick],
                                      **violinplot_kwargs)

        halfviolin_alpha=0.7
        halfviolin(v, fill_color=fcolors[j], alpha=halfviolin_alpha)
    
        ytick_color="black"
        es_marker_size=4

    #     # Plot the effect size.
        ax.plot([tick], current_effsize, marker='o',
                            color=ytick_color,
                            markersize=es_marker_size)
        # Plot the confidence interval.
        ax.plot([tick, tick],
                            [current_ci_low, current_ci_high],
                            linestyle="-",
                            color=ytick_color,
                            # linewidth=group_summary_kwargs['lw'],
                            linewidth=1)
        
    trans = transforms.blended_transform_factory(
                        ax.transData, ax.transAxes)
    ax.spines["bottom"].set_visible(False)
    
    if plottype == "twogroup":
        ax.axhline(color="black")
        ax.plot([barx[0], barx[1]], [-0.05, -0.05], transform=trans, color="black", clip_on=False)
        ax.plot([barx[2], barx[3]], [-0.05, -0.05], transform=trans, color="black", clip_on=False)
        
        for xtick in barx:
            ax.plot([xtick, xtick], [-0.1, -0.05], transform=trans, color="black", clip_on=False)
            
        for xtick in ticks_to_plot:
            ax.text(xtick, -0.12, "C-M", ha="center", va="top", transform=trans, color="black", clip_on=False, fontsize=6)
    
        ax.set_ylabel("Paired difference", fontsize=6)
    
    elif plottype == "onegroup":
        ax.axhline(color="black")
        ax.plot([barx[0], barx[1]], [-0.05, -0.05], transform=trans, color="black", clip_on=False)
        
        for xtick in barx:
            ax.plot([xtick, xtick], [-0.1, -0.05], transform=trans, color="black", clip_on=False)
            
        for xtick in ticks_to_plot:
            ax.text(xtick, -0.12, "C-M", ha="center", va="top", transform=trans, color="black", clip_on=False, fontsize=6)
    
        ax.set_ylabel("Paired difference", fontsize=6)
        
    elif plottype == "horiz":
        ax.axhline(color="grey", linestyle="dashed")
        
    elif plottype == "summary":
        ax.axhline(color="black")
        
        for xtick in ticks_to_plot:
            ax.text(xtick, -0.04, "vs.\nd1", ha="center", va="top", transform=trans, color="black", clip_on=False, fontsize=6)

        ax.set_ylabel("Diff. vs. Pref. 1", fontsize=6)
        
    else:
        print("Not a valid option for plottype.")
        
    return results

### Fig fx for Fig 6G, pie charts showing proportions of rats with sig diff activation
def get_proportions(df, session, diet, p="p-val"):
    
    df = df[(df["session"] == session) & (df["diet"] == diet)]
    n = len(df)

    
    cas = len(df[(df[p] < 0.05) & (df["t-stat"] > 0)])
    malt = len(df[(df[p] < 0.05) & (df["t-stat"] < 0)])
    nonsig = len(df[df[p] > 0.05])
    
    if cas + malt + nonsig != n:
        print("Something wrong with calculations")
        return    
    
    return cas/n, malt/n, nonsig/n

def makepie(data, ax, labelsOn=False):
    
    labels = "Casein", "Malt", "Non-sig"
    colors = ["blue", "black", "xkcd:light grey"]
    
    if labelsOn == True:
        ax.pie(data, explode = [0.1, 0.1, 0.1], labels=labels, colors=colors, wedgeprops={"edgecolor":"k"})
    else:
        ax.pie(data, explode = [0.1, 0.1, 0.1], colors=colors, wedgeprops={"edgecolor":"k", 'linewidth': 1})

### Fig fx for Ext Data 1-2, conditioning data
def condfigs(df, keys, dietmsk, cols, ax):
    
    a = [[df[keys[0]][dietmsk], df[keys[1]][dietmsk]],
          [df[keys[2]][dietmsk], df[keys[3]][dietmsk]]]

    ax, barx, _, _ = tp.barscatter(a, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = cols,
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 scattersize = 30,
                 ax=ax)

    return barx

### Fig fx for Ext Data 2-4, correlations between latency and peak
def remove_long_lats(x, y, threshold=20):
    
    # Checks for long latencies and removes all these entries from both x and y
    
    # converts input lists into arrays
    x = np.array(x)
    y = np.array(y)
    
    # removes trials with latencies greater than 20 s (time of maximum peak latency)
    y = y[x < threshold]
    x = x[x < threshold]
    
    return x, y

def scatter(x, y, ax, color="black", solution=""):
    # converts input lists into arrays
    x = np.array(x)
    y = np.array(y)
    
    # removes trials with latencies greater than 20 s (time of maximum peak latency)
    y = y[x<20]
    x = x[x<20]
    
    # ax.scatter(x, y, marker="o", edgecolor=color, facecolor="none")
    ax.scatter(x, y, s=12, marker="o", edgecolor="none", facecolor=color, alpha=0.1)
    
    slope, intercept, r, p, se = linregress(x, y)
    
    x_line = ax.get_xlim()
    y_line = [slope*x + intercept for x in x_line]
    ax.plot(x_line, y_line, color=color)
    
    print("{}: r={}, p={}".format(solution, r, p))
    
def scatter_plus_density(x1, y1, x2, y2, colors=["red", "black"]):

    # Tidies up data by removing long latencies
    x1, y1 = remove_long_lats(x1, y1)
    x2, y2 = remove_long_lats(x2, y2)
    
    # Set up figure grid
    gs = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.1,
                           bottom=0.2, left=0.2, right=0.85, top=0.85,
                           width_ratios=[1,0.2], height_ratios=[0.2,1])
    
    #Initialize figure
    f = plt.figure(figsize=(2.3,2.3))
    
    # Create main axis
    main_ax = f.add_subplot(gs[1, 0])
    
    scatter(x1, y1, main_ax, color=colors[0], solution="Casein")
    scatter(x2, y2, main_ax, color=colors[1], solution="Maltodextrin")
    
    main_ax.set_ylabel("Time to peak (s)", fontsize=6)
    main_ax.set_xlabel("Latency (s)", fontsize=6)
    
    # main_ax.tick
    
    # Create axis for latency density plot
    lat_ax = f.add_subplot(gs[0, 0], sharex=main_ax)
    lat_ax.tick_params(labelbottom=False)
    lat_ax.set_yticks([])
    lat_ax.set_ylabel("Density", fontsize=6)
    
    density1 = gaussian_kde(x1)
    density2 = gaussian_kde(x2)
    
    xs = np.linspace(0, lat_ax.get_xlim()[1])
    
    lat_ax.plot(xs, density1(xs), color=colors[0])
    lat_ax.plot(xs, density2(xs), color=colors[1])
    
    lat_ax.axvline(np.median(x1), color=colors[0], linestyle="--")
    lat_ax.axvline(np.median(x2), color=colors[1], linestyle="--")
    
    # Create axis for peak density plot
    peak_ax = f.add_subplot(gs[1, 1], sharey=main_ax)
    peak_ax.tick_params(labelleft=False)
    peak_ax.set_xticks([])
    peak_ax.set_xlabel("Density", fontsize=6)
    
    density1 = gaussian_kde(y1)
    density2 = gaussian_kde(y2)
    
    ys = np.linspace(0, peak_ax.get_ylim()[1])
    
    peak_ax.plot(density1(ys), ys, color=colors[0])
    peak_ax.plot(density2(ys), ys, color=colors[1])
    
    peak_ax.axhline(np.median(y1), color=colors[0], linestyle="--")
    peak_ax.axhline(np.median(y2), color=colors[1], linestyle="--")
    
    label_ax = f.add_subplot(gs[0,1])
    tp.invisible_axes(label_ax)

    label_ax.text(0,0.5,"Maltodextrin", color=colors[1], fontsize=6)
    label_ax.text(0,0.1,"Casein", color=colors[0], fontsize=6)
    
    label_ax.set_ylim([0,1])
    
    return f