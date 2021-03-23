# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:49:41 2021

@author: admin
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


from ppp_pub_figs_settings import *

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
    
    estimation_plot(df, barx=barx, ax=ax2, stats_args=stats_args, idx=(("control1", "test1"), ("control2", "test2")))
    
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
    
    estimation_plot_1group(df, barx=barx, ax=ax2, stats_args=stats_args)
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
    
    estimation_plot_summary(df, barx=barx, ax=ax2, stats_args=stats_args)
    
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
    
    estimation_plot_horiz(df, barx=barx, ax=ax2, stats_args=stats_args)
    
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

    return f

def estimation_plot(df, barx=[], ax=[], stats_args={}, idx=("control1", "test1"), plottype="onegroup"):
    
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

def estimation_plot_horiz(df, barx=[], ax=[], stats_args={}):
    
    est_stats = db.load(df, idx=(("control1", "test1"), ("control2", "test2")), id_col="rat", paired=True)
    
    e = est_stats.mean_diff
    results = e.results
    
    try:
        with pd.ExcelWriter(stats_args["file"], mode="a", engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name=stats_args["sheet"])
    except:
        print("No stats file to write to.")
    
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

    #     # Create the violinplot.
    #     # New in v0.2.6: drop negative infinities before plotting.
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
        
    ax.spines["bottom"].set_visible(False)
    # ax.spines["right"].set_visible(True)
    ax.axhline(color="grey", linestyle="dashed")
    
    trans = transforms.blended_transform_factory(
                    ax.transData, ax.transAxes)

def estimation_plot_1group(df, barx=[], ax=[], stats_args={}):
    
    est_stats = db.load(df, idx=("control1", "test1"), id_col="rat", paired=True)
    
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
        
    ax.spines["bottom"].set_visible(False)
    ax.axhline(color="black")
    
    trans = transforms.blended_transform_factory(
                    ax.transData, ax.transAxes)

    ax.plot([barx[0], barx[1]], [-0.05, -0.05], transform=trans, color="black", clip_on=False)
    
    for xtick in barx:
        ax.plot([xtick, xtick], [-0.1, -0.05], transform=trans, color="black", clip_on=False)
        
    for xtick in ticks_to_plot:
        ax.text(xtick, -0.12, "C-M", ha="center", va="top", transform=trans, color="black", clip_on=False, fontsize=6)

    ax.set_ylabel("Paired difference", fontsize=6)

def estimation_plot_summary(df, barx=[], ax=[], stats_args={}):
    
    est_stats = db.load(df, idx=("control1", "test1", "test2"), id_col="rat", paired=False)
    
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
    # if plot_kwargs["violinplot_kwargs"] is None:
    #     violinplot_kwargs = default_violinplot_kwargs
    # else:
    #     violinplot_kwargs = merge_two_dicts(default_violinplot_kwargs,
    #                                         plot_kwargs["violinplot_kwargs"])
        
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
        
    #     # Create the violinplot.
    #     # New in v0.2.6: drop negative infinities before plotting.
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
        
    ax.spines["bottom"].set_visible(False)
    ax.axhline(color="black")
    
    trans = transforms.blended_transform_factory(
                    ax.transData, ax.transAxes)

    # ax.plot([barx[0], barx[1]], [-0.05, -0.05], transform=trans, color="black", clip_on=False)
    # ax.plot([barx[2], barx[3]], [-0.05, -0.05], transform=trans, color="black", clip_on=False)
    
    # for xtick in barx:
    #     ax.plot([xtick, xtick], [-0.1, -0.05], transform=trans, color="black", clip_on=False)
        
    for xtick in ticks_to_plot:
        ax.text(xtick, -0.04, "vs.\nd1", ha="center", va="top", transform=trans, color="black", clip_on=False, fontsize=6)
    
    # ax.text(ticks_to_plot[0], -0.12, "2 - 1", ha="center", va="top", transform=trans, color="black", clip_on=False, fontsize=6)
    # ax.text(ticks_to_plot[1], -0.12, "3 - 1", ha="center", va="top", transform=trans, color="black", clip_on=False, fontsize=6)
    
    # ax.text(2.5, -0.12, "vs. pref1", ha="center", va="top", transform=trans, color="black", clip_on=False, fontsize=6)

    ax.set_ylabel("Diff. vs. Pref. 1", fontsize=6)
    
    
#Adds control column for calculating vs. 50%
con = [0.5] * 15
df_behav.insert(0, "control", con)

stats_args = {}
stats_args["file"] = "C:\\Github\\PPP_analysis\\stats\\estimation_stats.xlsx"
stats_args["file"] = "" # Comment this line out to write a new stats file

keys = ["pref1_malt_forced", "pref1_cas_forced"]
stats_args["sheet"] = "pref1_forced_licks"
data, df = prep4estimationstats(df_behav, ["NR", "PR"], keys)
fig2_p1 = barscatter_plus_estimation(data, df, ylabel="Licks", stats_args=stats_args)