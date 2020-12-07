# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:17:08 2020

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

keys = [['pref1_cas_licks_forced', 'pref1_malt_licks_forced'],
                      ['pref2_cas_licks_forced', 'pref2_malt_licks_forced'],
                      ['pref3_cas_licks_forced', 'pref3_malt_licks_forced']]

def summary_subfig_correl_diffdays(df_behav, df_delta, pref, colorgroup="control"):
    
    if colorgroup == "control":
        colors = [col["nr_cas"], col["pr_malt"]]
    else:
        colors = [col["pr_malt"], col["nr_cas"]]
    
    y_key = "pref" + pref
    x_key = "delta_" + pref
    
    dfy_NR = df_behav.xs("NR", level=1)[y_key]
    dfx_NR = df_delta.xs("NR", level=1)[x_key]
    
    dfy_PR = df_behav.xs("PR", level=1)[y_key]
    dfx_PR = df_delta.xs("PR", level=1)[x_key]
    
    f, ax = plt.subplots(figsize=(1.6, 1.75),
                     gridspec_kw={"left": 0.25, "right": 0.95, "bottom": 0.2})
    
    ax.scatter(dfx_NR, dfy_NR, color=colors[0], marker='o', edgecolors='k', s=20)
    ax.scatter(dfx_PR, dfy_PR, color=colors[1], marker='o', edgecolors='k', s=20)
    
    ax.set_ylabel("Casein preference")
    ax.set_xlabel('Diff. in z-score (Casein - Malt.)')
    
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0, 0.5, 1])
    
    ax.set_xlim([-6, 16])
    ax.set_xticks([-5, 0, 5, 10, 15])

    return f
    

epoch = [100, 149]

df_delta = find_delta(df_photo, keys, epoch=epoch)

# fig6_p2 = summary_subfig_casmalt(df_photo, "NR", keys)
# fig6_p2.savefig(savefolder + "fig6_p2.pdf")

# fig6_p3 = summary_subfig_correl(df_behav, df_delta, 'NR', use_zscore_diff=True)
# fig6_p3.savefig(savefolder + "fig6_p3.pdf")


# fig6_p5 = summary_subfig_casmalt(df_photo, "PR", keys)
# fig6_p5.savefig(savefolder + "fig6_p5.pdf")

# fig6_p6 = summary_subfig_correl(df_behav, df_delta, 'PR', use_zscore_diff=True)
# fig6_p6.savefig(savefolder + "fig6_p6.pdf")

f = summary_subfig_correl_diffdays(df_behav, df_delta, "1", colorgroup="control")
f.savefig(savefolder + "figs4_p1.pdf")

f = summary_subfig_correl_diffdays(df_behav, df_delta, "2", colorgroup="expt")
f.savefig(savefolder + "figs4_p2.pdf")

f = summary_subfig_correl_diffdays(df_behav, df_delta, "3", colorgroup="expt")
f.savefig(savefolder + "figs4_p3.pdf")