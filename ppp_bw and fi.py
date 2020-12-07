# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:07:17 2018

@author: James Rig
"""

# import JM_general_functions as jmf
# import JM_custom_figs as jmfig
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms

#import rpy2.robjects as ro
#from rpy2.robjects import r, pandas2ri, numpy2ri
#pandas2ri.activate()
#numpy2ri.activate()

from scipy import stats

import trompy as tp

import dabest as db
import pandas as pd

from ppp_pub_figs_settings import *

# col={}
# col['np_cas'] = 'xkcd:silver'
# col['np_malt'] = 'white'
# col['lp_cas'] = 'xkcd:kelly green'
# col['lp_malt'] = 'xkcd:light green'

# Loads in data

xlfile = "C:\\GitHub\\PPP_analysis\\data\\PPP_body weight and food intake.xlsx"
statsfolder = "C:\\GitHub\\PPP_analysis\\stats\\"
figsfolder = "C:\\Users\\jmc010\\Dropbox\\Publications in Progress\\PPP Paper\\04_JNS\\Figs\\"

# Body weight data
df = pd.read_excel(xlfile, sheet_name='PPP_bodyweight')
df.set_index('rat', inplace=True)

df.drop('cage', axis=1, inplace=True)

df.drop(['PPP1.8'], inplace=True)
df.drop(['PPP3.1'], inplace=True)
df.drop(['PPP3.6'], inplace=True)
df.drop(['PPP3.7'], inplace=True)
# df.drop(['PPP3.8'], inplace=True)
df.drop(['PPP4.2'], inplace=True)
df.drop(['PPP4.3'], inplace=True)
df.drop(['PPP4.5'], inplace=True)
df.drop(['PPP4.7'], inplace=True)
df.drop(['PPP4.8'], inplace=True)

df_days = df.loc[:,'d0':'d14']

nr_mean = df_days[df['diet'] == 'NR'].mean()
nr_sem = df_days[df['diet'] == 'NR'].std() / np.sqrt(len(df['diet'] == 'NR'))

pr_mean = df_days[df['diet'] == 'PR'].mean()
pr_sem = df_days[df['diet'] == 'PR'].std() / np.sqrt(len(df['diet'] == 'PR'))

nrd0 = df_days[df['diet'] == 'NR']['d0']
prd0 = df_days[df['diet'] == 'PR']['d0']

nrd14 = df_days[df['diet'] == 'NR']['d14']
prd14 = df_days[df['diet'] == 'PR']['d14']

# Prepare data for stats
df.set_index(['diet'], inplace=True, append=True)
df_days = df.loc[:,'d0':'d14']
data = df_days.stack()
data = data.to_frame()
data.reset_index(inplace=True) 
data.columns = ['rat', 'diet', 'day', 'bw']

data.to_csv(statsfolder+"df_days_stacked.csv")

# Food intake data
df = pd.read_excel(xlfile, sheet_name='PPP_foodintake')
df.set_index('cage', inplace=True)

df.drop(['cage_3.5'], inplace=True)
df.drop(['cage_4.4'], inplace=True)

df_days = df.loc[:,'d0':'d14'].mul(1/df['ratspercage'],axis=0)

foodintake_NR = df_days[df['diet'] == 'NR'].mean(axis=1)
foodintake_PR = df_days[df['diet'] == 'PR'].mean(axis=1)

fi = [foodintake_NR, foodintake_PR]

# Creates figure and sets general parameters, e.g. size, column widths


fig1A = plt.figure(figsize=(3.1,2))
fig1A.subplots_adjust(wspace=0.01, hspace=0.6, top=0.85, bottom=0.25, left=0.15, right=0.95)

markersize=5

# Makes bodyweight subplot
ax1 = fig1A.add_subplot()
nr_mean.plot(yerr=nr_sem, linewidth=1, color='xkcd:charcoal', marker='o', markersize=markersize, markerfacecolor='white', capthick=1, elinewidth=0.75, capsize=1.5)
pr_mean.plot(yerr=pr_sem, linewidth=1, color=col['pr_cas'], marker='o', markersize=markersize, markerfacecolor='white', capthick=1, elinewidth=0.75, capsize=1.5)
ax1.set_ylim([450, 570])
ax1.set_xlim([-1, 16])
ax1.set_xticks([1,8,15])
ax1.set_xticklabels(['0', '7', '14'], fontsize=8)
ax1.set_yticks([450, 500, 550])
ax1.set_yticklabels([450, 500, 550], fontsize=8)
ax1.set_ylabel('Body weight (g)', fontsize=8)
ax1.set_xlabel('Days since diet switch', fontsize=8)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)




# Saves figure
fig1A.savefig(figsfolder+"body weight.pdf")



"""
Code for running stats using R

This requires R to be installed and an Rscript written. At the moment I am using
the R package, EZ, which makes running mixed, between-within ANOVAs simple, and
tests for sphericity etc as appropriate.

EZ can be installed using the command install.packages('ez') in R. The package
seems to work best in R3.4.4 or later.

An R script is written to run the analysis and print the results. This script
is then called by Rscript.exe via the subprocess module in Python.

If it won't run it is probably because R has updated to a newer version so the 
Rscriptpath needs to change. Look for the most up to date version of R in the 
Program Files folder and amend this line appropriately. You will also have to 
open R (via the command line) to install the EZ ANOVA package using the install
command given above.'

"""

from subprocess import PIPE, run

Rscriptpath = 'C:\\Program Files\\R\\R-4.0.3\\bin\\Rscript'
Rprogpath = statsfolder+"bw_fi_stats.R"

result = run([Rscriptpath, "--vanilla", Rprogpath], stdout=PIPE, stderr=PIPE, universal_newlines=True)

print(result.returncode, result.stderr, result.stdout)

# d0_stats = stats.ttest_ind(nrd0, prd0)
# d14_stats = stats.ttest_ind(nrd14, prd14)

# print(d0_stats)
# print(d14_stats)

# # Stats on food intake
# fi_stats = stats.ttest_ind(foodintake_NR, foodintake_PR)
# print(fi_stats)

# To make food intake graph with estimation stats

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


# https://acclab.github.io/DABEST-python-docs/tutorial.html


df1 = pd.DataFrame(fi[0])
df1.reset_index(inplace=True)
df1.columns = ["cage", "NR"]

df2 = pd.DataFrame(fi[1])
df2.reset_index(inplace=True)
df2.columns = ["cage", "PR"]

df_new = pd.concat([df1, df2], sort=True)



# Creates figure and sets general parameters, e.g. size, column widths

gs = gridspec.GridSpec(1, 2, width_ratios=[2,1], wspace=0.2)
fig_s1B = plt.figure(figsize=(2,2))
fig_s1B.subplots_adjust(wspace=0.00, hspace=0.6, top=0.85, bottom=0.25, left=0.2, right=0.8)
    
ax2 = fig_s1B.add_subplot(gs[0,0])
_, barx, _, _ = tp.barscatter(fi, barfacecoloroption='individual',
                  barwidth = 0.8,
                  barfacecolor = [col['nr_cas'], col['pr_cas']],
                  scatteredgecolor = ['xkcd:charcoal'],
                  scattersize = 20,
                  ax=ax2)
ax2.set_yticks([0, 10, 20, 30])
ax2.set_xlim([0.25,2.75])
ax2.set_ylim([0, 35])
ax2.set_ylabel("Average food intake (g/day)", fontsize=8)

grouplabel=['NR', 'PR']

for x, label in zip(barx, grouplabel):
    ax2.text(x, -1, label, ha="center", va="top")
    
ax3 = fig_s1B.add_subplot(gs[0,1])
    
est_data = db.load(data=df_new, idx=("NR", "PR"))

e = est_data.mean_diff
results = e.results

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
    v = ax3.violinplot(current_bootstrap[~np.isinf(current_bootstrap)],
                                  positions=[tick],
                                  **violinplot_kwargs)



    halfviolin_alpha=0.7
    halfviolin(v, fill_color=fcolors[j], alpha=halfviolin_alpha)

    ytick_color="black"
    es_marker_size=4

#     # Plot the effect size.
    ax3.plot([tick], current_effsize, marker='o',
                        color=ytick_color,
                        markersize=es_marker_size)
    # Plot the confidence interval.
    ax3.plot([tick, tick],
                        [current_ci_low, current_ci_high],
                        linestyle="-",
                        color=ytick_color,
                        # linewidth=group_summary_kwargs['lw'],
                        linewidth=1)
    
ax3.spines["bottom"].set_visible(False)
ax3.spines["right"].set_visible(True)
ax3.spines["left"].set_visible(False)

ax3.yaxis.set_ticks_position("right")
ax3.yaxis.set_label_position("right")

# ax1.set_ylim([16, 34])
lims = ax2.get_ylim()
nr_mean = np.mean(fi[0])
lims2 = lims-nr_mean
ax3.set_ylim(lims2)

ax3.axhline(0, color='k', zorder=-20)

ax3.set_xticks([2])
ax3.set_xticklabels(["PR - NR"])
ax3.set_xlim([1.8, 2.5])

ax3.set_ylabel("Mean difference", rotation=270, fontsize=8, va="bottom")


fig_s1B.savefig(figsfolder+"foodintake.pdf")

