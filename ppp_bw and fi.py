# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:07:17 2018

@author: James Rig
"""

import JM_general_functions as jmf
import JM_custom_figs as jmfig
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#import rpy2.robjects as ro
#from rpy2.robjects import r, pandas2ri, numpy2ri
#pandas2ri.activate()
#numpy2ri.activate()

from scipy import stats

col={}
col['np_cas'] = 'xkcd:silver'
col['np_malt'] = 'white'
col['lp_cas'] = 'xkcd:kelly green'
col['lp_malt'] = 'xkcd:light green'

# Loads in data

xlfile = "C:\\GitHub\\PPP_analysis\\data\\PPP_body weight and food intake.xlsx"
statsfolder = "C:\\GitHub\\PPP_analysis\\stats\\"
figsfolder = "C:\\GitHub\\PPP_analysis\\figs\\"

# Body weight data
df = pd.read_excel(xlfile, sheet_name='PPP_bodyweight')
df.set_index('rat', inplace=True)

df.drop('cage', axis=1, inplace=True)

df.drop(['PPP1.8'], inplace=True)
df.drop(['PPP3.1'], inplace=True)
df.drop(['PPP3.6'], inplace=True)
df.drop(['PPP3.7'], inplace=True)
df.drop(['PPP3.8'], inplace=True)
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

df_days = df.loc[:,'d0':'d14'].mul(1/df['ratspercage'],axis=0)

foodintake_NR = df_days[df['diet'] == 'NR'].mean(axis=1)
foodintake_PR = df_days[df['diet'] == 'PR'].mean(axis=1)

fi = [foodintake_NR, foodintake_PR]

# Creates figure and sets general parameters, e.g. size, column widths

gs = gridspec.GridSpec(1, 2, width_ratios=[2,1], wspace=0.5)
fig1 = plt.figure(figsize=(5,2))
fig1.subplots_adjust(wspace=0.01, hspace=0.6, top=0.85, bottom=0.25, left=0.15, right=0.95)

# Makes bodyweight subplot
ax1 = fig1.add_subplot(gs[0,0])
nr_mean.plot(yerr=nr_sem, linewidth=1, color='xkcd:charcoal', marker='o', markerfacecolor='white', capthick=1, elinewidth=1, capsize=2)
pr_mean.plot(yerr=pr_sem, linewidth=1, color=col['lp_cas'], marker='o', markerfacecolor='white', capthick=1, elinewidth=1, capsize=2)
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

# Makes food intake subplot
ax2 = fig1.add_subplot(gs[0,1])
jmfig.barscatter(fi, barfacecoloroption='individual',
                 barwidth = 0.8,
                 barfacecolor = [col['np_cas'], col['lp_cas']],
                 scatteredgecolor = ['xkcd:charcoal'],
                 scattersize = 40,
                 ylabel = 'Average food intake (g/day)',
                 grouplabel=['NR', 'PR'],
                 ax=ax2)
ax2.set_yticks([0, 10, 20, 30])
ax2.set_xlim([0.25,2.75])
ax2.set_ylim([0, 35])

# Saves figure
fig1.savefig(figsfolder+"body weight and food intake.eps")



"""
Code for running stats using R

This requires R to be installed and an Rscript written. At the moment I am using
the R package, EZ, which makes running mixed, between-within ANOVAs simple, and
tests for sphericity etc as appropriate.

EZ can be installed using the command install.packages('ez') in R. The package
seems to work best in R3.4.4 or later.

An R script is written to run the analysis and print the results. This script
is then called by Rscript.exe via the subprocess module in Python.

"""

from subprocess import PIPE, run

Rscriptpath = 'C:\\Program Files\\R\\R-3.6.2\\bin\\Rscript'
Rprogpath = statsfolder+"bw_fi_stats.R"

result = run([Rscriptpath, "--vanilla", Rprogpath], stdout=PIPE, stderr=PIPE, universal_newlines=True)

print(result.returncode, result.stderr, result.stdout)

d0_stats = stats.ttest_ind(nrd0, prd0)
d14_stats = stats.ttest_ind(nrd14, prd14)

print(d0_stats)
print(d14_stats)

# Stats on food intake
fi_stats = stats.ttest_ind(foodintake_NR, foodintake_PR)
print(fi_stats)
