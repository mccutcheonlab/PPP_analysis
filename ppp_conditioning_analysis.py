# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 13:04:03 2018

For analysis of conditioning sessions

@author: James Rig
"""

import scipy.io as sio

import os
import string
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import dill

import trompy as tp

from ppp_pub_figs_settings import *

try:
    pickle_in = open('C:\\Github\\PPP_analysis\\data\\ppp_cond.pickle', 'rb')
except FileNotFoundError:
    print('Cannot access pickled file')

cond_sessions, rats = dill.load(pickle_in)

figsfolder = "C:\\Users\\jmc010\\Dropbox\\Publications in Progress\\PPP Paper\\04_JNS\\Figs\\"
    
rats = {}

for session in cond_sessions:
    x = cond_sessions[session]
    if x.rat not in rats.keys():
        rats[x.rat] = x.diet
    
cas_sessions = ['cond1-cas1', 'cond1-cas2']
malt_sessions = ['cond1-malt1', 'cond1-malt2']  

df_licks = pd.DataFrame([x for x in rats], columns=['rat'])
df_licks['diet'] = [rats.get(x) for x in rats]

# s = cond_sessions['PPP4-1_s6']

for cas, malt in zip(cas_sessions, malt_sessions):
    df_licks[cas] = [np.float(cond_sessions[x].cas) for x in cond_sessions if cond_sessions[x].sessiontype == cas]
    df_licks[malt] = [np.float(cond_sessions[x].malt) for x in cond_sessions if cond_sessions[x].sessiontype == malt]

df_licks['cond1-cas-all'] = df_licks['cond1-cas1'] + df_licks['cond1-cas2']
df_licks['cond1-malt-all'] = df_licks['cond1-malt1'] + df_licks['cond1-malt2']

# df_lickpeak = pd.DataFrame([x for x in rats])
# df_lickpeak['diet'] = [rats.get(x) for x in rats]

# for cas, malt, n in zip(cas_sessions, malt_sessions, [2,4]):
#     df_lickpeak[cas] = [np.mean(cond_sessions[x].cas['snips_licks']['peak']) for x in cond_sessions if cond_sessions[x].sessiontype == cas]
#     df_lickpeak[malt] = [np.mean(cond_sessions[x].malt['snips_licks']['peak']) for x in cond_sessions if cond_sessions[x].sessiontype == malt]

# df_sipperpeak = pd.DataFrame([x for x in rats])
# df_sipperpeak['diet'] = [rats.get(x) for x in rats]

# for cas, malt in zip(cas_sessions, malt_sessions):
#     df_sipperpeak[cas] = [np.mean(cond_sessions[x].cas['snips_sipper']['peak']) for x in cond_sessions if cond_sessions[x].sessiontype == cas]
#     df_sipperpeak[malt] = [np.mean(cond_sessions[x].malt['snips_sipper']['peak']) for x in cond_sessions if cond_sessions[x].sessiontype == malt]

def condfigs(df, keys, dietmsk, cols, ax):
    
    a = [[df[keys[0]][dietmsk], df[keys[1]][dietmsk]],
          [df[keys[2]][dietmsk], df[keys[3]][dietmsk]]]

    ax, barx, _, _ = tp.barscatter(a, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = cols,
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 scattersize = 50,
                 ax=ax)

    return barx

figcond, ax = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={
    "left": 0.2, "bottom": 0.2})
df = df_licks
dietmsk = df['diet'] == 'NR'
cols = [col["nr_malt"]]*2 + [col["nr_cas"]]*2
barx = condfigs(df, malt_sessions+cas_sessions, dietmsk, cols, ax[0])

dietmsk = df['diet'] == 'PR'
cols = [col["pr_malt"]]*2 + [col["pr_cas"]]*2
barx = condfigs(df, malt_sessions+cas_sessions, dietmsk, cols, ax[1])

ax[0].set_ylabel('Licks', fontsize=8)
ax[0].set_yticks([0, 1000, 2000, 3000, 4000])

yrange = ax[0].get_ylim()[1] - ax[0].get_ylim()[0]
grouplabel=['Maltodextrin', 'Casein', ]
barlabels=['1','2','1','2']
barlabeloffset=ax[0].get_ylim()[0] - yrange*0.02
grouplabeloffset=ax[0].get_ylim()[0] - yrange*0.08
for ax in ax:
    for x, label in zip(barx, barlabels):
        ax.text(x, barlabeloffset, label, va='top', ha='center', fontsize=8)
    for x, label in zip([1,2], grouplabel):
        ax.text(x, grouplabeloffset, label, va='top', ha='center', fontsize=8)
        
figcond.savefig(figsfolder + 'figS1b_conditioning.pdf')
    


# Preparing data for stats
df_cond1_behav = df_licks
df_cond1_behav.set_index(['rat', 'diet'], inplace=True)

pickle_out = open('C:\\Github\\PPP_analysis\\data\\ppp_dfs_cond1.pickle', 'wb')
dill.dump(df_cond1_behav, pickle_out)
pickle_out.close()



# for session in cond_sessions:
#     x = cond_sessions[session]
#     try:
#         print(np.shape(x.cas['snips_sipper']['peak']))
#     except:
#         print(np.shape(x.malt['snips_sipper']['peak']))


# binnedtrials = ['1-5','6-10','11-15','16-20','21-25','26-30','31-35','36-40']
# trialindex = []

# df_condtrials_cas = pd.DataFrame([x for x in rats])
# df_condtrials_cas['diet'] = [rats.get(x) for x in rats]

# need to add trials to average to this code!!!!

#for trials in zip(binnedtrials):
#    df[trials] = [np.mean(cond_sessions[x].cas['snips_sipper']['peak']) for x in cond_sessions if cond_sessions[x].sessiontype == 'cond1-cas1']
#    
