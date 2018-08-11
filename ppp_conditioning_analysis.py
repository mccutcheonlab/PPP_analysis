# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 13:04:03 2018

For analysis of conditioning sessions

@author: James Rig
"""

import scipy.io as sio
import JM_general_functions as jmf
import JM_custom_figs as jmfig

import os
import string
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import dill

try:
    type(sessions)
    print('Using existing data')
except NameError:
    print('Loading in data from pickled file')
    try:
        pickle_in = open('R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\ppp_rats_cond1.pickle', 'rb')
    except FileNotFoundError:
        print('Cannot access pickled file')
    sessions, rats = dill.load(pickle_in)
    
cond_sessions = {}
for session in sessions:
    x = sessions[session]
    try:
        len(x.data)
        cond_sessions[x.sessionID] = x
        
    except AttributeError:
        pass

rats = {}

for session in cond_sessions:
    x = cond_sessions[session]
    if x.rat not in rats.keys():
        rats[x.rat] = x.diet

cas_sessions = ['cond1-cas1', 'cond1-cas2']
malt_sessions = ['cond1-malt1', 'cond1-malt2']  

df_licks = pd.DataFrame([x for x in rats])
df_licks['diet'] = [rats.get(x) for x in rats]

for cas, malt in zip(cas_sessions, malt_sessions):
    df_licks[cas] = [cond_sessions[x].cas['lickdata']['total'] for x in cond_sessions if cond_sessions[x].sessiontype == cas]
    df_licks[malt] = [cond_sessions[x].malt['lickdata']['total'] for x in cond_sessions if cond_sessions[x].sessiontype == malt]

df_lickpeak = pd.DataFrame([x for x in rats])
df_lickpeak['diet'] = [rats.get(x) for x in rats]

for cas, malt, n in zip(cas_sessions, malt_sessions, [2,4]):
    df_lickpeak[cas] = [np.mean(cond_sessions[x].cas['snips_licks']['peak']) for x in cond_sessions if cond_sessions[x].sessiontype == cas]
    df_lickpeak[malt] = [np.mean(cond_sessions[x].malt['snips_licks']['peak']) for x in cond_sessions if cond_sessions[x].sessiontype == malt]

df_sipperpeak = pd.DataFrame([x for x in rats])
df_sipperpeak['diet'] = [rats.get(x) for x in rats]

for cas, malt in zip(cas_sessions, malt_sessions):
    df_sipperpeak[cas] = [np.mean(cond_sessions[x].cas['snips_sipper']['peak']) for x in cond_sessions if cond_sessions[x].sessiontype == cas]
    df_sipperpeak[malt] = [np.mean(cond_sessions[x].malt['snips_sipper']['peak']) for x in cond_sessions if cond_sessions[x].sessiontype == malt]

def condfigs(df, keys, dietmsk, cols, ax):
    
    a = [[df[keys[0]][dietmsk], df[keys[1]][dietmsk]],
          [df[keys[2]][dietmsk], df[keys[3]][dietmsk]]]

    ax, barx, _, _ = jmfig.barscatter(a, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = cols,
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 scattersize = 100,
                 ax=ax)

    return barx



figcond, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
df = df_licks
dietmsk = df['diet'] == 'NR'
cols = ['xkcd:silver']*2 + ['white']*2
barx = condfigs(df, cas_sessions+malt_sessions, dietmsk, cols, ax[0])

dietmsk = df['diet'] == 'PR'
cols = ['xkcd:kelly green']*2 + ['xkcd:light green']*2
barx = condfigs(df, cas_sessions+malt_sessions, dietmsk, cols, ax[1])

ax[0].set_ylabel('Licks')
ax[0].set_yticks([0, 1000, 2000, 3000, 4000])

yrange = ax[0].get_ylim()[1] - ax[0].get_ylim()[0]
grouplabel=['Casein', 'Maltodextrin']
barlabels=['1','2','3','4']
barlabeloffset=ax[0].get_ylim()[0] - yrange*0.04
grouplabeloffset=ax[0].get_ylim()[0] - yrange*0.12
for ax in ax:
    for x, label in zip(barx, barlabels):
        ax.text(x, barlabeloffset, label, va='top', ha='center')
    for x, label in zip([1,2], grouplabel):
        ax.text(x, grouplabeloffset, label, va='top', ha='center')
    





