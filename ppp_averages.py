# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 08:47:56 2017

@author: Jaime
"""

# Analysis of PPP1 grouped data

# Choice data
import scipy.io as sio
import JM_general_functions as jmf
import JM_custom_figs as jmfig

import os
import string
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#mpl.style.use('classic')

mpl.rcParams['figure.figsize'] = (4.8, 3.2)
mpl.rcParams['figure.dpi'] = 100

mpl.rcParams['font.size'] = 12.0
mpl.rcParams['axes.labelsize'] = 'medium'
mpl.rcParams['ytick.labelsize'] = 'small'

mpl.rcParams['figure.subplot.left'] = 0.15
mpl.rcParams['figure.subplot.bottom'] = 0.20

mpl.rcParams['errorbar.capsize'] = 5

mpl.rcParams['savefig.transparent'] = True

mpl.rcParams['axes.spines.top']=False
mpl.rcParams['axes.spines.right']=False

import dill

def choicetest(x):
    choices = []
    for trial, trial_off in zip(x.both['sipper'], x.both['sipper_off']):
        leftlick = [L for L in x.left['licks'] if (L > trial) and (L < trial_off)]
        rightlick = [L for L in x.right['licks'] if (L > trial) and (L < trial_off)]
        if len(leftlick) > 0:
            if len(rightlick) > 0:
                if leftlick < rightlick:
                    choices.append(x.bottleL[:3])
                else:
                    choices.append(x.bottleR[:3])
            else:
                choices.append(x.bottleL[:3])
        elif len(rightlick) > 0:
            choices.append(x.bottleR[:3])
        else:
            choices.append('missed')
    
    return choices

def prefcalc(x):
    cas = sum([1 for trial in x.choices if trial == 'cas'])
    malt = sum([1 for trial in x.choices if trial == 'mal'])
    pref = cas/(cas+malt)
    
    return pref


def doublesnipFig(ax1, ax2, df, diet, factor1, factor2):
    dietmsk = df.diet == diet    
    ax1.axis('off')
    ax2.axis('off')

    shadedError(ax1, df[factor1][dietmsk], linecolor='black')
    shadedError(ax1, df[factor2][dietmsk], linecolor='xkcd:bluish grey')
    ax1.plot([50,50], [0.02, 0.04], c='k')
    ax1.text(45, 0.03, '2% \u0394F', verticalalignment='center', horizontalalignment='right')
    
    shadedError(ax2, df[factor1][~dietmsk], linecolor='xkcd:kelly green')
    shadedError(ax2, df[factor2][~dietmsk], linecolor='xkcd:light green')
    ax2.plot([250,300], [-0.03, -0.03], c='k')
    ax2.text(275, -0.035, '5 s', verticalalignment='top', horizontalalignment='center')

def shadedError(ax, yarray, linecolor='black', errorcolor = 'xkcd:silver'):
    yarray = np.array(yarray)
    y = np.mean(yarray)
    yerror = np.std(yarray)/np.sqrt(len(yarray))
    x = np.arange(0, len(y))
    line = ax.plot(x, y, color=linecolor)
    patch = ax.fill_between(x, y-yerror, y+yerror, color=errorcolor, alpha=0.4)
    
    return ax, line, patch

def excluderats(rats, ratstoexclude):  
    ratsX = [x for x in rats if x not in ratstoexclude]        
    return ratsX

def makemeansnips(snips, noiseindex):
    if len(noiseindex) > 0:
        trials = np.array([i for (i,v) in zip(snips, noiseindex) if not v])
    meansnip = np.mean(trials, axis=0)
        
    return meansnip

def data2obj2D(data):
    obj = np.empty((np.shape(data)[0], np.shape(data)[1]), dtype=np.object)
    for i,x in enumerate(data):
        for j,y in enumerate(x):
            obj[i][j] = np.array(y)
    return obj

def data2obj1D(data):
    obj = np.empty(len(data), dtype=np.object)
    for i,x in enumerate(data):
        obj[i] = np.array(x)  
    return obj

def choicefig(df, keys, ax):
    dietmsk = df.diet == 'NR'
    
    a = [[df[keys[0]][dietmsk], df[keys[1]][dietmsk], df[keys[2]][dietmsk]],
          [df[keys[0]][~dietmsk], df[keys[1]][~dietmsk], df[keys[2]][~dietmsk]]]

    x = data2obj2D(a)
    
    cols = ['xkcd:silver', 'xkcd:kelly green']
    
    ax, x, _, _ = jmfig.barscatter(x, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [cols[0], cols[1], cols[1], cols[1], cols[0], cols[0]],
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel=['NR \u2192 PR', 'PR \u2192 NR'],
                 scattersize = 100,
                 ax=ax)
    
def onedaypreffig(df, key, ax):
    dietmsk = df.diet == 'NR'
    a = data2obj1D([df[key][dietmsk], df[key][~dietmsk]])

        
    jmfig.barscatter(a, barfacecoloroption = 'between', barfacecolor = ['xkcd:silver', 'xkcd:kelly green'],
                         scatteredgecolor = ['black'],
                         scatterlinecolor = 'black',
                         grouplabel=['NR', 'PR'],
                         barwidth = 0.8,
                         scattersize = 80,
                         ylabel = 'Casein preference',
                         ax=ax)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim([0.25,2.75])
    ax.set_ylim([0, 1.1])

def peakresponsebargraph(df, keys, ax):
    dietmsk = df.diet == 'NR'
    
    a = [[df[keys[0]][dietmsk], df[keys[1]][dietmsk]],
          [df[keys[0]][~dietmsk], df[keys[1]][~dietmsk]]]

    x = data2obj2D(a)
    
    cols = ['xkcd:silver', 'w', 'xkcd:kelly green', 'xkcd:light green']
    
    ax, x, _, _ = jmfig.barscatter(x, paired=True,
                 barfacecoloroption = 'individual',
                 barfacecolor = [cols[0], cols[1], cols[2], cols[3]],
                 scatteredgecolor = ['xkcd:charcoal'],
                 scatterlinecolor = 'xkcd:charcoal',
                 grouplabel=['NR', 'PR'],
                 scattersize = 100,
                 ax=ax)
    ax.set_ylim([-.02, 0.15])
    ax.set_yticks([0, 0.05, 0.1, 0.15])
#    ax.set_ylabel('\u0394F')

# Looks for existing data and if not there loads pickled file
try:
    type(sessions)
    print('Using existing data')
except NameError:
    print('Loading in data from pickled file')
    try:
        pickle_in = open('R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\ppp_rats.pickle', 'rb')
    except FileNotFoundError:
        print('Cannot access pickled file')
    rats = dill.load(pickle_in)

#ratsX = excluderats(rats, ['PPP1.8'])
#ratsX = rats
#
#testsessions = ['s10', 's11', 's16']
#
#for i in rats:
#    for j in testsessions:
#        x = rats[i].sessions[j]
#        ratkey = i
#              
#        x.choices = choicetest(x)
#        x.pref = prefcalc(x)
#
#df1 = pd.DataFrame([x for x in rats])
#df1.insert(1,'diet', [rats[x].dietgroup for x in rats])
#
#for j, n, ch, pr in zip(testsessions, [2,4,6], ['choices1', 'choices2', 'choices3'], ['pref1', 'pref2', 'pref3']):
#    df1.insert(n, ch, [(rats[x].sessions[j].choices) for x in rats])
#    df1.insert(n+1, pr, [rats[x].sessions[j].pref for x in rats])
#    
#for n, ch, cas, malt in zip([8,10,12],
#                            ['choices1', 'choices2', 'choices3'],
#                            ['ncas1', 'ncas2', 'ncas3'],
#                            ['nmalt1', 'nmalt2', 'nmalt3']):
#    df1.insert(n, cas, [c.count('cas') for c in df1[ch]])
#    df1.insert(n+1, malt, [m.count('mal') for m in df1[ch]])
#
##df1.to_csv('R:\\DA_and_Reward\\es334\\PPP1\\output\\choice-and-pref.csv')
#df1.to_csv('C:\\Users\\jaimeHP\\Documents\\GitHub\\PPP_analysis\\output\\choice-and-pref.csv')
## Figure showing one day preference data
#    
#mpl.rcParams['figure.subplot.left'] = 0.30
#fig = plt.figure(figsize=(3.2, 4.0))
#ax = plt.subplot(1,1,1)
#onedaypreffig(df1, 'pref1', ax)
#
##plt.savefig('R:/DA_and_Reward/es334/PPP1/figures/onedaypref.eps')
#
## Figure showing casein preference across all three test sessions
#mpl.rcParams['figure.subplot.left'] = 0.15
#fig = plt.figure(figsize=(4.4, 4.0))
#ax = plt.subplot(1,1,1)                
# 
#choicefig(df1, ['pref1', 'pref2', 'pref3'], ax)
#ax.set_ylabel('Casein preference')
#plt.yticks([0, 0.5, 1.0])
##plt.savefig('R:/DA_and_Reward/es334/PPP1/figures/alldayspref.eps')
#
## Figure showing licks divide into free choice and forced choice
#df2 = pd.DataFrame([x for x in rats])
#df2.insert(1,'diet', [rats[x].dietgroup for x in rats])
#
#for j, n, cas, malt in zip(testsessions, [2,4,6],
#                        ['forced1-cas', 'forced2-cas', 'forced3-cas'],
#                        ['forced1-malt', 'forced2-malt', 'forced3-malt']):
#    df2.insert(n, cas, [rats[x].sessions[j].cas['nlicks-forced'] for x in rats])
#    df2.insert(n+1, malt, [rats[x].sessions[j].malt['nlicks-forced'] for x in rats])
#
##df2.to_csv('R:\\DA_and_Reward\\es334\\PPP1\\output\\licks-forced.csv')
#df2.to_csv('C:\\Users\\jaimeHP\\Documents\\GitHub\\PPP_analysis\\output\\licks-forced.csv')
#
#df3 = pd.DataFrame([x for x in rats])
#df3.insert(1,'diet', [rats[x].dietgroup for x in rats])
#
#for j, n, cas, malt in zip(testsessions, [2,4,6],
#                        ['free1-cas', 'free2-cas', 'free3-cas'],
#                        ['free1-malt', 'free2-malt', 'free3-malt']):
#    df3.insert(n, cas, [rats[x].sessions[j].cas['nlicks-free'] for x in rats])
#    df3.insert(n+1, malt, [rats[x].sessions[j].malt['nlicks-free'] for x in rats])
#
##df3.to_csv('R:\\DA_and_Reward\\es334\\PPP1\\output\\licks-free.csv')
#df3.to_csv('C:\\Users\\jaimeHP\\Documents\\GitHub\\PPP_analysis\\output\\licks-free.csv')
#
### Creating new dataframe for photometry data so I can exclude rats
#
#ratsX = excluderats(rats, ['PPP1.8'])
#
#df4 = pd.DataFrame([x for x in ratsX])
#df4.insert(1,'diet', [rats[x].dietgroup for x in ratsX])
#
#for j, n, cas, malt in zip(testsessions, [2,4,6], ['cas1', 'cas2', 'cas3'], ['malt1', 'malt2', 'malt3']):
#    df4.insert(n, cas, [np.mean(rats[x].sessions[j].cas['snips_sipper']['diff'], axis=0) for x in ratsX])
#    df4.insert(n+1, malt, [np.mean(rats[x].sessions[j].malt['snips_sipper']['diff'], axis=0) for x in ratsX])
#
#for j, n, cas, malt in zip(testsessions, [8, 10, 12], ['cas1_licks', 'cas2_licks', 'cas3_licks'], ['malt1_licks', 'malt2_licks', 'malt3_licks']):
#    df4.insert(n, cas, [np.mean(rats[x].sessions[j].cas['snips_licks']['diff'], axis=0) for x in ratsX])
#    df4.insert(n+1, malt, [np.mean(rats[x].sessions[j].malt['snips_licks']['diff'], axis=0) for x in ratsX])
#
#for j, n, cas, malt in zip(testsessions, [14, 16, 18],
#                           ['cas1_licks_forced', 'cas2_licks_forced', 'cas3_licks_forced'],
#                           ['malt1_licks_forced', 'malt2_licks_forced', 'malt3_licks_forced']):
#    df4.insert(n, cas, [np.mean(rats[x].sessions[j].cas['snips_licks_forced']['diff'], axis=0) for x in ratsX])
#    df4.insert(n+1, malt, [np.mean(rats[x].sessions[j].malt['snips_licks_forced']['diff'], axis=0) for x in ratsX])
#
#for j, n, cas, malt in zip(testsessions, [20, 22, 24],
#                           ['cas1_licks_peak', 'cas2_licks_peak', 'cas3_licks_peak'],
#                           ['malt1_licks_peak', 'malt2_licks_peak', 'malt3_licks_peak']):
#    df4.insert(n, cas, [np.mean(rats[x].sessions[j].cas['snips_licks_forced']['peak'], axis=0) for x in ratsX])
#    df4.insert(n+1, malt, [np.mean(rats[x].sessions[j].malt['snips_licks_forced']['peak'], axis=0) for x in ratsX])
#    
#df4['pref1_peak_delta'] = df4['cas1_licks_peak'] - df4['malt1_licks_peak']
#df4['pref2_peak_delta'] = df4['cas2_licks_peak'] - df4['malt2_licks_peak']
#df4['pref3_peak_delta'] = df4['cas3_licks_peak'] - df4['malt3_licks_peak']
#    
## Figure to show malt vs cas in PR vs NR
#mpl.rcParams['figure.subplot.hspace'] = 0.15
#mpl.rcParams['figure.subplot.left'] = 0.05
#mpl.rcParams['figure.subplot.right'] = 0.95
#mpl.rcParams['figure.subplot.top'] = 0.95
#mpl.rcParams['figure.subplot.bottom'] = 0.05
#
##mpl.rcParams['axes.spines.bottom']=False
##mpl.rcParams['axes.spines.left']=False
#
#fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(3, 6))
#doublesnipFig(ax[0], ax[1], df4, 'NR', 'cas1', 'malt1')
#
#
#fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(3, 6))
#doublesnipFig(ax[0], ax[1], df4, 'NR', 'cas1_licks_forced', 'malt1_licks_forced')
##plt.savefig('R:/DA_and_Reward/es334/PPP1/figures/pref1photo_licks.eps')
#
#
#fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(3, 6))
#doublesnipFig(ax[0], ax[1], df4, 'NR', 'cas2_licks', 'malt2_licks')
##plt.savefig('R:/DA_and_Reward/es334/PPP1/figures/pref2photo_licks.eps')
#
#fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(3, 6))
#doublesnipFig(ax[0], ax[1], df4, 'NR', 'cas3_licks', 'malt3_licks')
##plt.savefig('R:/DA_and_Reward/es334/PPP1/figures/pref3photo_licks.eps')
#
## Figure for peak responses to casein and malto licks
##mpl.rcParams['figure.subplot.left'] = 0.30
##fig = plt.figure(figsize=(8, 4))
#fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(8, 4))
#peakresponsebargraph(df4, ['cas1_licks_peak', 'malt1_licks_peak'], ax[0])
##plt.savefig('R:/DA_and_Reward/es334/PPP1/figures/peak1_licks.eps')
#
#peakresponsebargraph(df4, ['cas2_licks_peak', 'malt2_licks_peak'], ax[1])
##plt.savefig('R:/DA_and_Reward/es334/PPP1/figures/peak1_licks.eps')
#
#peakresponsebargraph(df4, ['cas3_licks_peak', 'malt3_licks_peak'], ax[2])
#
#ax[0].set_ylabel('\u0394F')
##plt.savefig('R:/DA_and_Reward/es334/PPP1/figures/allpeaks_licks.eps')
#
#
##
### TO DO!!!
### remove noise trials from grouped data
### figure out a way of excluding certain rats (e.g. PPP1.8) maybe just a line that removes at beginning of this code
##