# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 08:47:56 2017

@author: Jaime
"""

# Analysis of PPP1 grouped data
# Need to run PPP1_analysis first to load sessions into
# Choice data
import os
import string
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['figure.figsize'] = (3.2, 2.4)
mpl.rcParams['figure.dpi'] = 100

mpl.rcParams['font.size'] = 8.0
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

#def side2subs(x):
#    
#    x.forcedtrials = {}
#    x.lickruns = {}
#    x.forcedtrialsx = {}
#    x.lickrunsx = {}
#    
#    for subs in ['cas', 'malt']:
#        if subs in x.bottleL and subs in x.bottleR:
#            print('Same substance in both bottles!')
#        if subs in x.bottleL:
#            x.forcedtrials[subs] = x.trialsLSnips
#            x.lickruns[subs] = x.licksLSnips            
#            x.forcedtrialsx[subs] = makemeansnips(x.trialsLSnips, x.trialsLnoise)
#            x.lickrunsx[subs] = makemeansnips(x.licksLSnips, x.licksLnoise)
#
#        if subs in x.bottleR:
#            x.forcedtrials[subs] = x.trialsRSnips
#            x.lickruns[subs] = x.licksRSnips
#            x.forcedtrialsx[subs] = makemeansnips(x.trialsRSnips, x.trialsRnoise)
#            x.lickrunsx[subs] = makemeansnips(x.licksRSnips, x.licksRnoise)

def doublesnipFig(ax1, ax2, df, factor1, factor2):
    dietmsk = df.diet == 'NR'    
    ax1.axis('off')
    ax2.axis('off')

    shadedError(ax1, df[factor1][dietmsk], linecolor='black')
    ax1 = shadedError(ax1, df[factor2][dietmsk], linecolor='xkcd:bluish grey')
    ax1.plot([50,50], [0.02, 0.04], c='k')
    ax1.text(45, 0.03, '2% \u0394F', verticalalignment='center', horizontalalignment='right')
    
    shadedError(ax2, df[factor1][~dietmsk], linecolor='xkcd:kelly green')
    ax2 = shadedError(ax2, df[factor2][~dietmsk], linecolor='xkcd:light green')
    ax2.plot([250,300], [-0.03, -0.03], c='k')
    ax2.text(275, -0.035, '5 s', verticalalignment='top', horizontalalignment='center')


def shadedError(ax, yarray, linecolor='black', errorcolor = 'xkcd:silver'):
    yarray = np.array(yarray)
    y = np.mean(yarray)
    yerror = np.std(yarray)/np.sqrt(len(yarray))
    x = np.arange(0, len(y))
    ax.plot(x, y, color=linecolor)
    ax.fill_between(x, y-yerror, y+yerror, color=errorcolor, alpha=0.4)
    
    return ax

def excluderats(rats, ratstoexclude):  
    ratsX = [x for x in rats if x not in ratstoexclude]
    print(ratsX) 
        
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
                 scattersize = 60,
                 ax=ax)
    
def onedaypreffig(df, key, ax):
    dietmsk = df.diet == 'NR'
    a = data2obj1D([df[key][dietmsk], df[key][~dietmsk]])

        
    jmfig.barscatter(a, barfacecoloroption = 'between', barfacecolor = ['xkcd:silver', 'xkcd:kelly green'],
                         scatteredgecolor = ['black'],
                         scatterlinecolor = 'black',
                         grouplabel=['NR', 'PR'],
                         barwidth = 0.8,
                         scattersize = 40,
                         ylabel = 'Casein preference',
                         ax=ax)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim([0.25,2.75])
    ax.set_ylim([0, 1.1])

# Looks for existing data and if not there loads pickled file
try:
    type(rats)
    print('Using existing data')
except NameError:
    print('Loading in data from pickled file')
    pickle_in = open('C:\\Users\\jaimeHP\\Documents\\rats.pickle', 'rb')
    rats = dill.load(pickle_in)

ratsX = excluderats(rats, ['PPP1.8'])
ratsX = rats

testsessions = ['s10', 's11', 's16']

for i in rats:
    for j in testsessions:
        x = rats[i].sessions[j]
        ratkey = i
              
        x.choices = choicetest(x)
        x.pref = prefcalc(x)
#        side2subs(x)

df = pd.DataFrame([x for x in ratsX])
df.insert(1,'diet', [rats[x].dietgroup for x in ratsX])

for j, n, ch, pr in zip(testsessions, [2,4,6], ['choices1', 'choices2', 'choices3'], ['pref1', 'pref2', 'pref3']):
    df.insert(n, ch, [[(rats[x].sessions[j].choices)] for x in ratsX])
    df.insert(n+1, pr, [rats[x].sessions[j].pref for x in ratsX])

# Figure showing one day preference data
    
mpl.rcParams['figure.subplot.left'] = 0.30
width = 1.8
fig = plt.figure(figsize=(width, width*(4/3)))
ax = plt.subplot(1,1,1)
onedaypreffig(df, 'pref1', ax)

plt.savefig('R:/DA_and_Reward/es334/PPP1/figures/onedaypref.eps')
plt.title('Casein preference')

# Figure showing casein preference across all three test sessions
fig = plt.figure(figsize=(2.4, 2.4))
ax = plt.subplot(1,1,1)                
 
choicefig(df, ['pref1', 'pref2', 'pref3'], ax)
ax.set_ylabel('Casein preference')
plt.yticks([0, 0.5, 1.0])
plt.savefig('R:/DA_and_Reward/es334/PPP1/figures/alldayspref.eps')

## Creating new dataframe for photometry data so I can exclude rats

ratsX = excluderats(rats, ['PPP1.8'])

df = pd.DataFrame([x for x in ratsX])
df.insert(1,'diet', [rats[x].dietgroup for x in ratsX])

for j, n, cas, malt in zip(testsessions, [2,4,6], ['cas1', 'cas2', 'cas3'], ['malt1', 'malt2', 'malt3']):
    df.insert(n, cas, [np.mean(rats[x].sessions[j].cas['snips_sipper']['diff'], axis=0) for x in ratsX])
    df.insert(n+1, malt, [np.mean(rats[x].sessions[j].malt['snips_sipper']['diff'], axis=0) for x in ratsX])

# Figure to show malt vs cas in PR vs NR
mpl.rcParams['figure.subplot.wspace'] = 0.1
mpl.rcParams['figure.subplot.left'] = 0.15

#mpl.rcParams['axes.spines.bottom']=False
#mpl.rcParams['axes.spines.left']=False

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6, 4))
doublesnipFig(ax[0], ax[1], df, 'cas1', 'malt1')

for j, n, cas, malt in zip(testsessions, [8, 10, 12], ['cas1_licks', 'cas2_licks', 'cas3_licks'], ['malt1_licks', 'malt2_licks', 'malt3_licks']):
    df.insert(n, cas, [np.mean(rats[x].sessions[j].cas['snips_licks']['diff'], axis=0) for x in ratsX])
    df.insert(n+1, malt, [np.mean(rats[x].sessions[j].malt['snips_licks']['diff'], axis=0) for x in ratsX])

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6, 4))
doublesnipFig(ax[0], ax[1], df, 'cas1_licks', 'malt1_licks')
plt.savefig('R:/DA_and_Reward/es334/PPP1/figures/pref1photo_licks.eps')


#
#
#np.shape(df.forcedtrialsCas[1])
##df.columns = ['choices']
#
## TO DO!!!
## remove noise trials from grouped data
## figure out a way of excluding certain rats (e.g. PPP1.8) maybe just a line that removes at beginning of this code
#
