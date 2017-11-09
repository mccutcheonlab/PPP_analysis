# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 08:47:56 2017

@author: Jaime
"""

# Analysis of PPP1 grouped data
# Need to run PPP1_analysis first to load sessions into
# Choice data
import string
import pandas as pd
import matplotlib as mpl

def choicetest(x):
    choices = []
    for trial, trial_off in zip(x.trialsboth, x.trialsboth_off):
        leftlick = [x for x in x.licksL if (x > trial) and (x < trial_off)]
        rightlick = [x for x in x.licksR if (x > trial) and (x < trial_off)]
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

def side2subs(x):
    
    x.forcedtrials = {}
    
    for subs in ['cas', 'malt']:
        if subs in x.bottleL and subs in x.bottleR:
            print('Same substance in both bottles!')
        if subs in x.bottleL:
            x.forcedtrials[subs] = x.trialsLSnips
        if subs in x.bottleR:
            x.forcedtrials[subs] = x.trialsRSnips

def prefhistFig(ax1, ax2, df, factor1, factor2):
    dietmsk = df.diet == 'NR'

    shadedError(ax1, df[factor1][dietmsk], linecolor='black')
    ax1 = shadedError(ax1, df[factor2][dietmsk], linecolor='xkcd:bluish grey')
#    ax1.set_xticks([0,10,20,30])
#    ax1.set_xticklabels(['0', '20', '40', '60'])
    
    shadedError(ax2, df[factor1][~dietmsk], linecolor='xkcd:kelly green')
    ax2 = shadedError(ax2, df[factor2][~dietmsk], linecolor='xkcd:light green')
#    ax2.set_xticks([0,10,20,30])
#    ax2.set_xticklabels(['0', '20', '40', '60'])

def shadedError(ax, yarray, linecolor='black', errorcolor = 'xkcd:silver'):
    yarray = np.array(yarray)
    y = np.mean(yarray)
    yerror = np.std(yarray)/np.sqrt(len(yarray))
    x = np.arange(0, len(y))
    ax.plot(x, y, color=linecolor)
    ax.fill_between(x, y-yerror, y+yerror, color=errorcolor, alpha=0.4)
    
    return ax

for i in rats:
    for j in ['s11']:
        x = rats[i].sessions[j]
        ratkey = i
              
        x.choices = choicetest(x)
        x.pref = prefcalc(x)
        side2subs(x)

#def makemeans

df = pd.DataFrame([x for x in rats])
df.insert(1,'diet', [rats[x].dietgroup for x in rats])
df.insert(2,'choices',[[(rats[x].sessions[j].choices)] for x in rats])
df.insert(3,'pref', [rats[x].sessions[j].pref for x in rats])

df.insert(4,'forcedtrialsCas', [np.mean(rats[x].sessions[j].forcedtrials['cas'], axis=0) for x in rats])
df.insert(5,'forcedtrialsMalt', [np.mean(rats[x].sessions[j].forcedtrials['malt'], axis=0) for x in rats])

# Figure to show malt vs cas in PR vs NR
mpl.rcParams['figure.subplot.wspace'] = 0.1
mpl.rcParams['figure.subplot.left'] = 0.15
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(3.2, 2.4))

prefhistFig(ax[0], ax[1], df, 'forcedtrialsCas', 'forcedtrialsMalt')
#fig.text(0.55, 0.04, 'Time (min)', ha='center')
#ax[0].set_ylabel('Licks per 2 min')



np.shape(df.forcedtrialsCas[1])
#df.columns = ['choices']

