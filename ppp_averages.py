# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 08:47:56 2017

@author: Jaime
"""

# Assembles data from PPP1 and PPP3 into pandas dataframes for plotting. Saves
# dataframes, df_behav and df_photo, as pickle object (ppp_dfs_pref)

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

def excluderats(rats, ratstoexclude):  
    ratsX = [x for x in rats if x not in ratstoexclude]        
    return ratsX

def makemeansnips(snips, noiseindex):
    if len(noiseindex) > 0:
        trials = np.array([i for (i,v) in zip(snips, noiseindex) if not v])
    meansnip = np.mean(trials, axis=0)
        
    return meansnip

def removenoise(snipdata):
    # returns blue snips with noisey ones removed
    new_snips = [snip for (snip, noise) in zip(snipdata['blue'], snipdata['noise']) if not noise]
    return new_snips

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
    sessions, rats = dill.load(pickle_in)

pref_sessions = {}
for session in sessions:
    x = sessions[session]
    try:
        len(x.data)
        pref_sessions[x.sessionID] = x
        
    except AttributeError:
        pass

rats = {}
included_sessions = []
for session in pref_sessions:
    x = pref_sessions[session]
    if x.rat not in rats.keys():
        rats[x.rat] = x.diet
    if x.session not in included_sessions:
        included_sessions.append(x.session)
        
for session in pref_sessions:
    x = pref_sessions[session]          
    x.choices = choicetest(x)
    x.pref = prefcalc(x)

df_behav = pd.DataFrame([x for x in rats], columns=['rat'])
df_behav['diet'] = [rats.get(x) for x in rats]
df_behav.set_index(['rat', 'diet'], inplace=True)

for j, ch, pr, cas, malt in zip(included_sessions,
                                ['choices1', 'choices2', 'choices3'],
                                ['pref1', 'pref2', 'pref3'],
                                ['ncas1', 'ncas2', 'ncas3'],
                                ['nmalt1', 'nmalt2', 'nmalt3']):
    df_behav[ch] = [pref_sessions[x].choices for x in pref_sessions if pref_sessions[x].session == j]
    df_behav[pr] = [pref_sessions[x].pref for x in pref_sessions if pref_sessions[x].session == j]
    df_behav[cas] = [c.count('cas') for c in df_behav[ch]]
    df_behav[malt] = [m.count('mal') for m in df_behav[ch]]

for j, forc_cas, forc_malt, free_cas, free_malt in zip(included_sessions,
                        ['forced1-cas', 'forced2-cas', 'forced3-cas'],
                        ['forced1-malt', 'forced2-malt', 'forced3-malt'],
                        ['free1-cas', 'free2-cas', 'free3-cas'],
                        ['free1-malt', 'free2-malt', 'free3-malt']):
    df_behav[forc_cas] = [pref_sessions[x].cas['nlicks-forced'] for x in pref_sessions if pref_sessions[x].session == j]
    df_behav[forc_malt] = [pref_sessions[x].malt['nlicks-forced'] for x in pref_sessions if pref_sessions[x].session == j]
    df_behav[free_cas] = [pref_sessions[x].cas['nlicks-free'] for x in pref_sessions if pref_sessions[x].session == j]
    df_behav[free_malt] = [pref_sessions[x].malt['nlicks-free'] for x in pref_sessions if pref_sessions[x].session == j]

# Assembles dataframe with photometry data
df_photo = pd.DataFrame([x for x in rats], columns=['rat'])
df_photo['diet'] = [rats.get(x) for x in rats]
df_photo.set_index(['rat', 'diet'], inplace=True)

for j, c_sip_diff, m_sip_diff, c_licks_diff, m_licks_diff in zip(included_sessions,
                             ['cas1_sip', 'cas2_sip', 'cas3_sip'],
                             ['malt1_sip', 'malt2_sip', 'malt3_sip'],
                             ['cas1_licks', 'cas2_licks', 'cas3_licks'],
                             ['malt1_licks', 'malt2_licks', 'malt3_licks']):

    df_photo[c_sip_diff] = [np.mean(pref_sessions[x].cas['snips_sipper']['diff'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_sip_diff] = [np.mean(pref_sessions[x].malt['snips_sipper']['diff'], axis=0) for x in pref_sessions if pref_sessions[x].session == j] 
    df_photo[c_licks_diff] = [np.mean(pref_sessions[x].cas['snips_licks']['diff'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_licks_diff] = [np.mean(pref_sessions[x].malt['snips_licks']['diff'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]

for j, c_licks_forc, m_licks_forc in zip(included_sessions,
                           ['cas1_licks_forced', 'cas2_licks_forced', 'cas3_licks_forced'],
                           ['malt1_licks_forced', 'malt2_licks_forced', 'malt3_licks_forced']):
    df_photo[c_licks_forc] = [np.mean(pref_sessions[x].cas['snips_licks_forced']['diff'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_licks_forc] = [np.mean(pref_sessions[x].malt['snips_licks_forced']['diff'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]

for j, c_licks_peak, m_licks_peak, delta_licks_peak in zip(included_sessions,
                           ['cas1_licks_peak', 'cas2_licks_peak', 'cas3_licks_peak'],
                           ['malt1_licks_peak', 'malt2_licks_peak', 'malt3_licks_peak'],
                           ['pref1_peak_delta', 'pref2_peak_delta', 'pref3_peak_delta']):
    
    df_photo[c_licks_peak] = [np.mean(pref_sessions[x].cas['snips_licks_forced']['peak'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[m_licks_peak] = [np.mean(pref_sessions[x].malt['snips_licks_forced']['peak'], axis=0) for x in pref_sessions if pref_sessions[x].session == j]
    df_photo[delta_licks_peak] = df_photo[c_licks_peak] - df_photo[m_licks_peak]



# Assembles dataframe for reptraces

    
groups = ['NR-cas', 'NR-malt', 'PR-cas', 'PR-malt']
rats = ['PPP1-7', 'PPP1-7', 'PPP1-4', 'PPP1-4']
traces = [16, 19, 6, 4]
s = 's10'
event = 'snips_licks_forced'
keys_traces = ['cas1_licks_forced', 'malt1_licks_forced']

df_reptraces = pd.DataFrame(groups, columns=['group'])
df_reptraces.set_index(['group'], inplace=True)

df_reptraces['pref1-photo-blue'] = ""
df_reptraces['pref1-photo-uv'] = ""
df_reptraces['pref1-licks'] = ""

for group, rat, trace in zip(groups, rats, traces):
    
    x = pref_sessions[rat + '_' + s]
    
    if 'cas' in group:
        trial = x.cas[event]    
        run = x.cas['lickdata']['rStart'][trace]
        all_licks = x.cas['licks']
    elif 'malt' in group:
        trial = x.malt[event]    
        run = x.malt['lickdata']['rStart'][trace]
        all_licks = x.malt['licks']
    
    df_reptraces.at[group, 'pref1-licks'] = [l-run for l in all_licks if (l>run-10) and (l<run+20)]    
    df_reptraces.at[group, 'pref1-photo-blue'] = trial['blue'][trace]
    df_reptraces.at[group, 'pref1-photo-uv'] = trial['uv'][trace]

rats = np.unique(rats)
df_heatmap = pd.DataFrame(rats, columns=['rat'])
df_heatmap.set_index(['rat'], inplace=True)

df_heatmap['pref1-cas'] = ""
df_heatmap['pref1-malt'] = ""

for rat in rats:
    x = pref_sessions[rat + '_' + s]
    
    df_heatmap.at[rat, 'pref1-cas'] = removenoise(x.cas[event])
    df_heatmap.at[rat, 'pref1-malt'] = removenoise(x.malt[event])


pickle_out = open('R:\\DA_and_Reward\\gc214\\PPP_combined\\output\\ppp_dfs_pref.pickle', 'wb')
dill.dump([df_behav, df_photo, df_reptraces, df_heatmap], pickle_out)
pickle_out.close()

##
### TO DO!!!
### remove noise trials from grouped data
### figure out a way of excluding certain rats (e.g. PPP1.8) maybe just a line that removes at beginning of this code
##
